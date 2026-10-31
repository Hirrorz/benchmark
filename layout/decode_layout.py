# coding=utf-8
"""Decode 权重排布转换（单机 P→D：prefill 跑完后一次性切到 decode 形态）。

两种 decode 形态
----------------
* `mode="ep"`：保持 EP8（每卡 48 专家 × 全 intermediate），专家权重从 raw ND 转成
  decode-GMM 的 **NZ** 布局（w13/w2 + scale），走 dispatch→GMM×2→combine。
* `mode="tp"`：本 rank 拥有全部 384 专家 × intermediate/tp（官方 `moe_infer_tp`，
  一次 all_reduce）；需要把 config 的 EP 布局改成 TP 并**重新装载**专家权重。

一次性转换
----------
`prepare_decode()` 在 prefill 结束后调用一次：
  1) （tp）改写 `parallel_config`：`moe_ep_size=1 / moe_tp_size=world`；
  2) 清掉 prefill 的 MegaMoE 融合缓存（`_CACHE` 的 src/w、`_PREP_AC_L`、`_PREP_GIDS`）；
  3) `source.invalidate_experts()`：丢掉所有 `ffn.experts.*` 的旧布局缓存；
  4) `prebuild=True` 时逐层构建 decode-GMM NZ 权重并**释放 raw**（同尺寸替换，显存净增 0）。
转换完成后 decode 步直接命中缓存，不再有“首步逐层 0.6 s”的抖动。

为什么需要它
------------
单机没有 PD 分离：prefill 用 EP + MegaMoE（ascendc 融合权重，raw 已释放），
decode 用另一套专家布局。必须在同一进程里、prefill 之后显式转换一次。

依赖注入
--------
`exec_moe` 在导入时调用 `configure(u8, reshape_mx_scale)`，避免循环导入；
`prepare_decode` 内部按需 `import exec_moe` 清理其缓存。
"""
from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass

import torch

__all__ = [
    "DECODE_CACHE", "configure", "cached_gmm", "clear_decode_cache",
    "build_decode_gmm", "ensure_decode_gmm", "switch_to_tp",
    "invalidate_experts", "clear_prefill_caches", "prepare_decode", "PrepareResult",
]

_U8 = None
_RESHAPE = None

# 按层 decode-GMM 权重缓存（键 = 层号；值 = dict(w13,w13s,w2,w2s,gids)）。
# exec_moe 以 `_DEC_W` 别名引用同一张表（兼容旧代码）。
DECODE_CACHE: dict = {}
# 常驻 ep8：AC 与 decode NZ 同时常驻放不下（AC 18 + NZ 36 + attn/pools ≈ 80 GiB，
# 图捕获还要 ~5 GiB）。NZ 放主机 pinned，decode 每层 H2D 一次（~9 ms/层，
# ~0.4 s/token），显存只留一层，图捕获可成功。tp8-fidelity 不需要（AC 已释放）。
DECODE_HOST: dict = {}


def configure(u8, reshape_mx_scale) -> None:
    """由 exec_moe 注入张量工具（避免 exec_moe ↔ decode_layout 循环导入）。"""
    global _U8, _RESHAPE
    _U8, _RESHAPE = u8, reshape_mx_scale


def cached_gmm(layer_idx):
    return DECODE_CACHE.get(int(layer_idx))


def clear_decode_cache() -> None:
    DECODE_CACHE.clear()


def build_decode_gmm(w, gids, I, H, dev, ctx=None):
    """本 rank 专家的 decode-GMM 布局（官方 W4A8MxFp4MoEGMMMethod.create_weights 口径）。

        w13 [E, 2I, H/2] uint8（w1 在前、w3 在后）+ scale [E, 2I, H/32]
        w2  [E, H, I/2]  uint8 + scale [E, H, I/32]
    调用时与官方一样 `.transpose(1, 2)`（视图，不拷贝）；GMM 要求 NZ 格式。
    """
    if _U8 is None or _RESHAPE is None:
        raise RuntimeError("decode_layout 未 configure()（应由 exec_moe 导入时注入）")
    E = len(gids)
    # 本 rank 的**局部** intermediate 宽（EP=48 专家×全 I；TP=384 专家×I/tp）
    _I = int(_U8(w[f"ffn.experts.{gids[0]}.w1.weight"]).shape[0])
    w13 = torch.empty((E, 2 * _I, H // 2), dtype=torch.uint8, device=dev)
    w13s = torch.empty((E, 2 * _I, H // 32), dtype=torch.uint8, device=dev)
    w2 = torch.empty((E, H, _I // 2), dtype=torch.uint8, device=dev)
    w2s = torch.empty((E, H, _I // 32), dtype=torch.uint8, device=dev)
    for i, g in enumerate(gids):
        p = f"ffn.experts.{g}."
        w13[i, :_I].copy_(_U8(w[p + "w1.weight"]))
        w13[i, _I:].copy_(_U8(w[p + "w3.weight"]))
        w13s[i, :_I].copy_(_U8(w[p + "w1.scale"]))
        w13s[i, _I:].copy_(_U8(w[p + "w3.scale"]))
        w2[i].copy_(_U8(w[p + "w2.weight"]))
        w2s[i].copy_(_U8(w[p + "w2.scale"]))
    # ★ GMM 要求 NZ（实测：Only AclnnGroupedMatmulNz supports fp4）。
    #   官方对**未转置**的 [E,2I,H/2] 直接 cast；apply 里再 transpose(1,2) 纯视图。
    _NZ = getattr(torch_npu_nz(), "FRACTAL_NZ", 29)
    w13_t = torch_npu_format_cast(w13.contiguous(), _NZ,
                                  customize_dtype=torch.float8_e4m3fn,
                                  input_dtype=torch.float4_e2m1fn_x2)
    w2_t = torch_npu_format_cast(w2.contiguous(), _NZ,
                                 customize_dtype=torch.float8_e4m3fn,
                                 input_dtype=torch.float4_e2m1fn_x2)
    return w13_t, _RESHAPE(w13s), w2_t, _RESHAPE(w2s)


def torch_npu_nz():
    import torch_npu
    return torch_npu


def torch_npu_format_cast(t, fmt, **kw):
    import torch_npu
    return torch_npu.npu_format_cast(t, fmt, **kw)


def _restore_host_nz(t):
    """pinned host → device 后**重新 cast 回 NZ**。

    ★ 2026-10-19 踩实：`nz.to("cpu")` / `.to(dev)` 只保留**逻辑值**，storage 格式变成 ND
      （实测 `npu_format_cast` 后 `get_npu_format` 从 FRACTAL_NZ 变 ND）。fp4 GMM 只接受
      NZ（`Only AclnnGroupedMatmulNz supports ...`），ND 权重会让每层 decode 静默失败
      （run_forward 按算子捕获异常并 continue → 输出退化成"跳过 MoE"的乱码）。
      这里用与 `build_decode_gmm` 完全相同的 cast 参数重建 NZ。
    """
    return torch_npu_format_cast(
        t.contiguous(), getattr(torch_npu_nz(), "FRACTAL_NZ", 29),
        customize_dtype=torch.float8_e4m3fn,
        input_dtype=torch.float4_e2m1fn_x2)


# ensure_decode_gmm 的返回契约：(w13, w13s, w2, w2s)；只有 w13/w2 是 NZ，
# w13s/w2s 是普通 uint8（4D reshape）。host 回载时必须还原的位置。
_HOST_NZ_IDX = (0, 2)


def _load_host_gmm(h, dev):
    ts = tuple(x.to(dev) for x in h)
    return tuple(_restore_host_nz(t) if i in _HOST_NZ_IDX else t for i, t in enumerate(ts))


def ensure_decode_gmm(w, gids, I, H, dev, ctx=None):
    """带按层缓存 + raw 释放的 decode-GMM 构建（decode 步用）。

    只有 `ctx["is_decode_step"]` 为真才缓存/释放；prefill（TP 路径）每层只算一次，
    缓存会在 prefill 中累计 40×850 MB 常驻 + 逐层 empty_cache（实测 TP 16k prefill
    在层 22 卡死，疑与此有关）。
    """
    li = int(ctx.get("idx") or 0) if ctx is not None else None
    if ctx is not None and not ctx.get("is_decode_step"):
        li = None
    if li is not None and li in DECODE_CACHE:
        c = DECODE_CACHE[li]
        if "w13" in c:
            return c["w13"], c["w13s"], c["w2"], c["w2s"]
        h = DECODE_HOST.get(int(li))
        if h is not None:
            # 常驻 ep8：权重在 pinned host，按层 H2D（用完随局部变量释放）。
            # ★ 必须重新 cast NZ：host 只有逻辑值，`.to(dev)` 得到的是 ND（见 _restore_host_nz）。
            return _load_host_gmm(h, dev)
    w13, w13s, w2, w2s = build_decode_gmm(w, gids, I, H, dev, ctx)
    if li is not None:
        DECODE_CACHE[li] = dict(w13=w13, w13s=w13s, w2=w2, w2s=w2s, gids=list(gids))
        if os.environ.get("DS41M_DEC_RELEASE_RAW", "1") != "0":
            src = ctx.get("_src")
            if src is not None and hasattr(src, "drop_layer_experts"):
                src.drop_layer_experts(li)
    return w13, w13s, w2, w2s


def switch_to_tp(cfg, rank: int, world: int) -> None:
    """把 parallel_config 从 EP 改成 TP（本 rank 全专家 × intermediate/tp）。"""
    pc = cfg["parallel_config"]
    pc["moe_ep_size"] = 1
    pc["moe_tp_size"] = int(world)
    pc["_moe_tp_rank"] = int(rank)


def switch_to_ep(cfg, rank: int, world: int) -> None:
    """D→P：把 parallel_config 从 TP 切回 EP（本 rank 1/world 专家、全 intermediate）。"""
    pc = cfg["parallel_config"]
    pc["moe_ep_size"] = int(world)
    pc["moe_tp_size"] = 1
    pc["_moe_tp_rank"] = 0
    pc["_ep_rank"] = int(rank)
    os.environ.pop("DS41M_TP_DECODE_ONLY", None)
    os.environ.pop("DS41M_MOE_TP", None)
    os.environ.setdefault("DS41M_MOEMODE", "decode")


def invalidate_experts(source, clear_host: bool = False) -> int:
    """丢掉 source 里所有专家缓存（不含注意力/engram），下次 iter_layer 按新布局重取。

    `clear_host=True`：连 host 缓存里的专家分片一起丢——**EP↔TP 换形态必须**；
    保持 EP 布局时传 False，保留 host 缓存（省 34 GB/rank 的重读）。
    """
    if source is None or not hasattr(source, "invalidate_experts"):
        return 0
    try:
        return int(source.invalidate_experts(clear_host=clear_host) or 0)
    except TypeError:                                  # 兼容旧签名
        return int(source.invalidate_experts() or 0)


def clear_prefill_caches() -> None:
    """释放 prefill 专用缓存（MegaMoE 融合权重 / transform / gids / AC 释放标记）。"""
    import exec_moe as em
    em._CACHE["src"] = None
    em._CACHE["w"] = None
    for d in ("_PREP_AC_L", "_PREP_GIDS", "_PREP_L"):
        getattr(em, d, {}).clear()          # _PREP_L = deepgemm 按层 transform（默认空）
    getattr(em, "_AC_RELEASED", set()).clear()
    getattr(em, "_DG_RELEASED", set()).clear()
    clear_decode_cache()


def _gids_of(w) -> list:
    ids = set()
    for k in w:
        m = re.search(r"(?:^|\.)ffn\.experts\.(\d+)\.", k)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


# ── EP→TP 专家权重 all2all（避免为 7/8 的专家重新读共享盘）──────────────────
_TP_KINDS = ("w1.weight", "w3.weight", "w2.weight",
             "w1.scale", "w3.scale", "w2.scale")


def _slice_for_tp(t, kind: str, I: int, H: int, rank: int, world: int,
                  align: int = 128):
    """从**全 intermediate** 的专家权重里取本 TP rank 的分片。

    ★ 必须与 `weights._tp_slice_expert` **完全同口径**：官方 TP 不是均匀切，
      `split_intermediate_aligned(I, world, 128)` 先按 128 对齐分块再分给各 rank
      （I=2304/8 → 前 2 个 rank 各 384、其余各 256）；均匀 I//world=288 是错的。
    维度：w1/w3/scale 切 dim0；w2/scale 切 dim1（列，行宽<4KB 页 → 读盘页放大）。
      打包比例：w1/w3 是 I；w2 是 I/2；scale 是 I/32。
    """
    from weights import split_intermediate_aligned
    sizes = split_intermediate_aligned(I, world, align)
    off_i, per_i = sum(sizes[:rank]), sizes[rank]
    if kind in ("w1.weight", "w3.weight", "w1.scale", "w3.scale"):
        return t.narrow(0, off_i, per_i)
    if kind == "w2.weight":
        total = t.shape[1]                       # I/2
        return t.narrow(1, off_i * total // I, per_i * total // I)
    if kind == "w2.scale":
        total = t.shape[1]                       # I/32
        return t.narrow(1, off_i * total // I, per_i * total // I)
    raise KeyError(kind)


def _slice_geom(full_shape, kind: str, I: int, rank: int, world: int, align: int):
    """返回 (slice_shape, dim, off, per)；dim 相对**单专家**张量（0/1）。"""
    from weights import split_intermediate_aligned
    sizes = split_intermediate_aligned(I, world, align)
    off_i, per_i = sum(sizes[:rank]), sizes[rank]
    if kind in ("w1.weight", "w3.weight", "w1.scale", "w3.scale"):
        dim = 0
        off, per = off_i, per_i
    else:
        dim = 1
        total = int(full_shape[1])
        off, per = off_i * total // I, per_i * total // I
    shape = tuple(per if i == dim else int(v) for i, v in enumerate(full_shape))
    return shape, dim, off, per


def _prebuild_tp_all2all_uneven(source, cfg, rank: int, world: int, n_layers: int,
                                verbose: bool = True) -> int:
    """all_to_all 版：每个 rank 只收自己需要的 1/tp 分片（流量约为 all_gather 的 1/3~1/8）。

    HCCL 的 `all_to_all_single` 在本运行时支持不等长 split（已用 8 rank 实测），
    因此可以直接按官方**非均匀**分片尺寸收发：
      send[d] = 本 rank 48 个专家在目标 rank d 上的分片（epl × per_d）；
      recv[s] = 源 rank s 的 48 个专家在本 rank 上的分片（epl × per_this）。
    """
    import torch.distributed as dist
    if not dist.is_initialized():
        raise RuntimeError("TP all2all 需要已初始化的 torch.distributed 进程组")
    E = int(cfg["model"]["n_routed_experts"])
    if E % world:
        raise RuntimeError(f"n_routed_experts={E} 不能被 world={world} 整除")
    epl = E // world
    lo, hi = source.expert_range()
    if (hi - lo) != epl:
        raise RuntimeError(f"TP all2all 必须在 EP 布局下调用（当前 expert_range={lo,hi}，"
                           f"期望每 rank {epl} 个专家）")
    H = int(cfg["model"]["hidden"])
    I = int(cfg["model"].get("moe_intermediate",
                              cfg["model"].get("moe_intermediate_size", 0)))
    align = int(cfg["parallel_config"].get("moe_tp_align", 128) or 128)
    dev = getattr(source, "device", None) or cfg["run"].get("device", "npu:0")
    grp = dist.group.WORLD
    # ★ 2026-10-22：P→D 的输入是 **EP raw 专家权重**，不是 ascendc/deepgemm 各自的融合布局
    #   （`transform_weights_for_mega_moe` 只新建副本、不改 raw）。后端切换不影响本转换。
    #   `DS41M_PD_VERIFY=1`：逐层校验 all2all 前后分片字节和（发送端 vs 接收端），
    #   用来抓切片/打包/索引错误；默认关（会多一次 all_gather + 每层 2×sum）。
    _verify = os.environ.get("DS41M_PD_VERIFY", "").strip().lower() in (
        "1", "on", "true", "yes")
    t0 = time.time()
    for li in range(int(n_layers)):
        w_ep = source.iter_layer(li, int(cfg["model"]["n_routed_experts"]))
        pref = f"layers.{li}."
        ws_ep = {k[len(pref):] if k.startswith(pref) else k: v for k, v in w_ep.items()}
        if f"ffn.experts.{lo}.w1.weight" not in ws_ep:
            raise RuntimeError(
                "TP all2all 需要 EP raw 专家权重（当前层已被融合/释放）；"
                "请设 DS41M_AC_CACHE_N=0 / DS41M_AC_RELEASE_RAW=0（ascendc）或 "
                "DS41M_PREP_CACHE_N=0 / DS41M_DG_RELEASE_RAW=0（deepgemm）后重跑")
        # ★ 6 类张量**合成一次 all_to_all**（每层 1 个集合通信，而不是 6 个）：
        #   每层的集合通信次数从 240 降到 40；实测 40 层 4.9s→3.8s 后又降到 ~1s 量级。
        #   发送顺序：dest 主序 → kind 次序 → 本 rank 的 48 个专家。
        per_dest = [[] for _ in range(world)]
        geoms, kind_off, _o = {}, {}, 0
        for kind in _TP_KINDS:
            full_shape = tuple(int(x) for x in _U8(ws_ep[f"ffn.experts.{lo}.{kind}"]).shape)
            shape_r, _dim, _off, _per = _slice_geom(full_shape, kind, I, rank, world, align)
            geoms[kind] = shape_r
            kind_off[kind] = _o
            _o += epl * int(math.prod(shape_r))
            local = torch.stack(
                [_U8(ws_ep[f"ffn.experts.{e}.{kind}"]) for e in range(lo, hi)], 0
            )
            for d in range(world):
                _sd, ddim, doff, dper = _slice_geom(full_shape, kind, I, d, world, align)
                per_dest[d].append(
                    local.narrow(ddim + 1, doff, dper).contiguous().reshape(-1))
            del local
        chunks = [torch.cat(x) for x in per_dest]
        in_splits = [int(c.numel()) for c in chunks]
        inp = torch.cat(chunks)
        n_r_sum = _o                                   # 每个源 rank 发来的一整段长度
        if _verify:
            # send[d] 的字节和：本 rank → dest d；all_gather 后 [src][dest]
            _sums = torch.stack([c.sum(dtype=torch.int64) for c in chunks])
            _exp = [torch.zeros_like(_sums) for _ in range(world)]
            dist.all_gather(_exp, _sums, group=grp)
        del chunks, per_dest
        out = torch.empty(world * n_r_sum, dtype=torch.uint8, device=dev)
        dist.all_to_all_single(out, inp,
                               output_split_sizes=[n_r_sum] * world,
                               input_split_sizes=in_splits,
                               group=grp)
        del inp
        if _verify:
            _bad = []
            for _s in range(world):
                _got = int(out.narrow(0, _s * n_r_sum, n_r_sum)
                           .sum(dtype=torch.int64).item())
                _want = int(_exp[_s][rank].item())      # 源 _s 发往本 rank 的字节和
                if _got != _want:
                    _bad.append((_s, _got, _want))
            if _bad:
                raise RuntimeError(
                    f"P→D all2all 校验失败（层 {li}）：{_bad[:4]}"
                    "（发送/接收分片字节和不一致，疑似切片或索引错位）")
            if verbose:
                print(f"  [P→D {li + 1}/{n_layers}] all2all 校验通过"
                      f"（{world} 源 × 6 类张量，字节和一致）", flush=True)
        w = {}
        for e in range(E):
            owner, le = divmod(e, epl)
            base = owner * n_r_sum
            for kind in _TP_KINDS:
                shape_r = geoms[kind]
                n_r = int(math.prod(shape_r))
                off = base + kind_off[kind] + le * n_r
                w[f"ffn.experts.{e}.{kind}"] = out.narrow(0, off, n_r).view(shape_r)
        ensure_decode_gmm(w, list(range(E)), I, H, dev,
                          ctx={"idx": li, "is_decode_step": True, "_src": source})
        if hasattr(source, "mark_all_experts_dropped"):
            source.mark_all_experts_dropped(li, E, cfg=cfg)
        if verbose:
            _ma = torch.npu.memory_allocated() / 2**30
            _mr = torch.npu.memory_reserved() / 2**30
            _cache = getattr(source, "_cache", {}) or {}
            _nek = sum(1 for k in _cache if ".ffn.experts." in str(k))
            print(f"  [P→D {li + 1}/{n_layers}] TP all2all(fused)："
                  f"累计 {time.time() - t0:.1f}s，alloc {_ma:.1f}/{_mr:.1f} GiB，"
                  f"EP专家缓存 {_nek} 项", flush=True)
        del out, w, w_ep, ws_ep
    try:
        torch.npu.empty_cache()
    except Exception:                                  # noqa: BLE001
        pass
    return int(n_layers)


def _prebuild_tp_hostcache_bulk(source, cfg, rank: int, world: int, n_layers: int,
                                verbose: bool = True) -> int:
    """从常驻 host cache 建 TP decode 权重：**按层批量 H2D**，而不是逐张量。

    每个 rank 需要 40 层 × 384 专家 × 6 张量 = 92,160 个张量；逐个 `.to(device)`
    实测要 ~100 s（每次 ~1 ms 的 Python/切片/H2D 固定开销）。这里每层：
      1) 在 **pinned host buffer**（每类张量一块，复用）里按 rank 的 TP 分片填好 384 个专家；
      2) 每类张量只做 **一次** H2D（pinned → device，~90 GB/s）；
      3) 用 device 视图构建 decode-GMM。
    实测目标：102 s → ~3–6 s。
    """
    mm = getattr(source, "_mc_mm", None)
    man = getattr(source, "_mc_man", None)
    if mm is None or not man:
        raise RuntimeError("host cache 未连接")
    E = int(cfg["model"]["n_routed_experts"])
    H = int(cfg["model"]["hidden"])
    I = int(cfg["model"].get("moe_intermediate",
                              cfg["model"].get("moe_intermediate_size", 0)))
    align = int(cfg["parallel_config"].get("moe_tp_align", 128) or 128)
    dev = getattr(source, "device", None) or cfg["run"].get("device", "npu:0")

    geoms = {}
    for kind in _TP_KINDS:
        ent = man[f"layers.0.ffn.experts.0.{kind}"]
        full_shape = tuple(int(x) for x in ent[1])
        geoms[kind] = (full_shape,) + _slice_geom(full_shape, kind, I, rank, world, align)

    # pinned host staging（每类一块，跨层复用）
    hbuf = {}
    for kind in _TP_KINDS:
        _fs, shape_r, _dim, _off, _per = geoms[kind]
        hbuf[kind] = torch.empty((E,) + shape_r, dtype=torch.uint8).pin_memory()
    if verbose:
        _mb = sum(v.numel() for v in hbuf.values()) / 2**20
        print(f"  [P→D] TP 权重来源 = host cache（按层批量 H2D，staging {_mb:.0f} MiB pinned）",
              flush=True)

    t0 = time.time()
    for li in range(int(n_layers)):
        if f"layers.{li}.ffn.experts.0.w1.weight" not in man:
            raise RuntimeError(f"host cache 缺少层 {li} 的专家权重（缓存范围与本次层数不一致）")
        for kind in _TP_KINDS:
            full_shape, shape_r, dim, off, per = geoms[kind]
            hb = hbuf[kind]
            for e in range(E):
                ent = man[f"layers.{li}.ffn.experts.{e}.{kind}"]
                src = torch.frombuffer(mm, dtype=torch.uint8,
                                       count=int(ent[3]), offset=int(ent[2])).reshape(full_shape)
                hb[e].copy_(src.narrow(dim, off, per))
        dv = {kind: hbuf[kind].to(dev) for kind in _TP_KINDS}
        # hbuf 跨层复用：必须等 H2D 完成，否则下一层的 host 写入会与在途 DMA 竞争
        torch.npu.synchronize()
        w = {}
        for e in range(E):
            for kind in _TP_KINDS:
                w[f"ffn.experts.{e}.{kind}"] = dv[kind][e]
        ensure_decode_gmm(w, list(range(E)), I, H, dev,
                          ctx={"idx": li, "is_decode_step": True, "_src": source})
        if hasattr(source, "mark_all_experts_dropped"):
            source.mark_all_experts_dropped(li, E, cfg=cfg)
        if verbose and (li + 1) % 10 == 0:
            print(f"  [P→D {li + 1}/{n_layers}] host cache 批量转换，"
                  f"累计 {time.time() - t0:.1f}s", flush=True)
        del dv, w
    del hbuf
    try:
        torch.npu.empty_cache()
    except Exception:                                  # noqa: BLE001
        pass
    return int(n_layers)


def _prebuild_tp_all2all(source, cfg, rank: int, world: int, n_layers: int,
                         verbose: bool = True) -> int:
    """不读盘：把每个 rank 手里的 EP 专家分片 all-gather，再切出 TP 1/world 分片。

    EP prefill 后 rank r 持有专家 [r*E/world, (r+1)*E/world) 的**全 intermediate** 权重；
    TP decode 每个 rank 需要**全部**专家 × I/world。这里按层：
      1) 取本 rank 的 EP 分片（source.iter_layer，仍在 EP 布局下）；
      2) 每类权重做一次 all_gather（world × 本 rank 分片）；
      3) 从 gathered[owner][local] 切出 rank 的 1/world，喂 build_decode_gmm。
    通信量 ≈ 每 rank 每层 E×I/world×(H/2+…) ≈ 0.85 GiB 接收；40 层秒级。
    """
    import torch.distributed as dist
    if not dist.is_initialized():
        raise RuntimeError("TP all2all 需要已初始化的 torch.distributed 进程组")
    E = int(cfg["model"]["n_routed_experts"])
    if E % world:
        raise RuntimeError(f"n_routed_experts={E} 不能被 world={world} 整除")
    epl = E // world
    lo, hi = source.expert_range()
    if (hi - lo) != epl:
        raise RuntimeError(f"TP all2all 必须在 EP 布局下调用（当前 expert_range={lo,hi}，"
                           f"期望每 rank {epl} 个专家）")
    H = int(cfg["model"]["hidden"])
    I = int(cfg["model"].get("moe_intermediate",
                              cfg["model"].get("moe_intermediate_size", 0)))
    dev = getattr(source, "device", None) or cfg["run"].get("device", "npu:0")
    grp = dist.group.WORLD
    has_into = hasattr(dist, "all_gather_into_tensor")
    t0 = time.time()
    for li in range(int(n_layers)):
        w_ep = source.iter_layer(li, int(cfg["model"]["n_routed_experts"]))
        pref = f"layers.{li}."
        ws_ep = {k[len(pref):] if k.startswith(pref) else k: v for k, v in w_ep.items()}
        if f"ffn.experts.{lo}.w1.weight" not in ws_ep:
            raise RuntimeError(
                "TP all2all 需要 EP raw 专家权重（当前层已被融合/释放）；"
                "请设 DS41M_AC_CACHE_N=0 / DS41M_AC_RELEASE_RAW=0（ascendc）或 "
                "DS41M_PREP_CACHE_N=0 / DS41M_DG_RELEASE_RAW=0（deepgemm）后重跑")
        gathered = {}
        for kind in _TP_KINDS:
            local = torch.stack(
                [_U8(ws_ep[f"ffn.experts.{e}.{kind}"]) for e in range(lo, hi)], 0
            ).contiguous()
            out = torch.empty((world,) + tuple(local.shape),
                              dtype=local.dtype, device=local.device)
            if has_into:
                dist.all_gather_into_tensor(out, local, group=grp)
            else:
                parts = [torch.empty_like(local) for _ in range(world)]
                dist.all_gather(parts, local, group=grp)
                out = torch.stack(parts, 0)
            gathered[kind] = out
            del local
        w = {}
        for e in range(E):
            owner, le = divmod(e, epl)
            for kind in _TP_KINDS:
                w[f"ffn.experts.{e}.{kind}"] = _slice_for_tp(
                    gathered[kind][owner][le], kind, I, H, rank, world,
                    align=int(cfg["parallel_config"].get("moe_tp_align", 128) or 128))
        ensure_decode_gmm(w, list(range(E)), I, H, dev,
                          ctx={"idx": li, "is_decode_step": True, "_src": source})
        # ★ 只 drop 已加载的 48 个 EP 专家还不够：decode 步的 manifest 在 TP 下会列出
        #   全部 384 个，未标记的 336 个会被重新读盘。把本层全部专家键都标 dropped。
        if hasattr(source, "mark_all_experts_dropped"):
            source.mark_all_experts_dropped(li, E, cfg=cfg)
        if verbose:
            gb = sum(v.numel() * v.element_size() for v in gathered.values()) / 2**30
            try:
                _ma = torch.npu.memory_allocated() / 2**30
                _mr = torch.npu.memory_reserved() / 2**30
                _mem = f"，alloc {_ma:.1f}/reserved {_mr:.1f} GiB"
            except Exception:                      # noqa: BLE001
                _mem = ""
            _cache = getattr(source, "_cache", {}) or {}
            _nek = sum(1 for k in _cache if ".ffn.experts." in str(k))
            _bek = sum(t.numel() * t.element_size() for k, t in _cache.items()
                       if ".ffn.experts." in str(k) and torch.is_tensor(t)) / 2**30
            print(f"  [P→D {li + 1}/{n_layers}] TP all2all："
                  f"gather {gb:.1f} GiB/层，累计 {time.time() - t0:.1f}s{_mem}"
                  f"，EP专家缓存 {_nek} 项/{_bek:.1f} GiB", flush=True)
        del gathered, w, w_ep, ws_ep
    # ★ 40 层构建完、EP raw 引用全部释放后清一次分配器缓存：TP 的 decode 缓存很大
    #   （rank0/1 因 384 对齐分片约 46 GiB），不 empty_cache 会残留十几 GiB reserved，
    #   第一步 decode 申请几 MiB 就报 OOM（实测 74.3 GiB allocated / 88.6 reserved / 3.8 MiB free）。
    try:
        torch.npu.empty_cache()
    except Exception:                                  # noqa: BLE001
        pass
    return int(n_layers)


@dataclass
class PrepareResult:
    mode: str
    prebuild: bool
    invalidated: int
    converted: int
    seconds: float
    message: str = ""


def prepare_decode(source, cfg, rank: int, world: int, *, mode: str = "ep",
                   prebuild: bool = True, n_layers: int = 40,
                   verbose: bool = True, keep_prefill: bool = False) -> PrepareResult:
    """**一次调用**把当前 prefill 权重切成 decode 形态（单机 P→D 的核心入口）。

    mode:
      · "ep"：保持 EP 布局，预构建 decode-GMM NZ 权重（走 dispatch/combine）；
      · "tp"：切到 TP 布局（全专家 × I/tp），按层重载并预构建 NZ 权重（走 all_reduce）。
    prebuild=True 时逐层完成转换并释放 raw；False 时只切模式/清缓存，首步按层懒构建。
    """
    mode = str(mode).lower()
    if mode not in ("ep", "tp"):
        raise ValueError(f"decode mode 只能是 ep/tp，收到 {mode!r}")
    t0 = time.time()
    xfer = os.environ.get("DS41M_PD_TP_XFER", "").strip().lower()
    use_a2a = (mode == "tp" and prebuild and int(world) > 1
               and xfer in ("all2all", "all2all_gather"))
    # 批量 H2D 版实测比逐张量慢（w2 列切片在 host 上是 strided copy，3 层 23.8s vs 8.8s），
    # 因此**只做 opt-in**，默认不开；TP 换布局首选 all2all（3.2s）。
    use_hc = (mode == "tp" and prebuild and not use_a2a
              and xfer == "hostcache_bulk"
              and getattr(source, "_mc_man", None) is not None)

    converted = 0
    n_inv = 0
    if keep_prefill and not (use_a2a or use_hc):
        raise RuntimeError("keep_prefill 的转换必须走 all2all 或 hostcache_bulk"
                           "（否则会 invalidate 掉 EP raw，D→P 无法回载）")
    if use_hc:
        if keep_prefill:
            clear_decode_cache()          # 保留 AC/_PREP_GIDS
        else:
            clear_prefill_caches()
        # 先丢掉 EP prefill 的设备专家缓存（34 GB/rank）；bulk 路径不再用它们。
        invalidate_experts(source, clear_host=True)
        converted = _prebuild_tp_hostcache_bulk(source, cfg, rank, int(world),
                                                int(n_layers), verbose)
        switch_to_tp(cfg, rank, int(world))
        os.environ["DS41M_TP_DECODE_ONLY"] = "1"
        # ★ 不能再调 invalidate_experts：会清空 dropped → decode 步重新加载专家。
        n_inv = 0
    elif use_a2a:
        # ★ 必须在 EP 布局下 gather：此时 source.expert_range 仍只返回本 rank 的 48 个专家。
        #   clear_prefill_caches() 会清 DECODE_CACHE → 必须放在构建之前。
        if keep_prefill:
            clear_decode_cache()
        else:
            clear_prefill_caches()
        if verbose:
            _how = ("all_to_all（不等长分片，流量最小）" if xfer == "all2all"
                    else "all_gather（兼容路径）")
            print(f"  [P→D] TP 权重来源 = {_how}（HCCL 交换 EP 分片，不读共享盘）", flush=True)
        if xfer == "all2all":
            converted = _prebuild_tp_all2all_uneven(source, cfg, rank, int(world),
                                                    int(n_layers), verbose)
        else:
            converted = _prebuild_tp_all2all(source, cfg, rank, int(world),
                                             int(n_layers), verbose)
        switch_to_tp(cfg, rank, int(world))
        os.environ["DS41M_TP_DECODE_ONLY"] = "1"          # mega_moe 在 decode 步选 TP 路径
        # ★ 不要再调 invalidate_experts：每层 ensure_decode_gmm 已 drop_layer_experts，
        #   设备侧 EP 原始权重已释放；而 invalidate_experts 会 **清空 dropped 集合** →
        #   decode 每步的 iter_layer 会重新按 TP 布局加载全部 384 专家（34 GB/层级重读）。
        #   实测：TP 512 第二步 OOM（74→90 GB，tried to allocate 2 MiB）。
        #   host 里的 EP 全量分片留着（8 rank 共 ~272 GB，1.5 TB 内存可控）。
        n_inv = 0
    else:
        if mode == "tp":
            switch_to_tp(cfg, rank, world)
            os.environ["DS41M_TP_DECODE_ONLY"] = "1"      # mega_moe 在 decode 步选 TP 路径
        else:
            # ★ 2026-10-22：必须**强制** decode，不能 setdefault。构建器生成的 launch 脚本会
            #   `export DS41M_MOEMODE=deepgemm`（强制 prefill 后端）；setdefault 不覆盖，
            #   decode 仍走 prefill MegaMoE → `_prepared_dg_cached` 去读 P→D 已释放的 raw
            #   → `KeyError: 'ffn.experts.0.w1.weight'`（2026-10-22 05:14 实测崩溃）。
            os.environ["DS41M_MOEMODE"] = "decode"   # mega_moe 在 decode 步选 dispatch/combine

        clear_prefill_caches()
        # TP 换形态必须清 host 分片（EP 全量 ≠ TP 1/tp）；EP 保持布局则保留 host 缓存
        n_inv = invalidate_experts(source, clear_host=(mode == "tp"))

        if prebuild:
            H = int(cfg["model"]["hidden"])
            dev = getattr(source, "device", None) or cfg["run"].get("device", "npu:0")
            I = int(cfg["model"].get("moe_intermediate", cfg["model"].get("moe_intermediate_size", 0)))
            _read_s = _build_s = 0.0
            for li in range(int(n_layers)):
                _t0 = time.perf_counter()
                w = source.iter_layer(li, int(cfg["model"]["n_routed_experts"]))
                _t1 = time.perf_counter()
                # iter_layer 返回的是 `layers.{li}.<suffix>`；算子侧吃的是去前缀的名字
                pref = f"layers.{li}."
                ws = {k[len(pref):] if k.startswith(pref) else k: v for k, v in w.items()}
                gids = _gids_of(ws)
                if not gids:
                    continue
                ensure_decode_gmm(ws, gids, I, H, dev,
                                  ctx={"idx": li, "is_decode_step": True, "_src": source})
                _t2 = time.perf_counter()
                _read_s += _t1 - _t0
                _build_s += _t2 - _t1
                converted += 1
                if verbose and (li + 1) % 10 == 0:
                    print(f"  [P→D {li + 1}/{n_layers}] decode 权重转换（{mode}）："
                          f"读权重 {_read_s:.1f}s + 构建 {_build_s:.1f}s", flush=True)

    secs = time.time() - t0
    try:
        import exec_moe as _em
        _be = _em._CACHE.get("backend") or os.environ.get("DS41M_MOEMODE", "auto")
    except Exception:                                  # noqa: BLE001
        _be = os.environ.get("DS41M_MOEMODE", "auto")
    msg = (f"  [P→D] prefill → decode 完成：mode={mode}，"
           f"输入=EP raw 专家权重（prefill 后端 {_be}），"
           f"清专家缓存 {n_inv} 项，预转换 {converted}/{int(n_layers)} 层，{secs:.1f}s")
    if verbose:
        try:
            msg += (f"；显存 alloc {torch.npu.memory_allocated() / 2**30:.1f}"
                    f"/reserved {torch.npu.memory_reserved() / 2**30:.1f} GiB")
        except Exception:                                  # noqa: BLE001
            pass
    if verbose:
        print(msg, flush=True)
    return PrepareResult(mode=mode, prebuild=prebuild, invalidated=n_inv,
                         converted=converted, seconds=secs, message=msg)


# ── D→P：TP decode → EP prefill raw（AC 常驻，不重建）────────────────────────
def _restore_ep_raw_bulk(source, cfg, rank: int, world: int, n_layers: int,
                         verbose: bool = True) -> int:
    """从 host cache 按层批量 H2D 回载本 rank 的 EP 专家 raw。

    为什么需要它：P→D all2all 后 raw 已逐层释放（`drop_layer_experts`），TP NZ 不能反向
    all2all 回 raw（NZ 是 decode-GMM 格式）。host cache 里有全量 raw，每 rank 只需自己的
    1/world（48 专家 × 40 层 ≈ 33.6 GiB）。按层每类张量一次 H2D，避免 11,520 次小拷贝。
    """
    mm = getattr(source, "_mc_mm", None)
    man = getattr(source, "_mc_man", None)
    if mm is None or not man:
        return -1
    import weights as _W
    E = int(cfg["model"]["n_routed_experts"])
    epl = E // int(world)
    lo, hi = source.expert_range()
    if (hi - lo) != epl:
        raise RuntimeError(f"D→P 必须在 EP 布局下调用（expert_range={lo,hi}，"
                           f"期望每 rank {epl} 专家）")
    dev = getattr(source, "device", None) or cfg["run"].get("device", "npu:0")
    geo = {}
    for kind in _TP_KINDS:
        ent = man.get(f"layers.0.ffn.experts.{lo}.{kind}")
        if ent is None:
            raise RuntimeError(f"host cache 缺少 layers.0.ffn.experts.{lo}.{kind}；"
                               f"缓存范围与 EP 分片不一致")
        dt_s, shp, _off, nb = ent
        geo[kind] = (str(dt_s), tuple(int(x) for x in shp), int(nb))
    hbuf = {kind: torch.empty((epl, geo[kind][2]), dtype=torch.uint8).pin_memory()
            for kind in _TP_KINDS}
    t0 = time.time()
    for li in range(int(n_layers)):
        pref = f"layers.{li}."
        # 1) 释放本层 TP decode NZ（腾显存）
        DECODE_CACHE.pop(int(li), None)
        # 2) 允许本 rank 的 EP 专家重新进入 _cache
        for e in range(lo, hi):
            for kind in _TP_KINDS:
                key = pref + f"ffn.experts.{e}.{kind}"
                source.dropped.discard(key)
                source._cache.pop(key, None)
        # 3) memfd → pinned staging（每类一块，跨层复用）
        #    ★ 6 类**并行**拷贝：串行时 33.6 GiB 单线程 memcpy 实测 6.5–8 s；
        #      并行后由内存带宽决定（TP8 D→P 目标 3.2–3.5 s）。
        def _stage(kind):
            hb = hbuf[kind]
            nb_exp = geo[kind][2]
            for j, e in enumerate(range(lo, hi)):
                ent = man.get(pref + f"ffn.experts.{e}.{kind}")
                if ent is None:
                    raise RuntimeError(
                        f"host cache 缺 {pref}ffn.experts.{e}.{kind}（缓存范围不足）")
                _dt, _shp, off, nb = ent
                if int(nb) != nb_exp:
                    raise RuntimeError(f"{pref}ffn.experts.{e}.{kind} 字节数不一致")
                src = torch.frombuffer(mm, dtype=torch.uint8,
                                       count=int(nb), offset=int(off))
                hb[j].copy_(src)
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=len(_TP_KINDS)) as _ex:
            for _f in [_ex.submit(_stage, k) for k in _TP_KINDS]:
                _f.result()
        # 4) 每类一次 H2D（pinned 源 + non_blocking：6 类在拷贝引擎上重叠）
        dv = {kind: hbuf[kind].to(dev, non_blocking=True) for kind in _TP_KINDS}
        torch.npu.synchronize()
        # 5) 还原 dtype/shape，写进 source._cache（视图持有整块 base，跨层不重复拷贝）
        for kind in _TP_KINDS:
            dt_s, shp, _nb = geo[kind]
            dt_c = _W.DTYPE.get(dt_s)
            for j, e in enumerate(range(lo, hi)):
                t = dv[kind][j]
                if dt_c is not None and dt_c is not torch.uint8:
                    t = t.view(dt_c)
                t = t.reshape(shp)
                source._cache[pref + f"ffn.experts.{e}.{kind}"] = t
        del dv
        if verbose:
            print(f"  [D→P {li + 1}/{n_layers}] 回载 EP raw："
                  f"累计 {time.time() - t0:.1f}s，"
                  f"alloc {torch.npu.memory_allocated() / 2**30:.1f} GiB", flush=True)
        if (li + 1) % 10 == 0:
            try:
                torch.npu.empty_cache()
            except Exception:                          # noqa: BLE001
                pass
    del hbuf
    try:
        torch.npu.empty_cache()
    except Exception:                                  # noqa: BLE001
        pass
    return int(n_layers)


def restore_ep_raw(source, cfg, rank: int, world: int, n_layers: int = 40,
                   verbose: bool = True, load_raw: bool = True):
    """D→P 入口：切回 EP 布局 + 回载 raw（AC 保持常驻，下一 prefill 直接命中）。

    `load_raw=False`：**延迟回载**。只释放 TP decode NZ、切回 EP，不向 host 回载
    EP raw（~33.6 GiB / 5.5 s）。下一段 prefill 用常驻 AC，不需要 raw；raw 留到
    下一次 P→D 之前再回载。预热 W2 用这条路径，避免"raw 33.6 + AC 33.6 同时
    常驻 + 首遍建 AC"把 W3 顶到 OOM（实测 reserved 90.09/91.75，1.31 GiB 建不出来）。
    """
    t0 = time.time()
    switch_to_ep(cfg, int(rank), int(world))
    if not load_raw:
        for li in range(int(n_layers)):
            DECODE_CACHE.pop(int(li), None)
        try:
            torch.npu.empty_cache()
        except Exception:                              # noqa: BLE001
            pass
        secs = time.time() - t0
        if verbose:
            print(f"  [D→P] TP → EP（延迟回载 raw）：释放 NZ {int(n_layers)} 层，"
                  f"{secs:.2f}s", flush=True)
        return {"layers": 0, "seconds": secs, "deferred": True}
    # ★ 先把 40 层 TP NZ 全部放掉再回载：回载阶段的 staging/raw 分配不必与
    #   ~44.8 GiB 的 NZ 抢空间（实测 D→P 从 4.0–5.7s 降到接近 W2 的 3.3s）。
    for _li in range(int(n_layers)):
        DECODE_CACHE.pop(int(_li), None)
    try:
        torch.npu.empty_cache()
    except Exception:                                  # noqa: BLE001
        pass
    n = _restore_ep_raw_bulk(source, cfg, rank, world, n_layers, verbose)
    if n < 0:
        # 没有 host cache：逐张量回载（走磁盘或已 pin 的 host，慢，仅保底）
        E = int(cfg["model"]["n_routed_experts"])
        n = 0
        for li in range(int(n_layers)):
            DECODE_CACHE.pop(int(li), None)
            pref = f"layers.{li}."
            for e in range(*source.expert_range()):
                for kind in _TP_KINDS:
                    source.dropped.discard(pref + f"ffn.experts.{e}.{kind}")
            source.iter_layer(li, E)
            n += 1
            if verbose and (li + 1) % 5 == 0:
                print(f"  [D→P {li + 1}/{n_layers}] 逐张量回载（无 host cache），"
                      f"累计 {time.time() - t0:.1f}s", flush=True)
    secs = time.time() - t0
    if verbose:
        try:
            mem = (f"，alloc {torch.npu.memory_allocated() / 2**30:.1f}"
                   f"/reserved {torch.npu.memory_reserved() / 2**30:.1f} GiB")
        except Exception:                              # noqa: BLE001
            mem = ""
        print(f"  [D→P] TP → EP 完成：{n}/{n_layers} 层，{secs:.1f}s{mem}", flush=True)
    return {"layers": int(n), "seconds": secs}

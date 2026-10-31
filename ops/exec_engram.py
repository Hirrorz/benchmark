"""engram 三件套（插件，见 exec_ops._load_plugins）—— **官方真实链路**。

官方链条（`modules/engram.py`，prefill 走 Engram.forward 的 tilelang 分支）：

    ① NgramHashState → tile_kernels.engram.engram_hash → 每层 [T, 24] 个取表 id
         （24 = (max_ngram_size-1) × n_heads = 3×8；词表先过压缩表 token_map）
    ② MultiHeadEmbedding.forward → 按 id 从 **host 上的表**（mxfp8，98 GB/层）取行
         `engram_buffer.engram_fetch(ids)` → `wait()`（厂商 ElasticBuffer，跨卡 RDMA；
         trace 里 = `EngramFetch` + `EngramFetchWait`），随后按 E8M0 scale 解量化 → bf16
         ★ 2026-10-15：**默认就走这条**（官方 hit97 = enable_engram_offload=True + tp=8）。
           以前默认是 host index_select 兜底（数值等价但 trace 里没有那两个算子，对拍缺项）；
           `DS41M_ENGRAM_HOST=1` 可退回。
    ③ embeddings [T, 24×256=6144] → `engram.wkv`（QMM）→ [T, 25600]
         → reshape [T, hc_mult+1=5, 5120] = kv
    ④ engram_gate_fwd(hidden [T,4,5120], kv [T,5,5120], fused_weight(q_weight,k_weight),
                      eps=rms_norm_eps, clamp_value=1e-6) → **覆盖残差流** st["x"]

★ 之前两处与官方不同（本次已修）：
  1. `engram_proj` 曾把 `hidden[:, :5120] + 占位 kv(1024)` 拼成 6144 喂 wkv —— 形状恰好
     也是 6144，算子图对齐看不出来，但官方喂的是**取到的表行**（24×256）；
  2. 官方的 engram 输出是**替换 hidden_states**（DecoderLayer.forward 的第一步），
     不是"另外存一份"。
"""
from __future__ import annotations

import os
import time

import torch

import engram_layout as el

ADDED = "engram_fetch/wait/proj"
ST = {"cfg": None, "path": None}


def _env_on(name, default=False):
    """布尔环境变量：`0/off/false/no` 与空串都算关。

    ★ 以前用 `bool(os.environ.get(name))`：`DS41M_ENGRAM_HOST=0` 仍是**非空字符串** →
      被当成开，bench_server 注释里的"设 0 强制厂商路径"实际无效（2026-10-20 实测）。
    """
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return bool(default)
    return str(v).strip().lower() in ("1", "on", "true", "yes")

# ★ 2026-10-19：结构参数由 model_spec 提供（默认 = 旧硬编码值）。
QUANT_BLOCK = 32         # E8M0 scale 组宽（= model_spec.quant_block）
PREFETCH_T = 2048        # 侧流预取默认 token 数（= model_spec.engram_prefetch_T）



def _apply_spec(cfg=None):
    import model_spec as _ms
    s = _ms.set_spec(cfg, strict=False) if cfg is not None else _ms.get()
    global QUANT_BLOCK, PREFETCH_T
    QUANT_BLOCK = int(s.quant_block)
    PREFETCH_T = int(s.engram_prefetch_T)
    if cfg is not None:                  # 布局参数（层号/表规模）跟随 config
        try:
            import engram_layout as _el
            _el.configure(cfg)
        except Exception as exc:         # noqa: BLE001
            print(f"  [engram] ⚠️ layout configure 失败（用默认布局）：{exc}", flush=True)
    return s


def _cfg_path(ctx):
    if ST["path"] is None:
        cfg = ST.get("cfg") or {}
        ST["path"] = (cfg.get("weights") or {}).get("path", "")
    return ST["path"]


def _ids_of(ctx, T):
    """输入 token id：对拍时由 runner 透传（ctx["input_ids"]）；合成模式返回 None。"""
    ids = ctx.get("input_ids")
    if ids is None:
        return None
    t = torch.as_tensor(ids).reshape(-1).to(torch.long)
    n_ctx = int(ctx.get("engram_ctx", 0) or 0)
    if n_ctx and t.numel() > T:
        # 增量 decode：只算 T 个位置，但 n-gram hash 需要上文 → 多带 n_ctx 个 id，
        # `_engram_fetch` 里取 hash 的**最后 T 行**（= 当前位置的 hash）
        return t[-(T + n_ctx):]
    return t[:T]


def _skip():
    """调试开关：DS41M_NO_ENGRAM=1 时三个算子全成 no-op（复现"官方跳过 engram"的情形）。"""
    return _env_on("DS41M_NO_ENGRAM")


# ── engram id 越界检查（不产生稳态 host 同步）────────────────────────────────
# ★ 2026-10-21 hostbound 根因之一：以前每次 fetch 都 `int(ids.max())` —— 设备 max +
#   `.item()`，命中稳态里每 pass 两个 engram 层各一次（trace 里 aten::item 前面正是
#   select→aclnnMax→max，每次 5.6ms 全是在等设备队列排空）。
#   现策略：
#     · 合成 id（`_synth_ids`）= randint(0, rows) → 必然合法，**直接跳过**；
#     · 真 id 默认每个 (层, T, 表行数) 只查**首次**（布局/hash 错误在首跑即暴露），
#       `DS41M_ENGRAM_BOUNDS=always` 恢复每次都查、`=off` 全关。
_BOUNDS_SEEN: set = set()


def _bounds_mode() -> str:
    v = os.environ.get("DS41M_ENGRAM_BOUNDS", "").strip().lower()
    if v in ("0", "off", "false", "no"):
        return "off"
    if v in ("1", "on", "true", "yes", "always"):
        return "always"
    return "once"


def _check_ids_range(ids, n_rows, layer, synthetic=False):
    """越界检查（见上方说明）。返回 max（仅作日志用；不查时返回 -1，不触碰设备）。"""
    if synthetic or not n_rows or _bounds_mode() == "off":
        return -1
    key = (int(layer), int(ids.shape[0]) if hasattr(ids, "shape") else -1, int(n_rows))
    if _bounds_mode() == "once":
        if key in _BOUNDS_SEEN:
            return -1
        _BOUNDS_SEEN.add(key)
    imax = int(ids.max())
    if imax >= int(n_rows):
        raise RuntimeError(f"engram id 越界：max={imax} 表行数={int(n_rows)}（层 {layer}）")
    return imax


import concurrent.futures as _fut

_ENG_POOL = _fut.ThreadPoolExecutor(max_workers=1)   # host 侧取表（纯 CPU，不碰 NPU）


def _engram_host_gather(ids, tbl, sf):
    """host 侧**只取字节**：fp8 行 + E8M0 scale 行（解量化放到卡上做，与官方一致）。

    ★ 官方（modules/engram.py:533-546）：`engram_buffer.engram_fetch(ids)` → wait 拿回
      `(fetched, fetched_sf)` —— 回的是**量化字节**，然后用 `(sf.view(uint8).to(int32) << 23).view(float32)`
      在**卡上**解量化。我们以前在主机上解量化再传 bf16（H2D 翻倍、主机多做一遍浮点）。
    ★ 不按 id 去重：搬运量按官方口径（官方也是把 [T×24, 6144] 整块搬一遍）。
    """
    idx = ids.reshape(-1).to(torch.int64).cpu()
    rows = torch.index_select(tbl.detach().cpu(), 0, idx)              # fp8  [N, 6144]
    sc = None if sf is None else torch.index_select(
        sf.detach().cpu().view(torch.uint8), 0, idx)                   # e8m0 [N, 192]
    return rows, sc


def _host_emb_from_ids(ids2d, tbl, sf, dev):
    """host 取表 + 卡上解量化（与 `_engram_wait` 的兜底分支同一口径）。"""
    rows, sc = _engram_host_gather(ids2d.to(torch.int32), tbl, sf)
    rows = rows.to(dev)
    if sc is not None:
        scale = (sc.to(dev).to(torch.int32) << 23).view(torch.float32)
        return (rows.to(torch.float32).unflatten(-1, (-1, QUANT_BLOCK))
                * scale.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
    return rows.to(torch.bfloat16)


def _engram_check_value(tag, emb_vendor, ids2d, w, ctx, dev):
    """DS41M_ENGRAM_CHECK=1：厂商取行 vs host 取行逐元素对比（定位数值差）。"""
    if os.environ.get("DS41M_ENGRAM_CHECK", "").strip().lower() not in (
            "1", "on", "true", "yes"):
        return
    try:
        host = _host_emb_from_ids(ids2d, w["engram.embed.weight"],
                                  w.get("engram.embed.scale"), dev)
        d = (emb_vendor.float() - host.float()).abs()
        nz = int((d > 0).sum())
        print(f"  [engram-check] L{int(ctx.get('idx', -1))} {tag} "
              f"max={float(d.max()):.6g} nonzero={nz}/{int(d.numel())} "
              f"contig={int(emb_vendor.is_contiguous())} "
              f"stride={tuple(emb_vendor.stride())} "
              f"ids0={ids2d.reshape(-1)[:6].tolist()}", flush=True)
    except Exception as _e:                            # noqa: BLE001
        print(f"  [engram-check] L{int(ctx.get('idx', -1))} {tag} 失败："
              f"{type(_e).__name__}: {_e}", flush=True)


_PREFETCH: dict = {}          # {layer_idx: future}：提前 1~2 层发起的 host 取表


def _host_gather_will_run(w):
    """只有"表在主机 + 厂商路径被关掉（DS41M_ENGRAM_HOST=1）"时才轮到 host 取表。

    表常驻设备（engram_offload=false）时取行在 wait 里由设备 gather 做；厂商路径下取表由
    ElasticBuffer/RDMA 做 —— 这两种情况 host 预取**都不会被消费**（`_engram_fetch` 一早就 return 了），
    白跑一趟（大 L 时一次 200+ MB）。
    """
    tbl = w.get("engram.embed.weight")
    return (tbl is not None and tbl.device.type == "cpu"
            and _env_on("DS41M_ENGRAM_HOST"))


def prefetch_layer(layer_idx, ctx, w, input_ids=None):
    """提前为 engram 层发起 host 取表（**异步**，不阻塞当前层）。

    官方能把 engram 与主干重叠，靠的是它的取行是设备/DMA（实测 2.8 ms/层）；
    我们的取行在**主机**上做（~33 ms/层），单层计算只有 ~5 ms → 藏不住，
    唯一办法是**提前发起**（engram 层是 config 里固定的 [1,14]，已知）。
    这里只做纯 host 的活（ids → hash → 取表），不碰设备，安全并发。
    """
    if _skip() or layer_idx in _PREFETCH or not _host_gather_will_run(w):
        return
    c = dict(ctx)
    c["idx"] = layer_idx
    T = int(c.get("_engram_T", PREFETCH_T))
    _PREFETCH[layer_idx] = _ENG_POOL.submit(_engram_prepare, c, w, input_ids, T)


def _hash_ids(ctx, ids, device=None):
    """n-gram hash；packed 多序列时**逐序列**做再拼接。

    n-gram 的 shift 不能跨序列边界（官方每条请求单独 hash），否则 seq2 前 3 行的
    id 会吃到 seq1 末尾的 token → 取表结果错。
    """
    cu = ctx.get("cu_seqlens")
    n = int(ctx.get("n_seqs", 1) or 1)
    dev = device if device is not None else torch.device("cpu")
    if n <= 1 or not cu:
        return el.ngram_hash_ids(_cfg_path(ctx), ids.cpu(), device=dev)
    parts = [el.ngram_hash_ids(_cfg_path(ctx), ids[int(a):int(b)].cpu(), device=dev)
             for a, b in zip(cu[:-1], cu[1:])]
    return torch.cat(parts, dim=0)


def precompute_kv(ctx, w, input_ids, T, device):
    """decode 图用：在**图外**算好 engram_kv（host 取表 + H2D + 解量化）。

    与 `_engram_fetch`+`_engram_wait` 同口径（含 `engram_ctx` 的 n-gram 上文），
    但整条链都在调用线程里同步做完；返回 [T, 24×256] bf16。
    """
    c = dict(ctx)
    if input_ids is not None:
        c["input_ids"] = input_ids
    ids = _ids_of(c, int(T))
    if ids is None:
        raise RuntimeError("precompute_kv 需要 input_ids（合成模式请走原路径）")
    # ★ engram_hash 内核要求 NPU 输入（CPU 会报 device_type mismatch 12 vs 1）；
    #   哈希在设备上算，随后 host 取表时 `_engram_host_gather` 自己搬回 CPU。
    h = _hash_ids(c, ids, device=device)
    li = el.layer_index(int(c.get("idx", 1)))
    eids = h[-int(T):, li, :].to(torch.long)
    tbl = w["engram.embed.weight"]
    _check_ids_range(eids, int(tbl.shape[0]), c.get("idx", -1))
    if tbl.device.type == "npu":
        # 表常驻设备：与 `_engram_wait` 的设备分支同口径（分片 + owned 掩码 + all_reduce）
        import torch.distributed as _d
        per = int(tbl.shape[0])
        rank = int(c.get("engram_tp_rank", 0))
        tp = max(1, int(c.get("engram_tp_size", 1) or 1))
        loc = eids.to(torch.int64).to(device) - rank * per
        owned = (loc >= 0) & (loc < per)
        loc = loc.masked_fill(~owned, 0)
        rows_t = tbl.index_select(0, loc)
        sf = w.get("engram.embed.scale")
        if sf is not None:
            s8 = sf.index_select(0, loc)
            if s8.dtype != torch.uint8:
                s8 = s8.view(torch.uint8)
            scale = (s8.to(torch.int32) << 23).view(torch.float32)
            emb = (rows_t.to(torch.float32).unflatten(-1, (-1, QUANT_BLOCK))
                   * scale.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
        else:
            emb = rows_t.to(torch.bfloat16)
        emb = emb * owned.unsqueeze(-1).to(emb.dtype)
        if tp > 1 and _d.is_initialized():
            _d.all_reduce(emb, group=_d.group.WORLD)
        return emb.reshape(int(T), -1)
    if os.environ.get("DS41M_ENGRAM_VENDOR", "").strip().lower() in ("1", "on", "true", "yes"):
        # 厂商设备取行：ElasticBuffer（主机窗口 + 设备侧/RDMA fetch）；在图外同步等待。
        try:
            _vt0 = time.perf_counter() if os.environ.get("DS41M_DG_TIME") else 0.0
            vb = _vendor_buffer(w, device, c, force=True)
            if vb is not None:
                _h = vb.engram_fetch(eids.reshape(-1).to(torch.int32))
                out = _h()
                fetched, fsf = (out if isinstance(out, tuple) else (out, None))
                if fsf is None:
                    emb = fetched.to(torch.bfloat16)
                else:
                    scale = (fsf.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
                    emb = (fetched.to(torch.float32).unflatten(-1, (-1, QUANT_BLOCK))
                           * scale.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
                if _vt0:
                    print(f"  [dg-time] engram L{int(c.get('idx', -1))} vendor_fetch="
                          f"{(time.perf_counter() - _vt0) * 1000:.1f}ms", flush=True)
                emb = emb.contiguous()
                _engram_check_value("precompute", emb, eids, w, c, device)
                return emb.reshape(int(T), -1)
        except Exception as _ve:                       # noqa: BLE001
            print(f"  [engram] vendor precompute 失败 → host 取表："
                  f"{type(_ve).__name__}: {_ve}", flush=True)
    rows, sc = _engram_host_gather(eids.to(torch.int32), tbl,
                                   w.get("engram.embed.scale"))
    rows = rows.to(device)
    if sc is not None:
        scale = (sc.to(device).to(torch.int32) << 23).view(torch.float32)
        emb = (rows.to(torch.float32).unflatten(-1, (-1, QUANT_BLOCK)) * scale.unsqueeze(-1)) \
            .flatten(-2).to(torch.bfloat16)
    else:
        emb = rows.to(torch.bfloat16)
    return emb.reshape(int(T), -1)


def _engram_prepare(ctx, w, input_ids, T):
    """线程里做：算 ids（含 ngram hash）→ 主机取表 → 返回 (rows, sc)。"""
    ids = None
    if input_ids is not None:
        ids = torch.as_tensor(input_ids).reshape(-1)[:T].to(torch.long)
    tbl = w["engram.embed.weight"]
    if ids is None:                                  # 合成模式：零 id（形状对即可）
        n = (el.L["max_ngram_size"] - 1) * el.L["n_heads"]
        idx = torch.zeros((T, n), dtype=torch.long)
    else:
        h = _hash_ids(ctx, ids, device=torch.device("cpu"))   # 纯 host；packed 逐序列
        li = el.layer_index(int(ctx.get("idx", 1)))
        idx = h[:, li, :].to(torch.long)
    if int(idx.max()) >= int(tbl.shape[0]):
        raise RuntimeError(f"engram id 越界：max={int(idx.max())} 表行数={int(tbl.shape[0])}")
    return _engram_host_gather(idx.to(torch.int32), tbl, w.get("engram.embed.scale"))


_VENDOR_BUFS: dict = {}   # key → ElasticBuffer（官方同款路径：主机窗口 + 设备侧/RDMA 取行）
_VENDOR_FAILED: set = set()   # 建失败的键：别每层重试（失败会静默退回 host 取表 → trace 缺算子）
_VENDOR_GROUPS: dict = {}     # (层号, tp) → 该层专用的 HCCL 进程组（见 _vendor_group）


def _engram_tp(ctx):
    """engram 分片口径（与 weights.engram_placement / runner 的 ctx 一致）→ (tp_size, rank)。"""
    tp = max(1, int(ctx.get("engram_tp_size", 1) or 1))
    rk = int(ctx.get("engram_tp_rank", 0) or 0) % tp
    return tp, rk


def _vendor_group(tp, rk, layer_idx):
    """**每层一个 HCCL 进程组** —— 官方 `engram_tp_group_{layer_id}`（modules/engram.py:341-345）同构。

    为什么非这样不可（2026-10-16 读厂商源码 + 实测）：
      · 厂商主机窗口是**进程级共享池、按通信域（HCCL comm name）为 tag** 的
        （`gEngramSharedBuffers[contextTag]`，elastic_buffer.cpp:535-570）；重建同一个组 = **复用同一块
        窗口**（docstring：「Rebuilding an ElasticBuffer on the same group reuses the pooled memory」）。
      · DS4.1 的 engram 表在**层 1 和层 14**（两张不同的表，384006168 / 384016682 行）。若两层共用一个
        组 → 共用一块窗口 → 层 14 的 `engram_write` 会把层 1 的行覆盖掉，取回的就是**错的表**（数值崩）。
      · 于是官方给每层单独建组（tag 里带 layer_id）→ 每层一块自己的窗口，两张表**并存**。
    mini 原来用 `group.WORLD`（单一 tag），只好每次换层都重写 11.44 GiB 分片（≈12 s/pass×2 层）——
    实测 3 pass 建 6 次、每 pass 2 次，正是 pass 耗时的大头（设备侧总时长只有 5.4 s）。
    `tp != world` 时退回 WORLD（多组分片场景 mini 还没用到）；建组必须在**所有 rank 上同序调用**
    （HCCL 集合操作）—— 触发点是固定层（层 1 头、层 2 头），各 rank 序列一致，安全。
    """
    import torch.distributed as _d
    if not _d.is_initialized():
        return None
    ws = _d.get_world_size()
    if tp != ws:
        return _d.group.WORLD
    key = (int(layer_idx), tp)
    g = _VENDOR_GROUPS.get(key)
    if g is None:
        ranks = list(range(ws))
        g = _d.new_group(ranks=ranks, backend="hccl")
        _VENDOR_GROUPS[key] = g
        print(f"  [engram] 层 {layer_idx} 专用进程组就绪（engram_tp_group_{layer_idx}，"
              f"{len(ranks)} rank）—— 厂商主机窗口按通信域池化，每层一块，两层表并存（官方同构）",
              flush=True)
    return g


def _vendor_buffer(w, dev, ctx, force=False):
    """官方同款（modules/engram.py:452-489 + 529-546）：表放**主机窗口**、按 engram_tp_size 分片，
    取表由设备侧/RDMA 完成 —— trace 里就是 `EngramFetch` + `EngramFetchWait` 两个算子。

    ★ 默认口径 = **官方 hit97 那一份**（light_8card_ep8_official_all_hit*_cp1_hitfake.yaml）：
      `enable_engram_offload: True` + `engram_tp_size: 8`。官方该点实测（64k，8 rank × 4 pass 合计）
      EngramFetch 46.2 ms / EngramFetchWait 176.9 ms（中位 2.76 ms/次）——这两个算子在 mini 的
      trace 里必须看得见，否则逐算子对拍就是缺项。
    `DS41M_ENGRAM_HOST=1` 退回旧的"后台线程 host index_select"路径（数值等价，但 trace 里没有
    那两个算子）；厂商路径要求 ≥2 rank 的进程组（官方 ElasticBuffer 硬约束），不满足自动退回。
    `force=True`：decode 图的 precompute 显式要求走厂商取行时，忽略 ENGRAM_HOST。
    """
    global _VENDOR_BUFS
    if _env_on("DS41M_ENGRAM_HOST") and not force:
        return None
    tbl = w["engram.embed.weight"]
    sf = w.get("engram.embed.scale")
    tp, rk = _engram_tp(ctx)
    # ★★ 2026-10-31（主控定位 L1 全 NaN）：**分片数必须 ≤ 实际 world**。
    #   实测 CP2（world=2）沿用模型配置 engram_tp_size=8：每 rank 只写 ceil(rows/8) 行、
    #   全表只覆盖 2/8；取到"未注册 rank"的 id 时厂商缓冲区返回垃圾 ⇒
    #   `engram_proj` 之后 x 全 NaN ⇒ L1 的 moe_topk_gate 断言。
    try:
        import torch.distributed as _dt
        if _dt.is_initialized():
            _ws = int(_dt.get_world_size())
            if tp > _ws > 0:
                tp, rk = _ws, (int(rk) % _ws)
    except Exception:                                  # noqa: BLE001
        pass
    # ★ 缓存键**不能用 `id(tbl)`**：真权重下 `RealSource.get()` 每次都从 safetensors 取一个新
    #   张量对象（mmap 视图，字节零成本但 id 每次都变）→ 缓存永不命中。用「层号 + 形状 + dtype + 分片」。
    # ★ 缓存必须**按层多槽**：层 1 与层 14 的表不同、块也不同（各 12.88 GiB）——单槽会被对方顶掉，
    #   每换一层就重建一次（实测单槽时 3 pass 建 6 次，每次 ≈12 s 的 mmap 读 + 拷贝）。
    key = (int(ctx.get("idx", -1)), tuple(tbl.shape), str(tbl.dtype), tp, rk)
    if os.environ.get("DS41M_DEBUG"):                  # 探针（DS41M_DEBUG=1 时打印，查完可删）
        import sys as _s
        _fr = _s._getframe(1)
        print(f"      [dbg-vendor] pid={os.getpid()} key={key[:2]} "
              f"have={sorted(k[0] for k in _VENDOR_BUFS)} "
              f"caller={_fr.f_code.co_name}:{_fr.f_lineno}", flush=True)
    buf = _VENDOR_BUFS.get(key)
    if buf is not None:
        return buf
    if key in _VENDOR_FAILED:
        return None                                     # 已经失败过：别每层重试
    try:
        import torch.distributed as _d
        if not _d.is_initialized() or _d.get_world_size() < 2:
            print("  [engram] 厂商 ElasticBuffer 需要 ≥2 rank 的进程组 → 退回主机取表", flush=True)
            return None
        _grp = _vendor_group(tp, rk, key[0])
        if _grp is None:
            print("  [engram] 厂商 ElasticBuffer 拿不到进程组 → 退回主机取表", flush=True)
            return None
        from cann_ops_transformer.ops import ElasticBuffer
        rows, dim = int(tbl.shape[0]), int(tbl.shape[1])
        # ★★ per 必须 **ceil**（= 官方 input_size_per_partition）：真权重两层的行数不是 tp 的整数倍
        #    （384006168 / 384016682），用 floor 时 per*tp < rows —— 而 id 域是 [0, rows)，
        #    厂商内部按 `rank = id // per` 映射，最后那几行的 id 会算出 **rank=tp（不存在）**。
        #    官方用 ceil + 末片零填充（`mxfp8_shard = torch.zeros(per, cols)`）把多出来的位置变成
        #    "合法但为 0 的行"，这里照抄。
        per = -(-rows // tp)
        row0 = rk * per
        end = min(row0 + per, rows)
        slab = tbl[row0:end]                           # safetensors mmap 视图：只拷本片，不读全表
        if int(slab.shape[0]) < per:
            slab = torch.cat([slab, torch.zeros((per - int(slab.shape[0]), dim),
                                                dtype=tbl.dtype, device=slab.device)], 0)
        # ★★ 池按**所有 engram 层的最大行数**申请：厂商窗口是进程级共享池、"重建不得超首建"，
        #    而 layer14 的表比 layer1 多 6514 行（per 差 815 行 ≈ 228 KB / 2 MiB 对齐后整差 2 MiB）
        #    → 第二个建时会 rolled back（静默失效）。行数从 safetensors 头读，秒级。
        _max_rows = rows
        try:
            import weights as _wsrc
            _m = _wsrc.engram_table_rows(ST.get("cfg") or {})
            if _m:
                _max_rows = max(rows, max(int(v) for v in _m.values()))
        except Exception:                              # noqa: BLE001
            pass
        max_per = -(-_max_rows // tp)
        hint = ElasticBuffer.get_engram_storage_size_hint(max_per, dim, tbl.dtype)
        # DS41M_ENGRAM_WIN_MB：只把**注册窗口**撑大（表内容不变）——官方每卡窗口 = 384M/8 行 ≈ 13.8 GB，
        # 用来验证"EngramFetchWait 的差是不是远端窗口跨度（页/TLB）造成的"。
        _win = int(os.environ.get("DS41M_ENGRAM_WIN_MB", "0") or 0)
        if _win > 0:
            _want = ((_win * 1024 * 1024) + (2 << 20) - 1) // (2 << 20) * (2 << 20)   # 2MB 对齐
            hint = max(hint, _want)
        buf = ElasticBuffer(_grp, num_cpu_bytes=hint)   # ★ 每层自己的组（见 _vendor_group）
        _sf = None
        if sf is not None:
            # ★★ sf 要传**整张表**、不是本卡那一片（官方 modules/engram.py:433-436 就是这么造的：
            #    `sf_full = torch.full((per*tp, cols), 127)` 再把真表拷进去）。
            #    原因（2026-10-15 实测 + 厂商源码注释）：`// SF 表仅本核经映射地址直读 (参数路由):
            #    只做 host->device 映射, 不自建内存、不入 HCCL 通信域` —— 远端行的 scale 根本取不到，
            #    只传本片时远端行取回全 0/垃圾 → 下游 topk_gate 直接 device assert（NaN）。
            #    fp8 存储 + E8M0 scale，dtype 必须是 fp8_e8m0fnu（厂商硬校验）。
            _u8 = sf if sf.dtype == torch.uint8 else sf.view(torch.uint8)
            rows_pad = per * tp                       # 官方口径：整表按 per×tp 补齐（补位用 127 = 2^0）
            if int(_u8.shape[0]) < rows_pad:          # ★ per 已是 ceil → 这里不再截掉尾部行
                _u8 = torch.cat([_u8, torch.full((rows_pad - int(_u8.shape[0]), int(_u8.shape[1])),
                                                 127, dtype=torch.uint8)], 0)
            _sf = _u8[:rows_pad].contiguous().view(torch.float8_e8m0fnu).to(dev)
        buf.engram_write(slab.contiguous(), sf=_sf)
        _VENDOR_BUFS[key] = buf
        print(f"  [engram] 厂商 ElasticBuffer 就绪：层 {key[0]} rank {rk}/{tp} 片 {tuple(slab.shape)}"
              f" {tbl.dtype}  表 {rows} 行（本地 {row0}..{end}）"
              f"  本片 {slab.numel() * slab.element_size() / 2**30:.2f} GiB"
              f"  池 {hint / 2**30:.2f} GiB（按最大层 {_max_rows} 行申请）"
              f"（主机窗口 + 设备侧 RDMA 取行，官方同款 → trace 里会有 EngramFetch/EngramFetchWait）",
              flush=True)
    except Exception as exc:                              # noqa: BLE001
        print(f"  [engram] 厂商 ElasticBuffer 不可用（{type(exc).__name__}: {exc}）→ 退回主机取表",
              flush=True)
        _VENDOR_FAILED.add(key)
    return _VENDOR_BUFS.get(key)


def _start_host_gather(st, w):
    """fetch 阶段发起：后台线程做 host 取字节（纯 CPU，不碰 NPU；id 越界先查）。"""
    ids = st["_engram_ids"]
    tbl = w["engram.embed.weight"]                    # fp8，host-resident（mmap 零拷贝）
    _check_ids_range(ids, int(tbl.shape[0]), -1,
                     synthetic=bool(st.get("_engram_synth")))
    st["_engram_fut"] = _ENG_POOL.submit(_engram_host_gather, ids, tbl, w.get("engram.embed.scale"))


_SYNTH_IDS: dict = {}   # 合成模式的取表 id 缓存（键 = T/n/词表行数/rank/层）


def _synth_ids(T, n, rows, dev, ctx):
    """合成模式的路由 id：**铺满整个词表**，不是全零。

    官方那份 id 来自真实 n-gram hash（`engram_hash` 内核），在词表上均匀铺开 → 每卡取的行里
    约 7/8 落在**别的 rank** 的片上（走 RDMA），这正是 EngramFetchWait 的 2.76 ms 来源。
    全零 id 会全部落在 rank0 的片上，等于把跨卡取行抽掉 —— 对拍时这个算子会"看起来很快"。
    每 rank / 每层各自一个种子（官方就是每 rank 不同请求、每层不同 hash 槽），互相同步即可。
    """
    key = (int(T), int(n), int(rows), str(dev), int(ctx.get("idx", 0) or 0))
    t = _SYNTH_IDS.get(key)
    if t is None:
        rank = int(ctx.get("engram_tp_rank", 0) or 0)
        seed = 20261015 + rank * 977 + int(ctx.get("idx", 0) or 0) * 31
        g = torch.Generator(device="cpu").manual_seed(seed)
        t = torch.randint(0, max(1, int(rows)), (int(T), int(n)),
                          generator=g, dtype=torch.int32).to(dev)
        _SYNTH_IDS[key] = t
    return t


# ── engram 侧流（官方 `enable_engram_multi_stream` → `self.engram_stream` 的对应物）─────────
# 官方排程（modeling_deepseek.py:2145 / 2244 / 2258）：整条链 hash → fetch → wait → 解量化 →
# wkv(precompute_kv) 全在**侧流**上跑，消费层只在主干上 `wait_event` + 做 gate：
#   · 早层(1)：在**层 0 之前**就在侧流发起 → 与层 0 的 attention/MoE 重叠
#   · 晚层(14)：在早层那一层发起 → 与中间 12 层重叠
# mini 以前是单流、三个算子串在 engram 层层首，取表延迟全落在关键路径上。


def _deepep_try(ctx, w, tbl, ids):
    """DeepInfer 式取表（DeepEP EngramBuffer：HBM 分片 + 对称内存直读）。

    返回惰性 wait（() → (fetched, sf|None)）或 None（回落厂商/host 路径）。
    `DS41M_ENGRAM_MAX_TOKENS` 决定 recv buffer 上限（首建即定，后到的更大 T 自动回落）。
    """
    try:
        import engram_deepep as _td
        if tbl is None or not _td.enabled():
            return None
        tp = max(1, int(ctx.get("engram_tp_size", 1) or 1))
        rk = int(ctx.get("engram_tp_rank", 0) or 0)
        T = int(ids.shape[0])
        mt = int(os.environ.get("DS41M_ENGRAM_MAX_TOKENS", "0") or 0) or T
        if T > mt:
            print(f"  [engram] deep_ep 的 max_tokens={mt} < 本 pass T={T} → 本次回落", flush=True)
            return None
        bm = _td.make_buffer(int(ctx.get("idx", -1)), tbl, w.get("engram.embed.scale"),
                             tp, rk, mt, int(ids.shape[-1]))
        if bm is None:
            return None
        return _td.fetch(bm, ids)
    except Exception as exc:                            # noqa: BLE001
        print(f"  [engram] deep_ep 取表失败（{type(exc).__name__}: {exc}）→ 回落", flush=True)
        return None


def _engram_fetch(st, ctx, w):
    """NgramHashState → engram_hash 内核（device 上算，与官方同一算子）+ 发起取表。

    取表两条路径：① 厂商 ElasticBuffer（主机窗口 + 设备侧 RDMA 取行，官方 hit97 同款，默认）；
    ② 主机线程取表（DS41M_ENGRAM_HOST=1 或厂商不可用时兜底）。
    """
    if _skip():
        return
    if ctx.get("_dg_engram_kv") is not None:
        # decode 图：kv 已在图外算好并写进固定缓冲；图里只跑 wkv/gate。
        st["engram_kv"] = ctx["_dg_engram_kv"]
        st["_dg_engram_done"] = True
        return
    T = st["x"].shape[0]
    dev = st["x"].device
    ids = _ids_of(ctx, T)                              # ① 先算 ids（两条路径都要）
    _tbl = w.get("engram.embed.weight")
    if ids is None:                                    # 合成模式：形状/分布对即可（数值无意义）
        n = (el.L["max_ngram_size"] - 1) * el.L["n_heads"]
        rows = int(ctx.get("engram_rows", 0) or 0) or (
            int(_tbl.shape[0]) if _tbl is not None else 1)
        st["_engram_ids"] = _synth_ids(T, n, rows, dev, ctx)
        st["_engram_synth"] = True
    else:
        h = _hash_ids(ctx, ids, device=dev)                    # [n_ids, L, 24]；packed 逐序列
        li = el.layer_index(int(ctx.get("idx", 1)))
        # 取**最后 T 行**：正常 prefill 时 n_ids==T（逐位不变）；增量 decode 时 n_ids=T+ctx
        st["_engram_ids"] = h[-T:, li, :].to(torch.int32)
        st["_engram_synth"] = False
    if _tbl is not None and _tbl.device.type == "npu":
        return                                         # ★ 表常驻设备：取行在 wait 里用设备 gather 做
    _n = int(_tbl.shape[0]) if _tbl is not None else 0
    _td_wait = _deepep_try(ctx, w, _tbl, st["_engram_ids"])   # ②a DeepInfer 式（HBM + 对称内存）
    if _td_wait is not None:
        _check_ids_range(st["_engram_ids"], _n, ctx.get("idx", -1),
                         synthetic=bool(st.get("_engram_synth")))
        st["_engram_vendor_wait"] = _td_wait
        return
    _vb = _vendor_buffer(w, dev, ctx)                  # ②b 厂商 ElasticBuffer
    if _vb is not None:
        # ★ 越界检查：厂商路径以前没有（host 路径有）；真权重 id 域到 384M，越界会让设备侧读飞。
        #   2026-10-21：改为合成跳过 / 真 id 每 (层,T,rows) 首次查 —— 不再每 pass 同步（见 _check_ids_range）。
        _check_ids_range(st["_engram_ids"], _n, ctx.get("idx", -1),
                         synthetic=bool(st.get("_engram_synth")))
        st["_engram_vendor_wait"] = _vb.engram_fetch(st["_engram_ids"].reshape(-1).to(torch.int32))
        return
    _pf = _PREFETCH.pop(int(ctx.get("idx", -1)), None)  # 提前发起的 host 取表（有就直接用）
    if _pf is not None:
        st["_engram_fut"] = _pf
        return
    _start_host_gather(st, w)


def _engram_wait(st, ctx, w):
    """join 后台取表 → [T, 6144] bf16（= 官方 embeddings），H2D。"""
    if _skip():
        return
    if st.pop("_dg_engram_done", False):
        return
    T = st["x"].shape[0]
    dev = st["x"].device
    _tbl = w.get("engram.embed.weight")
    if _tbl is not None and _tbl.device.type == "npu":
        # ★ 官方同款（modules/engram.py:551-563）：本卡只认领自己那片（engram_tp 分片），
        #   越界位置填 0 并置 mask，取行/解量化都在**设备**上做，最后 all_reduce 合并。
        import torch.distributed as _d
        per = int(_tbl.shape[0])
        rank = int(ctx.get("engram_tp_rank", 0))
        tp = max(1, int(ctx.get("engram_tp_size", 1) or 1))
        ids = st["_engram_ids"].reshape(-1).to(torch.int64)
        local = ids - rank * per
        owned = (local >= 0) & (local < per)
        local = local.masked_fill(~owned, 0)
        rows_t = _tbl.index_select(0, local)
        _sf = w.get("engram.embed.scale")
        if _sf is not None:
            s8 = _sf.index_select(0, local)
            if s8.dtype != torch.uint8:
                s8 = s8.view(torch.uint8)
            scale = (s8.to(torch.int32) << 23).view(torch.float32)          # 官方位技巧 2^(b-127)
            emb = (rows_t.to(torch.float32).unflatten(-1, (-1, QUANT_BLOCK)) * scale.unsqueeze(-1)) \
                .flatten(-2).to(torch.bfloat16)
        else:
            emb = rows_t.to(torch.bfloat16)
        emb = emb * owned.unsqueeze(-1).to(emb.dtype)                       # 官方：* owned
        if tp > 1 and _d.is_initialized():
            _d.all_reduce(emb, group=_d.group.WORLD)                        # 官方：跨片合并
        st["engram_kv"] = emb.reshape(T, -1)
        return
    _vw = st.pop("_engram_vendor_wait", None)
    if _vw is not None:                                    # 官方同款：wait → (字节, scale) → 卡上解量化
        out = _vw()
        fetched, fetched_sf = (out if isinstance(out, tuple) else (out, None))
        if fetched_sf is not None:
            # ★ 2026-10-26（T-023 / 对齐 DeepInfer 昇腾实现）：fetch 的 fp8 行 + e8m0 sf
            #   **不再展开**（旧实现的 `(sf<<23).view(f32)` + 逐元素乘 + bf16 物化已删除：
            #   256K@60 上那是 2×231ms/forward + 1.9GB int32 中间态）。改由 `_engram_wkv`
            #   直接喂 `npu_quant_matmul`（pertoken scale 形状 (M,K/64,2)，即 DeepInfer 的
            #   2×e8m0/int16 打包布局）。实测 4690→4381.7ms、峰值 48.81→47.60GiB。
            st["_engram_q_raw"] = fetched
            st["_engram_sf_raw"] = fetched_sf
            st["engram_kv"] = None
            return
        emb = fetched.to(torch.bfloat16).contiguous()      # 无 scale 的兜底（fp8 表未量化）
        _engram_check_value("wait", emb, st["_engram_ids"], w, ctx, dev)
        st["engram_kv"] = emb.reshape(T, -1)
        return
    fut = st.pop("_engram_fut", None)
    if fut is None:                                   # 兼容：没有 fetch 的路径走同步
        _start_host_gather(st, w)
        fut = st.pop("_engram_fut")
    _dg_t0 = time.perf_counter() if os.environ.get("DS41M_DG_TIME") else 0.0
    rows, sc = fut.result()
    _dg_t1 = time.perf_counter() if _dg_t0 else 0.0
    rows = rows.to(dev)                                # 全量 H2D（fp8 字节，与官方搬运量一致）
    if sc is not None:
        # 官方同款：`byte << 23` 直接拼出 fp32 的 2^(byte-127)（省掉 pow）
        scale = (sc.to(dev).to(torch.int32) << 23).view(torch.float32)         # [N, 192]
        emb = (rows.to(torch.float32).unflatten(-1, (-1, QUANT_BLOCK)) * scale.unsqueeze(-1)) \
            .flatten(-2).to(torch.bfloat16)
    else:
        emb = rows.to(torch.bfloat16)
    st["engram_kv"] = emb.reshape(T, -1)               # [T, 24×256 = 6144]
    if _dg_t0:
        print(f"  [dg-time] engram L{ctx.get('idx')} gather={(_dg_t1 - _dg_t0) * 1000:.1f}ms "
              f"h2d+deq={(time.perf_counter() - _dg_t1) * 1000:.1f}ms", flush=True)
    if os.environ.get("DS41M_DEBUG"):
        print(f"      [engram] 取表 {tuple(st['engram_kv'].shape)} "
              f"absmax={st['engram_kv'].float().abs().max().item():.4f}", flush=True)


def _engram_wkv(st, ctx, w):
    """官方 `precompute_kv`：wkv(QMM) + fused_weight。**侧流上做**（官方就在 engram_stream 上）。"""
    H = ctx["H"]
    M = ctx["hc_mult"]
    n_cols = (el.L["max_ngram_size"] - 1) * el.L["n_heads"] * el.L["head_dim"]
    kv = st.get("engram_kv")
    if st.get("_engram_q_raw") is not None:
        # ★ 融合：fetch 的 fp8 行 + e8m0 sf 直接进 QMM（跳过 `<<23` 展开与 bf16 物化）
        import torch_npu as _tn
        _q = st.pop("_engram_q_raw")
        _s = st.pop("_engram_sf_raw").view(torch.uint8)
        _W, _WS = w["_packed"]["engram.wkv.weight"]
        _n = int(n_cols)          # ★ 用逻辑 K（6144），不要用 W.shape[-1]（打包后是 N=25600）
        _Tq = int(_q.numel() // _n)
        # ★ 行分块：M=104998（256K@60）时单次 QMM 会 aicore 507015（实测），分块后与
        #   参考路径的 M 分块口径一致（参考也是 dynamic_mx_quant+QMM 按 M 分块跑）。
        _ch = int(os.environ.get("DS41M_ENGRAM_QMM_CHUNK", "8192") or 0)
        _x1 = _q.reshape(_Tq, _n)
        _sf = _s.reshape(_Tq, -1, 2)                      # 参考路径 xs=(M,K/64,2)
        if _ch > 0 and _Tq > _ch:
            _parts = []
            for _s0 in range(0, _Tq, _ch):
                _s1 = min(_Tq, _s0 + _ch)
                _parts.append(_tn.npu_quant_matmul(
                    _x1[_s0:_s1], _W, _WS, pertoken_scale=_sf[_s0:_s1],
                    output_dtype=torch.bfloat16,
                    scale_dtype=_tn.float8_e8m0fnu,
                    pertoken_scale_dtype=_tn.float8_e8m0fnu,
                    group_sizes=[1, 1, QUANT_BLOCK]))
            proj = torch.cat(_parts, 0) if len(_parts) > 1 else _parts[0]
        else:
            proj = _tn.npu_quant_matmul(
                _x1, _W, _WS, pertoken_scale=_sf,
                output_dtype=torch.bfloat16,
                scale_dtype=_tn.float8_e8m0fnu,
                pertoken_scale_dtype=_tn.float8_e8m0fnu,
                group_sizes=[1, 1, QUANT_BLOCK])
    else:
        if kv is None or kv.shape[-1] != n_cols:
            ph = st.get("_placeholder_used")
            if isinstance(ph, set):
                ph.add("engram_proj")
            kv = torch.zeros((st["x"].shape[0], n_cols), dtype=torch.bfloat16, device=st["x"].device)
        proj = core_qmm(kv, "engram.wkv.weight", w, ctx)         # [T, (M+1)*H]
    kv5 = proj.reshape(-1, M + 1, H).to(torch.bfloat16)
    from tile_kernels.engram import fused_weight
    wf = fused_weight(w["engram.q_weight"].to(torch.bfloat16),
                      w["engram.k_weight"].to(torch.bfloat16))
    return kv5, wf


def _engram_gate(st, ctx, w, kv5, wf):
    """官方 `engram_gate_fwd`（**主干**上、依赖 hidden_states）：覆盖残差流。"""
    from tile_kernels.engram import engram_gate_fwd
    M = ctx["hc_mult"]
    x = st["x"]
    x4 = x if x.dim() == 3 else x.unsqueeze(1).expand(-1, M, -1).contiguous()
    out = engram_gate_fwd(x4.contiguous(), kv5, wf, float(ctx.get("eps", 1e-20)), 1e-6,
                          save_for_backward=False, image_token_mask=None, inplace=False)[0]
    st["x"] = out                                                # ★ 覆盖残差流
    # ★ 官方顺序：engram 先跑，**之后**才 `residual = hidden_states`
    #   （DecoderLayer.forward: engram → residual = hidden_states → hc_pre）
    #   → engram 层的残差是**截断后**的值；mini 原来在层入口就克隆了残差 ✗
    st["residual"] = out.clone()
    st["engram_out"] = out
    if os.environ.get("DS41M_DEBUG"):
        print(f"      [engram/proj] x{tuple(x4.shape)} kv{tuple(kv5.shape)} → "
              f"{tuple(out.shape)} absmax={out.float().abs().max().item():.4f}", flush=True)


def _engram_proj(st, ctx, w):
    """wkv(QMM) → kv [T,5,H]；engram_gate_fwd → **覆盖残差流**（官方 DecoderLayer 第一步）。

    单流（2026-10-26 起，对齐 DeepInfer）：本层现算（融合 QMM 已在 `_engram_wkv` 里）。
    """
    if _skip():
        return
    kv5, wf = _engram_wkv(st, ctx, w)
    st["engram_proj_out"] = kv5
    _engram_gate(st, ctx, w, kv5, wf)


core_qmm = None


def register(OPS, SKIP, core, cfg=None):
    global core_qmm
    core_qmm = core._qmm
    _apply_spec(cfg)                     # ★ 常量对齐 config（默认值不变）
    ST["cfg"] = cfg
    ST["path"] = None
    OPS["engram_fetch"] = _engram_fetch
    OPS["engram_wait"] = _engram_wait
    OPS["engram_proj"] = _engram_proj
    for k in ("engram_fetch", "engram_wait", "engram_proj"):
        SKIP.pop(k, None)

# ── 点位边界（core/memlife.py）：按 T/层 缓存的路由 id（[T,n] int32）──
# 厂商 ElasticBuffer（`_VENDOR_BUFS`）**不登记**：键是层号+表形状，与点位几何无关，
# 重建一次要 ~12 s 重写 12.88 GiB 窗口 —— 跨点位复用正是收益。
POINT_CACHES = (_SYNTH_IDS,)

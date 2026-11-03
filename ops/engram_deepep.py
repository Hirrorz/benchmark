"""DeepInfer 式 engram 取表：DeepEP-Ascend `EngramBuffer`（HBM 分片 + 对称内存直读）。

与厂商 ElasticBuffer 路径（host 窗口 + EngramFetch/EngramFetchWait 两算子）的差别：
  · 表放 **NPU HBM**（`aclrtMalloc` 的对称内存），不是 host；
  · 取行是 **AscendC vector kernel**：`global_idx / per_rank` 定位源 rank，
    `get_sym_ptr` 解析对端同一 offset，`asc_copy_gm2ub_align` 直接 GM→UB；
  · 一次 launch 覆盖**本层全部 token**，host 只拿一个惰性 hook。

结构照 `DeepInfer/deepinfer/model_executor/models/v4_modules/engram.py`：
每层一个 buffer（DeepInfer 是所有层共用一个；mini 的权重是**按层懒加载**的，
所以这里保持"每层一个 buffer、首次用到时写表"，kernel/布局与 DeepInfer 完全一致）。

开关：`DS41M_ENGRAM_IMPL=deepep`（默认仍是厂商 ElasticBuffer）。不可用时返回 None，
调用方自动回落（与 `_vendor_buffer` 同一契约）。
"""
from __future__ import annotations

import os

import torch

_ALIGN = 2 << 20            # 2 MiB：`_C.get_num_allocation_alignment()` 的实测值（冒烟确认）

_BUFS: dict = {}            # key → (buffer, meta)
_FAILED: set = set()


def enabled() -> bool:
    return os.environ.get("DS41M_ENGRAM_IMPL", "").strip().lower() in (
        "deepep", "deepinfer", "1", "on", "true", "yes")


def _align_up(x: int) -> int:
    return (int(x) + _ALIGN - 1) // _ALIGN * _ALIGN


def _sf_packs(hidden: int) -> int:
    """每行 sf 的"包"数：每 32 列一个 ue8m0，Ascend 上 2 个 ue8m0 打进 1 个 int16。"""
    return (hidden // 32) // 2


def group_of(tp: int):
    import torch.distributed as dist
    if not dist.is_initialized():
        return None
    if tp == dist.get_world_size():
        return dist.group.WORLD
    return None                                   # mini 只支持 tp==world（与厂商路径一致）


def make_buffer(layer_idx: int, tbl: torch.Tensor, sf: torch.Tensor | None,
                tp: int, rank: int, max_tokens: int, ept: int):
    """给某一层建 buffer 并把本卡分片写进去。失败返回 None（调用方回落）。"""
    if not enabled():
        return None
    key = (int(layer_idx), tuple(tbl.shape), str(tbl.dtype), int(tp), int(rank))
    if key in _BUFS:
        return _BUFS[key]
    if key in _FAILED:
        return None
    try:
        from deep_ep import EngramBuffer

        grp = group_of(tp)
        if grp is None:
            print(f"  [engram] deep_ep 需要 tp==world 的进程组（tp={tp}）→ 回落", flush=True)
            _FAILED.add(key)
            return None

        rows, hidden = tbl.shape[0], tbl.shape[1]
        per = -(-rows // int(tp))                       # ceil：官方 input_size_per_partition
        r0 = int(rank) * per
        r1 = min(r0 + per, rows)
        slab = tbl[r0:r1]
        if slab.shape[0] < per:                    # 末片零填充（官方同款）
            slab = torch.cat([slab, torch.zeros((per - slab.shape[0], hidden),
                                                dtype=tbl.dtype)], 0)
        slab = slab.contiguous()

        esize = slab.element_size()
        num_sf = _sf_packs(hidden) if (sf is not None and esize == 1) else 0
        nsp = 1                                         # 本 buffer 只放"一层"
        recv = _align_up(nsp * int(max_tokens) * int(ept) * hidden * esize)
        storage = _align_up(int(per) * hidden * esize)
        # ★ 1G 大页在 torch 缓存分配器切碎后常常拿不到（2026-10-27 真权重 8 卡实测：
        #   31.75 GiB/91.8 GiB 仍报 EL0019 分配 11.54 GiB 失败）→ 默认关掉，回落 HUGE_FIRST。
        _huge1g = os.environ.get("DS41M_ENGRAM_HUGE1G", "0").strip().lower() in ("1", "on", "true", "yes")
        buf = EngramBuffer(grp, recv, storage, False, 0, 0,
                           explicitly_destroy=True,      # 显式销毁：必须在进程组活着时
                           use_huge1g_page=_huge1g)
        buf.set_config([int(per)], hidden, int(max_tokens), int(ept), slab.dtype, num_sf)
        sfs = None
        if num_sf:
            k = hidden // 32
            full = sf if sf.shape[0] >= per * int(tp) else torch.cat(
                [sf, torch.zeros((per * int(tp) - sf.shape[0], k),
                                 dtype=sf.dtype)], 0)
            full = full[: per * int(tp)]
            u8 = full.view(torch.uint8) if full.dtype != torch.uint8 else full
            _dev = torch.device("npu:%d" % torch.npu.current_device())
            sfs = [u8.contiguous().to(_dev).view(torch.int16)]   # 2×ue8m0/int16（必须在 NPU 上）
        buf.write([slab], sfs=sfs)
        meta = dict(per=per, hidden=hidden, dtype=slab.dtype, esize=esize,
                    ept=int(ept), max_tokens=int(max_tokens), num_sf=num_sf)
        _BUFS[key] = (buf, meta)
        print(f"  [engram] DeepEP EngramBuffer 就绪：层 {layer_idx} rank {rank}/{tp} "
              f"片 {tuple(slab.shape)} 行 {rows}（本地 {r0}..{r1}）"
              f" 本片 {slab.numel() * esize / 2**30:.2f} GiB（HBM）"
              f" recv {recv / 2**20:.1f} MiB sf_packs={num_sf}", flush=True)
        return _BUFS[key]
    except Exception as exc:                            # noqa: BLE001
        print(f"  [engram] DeepEP EngramBuffer 不可用（{type(exc).__name__}: {exc}）→ 回落", flush=True)
        _FAILED.add(key)
        return None


def fetch(buf_and_meta, ids: torch.Tensor):
    """ids: [T, ept] int32（全局行号）→ 惰性 hook（调用后返回 (fp8/bf16 张量, sf 或 None)）。"""
    buf, meta = buf_and_meta
    n, ept = ids.shape[0], ids.shape[-1]
    assert ept == meta["ept"], (ept, meta["ept"])
    dev = torch.device("npu:%d" % torch.npu.current_device())
    idx3 = ids.reshape(1, n, ept).to(dtype=torch.int32, device=dev).contiguous()
    hooks = buf.fetch(idx3, use_tma_aligned_col_major_sf=False)   # 行主序，与厂商路径的 sf 视图对齐

    def _wait():
        out = hooks[0]()
        if isinstance(out, tuple):
            fetched, sf = out
            return fetched, (sf if meta["num_sf"] else None)
        return out, None

    return _wait


def destroy_all():
    for key, (buf, _meta) in list(_BUFS.items()):
        try:
            buf.destroy()
        except Exception:                               # noqa: BLE001
            pass
        _BUFS.pop(key, None)

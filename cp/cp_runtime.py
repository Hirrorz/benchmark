"""DeepInfer 风格 CP 的运行时胶水：建 CP 组 + rank-major all-gather 还原。

与 cp_layout.py 分工：
  · cp_layout  纯 Python 布局（head-tail/mirror 索引），CPU 可回归；
  · cp_runtime 只做 dist 通信与设备张量，不碰模型算子。

all_gather_global 的语义与 DeepInfer `CPManager.all_gather` + `mirror_cp_reverse_index`
一致：先拼成 rank-major `[rank0 行 | rank1 行 | ...]`，再按 reverse_index 还原全局顺序。
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

from cp_layout import CpPlan, rank_layout


_CP_GROUPS: dict = {}

# 诊断（DS41M_CP_AG_TIME=1）：统计 CP all-gather 的**时长/字节/形状**，用于量化
# "gather hidden vs projected KV" 在 CP 层耗时里的占比（同步计时，会轻微扰动）。
_AG_STATS: dict = {}


def ag_stats_reset() -> None:
    _AG_STATS.clear()


def ag_stats() -> dict:
    return {k: dict(v) for k, v in _AG_STATS.items()}


def init_cp_group(world_size: int, cp_size: int, backend: str = "hccl", rank: int | None = None):
    """建 CP 组（连续分组）并返回本 rank 的组。所有 rank 必须按相同顺序调用。

    组按 (world, cp, backend) 缓存：同一个进程多次 forward 只建一次 HCCL 组。
    """
    if rank is None:
        rank = dist.get_rank()
    key = (int(world_size), int(cp_size), str(backend))
    cached = _CP_GROUPS.get(key)
    if cached is not None:
        return cached
    layout = rank_layout(world_size, cp_size)
    my_group = None
    for ranks in layout["cp_groups"]:
        group = dist.new_group(ranks, backend=backend)
        if rank in ranks:
            my_group = group
    if my_group is None:
        raise RuntimeError(f"rank {rank} 不在任何 CP 组（world={world_size}, cp={cp_size}）")
    _CP_GROUPS[key] = my_group
    return my_group


def _plan_tensor(plan: CpPlan, name: str, values, device) -> torch.Tensor:
    """plan 生命周期内缓存 device 侧索引张量（`reverse_index` / `inverse_reverse`）。

    ★ 2026-10-25 空泡根因：旧实现每次 all-gather 都 `torch.tensor(python_tuple,
    device=npu)` —— list→tensor 是 host CPU，H2D 又是**阻塞拷贝**（要等设备队列
    排空）。CP2 64K 实测每 forward 39 次 ≥10ms 的 `aten::to`（单次 20–115ms，
    host 被钉住），trace 里还表现为 kernel 前的 2ms 级空泡。plan 由 runner 按几何
    缓存（`_cp_state_cached`），这里只建一次；值不变 ⇒ 逐位不变。
    """
    t = plan._rt.get(name)
    dev = torch.device(device)
    if t is None or t.device != dev:
        t = torch.tensor(values, dtype=torch.long, device=dev)
        plan._rt[name] = t
    return t


# ★ DeepInfer userbuffer 式固定缓冲（2026-10-25）：per-plan 缓存 gather 的 full/g，
#   同一形状只向驱动分配一次，后续层/forward 复用（结果 alias 缓冲，寿命=单个算子）。
#   > `DS41M_CP_BUCKET_GB`（默认 4 GiB）不缓存（1M 的 10.7GB 走原分配，避免顶墙）。
_CP_BUCKET_BYTES = int(float(os.environ.get("DS41M_CP_BUCKET_GB", "4") or 0) * (1 << 30))


def _cached_buf(plan: CpPlan, key: str, shape, dtype, device):
    nbytes = 1
    for s in shape:
        nbytes *= int(s)
    nbytes *= torch.empty((), dtype=dtype).element_size()
    if _CP_BUCKET_BYTES <= 0 or nbytes > _CP_BUCKET_BYTES:
        return None
    k = ("cbuf", key, str(device))
    t = plan._rt.get(k)
    if t is not None and tuple(t.shape) == tuple(shape) and t.dtype == dtype:
        return t
    t = torch.empty(tuple(shape), dtype=dtype, device=device)
    plan._rt[k] = t
    return t


def release_plan_buffers(plan: CpPlan) -> None:
    """释放 plan 上的固定缓冲（runner 的 plan LRU 淘汰时调用）。"""
    for k in [k for k in list(plan._rt) if isinstance(k, tuple) and k and k[0] == "cbuf"]:
        plan._rt.pop(k, None)


def all_gather_global(x_local: torch.Tensor, plan: CpPlan, group,
                      chunk: int | None = None,
                      buf_key: str | None = None) -> torch.Tensor:
    """把本 rank 的 `[local_rows, ...]` all-gather 后按全局 token 顺序返回。

    x_local 的第 0 维必须等于 plan.local_rows（含 pad 行，pad 值任意，reverse 不会选它）。
    """
    if x_local.shape[0] != plan.local_rows:
        raise ValueError(f"x_local 行数 {x_local.shape[0]} != plan.local_rows {plan.local_rows}")
    x = x_local.contiguous()
    world = dist.get_world_size(group) if dist.is_initialized() else 1
    if world != plan.cp_size:
        raise ValueError(f"CP 组大小 {world} != plan.cp_size {plan.cp_size}")
    if world == 1:
        idx = _plan_tensor(plan, "reverse", plan.reverse_index, x.device)
        return x.index_select(0, idx)

    import os
    _ts = os.environ.get("DS41M_CP_AG_TIME", "").strip().lower() in (
        "1", "on", "true", "yes")
    if _ts:
        torch.npu.synchronize()
        import time as _tm
        _t0 = _tm.perf_counter()
    # ★ 分块 all-gather（2026-10-23，1M 显存）：原实现 parts+cat+index_select 同时
    #   存 3 份全长张量（1M×5120 bf16 ≈ 32 GiB）→ 1M@CP8 在层 8 OOM。
    #   分块：all_gather 只开 [world×chunk] 一块，按 inverse_reverse 直接写进 out，
    #   峰值 ≈ 1 份全长 + 1 个 chunk；**字节不变 ⇒ 数值逐位不变**。
    #   有 pad 行时 inverse_reverse 为 -1（无法直接 scatter）→ 退回旧实现。
    #   ★ 2026-10-25：只要没 pad 就走这条散写路径（不再要求单块超 chunk）。
    #   旧的 cat+index_select 分支要多两趟全长读写（一次 cat、一次 gather），
    #   且末尾 `torch.tensor(reverse_index)` 也是阻塞 H2D —— CP4/CP8 64K
    #   （S ≤ chunk）以前全落在旧分支上，forward 尾部 6–8ms 全花在这。
    _ch = (int(chunk) if chunk is not None
           else int(os.environ.get("DS41M_CP_AG_CHUNK", "16384") or 0))
    S = int(x.shape[0])
    # ★ 2026-10-31：`all(v>=0 ...)` 是 O(world*S) 的 Python 扫描（plain 64k 时 65536 项/
    #   次 × 40 层）。它是 plan 的**几何常量** ⇒ 每 plan 只算一次（plan._rt 缓存）。
    _inv_ok = plan._rt.get("inv_ok")
    if _inv_ok is None:
        _inv_ok = (len(plan.inverse_reverse) == world * int(plan.local_rows)
                   and all(v >= 0 for v in plan.inverse_reverse))
        plan._rt["inv_ok"] = _inv_ok
    _scatter = _ch > 0 and _inv_ok
    if _scatter:
        _rest = tuple(x.shape[1:])
        full = (_cached_buf(plan, f"{buf_key}:full", (plan.total_q,) + _rest,
                            x.dtype, x.device) if buf_key else None)
        if full is None:
            full = torch.empty((plan.total_q,) + _rest,
                               dtype=x.dtype, device=x.device)
        inv = _plan_tensor(plan, "inv", plan.inverse_reverse, x.device)
        for s0 in range(0, S, _ch):
            e0 = min(S, s0 + _ch)
            n = e0 - s0
            part = x[s0:e0]
            g = (_cached_buf(plan, f"{buf_key}:g{n}", (world * n,) + _rest,
                            x.dtype, x.device) if buf_key else None)
            if g is None:
                g = torch.empty((world * n,) + _rest,
                                dtype=x.dtype, device=x.device)
            _th = os.environ.get("DS41M_HOSTTIME", "").strip().lower() in ("1", "on", "true", "yes")
            _tc = __import__("time").perf_counter() if _th else 0.0
            try:
                dist.all_gather_into_tensor(g, part.contiguous(), group=group)
            except Exception:                           # noqa: BLE001
                parts = [torch.empty_like(part) for _ in range(world)]
                dist.all_gather(parts, part, group=group)
                g = torch.cat(parts, dim=0)
            if _tc:
                try:
                    import exec_ops as _eo
                    _eo._ht_add("ag.hccl", __import__("time").perf_counter() - _tc)
                except Exception:                   # noqa: BLE001
                    pass
            for r in range(world):
                full.index_copy_(0, inv[r * S + s0:r * S + e0],
                                 g[r * n:(r + 1) * n])
        out = full
    else:
        # ★★ 2026-10-31（host-bound 第 1 刀）：旧回退是
        #   `dist.all_gather(parts, x)`（list API，world 次 collective）+ `cat` +
        #   `index_select` —— 实测 143.5µs + 35.4µs + 21.6µs/次，而 pad 只发生在
        #   命中/尾段 plan（S 小）。改成：**一个** `all_gather_into_tensor` 进固定缓冲，
        #   再按 `inverse_reverse >= 0` 掩码散写（pad 行跳过）；字节/数值与原实现**逐位相同**。
        #   `DS41M_CP_AG_PADSCATTER=0` 回退旧实现（A/B 用）。
        _padsc = os.environ.get("DS41M_CP_AG_PADSCATTER", "1").strip().lower() not in (
            "0", "off", "false", "no")
        _th = os.environ.get("DS41M_HOSTTIME", "").strip().lower() in ("1", "on", "true", "yes")
        _tm2 = __import__("time")
        _tA = _tm2.perf_counter() if _th else 0.0
        if _padsc:
            _rest = tuple(x.shape[1:])
            g = _cached_buf(plan, f"{buf_key}:pg{world}x{S}", (world * S,) + _rest,
                            x.dtype, x.device)
            if g is None:
                g = torch.empty((world * S,) + _rest, dtype=x.dtype, device=x.device)
            _tB = _tm2.perf_counter() if _th else 0.0
            dist.all_gather_into_tensor(g, x, group=group)
            _tC = _tm2.perf_counter() if _th else 0.0
            # gathered 缓冲就是 rank 序拼接（= 旧 cat 的结果）⇒ 一次 index_select 还原全局序。
            # pad 行不会出现在 reverse_index 里，无需掩码；全程 1 个集合通信 + 1 个 gather。
            idx = _plan_tensor(plan, "reverse", plan.reverse_index, x.device)
            out = g.index_select(0, idx)
            if _th:
                try:
                    import exec_ops as _eo
                    _eo._ht_add("ag.alloc", _tB - _tA)
                    _eo._ht_add("ag.hccl", _tC - _tB)
                    _eo._ht_add("ag.post", _tm2.perf_counter() - _tC)
                except Exception:                   # noqa: BLE001
                    pass
        else:
            parts = [torch.empty_like(x) for _ in range(world)]
            _tB = _tm2.perf_counter() if _th else 0.0
            dist.all_gather(parts, x, group=group)
            _tC = _tm2.perf_counter() if _th else 0.0
            full = torch.cat(parts, dim=0)
            idx = _plan_tensor(plan, "reverse", plan.reverse_index, x.device)
            out = full.index_select(0, idx)
            if _th:
                try:
                    import exec_ops as _eo
                    _eo._ht_add("ag.alloc", _tB - _tA)
                    _eo._ht_add("ag.list", _tC - _tB)
                    _eo._ht_add("ag.post", _tm2.perf_counter() - _tC)
                except Exception:               # noqa: BLE001
                    pass
    if _ts:
        torch.npu.synchronize()
        _dt = _tm.perf_counter() - _t0
        _key = int(x.shape[1]) if x.dim() > 1 else 0
        _d = _AG_STATS.get(_key)
        if _d is None:
            _d = _AG_STATS[_key] = {"n": 0, "t": 0.0, "bytes": 0}
        _d["n"] += 1
        _d["t"] += _dt
        _d["bytes"] += int(x.numel()) * int(x.element_size()) * world
    return out


def _row_source(p: int, ps: int, cp_size: int) -> tuple:
    """全局位置 `p` 的值在哪个 rank 的本地第几行：`(owner_rank, local_row)`。

    mirror 布局：块 `b = p // ps`，持有者 `min(b, 2·cp−1−b)`；
    head 块（`b < cp`）在本 rank 的 `[0, ps)`，tail 块在 `[ps, 2·ps)`。
    """
    b = int(p) // int(ps)
    owner = int(min(b, 2 * int(cp_size) - 1 - b))
    base = 0 if b < int(cp_size) else int(ps)
    return owner, base + (int(p) - b * int(ps))


def merge_global_rows(x_local: torch.Tensor, plan: CpPlan, pos_lo: int, pos_hi: int,
                      group=None, buf_key: str | None = None) -> torch.Tensor:
    """**只取**全局位置 `[pos_lo, pos_hi)` 的行（B5 / GPT `partition=mirror_aligned` 的落地）。

    返回紧凑张量，第 i 行 = 全局位置 `pos_lo + i` 的值（与 `merge_global(...)[pos_lo:pos_hi]`
    **逐位相同**是硬要求，见 `DS41M_CP_KV_GATHER=check`）。

    只支持**单序列** plan：多序列时"全局位置"需要 cu 换算，等 G7/G10 的等长限制解除后再说；
    这里显式拒绝（不静默降级）。
    """
    if len(plan.seq_lens) != 1:
        raise ValueError(
            f"merge_global_rows 只支持单序列 plan（当前 {len(plan.seq_lens)} 个）；"
            f"多序列请继续用 merge_global")
    L = int(plan.seq_lens[0])
    ps = int(plan.local_lens[0]) // 2
    lo = max(0, int(pos_lo))
    hi = min(int(pos_hi), L)
    if hi <= lo:
        return x_local[:0]
    if ps <= 0:
        raise ValueError(f"merge_global_rows: ps={ps} 非法（plan.local_lens={plan.local_lens}）")
    world = dist.get_world_size(group) if dist.is_initialized() else 1
    if world == 1:
        return x_local[lo:hi]

    pos = list(range(lo, hi))
    src = [_row_source(p, ps, plan.cp_size) for p in pos]
    me = int(plan.cp_rank)
    my_rows = [r for (o, r) in src if o == me]
    dev = x_local.device
    if my_rows:
        _idx = torch.tensor(my_rows, dtype=torch.long, device=dev)
        send = x_local.index_select(0, _idx).contiguous()
    else:
        send = x_local[:0]
    # ① 交换各 rank 的发送行数（用 uint8 占位张量，避免 dtype 差异）
    _cnt = torch.tensor([len(my_rows)], dtype=torch.long, device=dev)
    _cnts = [torch.zeros_like(_cnt) for _ in range(world)]
    dist.all_gather(_cnts, _cnt, group=group)
    cnts = [int(v.item()) for v in _cnts]
    maxn = max(1, max(cnts))
    # ② pad 到 maxn 后一次性 all_gather_into_tensor
    #   ★ 注意 API 语义：`inp` 是**本地分片**（maxn 行），`out` 是 `world*maxn` 行。
    _tail = tuple(send.shape[1:])
    _inp = torch.zeros((maxn,) + _tail, dtype=x_local.dtype, device=dev)
    if len(my_rows):
        _inp[:len(my_rows)] = send
    _out = torch.empty((world * maxn,) + _tail, dtype=x_local.dtype, device=dev)
    dist.all_gather_into_tensor(_out, _inp.contiguous(), group=group)
    # ③ 每行在"本 rank 发送序列"里的序号（所有 rank 算得一致）
    _rank_counts = cnts
    _seen = [0] * world
    _rank_of = []
    for (o, _r) in src:
        _rank_of.append(o)
    _k = []
    for o in _rank_of:
        _k.append(_seen[o])
        _seen[o] += 1
    _gidx = [o * maxn + k for o, k in zip(_rank_of, _k)]
    _gt = torch.tensor(_gidx, dtype=torch.long, device=dev)
    return _out.index_select(0, _gt).contiguous()


# ── 侧流 async all-gather（2026-10-28，DeepInfer `cp.all_gather()` 的 wait-句柄同款）──
#   （实测 256k CP2 h0.97：主流水线等待占 AG 时长 84%，4.5ms/forward）。
#   现在 kv rope 一完成就在**侧流**发起，`_kv_epilog` 只 wait event。数值同一条
#   all-gather（单块、字节不变），逐位不变。
_AG_SIDE_STREAM = [None]


def ag_side_stream():
    """专用侧流（与 runner 的压缩器预取侧流分开：同一条流上排队会互相拖）。"""
    s = _AG_SIDE_STREAM[0]
    if s is None:
        s = torch.npu.Stream()
        _AG_SIDE_STREAM[0] = s
    return s


def all_gather_global_async(x_local: torch.Tensor, plan: CpPlan, group,
                            buf_key: str | None = None):
    """在侧流上**发起** all-gather，立刻返回 `(out, event)`；消费方 `cur.wait_event(event)`。

    与 `all_gather_global` 的唯一差别是执行的流；单块（chunk=local_rows）——侧流上
    分多块会让多个 collective 与主流交错互等（见 runner 的 CP 预取注释，实测 14ms 停顿）。
    """
    world = dist.get_world_size(group) if dist.is_initialized() else 1
    x = x_local.contiguous()
    if world <= 1 or world != plan.cp_size:
        return all_gather_global(x, plan, group, buf_key=buf_key), None
    S = int(x.shape[0])
    # 索引/固定缓冲在**主流**先建好（跨流复用要向 allocator 登记；见 runner 预取同处）
    _ok = (len(plan.inverse_reverse) == plan.cp_size * S
           and all(v >= 0 for v in plan.inverse_reverse))
    if _ok:
        _plan_tensor(plan, "inv", plan.inverse_reverse, x.device)
    else:
        _plan_tensor(plan, "reverse", plan.reverse_index, x.device)
    cur = torch.npu.current_stream()
    side = ag_side_stream()
    side.wait_stream(cur)
    try:
        x.record_stream(side)
    except Exception:                               # noqa: BLE001
        pass
    # 单块只在 S 不太大时用：单块缓冲 = world×S 行（1M/CP8 时 ≈1 GiB），
    # 超过阈值退回默认分块（字节不变 ⇒ 数值不变）。
    _max_rows = int(os.environ.get("DS41M_CP_KV_PREF_MAX_ROWS", "65536") or 65536)
    import time as _tm
    _ht = os.environ.get("DS41M_HOSTTIME", "").strip().lower() in ("1", "on", "true", "yes")
    _t0 = _tm.perf_counter() if _ht else 0.0
    with torch.npu.stream(side):
        out = all_gather_global(x, plan, group,
                                chunk=(S if S <= _max_rows else None), buf_key=buf_key)
        _t1 = _tm.perf_counter() if _ht else 0.0
        ev = torch.npu.Event()
        ev.record()
        if _ht:
            _t2 = _tm.perf_counter()
            try:
                import exec_ops as _eo
                _eo._ht_add("ag.call", _t1 - _t0)
                _eo._ht_add("ag.ev", _t2 - _t1)
            except Exception:                       # noqa: BLE001
                pass
    return out, ev


def cp_rank_in_group(world_size: int, cp_size: int, rank: int) -> int:
    """CP 组连续，所以组内 rank = global_rank % cp_size。"""
    layout = rank_layout(world_size, cp_size)
    for ranks in layout["cp_groups"]:
        if rank in ranks:
            return ranks.index(rank)
    raise RuntimeError(f"rank {rank} 不在任何 CP 组")

# ── 点位边界（core/memlife.py）──
# plan 上的固定 gather 缓冲（`plan._rt["cbuf",…]`）由 runner 的 POINT_RELEASE 通过
# `_CP_STATE_CACHE` 淘汰 + `release_plan_buffers()` 释放，这里只管统计表。
POINT_CACHES = (_AG_STATS,)

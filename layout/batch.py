# coding=utf-8
"""Batch 布局（单序列 / packed varlen）的**单一事实源**。

为什么单独成模块
----------------
packed 支持最初是"哪里需要就在哪里加一个 `if cu_seqlens`"，导致同一套
"token → (请求号, 请求内位置)" 的算术散落在 exec_ops / exec_indexer / runner 三处，
改一个口径要同步改三遍，而且每一处都要自己处理缓存（graph 捕获期还不能做 host→device 拷贝）。

本模块把这类算术收拢成两层：

* **纯布局**（Python 列表/整数）：cu_seqlens、每请求长度、块数；
* **设备张量**（带缓存）：局部位置、请求号、窗口索引、旋转块表、尾部切片索引。

缓存键只由几何参数（cu/T/pos_offset/device）决定，因此 graph 捕获期命中缓存、零 H2D。

约定
----
- `ctx` 仍然是 runner 传下来的普通 dict（`cu_seqlens` / `seq_lens` / `n_seqs` / `pos_offset`）；
  本模块不持有 ctx，也不修改它。
- `local_positions()` 返回的局部位置**已包含 pos_offset**：packed prefill 时它是 0，
  replay 裁尾时是 `S-W`（官方 tail 批的位置登记仍是绝对坐标）。
- 单序列（n_seqs<=1 或无 cu）一律返回 None，调用方走原有单序列路径——零行为变化。
"""
from __future__ import annotations

import os

import torch

__all__ = [
    "cu_of", "n_seqs", "is_packed", "seq_lens", "local_positions", "local_rows",
    "request_ids", "rows_per_seq", "rows_blocks_per_seq",
    "blocks_per_seq", "win_indices", "rot_block_table", "tail_slice", "clear_cache",
]

# 设备张量缓存：键只含几何参数 → 同一个 pass/同一个图里反复命中。
_CACHE: dict = {}
_CACHE_MAX = 64


def _cache(key, build):
    hit = _CACHE.get(key)
    if hit is None:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        hit = _CACHE[key] = build()
    return hit


def clear_cache() -> None:
    """几何变化/解码逐步换 T 时清缓存（runner 的 meta_clear 会一并调用）。"""
    _CACHE.clear()


def cu_of(ctx) -> list | None:
    """packed 的 cu_seqlens（int 列表）；无 cu → None。

    ★ 2026-10-21 统一实现：**n_seqs==1 且带 cu 也按 packed 走**（b=1 与 b>1 同一套代码）。
      以前 `n_seqs<=1 → None` 会把 b=1 分流到单序列实现，导致两边修复不同步。
      没有 cu 的调用（decode / 真权重 prompt 等旧路径）行为不变。
    """
    cu = (ctx or {}).get("cu_seqlens")
    if not cu or int((ctx or {}).get("n_seqs", 1) or 1) < 1:
        return None
    return [int(x) for x in cu]


def n_seqs(ctx) -> int:
    cu = cu_of(ctx)
    return len(cu) - 1 if cu else 1


def is_packed(ctx) -> bool:
    return cu_of(ctx) is not None


def seq_lens(ctx) -> list:
    """每请求长度。优先 `seq_lens`（replay 裁尾后它保持原始 S），否则由 cu 差分。"""
    lens = (ctx or {}).get("seq_lens")
    if lens:
        return [int(x) for x in lens]
    cu = cu_of(ctx)
    return [cu[i + 1] - cu[i] for i in range(len(cu) - 1)] if cu else []


def pos_offsets(ctx):
    """逐序列绝对起始位置（packed 各批不同命中率时用）；未设置 → None。

    ★ 2026-10-21（逐序列几何）：`ctx["pos_offsets"]` = 每请求 tail 的起始逻辑位置
      （plain=0；hit: L_i-R_i）。单序列/老路径仍用标量 `pos_offset`。
    """
    offs = (ctx or {}).get("pos_offsets")
    if not offs:
        return None
    return [int(x) for x in offs]


def local_positions(ctx, T: int, device):
    """→ (每 token 的请求内局部位置 [+pos_offset], 请求号)；单序列 → (None, None)。

    CP 时 `ctx["cp_positions"]` 是每行**全局 token 位置**（head-tail 两块不连续），
    直接覆盖局部位置口径；请求号仍按本地 cu 算。
    """
    cu = cu_of(ctx)
    if cu is None:
        return None, None
    cp_pos = (ctx or {}).get("cp_positions")
    if cp_pos is not None:
        # ★★ 2026-11-02 hostbound 修复（方案 B，比 A 更彻底）：
        #   过去这里每次调用都做 3 件纯浪费的事 ——
        #     ① `torch.tensor(cu, device=npu)` 同步 H2D（要先排空流 → host 阻塞，
        #        trace 里就是 DynamicMxQuant 之后的 aclrtSynchronizeStream）；
        #     ② `torch.arange` → Arange 设备算子；
        #     ③ `torch.searchsorted` → SearchSorted + 输出 ZerosLike。
        #   而 `sid` 在全仓**唯一**消费者是 `request_ids()`，它自身没有调用者
        #   （WORKPLAN 本文档 §3.5 已核对）。所以 CP 分支直接不造 sid：
        #   稳态零 H2D、零 Arange、零 SearchSorted。`request_ids()` 改为显式报错，
        #   杜绝"将来有人用却静默拿到 None"（铁律 0：不静默回退）。
        return cp_pos, None
    off = int((ctx or {}).get("pos_offset", 0) or 0)
    offs = pos_offsets(ctx)
    key = ("pos", tuple(cu), int(T), off, tuple(offs) if offs else None, str(device))
    return _cache(key, lambda: _build_positions(cu, int(T), off, device, offs))


def _build_positions(cu, T, off, device, offs=None):
    cu_t = torch.tensor(cu, dtype=torch.long, device=device)
    pos = torch.arange(T, device=device)
    sid = torch.searchsorted(cu_t[1:-1].contiguous(), pos, right=True)
    loc = pos - cu_t[sid]
    if offs:
        # 逐序列偏移：每 token 取所属请求的 off_i（len(offs)==n_seqs 由调用方保证）
        loc = loc + torch.tensor(offs, dtype=torch.long, device=device)[sid]
    elif off:
        loc = loc + off
    return loc, sid


def request_ids(ctx, T: int, device):
    """每 token 的请求号 [T]；单序列 → None。

    ★ 2026-11-02：CP 分支为消除阻塞 H2D 已不再构造 sid（见 local_positions）。
    本函数在全仓没有调用者；若将来要用，必须在 CP 场景下自行重建 sid——
    这里显式报错而不是静默返回 None。
    """
    _, sid = local_positions(ctx, T, device)
    if sid is None and (ctx or {}).get("cp_positions") is not None:
        raise RuntimeError(
            "request_ids：CP 场景下 sid 已被移除（消除阻塞 H2D 的性能修复）；"
            "需要请求号请自行由 cu 重建，勿依赖本函数的返回值"
        )
    return sid


def blocks_per_seq(ctx, BS: int) -> int:
    """每请求占用的物理块数（按**原始**序列长开池，replay 裁尾不缩池）。"""
    lens = seq_lens(ctx)
    return max(1, -(-max(lens) // int(BS))) if lens else 1


def rows_per_seq(ctx) -> list:
    """本 pass 每请求实际有**多少行**（cu 差分；replay 裁尾后就是 W/请求）。
    与 `seq_lens`（全序列逻辑长）区分：窗口 scratch 按行开池，压缩池按逻辑长开池。"""
    cu = cu_of(ctx)
    return [cu[i + 1] - cu[i] for i in range(len(cu) - 1)] if cu else []


def local_rows(ctx, T: int, device):
    """每 token 的**请求内行号**（不含 pos_offset）。

    ★ 窗口注意力（hit）的 scratch 口径（2026-10-20）：官方 adaptlib/hit.py 把窗口索引
      **重定基到本段局部坐标**（"窗口跑在本步 scratch 上、持久窗口永不读"，见
      exec_ops._win_indices 的长注释）。单序列 hit 一直是这个口径；packed-hit 之前误用了
      `local_positions`（含 pos_offset=15744 → 绝对逻辑位置），厂商 mqsmla 在该访问模式
      下 aicore trap / 输出 NaN（实测 16k b2 MqsmlaKernel trap、2k b2 gate 断言非有限值）。
      改为逐请求行号后：off==0（全量 prefill）时与 `local_positions` **逐位相同**，
      packed-hit 则与已验证的单序列 hit 完全同构（只是多请求拼在一个 kernel 里）。
    """
    cu = cu_of(ctx)
    if cu is None:
        return None, None
    key = ("lrow", tuple(cu), int(T), str(device))
    return _cache(key, lambda: _build_rows(cu, int(T), device))


def _build_rows(cu, T, device):
    cu_t = torch.tensor(cu, dtype=torch.long, device=device)
    pos = torch.arange(T, device=device)
    sid = torch.searchsorted(cu_t[1:-1].contiguous(), pos, right=True)
    return pos - cu_t[sid], sid


def rows_blocks_per_seq(ctx, BS: int) -> int:
    """窗口 scratch 每请求块数（按本 pass **行数**，不是逻辑长 L）。"""
    rows = rows_per_seq(ctx)
    return max(1, -(-max(rows) // int(BS))) if rows else 1


def _win_indices_build(local_pos, BS: int):
    pos = local_pos.to(torch.int64).unsqueeze(1)
    cols = torch.arange(BS, device=local_pos.device).unsqueeze(0)
    start = torch.clamp(pos - (BS - 1), min=0)
    idx = start + cols
    idx = torch.where(idx > pos, torch.full_like(idx, -1), idx)
    length = torch.clamp(pos + 1, max=BS).to(torch.int32)
    return idx.to(torch.int32).unsqueeze(1), length


# ★ 2026-10-26（T-020 P1 / T1）：packed 窗口索引按 local_pos **对象身份**缓存。
#   以前每层现造 → 11 个设备算子/层（实测 mqsmla「前置」8 ms/forward，占胶水 99%）；
#   同一 forward 的 40 层 local_pos 是同一个对象，缓存后每 forward 只算 1 次。
#   键里带 data_ptr 只为分桶，命中还必须 `is` 同一对象（防地址复用读到旧值）。
#   默认 0 = 旧行为逐字节不变；A/B 通过后再按 PERF_RULES §0 翻默认。
# ★ T-042：按 PERF_RULES §0 把默认从 0 翻成 **1**（T-020 eager A/B 实测 −5.1 ms@64k97：
#   `_win_indices_build` 的 arange/sub/clamp/add/gt/where/full_like/to 链每层 2 次）。
#   逐位由 T-042 的 digest A/B 复核（L0/1/2/20 + 末行，64k@0.97 与 256k@0.60）。
#   安全前提：`local_rows` 的返回来自 `_cache`（key = cu/T/device），同一几何下是
#   **同一个只读对象**；本函数只读 local_pos，不原地改。
_WIN_IDX_ON = os.environ.get("DS41M_WIN_IDX_CACHE", "1").strip().lower() in (
    "1", "on", "true", "yes")
_WIN_IDX_CACHE: dict = {}
_WIN_IDX_MAX = 2


def win_indices(local_pos, BS: int):
    """packed 窗口稀疏索引：逻辑位置 = 请求内局部位置（物理块由 block_table 映射）。

    返回 (indices [T,1,BS] int32, length [T,1] int32)，无效列 -1。
    """
    if not _WIN_IDX_ON:
        return _win_indices_build(local_pos, BS)
    key = ("winidx", int(local_pos.data_ptr()), local_pos.shape[0],
           int(BS), str(local_pos.device))
    ent = _WIN_IDX_CACHE.get(key)
    if ent is not None and ent[0] is local_pos:
        return ent[1]
    out = _win_indices_build(local_pos, BS)
    if len(_WIN_IDX_CACHE) >= _WIN_IDX_MAX:
        _WIN_IDX_CACHE.clear()
    _WIN_IDX_CACHE[key] = (local_pos, out)
    return out


def rot_block_table(nblk_per_seq: int, nseq: int, device):
    """每请求一段的旋转块表 [B, P]：[段末块, 段首块, …, 段末-1]（单序列形态的推广）。

    单序列时 `rot_block_table(n, 1)` == [n, 1, 2, …, n-1]，与官方 trace 实测一致。
    """
    P, n = int(nblk_per_seq), int(nseq)
    key = ("rotbt", P, n, str(device))
    return _cache(key, lambda: _build_rot_bt(P, n, device))


def _build_rot_bt(P, n, device):
    base = (torch.arange(n, dtype=torch.int32, device=device) * P).unsqueeze(1)
    last = base + P
    rest = base + torch.arange(1, P, dtype=torch.int32, device=device).unsqueeze(0)
    return torch.cat([last, rest], dim=1)


def tail_slice(ctx, W: int, device):
    """packed 裁尾：每条请求最后 W 行的行号（拼接）与新的 cu。

    replay 用；不等长请求也成立（每条各自取尾部）。
    """
    cu = cu_of(ctx)
    if cu is None:
        return None, None
    W = int(W)
    idx = torch.cat([torch.arange(cu[b + 1] - W, cu[b + 1], device=device)
                     for b in range(len(cu) - 1)])
    new_cu = [W * i for i in range(len(cu))]
    return idx, new_cu

# ── 点位边界（core/memlife.py）：位置/行索引缓存（按 T/cu/off）──
POINT_CACHES = (_CACHE, _WIN_IDX_CACHE)

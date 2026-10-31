# coding=utf-8
"""KV cache 管理（照 DeepInfer `worker/cache_engine.py` 的结构，2026-10-23）。

DeepInfer 的分工（`cache_engine.py:206-560`）：
  · `StateBlock`     —— 先算**布局**：每层 SWA/压缩 state 的字节数、stored/dropped 分区；
  · `KVCacheManager` —— 再**切视图**：一个 flat `kv_tensor`（按块） + `state_tensor`（按请求槽），
                        `process(layer)` 逐层切出 `ica_cache/ca_cache/swa_cache/ica_state/ca_state`；
  · `BlockManager`   —— 请求级块分配（vLLM 口径，DeepInfer 直接用 vLLM 的）。

mini 的对应物（本模块）：
  · `KVGeometry`  —— 从 model_spec 推每层的 cache 种类 + 行宽 + 块行数（等价 StateBlock 的输入）；
  · `KVLayout`    —— 先算布局（每层/每种 cache 的规模与 offset），可打印成一张表；
  · `KVCacheManager` —— 按现有 mini 的**槽位约定**分配/寻址（块 0 恒空、旋转块表、
                       `(1+p//BS)*BS + p%BS` …），几何必须与 inline 实现**逐位一致**
                       （并列 top-k 的取舍顺序依赖块表/池几何，见 NOTES/2026-10-21）；
  · `BlockManager` —— 请求级块分配（分页/多请求用；当前 mini 单序列，先备好）。

★ 为什么先独立成模块：CP/层 20 截尾（YOCO）需要在 rank 间**重分布 state**（hidden/
residual/hc/窗口 scratch + 压缩 state），这些都必须以"哪个层、哪种 cache、哪个槽位"为
单位来搬；散在 exec_ops/exec_indexer 里的 `_cache_buf(st, "_xxx")` 做不到。

几何常量（与 exec_ops/exec_indexer 保持一致）：
  BS=128（算子契约）、KV_ROW_W=544、CMP_ROW_W=320、INDEX 行 64B、WIN_SLOTS=384。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

# ── 与 exec_ops/exec_indexer 对齐的常量（唯一事实源在 model_spec/exec_ops）────
BS = 128
KV_ROW_W = 544
CMP_ROW_W = 320
IDX_ROW_W = 64          # PACKED_D（MXFP4 两值一字节）
IDX_SCALE_W = (2, 2)    # descale 形状
WIN_SLOTS = 384         # 3 × sliding_window
ROWS_PER_BLOCK = 128    # 压缩行块（= block_size）
CAND_BLOCK = 8
CAND_TOPK = 2048


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


@dataclass(frozen=True)
class KVGeometry:
    """每层需要哪些 cache、行宽/块行数是多少（DeepInfer `StateBlock` 的输入信息）。"""

    block_size: int = BS
    win_slots: int = WIN_SLOTS
    kv_row_w: int = KV_ROW_W
    cmp_row_w: int = CMP_ROW_W
    idx_row_w: int = IDX_ROW_W
    rows_per_block: int = ROWS_PER_BLOCK
    cand_block: int = CAND_BLOCK
    cand_topk: int = CAND_TOPK
    window: int = 128
    backtrack: int = 0

    @classmethod
    def from_cfg(cls, cfg: dict) -> "KVGeometry":
        m = cfg.get("model") or {}
        win = int(m.get("sliding_window", 128) or 128)
        return cls(window=win, win_slots=3 * win if win > 0 else WIN_SLOTS)

    # ── 池规模公式（与 exec_ops/exec_indexer 的 inline 版逐字一致）──────────
    def win_pool_rows(self) -> int:
        return int(self.win_slots)

    def full_pool_rows(self, nblk: int) -> int:
        return (1 + int(nblk)) * self.block_size

    def cmp_pool_nblk(self, L_full: int, ratio: int, n_out: int = 0) -> int:
        """压缩池块数：`n_full = max(n, L//ratio + 1)`（carry 行也要能写进去）。"""
        ratio = max(1, int(ratio))
        n_full = max(int(n_out), int(L_full) // ratio + 1)
        return max(1, _ceil_div(n_full, self.block_size))

    def idx_pool_nblk(self, n_rows: int, row0: int, L_full: int, ratio: int) -> int:
        """索引 K 池块数：`n_full=max(n_rows+row0, L//ratio)`，`nblk=ceil((n_full+1)/128)`。"""
        ratio = max(1, int(ratio))
        n_full = max(int(n_rows) + int(row0), int(L_full) // ratio)
        return max(1, _ceil_div(n_full + 1, self.rows_per_block))

    def qsli_pool_nblk(self, n_rows: int, r0max: int, cap: int = 0) -> tuple[int, int]:
        """QSLI 组合缓存：→ (rows_blk, nblk2)。`cap>0` 时是定长环（DS41M_QSLI_CAP）。"""
        rows_blk = max(1, self.rows_per_block // self.cand_block)
        if cap > 0:
            return rows_blk, 1 + max(1, _ceil_div(int(cap), rows_blk)) + 1
        return rows_blk, max(1 + max(1, _ceil_div(int(n_rows) + int(r0max), rows_blk)) + 1,
                             1 + _ceil_div(max(1, int(n_rows)), rows_blk))

    # ── 槽位/块表（与 inline 公式逐位一致）──────────────────────────────────
    def win_slots_for(self, wa: int, off: int, device) -> torch.Tensor:
        return (torch.arange(int(wa), dtype=torch.int32, device=device) + int(off)) % self.win_slots

    def full_slots(self, positions: torch.Tensor) -> torch.Tensor:
        """槽位 = (1 + p//BS)*BS + p%BS（块 0 恒空）。"""
        p = positions.to(torch.int64)
        return ((1 + p // self.block_size) * self.block_size + p % self.block_size).to(torch.int32)

    def cmp_slots(self, comp_rows: torch.Tensor) -> torch.Tensor:
        """压缩行号 → 槽位（同一约定）。"""
        return self.full_slots(comp_rows)

    def ca_block_table(self, nblk: int, device) -> torch.Tensor:
        """压缩/窗口池的块表：恒等 [1..nblk]（块 0 空）。"""
        return torch.arange(1, 1 + int(nblk), dtype=torch.int32, device=device).unsqueeze(0)

    def idx_block_table(self, nblk2: int, device) -> torch.Tensor:
        """索引 K / QSLI 的**旋转块表** [n-1, 1, 2, …, n-2]（写读同一张）。"""
        n = int(nblk2)
        return torch.cat([torch.tensor([n - 1], dtype=torch.int32, device=device),
                          torch.arange(1, n - 1, dtype=torch.int32, device=device)])

    def idx_slots(self, n_rows: int, row0: int, nblk: int, device) -> torch.Tensor:
        """索引 K 写槽位 = 旋转表[p//128]*128 + p%128。"""
        rpb = self.rows_per_block
        q = torch.arange(int(n_rows), dtype=torch.int64, device=device) + int(row0)
        tb = self.idx_block_table(int(nblk) + 1, device)
        return (tb[q // rpb].long() * rpb + q % rpb).to(torch.int32)

    def win_indices(self, pos: torch.Tensor, BS_: int | None = None):
        """窗口稀疏索引 [T,1,BS] + 有效长度 [T,1]（与 `_win_indices_from_pos` 同式）。"""
        bs = int(BS_ or self.block_size)
        p = pos.to(torch.int64).reshape(-1).unsqueeze(1)
        cols = torch.arange(bs, device=pos.device).unsqueeze(0)
        start = torch.clamp(p - (bs - 1), min=0)
        idx = start + cols
        idx = torch.where(idx > p, torch.full_like(idx, -1), idx)
        length = torch.clamp(p + 1, max=bs).to(torch.int32)
        return idx.to(torch.int32).unsqueeze(1), length


@dataclass
class KVLayerPlan:
    """一层的 cache 清单（DeepInfer `KVCache` 的 mini 版：按名字记规模）。"""

    layer: int
    ratio: int
    has_swa: bool = True
    has_cmp: bool = False        # 本层写压缩池（kv source）
    has_idx: bool = False        # 本层写索引 K（index source）
    has_candidate: bool = False  # 本层写 QSLI 组合缓存（candidate source）
    state_ring: int = 0          # 压缩 state ring 行数（0=无）


@dataclass
class KVLayout:
    """整模型布局表（DeepInfer `StateBlock` 的 mini 版）。

    mini 不落盘，这里只算**每层规模**与**状态区字节布局**，供：
      · `print_table()` 自检；
      · 未来接 kvstore/PE→CE 时按 stored/dropped 分区搬运。
    """

    geo: KVGeometry
    layers: list[KVLayerPlan] = field(default_factory=list)

    @classmethod
    def from_cfg(cls, cfg: dict) -> "KVLayout":
        m = cfg.get("model") or {}
        ratios = [int(x) for x in (m.get("compress_ratios") or [])]
        kv_src = {int(x) for x in (m.get("kv_source_layers") or [])}
        idx_src = {int(x) for x in (m.get("index_source_layers") or [])}
        cand = int(m.get("candidate_source_layer", -1))
        plans = []
        for i, r in enumerate(ratios):
            plans.append(KVLayerPlan(
                layer=i, ratio=max(0, r),
                has_swa=True,
                has_cmp=(i in kv_src and r > 0),
                has_idx=(i in idx_src and r > 0),
                has_candidate=(i == cand),
                # 压缩 state 只在**真正跑压缩器**的层（kv source，ratio>1）存在
                state_ring=(2 if (r > 1 and i in kv_src) else 0),
            ))
        return cls(geo=KVGeometry.from_cfg(cfg), layers=plans)

    def print_table(self) -> str:
        rows = ["layer ratio swa cmp idx cand state_ring"]
        for p in self.layers:
            rows.append(f"{p.layer:5d} {p.ratio:5d} {int(p.has_swa):3d} {int(p.has_cmp):3d} "
                        f"{int(p.has_idx):3d} {int(p.has_candidate):4d} {p.state_ring:10d}")
        return "\n".join(rows)


class KVCacheManager:
    """分配 + 寻址。mini 的池都在 `st["_shared"]` 里按名字惰性分配（跨层复用）。

    ★ 本类**不持有** tensor：`get_*` 走调用方给的 hold dict（= `exec_ops._hold(st)`），
      这样 CP/packed/decode 各条路径共用同一套几何，又不用改生命周期。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.layout = KVLayout.from_cfg(cfg)
        self.geo = self.layout.geo

    # ── 缓冲获取（与 `_cache_buf` 同语义：够大就复用，否则新开）──────────
    @staticmethod
    def _get(hold: dict, name: str, rows: int, row_elems: int, device, dtype):
        buf = hold.get(name)
        if buf is None or buf.shape[0] < rows or buf.dtype != dtype:
            buf = torch.zeros(rows, row_elems, dtype=dtype, device=device)
            hold[name] = buf
        return buf

    @staticmethod
    def get_pool(hold, name, rows, row_elems, device, dtype):
        """通用池获取（与 exec_ops._cache_buf 同语义）：够大复用、否则新开。"""
        return KVCacheManager._get(hold, name, rows, row_elems, device, dtype)

    def win_pool(self, hold, device, dtype=torch.float8_e4m3fn):
        return self._get(hold, "_win_pool", self.geo.win_pool_rows(), KV_ROW_W, device, dtype)

    def full_pool(self, hold, nblk, device, suffix="", dtype=torch.float8_e4m3fn):
        return self._get(hold, "_full_pool" + suffix, self.geo.full_pool_rows(nblk),
                         KV_ROW_W, device, dtype)

    def cmp_pool(self, hold, ratio, nblk, device, suffix=""):
        return self._get(hold, f"_cmp_pool_r{ratio}" + suffix,
                         self.geo.full_pool_rows(nblk), CMP_ROW_W, device, torch.uint8)

    def idx_k(self, hold, ratio, nblk, device):
        """索引 K + scale 池。★ 必须是 **4D/5D**（`[块, 行, 1, 64]` / `[块, 行, 1, 2, 2]`）——
        算子按块维度寻址；用二维 alloc 会让 K 缓存整段错（cmp 索引失效，digest 退化成
        NO_CMP 口径，2026-10-23 实测踩过）。"""
        kk, sk = f"_idx_k_r{ratio}", f"_idx_ks_r{ratio}"
        k = hold.get(kk)
        if k is None or k.shape[0] < 1 + nblk:
            k = torch.zeros(1 + nblk, self.geo.rows_per_block, 1, self.geo.idx_row_w,
                            dtype=torch.uint8, device=device)
            hold[kk] = k
        s = hold.get(sk)
        if s is None or s.shape[0] < 1 + nblk:
            s = torch.full((1 + nblk, self.geo.rows_per_block, 1, 2, 2), 127,
                           dtype=torch.uint8, device=device)
            hold[sk] = s
        return k, s

    def qsli(self, hold, ratio, nblk2, rows_blk, device):
        """组合缓存：**nblk2 块（已含块 0 空块）**，与 exec_indexer 的 inline 逐位同形。"""
        kk = f"_idx_qsli_r{ratio}"
        kc = hold.get(kk)
        if kc is None or kc.shape[0] < nblk2:
            cbs = self.geo.cand_block
            kc = torch.zeros(nblk2, rows_blk, 1, cbs * (self.geo.idx_row_w + 4),
                             dtype=torch.uint8, device=device)
            hold[kk] = kc
            hold[f"_idx_qslis_r{ratio}"] = (rows_blk, nblk2)
        return kc

    def cmp_state(self, hold, layer: int, ring: int, dim: int, device, nblocks: int = 2):
        """state ring；CP 官方流程每段一块（块 0 恒空）→ nblocks = 1 + 段数。

        ★ 需要更多块时**增长**（cat 零块），不重建：已有块里存着 stage A 写的
          前缀 carry，重建会把它们清零（实测会让 hit 段 head 的 carry 归零）。
        """
        key = f"_cmp_state_L{layer}"
        stc = hold.get(key)
        if stc is not None and int(stc.shape[1]) != int(ring):
            stc = None
        if stc is None:
            stc = torch.zeros(int(nblocks), int(ring), 2, int(dim), dtype=torch.float32,
                              device=device)
            hold[key] = stc
        elif int(stc.shape[0]) < int(nblocks):
            grow = torch.zeros(int(nblocks) - int(stc.shape[0]), int(ring), 2, int(dim),
                               dtype=torch.float32, device=stc.device)
            stc = torch.cat([stc, grow], 0)
            hold[key] = stc
        return stc


class BlockManager:
    """请求级块分配（vLLM `BlockManager` 口径；DeepInfer 直接复用 vLLM 的）。

    mini 目前单序列、池按 L 开；但多请求/分页/CP DP 组都要它：
      · `allocate(n_tokens)` → handle（块号列表，[0] 恒不分配=空块约定由调用方 +1）；
      · `free(handle)`；`block_table(handle)`。
    """

    def __init__(self, num_blocks: int, block_size: int = BS, first_block: int = 1):
        self.block_size = int(block_size)
        self.first_block = int(first_block)   # 池的块 0 恒空 → 从 1 起分配
        self.num_blocks = int(num_blocks)
        self._free = list(range(self.first_block, self.first_block + self.num_blocks))
        self._used: dict[int, list[int]] = {}

    def num_free_blocks(self) -> int:
        return len(self._free)

    def can_allocate(self, n_tokens: int) -> bool:
        return len(self._free) >= _ceil_div(int(n_tokens), self.block_size)

    def allocate(self, n_tokens: int) -> int:
        need = _ceil_div(int(n_tokens), self.block_size)
        if need > len(self._free):
            raise MemoryError(f"块不足：需要 {need}，空闲 {len(self._free)}")
        blocks = [self._free.pop(0) for _ in range(need)]
        h = len(self._used)
        self._used[h] = blocks
        return h

    def free(self, handle: int) -> None:
        self._free.extend(self._used.pop(handle))

    def block_table(self, handle: int) -> list[int]:
        return list(self._used[handle])


_MGR = None


def get_manager(cfg: dict) -> "KVCacheManager":
    """进程级单例（cfg 在整个 run 内不变；几何只算一次）。"""
    global _MGR
    if _MGR is None:
        _MGR = KVCacheManager(cfg)
    return _MGR


__all__ = [
    "BS", "KV_ROW_W", "CMP_ROW_W", "IDX_ROW_W", "WIN_SLOTS", "ROWS_PER_BLOCK",
    "KVGeometry", "KVLayerPlan", "KVLayout", "KVCacheManager", "BlockManager",
    "get_manager",
]

"""CP 运行时状态：把 cp_layout 的整数计划变成设备张量 + 本地/全量 ctx。

默认 `DS41M_CP_SIZE=1` 时 `from_env()` 返回 None，原有路径完全不变；
只有显式 `DS41M_CP_SIZE>1` 才建 CP 组、切本地行。

这一层只做：
  · split_local：全量输入 → 本 rank 的 head-tail/mirror 行（pad 行清零/填 pad token）
  · merge_global：本 rank 行 → all-gather → 全局顺序
  · local_ctx / full_ctx：给 exec_ops 的 ctx 加上 CP 坐标

不在这里做模型算子；模型接入见后续步骤。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

from cp_layout import CpPlan, build_cp_plan

# ★ 2026-10-28（P0-4）：CpState 绝对位置缓存开关（`=0` 回退到每次现算，A/B 归因用）。
_POS_CACHE_ON = os.environ.get("DS41M_CP_POS_CACHE", "1").strip().lower() in (
    "1", "on", "true", "yes")
from cp_runtime import (all_gather_global, all_gather_global_async,
                        cp_rank_in_group, init_cp_group)


@dataclass
class CpState:
    plan: CpPlan
    group: object | None
    cp_size: int
    cp_rank: int
    device: torch.device
    local_index: torch.Tensor       # [local_rows] int64，pad=-1
    reverse_index: torch.Tensor     # [total_q] int64
    local_positions: torch.Tensor   # [local_rows] int64，pad=0
    valid: torch.Tensor             # [local_rows] bool
    last_local_rows: tuple          # 每序列末 token 在**本地行**里的下标（诊断 digest 用）
    has_pads: bool = False          # 本地是否有 pad 行（非整除 L；host 侧常量，免每层同步）
    # ★ 层 20 截尾（compact 段）：本地行是"保留窗口"里的行，压缩/rope 要用**绝对位置**
    #   （= 本地 compact 位置 + pos_offset）；窗口 scratch 仍用本地 0..W-1。
    pos_offset: int = 0
    #   序列总长（compact 段 = 原 L）：ref-M/rope 表宽/seqused_ori_kv 沿用原序列口径。
    seq_len: int = 0
    # ★ 2026-10-28（P0-4）：`local_positions + pos_offset` 在本状态生命周期内是常量。
    #   以前每次访问都新发一个 int64 Add（runner/mqsmla/idx_prolog 各一次），且
    #   `cmp_cl_cp` / `win_cp` 的 `_meta` key 带 data_ptr → 每层变 ⇒ 压缩有效长度/窗口
    #   索引每层重建（div/clamp/cast/where 链）。缓存后同一指针跨层复用。
    _abs_cache: tuple | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_plan(cls, plan: CpPlan, *, group=None, device="cpu",
                  pos_offset: int = 0, seq_len: int = 0) -> "CpState":
        dev = torch.device(device)
        li = torch.tensor(plan.local_index, dtype=torch.long, device=dev)
        ri = torch.tensor(plan.reverse_index, dtype=torch.long, device=dev)
        # 每序列末 token 的本地行号：非整除 L 时最后一个块带 pad，`x[-1]` 是 pad 行，
        # 不能拿来做对拍摘要（会被误判成数值不一致）。末 token 的全局号要加 cu_q；
        # head-tail 下它恒在块 2*CP-1（= rank0 的尾块）⇒ rank0 总能找到，其余 rank 记 -1。
        last = []
        cu_q = 0
        for sq in plan.seq_lens:
            gi = cu_q + int(sq) - 1
            last.append(plan.local_index.index(gi) if gi in plan.local_index else -1)
            cu_q += int(sq)
        return cls(
            plan=plan,
            group=group,
            cp_size=plan.cp_size,
            cp_rank=plan.cp_rank,
            device=dev,
            local_index=li,
            reverse_index=ri,
            local_positions=li.clamp_min(0),
            valid=li >= 0,
            last_local_rows=tuple(last),
            has_pads=any(int(v) < 0 for v in plan.local_index),
            pos_offset=int(pos_offset),
            seq_len=int(seq_len or plan.total_q),
        )

    # ── compact 段的两种位置口径 ────────────────────────────────────────
    @property
    def abs_positions(self) -> torch.Tensor:
        """绝对位置（rope / 压缩长度 / 压缩池行号用）。全量段 pos_offset=0 → 恒等。"""
        if self.pos_offset == 0:
            return self.local_positions
        if not _POS_CACHE_ON:
            return self.local_positions + int(self.pos_offset)
        c = self._abs_cache
        if c is None or c[0] != int(self.pos_offset):
            c = (int(self.pos_offset), self.local_positions + int(self.pos_offset))
            self._abs_cache = c
        return c[1]

    def with_compact(self, keep_lens, *, pos_offset: int, seq_len: int) -> "CpState":
        """同一 rank/组上，为"截尾后的 compact 序列"建一个新的 CpState。"""
        plan = build_cp_plan([int(x) for x in keep_lens], self.cp_size, self.cp_rank)
        return CpState.from_plan(plan, group=self.group, device=self.device,
                                 pos_offset=pos_offset, seq_len=seq_len)

    @classmethod
    def from_env(cls, seq_lens, world_size: int, device, *, backend: str = "hccl",
                 cp_size: int | None = None) -> "CpState | None":
        """从 `DS41M_CP_SIZE` 建 CP 状态；CP=1 直接返回 None（零行为变化）。

        所有 rank 必须按相同顺序调用（内部会 `dist.new_group`）。
        """
        cp_size = int(cp_size if cp_size is not None else os.environ.get("DS41M_CP_SIZE", "1") or 1)
        if cp_size <= 1:
            return None
        import torch.distributed as dist
        rank = int(os.environ.get("RANK", dist.get_rank() if dist.is_initialized() else 0))
        if world_size % cp_size:
            raise ValueError(f"world_size={world_size} 不能被 DS41M_CP_SIZE={cp_size} 整除")
        group = init_cp_group(world_size, cp_size, backend=backend, rank=rank)
        cr = cp_rank_in_group(world_size, cp_size, rank)
        plan = build_cp_plan(seq_lens, cp_size, cr)
        return cls.from_plan(plan, group=group, device=device)

    @property
    def local_rows(self) -> int:
        return int(self.local_index.numel())

    def split_local(self, x: torch.Tensor, pad: int | float = 0) -> torch.Tensor:
        """全量 [total_q, ...] → 本地 [local_rows, ...]；pad 行填 pad。"""
        if int(x.shape[0]) != self.plan.total_q:
            raise ValueError(f"x 行数 {x.shape[0]} != total_q {self.plan.total_q}")
        out = x.index_select(0, self.local_index.clamp_min(0))
        # ★ 2026-10-25：`bool(self.valid.all())` 是**设备归约 + host 同步**，每 forward
        #   在 split_local 卡一次（trace：ReduceAll[T,1] + aten::is_nonzero 4.5–10ms，
        #   恰好落在 forward 边界空泡里）。`has_pads` 是 from_plan 里的 host 常量。
        if self.has_pads:
            shape = (self.local_rows,) + (1,) * (out.dim() - 1)
            out = torch.where(self.valid.view(shape), out, torch.full_like(out, pad))
        return out

    def merge_global(self, x_local: torch.Tensor, chunk: int | None = None,
                     buf_key: str | None = None) -> torch.Tensor:
        """本地 [local_rows, ...] → all-gather → 全局 [total_q, ...]。

        `chunk`：只给**侧流预取**用——强制一次 all-gather 完成（见 runner 的 prefetch
        注释：分块时侧流会同时有多个 collective 与主流交错，HCCL 上会互相等，实测
        CP2 64K 出现 14ms 停顿；预取改成单块后消失）。普通调用不传，行为不变。
        `buf_key`：DeepInfer userbuffer 式固定缓冲的键；只给**结果寿命在单算子内**
        的调用用（kv_compress/kv_epilog）。同一 key 复用同一块显存，逐位不变。
        """
        if int(x_local.shape[0]) != self.local_rows:
            raise ValueError(f"x_local 行数 {x_local.shape[0]} != local_rows {self.local_rows}")
        if self.cp_size == 1:
            return x_local.index_select(0, self.reverse_index)
        if self.group is None:
            raise ValueError("cp_size>1 但没有 CP 组；测试请用 cp_layout.simulate_all_gather")
        if chunk is None and buf_key is None:   # 兼容旧签名（A/B 回退用）
            return all_gather_global(x_local, self.plan, self.group)
        return all_gather_global(x_local, self.plan, self.group,
                                 chunk=chunk, buf_key=buf_key)

    def merge_global_async(self, x_local: torch.Tensor, buf_key: str | None = None):
        """`merge_global` 的**侧流发起版**：立刻返回 `(out, event)`，消费方 wait_event。

        DeepInfer `cp.all_gather()` 的 wait-句柄同款。单块 + 固定缓冲；数值逐位不变。
        """
        if int(x_local.shape[0]) != self.local_rows:
            raise ValueError(f"x_local 行数 {x_local.shape[0]} != local_rows {self.local_rows}")
        if self.cp_size == 1:
            return x_local.index_select(0, self.reverse_index), None
        if self.group is None:
            raise ValueError("cp_size>1 但没有 CP 组；测试请用 cp_layout.simulate_all_gather")
        return all_gather_global_async(x_local, self.plan, self.group, buf_key=buf_key)

    def local_ctx(self, ctx: dict) -> dict:
        """原 ctx → 本地 CP 视图（q/attention/MoE 用本地行）。"""
        c = dict(ctx)
        c.update(
            T=self.local_rows,
            cu_seqlens=list(self.plan.local_cu),
            seq_lens=list(self.plan.local_lens),
            row_lens=list(self.plan.local_lens),
            n_seqs=len(self.plan.seq_lens),
            pos_offsets=None,
            pos_offset=0,
            cp_size=self.cp_size,
            cp_rank=self.cp_rank,
            cp_positions=self.local_positions,
            cp_local_index=self.local_index,
            cp_valid=self.valid,
            cp_plan=self.plan,
        )
        return c

    def full_ctx(self, ctx: dict) -> dict:
        """原 ctx → 全局视图（SWA/压缩池写 full 用）。"""
        lens = list(self.plan.seq_lens)
        cu = [0]
        for n in lens:
            cu.append(cu[-1] + n)
        c = dict(ctx)
        c.update(
            T=int(self.plan.total_q),
            cu_seqlens=cu,
            seq_lens=lens,
            row_lens=lens,
            n_seqs=len(lens),
            pos_offsets=[0] * len(lens),
            pos_offset=0,
            L_full=int(self.plan.total_q),
            cp_size=self.cp_size,
            cp_rank=self.cp_rank,
            cp_positions=None,
            cp_local_index=None,
            cp_valid=None,
            cp_plan=self.plan,
        )
        return c

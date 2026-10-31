"""DeepInfer 风格 CP 的纯布局层（2026-10-23）。

只做**纯 Python** 的 rank/head-tail/mirror 布局，不碰 NPU、不碰 torch.distributed，
因此可以在 CPU 上单独跑回归；runner/exec_ops 再把它转成设备张量。

DeepInfer 口径（`deepinfer/model_executor/parallel_utils/parallel_state.py:92-99`）：
  · CP 组连续：`[r*CP, (r+1)*CP)`；
  · DP 组按 CP 跨步：同一 cp 坐标的 replica 组成一组；
  · EP 组默认是全 world（EP8 时）。
  · `dp_size = ep_size // cp_size`（config.py:162）。

每个请求的序列切分（`prepare_cp_mirror.py:135-180`）：
  ps = ceil(L / (2*CP))
  rank r 拿两块：块 r 和块 2*CP-1-r，每块 ps 行
  有效行 < L 才保留，否则 pad=-1
  · local_index[row]      = 该行的全局 token 号（pad=-1）
  · reverse_index[token]  = 该全局 token 在 rank-major all-gather 缓冲中的行号
  · local_cu               = 本 rank 每个请求的行数（含 pad，2*ps）

CP=1 时故意走 identity 快路径，保证当前单卡行为逐位不变（不引入 padding）。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CpPlan:
    cp_size: int
    cp_rank: int
    seq_lens: tuple[int, ...]
    local_lens: tuple[int, ...]      # 每请求本地行数（含 pad）
    local_cu: tuple[int, ...]        # len(seq_lens)+1
    local_index: tuple[int, ...]     # 本地行 -> 全局 token，pad=-1
    reverse_index: tuple[int, ...]   # 全局 token -> rank-major 行
    total_q: int
    total_per_rank: int
    # reverse_index 的逆：rank-major 缓冲行 → 全局 token 号（分块 all-gather 用）
    inverse_reverse: tuple[int, ...] = ()
    # 运行时缓存（**不参与相等性/哈希**）：device 侧索引张量等，见
    # `cp_runtime._plan_tensor`。plan 本身不可变，缓存随 plan 生命周期走。
    _rt: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def local_rows(self) -> int:
        return len(self.local_index)

    def local_valid_mask(self) -> tuple[bool, ...]:
        return tuple(i >= 0 for i in self.local_index)

    def local_blocks(self) -> tuple[tuple[int, int, int], ...]:
        """本 rank 每个请求的 head/tail 块 → `(local_start, global_start, n_valid)`。

        `local_start` 是本 rank 本地张量里的行号（**含前一块的 pad**）；`n_valid`
        是真正属于序列的行数（块尾其余为 pad=-1）。非整除 L 时最后一个块可能
        不满；整块越界的块 `n_valid=0`（本地仍有 ps 行 pad）。
        压缩/indexer 的 CP 实现按块跑（每块本身就是"前缀的尾部"→ 语义与 prefill
        的右对齐一致）。
        """
        out = []
        cu_local = 0
        cu_q = 0
        for sq in self.seq_lens:
            ps = _ceil_div(sq, 2 * self.cp_size) if self.cp_size > 1 else sq
            for half, block in enumerate((self.cp_rank, 2 * self.cp_size - 1 - self.cp_rank)
                                         if self.cp_size > 1 else (self.cp_rank,)):
                gstart = int(cu_q) + block * ps
                nv = max(0, min(ps, int(sq) - block * ps))
                out.append((cu_local + half * ps, gstart, nv))
            cu_local += 2 * ps if self.cp_size > 1 else sq
            cu_q += sq
        return tuple(out)


def rank_layout(world_size: int, cp_size: int) -> dict:
    """返回 DeepInfer 的 CP/DP/EP 分组。

    world_size 必须能被 cp_size 整除；dp_size = world_size // cp_size。
    返回的列表都是 rank 号列表，按组排列。
    """
    world_size = int(world_size)
    cp_size = int(cp_size)
    if cp_size < 1 or world_size % cp_size:
        raise ValueError(f"world_size={world_size} 不能被 cp_size={cp_size} 整除")
    dp_size = world_size // cp_size
    cp_groups = [list(range(r * cp_size, (r + 1) * cp_size)) for r in range(dp_size)]
    # DeepInfer: (replica dp cp) -> (replica cp) dp
    dp_groups = [[r * cp_size + c for r in range(dp_size)] for c in range(cp_size)]
    return {
        "world_size": world_size,
        "cp_size": cp_size,
        "dp_size": dp_size,
        "cp_groups": cp_groups,
        "dp_groups": dp_groups,
        "ep_group": list(range(world_size)),
    }


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def build_cp_plan(seq_lens, cp_size: int, cp_rank: int) -> CpPlan:
    """构造一个 rank 的 head-tail/mirror 布局。

    seq_lens: 每个请求的逻辑长度（列表）。
    cp_size/cp_rank: 上下文并行大小/本 rank。
    """
    lens = tuple(int(x) for x in seq_lens)
    if not lens or any(x <= 0 for x in lens):
        raise ValueError(f"seq_lens 必须非空且为正：{lens}")
    cp_size = int(cp_size)
    cp_rank = int(cp_rank)
    if cp_size < 1 or not (0 <= cp_rank < cp_size):
        raise ValueError(f"非法 cp_size={cp_size}, cp_rank={cp_rank}")

    total_q = sum(lens)
    if cp_size == 1:
        # ★ 单卡 identity：不引入 pad，保证 cp1 与旧行为逐位一致。
        local_index = tuple(range(total_q))
        local_lens = lens
        local_cu = [0]
        for n in local_lens:
            local_cu.append(local_cu[-1] + n)
        return CpPlan(
            cp_size=1,
            cp_rank=0,
            seq_lens=lens,
            local_lens=lens,
            local_cu=tuple(local_cu),
            local_index=local_index,
            reverse_index=tuple(range(total_q)),
            total_q=total_q,
            total_per_rank=total_q,
            inverse_reverse=tuple(range(total_q)),
        )

    local_index: list[int] = []
    local_lens: list[int] = []
    reverse = [0] * total_q
    cu_q = 0
    cu_elem = 0
    total_per_rank = sum(2 * _ceil_div(sq, 2 * cp_size) for sq in lens)
    for sq in lens:
        ps = _ceil_div(sq, 2 * cp_size)
        # reverse_index 是**全局**映射：每个 rank 都要知道每个 token 落在哪个
        # rank 的 rank-major 行；local_index 只填本 rank。
        for rank in range(cp_size):
            for half, block in enumerate((rank, 2 * cp_size - 1 - rank)):
                for j in range(ps):
                    local_idx = block * ps + j
                    if local_idx >= sq:
                        if rank == cp_rank:
                            local_index.append(-1)
                        continue
                    gi = cu_q + local_idx
                    reverse[gi] = rank * total_per_rank + cu_elem + half * ps + j
                    if rank == cp_rank:
                        local_index.append(gi)
        local_lens.append(2 * ps)
        cu_q += sq
        cu_elem += 2 * ps

    local_cu = [0]
    for n in local_lens:
        local_cu.append(local_cu[-1] + n)
    # ★ pad 行（块尾不满）没有任何全局 token 指向 → 必须是 -1，scatter 路径据此退回。
    #   旧实现初始化为 0：`_inv_ok` 检查形同虚设，L 不整除 2*cp 且本地行 > chunk 时
    #   会把 pad 行写到全局 token 0（本项目的 CP 尺寸都整除，未触发；2026-10-25 修）。
    inv = [-1] * (cp_size * total_per_rank)
    for g, b in enumerate(reverse):
        inv[b] = g
    return CpPlan(
        cp_size=cp_size,
        cp_rank=cp_rank,
        seq_lens=lens,
        local_lens=tuple(local_lens),
        local_cu=tuple(local_cu),
        local_index=tuple(local_index),
        reverse_index=tuple(reverse),
        total_q=total_q,
        total_per_rank=total_per_rank,
        inverse_reverse=tuple(inv),
    )


def simulate_all_gather(shards: list[list], plan: CpPlan) -> list:
    """纯 Python 模拟 `all_gather` + `mirror_cp_reverse_index`：

    shards[r] 是 rank r 的本地行（长度必须 == plan.total_per_rank）。
    返回按**全局 token 顺序**排列的 full 序列。
    """
    if len(shards) != plan.cp_size:
        raise ValueError(f"shards 数 {len(shards)} != cp_size {plan.cp_size}")
    flat: list = []
    for r, shard in enumerate(shards):
        if len(shard) != plan.total_per_rank:
            raise ValueError(
                f"rank {r} shard 长度 {len(shard)} != total_per_rank {plan.total_per_rank}"
            )
        flat.extend(shard)
    return [flat[plan.reverse_index[i]] for i in range(plan.total_q)]


def boundary_indices(seq_lens, keep_lens, cp_size: int, cp_rank: int):
    """层 20（YOCO）边界：**截尾后跨 CP 重分布**的索引（DeepInfer 同口径）。

    移植自 `prepare_cp_mirror.py:160-186 yoco_cp_boundary_fused_ref`：
      · 每个请求保留最后 `keep` 个 token（`clip = sq - keep`）；
      · `contribs[r]`：rank r 的**本地 mirror 行**里"截尾后仍存在"的行号（按全局序）；
      · `gathered_reverse[compact_idx]`：compact 全局序 → 拼接缓冲行号
        （`rank * pad + ordinal`）；
      · `pad = max(每 rank 贡献数)`（各 rank 等长，便于 all_gather）。

    返回 `(contribs, gathered_reverse, pad)`；纯 Python，CPU 可回归。
    """
    lens = [int(x) for x in seq_lens]
    keeps = [int(x) for x in keep_lens]
    if len(lens) != len(keeps):
        raise ValueError(f"seq_lens/keep_lens 长度不一致：{lens} vs {keeps}")
    cp_size = int(cp_size)
    cp_rank = int(cp_rank)
    contribs: list[list[int]] = [[] for _ in range(cp_size)]
    entries: list[tuple[int, int, int]] = []      # (compact_idx, rank, ordinal)
    cu_elem = 0
    cu_c = 0
    for sq, keep in zip(lens, keeps):
        keep = max(1, min(keep, sq))
        clip = sq - keep
        if cp_size == 1:
            # identity（mini 的 cp1 不做 mirror 切分）：保留行就是本地行号的连续段
            for j in range(keep):
                entries.append((cu_c + j, 0, len(contribs[0])))
                contribs[0].append(cu_elem + clip + j)
            cu_elem += sq
            cu_c += keep
            continue
        ps = _ceil_div(sq, 2 * cp_size)
        for rank in range(cp_size):
            for half, block in enumerate((rank, 2 * cp_size - 1 - rank)):
                for j in range(ps):
                    local_idx = block * ps + j
                    if clip <= local_idx < sq:
                        entries.append((cu_c + local_idx - clip, rank, len(contribs[rank])))
                        contribs[rank].append(cu_elem + half * ps + j)
        cu_elem += 2 * ps
        cu_c += keep
    pad = max((len(c) for c in contribs), default=1) or 1
    gathered = [-1] * cu_c
    for compact_idx, rank, ordinal in entries:
        gathered[compact_idx] = rank * pad + ordinal
    return contribs, gathered, pad


def compact_boundary_index(compact_plan: CpPlan, gathered_reverse) -> tuple[int, ...]:
    """compact-mirror 本地行 → 拼接缓冲行号（pad 行 = -1）。"""
    out = []
    for gi in compact_plan.local_index:
        out.append(int(gathered_reverse[gi]) if gi >= 0 else -1)
    return tuple(out)

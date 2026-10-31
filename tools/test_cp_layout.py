#!/usr/bin/env python3
"""cp_layout 纯 CPU 回归：head-tail/mirror/all-gather 口径。"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cp_layout import (  # noqa: E402
    boundary_indices, build_cp_plan, compact_boundary_index, rank_layout,
    simulate_all_gather,
)


def _check_layout():
    assert rank_layout(8, 1)["cp_groups"] == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert rank_layout(8, 1)["dp_groups"] == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert rank_layout(8, 2)["cp_groups"] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert rank_layout(8, 2)["dp_groups"] == [[0, 2, 4, 6], [1, 3, 5, 7]]
    assert rank_layout(8, 4)["cp_groups"] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert rank_layout(8, 4)["dp_groups"] == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert rank_layout(8, 8)["cp_groups"] == [list(range(8))]
    assert rank_layout(8, 8)["dp_groups"] == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert rank_layout(8, 8)["ep_group"] == list(range(8))


def _check_case(lens, cp):
    total_q = sum(lens)
    plans = [build_cp_plan(lens, cp, r) for r in range(cp)]
    rows = {p.total_per_rank for p in plans}
    assert len(rows) == 1, (lens, cp, rows)
    total_per_rank = rows.pop()
    assert all(p.total_q == total_q for p in plans)

    seen = []
    for r, p in enumerate(plans):
        assert p.local_rows == sum(p.local_lens) == p.local_cu[-1]
        assert len(p.reverse_index) == total_q
        valid = [v for v in p.local_index if v >= 0]
        seen.extend(valid)
        # 本地有效行 = 全局 token 号
        assert valid == sorted(set(valid)), (lens, cp, r, valid[:20])
        for gi in valid:
            assert 0 <= gi < total_q
            # reverse 指向本 rank 的 rank-major 段
            assert r * total_per_rank <= p.reverse_index[gi] < (r + 1) * total_per_rank, (
                lens, cp, r, gi, p.reverse_index[gi], total_per_rank
            )
        # reverse 是全局映射（跨所有 rank），不能按本 rank 段检查；只检查唯一/范围
        vals = [p.reverse_index[i] for i in range(total_q)]
        assert len(set(vals)) == total_q
        assert min(vals) >= 0
        assert max(vals) < cp * total_per_rank
        # 本 rank 自己的行号段（用于写 full buffer）
        own = [p.reverse_index[gi] for gi in valid]
        assert all(r * total_per_rank <= v < (r + 1) * total_per_rank for v in own)

    # 每个全局 token 恰好出现一次
    assert sorted(seen) == list(range(total_q)), (lens, cp, seen[:50], len(seen))

    # all-gather：每 rank 的 shard 是 local_index（pad=-1）；拼起来按 reverse 取，
    # 非 pad 行必须得到全局 token 号
    shards = [list(p.local_index) for p in plans]
    full = simulate_all_gather(shards, plans[0])
    assert full == list(range(total_q)), (lens, cp, full[:20])

    # local_blocks 自证：按块铺回去必须与 local_index 逐行一致（pad 行 = 块外）
    for r, p in enumerate(plans):
        rebuilt = {}
        for ls, gs, nv in p.local_blocks():
            for j in range(nv):
                rebuilt[ls + j] = gs + j
        for i, v in enumerate(p.local_index):
            if v < 0:
                assert i not in rebuilt, (lens, cp, r, i)
            else:
                assert rebuilt.get(i) == v, (lens, cp, r, i, v, rebuilt.get(i))
        assert len(rebuilt) == sum(1 for v in p.local_index if v >= 0)
        # 块不重叠、起点递增
        starts = [b[0] for b in p.local_blocks()]
        assert starts == sorted(set(starts)), (lens, cp, r, starts)

    # 头部/尾部块的检查：rank0 拿块 0 与最后一块；rank cp-1 拿中间两块
    if cp > 1 and lens == [17]:
        p0_valid = [v for v in plans[0].local_index if v >= 0]
        p_last_valid = [v for v in plans[cp - 1].local_index if v >= 0]
        assert p0_valid[0] == 0
        if cp == 2:
            assert p0_valid[-1] == 16
        assert p_last_valid[0] > 0


def _ref_boundary(seq_lens, keep_lens, cp):
    """DeepInfer `yoco_cp_boundary_fused_ref` 逐字移植（cp>1 的 oracle）。"""
    contribs = [[] for _ in range(cp)]
    entries = []
    cu_elem = 0
    cu_c = 0
    for sq, keep in zip(seq_lens, keep_lens):
        clip = sq - keep
        ps = (sq + 2 * cp - 1) // (2 * cp)
        for rank in range(cp):
            for half, block in enumerate((rank, 2 * cp - 1 - rank)):
                for j in range(ps):
                    li = ps * block + j
                    if clip <= li < sq:
                        entries.append((cu_c + li - clip, rank, len(contribs[rank])))
                        contribs[rank].append(cu_elem + half * ps + j)
        cu_elem += 2 * ps
        cu_c += keep
    pad = max(len(c) for c in contribs)
    gathered = [-1] * cu_c
    for ci, rank, o in entries:
        gathered[ci] = rank * pad + o
    return contribs, gathered, pad


def _check_boundary():
    cases = [
        ([1000], [512]), ([1000], [140]), ([17], [8]), ([8192], [512]),
        ([3, 5, 2], [2, 3, 1]), ([127, 129], [64, 100]),
    ]
    for cp in (2, 4, 8):
        for lens, keeps in cases:
            c, g, p = boundary_indices(lens, keeps, cp, 0)
            rc, rg, rp = _ref_boundary(lens, keeps, cp)
            assert (c, g, p) == (rc, rg, rp), (lens, keeps, cp)
            # 每个 rank 的 contrib 必须落在该 rank 的本地行号范围内
            for r in range(cp):
                plan = build_cp_plan(lens, cp, r)
                for v in c[r]:
                    assert 0 <= v < plan.local_rows, (lens, cp, r, v)
            # compact 计划的每个本地行都能映射到拼接缓冲（pad=-1）
            cplan = build_cp_plan(keeps, cp, 0)
            bidx = compact_boundary_index(cplan, g)
            for row, gi in enumerate(cplan.local_index):
                assert bidx[row] == (g[gi] if gi >= 0 else -1)
            seen = [x for x in g if x >= 0]
            assert len(seen) == len(set(seen)) == sum(keeps)


def main():
    _check_layout()
    _check_boundary()
    lens_cases = [
        [1], [2], [3], [5], [8], [17], [128], [1000], [8191],
        [3, 5, 2], [1, 1, 1, 1], [127, 129], [131072],
    ]
    for cp in (1, 2, 4, 8):
        for lens in lens_cases:
            _check_case(lens, cp)
    # cp1 必须与旧行为逐位一致：无 pad、identity
    for lens in lens_cases:
        p = build_cp_plan(lens, 1, 0)
        assert p.local_index == tuple(range(sum(lens)))
        assert p.local_lens == tuple(lens)
        assert p.reverse_index == tuple(range(sum(lens)))
    print(f"cp_layout: PASS ({len(lens_cases)} lens cases × cp 1/2/4/8)")


if __name__ == "__main__":
    main()

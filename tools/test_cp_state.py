#!/usr/bin/env python3
"""cp_state 纯 CPU 回归：split_local / local_ctx / full_ctx。"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cp_layout import build_cp_plan, simulate_all_gather  # noqa: E402
from cp_state import CpState  # noqa: E402

CASES = [[1], [2], [3], [5], [17], [128], [3, 5, 2], [127, 129]]


def _check(lens, cp):
    plans = [build_cp_plan(lens, cp, r) for r in range(cp)]
    states = [CpState.from_plan(p, device="cpu") for p in plans]
    total_q = sum(lens)
    full = torch.arange(total_q, dtype=torch.float32)
    for r, st in enumerate(states):
        loc = st.split_local(full, pad=-1.0)
        assert loc.numel() == st.local_rows
        # 有效行 = 全局 token 号
        vals = loc[st.valid].tolist()
        assert vals == [float(v) for v in plans[r].local_index if v >= 0]
        # pad 行 = -1
        assert torch.all(loc[~st.valid] == -1.0)
        lc = st.local_ctx({"pos_offset": 0, "n_seqs": 1})
        assert lc["T"] == st.local_rows
        assert lc["cu_seqlens"] == list(plans[r].local_cu)
        assert lc["cp_positions"].tolist() == [
            v if v >= 0 else 0 for v in plans[r].local_index
        ]
        fc = st.full_ctx({"pos_offset": 0, "n_seqs": 1})
        assert fc["T"] == total_q
        assert fc["seq_lens"] == list(lens)
        assert fc["cu_seqlens"][-1] == total_q
    # 模拟 all-gather：每 rank 的 local shard 拼起来，按 reverse 还原全局
    shards = [s.split_local(full, pad=-1.0).tolist() for s in states]
    merged = simulate_all_gather(shards, plans[0])
    assert merged == [float(i) for i in range(total_q)], (lens, cp, merged[:20])


def main():
    for cp in (1, 2, 4, 8):
        for lens in CASES:
            _check(lens, cp)
    print(f"cp_state: PASS ({len(CASES)} lens cases × cp 1/2/4/8)")


if __name__ == "__main__":
    main()

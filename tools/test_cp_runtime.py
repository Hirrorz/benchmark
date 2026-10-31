#!/usr/bin/env python3
"""cp_runtime 的 8 进程 gloo 回归（不需要 NPU）。"""
import os
import socket
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cp_layout import build_cp_plan  # noqa: E402
from cp_runtime import all_gather_global, cp_rank_in_group, init_cp_group  # noqa: E402

CASES = [[1], [2], [3], [5], [17], [128], [3, 5, 2], [127, 129]]


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _worker(rank, world, port, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        for cp in (1, 2, 4, 8):
            group = init_cp_group(world, cp, backend="gloo", rank=rank)
            cp_rank = cp_rank_in_group(world, cp, rank)
            for lens in CASES:
                plan = build_cp_plan(lens, cp, cp_rank)
                vals = [float(gi) if gi >= 0 else -1.0 for gi in plan.local_index]
                x = torch.tensor(vals, dtype=torch.float32)
                full = all_gather_global(x, plan, group)
                want = torch.arange(sum(lens), dtype=torch.float32)
                if not torch.equal(full.cpu(), want):
                    q.put((rank, cp, lens, full.tolist()[:20]))
                    return
                dist.barrier()
        if rank == 0:
            q.put(None)
    finally:
        dist.destroy_process_group()


def main():
    world = 8
    port = _free_port()
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    mp.spawn(_worker, args=(world, port, q), nprocs=world, join=True)
    err = q.get()
    if err is not None:
        print(f"cp_runtime: FAIL {err}")
        raise SystemExit(1)
    print(f"cp_runtime: PASS (8 ranks gloo, cp 1/2/4/8, {len(CASES)} lens cases)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""最小验证：torch.npu.NPUGraph 能否捕获 HCCL all_reduce，并用更新后的静态输入重放。

用法（8 卡，新机）：
    source ./env_pack.sh
    python3.11 -m torch.distributed.run --nproc_per_node=8 tools/test_graph_hccl.py

判定：
  · capture_ok=1；
  · 第二次 replay 的结果 == 所有 rank 的 (rank+2) 之和（fp16 精确可表示）；
  · 与同输入的 eager all_reduce 逐位一致。
"""
import os
import sys

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    dev = f"npu:{local}"
    torch.npu.set_device(dev)
    dist.init_process_group("hccl", rank=rank, world_size=world)

    x = torch.full((1, 4096), float(rank + 1), dtype=torch.float16, device=dev)
    y = x.clone()
    dist.all_reduce(y)
    torch.npu.synchronize()
    eager0 = float(y.flatten()[0].item())

    static_in = torch.zeros_like(x)
    static_in.copy_(x)
    graph = torch.npu.NPUGraph()
    cap_ok, err = 0, ""
    out = None
    try:
        with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
            out = static_in.clone()
            dist.all_reduce(out)
        cap_ok = 1
    except Exception as e:                      # noqa: BLE001
        err = f"{type(e).__name__}: {e}"

    replay_ok, replay_val, eager1 = 0, float("nan"), float("nan")
    if cap_ok:
        try:
            static_in.fill_(float(rank + 2))
            graph.replay()
            torch.npu.synchronize()
            replay_val = float(out.flatten()[0].item())
            z = static_in.clone()
            dist.all_reduce(z)
            torch.npu.synchronize()
            eager1 = float(z.flatten()[0].item())
            replay_ok = 1
        except Exception as e:                  # noqa: BLE001
            err = f"replay {type(e).__name__}: {e}"

    print(f"[rank{rank}] capture_ok={cap_ok} replay_ok={replay_ok} "
          f"eager0={eager0:.1f} eager1={eager1:.1f} replay={replay_val:.1f} "
          f"match={int(replay_ok and replay_val == eager1)} err={err!r}", flush=True)
    sys.stdout.flush()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

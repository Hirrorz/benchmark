#!/usr/bin/env python3
"""厂商 ElasticBuffer 的 engram 取表路径最小验证（2 rank）：id→rank 映射 / 字节口径对不对。

背景（2026-10-15）：mini 的 engram 从"表常驻设备"切到官方口径（`enable_engram_offload=True`
+ `engram_tp_size=8`，走 `ElasticBuffer.engram_fetch/wait`）后，第一层 engram 跑完，下一层
MoE gate 就吃到 NaN —— 怀疑取回的数据不对。本探针把链路缩到最小：

    每卡 engram_write 自己那 1/tp 片表（**行内容 = 行号**，一眼可辨）→ engram_fetch(ids) → wait()
    → 和 host 侧 index_select 参考逐元素比。

先验 bf16（无 sf，排除量化干扰），再验 fp8_e4m3 + E8M0 scale（官方真实口径）。

用法（占卡，默认 0,1）：
    cd ds41_mini && source ./env_pack.sh && python3.11 tools/probe_engram_vendor.py --cards 0,1
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys

ROWS, DIM = 8192, 256          # 表 8192 行 × 256 列（fp8 时 8 个 E8M0 pack/行）


def free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def worker(rank: int, world: int) -> int:
    import torch                                    # noqa: E402
    import torch_npu                                # noqa: E402
    import torch.distributed as dist                # noqa: E402

    torch.npu.set_device("npu:0")
    master = os.environ["PEV_MASTER"].split(":")
    os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(world)
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = master[0], master[1]
    dist.init_process_group("hccl", rank=rank, world_size=world)
    dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(rank, init_comm=True)
    print(f"[rk{rank}] 进程组就绪 world={world}", flush=True)

    from cann_ops_transformer.ops import ElasticBuffer

    per = ROWS // world
    dev = "npu:0"
    # 行内容 = 行号（可辨认）；列上做一个固定小图案，方便定位"取错行/取错列"
    full = (torch.arange(ROWS, dtype=torch.float32)[:, None]
            + torch.arange(DIM, dtype=torch.float32)[None, :] * 0.001)
    # 覆盖：本 rank 片头/片尾、跨片边界、末行、以及均匀撒开几个点（都在 [0, ROWS) 内）
    cand = [0, 1, per - 1, per, per + 1, 2 * per + 7, ROWS - 1,
            ROWS // 3, ROWS * 2 // 3 + 5]
    ids = torch.tensor(sorted({c % ROWS for c in cand}), dtype=torch.int32, device=dev)

    case = os.environ.get("PEV_CASE", "fp8")

    def show(tag, out, ref, out_sf=None):
        ok = bool(torch.equal(out, ref))
        print(f"[rk{rank}] {tag} 取回 {tuple(out.shape)} 与 host 参考逐位一致 = {ok}"
              + (f"；sf 字节 {sorted(set(out_sf.view(torch.uint8).flatten().cpu().tolist()))[:6]}"
                 if out_sf is not None else ""), flush=True)
        if not ok:
            for k, i in enumerate(ids.tolist()):
                d = float((out[k].float() - ref[k].float()).abs().max())
                print(f"[rk{rank}]    id={i:7d} 期望 {float(ref[k, 0]):10.4f} 实际 {float(out[k, 0]):10.4f}"
                      f" 最大差 {d:.4g} {'✓' if d == 0 else '✗'}", flush=True)

    # ── ① bf16（无 sf）：只验 id→rank 映射与数据通路 ──
    if case in ("bf16", "both"):
        t_bf = full.to(torch.bfloat16)
        hint = ElasticBuffer.get_engram_storage_size_hint(per, DIM, torch.bfloat16)
        buf = ElasticBuffer(dist.group.WORLD, num_cpu_bytes=hint)
        buf.engram_write(t_bf[rank * per:(rank + 1) * per].contiguous())
        show("① bf16", buf.engram_fetch(ids)(),
             t_bf.index_select(0, ids.cpu().long()).to(dev))

    # ── ② fp8_e4m3 + E8M0 scale（官方真实口径）：验 sf 的字节与解量化 ──
    #   case=fp8     storage 走 CPU（doc 里写的用法）
    #   case=fp8npu  storage 先搬 NPU 再写（**官方 modules/engram.py:481-484 的用法**）
    #   case=fp8nosf 只写 fp8、不带 sf（隔离"sf 参与"这个变量）
    if case in ("fp8", "fp8fullsf", "fp8npu", "fp8nosf", "both"):
        q = (full % 7 + 1).to(torch.float8_e4m3fn)                  # 正值、范围 1..7
        sf_u8 = torch.full((ROWS, DIM // 32), 127, dtype=torch.uint8)  # E8M0 127 = 2^0 = 1
        hint8 = ElasticBuffer.get_engram_storage_size_hint(per, DIM, torch.float8_e4m3fn)
        buf8 = ElasticBuffer(dist.group.WORLD, num_cpu_bytes=hint8)
        shard_q = q[rank * per:(rank + 1) * per].contiguous()
        if case == "fp8npu":
            shard_q = shard_q.to(dev)
        if case == "fp8nosf":
            buf8.engram_write(shard_q)
        elif case == "fp8fullsf":
            # 假设：sf 不入通信域 → 远端行的 sf 取不到；那每卡带上**整表 sf** 自己解量化
            buf8.engram_write(shard_q, sf=sf_u8.contiguous().view(torch.float8_e8m0fnu).to(dev))
        else:
            buf8.engram_write(shard_q,
                              sf=sf_u8[rank * per:(rank + 1) * per].contiguous()
                              .view(torch.float8_e8m0fnu).to(dev))
        print(f"[rk{rank}] ② buffer 元信息：num_entries={buf8._engram_num_entries} "
              f"hidden={buf8._engram_hidden_size} rank_size={buf8._rank_size} "
              f"comm_buf={buf8._comm_buffer_size}", flush=True)
        got = buf8.engram_fetch(ids)()
        got, got_sf = got if isinstance(got, tuple) else (got, None)
        refq = q.index_select(0, ids.cpu().long()).to(torch.float32).to(dev)
        if got_sf is None:
            print(f"[rk{rank}] ② 没返回 sf（fp8nosf 就是预期的）", flush=True)
            show("② fp8(无sf)", got.to(torch.float32), refq)
        else:
            scale = (got_sf.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
            deq = (got.to(torch.float32).unflatten(-1, (-1, 32)) * scale.unsqueeze(-1)).flatten(-2)
            refd = (refq.unflatten(-1, (-1, 32)) * 1.0).flatten(-2)       # sf=1 → 参考就是原值

            # ③ 真实规模计时：16k 命中点每层 640 token × 24 个 id。官方同点 EngramFetch 256 µs /
            #    EngramFetchWait 862 µs（它传的是**本片 sf**）—— 这里就看 sf 传法会不会把 wait 拖慢。
            import time
            ids_big = torch.randint(0, ROWS, (640 * 24,), dtype=torch.int32, device=dev)
            for _ in range(3):
                _ = buf8.engram_fetch(ids_big)()
            torch.npu.synchronize()
            t0 = time.perf_counter()
            bw = buf8.engram_fetch(ids_big)
            t1 = time.perf_counter()
            _ = bw()
            torch.npu.synchronize()
            t2 = time.perf_counter()
            print(f"[rk{rank}] ③ 计时 {ids_big.numel()} ids：fetch {(t1-t0)*1e6:.1f} µs  "
                  f"wait+sync {(t2-t1)*1e6:.1f} µs（sf 传法 = {case}）", flush=True)
            show("② fp8+E8M0", deq, refd, got_sf)

    dist.barrier()
    print(f"[rk{rank}] 探针结束", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="0,1")
    ap.add_argument("--case", default="fp8", choices=["fp8", "fp8fullsf", "fp8npu", "fp8nosf", "bf16", "both"],
                    help="一个进程里只建一个 ElasticBuffer（厂商的 host 窗口是**进程级共享池**，"
                         "同 group 建第二个会复用第一个的存储 → 混着跑会互相污染）")
    args, _ = ap.parse_known_args()
    cards = [c.strip() for c in args.cards.split(",") if c.strip()]

    if os.environ.get("PEV_RANK") is None:                 # 父进程：起 world 个子进程
        world = len(cards)
        port = free_port()
        here = os.path.abspath(__file__)
        procs = []
        for r, card in enumerate(cards):
            env = os.environ.copy()
            env["PEV_RANK"], env["PEV_MASTER"] = str(r), f"127.0.0.1:{port}"
            env["PEV_WORLD"] = str(world)                 # 子进程 argv 里没有 --cards，world 走 env
            env["PEV_CASE"] = args.case
            env["ASCEND_RT_VISIBLE_DEVICES"] = card       # 每进程只见自己那张卡（逻辑恒 0）
            env.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "3")
            procs.append(subprocess.Popen([sys.executable, "-u", here], env=env))
        codes = [p.wait() for p in procs]
        print(f"== 全部结束：{[f'rank{r}={c}' for r, c in enumerate(codes)]}")
        return 0 if all(c == 0 for c in codes) else 1

    return worker(int(os.environ["PEV_RANK"]), int(os.environ.get("PEV_WORLD", "2")))


if __name__ == "__main__":
    sys.exit(main())

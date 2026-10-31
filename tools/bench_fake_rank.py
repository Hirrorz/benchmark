#!/usr/bin/env python3.11
# coding=utf-8
"""bench_fake_rank.py — 无 NPU 的假 rank，仅用于自测 bench_server/bench_cli 协议。

它和 runner.py --serve 的**控制协议完全一致**：ctl socket 收 JSONL，先 ack 再 result；
stdout 打印和真 rank 相同的进度行（[W1 1/40]、[ 1/40] layer、[inc 1/3]），
所以 server 的进度转发、ACK 等待、结果聚合、客户端进度条都能在没卡时验证。

用法（由 bench_server.py --fixture 拉起，不直接手跑）：
  python3 bench_fake_rank.py --rank 0 --control /tmp/.../rank0.ctl --world 8
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_protocol as P


def send(conn, obj):
    conn.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--control", required=True)
    ap.add_argument("--serve-l-max", type=int, default=4096)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.control), exist_ok=True)
    try:
        os.unlink(a.control)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(a.control)
    srv.listen(1)
    os.chmod(a.control, 0o600)
    mode = "fixture"
    print(f"[bench] rank {a.rank} serve ready mode={mode} l_pool={a.serve_l_max} "
          f"ctl={a.control}", flush=True)
    conn, _ = srv.accept()
    srv.close()
    f = conn.makefile("r", encoding="utf-8", errors="replace", newline="\n")
    warmed = False
    try:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError:
                continue
            op = cmd.get("op")
            cid = cmd.get("cmd_id")
            if op == "stop":
                send(conn, {"type": "bye", "cmd_id": cid})
                break
            if op == "cancel":
                send(conn, {"type": "ack", "cmd_id": cid, "op": "cancel"})
                continue
            send(conn, {"type": "ack", "cmd_id": cid, "op": op})
            try:
                if op == "status":
                    send(conn, {"type": "result", "cmd_id": cid, "result": {
                        "ok": True, "rank": a.rank, "pid": os.getpid(),
                        "mode": mode, "l_pool": a.serve_l_max, "warmed": warmed,
                        "ac_layers": 40 if warmed else 0,
                        "nz_layers": 40 if warmed else 0,
                        "hostcache_attached": False, "hostcache_hits": 0,
                        "allocated_gib": 0.0,
                    }})
                elif op == "hostcache":
                    send(conn, {"type": "result", "cmd_id": cid, "result": {
                        "ok": True, "hostcache": {"ok": True, "fixture": True}}})
                elif op == "warmup":
                    # W1
                    for i in range(1, 41):
                        print(f"  [W1 {i:>2}/40] layer {i-1} win T=128 算子 24",
                              flush=True)
                        time.sleep(0.005)
                    for i in range(1, 41):
                        print(f"  [W2 {i:>2}/40] layer {i-1} win T=1 算子 24",
                              flush=True)
                        time.sleep(0.005)
                    lengths = cmd.get("lengths") or [128]
                    for i, L in enumerate(lengths, 1):
                        time.sleep(0.02)
                        print(f"  [W3 {i}/{len(lengths)}] L={L} 0.900s", flush=True)
                    warmed = True
                    send(conn, {"type": "result", "cmd_id": cid, "result": {
                        "ok": True, "mode": mode,
                        "w1_s": 0.9 + a.rank * 0.01, "w2_s": 0.4,
                        "ac_layers": 40, "nz_layers": 40, "dropped_raw": 15360,
                        "max_allocated_gib": 70.0 + a.rank * 0.1,
                        "buckets": [{"L": L, "prefill_s": 0.9} for L in lengths],
                    }})
                elif op == "run":
                    n_new = int(cmd.get("decode", 0) or 0)
                    rep = max(1, int(cmd.get("repeat", 1) or 1))
                    runs = []
                    for k in range(rep):
                        for i in range(1, 41, 8):
                            print(f"  [{i:>2}/40] layer {i-1} win T=512 算子 24",
                                  flush=True)
                            time.sleep(0.005)
                        ids = [1, 2, 3, 396]
                        for st in range(1, n_new + 1):
                            print(f"  [inc {st}/{n_new}] 预测位置 {3+st} → id=396 "
                                  f"本步 0.69s", flush=True)
                            time.sleep(0.003)
                        runs.append({
                            "prefill_s": 1.00 + a.rank * 0.01 + k * 0.001,
                            "p2d_s": 0.0, "d2p_s": 0.0,
                            "decode_ms_per_token": 690.0 + a.rank,
                            "first_token": 396, "text": " that",
                        })
                    send(conn, {"type": "result", "cmd_id": cid, "result": {
                        "ok": True, "mode": mode, "runs": runs,
                        "summary": {"prefill_s": runs[-1]["prefill_s"],
                                    "p2d_s": 0.0, "d2p_s": 0.0,
                                    "decode_ms_per_token": runs[-1]["decode_ms_per_token"]},
                        "n_new": n_new, "ids": [1, 2, 3, 396], "text": " that",
                    }})
                else:
                    send(conn, {"type": "error", "cmd_id": cid,
                                "error": f"unknown op {op}", "recoverable": True})
            except Exception as exc:                    # noqa: BLE001
                send(conn, {"type": "error", "cmd_id": cid,
                            "error": f"{type(exc).__name__}: {exc}",
                            "recoverable": False})
    finally:
        try:
            f.close()
            conn.close()
        except Exception:                               # noqa: BLE001
            pass
        try:
            os.unlink(a.control)
        except Exception:                               # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

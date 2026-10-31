#!/usr/bin/env python3
"""尾段空泡分析：按 HcPre 切 forward/层，算 union-busy 空泡，并按 next_op / prev 类型聚合。

用法：
  python3.11 tools/analyze_tail.py <trace 根目录或 mindstudio_profiler_output 目录> \
      [--layers 40] [--tail-lo 21] [--tail-hi 40] [--device 0] [-o OUTDIR]

口径（与 NOTES/2026-10-28-tail-bubble-host-bound-opt.md 一致）：
  · 每层固定 2 个 HcPre → 80 个一行 forward（40 层）；
  · busy = 所有非等待类 kernel 在时间轴上的并集（EVENT WAIT / NOTIFY WAIT 等不算忙）；
  · idle% = 1 - busy/span；
  · gap = busy 并集的补集，next_op = gap 结束后开始的第一个算子，
    prev = gap 开始前结束的最后一个算子（按 kernel_type 聚合）。

输出：终端摘要 + OUTDIR/TAILBUBBLE_LAYERS.txt + OUTDIR/TAILBUBBLE_GAPS.txt
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys

WAIT = {"NOTIFY_WAIT_SQE", "EVENT WAIT", "NOTIFY WAIT", "EVENT RECORD",
        "NOTIFY_RECORD_SQE", "NOTIFY RECORD"}


def find_trace_dir(root):
    if os.path.basename(root.rstrip("/")) == "mindstudio_profiler_output":
        return root
    hits = glob.glob(os.path.join(root, "**", "mindstudio_profiler_output"), recursive=True)
    if not hits:
        return None
    return sorted(hits)[0]


def load_task_time(td):
    f = glob.glob(os.path.join(td, "task_time_*.csv"))
    if not f:
        raise SystemExit(f"没有 task_time_*.csv：{td}")
    rows = list(csv.DictReader(open(f[0])))
    # 多 device 时取行数最多的那颗（通常就是被采集的 rank）
    by_dev = {}
    for r in rows:
        by_dev.setdefault(r.get("Device_id", "0"), []).append(r)
    dev = max(by_dev, key=lambda k: len(by_dev[k]))
    rows = by_dev[dev]
    iv = []
    for r in rows:
        s = float(r["task_start(us)"])
        iv.append((s, s + float(r["task_time(us)"]), r["stream_id"],
                   r["kernel_type"], r["kernel_name"]))
    iv.sort()
    return iv, dev


def load_hcpre(td):
    f = glob.glob(os.path.join(td, "msprof_*.json"))
    if not f:
        raise SystemExit(f"没有 msprof_*.json：{td}")
    ev = json.load(open(f[0]))
    return sorted(float(e["ts"]) for e in ev
                  if isinstance(e, dict) and e.get("ph") == "X"
                  and str(e.get("name")) == "HcPre")


def union(segs):
    out = []
    for s, e in sorted(segs):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def clip(segs, s, e):
    out = []
    for a, b in segs:
        a2, b2 = max(a, s), min(b, e)
        if b2 > a2:
            out.append((a2, b2))
    return out


def busy_of(iv, s, e):
    return union(clip([(a, b) for a, b, _st, k, _n in iv if k not in WAIT], s, e))


def gaps_of(busy, s, e):
    gs = []
    cur = s
    for a, b in busy:
        if a > cur:
            gs.append((cur, a))
        cur = max(cur, b)
    if e > cur:
        gs.append((cur, e))
    return gs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--tail-lo", type=int, default=21)
    ap.add_argument("--tail-hi", type=int, default=40)
    ap.add_argument("-o", "--outdir", default="")
    a = ap.parse_args()

    td = find_trace_dir(a.trace)
    if not td:
        raise SystemExit(f"找不到 mindstudio_profiler_output：{a.trace}")
    iv, dev = load_task_time(td)
    hc = load_hcpre(td)
    per = 2 * a.layers
    nf = len(hc) // per
    if nf < 2:
        raise SystemExit(f"HcPre 只有 {len(hc)} 个，不足 2 个 forward（per={per}）")
    print(f"trace={td}\ndevice={dev} tasks={len(iv)} HcPre={len(hc)} forwards={nf}")

    spans, tails, tidles, layer_rows = [], [], [], {}
    for k in range(nf - 1):
        f0, f1 = hc[per * k], hc[per * (k + 1)]
        spans.append((f1 - f0) / 1000)
        ts, te = hc[per * k + 2 * a.tail_lo], f1
        b = busy_of(iv, ts, te)
        tails.append((te - ts) / 1000)
        tidles.append((1 - sum(y - x for x, y in b) / (te - ts)) * 100)
        for li in range(a.layers):
            s, e = hc[per * k + 2 * li], hc[per * k + 2 * li + 2]
            bb = busy_of(iv, s, e)
            layer_rows.setdefault(li, []).append(
                ((e - s) / 1000, (1 - sum(y - x for x, y in bb) / (e - s)) * 100))

    print(f"whole: steady(median fwd, 剔除首轮)={statistics.median(spans):.1f} ms")
    print(f"tail L{a.tail_lo}-{a.tail_hi - 1}: span={statistics.median(tails):.1f} ms "
          f"idle={statistics.median(tidles):.1f}%")

    # 尾段 gap 聚合（所有 steady forward 合并）
    import bisect
    starts = [x[0] for x in iv]
    next_agg, prev_agg = {}, {}
    for k in range(nf - 1):
        f1 = hc[per * (k + 1)]
        ts, te = hc[per * k + 2 * a.tail_lo], f1
        busy = busy_of(iv, ts, te)
        for gs, ge in gaps_of(busy, ts, te):
            i = bisect.bisect_left(starts, ge - 1e-6)
            nxt = iv[i] if i < len(iv) else None
            prv = None
            j = bisect.bisect_left(starts, gs)
            for x in reversed(iv[max(0, j - 200):j]):
                if x[1] <= gs + 1e-6:
                    prv = x
                    break
            if nxt is not None:
                key = nxt[4]
                e = next_agg.setdefault(key, [0, 0.0])
                e[0] += 1
                e[1] += ge - gs
            if prv is not None:
                key = prv[3]
                e = prev_agg.setdefault(key, [0, 0.0])
                e[0] += 1
                e[1] += ge - gs

    print(f"\n尾段 top gap（按 next_op，{nf - 1} 个 forward 合计）")
    for name, (n, us) in sorted(next_agg.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"  {name:<40s} n={n:<5d} total={us / 1000:8.2f} ms  avg={us / n:7.1f} us")
    print("\n尾段 gap 前驱类型")
    for name, (n, us) in sorted(prev_agg.items(), key=lambda kv: -kv[1][1])[:10]:
        print(f"  {name:<18s} n={n:<5d} total={us / 1000:8.2f} ms  avg={us / n:7.1f} us")

    if a.outdir:
        os.makedirs(a.outdir, exist_ok=True)
        lp = os.path.join(a.outdir, "TAILBUBBLE_LAYERS.txt")
        with open(lp, "w") as fh:
            fh.write("layer\tspan_med_ms\tidle_med_pct\n")
            for li in range(a.layers):
                rs = layer_rows[li]
                fh.write(f"{li}\t{statistics.median([x for x, _ in rs]):.3f}\t"
                         f"{statistics.median([y for _, y in rs]):.1f}\n")
        gp = os.path.join(a.outdir, "TAILBUBBLE_GAPS.txt")
        with open(gp, "w") as fh:
            fh.write("# next_op\tcount\ttotal_us\tavg_us\n")
            for name, (n, us) in sorted(next_agg.items(), key=lambda kv: -kv[1][1]):
                fh.write(f"{name}\t{n}\t{us:.1f}\t{us / n:.1f}\n")
            fh.write("# prev_type\tcount\ttotal_us\tavg_us\n")
            for name, (n, us) in sorted(prev_agg.items(), key=lambda kv: -kv[1][1]):
                fh.write(f"{name}\t{n}\t{us:.1f}\t{us / n:.1f}\n")
        print(f"\n已写 {lp}\n已写 {gp}")


if __name__ == "__main__":
    main()

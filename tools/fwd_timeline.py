#!/usr/bin/env python3
"""把一个已导出的 `task_time_*.csv` 切成「逐 forward」时间线，回答两个问题：

  1) 每个 forward 的 **wall span**（首个 kernel 开始 → 末个 kernel 结束）是多少；
  2) span 内 **设备空泡**（所有 stream 都没 kernel 的区间）有多少、最大几段在哪个 kernel 之后。

用法：
  python3 tools/fwd_timeline.py <mindstudio_profiler_output 目录 | task_time_*.csv> \
      [--split-us 2000] [--top 8] [--csv out.csv]

切分规则：相邻 task 的间隔 > --split-us 视为 forward 边界（默认 2 ms；forward 内空泡都能被 --top 列出来）。
"""
from __future__ import annotations
import argparse, csv, glob, os, sys


def find_csv(path):
    if os.path.isdir(path):
        cand = sorted(glob.glob(os.path.join(path, "task_time_*.csv")))
        if not cand:
            sys.exit(f"目录里没有 task_time_*.csv: {path}")
        return cand[0]
    return path


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                s = float((r.get("task_start(us)") or "").strip())
                e = float((r.get("task_stop(us)") or "").strip())
            except ValueError:
                continue
            if e < s:
                continue
            rows.append((s, e, (r.get("stream_id") or "").strip(),
                         (r.get("kernel_name") or "").strip()))
    rows.sort(key=lambda x: x[0])
    return rows


def merge_busy(iv):
    """把所有 stream 的 kernel 区间并起来 → 返回总 busy 时长与并集区间列表。"""
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return sum(e - s for s, e in out), out


def split_forwards(rows, split_us):
    fwds, cur = [], [rows[0]] if rows else []
    for prev, nxt in zip(rows, rows[1:]):
        if nxt[0] - prev[1] > split_us:
            fwds.append(cur)
            cur = []
        cur.append(nxt)
    if cur:
        fwds.append(cur)
    return fwds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--split-us", type=float, default=2000.0)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    rows = load(find_csv(a.path))
    if not rows:
        sys.exit("没有可用 task 行")
    fwds = split_forwards(rows, a.split_us)
    t0 = rows[0][0]

    print(f"# {len(rows)} tasks / {len(fwds)} forwards（切分阈值 {a.split_us:.0f} µs）")
    print(f"{'idx':>3} {'start(s)':>10} {'span(ms)':>9} {'busy(ms)':>9} {'idle(ms)':>9} {'idle%':>6} {'streams':>7}")
    recs = []
    for i, f in enumerate(fwds):
        span = f[-1][1] - f[0][0]
        busy, _ = merge_busy([(s, e) for s, e, _, _ in f])
        streams = len({st for _, _, st, _ in f})
        idle = span - busy
        print(f"{i:>3} {(f[0][0]-t0)/1e6:>10.3f} {span/1e3:>9.2f} {busy/1e3:>9.2f} {idle/1e3:>9.2f} "
              f"{100.0*idle/max(span,1e-9):>5.1f}% {streams:>7}")
        recs.append((i, f, span, busy))

    # 只看最后一个 forward（稳态）里的空泡分布
    if recs:
        i, f, span, busy = recs[-1]
        print(f"\n## forward #{i} 内空泡 top{a.top}（设备全静默的区间）")
        _, uni = merge_busy([(s, e) for s, e, _, _ in f])
        gaps = []
        for (s0, e0), (s1, _) in zip(uni, uni[1:]):
            if s1 - e0 > 0:
                gaps.append((s1 - e0, e0, s1))
        gaps.sort(reverse=True)
        for g, gs, ge in gaps[:a.top]:
            before = [k for s, e, _, k in f if abs(e - gs) < 1.0]
            after = [k for s, e, _, k in f if abs(s - ge) < 1.0]
            print(f"  {g/1e3:>7.2f} ms  @{ (gs-f[0][0])/1e3:>8.2f} ms  "
                  f"after={before[0][:60] if before else '?'}  next={after[0][:50] if after else '?'}")
        print(f"  合计空泡 {sum(g for g, _, _ in gaps)/1e3:.2f} ms / span {span/1e3:.2f} ms")

        # 空泡按「后一个 kernel」聚合：谁把设备等住了，一目了然
        agg = {}
        for g, gs, ge in gaps:
            nxt = [k for s, e, _, k in f if abs(s - ge) < 1.0]
            key = nxt[0][:48] if nxt else "?"
            n, tot = agg.get(key, (0, 0.0))
            agg[key] = (n + 1, tot + g)
        print("  按 next-kernel 聚合（次数 × 合计 ms）:")
        for k, (n, tot) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:a.top]:
            print(f"    {tot/1e3:>7.2f} ms  n={n:<4d} avg={tot/max(n,1):>7.0f} us  {k}")

    if a.csv:
        with open(a.csv, "w", newline="") as fo:
            w = csv.writer(fo)
            w.writerow(["fwd", "span_ms", "busy_ms", "idle_ms", "tasks", "streams"])
            for i, f, span, busy in recs:
                w.writerow([i, f"{span/1e3:.2f}", f"{busy/1e3:.2f}", f"{(span-busy)/1e3:.2f}",
                            len(f), len({st for _, _, st, _ in f})])
        print(f"\n→ {a.csv}")


if __name__ == "__main__":
    main()

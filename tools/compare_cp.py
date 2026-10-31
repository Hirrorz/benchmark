#!/usr/bin/env python3
"""两套采集（不同 CP/并行策略）的逐点位对照：ms、比值、按 L / 按 T_pass 的汇总。

输入：两份 `TRACE_POINTS.csv`（`tools/trace_matrix.py` 的产物，trace 口径）。
输出：Markdown 对照表（默认写到第二份的目录）+ 终端摘要。

口径提醒：这里比的是 **trace 设备时间线**（相邻 forward 起点差的中位、剔除首个）；
**不能**用 `results.tsv`（含 msprof stop，小 T 虚高 1.4–25×）。

用法：
    python3 tools/compare_cp.py <A.csv> <B.csv> [-o out.md] [--labels CP2 CP4]
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys


def load(p):
    out = {}
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                ms = float(r["steady_ms"])
                L = int(r["L"])
                h = float(r["hit"])
                T = int(r["T_pass"])
            except (KeyError, ValueError):
                continue
            if ms != ms:
                continue
            out[(L, h)] = dict(ms=ms, T=T, idle=float(r.get("idle_pct") or "nan"))
    return out


def fmt_L(L):
    for unit, div in (("m", 1 << 20), ("k", 1 << 10)):
        if L >= div and L % div == 0:
            return f"{L // div}{unit}"
    return str(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("-o", "--out", default="")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    la, lb = a.labels
    common = sorted(set(A) & set(B), key=lambda k: (k[0], k[1]))
    if not common:
        sys.exit("✗ 两套没有共同点位（检查命名/点位集）")

    Ls = sorted({k[0] for k in common})
    hs = sorted({k[1] for k in common})
    out = [f"# {la} vs {lb}（trace 口径，{len(common)} 个共同点位）", "",
           f"- {la}: `{a.a}`", f"- {lb}: `{a.b}`",
           "- 单元格 = `B/A` 的 ms 比值；<1 = B 更快", "",
           "| L \\ hit | " + " | ".join(f"{h:g}" for h in hs) + " |",
           "|" + "---|" * (len(hs) + 1)]
    ratios = []
    for L in Ls:
        cells = []
        for h in hs:
            k = (L, h)
            if k in A and k in B and A[k]["ms"] > 0:
                r = B[k]["ms"] / A[k]["ms"]
                ratios.append((r, L, h, A[k]["ms"], B[k]["ms"]))
                cells.append(f"{r:.2f}")
            else:
                cells.append("—")
        out.append(f"| {fmt_L(L)} | " + " | ".join(cells) + " |")

    out += ["", "## 按 L 汇总（比值中位）", "", "| L | 中位比值 | 点数 |", "|---|---:|---:|"]
    for L in Ls:
        v = [r for r, LL, _h, _ma, _mb in ratios if LL == L]
        if v:
            out.append(f"| {fmt_L(L)} | {statistics.median(v):.2f} | {len(v)} |")

    allr = [r for r, *_ in ratios]
    out += ["", "## 总体", "",
            f"- 比值中位 **{statistics.median(allr):.2f}**，范围 "
            f"{min(allr):.2f}–{max(allr):.2f}（{len(allr)} 个点）"]
    # 高命中（命中路径）与低命中分开看
    hi = [r for r, _, h, *_ in ratios if h >= 0.9]
    lo = [r for r, _, h, *_ in ratios if h <= 0.6]
    if hi:
        out.append(f"- 高命中（hit≥0.9）中位 {statistics.median(hi):.2f}（n={len(hi)}）")
    if lo:
        out.append(f"- 低命中（hit≤0.6）中位 {statistics.median(lo):.2f}（n={len(lo)}）")

    worst = sorted(ratios, reverse=True)[:5]
    best = sorted(ratios)[:5]
    out += ["", f"## {lb} 相对最慢的 5 个点", "",
            "| L | hit | T_pass | " + la + " ms | " + lb + " ms | 比值 |",
            "|---|---:|---:|---:|---:|---:|"]
    for r, L, h, ma, mb in worst:
        out.append(f"| {fmt_L(L)} | {h:g} | {A[(L, h)]['T']} | {ma:.0f} | {mb:.0f} | {r:.2f} |")
    out += ["", f"## {lb} 相对最快的 5 个点", "",
            "| L | hit | T_pass | " + la + " ms | " + lb + " ms | 比值 |",
            "|---|---:|---:|---:|---:|---:|"]
    for r, L, h, ma, mb in best:
        out.append(f"| {fmt_L(L)} | {h:g} | {A[(L, h)]['T']} | {ma:.0f} | {mb:.0f} | {r:.2f} |")

    p = a.out or os.path.join(os.path.dirname(os.path.abspath(a.b)), "COMPARE.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n→ {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

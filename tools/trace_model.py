#!/usr/bin/env python3
"""检验并拟合**时间模型**：`ms/forward` 到底由什么决定（T_pass？L？命中率？）。

输入：`tools/trace_matrix.py` 产的 `TRACE_POINTS.csv`
      （列：L, L_text, hit, T_pass, steady_ms, busy_ms, idle_pct, forwards, note）

做四件事：
  1. **同一 T_pass 跨 L 的一致性**——若 ms 只由 T_pass 决定，同一 T 下不同 L/命中率的点
     应该落在同一水平（这是"L 不影响 forward"的判据，也是 2026-10-29 用户纠正的模型）；
  2. 分箱（按 T_pass 对数分箱）给中位/IQR，看曲线形状；
  3. 最小二乘 `ms = a + b·T_pass`（可选对数形式），给 R²；
  4. 残差按 L 聚合——找出**偏离模型**的 L 档（那些才是 L 真的进来地方，例如大 L 的
     固定额外开销）。

用法： python3 tools/trace_model.py <TRACE_POINTS.csv>
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys


def load(p):
    rows = []
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                ms = float(r["steady_ms"])
            except (KeyError, ValueError):
                continue
            if ms != ms:
                continue
            rows.append(dict(L=int(r["L"]), Ls=r.get("L_text") or r["L"],
                             h=float(r["hit"]), T=int(r["T_pass"]), ms=ms,
                             busy=float(r.get("busy_ms") or "nan"),
                             idle=float(r.get("idle_pct") or "nan")))
    return rows


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, (1 - ss_res / ss_tot if ss_tot else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default="")
    a = ap.parse_args()
    rows = load(a.csv)
    if len(rows) < 4:
        sys.exit(f"✗ 有效点位只有 {len(rows)} 个")
    outdir = a.out or os.path.dirname(os.path.abspath(a.csv))
    out = [f"# forward 时间模型（{len(rows)} 个点位，trace 口径）", ""]

    # ① 同一 T_pass 跨 L
    out += ["## 1. 同一 T_pass、不同 L 的一致性", "",
            "同一 `T_pass` 上把不同 L 的点放一起：若 ms 只由 T_pass 决定，它们应该同水平。", ""]
    by_t: dict[int, list] = {}
    for r in rows:
        by_t.setdefault(r["T"], []).append(r)
    checked = 0
    out += ["| T_pass | 点数 | L 范围 | ms 中位 | ms 全距 | 相对全距 |", "|---:|---:|---|---:|---:|---:|"]
    worst = []
    for T in sorted(by_t):
        g = by_t[T]
        if len(g) < 2 or len({x["L"] for x in g}) < 2:
            continue
        v = [x["ms"] for x in g]
        med = statistics.median(v)
        spread = max(v) - min(v)
        rel = spread / med if med else float("nan")
        checked += 1
        if rel > 0.15:
            worst.append((rel, T, g))
        if len(by_t) <= 12 or checked <= 12:
            out.append(f"| {T} | {len(g)} | {min(x['Ls'] for x in g)}–{max(x['Ls'] for x in g)} "
                       f"| {med:.0f} | {spread:.0f} | {rel * 100:.0f}% |")
    out += ["", f"可比的（同 T、≥2 个 L）：**{checked} 组**；相对全距 >15% 的：**{len(worst)} 组**"]
    for rel, T, g in sorted(worst, reverse=True)[:5]:
        det = "，".join(f"L{x['Ls']}/h{x['h']:g}={x['ms']:.0f}" for x in sorted(g, key=lambda y: y["ms"]))
        out.append(f"- T={T}（全距 {rel * 100:.0f}%）：{det}")

    # ② 分箱曲线
    out += ["", "## 2. 分箱中位（按 T_pass 二分箱，单调展示）", "",
            "| T_pass 区间 | 点数 | ms 中位 | IQR |", "|---|---:|---:|---:|"]
    ts = sorted(by_t)
    bins = [ts[i:i + max(1, len(ts) // 10)] for i in range(0, len(ts), max(1, len(ts) // 10))]
    for b in bins:
        g = [x for T in b for x in by_t[T]]
        v = sorted(x["ms"] for x in g)
        q1 = v[len(v) // 4]
        q3 = v[(3 * len(v)) // 4]
        out.append(f"| {b[0]}–{b[-1]} | {len(g)} | {statistics.median(v):.0f} | {q3 - q1:.0f} |")

    # ③ 线性拟合
    xs = [r["T"] for r in rows]
    ys = [r["ms"] for r in rows]
    f = ols(xs, ys)
    out += ["", "## 3. 拟合", ""]
    if f:
        a0, b0, r2 = f
        out.append(f"- `ms ≈ {a0:.1f} + {b0 * 1e3:.3f} µs × T_pass`（R² = {r2:.4f}，n = {len(rows)}）")
        out.append(f"- 每 10k token ≈ {b0 * 1e4:.0f} ms；T=0 截距 ≈ {a0:.0f} ms")
        resid = [(r["ms"] - (a0 + b0 * r["T"]), r) for r in rows]
        by_l: dict[str, list] = {}
        for d, r in resid:
            by_l.setdefault(r["Ls"], []).append(d)
        out += ["", "## 4. 残差按 L（>0 = 比模型慢）", "", "| L | 残差中位 ms | 最大正残差 |", "|---|---:|---:|"]
        for Ls in sorted(by_l, key=lambda s: rows[0]["L"]):
            v = by_l.get(Ls)
            if not v:
                continue
            out.append(f"| {Ls} | {statistics.median(v):+.0f} | {max(v):+.0f} |")

    # ⑤ idle
    idl = [r["idle"] for r in rows if r["idle"] == r["idle"]]
    if idl:
        out += ["", f"## 5. 设备空闲占比：中位 {statistics.median(idl):.0f}%"
                    f"（最小 {min(idl):.0f}% / 最大 {max(idl):.0f}%）"]

    md = os.path.join(outdir, "TRACE_MODEL.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n→ {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

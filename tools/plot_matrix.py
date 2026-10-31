#!/usr/bin/env python3
"""把 run_hit 多点位的 `results.tsv` 画成 SVG（不依赖 matplotlib）。

口径：每个点是 **N 个稳态 forward 的总时间 ÷ N**（run_hit 的 `纯 forward` 输出）。
用法：
  python3 tools/plot_matrix.py <results.tsv> [--out <prefix>] [--title ...]
输出：
  <prefix>.svg        —— x=命中率、y=ms/forward（log10），每条线一个 L
  <prefix>.md         —— 同数据的 Markdown 表
  <prefix>.csv        —— 原始行（tag, L, hit, ms_per_forward, errors）
"""
import argparse
import csv
import math
import os
import re
import sys

TAG_RE = re.compile(r"\((\d+)_([0-9.]+)\)_r1")

COLORS = ["#2f6fed", "#e05555", "#1f9d63", "#b26ad4", "#d98a1f", "#4bb3c9"]


def load(tsv):
    rows = []
    with open(tsv) as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for r in rd:
            m = TAG_RE.search(r.get("tag", ""))
            if not m:
                continue
            L = int(m.group(1))
            hit = float(m.group(2))
            try:
                ms = float(r.get("ms_per_forward", r.get("ms_per_pass", "nan")))
            except ValueError:
                continue
            if not math.isfinite(ms) or ms <= 0:
                continue
            rows.append({"L": L, "hit": hit, "ms": ms,
                         "errors": int(r.get("errors", 0) or 0)})
    return rows


def nice_ticks(lo, hi):
    """log10 轴刻度：1/2/5 × 10^k。"""
    out = []
    k = math.floor(math.log10(lo))
    while 10 ** k <= hi * 10:
        for m in (1, 2, 5):
            v = m * 10 ** k
            if lo <= v <= hi:
                out.append(v)
        k += 1
    return sorted(set(out))


def svg(rows, title, path):
    W, H = 1100, 640
    ML, MR, MT, MB = 95, 210, 70, 75
    pw, ph = W - ML - MR, H - MT - MB
    Ls = sorted({r["L"] for r in rows})
    hits = sorted({r["hit"] for r in rows})
    x_lo, x_hi = min(hits), max(hits)
    if x_hi == x_lo:
        x_hi = x_lo + 1e-6
    ms_lo = min(r["ms"] for r in rows)
    ms_hi = max(r["ms"] for r in rows)
    y_lo, y_hi = ms_lo * 0.75, ms_hi * 1.35
    y0, y1 = math.log10(y_lo), math.log10(y_hi)

    def X(h):
        return ML + (h - x_lo) / (x_hi - x_lo) * pw

    def Y(ms):
        return MT + (y1 - math.log10(ms)) / (y1 - y0) * ph

    def fmt_ms(v):
        return f"{v/1000:.2f}s" if v >= 1000 else f"{v:.1f}"

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="DejaVu Sans, Noto Sans CJK SC, sans-serif">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="{ML}" y="38" font-size="20" font-weight="bold" fill="#222">{title}</text>',
           f'<text x="{ML}" y="60" font-size="12" fill="#666">ms per pass (log scale) · '
           f'total_time / n_passes · DP8 · 1 batch/DP · KEEP=140 · deepgemm · graph on</text>']
    # grid + y ticks
    for v in nice_ticks(y_lo, y_hi):
        y = Y(v)
        out.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+pw}" y2="{y:.1f}" '
                   f'stroke="#e6e9ef" stroke-width="1"/>')
        out.append(f'<text x="{ML-10}" y="{y+4:.1f}" font-size="11" fill="#555" '
                   f'text-anchor="end">{fmt_ms(v)}</text>')
    # x ticks
    for h in hits:
        x = X(h)
        out.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{MT+ph}" '
                   f'stroke="#f0f2f6" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{MT+ph+22}" font-size="12" fill="#555" '
                   f'text-anchor="middle">{h*100:.0f}%</text>')
    out.append(f'<line x1="{ML}" y1="{MT+ph}" x2="{ML+pw}" y2="{MT+ph}" stroke="#888" stroke-width="1.2"/>')
    out.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+ph}" stroke="#888" stroke-width="1.2"/>')
    out.append(f'<text x="{ML+pw/2:.1f}" y="{H-28}" font-size="14" fill="#333" '
               f'text-anchor="middle">cache hit rate</text>')
    out.append(f'<text transform="translate(28,{MT+ph/2:.1f}) rotate(-90)" font-size="14" '
               f'fill="#333" text-anchor="middle">ms / pass</text>')
    # lines per L
    for i, L in enumerate(Ls):
        pts = sorted([r for r in rows if r["L"] == L], key=lambda r: r["hit"])
        if not pts:
            continue
        color = COLORS[i % len(COLORS)]
        poly = " ".join(f"{X(r['hit']):.1f},{Y(r['ms']):.1f}" for r in pts)
        out.append(f'<polyline points="{poly}" fill="none" stroke="{color}" '
                   f'stroke-width="2.2" stroke-linejoin="round"/>')
        for r in pts:
            x, y = X(r["hit"]), Y(r["ms"])
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
            out.append(f'<text x="{x:.1f}" y="{y-9:.1f}" font-size="9.5" fill="{color}" '
                       f'text-anchor="middle">{fmt_ms(r["ms"])}</text>')
        # legend
        lx, ly = ML + pw + 28, MT + 8 + i * 26
        out.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+26}" y2="{ly}" stroke="{color}" stroke-width="2.6"/>')
        out.append(f'<text x="{lx+34}" y="{ly+4}" font-size="12.5" fill="#333">L={L//1024}K</text>')
    out.append('</svg>')
    with open(path, "w") as fh:
        fh.write("\n".join(out))


def table(rows, path):
    Ls = sorted({r["L"] for r in rows})
    hits = sorted({r["hit"] for r in rows})
    lines = ["# DeepInfer 28-point matrix", "",
             "单位：ms/forward（N 个稳态 forward 总时间 ÷ N；DP8、每 DP 1 batch、KEEP=140、deepgemm、graph on）", "",
             "| L \\ hit | " + " | ".join(f"{h*100:.0f}%" for h in hits) + " |",
             "|---|" + "---|" * len(hits)]
    for L in Ls:
        cells = []
        for h in hits:
            r = next((x for x in rows if x["L"] == L and abs(x["hit"] - h) < 1e-9), None)
            cells.append(f"{r['ms']:.1f}" if r else "—")
        lines.append(f"| {L//1024}K | " + " | ".join(cells) + " |")
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--out", default="")
    ap.add_argument("--title", default="DeepInfer 28-point matrix")
    a = ap.parse_args()
    rows = load(a.tsv)
    if not rows:
        sys.exit(f"没有可用数据：{a.tsv}")
    pref = a.out or os.path.splitext(a.tsv)[0]
    svg(rows, a.title, pref + ".svg")
    table(rows, pref + ".md")
    with open(pref + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "L", "hit", "ms_per_forward", "errors"])
        for r in sorted(rows, key=lambda r: (r["L"], r["hit"])):
            w.writerow([f"({r['L']}_{r['hit']})_r1", r["L"], r["hit"], r["ms"], r["errors"]])
    print(f"rows={len(rows)} -> {pref}.svg / .md / .csv")


if __name__ == "__main__":
    sys.exit(main())

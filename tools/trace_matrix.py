#!/usr/bin/env python3
"""从 **trace 口径** 的 `forward_times.csv` 出 L×hit 矩阵 + `ms vs T_pass` 拟合。

**为什么不用 `results.tsv`**：`points_runner` 的墙钟把 **msprof stop** 也算进去
（最后一次 forward 带 `stop_profiler=True`，整段停 profiler 的秒数被摊到 iters 上），
小 T 时虚高 1.4–25×（1k 点：墙钟 3067 ms vs trace 143 ms ≈ 21×）。
性能口径只认 trace 里的**设备时间线**（相邻 forward 起点差的中位，剔除首个）。

输入：`tools/trace_report.py ... --export` 产出的 `<out>/forward_times.csv`
      （列：point,forwards,steady_ms_per_forward,span_ms,busy_ms,idle_pct,note；
       point 形如 `cp2dp4ep8_L256k_hit0.6_batch1_replayon_prefill`（过渡格式 `_b1_r1` 也认））

输出（默认写到 CSV 同目录）：
  TRACE_MATRIX.md    L × hit 的 ms/forward 矩阵（含 idle%、点数）
  TRACE_POINTS.csv   长表：L / hit / T_pass / steady / busy / idle / forwards
  终端：矩阵 + 拟合 ms ≈ a + b·T_pass + 各 L 的高命中地板值

用法：
    python3 tools/trace_matrix.py <forward_times.csv 或包含它的目录>
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics
import sys

# 标准格式（2026-10-29 用户口径）：cp<CP>dp<DP>ep<EP>_L<Lk>_hit<h>_batch<b>_replay<on|off>_prefill
# 过渡格式（同日短暂用过）：          …_b<b>_r<0|1>_prefill
_RX = re.compile(
    r"^cp(?P<cp>\d+)dp(?P<dp>\d+)ep(?P<ep>\d+)_L(?P<L>[0-9.]+[km]?)_hit(?P<h>[0-9.]+)_"
    r"(?:b(?P<b>\d+)_r(?P<r>[01])|batch(?P<b2>\d+)_replay(?P<r2>on|off))_prefill$")
_MUL = {"": 1, "k": 1 << 10, "m": 1 << 20}


def parse_L(s: str) -> int | None:
    m = re.fullmatch(r"([0-9.]+)([km]?)", s)
    if not m:
        return None
    return int(float(m.group(1)) * _MUL[m.group(2)])


def fmt_L(L: int) -> str:
    for unit, div in (("m", 1 << 20), ("k", 1 << 10)):
        if L >= div and L % div == 0:
            return f"{L // div}{unit}"
    return str(L)


def t_pass(L: int, h: float, keep: int = 140) -> int:
    """与点位集同一口径：h=0 → 全长；h>0 → round((1-h)L) + KEEP(140)。"""
    return L if h <= 0 else int(round((1.0 - h) * L)) + keep


def load(path: str):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = _RX.match((r.get("point") or "").strip())
            if not m:
                continue
            L = parse_L(m.group("L"))
            if L is None:
                continue
            h = float(m.group("h"))

            def f(k):
                try:
                    return float(r.get(k) or "nan")
                except ValueError:
                    return float("nan")
            _b = m.group("b") or m.group("b2") or "1"
            _r = m.group("r")
            _rep = (_r == "1") if _r is not None else (m.group("r2") == "on")
            rows.append(dict(
                cdp=m.group("cp"), L=L, h=h, batch=int(_b),
                replay=_rep, T=t_pass(L, h),
                steady=f("steady_ms_per_forward"), span=f("span_ms"),
                busy=f("busy_ms"), idle=f("idle_pct"), n=f("forwards"),
                note=(r.get("note") or "").strip()))
    return rows


def fit(rows):
    """最小二乘 ms = a + b·T（只用 h>0 且 steady 有效的点）。"""
    pts = [(r["T"], r["steady"]) for r in rows
           if r["h"] > 0 and r["steady"] == r["steady"]]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den == 0:
        return None
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / den
    a = my - b * mx
    ss_tot = sum((p[1] - my) ** 2 for p in pts)
    ss_res = sum((p[1] - (a + b * p[0])) ** 2 for p in pts)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(a=a, b=b, r2=r2, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="forward_times.csv，或包含它的目录")
    ap.add_argument("-o", "--out", default="")
    a = ap.parse_args()

    p = a.path
    if os.path.isdir(p):
        cand = [os.path.join(p, "forward_times.csv")] + \
            sorted(glob.glob(os.path.join(p, "**", "forward_times.csv"), recursive=True))
        p = next((c for c in cand if os.path.isfile(c)), "")
    if not p or not os.path.isfile(p):
        sys.exit(f"✗ 找不到 forward_times.csv（先跑 tools/trace_report.py --export）：{a.path}")
    rows = load(p)
    if not rows:
        sys.exit(f"✗ {p} 里没有可解析的点位名")
    outdir = a.out or os.path.dirname(os.path.abspath(p))

    Ls = sorted({r["L"] for r in rows})
    hs = sorted({r["h"] for r in rows})
    cell = {(r["L"], r["h"]): r for r in rows}
    bad = [r for r in rows if r["steady"] != r["steady"] or r["note"]]

    lines = [f"# trace 口径点位矩阵（{os.path.basename(os.path.dirname(p)) or outdir}）", "",
             f"- 来源：`{p}`（{len(rows)} 个点位）",
             "- 口径：trace 设备时间线，**相邻 forward 起点差的中位、剔除首个**；"
             "不含 msprof stop（墙钟 `results.tsv` 不能用于性能）", ""]
    lines += ["| L \\ hit | " + " | ".join(f"{h:g}" for h in hs) + " |",
              "|" + "---|" * (len(hs) + 1)]
    for L in Ls:
        vals = []
        for h in hs:
            r = cell.get((L, h))
            vals.append("—" if r is None or r["steady"] != r["steady"]
                        else f"{r['steady']:.0f}")
        lines.append(f"| {fmt_L(L)} | " + " | ".join(vals) + " |")

    ft = fit(rows)
    lines += ["", "## 拟合（h>0 的点）", ""]
    if ft:
        lines.append(f"- `ms/forward ≈ {ft['a']:.1f} + {ft['b'] * 1000:.3f} µs × T_pass`"
                     f"（R² = {ft['r2']:.3f}，n = {ft['n']}）")
    else:
        lines.append("- 点数不足，未拟合")
    lines += ["", "## 各 L 的高命中地板（h≥0.9，取中位）", "",
              "| L | 地板 ms/forward | h≥0.9 点数 |", "|---|---:|---:|"]
    for L in Ls:
        v = [r["steady"] for r in rows if r["L"] == L and r["h"] >= 0.9
             and r["steady"] == r["steady"]]
        if v:
            lines.append(f"| {fmt_L(L)} | {statistics.median(v):.0f} | {len(v)} |")

    md = os.path.join(outdir, "TRACE_MATRIX.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    cs = os.path.join(outdir, "TRACE_POINTS.csv")
    with open(cs, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["L", "L_text", "hit", "T_pass", "steady_ms", "busy_ms",
                    "idle_pct", "forwards", "note"])
        for r in sorted(rows, key=lambda x: (x["L"], x["h"])):
            w.writerow([r["L"], fmt_L(r["L"]), f"{r['h']:g}", r["T"],
                        f"{r['steady']:.1f}", f"{r['busy']:.1f}",
                        f"{r['idle']:.1f}", f"{r['n']:.0f}", r["note"]])

    print(f"点位 {len(rows)} 个（{len(rows) - len(bad)} 个有效）· L {len(Ls)} 档 · hit {len(hs)} 档")
    print("\n".join(lines[5:]))
    print(f"\n→ {md}\n→ {cs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

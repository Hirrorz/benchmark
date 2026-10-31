#!/usr/bin/env python3
"""把 `--points` 的 `results.tsv` 摊成 **L × hit 矩阵**（每点一个数），并写 Markdown/CSV。

口径（PERF_RULES §0）：`results.tsv` 由 `--points` 的 **report rank**（默认 rank1，
即第一个**未被 msprof 采集**的 rank）写入 ⇒ 这里的数**不含 profiler 开销**；
trace 口径（`tools/trace_report.py`）只用于看结构/空泡，两者不能混比。

点位名（tag）形如 `(262144_0.6)_r1`，直接解析出 L 与 hit，不依赖 points.json。

用法：
    python3 tools/points_results_table.py <results.tsv 或 目录> [-o OUTDIR]
                                         [--field steady|ms|errors]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

_TAG_RX = re.compile(r"^\((\d+)_([\d.]+)\)_r([01])$")
# 2026-10-29 起采集目录/结果 tag 直接用标准扁平名
_FLAT_RX = re.compile(
    r"^cp\d+dp\d+ep\d+_L([0-9.]+[km]?)_hit([0-9.]+)"
    r"(?:_batch\d+_replay(?P<ron>on|off)|_b\d+_r(?P<rnum>[01]))_prefill$")
_MUL = {"": 1, "k": 1 << 10, "m": 1 << 20}


def _parse_flat(tag):
    """`cp2dp4ep8_L256k_hit0.97_b1_r1_prefill` → (L, hit, "0"/"1")；多批（`+`）只取第一段。"""
    m = _FLAT_RX.match(tag)
    if not m:
        return None
    lm = re.fullmatch(r"([0-9.]+)([km]?)", m.group(1))
    if not lm:
        return None
    _r = m.group("rnum")
    _rep = (_r == "1") if _r is not None else (m.group("ron") == "on")
    return (int(float(lm.group(1)) * _MUL[lm.group(2)]),
            float(m.group(2).split("+")[0]), "1" if _rep else "0")


def load(path: str):
    rows = []
    with open(path, encoding="utf-8") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for r in rd:
            tag = (r.get("tag") or "").strip()
            m = _TAG_RX.match(tag)
            if m:
                L0, h0, r0 = int(m.group(1)), float(m.group(2)), m.group(3)
            else:
                pf = _parse_flat(tag)
                if not pf:
                    continue
                L0, h0, r0 = pf
            rows.append(dict(L=L0, h=h0,
                             replay=r0 == "1",
                             steady=float(r.get("steady_s") or "nan"),
                             ms=float(r.get("ms_per_forward") or "nan"),
                             errors=int(float(r.get("errors") or 0)),
                             ts=r.get("ts") or ""))
    return rows


def fmt_L(L: int) -> str:
    for unit, div in (("m", 1 << 20), ("k", 1 << 10)):
        if L >= div and L % div == 0:
            return f"{L // div}{unit}"
    return str(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="results.tsv，或包含它的点位根目录")
    ap.add_argument("-o", "--out", default="", help="输出目录（默认与 results.tsv 同目录）")
    ap.add_argument("--field", default="ms", choices=("steady", "ms", "errors"),
                    help="矩阵里放哪个字段（默认 ms = ms/forward）")
    a = ap.parse_args()

    p = a.path
    if os.path.isdir(p):
        p = os.path.join(p, "results.tsv")
    if not os.path.isfile(p):
        sys.exit(f"✗ 找不到 {p}")
    rows = load(p)
    if not rows:
        sys.exit(f"✗ {p} 里没有可解析的点位（tag 形如 `(262144_0.6)_r1`）")

    outdir = a.out or os.path.dirname(os.path.abspath(p))
    Ls = sorted({r["L"] for r in rows})
    hs = sorted({r["h"] for r in rows})
    cell = {}
    for r in rows:
        cell[(r["L"], r["h"])] = r
    unit = {"steady": "s", "ms": "ms", "errors": "err"}[a.field]

    # ── Markdown 矩阵 ──
    lines = [f"# 点位矩阵（{os.path.basename(os.path.dirname(p)) or outdir}）", "",
             f"- 来源：`{p}`（{len(rows)} 个点位；口径 = report rank，**不含 msprof 开销**）",
             f"- 单元格 = `{a.field}`（{unit}）；`—` = 该点位没跑/缺行", ""]
    head = "| L \\ hit | " + " | ".join(f"{h:g}" for h in hs) + " |"
    lines += [head, "|" + "---|" * (len(hs) + 1)]
    for L in Ls:
        vals = []
        for h in hs:
            r = cell.get((L, h))
            if r is None:
                vals.append("—")
            elif a.field == "errors":
                vals.append(str(r["errors"]))
            else:
                vals.append(f"{r[a.field]:.1f}" if r[a.field] == r[a.field] else "—")
        lines.append(f"| {fmt_L(L)} | " + " | ".join(vals) + " |")
    md = os.path.join(outdir, f"RESULTS_MATRIX_{a.field}.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ── 长表 CSV（便于画图/二次统计）──
    cs = os.path.join(outdir, "RESULTS_POINTS.csv")
    with open(cs, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["L", "L_text", "hit", "replay", "steady_s", "ms_per_forward",
                    "errors", "ts"])
        for r in sorted(rows, key=lambda x: (x["L"], x["h"])):
            w.writerow([r["L"], fmt_L(r["L"]), f"{r['h']:g}", int(r["replay"]),
                        f"{r['steady']:.4f}", f"{r['ms']:.2f}", r["errors"], r["ts"]])

    # ── 终端摘要 ──
    print(f"点位 {len(rows)} 个 · L {len(Ls)} 档 · hit {len(hs)} 档 · "
          f"有算子错误的点位 {sum(1 for r in rows if r['errors'])} 个")
    print("\n".join(lines[4:]))
    print(f"\n→ {md}\n→ {cs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

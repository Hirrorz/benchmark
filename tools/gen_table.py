#!/usr/bin/env python3.11
"""把三 CP 的网格产物汇总成最终表格（**按 points_*.json 过滤**掉新约束下已跳过的点位）。

输出列：seq_len  hit_rate  strategy  per_die_tokens  forward_40_layers(ms)  TPS/card
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

TAG_RE = re.compile(r"^cp(?P<cp>\d+)dp(?P<dp>\d+)ep(?P<ep>\d+)_L(?P<L>\d+)_hit(?P<h>[\d.]+)")
COLS = ["seq_len", "hit_rate", "strategy", "per_die_tokens",
        "forward_40_layers(ms)", "TPS/card"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()

    rows, n_drop_nojson, n_drop_bad = [], 0, 0
    for tsv in sorted(glob.glob(os.path.join(args.root, "noprof_cp*", "results.tsv"))):
        cpm = re.search(r"noprof_cp(\d+)", tsv)
        if not cpm:
            continue
        cp = int(cpm.group(1))
        pj = os.path.join(args.root, f"points_cp{cp}.json")
        allowed = None
        if os.path.exists(pj):
            allowed = {(int(x["L"]), float(x["hit"])) for x in json.load(open(pj))}
        with open(tsv, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                tag = (r.get("tag") or "").strip()
                m = TAG_RE.match(tag)
                if not m:
                    continue
                L, h = int(m["L"]), float(m["h"])
                if allowed is not None and (L, h) not in allowed:
                    n_drop_nojson += 1
                    continue
                try:
                    steady = float(r.get("steady_s") or 0)
                except Exception:
                    continue
                if steady <= 0:
                    continue
                err = int(r.get("errors") or 0)
                if err > 0:
                    n_drop_bad += 0            # 保留但标注
                dp, ep = int(m["dp"]), int(m["ep"])
                rows.append({
                    "seq_len": L,
                    "hit_rate": h,
                    "strategy": f"cp{cp}_dp{dp}_ep{ep}",
                    "per_die_tokens": int(round((1.0 - h) * L)),
                    "forward_40_layers(ms)": round(steady * 1000.0, 2),
                    "TPS/card": round(L / steady, 0),
                    "err": err,
                })

    if not rows:
        sys.exit("✗ 没有数据")
    seen = {}
    for r in rows:
        seen[(r["strategy"], r["seq_len"], r["hit_rate"])] = r
    rows = sorted(seen.values(), key=lambda x: (x["strategy"], x["seq_len"], x["hit_rate"]))

    out = args.out or os.path.join(args.root, "grid_final.tsv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS + ["err"], delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n_err = sum(1 for r in rows if r["err"] > 0)
    print(f"  ✓ {len(rows)} 行 → {out}")
    print(f"    过滤掉 {n_drop_nojson} 条（新约束下已跳过）")
    print(f"    其中 err>0 的 {n_err} 行（DeepSelect 偶发，时延可能偏低）")
    by = {}
    for r in rows:
        by[r["strategy"]] = by.get(r["strategy"], 0) + 1
    for k in sorted(by):
        print(f"    {k}: {by[k]} 行")

    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write("| " + " | ".join(COLS + ["err"]) + " |\n")
            f.write("|" + "---|" * (len(COLS) + 1) + "\n")
            for r in rows:
                f.write("| " + " | ".join(str(r[c]) for c in COLS + ["err"]) + " |\n")
        print(f"  ✓ Markdown → {args.md}")


if __name__ == "__main__":
    main()

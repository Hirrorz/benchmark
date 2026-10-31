"""逐算子**耗时对齐表**：mini 的 trace × 官方同点 op_summary.csv（时间口径；不做数值对拍）。

mini 是逐算子参考实现，端到端延迟跟官方引擎没法直接比；能比、也值得比的是
**每个算子的调用次数与单次耗时**（同一点、同一个 T，形状一致）。

用法：
  python3.11 tools/op_time_table.py \
      --trace-dir <mini 的 prefill 目录，含 8 个 *_ascend_pt> \
      --ref /mnt/xds/sfs/l00972766/ds41_light/res/20261010/hit97_prof_unified/L64k/op_summary.csv \
      --out <输出 csv>

口径：
  · mini：8 个 rank 的 `ASCEND_PROFILER_OUTPUT/op_statistic.csv` 逐算子求和（Count / Total Time(us)），
    单次耗时 = Total/Count（trace 里已含 N 个 pass，不除 passes —— 与官方 op_summary 同口径，
    官方那份是 8 rank × 4 pass 的合计）。
  · 官方：`# ...` 注释行跳过；带 mangled 名的行按 align_ops.ALIAS 的正则归并到短名。
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from align_ops import ALIAS                                    # noqa: E402


# 名字对不上的少数几处（mini 走 tilelang/自研，官方走厂商算子）：按**语义**手动归并
# ⚠️ 2026-10-15 撤掉两条错归并：以前把 `EngramFetch → engram_gate_fwd`、`EngramFetchWait →
#    engram_fused_weight` 归成一行，结论是错的 —— 官方 op_summary 里这**四个算子同时存在**
#    （64k：EngramFetch 64 次 / EngramFetchWait 64 次 / engram_gate_fwd 64 / engram_fused_weight 64），
#    归并后那张表把官方两类合并计数（128）拿去比 mini 一类（64），还比出 0.0× 的假差距。
#    mini 现在**本来就不发** EngramFetch/EngramFetchWait（表常驻设备 → 没有"取表"这个动作，
#    exec_engram.py:186-188 直接 return），就该显示成 mini_count=0 的缺口，而不是被抹平。
EXTRA_ALIAS = [
    ("engram_hash", r"engram_hash"),
]


def short(name):
    for s, rx in ALIAS + EXTRA_ALIAS:
        if re.search(rx, name):
            return s
    # aclnn 包装名（`aclnnXxx_Yyy`）与带后缀的变体 → 退化到"包含式"匹配
    for s, rx in ALIAS + EXTRA_ALIAS:
        if re.search(rx.strip("^$"), name):
            return s
    # 没归并到的：去掉 mangled 前缀，截断到 60 字符，仍然可读
    n = re.sub(r"^_Z\d+", "", name)
    n = re.sub(r"^_ZN?9deep_gemm8", "deep_gemm.", n)
    return n[:60]


def mini_totals(trace_dir):
    """→ {短名: [count, total_us]}（8 rank 求和）"""
    agg = collections.defaultdict(lambda: [0, 0.0])
    files = glob.glob(os.path.join(trace_dir, "*_ascend_pt", "ASCEND_PROFILER_OUTPUT",
                                   "op_statistic.csv"))
    if not files:
        raise SystemExit(f"✗ {trace_dir} 下没有 *_ascend_pt/ASCEND_PROFILER_OUTPUT/op_statistic.csv")
    for f in files:
        for r in csv.DictReader(open(f, errors="ignore")):
            if not r.get("OP Type"):
                continue
            c = int(float(r.get("Count") or 0))
            t = float(r.get("Total Time(us)") or 0.0)
            cell = agg[short(r["OP Type"])]
            cell[0] += c
            cell[1] += t
    print(f"  mini：{len(files)} 个 rank 的 op_statistic.csv 已汇总（{sum(v[0] for v in agg.values())} 次调用）")
    return agg


def ref_totals(path):
    """→ {短名: [count, total_us]}（官方 op_summary.csv：Total 列是 ms）"""
    agg = collections.defaultdict(lambda: [0, 0.0])
    for line in open(path, errors="ignore"):
        if line.startswith("#") or line.startswith("OP Type"):
            continue
        parts = line.rstrip("\n").split(",")
        if len(parts) < 4:
            continue
        try:
            c, t_ms = int(float(parts[1])), float(parts[2])
        except ValueError:
            continue
        cell = agg[short(parts[0])]
        cell[0] += c
        cell[1] += t_ms * 1000.0
    print(f"  官方：{os.path.basename(path)} 已读（{sum(v[0] for v in agg.values())} 次调用）")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True, help="mini 的 --out 目录（含 *_ascend_pt）")
    ap.add_argument("--ref", default="", help="官方 op_summary.csv（给了才出对比列）")
    ap.add_argument("--out", default="", help="输出 csv（默认打印到屏幕）")
    ap.add_argument("--top", type=int, default=0, help="只打印 mini 总耗时前 N 项")
    a = ap.parse_args()

    mini = mini_totals(a.trace_dir)
    ref = ref_totals(a.ref) if a.ref else {}

    keys = sorted(set(mini) | set(ref), key=lambda k: -mini.get(k, [0, 0.0])[1])
    rows = []
    for k in keys:
        mc, mt = mini.get(k, [0, 0.0])
        rc, rt = ref.get(k, [0, 0.0])
        rows.append(dict(op=k,
                         mini_count=mc, mini_total_ms=round(mt / 1000, 3),
                         mini_avg_us=round(mt / mc, 2) if mc else "",
                         ref_count=rc, ref_total_ms=round(rt / 1000, 3),
                         ref_avg_us=round(rt / rc, 2) if rc else "",
                         avg_ratio=round((mt / mc) / (rt / rc), 2) if (mc and rc) else ""))
    if a.top:
        rows = rows[:a.top]

    hdr = ["op", "mini_count", "mini_total_ms", "mini_avg_us", "ref_count", "ref_total_ms",
           "ref_avg_us", "avg_ratio"]
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            w.writeheader()
            w.writerows(rows)
        print(f"  → {a.out}（{len(rows)} 个算子）")
    print(f"\n{'算子':<34}{'mini次数':>9}{'mini ms':>10}{'mini µs/次':>12}"
          f"{'官方次数':>9}{'官方 ms':>10}{'官方 µs/次':>12}{'倍数':>7}")
    for r in rows:
        print(f"{r['op']:<34}{r['mini_count']:>9}{r['mini_total_ms']:>10}"
              f"{str(r['mini_avg_us']):>12}{r['ref_count']:>9}{r['ref_total_ms']:>10}"
              f"{str(r['ref_avg_us']):>12}{str(r['avg_ratio']):>7}")


if __name__ == "__main__":
    main()

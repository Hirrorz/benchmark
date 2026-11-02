#!/usr/bin/env python3
"""B5 收益报告（**已按 2026-11-01 实测修正口径**）。

    python3.11 tools/halo_report.py            # 默认矩阵
    python3.11 tools/halo_report.py --Ls 8192,16384 --cps 2,4,8

★ 撤回说明（作业 249/251）：早先版本把**全序列 L** 当作 gather 的分母，得出
"replay 段选择性 gather 可省 73.9%–99.6%"。实测证明该口径错误：

  · 层 20（裁剪前）本层**全部 T 行**参与 attention ⇒ 需要的 KV 就是全部 T 行；
  · 层 21+（compact 段）`T ≈ keep ≈ 140` ⇒ 全量 gather 本来就只有 ~140 行。

所以真实要 gather 的量是**本层 T**，不是 L。本脚本现在同时打印两种口径，
并把"按本层口径"的结论明确写出来（当前 bench 点位：**无可省量**）。
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("cp", "layout", "core", "meta", "ops", "trace"):
    sys.path.insert(0, os.path.join(ROOT, _p))

import cp_plan            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="B5：halo 清单与 gather 收益（修正口径）")
    ap.add_argument("--Ls", default="1024,4236,8192,16384", help="序列长")
    ap.add_argument("--cps", default="2,4,8", help="cp 尺寸")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--keep", type=int, default=140)
    ap.add_argument("--Ts", default="140,2118,4236",
                    help="本层实际行数 T（真实 gather 量的分母）")
    a = ap.parse_args()

    Ls = [int(x) for x in a.Ls.split(",") if x.strip()]
    cps = [int(x) for x in a.cps.split(",") if x.strip()]
    Ts = [int(x) for x in a.Ts.split(",") if x.strip()]

    print(f"[halo] window={a.window} keep={a.keep}   （★ 口径已修正：分母=本层 T，不是 L）")
    print("\n== ① 尾部窗口需要多少行（与 T 无关，恒为 keep+window-1） ==")
    print(f"{'cp':>3} {'L':>7} {'blocks(need/total)':>18} {'rows_need':>10}")
    for cp in cps:
        for L in Ls:
            v = cp_plan.build_plan([L], cp, 0)
            s = cp_plan.gather_stats(v, window=a.window, keep=a.keep, mode="replay")
            print(f"{cp:>3} {L:>7} {s['blocks_needed']:>8}/{s['blocks_total']:<9} "
                  f"{s['rows_needed']:>10}")

    print("\n== ② 真实收益：按**本层 query 范围**分别算（分母 = 本层 T） ==")
    print("   · 层 20（裁剪前）：本层全部 T 行参与 attention ⇒ need = T ⇒ 0%")
    print("   · 层 21+（compact）：query 只覆盖尾部 keep 行 ⇒ need = keep+window−1")
    print(f"{'cp':>3} {'T':>7} {'query':>10} {'rows_need':>10} {'省%':>6}")
    bad = 0
    for cp in cps:
        for T in Ts:
            v = cp_plan.build_plan([T], cp, 0)
            s = cp_plan.gather_stats(v, window=a.window, keep=a.keep, mode="replay")
            tail_need = min(int(s["rows_needed"]), T)
            for scope in ("全行", "仅尾部"):
                need = T if scope == "全行" else tail_need
                saved = 100.0 * max(0, T - need) / max(1, T)
                print(f"{cp:>3} {T:>7} {scope:>10} {need:>10} {saved:>5.1f}%")
                if need > T:
                    bad += 1

    print("\n结论：层 20 必须全量（0%）；层 21+ 虽是「仅尾部」，但那里 `T≈keep≈140`，"
          "全量本来就只有 140 行 ⇒ **当前 bench 点位没有可省的 gather 量**。"
          "只有 query 只覆盖尾部**且** T 远大于 keep（长序列 chunked prefill / decode）时才有收益。")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
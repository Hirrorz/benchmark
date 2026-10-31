#!/usr/bin/env python3
"""看 trace 里到底有哪些算子、各多少次（与我们的模型对照）。

用法：python3 tools/trace_coverage.py <trace 目录或 kernel_details.csv> [--top 30]
"""
import collections
import csv
import os
import sys


def main():
    path = sys.argv[1]
    top = 30
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])
    if os.path.isdir(path):
        path = os.path.join(path, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
    rows = list(csv.DictReader(open(path, errors="ignore")))
    c = collections.Counter(r["Type"] for r in rows)
    print(f"{path}\n总内核调用 {sum(c.values())}，{len(c)} 种：")
    for k, v in c.most_common(top):
        print(f"  {v:>7}  {k[:70]}")
    if len(c) > top:
        rest = sum(v for k, v in c.most_common()[top:])
        print(f"  {rest:>7}  （其余 {len(c) - top} 种合计）")


if __name__ == "__main__":
    main()

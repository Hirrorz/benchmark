#!/usr/bin/env python3
"""从 kernel_details.csv 解出某个算子的**层内形状序列** —— M1 校准的关键手段。

原理：一个前向里 40 层是按序执行的，所以把同一算子的所有行按 Start Time 排序，
它的 Input Shapes 序列就暴露了"哪些层多跑了一次、多跑的那个是什么形状"。
配合"按形状分组计数"（--group），就能把 40 层拆成 基础次数 + 按层类型的增量。

用法：
  python3 tools/shape_seq.py <kernel_details.csv> --op KvCompressEpilogV2
  python3 tools/shape_seq.py <kernel_details.csv> --op QuantBatchMatmulV3 --key Type --step 0
  python3 tools/shape_seq.py <kernel_details.csv> --op DynamicMxQuant --group

--key   : 按 Name 还是 Type 匹配（默认自动：Type 命中就用 Type）
--step  : 只看某个 Step（默认取最小 Step，即第一个 pass）
--group : 只出"按形状分组计数"（不排序），适合先看有几种形状
注意：多 stream 并行时 Start Time 顺序只是**近似**层序 —— 次数/形状是硬证据，顺序不是。
"""
import argparse
import collections
import csv
import sys


def load(path):
    return list(csv.DictReader(open(path, errors="ignore")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--op", required=True)
    ap.add_argument("--key", default="auto", choices=["auto", "Name", "Type"])
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--passes", type=int, default=1, help="归一化到单 pass（trace 里有几个前向）")
    ap.add_argument("--group", action="store_true")
    a = ap.parse_args()

    rows = load(a.csv)
    key = a.key
    if key == "auto":
        key = "Type" if any(r["Type"] == a.op for r in rows) else "Name"
    hits = [r for r in rows if r[key] == a.op]
    if not hits:
        sys.exit(f"没找到 {a.op}（key={key}）—— 用 --key Name 试试？")

    def sig(r):
        s = r["Input Shapes"].strip()
        return None if s in ("", "-") else s

    shapes = []
    for r in hits:
        s = sig(r)
        if s not in shapes:
            shapes.append(s)
    mp = {s: chr(65 + i) for i, s in enumerate(shapes)}

    print(f"=== {a.op}（key={key}）{len(hits)} 行 = {len(hits)/a.passes:.1f}/pass，{len(shapes)} 种形状 ===")
    if a.group:
        c = collections.Counter(sig(r) for r in hits)
        for s, n in c.most_common():
            print(f"  {n/a.passes:>7.1f}/pass  {str(s)[:110]}")
        return

    h = hits if a.step is None else [r for r in hits if int(r["Step Id"]) == a.step]
    if a.step is None:
        sid = min(int(r["Step Id"]) for r in h)
        h = [r for r in h if int(r["Step Id"]) == sid]
    else:
        sid = a.step
    h.sort(key=lambda r: float(r["Start Time(us)"]))
    seq = "".join(mp[sig(r)] for r in h)
    print(f"（Step {sid}，{len(h)} 个，按 Start Time 排序）")
    for i in range(0, len(seq), 100):
        print("   ", seq[i:i + 100])
    for s in shapes:
        print(f"   {mp[s]} = {str(s)[:110]}")


if __name__ == "__main__":
    main()

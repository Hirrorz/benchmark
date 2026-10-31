#!/usr/bin/env python3
"""scan_first_nan.py — 从 `DS41M_DUMP_OPS` 的逐算子快照里，按**执行序**找出第一个含非有限值的。

原理（2026-10-16 首次用、2026-10-17 固化）：runner 的 `DS41M_DUMP_OPS=<dir>` 会在**每个算子
执行前**落一份 st/shared 快照（文件名含 rank/layer/op/序号）。文件按 **mtime 升序 = 执行序** ⇒
**第一个含 NaN/Inf 的快照**的**前一个算子**就是产出 NaN 的元凶 ✓（快照是"执行前"的 ✓）。

用法：
    python3.11 tools/scan_first_nan.py <dump_dir> [--rank 7] [--max 4000] [--keys]

输出：前若干个快照的 mtime/名字/非有限统计；第一个坏的 → 打印其**前一个**快照名（= 元凶算子 ✓）。
"""
import argparse
import os
import sys

import torch


def walk_nonfinite(obj, prefix=""):
    """递归统计张量里的非有限元素，返回 [(键路径, 非有限数, 形状)]。"""
    bad = []
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            n = int((~torch.isfinite(obj)).sum())
            if n:
                bad.append((prefix or "<tensor>", n, tuple(obj.shape)))
        return bad
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad += walk_nonfinite(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += walk_nonfinite(v, f"{prefix}[{i}]")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--rank", default=None, help="只看该 rank 的文件（文件名里含 rankN）")
    ap.add_argument("--max", type=int, default=3000, help="最多扫多少个快照")
    ap.add_argument("--head", type=int, default=6, help="开头打印几个快照")
    args = ap.parse_args()

    files = sorted(
        (os.path.join(args.dump, f) for f in os.listdir(args.dump) if f.endswith(".pt")),
        key=os.path.getmtime)
    if args.rank is not None:
        files = [f for f in files if f"rank{args.rank}" in os.path.basename(f)]
    if not files:
        print(f"✗ {args.dump} 里没有 .pt（或 rank 过滤后为空）")
        return 2
    print(f"快照 {len(files)} 份（按 mtime = 执行序 ✓）；rank 过滤 = {args.rank}")
    for f in files[:args.head]:
        print(f"    {os.path.getmtime(f):.3f}  {os.path.basename(f)}")

    first_bad = None
    n_scan = min(len(files), args.max)
    for i, f in enumerate(files[:n_scan]):
        try:
            obj = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [warn] 读不了 {os.path.basename(f)}：{type(exc).__name__}: {exc}")
            continue
        bad = walk_nonfinite(obj)
        if bad:
            first_bad = (i, f, bad)
            break
    if first_bad is None:
        print(f"✓ 扫了 {n_scan} 份，**全有限**（本跑没崩，或崩在更晚的快照之后 ✗）")
        return 1
    i, f, bad = first_bad
    print(f"\n✗✗ 第一个含非有限的快照（第 {i} 份 / mtime={os.path.getmtime(f):.3f}）：")
    print(f"     {os.path.basename(f)}")
    for k, n, shp in bad[:12]:
        print(f"       · {k}: {n} 个非有限 | shape={shp}")
    if i > 0:
        print("\n★ 元凶 = 它的**前一个算子**（快照是「执行前」落的 ✓）：")
        print(f"     第 {i-1} 份 = {os.path.basename(files[i-1])}")
        for j in range(max(0, i - 3), i):
            print(f"     （执行序 {j}）{os.path.basename(files[j])}")
    else:
        print("\n（它已经是第一份 ⇒ 元凶在更早：输入/权重加载阶段 ✗）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

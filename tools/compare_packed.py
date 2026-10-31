#!/usr/bin/env python3
"""对比 packed 一次前向 与 两条独立单序列前向 的逐层 dump（--collect-layers 产物）。

用法：
  python3 tools/compare_packed.py <packed.pt> <seqA.pt> <seqB.pt>

判据：packed 的 [0:n] 应等于 seqA、[n:2n] 应等于 seqB（n = packed 行数 / 2）。
输出每层的 max|Δ| 与逐位相等比例；非 0 即说明 packed 的序列隔离/块表/槽位有问题。
"""
from __future__ import annotations

import sys

import torch


def load(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d["layers"] if isinstance(d, dict) and "layers" in d else d


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    pk, da, db = (load(p) for p in sys.argv[1:4])
    keys = sorted(set(pk) & set(da) & set(db), key=lambda x: int(x))
    if not keys:
        print("✗ 三份 dump 没有共同层号")
        return 2
    worst = 0.0
    for k in keys:
        p = pk[k]
        if p.dim() != 2:
            print(f"layer {k}: 形状异常 {tuple(p.shape)}")
            continue
        n = p.shape[0] // 2
        a, b = da[k], db[k]
        if a.shape != (n, p.shape[1]) or b.shape != (n, p.shape[1]):
            print(f"layer {k}: 形状不匹配 packed={tuple(p.shape)} A={tuple(a.shape)} B={tuple(b.shape)}")
            continue
        ea = (p[:n] - a).abs().max().item()
        eb = (p[n:] - b).abs().max().item()
        eqa = (p[:n] == a).float().mean().item()
        eqb = (p[n:] == b).float().mean().item()
        worst = max(worst, ea, eb)
        print(f"layer {k:>2}: max|Δ| A={ea:.3e} (逐位 {eqa*100:.3f}%)  "
              f"B={eb:.3e} (逐位 {eqb*100:.3f}%)")
    print(f"\n最差 max|Δ| = {worst:.3e}")
    return 0 if worst == 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())

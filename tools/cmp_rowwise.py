#!/usr/bin/env python3
"""按 **token 行** 定位分歧：mini collect vs 官方 collect（层边界，全 T 长度）。

层界面对拍（compare_layers.py）只给“层 1 相对误差 9e-3”，但 max|Δ|=7e-2、mean|Δ|=4.6e-4
——这种"少数行大错"通常是**路由翻转**。本工具报：
  · 每层有多少行不同、第一处不同的行号、第一处 >1e-2 的行号；
  · 最差 N 行（行号 + max/mean |Δ| + 参考幅值）；
  · 不同行是**连续区间**还是**散点**（决定逐算子 dump 的行窗口）。

用法：
  python3.11 tools/cmp_rowwise.py --ref <官方 collect_rank0.pt> \
      --ours <mini collect_rank0.pt> --layers 1,2,3 [--top 12]
"""
import argparse
import sys

import torch


def load(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--layers", default="1,2,3")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    R = load(a.ref)
    O = load(a.ours)
    rl, ol = R.get("layers") or R, O.get("layers") or O
    print(f"ref = {a.ref}\nours = {a.ours}\n层数 ref={len(rl)} ours={len(ol)}")

    for L in [int(s) for s in a.layers.split(",") if s.strip()]:
        r, o = rl.get(L), ol.get(L)
        if r is None or o is None:
            print(f"\n层 {L}: 缺数据（ref={r is not None} ours={o is not None}）")
            continue
        r = r.float()
        o = o.float()
        if r.shape != o.shape:
            print(f"\n层 {L}: 形状不同 ref{tuple(r.shape)} ours{tuple(o.shape)}")
            continue
        d = (r - o).abs()                       # [T,H]
        rowmax = d.max(dim=1).values             # [T]
        rowmean = d.mean(dim=1)
        T = d.shape[0]
        n_any = int((rowmax > 0).sum())
        n_1e3 = int((rowmax > 1e-3).sum())
        n_1e2 = int((rowmax > 1e-2).sum())
        bad = (rowmax > 1e-3).nonzero(as_tuple=False).flatten().tolist()
        print(f"\n===== 层 {L}：形状 {tuple(r.shape)} =====")
        print(f"  不同(>0) 行={n_any}/{T}   >1e-3 行={n_1e3}   >1e-2 行={n_1e2}"
              f"   nan ref={int(torch.isnan(r).sum())} ours={int(torch.isnan(o).sum())}")
        if n_any == 0:
            print("  ✅ 全 T 逐位相同")
            continue
        first0 = int((rowmax > 0).nonzero()[0])
        print(f"  第一处不同: 行 {first0}（max|Δ|={rowmax[first0]:.3e}）")
        if bad:
            first = bad[0]
            # 不同行是否连续？
            gaps = [b - a2 for a2, b in zip(bad, bad[1:]) if b - a2 > 1]
            print(f"  第一处 >1e-3: 行 {first}（max|Δ|={rowmax[first]:.3e}）"
                  f"    不同行区间数={(len(gaps) + 1) if bad else 0}（散点则区间数≈行数）")
            print(f"  前 20 个不同行: {bad[:20]}")
        top = torch.topk(rowmax, k=min(a.top, T))
        print(f"  {'行号':>7} {'max|Δ|':>11} {'mean|Δ|':>11} {'ref行幅值':>11} {'ours行幅值':>11}")
        for idx, v in zip(top.indices.tolist(), top.values.tolist()):
            print(f"  {idx:>7} {v:>11.3e} {rowmean[idx]:>11.3e} "
                  f"{r[idx].abs().max():>11.3e} {o[idx].abs().max():>11.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

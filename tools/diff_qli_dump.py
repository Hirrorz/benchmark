#!/usr/bin/env python3
"""比对两套 QLI dump（`DS41M_QLI_DUMP`），定位**第一处**不一致。

用途：命中 + CP 的索引器是否该按 CP 切（分块版 `_qli_cp`）——
两版输出的 `cmp_sparse_indices` 必须逐位相同，否则不能用分块版（会改 top-k 选择）。

dump 文件：`r<rank>_L<layer>_p<pass>.pt`，内含
    sp   : [T_loc, 1, topk] int32（本地行的 top-k 索引，-1 填充）
    tl   : [T_loc, 1] int32（有效条数）
    off  : 本段绝对起点
    Tloc : 本地行数
    q_rows: idx_q 的行数

用法：
    python3 tools/diff_qli_dump.py <dirA> <dirB> [--top 5]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def key_of(fn):
    """`r0_L2_p3.pt` → ("0", "2", "3")（rank / 层 / pass）。"""
    base = os.path.basename(fn)[:-3]
    d = {}
    for x in base.split("_"):
        d[x[0]] = x[1:]
    return (d.get("r", "?"), d.get("L", "?"), d.get("p", "?"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    fa = {key_of(f): f for f in glob.glob(os.path.join(a.a, "*.pt"))}
    fb = {key_of(f): f for f in glob.glob(os.path.join(a.b, "*.pt"))}
    common = sorted(set(fa) & set(fb))
    if not common:
        sys.exit(f"✗ 两套没有同名 dump（A {len(fa)} 个 / B {len(fb)} 个）")
    print(f"可比 dump：{len(common)} 个（A {len(fa)} / B {len(fb)}）")
    nbad = 0
    shown = 0
    for k in common:
        A, B = load(fa[k]), load(fb[k])
        sa, sb = A["sp"], B["sp"]
        if tuple(sa.shape) != tuple(sb.shape):
            print(f"  ✗ {k}: 形状不同 {tuple(sa.shape)} vs {tuple(sb.shape)}")
            nbad += 1
            continue
        # 先看「输入侧」是否就不同：K 缓存摘要 / 块表
        if A.get("k_sha") != B.get("k_sha") or A.get("bt_sha") != B.get("bt_sha"):
            print(f"  ⚠ 层{k[1]} pass{k[2]} rank{k[0]}: **输入侧就不同**  "
                  f"K sha {A.get('k_sha')} vs {B.get('k_sha')}"
                  f"（形状 {A.get('k_shape')} vs {B.get('k_shape')}）；"
                  f"块表 len {A.get('bt_len')} vs {B.get('bt_len')}，"
                  f"sha {A.get('bt_sha')} vs {B.get('bt_sha')}，前8 {A.get('bt')} vs {B.get('bt')}")
        diff = (sa != sb)
        n = int(diff.sum())
        if n == 0:
            continue
        nbad += 1
        rows = torch.nonzero(diff.any(-1).any(-1)).flatten().tolist()
        print(f"  ✗ 层{k[1]} pass{k[2]} rank{k[0]}: {n} 个元素不同，"
              f"涉及 {len(rows)} 行（共 {sa.shape[0]}），首行 {rows[0]}"
              f"（off={A['off']} q_rows={A['q_rows']}）")
        if shown < a.top:
            r = rows[0]
            idxs = torch.nonzero(diff[r].any(-1)).flatten().tolist()
            print(f"      行 {r}: 首个不同列 {idxs[0]}  A={sa[r, 0, idxs[0]].item()} "
                  f"B={sb[r, 0, idxs[0]].item()}")
            # 两行 top-k 的差异规模（有多少个索引项不同、多少只是顺序不同）
            va = sorted(x for x in sa[r, 0].tolist() if x >= 0)
            vb = sorted(x for x in sb[r, 0].tolist() if x >= 0)
            print(f"      该行有效条数 A={len(va)} B={len(vb)}；"
                  f"集合差 {len(set(va) ^ set(vb))}；顺序差 {sum(1 for x, y in zip(sa[r,0].tolist(), sb[r,0].tolist()) if x != y)}")
            shown += 1
    print()
    if nbad == 0:
        print("✅ 全部 dump 逐位相同")
        return 0
    print(f"❌ {nbad}/{len(common)} 个 dump 不一致")
    return 1


if __name__ == "__main__":
    sys.exit(main())

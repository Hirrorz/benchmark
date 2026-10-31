#!/usr/bin/env python3
"""精度 gate：**mini 的贪心下一 token × 官方真权重参考**（真权重 EP8 采集的"带精度"凭据）。

为什么这一条就够硬：官方参考（`parity/ref/`）与 mini 跑的是**同一份 16k 真 ids、同一份真权重、
同为 EP8**；末位 token 的预测只依赖整条 16k 上下文，stage A+B 命中口径在数学上又等价于全量 prefill
→ **top-1 相等**同时证明了：真权重装载、自然路由（无强均）、engram 真表取行、命中前缀复用 四件事。

用法：
    # ① 从 mini 的 rank0 日志抓（runner 跑完会打 "== 贪心下一个 token ==" + top-8 + 分布诊断）
    python3.11 tools/check_next_token.py --log <L16384/logs_hit/rank0.log> \
        --ref-dump /mnt/xds/sfs/l00972766/parity/ref/dump_rank0_call1.pt [--json out.json]
    # ② 更严：mini 侧也用 --dump-hidden 落的 logits，逐元素比（同源口径）
    python3.11 tools/check_next_token.py --mini-dump <mini_dump.pt> --ref-dump <ref.pt>

判据：**top-1 id 相等 = PASS**（唯一硬判据）；另报 top-8 重合数、逐位 |Δscore|、归一化熵对比。
历史基准：官方 = `id=396 ' that'`，score 37.750（mini 单卡时代打出的就是同一个值）。
退出码：0=PASS，1=FAIL，2=解析失败。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys

import torch

RE_TOP = re.compile(r"^\s*(-?[\d.]+)\s+id=(\d+)\s+(.*)$")
RE_ENT = re.compile(r"归一化熵\s+([\d.]+)")
RE_TOP1P = re.compile(r"top1 概率\s+([\d.eE+-]+)")


def topk_of(logits: torch.Tensor, k: int = 8):
    """→ [(id, score, piece?)]、归一化熵、top1 概率（与 runner 的打印口径一致）。"""
    lg = logits.reshape(-1).float()
    top = torch.topk(lg, k)
    p = torch.softmax(lg, dim=-1)
    ent = float(-(p * (p + 1e-12).log()).sum())
    return (list(zip(top.indices.tolist(), [float(x) for x in top.values])),
            ent / math.log(lg.numel()), float(p.max()))


def parse_log(path: str):
    """抓**最后一段** `== 贪心下一个 token ==` 的 top-8 + 诊断行。"""
    lines = open(path, errors="ignore").read().splitlines()
    idx = [i for i, l in enumerate(lines) if "== 贪心下一个 token ==" in l]
    if not idx:
        return None
    out, ent, p1 = [], None, None
    for l in lines[idx[-1]:]:
        if l.strip().startswith("分布诊断"):
            m = RE_ENT.search(l);  ent = float(m.group(1)) if m else None
            m = RE_TOP1P.search(l); p1 = float(m.group(1)) if m else None
            break
        m = RE_TOP.match(l)
        if m and len(out) < 8:
            out.append((int(m.group(2)), float(m.group(1))))
    return {"top": out, "ent": ent, "top1p": p1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="", help="mini rank0 日志（里面有贪心 top-8）")
    ap.add_argument("--mini-dump", default="", help="mini --dump-hidden 落的 .pt（含 logits，更严）")
    ap.add_argument("--ref-dump", default="/mnt/xds/sfs/l00972766/parity/ref/dump_rank0_call1.pt",
                    help="官方参考 dump（logits [1,1,129280] fp32）")
    ap.add_argument("--json", default="", help="把结论写成 JSON（供驱动写进 SUMMARY）")
    a = ap.parse_args()
    if not (a.log or a.mini_dump):
        ap.error("--log 或 --mini-dump 至少给一个")

    ref = torch.load(a.ref_dump, map_location="cpu")
    rtop, rent, rp1 = topk_of(ref["logits"])

    if a.mini_dump:
        d = torch.load(a.mini_dump, map_location="cpu")
        mtop, ment, mp1 = topk_of(d["logits"])
        src = a.mini_dump
    else:
        got = parse_log(a.log)
        if not got or not got["top"]:
            print(f"⚠️ {a.log} 里没有 '== 贪心下一个 token ==' 段（没给 --ids-file/--prompt？或用 --layers 跑了？）")
            return 2
        mtop, ment, mp1 = got["top"], got["ent"], got["top1p"]
        src = a.log

    m_top = dict(mtop)                              # id → score
    ref_ids = [i for i, _ in rtop]
    overlap = sum(1 for i, _ in mtop if i in ref_ids)
    dscore = max((abs(m_top[i] - s) for i, s in rtop if i in m_top), default=float("nan"))
    match = bool(mtop and mtop[0][0] == rtop[0][0])

    print(f"NEXT_TOKEN mini={mtop[0][0]} ref={rtop[0][0]} match={'YES' if match else 'NO'} "
          f"top8_overlap={overlap}/8 maxscore_delta={dscore:.3f} "
          f"norm_entropy={ment if ment is not None else float('nan'):.3f}/{rent:.3f} "
          f"top1p={mp1 if mp1 is not None else float('nan'):.3e}/{rp1:.3e}")
    print(f"  mini top-8: {[i for i, _ in mtop]}")
    print(f"  官方 top-8: {ref_ids}（分数 {[round(s, 3) for _, s in rtop][:4]}…）")
    if not match:
        print(f"  ✗ top-1 不一致：mini={mtop[0][0]}（{mtop[0][1]:.3f}）vs 官方={rtop[0][0]}（{rtop[0][1]:.3f}）")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"ok": match, "mini_top1": mtop[0][0] if mtop else None,
                       "ref_top1": rtop[0][0], "top8_overlap": overlap,
                       "max_score_delta": None if dscore != dscore else dscore,
                       "norm_entropy_mini": ment, "norm_entropy_ref": rent,
                       "top1_prob_mini": mp1, "top1_prob_ref": rp1,
                       "source": src, "ref": a.ref_dump}, fh, ensure_ascii=False, indent=1)
    print("精度 gate " + ("✅ PASS（top-1 与官方一致）" if match else "❌ FAIL"))
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())

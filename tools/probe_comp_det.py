#!/usr/bin/env python3



"""probe_comp_det.py — d.compressor 的**可重复性**探针。

背景（2026-10-21 统一路径 A/B）：同一进程里 layer2 的 kv_compress 输入
x/q/kv 逐位相同、state_cache 全 0（host 侧 sum=0），但 3 个 pass 的输出
cmp_kv 每次都不同；而**新进程的第一次调用**又跨运行可重复。
本探针在单进程里直接调 op，回答：
  1. 相同实参连续调 N 次 → 输出是否逐位相同？
  2. 每次换新的 zeros state / 新 clone 的 x → 是否仍不同？
  3. 若不同，差在哪些行（前 T/ratio 行 vs 尾部的 carry/垃圾行）？

用法：
  cd ds41_mini && source ./env_pack.sh && python3.11 tools/probe_comp_det.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_ds41_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    _q = _os.path.join(_ds41_root, _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import torch                                       # noqa: E402
import torch_npu                                   # noqa: E402,F401

import exec_ops                                    # noqa: E402

DEV = os.environ.get("DS41M_PROBE_DEV", "npu:0")
T, RATIO, H, D, OFF = 3456, 2, 5120, 512, 78464


def sha(t):
    return hashlib.sha1(t.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()[:12]


def main() -> int:
    torch.npu.set_device(int(DEV.split(":")[-1]))
    exec_ops.ensure_ops()
    torch.manual_seed(0)
    wkv = (torch.randn(D, H, dtype=torch.float32) * 0.02).to(torch.bfloat16).to(DEV).contiguous()
    wgate = (torch.randn(D, H, dtype=torch.float32) * 0.02).to(torch.bfloat16).to(DEV).contiguous()
    x = torch.randn(T, H, dtype=torch.bfloat16, device=DEV).contiguous()
    d = torch.ops.cann_ops_transformer.ds41

    def call(xt, stc):
        return d.compressor(
            x=xt, wkv=wkv, wgate=wgate,
            seqused=torch.tensor([T], dtype=torch.int32, device=DEV),
            start_pos=torch.tensor([OFF], dtype=torch.int32, device=DEV),
            cmp_ratio=RATIO,
            cu_seqlens=torch.tensor([0, T], dtype=torch.int32, device=DEV),
            state_cache=stc.flatten(-2),
            state_block_table=torch.ones(1, dtype=torch.int32, device=DEV))

    # 1) 同一组实参（同一个 stc）连续调用
    stc = torch.zeros(2, RATIO, 2, D, dtype=torch.float32, device=DEV)
    outs = []
    for i in range(4):
        o = call(x, stc)
        outs.append(o)
        print(f"[same objects] call{i}: {tuple(o.shape)} {sha(o)} stc_sum_after={float(stc.float().sum()):.3e}",
              flush=True)

    # 2) 每次全新 zeros state（同一 x）
    outs2 = []
    for i in range(3):
        stc2 = torch.zeros(2, RATIO, 2, D, dtype=torch.float32, device=DEV)
        o = call(x, stc2)
        outs2.append(o)
        print(f"[fresh state ] call{i}: {sha(o)}", flush=True)

    # 3) 每次 clone x + 全新 state
    outs3 = []
    for i in range(3):
        stc3 = torch.zeros(2, RATIO, 2, D, dtype=torch.float32, device=DEV)
        o = call(x.clone().contiguous(), stc3)
        outs3.append(o)
        print(f"[fresh x+st ] call{i}: {sha(o)}", flush=True)

    # 4) 逐行定位：同 objects 的 call0 vs call1 哪些行不同
    a, b = outs[0].float(), outs[1].float()
    row_diff = (a != b).any(dim=1)
    idx = torch.nonzero(row_diff).flatten().cpu().tolist()
    n = a.shape[0]
    print(f"\n[rows] total={n} 不同行数={len(idx)} 前10={idx[:10]} 后10={idx[-10:]}", flush=True)
    if idx:
        i0 = idx[0]
        print(f"[rows] 行{i0}: max|Δ|={float((a[i0] - b[i0]).abs().max()):.6e} "
              f"a[{i0}][:4]={a[i0][:4].cpu().tolist()} b[{i0}][:4]={b[i0][:4].cpu().tolist()}", flush=True)
        nb = int(row_diff[: n - 1].sum()) if n > 1 else 0
        print(f"[rows] 前 n-1={n-1} 行里不同 {nb} 行；最后一行不同={bool(row_diff[-1])}", flush=True)
    for tag, lst in (("same", outs), ("fresh_state", outs2), ("fresh_x_state", outs3)):
        print(f"[hashes] {tag}: {[sha(t) for t in lst]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

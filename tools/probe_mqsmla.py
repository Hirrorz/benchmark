#!/usr/bin/env python3
import os as _os, sys as _sys
_ds41_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    _q = _os.path.join(_ds41_root, _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)



"""单独验证 mqsmla 的最小调用（不进 runner，方便定位崩在哪一步）。

用法：python3 tools/probe_mqsmla.py [--T 1024] [--cmp 0|1]
"""
import argparse
import os
import sys

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exec_ops  # noqa: E402

exec_ops.ensure_ops()

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, default=1024)
ap.add_argument("--cmp", type=int, default=0)
ap.add_argument("--BS", type=int, default=128)
a = ap.parse_args()

T, BS, dev = a.T, a.BS, "npu:0"
N1, D, KD = 64, 512, 512
ratio = 2 if a.cmp else 0
print(f"T={T} cmp={a.cmp} BS={BS}", flush=True)

d = torch.ops.cann_ops_transformer.ds41
q = torch.randn(T, N1, D, dtype=torch.bfloat16, device=dev).contiguous()
print("q ok", flush=True)

nblk = (T + BS - 1) // BS
pool = torch.zeros(1 + nblk, BS, 1, 544, dtype=torch.float8_e4m3fn, device=dev)
print("pool ok", flush=True)

idx, length = exec_ops._win_indices(T, dev, BS)
cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
bt = torch.arange(1, 1 + nblk, dtype=torch.int32, device=dev).unsqueeze(0)
sinks = torch.zeros(N1, dtype=torch.float32, device=dev)
print(f"idx{tuple(idx.shape)} len{tuple(length.shape)} cu{tuple(cu.shape)} bt{tuple(bt.shape)}", flush=True)

kw = dict(quant_mode=1, ori_kv=pool.view(torch.uint8), ori_sparse_indices=idx,
          ori_block_table=bt, cu_seqlens_q=cu, ori_topk_length=length, sinks=sinks,
          softmax_scale=D ** -0.5, layout_q="TND", layout_kv="PA_BBND",
          return_softmax_lse=False)
cmp_len_t = torch.zeros_like(length)
if a.cmp:
    clen = max(1, T // ratio)
    cnblk = max(1, (clen + BS - 1) // BS)
    kw.update(cmp_kv=torch.zeros(cnblk, BS, 1, 320, dtype=torch.uint8, device=dev),
              cmp_sparse_indices=torch.full((T, 1, 512), -1, dtype=torch.int32, device=dev),
              cmp_block_table=torch.arange(cnblk, dtype=torch.int32, device=dev).unsqueeze(0),
              cmp_topk_length=cmp_len_t)
print("调用 metadata …", flush=True)
md = d.mixed_quant_sparse_flash_mla_metadata(
    length, cmp_len_t, cu_seqlens_q=cu, num_heads_q=N1, num_heads_kv=1, head_dim=D,
    quant_mode=1, layout_q="TND", layout_kv="PA_BBND",
    has_ori_kv=True, has_cmp_kv=bool(a.cmp))
torch.npu.synchronize()
print(f"metadata ok: {tuple(md.shape)} {md.dtype}", flush=True)

kw["metadata"] = md
print("调用 mqsmla …", flush=True)
out, lse = d.mixed_quant_sparse_flash_mla(q, **kw)
torch.npu.synchronize()
print(f"✓ mqsmla ok: out{tuple(out.shape)} {out.dtype} lse{tuple(lse.shape)}", flush=True)
print(f"  out 统计: absmax={out.abs().max().item():.4f} mean={out.float().mean().item():.6f}", flush=True)

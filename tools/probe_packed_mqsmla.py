#!/usr/bin/env python3



"""单卡探针：验证 packed（2×2048，varlen）与两条独立 2048 序列的**窗口注意力**逐位一致。

绕开 hc_pre 等对 T 敏感的算子，直接测我们改的胶水：
  `_kv_epilog`（槽位/块表）+ `_mqsmla`（cu_seqlens / per-request block table / 局部索引）。

用法（需要 runtime 环境）：
  env -i ... bash -c 'cd ds41_mini && source ./env_pack.sh && python3.11 tools/probe_packed_mqsmla.py'

判据：o_packed[:2048] == o_A、o_packed[2048:] == o_B（逐位）。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ds41_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    _q = _os.path.join(_ds41_root, _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import torch                                       # noqa: E402
import torch_npu                                   # noqa: E402,F401

import exec_ops                                    # noqa: E402

DEV = os.environ.get("DS41M_PROBE_DEV", "npu:0")
BS = 128
B, S = 2, 2048
T = B * S


def _mk_ctx(n_seqs, lens=None):
    cu = None
    if n_seqs > 1:
        cu = [0]
        for x in lens:
            cu.append(cu[-1] + int(x))
    return dict(T=int(sum(lens)) if lens else S, ratio=0, index_topk=512,
                pos_offset=0, L_full=int(sum(lens)) if lens else S,
                S=int(exec_ops.WIN_SLOTS // 3), head_dim=int(exec_ops.HEAD_DIM),
                n_seqs=int(n_seqs), cu_seqlens=cu, seq_lens=lens, idx=0)


def _run(q, kv, ctx, sink):
    st = {"q": q, "_shared": {}, "_backends": {}}
    w = {"attn.attn_sink": sink}
    st["kv"] = kv
    exec_ops._kv_epilog(st, ctx, w)
    exec_ops._mqsmla(st, ctx, w)
    return st["attn_out"]


def main() -> int:
    torch.npu.set_device(int(DEV.split(":")[-1]))
    exec_ops.ensure_ops()          # 注册 vendor 命名空间 + 插件（默认 spec）
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    q = torch.randn(T, exec_ops.N_HEADS, exec_ops.HEAD_DIM,
                    dtype=torch.bfloat16, device=DEV)
    kv = torch.randn(T, exec_ops.HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    sink = torch.zeros(exec_ops.N_HEADS, dtype=torch.float32, device=DEV)

    print(f"[probe] q{tuple(q.shape)} kv{tuple(kv.shape)} "
          f"N_HEADS={exec_ops.N_HEADS} HEAD_DIM={exec_ops.HEAD_DIM}", flush=True)

    o_pk = _run(q, kv, _mk_ctx(2, [S, S]), sink)
    torch.npu.synchronize()
    o_a = _run(q[:S].contiguous(), kv[:S].contiguous(), _mk_ctx(1), sink)
    o_b = _run(q[S:].contiguous(), kv[S:].contiguous(), _mk_ctx(1), sink)
    torch.npu.synchronize()

    ea = (o_pk[:S].float() - o_a.float()).abs().max().item()
    eb = (o_pk[S:].float() - o_b.float()).abs().max().item()
    print(f"[probe] packed vs A: max|Δ|={ea:.3e}  ({o_pk[:S].shape})")
    print(f"[probe] packed vs B: max|Δ|={eb:.3e}  ({o_pk[S:].shape})")
    print("✅ PASS（逐位）" if ea == 0.0 and eb == 0.0 else "❌ 有差异")
    return 0 if (ea == 0.0 and eb == 0.0) else 1


if __name__ == "__main__":
    sys.exit(main())

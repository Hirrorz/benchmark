#!/usr/bin/env python3



"""单卡探针：摸清 ds41.compressor 在 **batch（cu_seqlens 多请求）** 下的输出契约。

要回答两件事：
  1. 输出行数 = 各请求的 (s_b//ratio [+carry]) 怎么拼；
  2. 逐请求的 state_cache / start_pos / seqused 怎么传。

用法：
  env -i ... bash -c 'cd ds41_mini && source ./env_pack.sh && python3.11 tools/probe_packed_compressor.py'
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
S, RATIO, H, D = 2048, 2, 5120, 512


def _call(x, seqused, start_pos, cu, nstate):
    d = torch.ops.cann_ops_transformer.ds41
    stc = torch.zeros(1 + nstate, RATIO, 2, D, dtype=torch.float32, device=DEV)
    tbl = torch.arange(1, 1 + nstate, dtype=torch.int32, device=DEV)
    return d.compressor(
        x=x, wkv=WKV, wgate=WGATE,
        seqused=torch.tensor(seqused, dtype=torch.int32, device=DEV),
        start_pos=torch.tensor(start_pos, dtype=torch.int32, device=DEV),
        cmp_ratio=RATIO,
        cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=DEV),
        state_cache=stc.flatten(-2), state_block_table=tbl)


def main() -> int:
    global WKV, WGATE
    torch.npu.set_device(int(DEV.split(":")[-1]))
    exec_ops.ensure_ops()
    torch.manual_seed(0)
    WKV = torch.randn(D, H, dtype=torch.bfloat16, device=DEV) * 0.02
    WGATE = torch.randn(D, H, dtype=torch.bfloat16, device=DEV) * 0.02
    x = torch.randn(2 * S, H, dtype=torch.bfloat16, device=DEV)

    o1 = _call(x[:S].contiguous(), [S], [0], [0, S], 1)
    o2 = _call(x[S:].contiguous(), [S], [0], [0, S], 1)
    of = _call(x, [2 * S], [0], [0, 2 * S], 1)      # 单序列 4096（对照组）
    op = _call(x, [S, S], [0, 0], [0, S, 2 * S], 2)
    # 变体：cu/seqused 的压缩口径
    variants = {}
    for name, su, cu in (("B cu=compressed", [S, S], [0, S // RATIO, 2 * (S // RATIO)]),
                         ("C seqused=compressed", [S // RATIO, S // RATIO], [0, S, 2 * S]),
                         ("D both compressed", [S // RATIO, S // RATIO], [0, S // RATIO, 2 * (S // RATIO)])):
        try:
            o = _call(x, su, [0, 0], cu, 2)
            torch.npu.synchronize()
            variants[name] = o
            print(f"[probe] {name}: out {tuple(o.shape)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[probe] {name}: FAIL {type(e).__name__}: {str(e)[:120]}", flush=True)
    torch.npu.synchronize()
    print(f"[probe] single out {tuple(o1.shape)} / {tuple(o2.shape)}；packed out {tuple(op.shape)}", flush=True)

    # 猜测布局：每个请求 n = o1.shape[0] 行；比较候选拼接
    n = o1.shape[0]
    print(f"[probe] 单序列4096 out {tuple(of.shape)}", flush=True)
    if of.shape[0] >= n and op.shape[0] >= n:
        e = (op[:n].float() - of[:n].float()).abs().max().item()
        print(f"[probe] packed[:n] vs 单序列4096[:n]: max|Δ|={e:.3e}")
    # 假设1：packed 连续压缩（等价单序列4096）→ packed[1:2049] == of[1:2049]、packed[1024:] == o2
    for name, lo, hi, ref in (("packed[1025:2049] vs of[1025:2049]", 1025, 2049, of),
                              ("packed[1024:2049] vs o2", 1024, 2049, o2),
                              ("packed[1025:2050] vs o2", 1025, 2050, o2)):
        a = op[lo:hi]
        b = ref[(lo - 1024) if ref is o2 else lo:(hi - 1024) if ref is o2 else hi]
        if a.shape[0] == b.shape[0]:
            print(f"[probe] {name}: max|Δ|={(a.float()-b.float()).abs().max().item():.3e}")
    for name, aa, bb in (("packed[:8] vs A[:8]", op[:8], o1[:8]),
                         ("packed[n:n+8] vs B[:8]", op[n:n + 8], o2[:8])):
        d = (aa.float() - bb.float()).abs().max().item()
        print(f"[probe] {name}: max|Δ|={d:.3e}  A0={aa[0,:3].tolist()} B0={bb[0,:3].tolist()}")
    for name, o in variants.items():
        if o.shape[0] == 2 * n:
            ea = (o[:n].float() - o1.float()).abs().max().item()
            eb = (o[n:].float() - o2.float()).abs().max().item()
            print(f"[probe] 变体 {name}: A={ea:.3e} B={eb:.3e}")
    # ★ 官方 TND 口径（attention_data.generate_compressed_position_ids）：
    #   各请求的**真实压缩行打包在前**（每请求 s//ratio），pad 行统一甩到尾部。
    #   probe 条件：ratio=2、S=2048 → 每请求 1024 real + 1 carry/pad，总 2050。
    real = S // RATIO
    if op.shape[0] == 2 * real + 2:
        ea = (op[:real].float() - o1[:real].float()).abs().max().item()
        eb = (op[real:2 * real].float() - o2[:real].float()).abs().max().item()
        print(f"[probe] 官方布局 [A real | B real | pad×2]: "
              f"A max|Δ|={ea:.3e}  B max|Δ|={eb:.3e}")
        if ea == 0.0 and eb == 0.0:
            print("[probe] ✅ 官方 TND 布局逐位一致")
    if op.shape[0] == 2 * n:
        ea = (op[:n].float() - o1.float()).abs().max().item()
        eb = (op[n:].float() - o2.float()).abs().max().item()
        print(f"[probe] 布局 [A | B]，每段 {n} 行：A max|Δ|={ea:.3e}  B max|Δ|={eb:.3e}")
        ok = ea == 0.0 and eb == 0.0
    else:
        print(f"[probe] 行数不是 2×{n}（{op.shape[0]}），需要人工看布局")
        ok = False
    print("✅ 逐位一致" if ok else "⚠️ 需进一步确认布局/state 契约")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

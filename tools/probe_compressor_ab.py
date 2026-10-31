#!/usr/bin/env python3
"""probe_compressor_ab.py — 压缩器 `d.compressor` 的"stage A → stage B"序列复现器。

背景（2026-10-16 深夜定位）：命中口径的崩点= L14 的 `kv_compress` **算子里**产出整行 NaN
（输入 x/kv/q/residual 全有限 ✓，rank 随数据变 ✗，层恒为 L14 ✓，prefill 从不崩 ✓）。
命中 stage B 的调用与 prefill 的两点不同：`start_pos=段基址(15744)` 与 **复用 stage A 推进过的
state_cache（ring）**。

本探针在一个进程里复刻这个序列，并给出三组对照：
  A) 序列：先 start_pos=0/T=H 跑一次（推进 ring），再 start_pos=15744/T=640（命中场景 ✗）
  B) 对照1：第二次仍 start_pos=0（= prefill 语义 ✓）
  C) 对照2：第二次用全新 zeros 的 state（= 不复用 ring ✓）
判据：任一组输出出现 NaN ⇒ 复现成功 ✓（且能立刻看出是哪一因素触发 ✗）。

用法：python3.11 tools/probe_compressor_ab.py [--real]   # --real 用真权重（默认随机）
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

for _m in ("custom_ops", "cann_ops_transformer", "cann_ops_nn"):
    try:
        __import__(_m)
    except Exception as _e:                              # noqa: BLE001
        print(f"  [warn] import {_m} 失败：{type(_e).__name__}: {_e}")

DEV = "npu:0"
H, D, RATIO = 5120, 512, 2
T_A, T_B, OFF = 15872, 640, 15744
d = torch.ops.cann_ops_transformer.ds41


def mkW():
    if "--real" in sys.argv:
        raise SystemExit("--real 暂未接（先用随机权重）")
    torch.manual_seed(0)
    wkv = (torch.randn(D, H, dtype=torch.float32) * 0.02).to(torch.bfloat16).to(DEV)
    wgate = (torch.randn(D, H, dtype=torch.float32) * 0.02).to(torch.bfloat16).to(DEV)
    return wkv.contiguous(), wgate.contiguous()


def mkX(T):
    torch.manual_seed(T)
    return (torch.randn(T, H, dtype=torch.float32) * 1.0).to(torch.bfloat16).to(DEV).contiguous()


def call(x, wkv, wgate, T, start, stc):
    return d.compressor(
        x=x, wkv=wkv, wgate=wgate,
        seqused=torch.tensor([T], dtype=torch.int32, device=DEV),
        start_pos=torch.tensor([start], dtype=torch.int32, device=DEV),
        cmp_ratio=RATIO,
        cu_seqlens=torch.tensor([0, T], dtype=torch.int32, device=DEV),
        state_cache=stc.flatten(-2),
        state_block_table=torch.ones(1, dtype=torch.int32, device=DEV))


def chk(tag, kv):
    n = int((~torch.isfinite(kv)).sum())
    print(f"    {tag}: shape={tuple(kv.shape)} 非有限={n}"
          f"{'   ← ✗ 复现！' if n else '   ✓'}")
    return n


W1, W2 = mkW()
XA, XB = mkX(T_A), mkX(T_B)
print(f"权重随机、x 随机；ratio={RATIO}  A: T={T_A},start=0   B: T={T_B},start={OFF}")

print("\n[A] 序列（命中场景 ✗）：A(start=0) → B(start=段基址，复用 A 的 state）")
stc = torch.zeros(2, RATIO, 2, D, dtype=torch.float32, device=DEV)
chk("A 输出", call(XA, W1, W2, T_A, 0, stc))
chk("B 输出(start=15744, 复用 state)", call(XB, W1, W2, T_B, OFF, stc))

print("\n[B] 对照1（prefill 语义）：A(start=0) → B2(start=0，复用 state）")
stc = torch.zeros(2, RATIO, 2, D, dtype=torch.float32, device=DEV)
chk("A 输出", call(XA, W1, W2, T_A, 0, stc))
chk("B2 输出(start=0)", call(XB, W1, W2, T_B, 0, stc))

print("\n[C] 对照2（不复用）：A → B3(start=15744，但 state 全新）")
stc = torch.zeros(2, RATIO, 2, D, dtype=torch.float32, device=DEV)
chk("A 输出", call(XA, W1, W2, T_A, 0, stc))
stc2 = torch.zeros(2, RATIO, 2, D, dtype=torch.float32, device=DEV)
chk("B3 输出(新 state)", call(XB, W1, W2, T_B, OFF, stc2))
print("\n（若 [A] 出 NaN 而 [B]/[C] 干净 ⇒ 与 start_pos 或 state 复用直接相关 ✓）")

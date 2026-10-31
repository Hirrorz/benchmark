#!/usr/bin/env python3
"""probe_epilog_slot_m1.py — 判定 `kv_compress_epilog_v2` 对 **slot = -1** 的语义。

背景（2026-10-16）：命中口径的"重放段压缩行写入抑制"按官方 `adaptlib/hit.py` 的做法
给这些行传 slot=-1（官方语义 = 跳过）。但本 mini 用的是**厂商自定义算子**——若它把 -1
当成无符号大数写到别处，就是**越界写 → 内存污染 → 时崩时错的 NaN**（残余病C + 跑批翻盘的
共同嫌疑 ✗）。

做法：
  1. 池填**哨兵字节 0xAB**（整块，含 scale 区域）；
  2. 用混合槽位写一次：前一半行 → 有效槽位（按 (1+p//BS)*BS + p%BS）；后一半 → -1；
  3. 回读**逐字节**比对：
     · 有效槽位对应行是否**按预期改变**（不是 0xAB 了）✓
     · 其余所有字节是否**一个都没动**（仍是 0xAB）✓ ← 这一条是判据
  4. 另做一组全 -1 的对照（若全 -1 也动了别处，说明 -1 非法 ✗）。

用法：python3.11 tools/probe_epilog_slot_m1.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

# ★ 必须显式 import 才会注册 vendor 命名空间（见 exec_ops.py:37-51 的实测注释）
for _m in ("custom_ops", "cann_ops_transformer", "cann_ops_nn"):
    try:
        __import__(_m)
    except Exception as _e:                     # noqa: BLE001
        print(f"  [warn] import {_m} 失败：{type(_e).__name__}: {_e}")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEV = "npu:0"
BS = 128
ROWS, NELEM = 1 + 4 * BS, 320          # 池：5 块（块 0 空 + 4 块数据）；行宽 320 = mxfp4 的实际口径
N = 256                                # 写入行数（256 = 2 块）
D = 512                                # kv 维度


def run_case(tag, slots_first_half_valid: bool):
    pool = torch.full((ROWS, NELEM), 0xAB, dtype=torch.uint8, device=DEV)
    kv = torch.randn(N, D, dtype=torch.bfloat16, device=DEV).contiguous()

    pos = torch.arange(N, dtype=torch.int32, device=DEV)
    slot = (1 + pos // BS) * BS + pos % BS                 # 有效槽位（块 1..2）
    if slots_first_half_valid:
        slot[N // 2:] = -1                                 # 后一半 -1
    else:
        slot[:] = -1                                       # 全 -1（对照）

    before = pool.clone()
    torch.ops.custom.kv_compress_epilog_v2(
        pool.view(-1, pool.shape[-1]), kv, slot,
        quant_group_size=16, quant_mode="mxfp4_bf16", round_scale=True, x_scale=1.0)
    torch.npu.synchronize()

    changed = (pool != before)
    rows_touched = changed.any(dim=1).nonzero().flatten().tolist()
    # 期望被写的行集合
    valid_rows = set()
    if slots_first_half_valid:
        for s in slot[:N // 2].tolist():
            valid_rows.add(int(s))
    unexpected = [r for r in rows_touched if r not in valid_rows]

    print(f"[{tag}] 行被改动的：{rows_touched}")
    print(f"[{tag}] 期望被写：{sorted(valid_rows)}")
    print(f"[{tag}] **意外被改**：{unexpected}  → "
          f"{'✓ -1 安全（跳过）' if not unexpected else '✗ -1 会写到别处（越界/污染）'}")
    # 被写行的内容是否真变了（有效写成功的证据）
    if valid_rows:
        r0 = sorted(valid_rows)[0]
        print(f"[{tag}] 有效行 {r0} 前 16 字节：{pool[r0, :16].tolist()}")
    return unexpected


print(f"设备 {torch.npu.get_device_name(0)}；池 {ROWS}×{NELEM}（哨兵 0xAB）")
u1 = run_case("混合槽位(有效+-1)", True)
print()
u2 = run_case("全 -1（对照）", False)
print()
print("结论：", "✓ 官方 '-1=跳过' 语义成立，抑制不会污染内存"
      if not u1 and not u2 else "✗ **-1 在本厂商算子上不安全** ⇒ 抑制必须换写法（如指向 scratch 块）")

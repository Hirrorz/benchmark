#!/usr/bin/env python3
"""C6 #7 / GPT §14：`CpState` **逐请求坐标**（去掉"只能等长同起点"的限制）。

判据（纯 CPU + torch，无需 npu）：
  1. **identity**：`pos_offsets=[o,o]`（各请求同起点）与标量 `pos_offset=o`
     逐元素相同 —— 证明新路径不改变旧语义；
  2. **逐请求**：`pos_offsets=[a,b]` 时，第 i 个请求的行 = 自己的 local_positions + 各自起点；
     且两请求的绝对位置区间可**不重叠/不等长**（旧的标量口径做不到）；
  3. **pad 行**：有 pad 时 `valid=False` 的行不参与（`local_positions` 已 clamp_min(0)），
     逐请求路径不得抛错；
  4. 长度校验：`pos_offsets` 长度 ≠ 请求数 ⇒ 显式 `ValueError`（不静默截断）。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("cp", "layout", "core", "meta", "ops", "trace"):
    sys.path.insert(0, os.path.join(ROOT, _p))

import torch                       # noqa: E402
import cp_layout                   # noqa: E402
from cp_state import CpState       # noqa: E402

FAIL = []


def chk(name, cond, detail=""):
    print(("PASS" if cond else "FAIL") + " · " + name + ("" if cond else f"  <<< {detail}"))
    if not cond:
        FAIL.append(name)


def _pos(lens, cp, rank, off=None, scalar=0):
    plan = cp_layout.build_cp_plan(lens, cp, rank)
    st = CpState.from_plan(plan, device="cpu", pos_offset=scalar,
                           pos_offsets=off)
    return st, plan


def check_identity():
    """① 同起点的逐请求 == 标量。"""
    for lens, cp, rank in ([([8192], 2, 0), ([8192], 4, 1),
                            ([4096, 4096], 2, 0), ([4096, 4096], 8, 3)]):
        a, plan = _pos(lens, cp, rank, off=[512] * len(lens), scalar=512)
        b, _ = _pos(lens, cp, rank, off=None, scalar=512)
        same = torch.equal(a.abs_positions, b.abs_positions)
        chk(f"identity lens={lens} cp={cp} r={rank}：逐请求==标量", same,
            f"{a.abs_positions.tolist()[:6]} vs {b.abs_positions.tolist()[:6]}")


def check_per_request():
    """② 不等起点：每个请求的行用**自己**的起点，且区间可不等长/不重叠。"""
    lens = [2048, 4096]
    cp = 2
    st, plan = _pos(lens, cp, 0, off=[100, 500])
    pos = st.abs_positions.tolist()
    loc = st.local_positions.tolist()
    # 本 rank 行边界：plan.local_cu
    n0 = int(plan.local_lens[0])
    seg0 = pos[:n0]
    seg1 = pos[n0:]
    ok0 = all(p == l + 100 for p, l in zip(seg0, loc[:n0]))
    ok1 = all(p == l + 500 for p, l in zip(seg1, loc[n0:]))
    chk("不等起点：请求 0 用 +100、请求 1 用 +500", ok0 and ok1,
        f"seg0={seg0[:4]} seg1={seg1[:4]}")
    chk("两个请求的绝对位置区间不重叠（旧标量口径做不到）",
        max(seg0) < min(seg1), f"{max(seg0)} vs {min(seg1)}")


def check_pads_and_guard(torch_mod=None):
    """③ pad 行不抛错；④ 长度不符必须报错。"""
    # L 不整除 2·cp 时会产生 pad（-1）
    st, plan = _pos([1000, 1000], 8, 0, off=[0, 7])
    chk("有 pad 的几何：逐请求路径不抛错且行数一致",
        st.abs_positions.shape[0] == len(plan.local_index),
        f"{st.abs_positions.shape} vs {len(plan.local_index)}")
    try:
        _pos([1000, 1000], 8, 0, off=[0])
        chk("pos_offsets 长度不符 ⇒ ValueError", False, "竟然通过")
    except ValueError as e:
        chk("pos_offsets 长度不符 ⇒ ValueError（不静默截断）", "pos_offsets" in str(e), str(e)[:80])


def main():
    print("== C7：逐请求坐标 ==")
    check_identity()
    check_per_request()
    check_pads_and_guard()
    print()
    if FAIL:
        print(f"✗ cp_state_offsets: {len(FAIL)} 项失败")
        for n in FAIL:
            print(f"   - {n}")
        return 1
    print("✓ cp_state_offsets: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""`core/memlife.py` 的 CPU 自检（不碰卡，秒级）。

跑法： python3 tools/selfcheck_memlife.py   （或 python3.11）

覆盖 4 条机制：
  1. **发现**：按 `__file__` 目录找到本树的模块（裸名 `exec_ops`/`runner` 也算）——
     这条曾经写成按模块名过滤，结果一个都扫不到、sweep 静默变空操作；
  2. **登记释放**：`POINT_CACHES` 会被 `sweep()` 清掉，`POINT_RELEASE()` 会被调用；
  3. **归因**：`container_snapshot()` 能按容器统计字节；
  4. **兜底**：`sweep_unknown()` 清「未登记且增长超阈值」的容器，
     但**不动**已登记的和 `RESIDENT` 白名单里的。

为了脱离 NPU 也能跑，本脚本把 `memlife._is_npu_tensor` 换成"任何 torch.Tensor"。
"""
from __future__ import annotations

import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

import torch                                                   # noqa: E402
import memlife                                                 # noqa: E402

memlife._is_npu_tensor = lambda o: isinstance(o, torch.Tensor)   # CPU 自检：不认设备
FAILS = []


def check(name, cond, extra=""):
    print(("  ✅ " if cond else "  ✗ ") + name + (f"：{extra}" if extra else ""))
    if not cond:
        FAILS.append(name)


def fake_module(name, path, **attrs):
    m = types.ModuleType(name)
    m.__file__ = path
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def main():
    print("== memlife selfcheck")
    # ① 发现
    found = {n for n, _ in memlife._our_modules()}
    check("发现本树模块（裸名也算）", "exec_ops" in found or "memlife" in found,
          f"{len(found)} 个：{sorted(found)[:5]}…")
    outside = fake_module("notours", "/tmp/notours.py")
    found2 = {n for n, _ in memlife._our_modules()}
    check("不误认树外模块", "notours" not in found2)

    # ② 登记释放
    t = torch.zeros(1024, 1024, dtype=torch.float32)             # 4 MiB
    reg = {"a": t}
    hook = []
    fake_module("ops.selfcheck", os.path.join(ROOT, "ops", "selfcheck.py"),
                POINT_CACHES=(reg,), POINT_RELEASE=lambda: hook.append(1))
    memlife.sweep()
    check("POINT_CACHES 被清空", reg == {})
    check("POINT_RELEASE 被调用", hook == [1])

    # ③ 归因：按**模块属性**统计（真实模块的缓存都是独立的模块级名字）
    reg2 = {"b": torch.zeros(2 * 1024 * 1024, dtype=torch.float32)}   # 8 MiB
    fake_module("ops.selfcheck2", os.path.join(ROOT, "ops", "selfcheck2.py"),
                _REG_SELFCHECK2=reg2, POINT_CACHES=(reg2,))
    snap = memlife.container_snapshot()
    nb = snap.get("ops.selfcheck2._REG_SELFCHECK2", 0)
    check("container_snapshot 计入模块级缓存", 7.5e6 < nb < 8.5e6, f"{nb / 2**20:.2f} MiB")

    # ④ 兜底：未登记 + 增长超阈值 → 清；已登记 / RESIDENT → 不动
    unreg = {"e": torch.zeros(4 * 1024 * 1024, dtype=torch.float32)}  # 16 MiB
    reg3 = {"c": torch.zeros(4 * 1024 * 1024, dtype=torch.float32)}
    resident = {"d": torch.zeros(4 * 1024 * 1024, dtype=torch.float32)}
    fake_module("ops.selfcheck3", os.path.join(ROOT, "ops", "selfcheck3.py"),
                _UNREG_SELFCHECK=unreg, _REG_SELFCHECK3=reg3,
                _PACK_CACHE=resident, POINT_CACHES=(reg3,))      # _PACK_CACHE 在 RESIDENT
    memlife._CONTAINER_BASE.clear()
    memlife.sweep_unknown(1.0)                                   # 建基线（当前值）
    unreg["f"] = torch.zeros(8 * 1024 * 1024, dtype=torch.float32)   # 再长 32 MiB
    reg3["g"] = torch.zeros(8 * 1024 * 1024, dtype=torch.float32)    # 登记的也长（不该被兜底清）
    freed = memlife.sweep_unknown(0.001)                         # 1 MiB 阈值 → 触发
    check("sweep_unknown 清掉未登记的增长", "f" not in unreg, str([k for _, k in freed]))
    check("sweep_unknown 不动 RESIDENT", "d" in resident)
    check("sweep_unknown 不动已登记容器", "c" in reg3 and "g" in reg3)

    # 收尾：把自检造的假模块摘掉，别污染别人的进程（本脚本本来就短命）
    for n in ("notours", "ops.selfcheck", "ops.selfcheck2", "ops.selfcheck3"):
        sys.modules.pop(n, None)
    print("== 结果：" + ("全过 ✅" if not FAILS else f"失败 {len(FAILS)} 项 ❌ {FAILS}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())

"""cannbotdsl ABI 契约缓存 —— 纯 host 优化，数值零影响（默认开）。

## 为什么

`torch.ops.cann_ops_transformer.ds41.*`（mqsmla / qli / qsli / idx_prolog）在 runtime 走
cannbotdsl 的 provider 调用桥：

    op(...) -> runtime_bridge.invoke_provider_call -> resolve_provider_call
            -> parameter_lowering.LoweredCall.bind_physical -> cannbotdsl._host_runtime.invoke

其中 `CallContract.runtime_parameters` / `.abi_arguments` 是 frozen dataclass 上的
**普通 @property**，每次访问都重做一遍 tuple 推导：

    cannbotdsl/core/contracts/model.py:177  runtime_parameters -> tuple(p for p in self.parameters if p.role=="runtime")
    cannbotdsl/core/contracts/model.py:181  abi_arguments      -> tuple(p.argument for p in self.runtime_parameters if p.argument is not None)

而 `parameter_lowering.py:177 bind_physical` 在 **for 循环体内**访问 `self.abi_arguments[index]`
⇒ 每个 runtime 参数都重建一次整条 tuple（O(n²)）。

实测来源
--------
- cProfile `tmp/t042/t042_prof/ds41m_host_r0.prof`（4 个 forward）：
  244 次 `invoke_provider_call`；`model.py:181 abi_arguments` **11352 次**（46.5 次/invoke）、
  `model.py:177 runtime_parameters` **11596 次**；两者 cum 0.109 s / 0.864 s = 12.6% 的 host 采样。
- CPU 微基准（本机、无卡，47 参数契约 / 20 个 runtime 参数）：单次 `abi_arguments` 访问 2.63 µs；
  按 profile 的每次 invoke 46+47 次访问 ⇒ **195 µs/invoke**；61 invoke/forward ⇒ **≈11.9 ms/forward**。

## 为什么安全

`CallContract` 是 `@dataclass(frozen=True, slots=True)`，`parameters` 是 tuple，
没有任何代码写入 `.parameters`（全仓 grep 无赋值），也没有子类覆盖这两个属性 ⇒
属性的值只由 `self.parameters` 决定，缓存后**逐位不变**。缓存的强引用保证 id 不会被复用。

## 开关

- `DS41M_FAST_ABI=0` 关闭（回退到 vendor 原实现）。
- `DS41M_FAST_ABI_STATS=1` 退出时打印命中/未命中和估算省下的时间。

## 预期

- host：≈11.9 ms/forward 的纯 Python 归零（换算成 wall 只有 replay 尾段能兑现，见
  NOTES/2026-10-26-a4-deepinfer-host.md §3：尾段是 host-bound，头段 host 被设备完全盖住）。
- 设备/数值：零影响（不改变任何张量、 stream、参数顺序）。
"""

from __future__ import annotations

import os
import time

_ON = os.environ.get("DS41M_FAST_ABI", "1").strip().lower() not in ("0", "off", "false", "no")
_STATS = os.environ.get("DS41M_FAST_ABI_STATS", "").strip().lower() in ("1", "on", "true", "yes")

# id(contract) -> (contract, runtime_parameters, abi_arguments)
#   ★ 强引用 contract：契约是长生命周期对象，数量只有几十个；强引用同时保证 id 不被复用。
_MEMO: dict[int, tuple[object, tuple, tuple]] = {}

_HIT = 0
_MISS = 0
_SAVED_S = 0.0      # 未命中时"如果按原实现算"的耗时估算（用于自证收益）
_ACCESS = 0         # 每次属性访问（含命中）


def _pair(contract):
    """返回 (runtime_parameters, abi_arguments)，只按 contract 计算一次。"""
    global _HIT, _MISS, _SAVED_S, _ACCESS
    _ACCESS += 1
    key = id(contract)
    rec = _MEMO.get(key)
    if rec is not None and rec[0] is contract:
        _HIT += 1
        return rec[1], rec[2]
    t0 = time.perf_counter()
    rt = tuple(p for p in contract.parameters if p.role == "runtime")
    abi = tuple(p.argument for p in rt if p.argument is not None)
    if _STATS:
        _SAVED_S += time.perf_counter() - t0
    _MISS += 1
    rec = (contract, rt, abi)
    _MEMO[key] = rec
    return rt, abi


def _get_runtime_parameters(self):
    return _pair(self)[0]


def _get_abi_arguments(self):
    return _pair(self)[1]


_installed = False


def install(force: bool = False) -> bool:
    """幂等安装；返回是否（或已经）生效。失败一律静默回退（不改变任何行为）。"""
    global _installed
    if _installed:
        return True
    if not _ON and not force:
        return False
    try:
        from cannbotdsl.core.contracts.model import CallContract
    except Exception:                                     # noqa: BLE001  vendor 缺失/改名 → 不动
        return False
    try:
        # 只在确认是"原实现"时替换，避免把别人的补丁再包一层
        if isinstance(vars(CallContract).get("abi_arguments"), property):
            CallContract.runtime_parameters = property(_get_runtime_parameters)
            CallContract.abi_arguments = property(_get_abi_arguments)
        _installed = True
    except Exception:                                     # noqa: BLE001
        return False
    if _STATS:
        import atexit

        @atexit.register
        def _dump():                                      # noqa: ANN202
            miss = _MISS
            if miss:
                print(f"  [fast-abi] 访问 {_ACCESS} 次 / 未命中 {miss}（每个契约一次）"
                      f" / 命中率 {100.0 * _HIT / max(1, _ACCESS):.1f}%"
                      f" / 未命中累计 {_SAVED_S * 1e3:.2f} ms"
                      f" ≈ 省下 {_SAVED_S * max(0.0, _ACCESS / max(1, miss) - 1) * 1e3:.0f} ms host",
                      flush=True)
    return _installed


def stats() -> dict:
    return {"access": _ACCESS, "hit": _HIT, "miss": _MISS, "contracts": len(_MEMO)}

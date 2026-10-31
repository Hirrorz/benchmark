"""bind_probe —— 测「vendor 绑定后直发」的 host 上限（诊断，默认关）。

`DS41M_BIND_PROBE=1` 时，等 mq_fastcall2 的桥安装完成后，在
`ProviderCallable.__call__` 外再包一层：抓一次真实调用的 `(self, args, kwargs)`，
然后调用 vendor 的 benchmark 接口：

    res   = self._benchmark_resolve(*args, **kwargs)     # 完整解析（一次）
    bound = self._benchmark_bind(res)                     # 绑定（一次）
    bound.invoke()                                        # 直发（每次）

对同一组张量把 `bound.invoke()` 与正常 `self(*args, **kwargs)` 各跑 N 次求均值，
打印 resolve/bind/直发/正常 的 host µs/次。

安全性：同一 kernel、同一组张量、同一流；bound.invoke 只是重复下同一发，
输出张量会被同样内容覆盖，数值不变。**这只测量，不接入任何生产路径。**
"""
from __future__ import annotations

import os
import threading
import time

_ON = os.environ.get("DS41M_BIND_PROBE", "").strip().lower() in ("1", "on", "true", "yes")
_N = int(os.environ.get("DS41M_BIND_PROBE_N", "20") or 20)
_AT = int(os.environ.get("DS41M_BIND_PROBE_AT", "4") or 4)   # 第几次调用后测
_TRIED = [False]


def install():
    if not _ON or _TRIED[0]:
        return
    _TRIED[0] = True
    threading.Thread(target=_wait_and_patch, daemon=True).start()


def _wait_and_patch():
    try:
        import mq_fastcall2
    except Exception:                                    # noqa: BLE001
        return
    deadline = time.time() + 180
    while time.time() < deadline:
        if getattr(mq_fastcall2, "_BRIDGE_OK", False) and getattr(
                mq_fastcall2, "_ORIG_CALL", None) is not None:
            break
        time.sleep(0.2)
    else:
        return
    from cannbotdsl.core.compiler import function as _fn
    PC = _fn.ProviderCallable
    cur = PC.__call__
    st = {"n": 0, "done": False}

    _DBG = os.environ.get("DS41M_BIND_PROBE_DEBUG", "").strip().lower() in (
        "1", "on", "true", "yes")

    def _call(self, *a, **kw):
        out = cur(self, *a, **kw)
        st["n"] += 1
        if _DBG and st["n"] % 10 == 0:
            print(f"  [bind-probe] seen={st['n']} type={type(self).__name__}", flush=True)
        if st["n"] == _AT and not st["done"]:
            st["done"] = True
            try:
                _probe(self, a, kw)
            except BaseException as e:                   # noqa: BLE001  诊断失败不影响主流程
                print(f"  [bind-probe] fail: {type(e).__name__}: {e}", flush=True)
        return out

    PC.__call__ = _call
    print("  [bind-probe] 已挂载（等真实调用）", flush=True)


def _probe(callable_obj, args, kwargs):
    import torch
    t0 = time.perf_counter()
    res = callable_obj._benchmark_resolve(*args, **kwargs)
    t1 = time.perf_counter()
    bound = callable_obj._benchmark_bind(res)
    t2 = time.perf_counter()
    bound.invoke()
    torch.npu.synchronize()
    t3 = time.perf_counter()
    for _ in range(_N):
        bound.invoke()
    t4 = time.perf_counter()
    torch.npu.synchronize()
    t5 = time.perf_counter()
    for _ in range(_N):
        callable_obj(*args, **kwargs)
    t6 = time.perf_counter()
    torch.npu.synchronize()
    t7 = time.perf_counter()
    print(f"  [bind-probe] resolve={(t1-t0)*1e6:.0f}us bind={(t2-t1)*1e6:.0f}us "
          f"bound.invoke={(t4-t3)/_N*1e6:.0f}us normal={(t6-t5)/_N*1e6:.0f}us "
          f"（N={_N}；含队列排空 {((t5-t4)+(t7-t6))*1e3:.0f}ms）", flush=True)

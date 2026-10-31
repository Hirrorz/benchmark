"""bind_fast —— vendor「绑定后直发」的生产原型（2026-10-29，默认关）。

实测（`ops/bind_probe.py`，8k CP2）：同一 kernel 同一张量
    正常 `ProviderCallable(...)`      ≈ 361–363 µs/次
    `_benchmark_resolve`              ≈ 385–397 µs（一次性）
    `_benchmark_bind`                 ≈  83–88 µs（一次性）
    `bound.invoke()`                  ≈ **21–23 µs/次**
⇒ 命中绑定时每次调用省 ~340 µs。

做法：在 `ProviderCallable.__call__` 外挂一层（等 mq_fastcall2 的桥装好后包装它）：
  · 计算**完整实参指纹**（张量与标量，见 `_fp`）；带当前 stream handle；
  · 命中 → `bound.invoke()`（等价于正常调用；out/lse 是预分配张量，内容由 kernel 写入）；
  · 未命中 → 走原路；同一指纹**第二次**出现时建 binding
    （`_benchmark_resolve`+`_benchmark_bind`，一次性 ~0.5 ms），之后命中。
安全闸：
  · 每调用张量总字节 ≤ `DS41M_BIND_FAST_MAX_MB`（默认 512 MB；16k 尾段约 20 MB，
    256k 尾段的池视图约 84 MB，stage A 的大 q 仍会被挡），
    全局 pinned ≤ `DS41M_BIND_FAST_TOTAL_MB`（默认 2048 MB）；
  · 指纹含 data_ptr/shape/stride/dtype/storage_offset/npu_format/is_neg/is_conj + 标量值 +
    `id(callable)` + stream handle；任一不同 → miss；
  · 任何异常 → 回退原路（不吞算子错误）。

开关：`DS41M_BIND_FAST=1`；`DS41M_BIND_FAST_STATS=1` 打印命中统计。
"""
from __future__ import annotations

import atexit
import os
import threading
import time

_ON = os.environ.get("DS41M_BIND_FAST", "1").strip().lower() in ("1", "on", "true", "yes")
_STATS_ON = os.environ.get("DS41M_BIND_FAST_STATS", "").strip().lower() in (
    "1", "on", "true", "yes")
_MAX_MB = float(os.environ.get("DS41M_BIND_FAST_MAX_MB", "512") or 512)
_TOTAL_MB = float(os.environ.get("DS41M_BIND_FAST_TOTAL_MB", "2048") or 2048)
_BIND_AT = int(os.environ.get("DS41M_BIND_FAST_AT", "2") or 2)   # 同 key 第几次出现才建绑定

_CACHE: dict = {}
_SEEN: dict = {}
_PINNED = [0.0]
_ST = {"hit": 0, "miss": 0, "bound": 0, "fallback": 0, "toobig": 0, "small": 0}
_MIN_NUMEL = int(float(os.environ.get("DS41M_BIND_FAST_MIN_NUMEL", "262144") or 262144))
_TT = None
_DUMP = os.environ.get("DS41M_BIND_FAST_DUMP", "").strip().lower() in (
    "1", "on", "true", "yes")
_LAST: dict = {}
_DIFF_N = [0]


def _diff(key):
    """诊断：同一实参个数下，本次 key 与上一次 key 的首个差异位。"""
    if _DIFF_N[0] >= 8:
        return
    n = len(key) - 2                                   # 去掉 id(self) 和 stream
    prev = _LAST.get(n)
    _LAST[n] = key
    if prev is None or len(prev) != len(key):
        return
    for i in range(1, len(key) - 1):
        if prev[i] != key[i]:
            d = key[i]
            p = prev[i]
            tag = (f"tensor dp={d[1]} shape={d[2]} fmt={d[6]} off={d[5]}"
                   if isinstance(d, tuple) and d and d[0] == 1 else f"val={d}")
            ptag = (f"dp={p[1]} shape={p[2]} fmt={p[6]}"
                    if isinstance(p, tuple) and p and p[0] == 1 else f"val={p}")
            print(f"  [bind-fast] diff arg#{i - 1}: now[{tag}] prev[{ptag}]", flush=True)
            _DIFF_N[0] += 1
            return


def install():
    if not _ON:
        return
    threading.Thread(target=_wait_and_patch, daemon=True).start()
    atexit.register(_dump)


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

    def _call(self, *a, **kw):
        global _TT
        if _TT is None:
            import torch as _t
            _TT = _t.Tensor
        # 小 T（decode 步、metadata 等）不参与绑定：键每步都变，绑定只添开销。
        if _MIN_NUMEL > 0:
            try:
                a0 = a[0]
                if type(a0) is not _TT or int(a0.numel()) < _MIN_NUMEL:
                    _ST["small"] += 1
                    return cur(self, *a, **kw)
            except Exception:                            # noqa: BLE001
                pass
        try:
            key, nbytes = _fp(self, a, kw)
        except Exception:                                # noqa: BLE001
            _ST["fallback"] += 1
            return cur(self, *a, **kw)
        if _DUMP and 15 <= len(a) <= 26:
            _diff(key)
        b = _CACHE.get(key)
        if b is not None:
            try:
                b.invoke()
                _ST["hit"] += 1
                return None
            except Exception:                            # noqa: BLE001  binding 失效 → 原路
                _CACHE.pop(key, None)
                _ST["fallback"] += 1
        if nbytes > _MAX_MB * 1024 * 1024 or _PINNED[0] + nbytes > _TOTAL_MB * 1024 * 1024:
            _ST["toobig"] += 1
            return cur(self, *a, **kw)
        if len(_SEEN) > 8192:
            _SEEN.clear()
        n = _SEEN.get(key, 0) + 1
        _SEEN[key] = n
        out = cur(self, *a, **kw)                        # 先走原路
        _ST["miss"] += 1
        if n >= _BIND_AT:
            try:
                res = self._benchmark_resolve(*a, **kw)
                bd = self._benchmark_bind(res)
            except Exception:                            # noqa: BLE001
                _ST["fallback"] += 1
                _SEEN.pop(key, None)
                return out
            _CACHE[key] = bd
            _PINNED[0] += nbytes
            _ST["bound"] += 1
        return out

    PC.__call__ = _call


def _fp(self, args, kwargs):
    """完整指纹 + 本次张量总字节数；不可判定 → 抛异常（调用方回退）。"""
    import torch_npu
    if kwargs:
        raise TypeError("kwargs present")
    parts = [id(self)]
    nbytes = 0.0
    for v in args:
        if type(v) is _TT:
            dev = v.device
            if dev.type != "npu":
                raise TypeError("not npu")
            # 逻辑字节数（不是 untyped_storage）：pool/权重的视图共享常驻 storage，
            # 绑它们不增加实际内存；真正被 pin 住的是每层新分配的 q/out 这类瞬态张量。
            nbytes += float(v.numel() * v.element_size())
            parts.append((1, int(v.data_ptr()), tuple(v.shape), tuple(v.stride()),
                          v.dtype, int(v.storage_offset()),
                          int(torch_npu.get_npu_format(v)),
                          bool(v.is_neg()), bool(v.is_conj()), dev.index))
        elif v is None:
            parts.append((2,))
        elif isinstance(v, (int, float, bool, str)):
            parts.append((3, type(v).__name__, v))
        else:
            raise TypeError(f"unsupported arg {type(v).__name__}")
    import torch
    parts.append(("stream", int(torch.npu.current_stream().npu_stream)))
    return tuple(parts), nbytes


def _dump():
    if _STATS_ON and (_ST["hit"] or _ST["miss"]):
        hist: dict = {}
        for v in _SEEN.values():
            hist[v] = hist.get(v, 0) + 1
        h = " ".join(f"{k}x:{v}" for k, v in sorted(hist.items())[:8])
        print(f"  [bind-fast] hit={_ST['hit']} miss={_ST['miss']} bound={_ST['bound']} "
              f"fallback={_ST['fallback']} toobig={_ST['toobig']} small={_ST['small']} "
              f"cache={len(_CACHE)} pinned={_PINNED[0]/2**20:.1f}MiB "
              f"keys={len(_SEEN)} seen_hist[{h}]", flush=True)

# ── 点位边界（core/memlife.py）：vendor 绑定**钉住**实参张量（含 base storage）──
# `_CACHE` 以 data_ptr 为键：每个新点位/新几何都造一批新绑定，旧绑定无人清 ⇒
# 设备上的 q/out/池视图被钉死。预算上限（`_PINNED`）只统计"实参逻辑字节"，
# 不是真实占用 ⇒ 必须随点位整体放掉。`_SEEN` 同步清（否则第一次见到的新地址
# 会沿用旧计数直接建绑定）。
POINT_CACHES = (_CACHE, _SEEN)


def POINT_RELEASE():                                     # noqa: N802 —— memlife 约定名
    _CACHE.clear()
    _SEEN.clear()
    _PINNED[0] = 0.0

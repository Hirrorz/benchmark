"""Phase 1.5：上下文读写探针（默认关；开启后不改任何数值）。

用途：用**实测**的 ctx/st 读写修正 `STAGES/op_iface.py` 的接口声明，并证明
"探针开/关逐位一致"。

开关（环境变量，进程启动前设置）：
  DS41M_CTX_TRACE=1          启用探针（默认关）
  DS41M_CTX_TRACE_OUT=path   结束时把报告写成 JSON（默认只打印摘要）
  DS41M_CTX_TRACE_VERBOSE=1  每次 forward 结束都写一次（默认进程退出写）

实现要点：
  · `CtxProbe` / `StProbe` 是 **dict 子类**（底层真实持有数据），因此 `dict(ctx)`、
    `ctx.items()`、`st["_shared"]` 等现有用法语义不变；
  · 只重写 getitem/get/contains/setitem/pop/update/setdefault，记录 (op, side, action, key)；
  · 当前 op 用**线程局部**变量（engram 侧流不会被算到主线程的算子上）；
  · 记录键同时保留原始名与 `names.canonical` 规范名。
"""
from __future__ import annotations

import atexit
import json
import os
import threading
from collections import defaultdict

from modules import names

_LOCAL = threading.local()
_LOCK = threading.RLock()
_RECORDS: dict = {}
_ON = False


def trace_enabled() -> bool:
    return os.environ.get("DS41M_CTX_TRACE", "").strip().lower() in (
        "1", "on", "true", "yes")


def maybe_enable() -> bool:
    """runner 在导入/启动时调用一次；返回是否启用（供热循环做布尔判断）。"""
    global _ON
    _ON = trace_enabled()
    if _ON:
        atexit.register(flush)
    return _ON


def is_on() -> bool:
    return _ON


def set_current(op: str | None) -> None:
    _LOCAL.current = op


def _current() -> str:
    return getattr(_LOCAL, "current", None) or "<runner>"


def _bucket(op: str, side: str, action: str) -> dict:
    with _LOCK:
        d = _RECORDS.setdefault(op, {})
        d = d.setdefault(side, {})
        return d.setdefault(action, defaultdict(int))


def _record(side: str, action: str, key) -> None:
    if not _ON:
        return
    try:
        k = str(key)
    except Exception:                              # noqa: BLE001
        k = repr(key)
    _bucket(_current(), side, action)[k] += 1


class CtxProbe(dict):
    SIDE = "ctx"

    def __getitem__(self, k):
        _record(self.SIDE, "read", k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        # 命中才是真读；未命中只是"探测"（optional 依赖，常见于 `for k in (...): st.get(k)`）。
        # 旧版把 miss 也记成 read → 报告里出现 `k` 这种假依赖。
        _record(self.SIDE, "read" if dict.__contains__(self, k) else "probe", k)
        return super().get(k, default)

    def __contains__(self, k):
        _record(self.SIDE, "probe", k)
        return super().__contains__(k)

    def __setitem__(self, k, v):
        _record(self.SIDE, "write", k)
        return super().__setitem__(k, v)

    def pop(self, k, *a):
        _record(self.SIDE, "pop", k)
        return super().pop(k, *a)

    def setdefault(self, k, default=None):
        _record(self.SIDE, "write", k)
        return super().setdefault(k, default)

    def update(self, *args, **kw):
        for k in dict(*args, **kw).keys():
            _record(self.SIDE, "write", k)
        return super().update(*args, **kw)


class StProbe(CtxProbe):
    SIDE = "st"


def wrap_ctx(ctx):
    if not _ON or isinstance(ctx, CtxProbe):
        return ctx
    p = CtxProbe()
    dict.update(p, ctx)                            # 不经过重写的 update（不计数）
    return p


def wrap_st(st):
    if not _ON or isinstance(st, CtxProbe):
        return st
    p = StProbe()
    dict.update(p, st)
    return p


# ── 报告 ────────────────────────────────────────────────────────────
def snapshot() -> dict:
    with _LOCK:
        return json.loads(json.dumps(_RECORDS))


def flush(path: str | None = None) -> None:
    if not _ON:
        return
    path = path or os.environ.get("DS41M_CTX_TRACE_OUT", "")
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot(), f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except Exception as e:                         # noqa: BLE001
        print(f"  [ctx-trace] 写报告失败：{type(e).__name__}: {e}", flush=True)


def summary(limit: int = 12) -> str:
    """一行式摘要：记录到的 op 数 / 键数。"""
    with _LOCK:
        ops = len(_RECORDS)
        keys = set()
        for d in _RECORDS.values():
            for side in d.values():
                for act in side.values():
                    keys |= set(act)
    return f"ctx-trace：{ops} 个 op / {keys and len(keys)} 个键"


def _canon_counts(d: dict) -> dict:
    out = defaultdict(int)
    for raw, n in d.items():
        out[names.canonical(raw)] += n
    return dict(out)


def compare(report: dict, iface_of=None) -> dict:
    """实测读写 vs OP_IFACE 声明 → 漏声明 / 未观测 / 动态键。"""
    from modules.op_iface import OP_IFACE
    iface_of = iface_of or (lambda op: OP_IFACE[op])
    out = {}
    for op, sides in sorted(report.items()):
        iface = iface_of(op) if op in OP_IFACE else None
        if iface is None:
            out[op] = dict(no_iface=True)
            continue
        req = {k.name for k in iface.requires}
        prov = {k.name for k in iface.provides}
        obs_read = set()
        obs_probe = set()
        obs_write = set()
        for side in ("ctx", "st"):
            d = sides.get(side, {})
            obs_read |= set(_canon_counts(d.get("read", {})).keys())
            # 命中不了的 .get / in 检查 = optional 探测，不与真实读混在一起
            obs_probe |= set(_canon_counts(d.get("probe", {})).keys())
            obs_probe |= set(_canon_counts(d.get("contains", {})).keys())
            obs_write |= set(_canon_counts(d.get("write", {})).keys())
            obs_write |= set(_canon_counts(d.get("pop", {})).keys())
        dynamic = {k for k in (obs_read | obs_probe | obs_write) if k.startswith("_")}
        obs_read -= dynamic
        obs_probe -= dynamic
        obs_write -= dynamic
        out[op] = dict(
            # 读了但既没声明 require 也不是自己 provide 的键
            read_not_declared=sorted(obs_read - req - prov),
            # 探测了（optional/不存在）但没声明的键：可声明为 optional
            probe_not_declared=sorted(obs_probe - obs_read - req - prov),
            # 声明了 require 但实测没读到（条件分支/帮助函数）
            declared_not_observed=sorted(req - obs_read - obs_probe),
            # 写了但没声明 provide（顺带写 require 的键不算）
            write_not_declared=sorted(obs_write - prov - req),
            declared_not_written=sorted(prov - obs_write),
            dynamic=sorted(dynamic),
        )
    return out


def self_test() -> None:
    """纯 CPU 自检：记录/规范化/别名/compare。"""
    global _ON
    was = _ON
    _ON = True
    try:
        with _LOCK:
            _RECORDS.clear()
        ctx = wrap_ctx({"T": 8, "H": 5120, "cu_seqlens": [0, 8]})
        st = wrap_st({"x": 1, "residual": 2, "_shared": {}})
        set_current("wq_a")
        assert ctx["T"] == 8 and ctx.get("H") == 5120
        assert st["x"] == 1
        st["qr"] = 3
        ctx["n_seqs"] = 1
        set_current("q_norm")
        assert st.get("qr") == 3
        _ = st["x"]                     # q_norm 没声明 hidden.x → 应报“读漏”
        ctx.pop("n_seqs", None)
        r = snapshot()
        assert r["wq_a"]["ctx"]["read"]["T"] == 1
        assert r["wq_a"]["st"]["write"]["qr"] == 1
        assert r["q_norm"]["ctx"]["pop"]["n_seqs"] == 1
        # dict 子类底层真实持有数据（dict(ctx) 不能为空）
        assert dict(ctx)["T"] == 8
        c = compare(r)
        # wq_a 没声明 T（几何行数）→ 读漏；q_norm 读了未声明的 hidden.x
        assert "geometry.rows" in c["wq_a"]["read_not_declared"]
        assert "attn.qr" not in c["q_norm"]["read_not_declared"]      # 已声明提供/消费
        assert "hidden.x" in c["q_norm"]["read_not_declared"]
    finally:
        with _LOCK:
            _RECORDS.clear()
        _ON = was


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("report", help="DS41M_CTX_TRACE_OUT 写出的 JSON")
    ap.add_argument("--full", action="store_true", help="打印全部 op")
    a = ap.parse_args(argv)
    with open(a.report, encoding="utf-8") as f:
        report = json.load(f)
    cmp = compare(report)
    bad = 0
    for op, d in cmp.items():
        if d.get("no_iface"):
            print(f"{op:<18} (OP_IFACE 无此算子)")
            continue
        hot = (d["read_not_declared"] or d["write_not_declared"]
               or d["probe_not_declared"]
               or d["declared_not_observed"] or d["declared_not_written"])
        if not hot and not a.full:
            continue
        bad += bool(d["read_not_declared"] or d["write_not_declared"])
        print(f"{op:<18} 读漏={d['read_not_declared']} 写漏={d['write_not_declared']}"
              f" 探测={d['probe_not_declared']}"
              f" 未观测(require)={d['declared_not_observed']}"
              f" 未观测(provide)={d['declared_not_written']}"
              f" 动态={len(d['dynamic'])}")
    print(f"\n{len(cmp)} 个算子；漏声明 {bad} 个（动态/内部键已单列）")


if __name__ == "__main__":
    raise SystemExit(main())

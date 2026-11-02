"""点位边界的内存生命周期：**统一释放** + **归因审计**（2026-10-29）。

背景（099 实测，`tmp/COLLECT_CP2CP4_1PROC_1029_0447`）：
一个进程连跑 205 个点位时，point 192（L=256k、T=2761，纯 forward 只要 ~2 GiB）
的峰值已经 **66.9 GiB**；同量级点位在**全新进程**里只要 15–34 GiB。OOM 现场是
`84.16 GiB already allocated / 84.16 GiB current active / 197 MiB free` ——
`allocated == active` 说明**不是分配器碎片**，是 65 GiB **活张量**被进程级容器钉住。

点间的 `gc.collect()+torch.npu.empty_cache()`（`points_runner._free_device`）只能
回收**没人引用**的块；被模块级 dict/list/set 或 vendor 绑定钉住的张量必须**显式**放掉。

本模块提供两件事（PERF_RULES §0：性能开关默认开；诊断默认关）：

1. **登记表 —— 统一释放**
   每个模块把自己「随点位几何（T/L/cut/池地址/图）变化」的容器写进模块级
   `POINT_CACHES = (容器, ...)`，需要额外动作的再定义 `POINT_RELEASE()`。
   `sweep()` 遍历 `sys.modules` 统一清。**不再手改 `points_runner` 的清表** ——
   那正是以前漏一处就 OOM 的原因（新增缓存没人登记 → 跨点位累积）。

   注意纪律：**权重/布局类缓存不许登记**（`_PACK_CACHE*`/`_PACKED_LAYER_CACHE`/
   `_NZ_CACHE`/`_PREP_L`/`_PREP_AC_L`/`DECODE_CACHE`/`_WOA_PLAN`/`_DG_CONST`/engram
   表窗口…）——它们与几何无关，跨点位复用正是「一次预热跑多点位」的收益来源。

2. **审计 —— 点名**
   `report()` 在点位边界打印 `allocated` 相对基线的**留存**，并按
   「storage 形状」与「哪个模块容器」归因。留存超阈值时自动打印 top holder ——
   以后谁加了没登记的几何缓存，日志直接点名，而不是跑 200 点后 OOM。

3. **兜底 —— 未登记的容器**
   `sweep_unknown()`：没登记、但本点位期间**长超过 2 GiB 的 npu 张量**的模块级容器，
   默认**清掉并打印容器名**（`RESIDENT` 与已登记的不动）。这样"忘了登记"不再等于
   "静默累积 200 点后 OOM"，而是"慢一点 + 日志里点名"；看到点名行就去补
   `POINT_CACHES`。

配套工具：
   `tools/selfcheck_memlife.py`   CPU 自检以上 4 条机制（秒级，不碰卡）
   `tools/check_point_caches.py`  静态扫"装了设备张量却没登记"的模块级容器

用法（points_runner 已接好）：
    DS41M_MEM_AUDIT=1             每个点位边界打印一行留存；超阈值打印归因
    DS41M_MEM_AUDIT_EVERY=N       每 N 个点位审计一次（默认 1）
    DS41M_MEM_GUARD_GB=4          ｜留存｜ > 该值时**总是**打印归因（默认 4）
    DS41M_MEM_SWEEP_UNKNOWN=2     未登记容器的兜底清理阈值 GiB（默认 2；0=关）
"""
from __future__ import annotations

import gc
import os
import sys

_PKGS = ("core", "meta", "layout", "ops", "cp", "trace", "weights", "modules")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 归因时忽略的模块级容器（不是缓存，或与显存无关）
_SKIP_ATTRS = {
    "OPS", "SKIP", "SCRIPTS", "PATHS", "CONTAINERS", "OP_IFACE", "PHASE",
    "__builtins__", "__annotations__", "_ENV_SNAP", "_ST",
}

# 与点位几何**无关**的进程级常驻（权重/布局/通信域/事件环…）：**不许**登记，
# 也不参与「未知容器自动清理」。`tools/check_point_caches.py` 复用同一份名单。
RESIDENT = {
    # 权重与布局转换（跨点位复用 = 一次预热跑多点位的收益）
    "_PACK_CACHE", "_PACKED_LAYER_CACHE", "_PACK_CACHE_L", "_PACKED_SUB_CACHE",
    "_STRIP_CACHE", "_NZ_CACHE", "_WOA_PLAN", "_DG_CONST", "_PREP_L",
    "_PREP_AC_L", "_PREP_GIDS", "_AC_RELEASED", "_DG_RELEASED", "DECODE_CACHE",
    "_ENGRAM_ROWS_CACHE", "_EXPERT_CACHE_CLEARERS", "_RECIPES_CACHE",
    # 通信域 / 对称缓冲 / engram 表窗口
    "_CP_GROUPS", "_VENDOR_BUFS", "_VENDOR_GROUPS", "_VENDOR_FAILED", "_BUFS",
    "_SB", "_MC2", "_BACKEND_LAST", "_MGR", "_FAILED",
    # 事件环（只放 Event，不放张量）与统计表
    "_DF_RING", "_DF_POS",
    "_DF_ARMED", "_AG_STATS", "_SUB", "_STATS", "_N", "_QD_N", "_BOUNDS_SEEN",
    # 注册表
    "OPS", "SKIP", "OP_IFACE", "CONTAINERS", "PATHS", "_IMPL", "_PKT", "_ST",
    "_FP_MEMO", "_LAST", "_DUMP_N", "_ENV_SNAP", "PHASE", "META", "_CS", "_CS_STEP",
}


def _our_modules():
    """本树的已导入模块。

    ★ 不能按**模块名**过滤：`run.py` 把 `core/ops/layout/…` 逐个塞进 `sys.path`，
      这些模块是**裸名**（`exec_ops` / `runner` / `cp_runtime`…），`name.split('.')[0]`
      永远不在包名集合里 —— 以前按名字过滤会一个都扫不到（sweep 静默变成空操作）。
      改成按 `__file__` 的**所在目录**判归属。
    """
    root = _ROOT + os.sep
    for name, m in list(sys.modules.items()):
        if m is None:
            continue
        f = getattr(m, "__file__", None) or ""
        if not f.startswith(root):
            continue
        rel = os.path.relpath(os.path.dirname(os.path.abspath(f)), _ROOT)
        if rel.split(os.sep)[0] in _PKGS:
            yield name, m


# ───────────────────────── 统计工具 ─────────────────────────
def _is_npu_tensor(o) -> bool:
    t = type(o)
    if t.__name__ != "Tensor" or t.__module__ != "torch":
        return False
    d = getattr(o, "device", None)
    return d is not None and getattr(d, "type", "") == "npu"


def _storage_key(t):
    try:
        s = t.untyped_storage()
        return (int(s.data_ptr()), int(s.nbytes()))
    except Exception:                                    # noqa: BLE001
        return (int(t.data_ptr()), int(t.numel()) * int(t.element_size()))


def _bytes_of(obj, max_items: int = 200_000):
    """粗略统计容器里（浅层 + 一层嵌套）npu 张量的**去重 storage** 字节。

    只用来排序「谁最大」，不要求精确到字节。
    """
    seen: dict = {}
    n = 0
    stack = [(obj, 0)]
    while stack:
        cur, depth = stack.pop()
        n += 1
        if n > max_items:
            break
        if _is_npu_tensor(cur):
            k, nb = _storage_key(cur)
            seen[k] = nb
            continue
        if depth >= 3:
            continue
        if isinstance(cur, dict):
            stack.extend((v, depth + 1) for v in cur.values())
        elif isinstance(cur, (list, tuple, set, frozenset)):
            stack.extend((v, depth + 1) for v in cur)
    return sum(seen.values())


def container_snapshot() -> dict:
    """{ "模块.属性": bytes }：本树所有模块级容器里钉住的 npu 张量（去重 storage）。"""
    out: dict = {}
    for name, m in _our_modules():
        for attr, val in list(vars(m).items()):
            if attr.startswith("__") or attr in _SKIP_ATTRS:
                continue
            if not isinstance(val, (dict, list, set, frozenset)):
                continue
            try:
                if len(val) > 200_000:
                    continue
            except Exception:                            # noqa: BLE001
                continue
            nb = _bytes_of(val)
            if nb > 0:
                out[f"{name}.{attr}"] = nb
    return out


def container_holders(top: int = 8, base: dict | None = None):
    """[(bytes, "模块.属性"), …]：按**增长量**排序（给了 `base` 时就比 baseline）。"""
    now = container_snapshot()
    if base:
        rows = [(nb - base.get(k, 0), nb, k) for k, nb in now.items()]
        rows = [r for r in rows if r[0] > 0]
        rows.sort(reverse=True)
        return [(d, f"{k}（现在 {n / 2**30:.2f}G）") for d, n, k in rows[:top]]
    rows = sorted(((nb, k) for k, nb in now.items()), reverse=True)
    return rows[:top]


# 第一次 report() 时的容器快照 = 「全新进程加载完权重」的基线；之后按增长量归因。
_CONTAINER_BASE: dict = {}


def tensor_census(top: int = 10):
    """[(bytes, "shape×count"), …]：进程内**全部**活着的 npu 张量按形状聚合（去重 storage）。"""
    by_shape: dict = {}
    seen: dict = {}
    for o in gc.get_objects():
        if not _is_npu_tensor(o):
            continue
        k, nb = _storage_key(o)
        if k in seen:
            continue
        seen[k] = nb
        shp = tuple(int(d) for d in o.shape)
        c = by_shape.get(shp)
        if c is None:
            c = by_shape[shp] = [0, 0]
        c[0] += nb
        c[1] += 1
    rows = sorted(((b, f"{s} ×{n}") for s, (b, n) in by_shape.items()), reverse=True)
    return rows[:top]


def live_bytes() -> float:
    """全部活着的 npu 张量字节（去重 storage）；拿不到就返回 nan。"""
    try:
        return float(sum(_storage_key(o)[1] for o in _all_tensors()))
    except Exception:                                    # noqa: BLE001
        return float("nan")


def _all_tensors():
    for o in gc.get_objects():
        if _is_npu_tensor(o):
            yield o


# ───────────────────────── 释放 ─────────────────────────
def sweep(verbose: bool = False, measure: bool = False):
    """清掉**所有登记过的**几何相关容器 + 调 `POINT_RELEASE()`。返回 [(bytes, name)]。

    `measure=False`（默认）不做字节统计 —— 热点路径上省一次容器遍历。
    """
    freed = []
    for name, m in _our_modules():
        caches = getattr(m, "POINT_CACHES", None)
        if caches:
            for c in caches:
                try:
                    nb = _bytes_of(c) if measure else 0
                    c.clear()
                    if nb:
                        freed.append((nb, name))
                except Exception as exc:                 # noqa: BLE001
                    print(f"  [memlife] ⚠️ 清 {name} 失败：{type(exc).__name__}: {exc}",
                          flush=True)
        rel = getattr(m, "POINT_RELEASE", None)
        if callable(rel):
            try:
                rel()
            except Exception as exc:                     # noqa: BLE001
                print(f"  [memlife] ⚠️ {name}.POINT_RELEASE 失败："
                      f"{type(exc).__name__}: {exc}", flush=True)
    freed.sort(reverse=True)
    if verbose and freed:
        _p("  [memlife] 释放：" + "  ".join(f"{n}={b / 2**30:.2f}G" for b, n in freed[:8]),
           flush=True)
    return freed


# ───────────────────────── 审计 ─────────────────────────
def _env_gb(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, "") or default)
    except Exception:                                    # noqa: BLE001
        return default


def _p(*a, **kw):
    print(*a, **kw)


def report(tag: str = "", base: float | None = None, prev: float | None = None,
           guard_gb: float | None = None, top: int = 8) -> dict:
    """打印一行留存 + （超阈值时）归因。返回 {alloc, reserved, retain, delta, holders}。"""
    try:
        import torch
        alloc = torch.npu.memory_allocated() / 2 ** 30
        reserved = torch.npu.memory_reserved() / 2 ** 30
    except Exception:                                    # noqa: BLE001
        return {}
    gh = _env_gb("DS41M_MEM_GUARD_GB", 4.0) if guard_gb is None else guard_gb
    retain = alloc - base if base is not None else float("nan")
    delta = alloc - prev if prev is not None else float("nan")
    msg = (f"  [mem] {tag} allocated={alloc:.2f} reserved={reserved:.2f}"
           + (f" 留存={retain:+.2f}" if base is not None else "")
           + (f" Δ={delta:+.2f}" if prev is not None else "") + " GiB")
    _p(msg, flush=True)
    over = (retain == retain and abs(retain) > gh)      # NaN 安全
    out = {"alloc": alloc, "reserved": reserved, "retain": retain, "delta": delta}
    if over:
        if not _CONTAINER_BASE:
            _CONTAINER_BASE.update(container_snapshot())
        hold = container_holders(top, base=_CONTAINER_BASE)
        census = tensor_census(top)
        _p(f"       ⚠️ 留存 {retain:+.2f} GiB > {gh:.1f} GiB → 归因（top {top}，"
           f"容器按相对首次审计的增长量）：", flush=True)
        for nb, name in hold:
            _p(f"         {nb / 2**30:+8.2f}G  容器 {name}", flush=True)
        for nb, desc in census:
            _p(f"         {nb / 2**30:8.2f}G  形状 {desc}", flush=True)
        out["holders"] = hold
        out["census"] = census
    return out


def _ensure_base():
    if not _CONTAINER_BASE:
        _CONTAINER_BASE.update(container_snapshot())


def unknown_guard_gb() -> float:
    """兜底清理的阈值（GiB）。**默认 2**；`DS41M_MEM_SWEEP_UNKNOWN=0` 关闭。"""
    v = os.environ.get("DS41M_MEM_SWEEP_UNKNOWN", "").strip().lower()
    if v in ("0", "off", "false", "no"):
        return 0.0
    if not v:
        return 2.0
    try:
        return float(v)
    except Exception:                                    # noqa: BLE001
        return 2.0


def sweep_unknown(guard_gb: float = 1.0):
    """兜底（`DS41M_MEM_SWEEP_UNKNOWN=<GB>`，默认关）：清掉**没登记**、但在点位期间
    增长超过阈值的模块级容器。

    用途：新加的几何缓存忘了登记时，先靠它止血 + 看日志点名，再补 `POINT_CACHES`。
    安全边界：只清 dict/list/set；`RESIDENT` 与 `POINT_CACHES` 里的**不动**；
    只清「本点位期间增长 > 阈值且装着 npu 张量」的 ⇒ 权重/注册表不会被误伤。
    """
    _ensure_base()
    freed = []
    for name, m in _our_modules():
        declared = {id(c) for c in (getattr(m, "POINT_CACHES", None) or ())}
        for attr, val in list(vars(m).items()):
            if attr.startswith("__") or attr in RESIDENT or attr in _SKIP_ATTRS:
                continue
            if not isinstance(val, (dict, list, set)):
                continue
            if id(val) in declared:
                continue
            key = f"{name}.{attr}"
            try:
                nb = _bytes_of(val)
            except Exception:                            # noqa: BLE001
                continue
            grow = nb - _CONTAINER_BASE.get(key, 0)
            if grow <= guard_gb * 2 ** 30:
                continue
            try:
                val.clear()
                _CONTAINER_BASE[key] = 0
                freed.append((grow, key))
            except Exception:                            # noqa: BLE001
                pass
    if freed:
        freed.sort(reverse=True)
        _p("  [memlife] ⚠️ 未知容器兜底清理（未登记！请补进该模块的 POINT_CACHES）："
           + "  ".join(f"{k}={b / 2**30:.2f}G" for b, k in freed[:8]), flush=True)
    return freed


def audit_on() -> bool:
    return os.environ.get("DS41M_MEM_AUDIT", "").strip().lower() in (
        "1", "on", "true", "yes")


def audit_every() -> int:
    try:
        return max(1, int(os.environ.get("DS41M_MEM_AUDIT_EVERY", "1") or 1))
    except Exception:                                    # noqa: BLE001
        return 1

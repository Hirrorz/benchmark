"""模块注册表：每个 op 一个 Module 类（由 OP_IFACE 生成，接口唯一事实源）。

Phase 1 只声明、不执行；`execute` 在 Phase 2 才被调用（惰性 import exec_ops，
所以本模块可以在纯 CPU 导出器里安全导入）。
"""
from __future__ import annotations

import re
from typing import Iterable

MODULES: dict[str, type] = {}


def _camel(op: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in re.split(r"[_@]+", op) if p)


class LeafModule:
    """所有 op 模块的基类。类属性 = 接口；实例 = 一次调用（path/occurrence）。"""

    OP: str = ""
    IFACE = None                     # modules.spec.OpInterface
    PATH: str | None = None          # 装配时填（层内规范路径）
    KIND = "leaf"

    def __init__(self, path: str | None = None, occurrence: int = 0):
        self.path = path or self.PATH or self.OP
        self.occurrence = int(occurrence)

    # ── Phase 2：执行（惰性 import，Phase 1 不会被调用）────────────────
    def execute(self, st, ctx, w):
        if self.OP == "boundary_cut":
            raise RuntimeError(
                "boundary_cut 由 runner 特判（runner.py:1846），不是 OPS 函数；"
                "模块运行时请调用 runner 的边界切分钩子")
        from exec_ops import OPS           # noqa: PLC0415 —— 故意惰性
        fn = OPS.get(self.OP)
        if fn is None:
            raise RuntimeError(f"算子 {self.OP!r} 不在 exec_ops.OPS（no_impl）")
        return fn(st, ctx, w)

    def to_json(self) -> dict:
        d = dict(op=self.OP, path=self.path, kind=self.KIND)
        d.update(self.IFACE.to_json())
        return d

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.path}>"


def register_module(cls: type) -> type:
    MODULES[cls.OP] = cls
    return cls


def define_module(op: str, iface) -> type:
    """按 OP_IFACE 生成一个 Module 类（每个 op 一个类）。"""
    name = _camel(op)
    cls = type(name, (LeafModule,), {
        "OP": op,
        "IFACE": iface,
        "__doc__": iface.doc or f"{op}（kernel={getattr(iface, 'impl', None)}）",
        "__module__": "modules.ops",
    })
    return register_module(cls)


def register_group(op_names: Iterable[str]) -> list[type]:
    """把一组 op 的接口表条目实例化成 Module 类。"""
    from modules.op_iface import OP_IFACE          # noqa: PLC0415
    out = []
    for op in op_names:
        iface = OP_IFACE.get(op)
        if iface is None:
            raise KeyError(f"OP_IFACE 缺少 {op!r}（先补接口再注册）")
        out.append(define_module(op, iface))
    return out


def load_plugins(names=None) -> None:
    """Phase 2：按 `DS41M_MODULE_PLUGINS=a.b,c.d` 加载外部模块。"""
    import importlib
    import os
    if names is None:
        raw = os.environ.get("DS41M_MODULE_PLUGINS", "")
        names = [x.strip() for x in raw.split(",") if x.strip()]
    for n in names:
        importlib.import_module(n)

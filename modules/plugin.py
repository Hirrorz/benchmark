"""插件开发 API：**模块类即事实源**。

用法（推荐用 `python3.11 tools/new_module.py` 生成骨架）：

    from modules.plugin import K, W, module
    from modules.registry import LeafModule

    @module(op="my_op", category="attention", container="attn.core",
            after=("mmsmla",),
            requires=(K("hidden.x", "Tensor", "[T,H]"),),
            provides=(K("attn.my_out", "Tensor", "[T,H]"),),
            weights=(W("attn.my_op.weight"),),
            impl="exec_ops:_my_op", kernel="MyKernel", doc="一句话说明")
    class MyOp(LeafModule):
        def execute(self, st, ctx, w):
            from exec_ops import OPS
            return OPS[self.OP](st, ctx, w)

注册后：
  · `OP_IFACE` / `MODULES` 立即出现该算子（导出器/门禁/查看器都能看到）；
  · 把 op 名加进 `recipes.yaml` 的对应段，就会被装配进层树；
  · `modules/export_model_html.py` 重新导出后，HTML 的结构/积木库/蓝图都会更新。
"""
from __future__ import annotations

from modules.op_iface import K, OP_IFACE, W, register_op
from modules.registry import LeafModule, register_module
from modules.spec import OpInterface

__all__ = ["module", "K", "W", "OP_IFACE"]


def module(op, *, category="misc", requires=(), provides=(), weights=(),
           impl=None, source=None, flags=(), container=None, after=(),
           dispatch=None, rows_phase="same", doc="", kernel=None, calls=1,
           intrinsics=()):
    """把一个 Module 类登记为插件（接口声明写在这里，类只负责执行）。"""
    def deco(cls):
        iface = OpInterface(
            op=op, impl=impl, source=source,
            requires=tuple(requires), provides=tuple(provides),
            weights=tuple(weights), flags=frozenset(flags), calls=int(calls),
            dispatch=dispatch, rows_phase=rows_phase,
            doc=doc or (cls.__doc__ or "").strip(),
            intrinsics=tuple(intrinsics), category=category,
            container=container, after=tuple(after), kernel=kernel,
            module_cls=cls.__name__,
        )
        iface = register_op(iface)
        cls.OP = op
        cls.IFACE = iface
        cls.PATH = container
        # 插件类必须叫得出来：registry 用类名展示
        if not issubclass(cls, LeafModule):
            raise TypeError(f"{cls.__name__} 必须继承 LeafModule")
        register_module(cls)
        return cls
    return deco


def load_plugins() -> list[str]:
    """显式加载 `modules/ops/plugins/`（导入 modules.ops 时会自动做一次）。"""
    import importlib
    import modules.ops.plugins as pkg
    import pkgutil
    out = []
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name.startswith("_"):
            continue
        importlib.import_module(f"{pkg.__name__}.{m.name}")
        out.append(m.name)
    return out


def registered_plugins() -> list[dict]:
    """积木库视图：全部已注册插件（含未被配方使用的）。"""
    return [
        dict(op=i.op, category=i.category, module=i.module_cls or "",
             kernel=i.kernel, impl=i.impl, container=i.container,
             after=list(i.after), flags=sorted(i.flags),
             requires=len(i.requires), provides=len(i.provides),
             weights=len(i.weights), doc=i.doc)
        for i in sorted(OP_IFACE.values(), key=lambda x: (x.category, x.op))
    ]

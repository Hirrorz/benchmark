"""用户插件目录：这里的每个 .py 会在 `import modules.ops` 时自动加载。

用 `python3.11 tools/new_module.py` 生成骨架；文件里用
`@module(op=..., requires=..., provides=..., ...)` 声明接口。
"""
import importlib
import pkgutil


def load_all() -> list[str]:
    out = []
    for m in pkgutil.iter_modules(__path__):
        if m.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{m.name}")
        out.append(m.name)
    return out


load_all()

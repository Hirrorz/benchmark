"""接口声明快捷构造（K/W）——配置加载器与 Python 插件共用。

这些函数只做 dataclass 构造，不含任何具体模块定义；模块定义在
`modules/specs/model_modules.yaml`（唯一事实源）。
"""
from __future__ import annotations

from modules import names
from modules.spec import KeySpec, WeightRef

_ST_NS = {"hidden", "attn", "cmp", "idx", "moe", "engram", "hc"}


def K(name, type="int", shape=None, storage=None, tier=None, **kw):
    """KeySpec 快捷构造：storage/tier 默认按命名空间推断。"""
    if storage is None:
        ns = names.namespace(name)
        storage = "cache" if ns == "cache" else ("external" if ns == "external"
                                                 else ("st" if ns in _ST_NS else "ctx"))
    if tier is None:
        tier = names.default_tier(name)
    return KeySpec(name, type=type, shape=shape, storage=storage, tier=tier,
                   legacy=tuple(k for k, v in names.LEGACY.items() if v == name), **kw)


def W(suffix, via="raw", packed=None, cond=None, role=None, doc=""):
    if role is None:
        role = "read_packed" if via in ("_packed", "synthetic") else "read"
    return WeightRef(suffix=suffix, via=via, packed_key=packed, condition=cond,
                     role=role, doc=doc)

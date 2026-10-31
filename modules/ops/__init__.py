"""op 模块类（每个 op 一个类，由 OP_IFACE 生成）。

按语义组拆文件只是组织手段；**接口唯一事实源**是 `modules/op_iface.py`。
导入本包即完成 `modules.registry.MODULES` 的注册。
"""
from modules.ops import common, attention, compressor, indexer, boundary, moe, engram  # noqa: F401
from modules.ops import plugins  # noqa: F401 —— 自动加载用户插件（可能新增 op）

from modules.op_iface import OP_IFACE                       # noqa: E402
from modules.registry import MODULES, define_module         # noqa: E402

# 配置驱动的兜底：YAML / modules.d / 插件里新增、但还没归到上面任何分组的 op，
# 也自动生成 Module 类（否则 len(MODULES) != len(OP_IFACE)，门禁必红）。
for _op, _iface in OP_IFACE.items():
    if _op not in MODULES:
        define_module(_op, _iface)

# 同步 opgraph 的默认条目：新模块进 recipes 时不必先改 opgraph.py。
try:
    import opgraph as _opgraph
    _opgraph.refresh_defaults()
except ImportError:      # 单独拷走 modules/ 的场景：没有顶层 opgraph 也能导入
    pass

__all__ = ["common", "attention", "compressor", "indexer", "boundary", "moe", "engram"]

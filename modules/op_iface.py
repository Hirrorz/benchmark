"""算子接口表——**配置驱动**（唯一事实源 = `modules/specs/model_modules.yaml`）。

2026-10-25 用户要求：模块定义与其所需参数不在 Python 里硬编码，改为配置文件解析；
父节点不携带子节点所需 I/O，子节点缺失的键沿祖先作用域解析（见 `modules.context`）。

本模块只保留：
  · `OP_IFACE` 注册表（从 YAML 加载，Python 插件可继续 `register_op` 追加）；
  · `K/W/register_op` 兼容 API（`modules.keys` 的再导出）。
如 YAML 缺失，才回退到 `modules/op_iface_builtin.py`（历史 Python 表，仅供迁移兜底）。
"""
from __future__ import annotations

import os
import sys

from modules.keys import K, W
from modules.spec import OpInterface
from modules import spec_loader

__all__ = ["OP_IFACE", "K", "W", "register_op", "_add", "SPEC_PATH"]

SPEC_PATH = os.environ.get("DS41M_MODULE_SPEC") or spec_loader.DEFAULT_SPEC

OP_IFACE: dict[str, OpInterface] = {}


def _add(iface: OpInterface) -> None:
    if iface.op in OP_IFACE:
        raise KeyError(f"OP_IFACE 重复登记 {iface.op!r}")
    OP_IFACE[iface.op] = iface


def register_op(iface: OpInterface) -> OpInterface:
    """插件注册入口（`modules.plugin.module` 装饰器调用）。"""
    _add(iface)
    return iface


def _load() -> None:
    strict = os.environ.get("DS41M_SPEC_STRICT", "1").strip().lower() not in (
        "0", "off", "false", "no")
    if SPEC_PATH and os.path.exists(SPEC_PATH):
        try:
            for iface in spec_loader.load_spec_file(SPEC_PATH):
                _add(iface)
            return
        except Exception as exc:                         # noqa: BLE001
            if strict:
                raise
            print(f"  ⚠️ [module-spec] {SPEC_PATH} 解析失败，回退 Python 表："
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    else:
        print(f"  ⚠️ [module-spec] 未找到 {SPEC_PATH}，回退 Python 表"
              f"（建议用 DS41M_MODULE_SPEC 指向配置）", file=sys.stderr, flush=True)

    from modules import op_iface_builtin as _b               # 迁移兜底
    for iface in _b.OP_IFACE.values():
        _add(iface)


_load()

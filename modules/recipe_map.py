"""配方段 → 容器路径的 sidecar——**配置驱动**。

2026-10-25 用户要求：模块/参数/装配关系都应通过配置文件解析，开发者改配置而不是改代码。
唯一事实源 = `modules/specs/recipe_map.yaml`（可用 `DS41M_RECIPE_MAP` 覆盖）；
文件缺失时才回退 `modules/recipe_map_builtin.py`（历史 Python 表，迁移兜底）。

`recipes.yaml` 仍只写"跑哪些算子"；本文件补"算子属于哪个容器/路径"。父容器不需要
汇总子节点的 I/O——子节点只声明自己的 requires，解析时沿祖先作用域查找。
"""
from __future__ import annotations

import os
import sys

from modules.spec import ContainerInfo

SCHEMA = "ds41_mini/recipe-map@1"
DEFAULT_SPEC = os.path.join(os.path.dirname(__file__), "specs", "recipe_map.yaml")
SPEC_PATH = os.environ.get("DS41M_RECIPE_MAP") or DEFAULT_SPEC

CONTAINERS: dict[str, ContainerInfo] = {}
PATHS: dict[tuple[str, str, int], str] = {}


def path_of(segment: str, op: str, occurrence: int) -> str | None:
    return PATHS.get((segment, op, occurrence))


def _err(path, msg):
    raise ValueError(f"{path}: {msg}")


def _load_yaml(path: str) -> None:
    import yaml
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        _err(path, "顶层必须是映射")
    if doc.get("schema") != SCHEMA:
        _err(path, f"schema={doc.get('schema')!r}，期望 {SCHEMA!r}")
    for c in doc.get("containers") or ():
        p = str(c["path"])
        if p in CONTAINERS:
            _err(path, f"重复容器 {p!r}")
        CONTAINERS[p] = ContainerInfo(p, str(c.get("title") or p),
                                      str(c.get("doc") or ""), int(c.get("order", 0)))
    for r in doc.get("paths") or ():
        key = (str(r["segment"]), str(r["op"]), int(r.get("occurrence", 0)))
        if key in PATHS:
            _err(path, f"重复路径映射 {key!r}")
        PATHS[key] = str(r["path"])


def _load_builtin() -> None:
    from modules import recipe_map_builtin as b
    CONTAINERS.update(b.CONTAINERS)
    PATHS.update(b.PATHS)


def _load() -> None:
    strict = os.environ.get("DS41M_SPEC_STRICT", "1").strip().lower() not in (
        "0", "off", "false", "no")
    if SPEC_PATH and os.path.exists(SPEC_PATH):
        try:
            _load_yaml(SPEC_PATH)
            return
        except Exception as exc:                         # noqa: BLE001
            if strict:
                raise
            print(f"  ⚠️ [recipe-map] {SPEC_PATH} 解析失败，回退 Python 表："
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    else:
        print(f"  ⚠️ [recipe-map] 未找到 {SPEC_PATH}，回退 Python 表",
              file=sys.stderr, flush=True)
    _load_builtin()


_load()

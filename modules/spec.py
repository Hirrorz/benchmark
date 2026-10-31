"""模块规格的数据结构（纯 dataclass，无依赖）。

术语：
  · KeySpec     —— 上下文接口的一条键（谁需要 / 谁提供）；
  · WeightRef   —— 权重读取（raw / `_packed` / 运行期合成）；
  · OpInterface —— 一个算子（= 一个 Module 类）的完整接口声明；
  · Step        —— 配方展开后的一次算子调用（含容器路径与出现序号）；
  · LayerSpec   —— 一层（40 层里的一个）。

`storage` 说明键"活在哪里"：
  ctx      —— runner 的扁平 ctx（常量/几何/批次/层角色/CP…）；
  st       —— 层内激活字典；
  cache    —— `st["_shared"]`（跨层池/行账）；
  external —— 进程级全局（`exec_ops.CP_STATE` 等）；
  global   —— 模块/进程常量（不是运行时数据）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Tier = Literal["const", "pass", "layer", "container", "module", "cache", "external"]
Storage = Literal["ctx", "st", "cache", "external", "global"]
Shadow = Literal["single_writer", "stream", "allow_multi", "in_place", "hidden"]
Mutation = Literal["new", "replace", "in_place"]
Via = Literal["raw", "_packed", "synthetic"]


@dataclass(frozen=True)
class KeySpec:
    """上下文接口的一条键。

    shape 是**符号式**（如 `[T,H]`），由 `modules.shapes` 按几何预设求值；
    非 Tensor 键可留空。`condition` 用简单表达式描述生效条件（`ratio>0`、
    `is_candidate`、`engram`、`cp.active`、`packed`…），导出器不求值，只展示。
    """
    name: str
    type: str = "int"
    shape: str | None = None
    storage: Storage = "ctx"
    tier: Tier = "module"
    optional: bool = False
    mutation: Mutation = "new"
    shadow: Shadow = "single_writer"
    condition: str | None = None
    legacy: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    doc: str = ""

    def to_json(self) -> dict:
        d = dict(key=self.name, type=self.type, storage=self.storage, tier=self.tier,
                 mutation=self.mutation, shadow=self.shadow, optional=self.optional)
        if self.mutation != "new":
            d["mutation"] = self.mutation
        if self.shape:
            d["shape"] = self.shape
        if self.condition:
            d["condition"] = self.condition
        if self.legacy:
            d["legacy"] = list(self.legacy)
        if self.roles:
            d["roles"] = list(self.roles)
        if self.doc:
            d["doc"] = self.doc
        return d


@dataclass(frozen=True)
class WeightRef:
    """权重读取声明。

    suffix   : 权重键（可含 `{E}` 专家模板、`{stage}` 阶段模板）；
    via      : raw（原始权重）/ _packed（加载期打包的派生键）/ synthetic（运行期合成键）；
    packed_key: `_packed` 字典里的真实键（None 时 = suffix）；
    role     : read / read_packed / release_raw（打包后释放 raw 的 fp8 权重）。
    """
    suffix: str
    via: Via = "raw"
    packed_key: str | None = None
    condition: str | None = None
    role: str = "read"
    doc: str = ""

    def to_json(self) -> dict:
        d = dict(suffix=self.suffix, via=self.via, role=self.role)
        if self.packed_key:
            d["packed_key"] = self.packed_key
        if self.condition:
            d["condition"] = self.condition
        if self.doc:
            d["doc"] = self.doc
        return d


@dataclass(frozen=True)
class OpInterface:
    """一个算子 = 一个 Module 类的接口声明。"""
    op: str
    impl: str | None = None                     # "exec_ops:_wq_a" / "runner:_boundary_cut"
    source: tuple[str, int, int] | None = None  # (file, def_line, end_line)
    requires: tuple[KeySpec, ...] = ()
    provides: tuple[KeySpec, ...] = ()
    weights: tuple[WeightRef, ...] = ()
    flags: frozenset[str] = frozenset()         # implicit|packed_variant|...
    calls: int = 1
    dispatch: str | None = None                 # "packed_if:batch.is_packed && !cp.active"
    rows_phase: str = "same"                    # full|compact|same
    doc: str = ""
    intrinsics: tuple[str, ...] = ()            # 融合算子内部的 kernel 列表
    # ── 插件/组装元数据（"积木"的信息）──────────────────────────
    category: str = "misc"                      # attention|compressor|indexer|moe|engram|...
    container: str | None = None                # 默认插入容器（新插件无 recipe_map 时用）
    after: tuple[str, ...] = ()                 # 顺序约束：这些算子应先于本算子（若同层存在）
    kernel: str | None = None                   # 插件自报内核名（opgraph 缺失时用）
    module_cls: str | None = None               # Module 类名
    # 模块可配置参数：字面量或 "${key}" 引用（运行时沿祖先作用域解析）。
    # 例：{"candidate_block_size": 8, "topk": "${model.moe_topk}"}
    params: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = dict(op=self.op, impl=self.impl, requires=[k.to_json() for k in self.requires],
                 provides=[k.to_json() for k in self.provides],
                 weights=[w.to_json() for w in self.weights],
                 flags=sorted(self.flags), calls=self.calls, rows_phase=self.rows_phase)
        if self.source:
            d["source"] = dict(file=self.source[0], line=self.source[1],
                               end_line=self.source[2])
        if self.dispatch:
            d["dispatch"] = self.dispatch
        if self.intrinsics:
            d["intrinsics"] = list(self.intrinsics)
        if self.doc:
            d["doc"] = self.doc
        d["category"] = self.category
        if self.container:
            d["container"] = self.container
        if self.after:
            d["after"] = list(self.after)
        if self.kernel:
            d["kernel"] = self.kernel
        if self.params:
            d["params"] = dict(self.params)
        return d


@dataclass(frozen=True)
class Step:
    """配方展开后的一次算子调用。path = 容器路径（层内规范名）。"""
    op: str
    path: str
    segment: str
    occurrence: int = 0
    rows_phase: str = "same"
    view: str = "default"


@dataclass(frozen=True)
class ContainerInfo:
    """容器的展示信息（标题/说明/顺序）。"""
    path: str
    title: str
    doc: str = ""
    order: int = 0


@dataclass(frozen=True)
class LayerRoles:
    kv_source: bool = False
    index_source: bool = False
    candidate_source: bool = False
    engram: bool = False

    def to_json(self) -> dict:
        return dict(kv_source=self.kv_source, index_source=self.index_source,
                    candidate_source=self.candidate_source, engram=self.engram)


@dataclass(frozen=True)
class LayerSpec:
    idx: int
    kind: str
    ratio: int
    roles: LayerRoles
    engram: bool = False
    template: str = ""
    split: bool = False

    def to_json(self) -> dict:
        return dict(idx=self.idx, kind=self.kind, ratio=self.ratio,
                    roles=self.roles.to_json(), engram=self.engram,
                    template=self.template, split=self.split)


@dataclass
class ModelTree:
    """导出器的中间产物：模型常量 + 层规格 + 每层模块树 + 诊断。"""
    model: dict
    layers: list[LayerSpec] = field(default_factory=list)
    trees: dict = field(default_factory=dict)       # key -> 树（templates 用）
    templates: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    presets: dict = field(default_factory=dict)
    variants: dict = field(default_factory=dict)
    blueprint: dict = field(default_factory=dict)
    builder: dict = field(default_factory=dict)

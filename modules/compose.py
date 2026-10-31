"""把 `recipes.yaml` 展开成"带容器路径的步骤序列"并装配成模块树（Phase 1 纯装配）。

硬不变量：`[s.op for s in steps] == run.expand_layer(cfg, recipes, layer)`。
`tools/check_module_spec.py` 会对 40 层 × 6 种口径逐位断言（见计划）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from modules import recipe_map
from modules.op_iface import OP_IFACE
from modules.spec import Step

_PAD_OPS = {"mul", "add", "cast", "tmove"}


@dataclass
class ComposedLayer:
    idx: int
    kind: str
    view: str
    steps: list[Step] = field(default_factory=list)
    tree: dict = field(default_factory=dict)

    @property
    def op_sequence(self) -> list[str]:
        return [s.op for s in self.steps]


def moe_key(cfg) -> str:
    """与 run.expand_layer 同口径：ascendc/deepgemm → fused，否则 unfused。"""
    backend = (cfg.get("backends") or {}).get("megamoe", "deepgemm")
    return "fused" if backend in ("ascendc", "deepgemm") else "unfused"


def _skip_env(name: str) -> bool:
    import os
    return os.environ.get(name, "").strip().lower() in ("1", "on", "true", "yes")


def expand_tracked(recipes, items, attn_kind, moe_key_, segment, out):
    """recipes 展开（与 run.expand_items 同构），额外记录最内层段名。"""
    for it in items:
        if not it.startswith("_"):
            out.append((it, segment))
            continue
        key = it[1:]
        if key == "attention":
            sub = recipes["attention"][attn_kind]
            seg = f"attention.{attn_kind}"
        elif key == "moe":
            sub = [] if _skip_env("DS41M_SKIP_MOE") else recipes["moe"][moe_key_]
            seg = f"moe.{moe_key_}"
        else:
            sub = recipes[key]
            seg = key
        expand_tracked(recipes, sub, attn_kind, moe_key_, seg, out)


def tracked_sequence(cfg, recipes, layer, view="default") -> list[tuple[str, str]]:
    """→ [(op, segment)]；与 run.expand_layer 的顺序/过滤完全一致。"""
    attn_kind = layer["attn"]
    if view == "split":
        attn_kind = "c1a_source_split"
    out: list[tuple[str, str]] = []
    if layer.get("engram") and not _skip_env("DS41M_SKIP_ENGRAM"):
        expand_tracked(recipes, recipes["engram"], attn_kind, moe_key(cfg), "engram", out)
    expand_tracked(recipes, recipes["common"], attn_kind, moe_key(cfg), "common", out)
    if not _skip_env("DS41M_PAD_OPS"):
        out = [(op, seg) for op, seg in out if op not in _PAD_OPS]
    return out


def _title(path: str) -> str:
    """容器默认标题：末段下划线 → 空格，首字母大写。"""
    return path.rsplit(".", 1)[-1].replace("_", " ").capitalize()


def _new_container(path: str) -> dict:
    info = recipe_map.CONTAINERS.get(path)
    return dict(id=path, name=(info.title if info else _title(path)), kind="container",
                children=[], doc=(info.doc if info else ""))


def build_tree(composed: ComposedLayer) -> dict:
    """按执行序建树：**每次插入都追加到当前路径链的尾部**，因此 DFS 序 == 执行序。

    容器只在"它是父节点最后一个孩子"时复用；若同名容器已被别的分支隔开，
    就在当前位置再开一个（id 加 `#k`）——这保证 split 视图这种"进入→离开→再进入"
    的配方不会把后插入的叶子排到前面去。
    """
    root = dict(id="layer", name="Layer", kind="container", children=[], doc="")
    used: dict[str, int] = {}

    def uid(path: str) -> str:
        n = used.get(path, 0)
        used[path] = n + 1
        return path if n == 0 else f"{path}#{n + 1}"

    def ensure_container(parent: dict, path: str) -> dict:
        last = parent["children"][-1] if parent["children"] else None
        if last is not None and last.get("kind") == "container" and \
                last.get("_path") == path:
            return last
        node = _new_container(path)          # id 可能是带 #k 的
        node["id"] = uid(path)
        node["_path"] = path
        parent["children"].append(node)
        return node

    for step in composed.steps:
        parts = step.path.split(".")
        parent = root
        cur = ""
        for seg in parts[:-1]:
            cur = seg if not cur else f"{cur}.{seg}"
            parent = ensure_container(parent, cur)
        leaf = dict(id=uid(step.path), name=parts[-1], kind="op", op=step.op,
                    segment=step.segment, occurrence=step.occurrence,
                    rows_phase=step.rows_phase, children=[])
        parent["children"].append(leaf)
    composed.tree = root
    return root


def compose_layer(cfg, recipes, layer, view="default") -> ComposedLayer:
    """默认视图 / split 视图（层 20 的 c1a_source_split）。"""
    kind = "c1a_source_split" if view == "split" else layer["attn"]
    composed = ComposedLayer(idx=int(layer["idx"]), kind=kind, view=view)
    occ: dict[tuple[str, str], int] = {}
    after_cut = False
    for op, seg in tracked_sequence(cfg, recipes, layer, view):
        n = occ.get((seg, op), 0)
        occ[(seg, op)] = n + 1
        path = recipe_map.path_of(seg, op, n)
        if path is None:
            # 新插件：优先用插件自己声明的 container（如 "attn.core"）
            iface = OP_IFACE.get(op)
            hint = getattr(iface, "container", None) if iface else None
            path = (f"{hint}.{op}" if hint else f"{seg.replace('.', '_')}.{op}")
        if view == "split":
            # 截尾前 = full 行；截尾后（含 common 的 wo_b/ffn） = compact 行
            rows_phase = "compact" if after_cut else "full"
        else:
            rows_phase = "same"
        composed.steps.append(Step(op=op, path=path, segment=seg,
                                   occurrence=n, rows_phase=rows_phase, view=view))
        if op == "boundary_cut":
            after_cut = True
    build_tree(composed)
    return composed

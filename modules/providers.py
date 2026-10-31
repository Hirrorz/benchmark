"""静态上下文解析：最近祖先 + 前序写者 + cache 写者索引。

规则（与计划一致）：
  1. 叶子读出：容器作用域（含 enter 提供的键）→ 层工作集（前序叶子 provide 上浮）
     → runner_iface 的 process/pass/layer/st 提供者；
  2. **前序可见**：provide 按执行序发布到层工作集，后面的兄弟/其它容器都能读到；
  3. `cache.*` 不走词法链：取"执行序最近的前序写者"，无写者 = zero-init；
  4. 找不到 → unresolved（optional 键记 optional）；不静默吞掉；
  5. 单写者键被第二次发布 → shadowed 诊断。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from modules import names
from modules.spec import KeySpec


@dataclass
class Binding:
    key: str
    provider: str          # 叶子路径 / "runner.py:..." / "cache.writer:..."
    tier: str
    status: str = "ok"     # ok|in_place|optional|unresolved|cache|zero-init
    condition: str | None = None
    roles: tuple[str, ...] = ()


@dataclass
class LayerResolution:
    nodes: dict = field(default_factory=dict)      # node_id -> {"requires": [...], "provides": [...]}
    diagnostics: dict = field(default_factory=lambda: {
        "unresolved": [], "optional": [], "shadowed": [], "phantom": [],
        "cache_writers": {}, "providers": {},
    })
    consumers: dict = field(default_factory=dict)   # key -> [leaf path]


CONTAINER_ENTER = {
    "attn": {"hc.stage": Binding("hc.stage", "container:attn", "container")},
    "ffn": {"hc.stage": Binding("hc.stage", "container:ffn", "container")},
}


def _base_bindings(base_providers) -> dict:
    """runner_iface.Provider 列表 → key → Binding（同键取最后一个）。"""
    out = {}
    for p in base_providers:
        out[p.key] = Binding(p.key, p.source, p.tier, condition=p.condition)
    return out


def _append(diag: dict, name: str, item) -> None:
    diag.setdefault(name, []).append(item)


def resolve_layer(composed, iface_of, base_providers) -> LayerResolution:
    """遍历模块树，给每个叶子补 requires(resolved)/provides(consumers)。"""
    res = LayerResolution()
    diag = res.diagnostics
    base = _base_bindings(base_providers)
    layer_scope: dict[str, Binding] = {}
    stack: list[dict[str, Binding]] = [layer_scope]
    # cache 写者：canonical -> [(order, path)]
    cache_writers: dict[str, list[tuple[int, str]]] = {}
    order = 0

    def lookup(key: str):
        for scope in reversed(stack):
            if key in scope:
                return scope[key]
        return base.get(key)

    def publish(binding: Binding, leaf_path: str, policy: str) -> None:
        prev = layer_scope.get(binding.key)
        if (prev is not None and prev.status != "in_place"
                and policy not in ("stream", "allow_multi", "in_place", "hidden")):
            _append(diag, "shadowed", dict(key=binding.key, first=prev.provider,
                                           second=leaf_path))
        layer_scope[binding.key] = binding
        res.diagnostics["providers"].setdefault(binding.key, []).append(leaf_path)

    def walk(node: dict) -> None:
        nonlocal order
        if node.get("kind") == "container":
            # enter 提供的键（如 hc.stage）只在本容器作用域内可见，**不上浮**到层
            scope = dict(CONTAINER_ENTER.get(node.get("_path", node["id"]), {}))
            stack.append(scope)
            for ch in node.get("children") or []:
                walk(ch)
            stack.pop()
            return

        # ── 叶子 ──────────────────────────────────────────────────
        iface = iface_of(node["op"])
        order += 1
        node["order"] = order
        req_out, prov_out = [], []
        for spec in iface.requires:
            req_out.append(_resolve_req(spec, node["id"], lookup, cache_writers,
                                        order, diag, res))
        for spec in iface.provides:
            # in_place 是"改写已有键"，不是第二写者
            policy = "in_place" if spec.mutation == "in_place" else spec.shadow
            b = Binding(spec.name, node["id"], spec.tier,
                        status=("in_place" if spec.mutation == "in_place" else "ok"),
                        condition=spec.condition, roles=spec.roles)
            publish(b, node["id"], policy)
            p = spec.to_json()
            p["consumers"] = []
            prov_out.append(p)
            if spec.name.startswith("cache."):
                cache_writers.setdefault(spec.name, []).append((order, node["id"]))
        node["iface"] = dict(op=node["op"], requires=req_out, provides=prov_out)
        res.nodes[node["id"]] = node["iface"]
        # 后续消费者的全局索引
        for spec in iface.requires:
            res.consumers.setdefault(spec.name, []).append(node["id"])

    walk(composed.tree)
    _finalize_consumers(res, composed)
    for item in phantom_provides(composed, res):
        _append(diag, "phantom", item)
    diag["cache_writers"] = {k: [p for _, p in v] for k, v in cache_writers.items()}
    return res


def _resolve_req(spec: KeySpec, leaf: str, lookup, cache_writers, order,
                 diag, res: LayerResolution) -> dict:
    key = spec.name
    b = lookup(key)
    if b is None and key.startswith("cache."):
        writers = [p for o, p in cache_writers.get(key, []) if o < order]
        if writers:
            b = Binding(key, f"cache.writer:{writers[-1]}", "cache", status="cache")
        else:
            b = Binding(key, "zero-init", "cache", status="zero-init")
    if b is None:
        status = "optional" if spec.optional else "unresolved"
        b = Binding(key, "", "unknown", status=status)
        _append(diag, "optional" if spec.optional else "unresolved",
                dict(leaf=leaf, key=key))
    d = spec.to_json()
    d.update(status=b.status, **({"from": b.provider} if b.provider else {}),
             from_tier=b.tier)
    d["overload"] = names.overload_roles(key)
    return d


def _finalize_consumers(res: LayerResolution, composed) -> None:
    """给每个 provide 标出本层内的后续消费者（跨层 cache 消费者在导出器另算）。"""
    by_key = res.consumers
    for node in _iter_leaves(composed.tree):
        iface = node.get("iface") or {}
        for p in iface.get("provides", []):
            p["consumers"] = [c for c in by_key.get(p["key"], []) if c != node["id"]]


def _iter_leaves(tree):
    stack = [tree]
    while stack:
        n = stack.pop()
        if n.get("kind") == "container":
            stack.extend(reversed(n.get("children") or []))
        else:
            yield n


def phantom_provides(composed, res: LayerResolution) -> list[dict]:
    """提供但本层无人消费的键（导出器汇总用）。"""
    out = []
    for node in _iter_leaves(composed.tree):
        for p in (node.get("iface") or {}).get("provides", []):
            if p.get("consumers") or p.get("mutation") == "in_place":
                continue
            # cache.* 的消费者在后续层（跨层写者索引），本层看不到；
            # shadow=hidden 的是模块内部瞬态（如 rope done-set），不算幽灵
            if p["key"].startswith("cache.") or p.get("shadow") == "hidden":
                continue
            out.append(dict(leaf=node["id"], key=p["key"]))
    return out

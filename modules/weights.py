"""权重 ↔ 模块映射 + `_packed` 合成键（不加载真实权重）。

`weights_manifest.layer_manifest()` 给出"本层该有哪些权重"；
`runner.pack_layer_weights`（runner.py:358-451）给出 `_packed` 派生键的生成规则。
本模块只做**声明侧**的覆盖检查与展示，不接触权重数据。
"""
from __future__ import annotations

import os
import re

import weights_manifest as wm

from modules import names

# manifest 里有、但运行期没有任何算子读取的键（显式忽略，避免"幽灵权重"）
IGNORE = {
    "ffn.gate.bias_vl": "多模态门控 bias；mini 不用（无读取方）",
}


def layer_manifest(layer_idx: int, cfg: dict) -> list[tuple[str, str, tuple]]:
    """本层权重清单（专家按 EP 折叠成 `{E}` 模板，避免 128×6 行）。"""
    spec = _spec_from_cfg(cfg)
    e_local = spec.n_routed_experts // max(1, spec.moe_ep_size)
    rows = wm.layer_manifest(layer_idx, n_experts=spec.n_routed_experts,
                             e_range=(0, e_local), cfg=cfg)
    # 合并专家模板：ffn.experts.<n>.w1.weight → ffn.experts.{E}.w1.weight
    folded: dict[str, tuple[str, tuple]] = {}
    for name, dt, shape in rows:
        m = re.match(r"(ffn\.experts)\.(\d+)\.(.+)$", name)
        key = f"{m.group(1)}.{{E}}.{m.group(3)}" if m else name
        folded.setdefault(key, (dt, shape))
    return [(k, dt, shp) for k, (dt, shp) in folded.items()]


def _spec_from_cfg(cfg):
    """模型结构参数（model_spec 是唯一事实源；纯 Python，可 CPU 导入）。"""
    import model_spec
    return model_spec.build(cfg)


def packed_synthetic(manifest: list[tuple[str, str, tuple]],
                     layer_idx: int, cfg: dict) -> dict[str, str]:
    """`_packed` 派生键 → 生产者说明（与 runner.py:380-448 同口径）。"""
    names_ = {n for n, _, _ in manifest}
    out: dict[str, str] = {}
    for name, dt, _ in manifest:
        if name.endswith(".weight") and dt.startswith("F8") and \
                name.replace(".weight", ".scale") in names_:
            out[name] = "pack_qmm（runner.py:380-403）"
    if "ffn.gate.weight" in names_:
        out["ffn.gate.weight.T"] = "gate 转置（runner.py:421-424）"
    if "ffn.gate.bias" in names_:
        out["ffn.gate.bias.f32"] = "gate bias fp32（runner.py:425-428）"
    if "attn.indexer.wq_b.scale" in names_:
        out["attn.indexer.wq_b.scale.pairs"] = "末维配对（runner.py:431-436）"
    if "attn.indexer.k_norm.weight" in names_:
        out["attn.indexer.k_norm.weight.f32"] = "k_norm fp32（runner.py:437-440）"
    if "attn.compressor.norm.weight" in names_:
        out["attn.compressor.norm.weight.f32"] = "compressor norm fp32（runner.py:441-447）"
    # ★ T-043 ④（实验，默认关）：wo_a einsum 版权重对（原始 ND view + k-major 配对 SF）。
    if (os.environ.get("DS41M_WOA_EINSUM", "1").strip().lower()   # 2026-10-26 T-054 默认开
            in ("1", "on", "true", "yes")
            and "attn.wo_a.weight" in names_):
        out["attn.wo_a.weight.einsum"] = (
            "wo_a einsum 权重对（runner.pack_layer_weights，T-043 ④）")
    return out


def _match(pattern: str, key: str) -> bool:
    if pattern == key:
        return True
    if "{E}" not in pattern:
        return False
    rx = "^" + re.escape(pattern).replace(r"\{E\}", r"\d+") + "$"
    return re.match(rx, key) is not None


def _cond_true(cond: str | None, layer: dict) -> bool:
    """权重条件的**粗粒度**求值（只用于覆盖检查；未知条件按 True）。"""
    if not cond:
        return True
    ratio = int(layer.get("ratio") or 0)
    if cond == "ratio>1":
        return ratio > 1
    if cond == "ratio==2":
        return ratio == 2
    if cond == "is_candidate":
        return bool(layer.get("is_candidate"))
    if cond == "engram":
        return bool(layer.get("engram"))
    return True


def collect_refs(composed, iface_of, layer=None) -> dict[str, list[str]]:
    """本层模块树里所有 WeightRef.suffix → [叶子路径]（按层条件过滤）。"""
    from modules.providers import _iter_leaves
    out: dict[str, list[str]] = {}
    for node in _iter_leaves(composed.tree):
        for w in iface_of(node["op"]).weights:
            if layer is not None and not _cond_true(w.condition, layer):
                continue
            out.setdefault(w.suffix, []).append(node["id"])
    return out


def coverage(composed, layer: dict, cfg: dict, iface_of) -> dict:
    """双向覆盖检查：manifest ⊆ 模块声明（或 IGNORE）；声明 ⊆ manifest（或表示合成键）。"""
    layer_idx = int(layer["idx"])
    manifest = layer_manifest(layer_idx, cfg)
    names_ = {n for n, _, _ in manifest}
    packed = packed_synthetic(manifest, layer_idx, cfg)
    refs = collect_refs(composed, iface_of, layer)
    unmapped = [n for n in names_ if n not in IGNORE
                and not any(_match(p, n) for p in refs)]
    missing = []
    for p in refs:
        if any(_match(p, n) for n in names_):
            continue
        if p in packed:
            continue
        missing.append(p)
    return dict(layer=layer_idx, unmapped=sorted(unmapped), missing=sorted(missing),
                manifest=len(names_), declared=len(refs), packed=sorted(packed))


_DERIVED = {
    "ffn.gate.weight.T": "ffn.gate.weight",
    "ffn.gate.bias.f32": "ffn.gate.bias",
    "attn.indexer.wq_b.scale.pairs": "attn.indexer.wq_b.scale",
    "attn.indexer.k_norm.weight.f32": "attn.indexer.k_norm.weight",
    "attn.compressor.norm.weight.f32": "attn.compressor.norm.weight",
}


def ownership(composed, layer: dict, cfg: dict, iface_of) -> list[dict]:
    """给 HTML 的权重表：每行 manifest 权重 + 宿主模块 + 是否 `_packed`。"""
    layer_idx = int(layer["idx"])
    manifest = layer_manifest(layer_idx, cfg)
    packed = packed_synthetic(manifest, layer_idx, cfg)
    refs = collect_refs(composed, iface_of, layer)
    out = []
    for name, dt, shape in manifest:
        owners = [p for pat, ps in refs.items() if _match(pat, name) for p in ps]
        for pat, ps in refs.items():
            base = _DERIVED.get(pat, pat)
            if base == name and pat in packed:
                owners.extend(ps)
        out.append(dict(key=name, dtype=dt, shape=list(shape),
                        owners=sorted(set(owners)),
                        packed=(name in packed),
                        ignored=IGNORE.get(name)))
    return out

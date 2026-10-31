"""模型树构建：40 层 → templates（去重）+ 几何预设 + 诊断（Phase 1，CPU-only）。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import assemble
import model_spec
import opgraph

from modules import compose, providers, runner_iface, weights
from modules.op_iface import OP_IFACE
from modules import ops as _ops                      # noqa: F401 —— 注册 Module 类
from modules.registry import MODULES
from modules.spec import LayerRoles, LayerSpec, ModelTree


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def iface_of(op: str):
    return OP_IFACE[op]


def _template_sig(composed) -> str:
    return "|".join(f"{s.op}@{s.path}@{s.rows_phase}" for s in composed.steps)


def _template_id(composed, seen: dict) -> str:
    tid = f"tpl:{composed.kind}"
    if any(s.segment == "engram" for s in composed.steps):
        tid += "+engram"
    if composed.view != "default":
        tid += f":{composed.view}"
    if seen.get(tid):
        tid += f"#{seen[tid] + 1}"
    seen[tid] = seen.get(tid, 0) + 1
    return tid


@dataclass
class TreeBuild:
    tree: ModelTree
    templates_raw: dict = field(default_factory=dict)   # tid -> ComposedLayer
    resolutions: dict = field(default_factory=dict)     # tid -> LayerResolution


def build(cfg, recipes) -> TreeBuild:
    stack, meta = assemble.build_stack(cfg)
    spec = model_spec.build(cfg)
    tbuild = TreeBuild(tree=ModelTree(model=_model_dict(cfg, spec, meta)))

    # 边界 split 的层号只由模型结构决定（是否启用是运行期开关）；导出器始终生成该变体
    cut_model, _w_model = assemble.replay_cut_w(cfg)

    sigs: dict[str, str] = {}
    name_seen: dict[str, int] = {}
    split_tids: dict[int, str] = {}
    layer_dicts: dict[int, dict] = {}
    for layer in stack:
        layer_dicts[int(layer["idx"])] = layer
        dflt = compose.compose_layer(cfg, recipes, layer, "default")
        sig = _template_sig(dflt)
        if sig not in sigs:
            tid = _template_id(dflt, name_seen)
            sigs[sig] = tid
            tbuild.templates_raw[tid] = dflt
            tbuild.resolutions[tid] = providers.resolve_layer(
                dflt, iface_of, runner_iface.providers_for(layer, cfg))
            tbuild.tree.templates[tid] = dict(kind=dflt.kind, view="default",
                                              steps=[s.op for s in dflt.steps],
                                              root=dflt.tree)
        layer_tid = sigs[sig]

        split_tid = None
        # 边界 split 只可能发生在 cut-1 且是 c1a_source 的层（runner.py:1747-1758）
        if int(layer["idx"]) == int(cut_model) - 1 and \
                str(layer["attn"]) == "c1a_source":
            spl = compose.compose_layer(cfg, recipes, layer, "split")
            ssig = _template_sig(spl)
            if ssig not in sigs:
                stid = _template_id(spl, name_seen)
                sigs[ssig] = stid
                tbuild.templates_raw[stid] = spl
                tbuild.resolutions[stid] = providers.resolve_layer(
                    spl, iface_of, runner_iface.providers_for(layer, cfg))
                tbuild.tree.templates[stid] = dict(kind=spl.kind, view="split",
                                                   steps=[s.op for s in spl.steps],
                                                   root=spl.tree)
            split_tid = sigs[ssig]
            split_tids[int(layer["idx"])] = split_tid

        roles = LayerRoles(kv_source=bool(layer.get("is_kv_src")),
                           index_source=bool(layer.get("is_idx_src")),
                           candidate_source=bool(layer.get("is_candidate")),
                           engram=bool(layer.get("engram")))
        tbuild.tree.layers.append(LayerSpec(
            idx=int(layer["idx"]), kind=str(layer["attn"]), ratio=int(layer["ratio"]),
            roles=roles, engram=bool(layer.get("engram")), template=layer_tid,
            split=split_tid is not None))

    # 未融合 MoE 变体（megamoe=none）——只作对照，不分配给任何层
    if compose.moe_key(cfg) == "fused" and stack:
        import copy
        cfg_uf = copy.deepcopy(cfg)
        cfg_uf.setdefault("backends", {})["megamoe"] = "none"
        rep = layer_dicts[int(stack[0]["idx"])]
        uf = compose.compose_layer(cfg_uf, recipes, rep, "default")
        tid = "tpl:unfused"
        tbuild.templates_raw[tid] = uf
        tbuild.resolutions[tid] = providers.resolve_layer(
            uf, iface_of, runner_iface.providers_for(rep, cfg_uf))
        tbuild.tree.templates[tid] = dict(kind=uf.kind, view="unfused",
                                          steps=[s.op for s in uf.steps],
                                          root=uf.tree)

    # 每层：模板引用 + 权重归属 + 覆盖诊断
    layer_json = []
    cov_agg = dict(unmapped=[], missing=[])
    for ls in tbuild.tree.layers:
        composed = tbuild.templates_raw[ls.template]
        layer = layer_dicts[ls.idx]
        cov = weights.coverage(composed, layer, cfg, iface_of)
        cov_agg["unmapped"] += [dict(layer=ls.idx, key=k) for k in cov["unmapped"]]
        cov_agg["missing"] += [dict(layer=ls.idx, key=k) for k in cov["missing"]]
        row = ls.to_json()
        row["weights"] = weights.ownership(composed, layer, cfg, iface_of)
        row["coverage"] = dict(unmapped=cov["unmapped"], missing=cov["missing"])
        if ls.idx in split_tids:
            row["split_template"] = split_tids[ls.idx]
        layer_json.append(row)

    tbuild.tree.layers = layer_json
    tbuild.tree.presets = geometry_presets(cfg, meta)
    tbuild.tree.diagnostics = _diagnostics(tbuild, cfg, meta, cov_agg, recipes)
    tbuild.tree.variants = _variants(tbuild)
    tbuild.tree.blueprint = _blueprint(recipes, stack, cfg)
    tbuild.tree.builder = _builder(cfg, stack)
    return tbuild


def _model_dict(cfg, spec, meta) -> dict:
    m = cfg.get("model") or {}
    pc = cfg.get("parallel_config") or {}
    return dict(
        name="deepseek_v4_1_flash_mini",
        n_layers=spec.n_layers,
        L=int(meta.get("L") or 0),
        dims=dict(hidden=spec.hidden, moe_intermediate=spec.moe_intermediate,
                  head_dim=spec.head_dim, n_attention_heads=spec.n_attention_heads,
                  q_lora_rank=spec.q_lora_rank, o_lora_rank=spec.o_lora_rank,
                  o_groups=spec.o_groups, index_n_heads=spec.index_n_heads,
                  index_head_dim=spec.index_head_dim, index_topk=spec.index_topk,
                  hc_mult=spec.hc_mult, sliding_window=spec.sliding_window,
                  n_routed_experts=spec.n_routed_experts, topk=spec.topk,
                  qk_rope_head_dim=spec.qk_rope_head_dim),
        derived=spec.derived(),
        roles=dict(kv_source_layers=list(spec.kv_source_layers),
                   index_source_layers=list(spec.index_source_layers),
                   engram_layers=list(spec.engram_layers),
                   candidate_source_layer=spec.candidate_source_layer),
        parallel=dict(world_size=int(pc.get("world_size", spec.world_size) or spec.world_size),
                      moe_ep_size=int(pc.get("moe_ep_size", spec.moe_ep_size) or spec.moe_ep_size),
                      cp_size=int(spec.cp_size)),
        features=dict(cfg.get("features") or {}),
        backends=dict(cfg.get("backends") or {}),
        run=dict(seq_len=int((cfg.get("run") or {}).get("seq_len", meta.get("L") or 0)),
                 moe_chunk_max_len=int((cfg.get("run") or {}).get("moe_chunk_max_len", 0) or 0)),
        meta=dict(T_main=meta.get("T_main"), T_pass=meta.get("T_pass"),
                  cut=meta.get("cut"), w=meta.get("w")),
    )


def geometry_presets(cfg, meta) -> dict:
    """HTML 顶部可切换的几何（只影响展示的形状/T，不影响结构）。"""
    L = int(meta.get("L") or 1)
    cut_model, w_keep = assemble.replay_cut_w(cfg)      # 结构决定的截尾点/宽度
    w = int(meta.get("w") or w_keep)
    cut = meta.get("cut")
    presets = {
        "prefill": dict(T=L, T_main=L, T_pass=L, W=w, W_cut=max(w, 1),
                        NSEQ=1, N=1, B=1, full_len=L, cut=cut, cp_size=1,
                        desc=f"当前配置 prefill L={L}" + (f"（层 {cut}+ replay w={w}）" if cut else "")),
        "prefill_replay": dict(T=L, T_main=L, T_pass=L, W=w_keep,
                               W_cut=w_keep, NSEQ=1, N=1, B=1, full_len=L,
                               cut=cut_model, cp_size=1,
                               desc=f"prefill + SWA replay：层 {cut_model}+ 只算最后 {w_keep} token"),
        "prefill_noreplay": dict(T=L, T_main=L, T_pass=L, W=0, W_cut=1,
                                 NSEQ=1, N=1, B=1, full_len=L, cut=None, cp_size=1,
                                 desc=f"全量 prefill L={L}（不裁尾）"),
        "replay_1024": dict(T=1024, T_main=1024, T_pass=1024, W=w_keep,
                            W_cut=w_keep, NSEQ=1, N=1, B=1, full_len=1024,
                            cut=cut_model, cp_size=1,
                            desc=f"小序列 replay（1024；层 {cut_model}+ 宽 {w_keep}）"),
        "packed_2x1024": dict(T=1024, T_main=1024, T_pass=2048, W=w or 140,
                              W_cut=max(w, 1), NSEQ=2, N=2, B=2, L=1024,
                              full_len=1024, cut=cut, cp_size=1,
                              desc="packed batch 2×1024（tokenwise 打包）"),
        "cp8_L8192": dict(T=1024, T_main=8192, T_pass=8192, W=w or 140,
                          W_cut=max(1, _ceil_div(w or 140, 16) * 2),
                          NSEQ=1, N=1, B=1, full_len=8192, cut=cut, cp_size=8,
                          desc="CP8 L=8192（每 rank 1024 行；compact 每 rank ≤18 行）"),
        "decode_T1": dict(T=1, T_main=1, T_pass=1, W=1, W_cut=1, NSEQ=1, N=1, B=1,
                          full_len=L, cut=None, cp_size=1, pos_offset=L - 1,
                          desc="decode 单 token（T=1）"),
    }
    return presets


def env_for(spec, preset: dict, ratio: int, rows_phase: str = "same",
            layer_idx: int | None = None) -> dict:
    """符号表：模型维度 + 几何（Python 侧；JS 侧同构，见模板里的 evalShape）。

    replay：层号 >= preset["cut"] 的层整层只跑最后 W 行（split 的 full/compact 相位除外）。
    """
    T = int(preset["T"])
    if rows_phase == "compact":
        T = int(preset.get("W_cut") or T)
    elif preset.get("cut") is not None and layer_idx is not None \
            and int(layer_idx) >= int(preset["cut"]):
        T = int(preset.get("W") or T)
    if rows_phase == "full" and preset.get("cut") is not None:
        T = int(preset.get("T_main") or T)
    ratio = max(1, int(ratio))
    nblk = max(1, _ceil_div(max(T // ratio + 1, 1), 128))
    nblk_full = max(1, _ceil_div(T, 128))
    nblk_idx = max(1, _ceil_div(T // ratio + 1, 128))
    d = spec.derived()
    env = dict(
        H=spec.hidden, I=spec.moe_intermediate, Q_LORA=spec.q_lora_rank,
        O_LORA=spec.o_lora_rank, O_GROUPS=spec.o_groups, HEAD_DIM=spec.head_dim,
        N_HEADS=spec.n_attention_heads, ROPE_DIM=spec.qk_rope_head_dim,
        INDEX_N_HEADS=spec.index_n_heads, INDEX_HEAD_DIM=spec.index_head_dim,
        INDEX_Q_DIM=d["index_q_dim"], INDEX_PACKED_D=d["index_packed_d"],
        INDEX_TOPK=spec.index_topk, S=spec.sliding_window, HC_MULT=spec.hc_mult,
        N_EXPERTS=spec.n_routed_experts, TOPK=spec.topk, WIN_SLOTS=spec.win_pool_slots,
        KV_ROW_W=spec.kv_row_width, CMP_ROW_W=spec.cmp_row_width,
        BS=spec.block_size, N_LAYERS=spec.n_layers,
        T=T, T_MAIN=int(preset.get("T_main") or T), T_PASS=int(preset.get("T_pass") or T),
        W=int(preset.get("W") or 0), W_CUT=int(preset.get("W_cut") or 1),
        NSEQ=int(preset.get("NSEQ") or 1), N=int(preset.get("N") or 1),
        B=int(preset.get("B") or 1), L=int(preset.get("L") or preset.get("full_len") or T),
        FULL_LEN=int(preset.get("full_len") or T), RATIO=ratio, ratio=ratio,
        NBLK=nblk, NBLK_FULL=nblk_full, NBLK_IDX=nblk_idx,
        E_LOCAL=spec.n_routed_experts // max(1, spec.moe_ep_size),
        E=spec.n_routed_experts // max(1, spec.moe_ep_size),
        POS_OFFSET=int(preset.get("pos_offset") or 0),
        CP_SIZE=int(preset.get("cp_size") or 1),
    )
    # 小写别名（OP_IFACE 里用的是小写）
    for k in ("q_lora", "o_lora", "o_groups", "head_dim", "n_heads", "n_experts",
              "topk", "hc_mult", "index_topk", "index_head_dim", "index_n_heads",
              "win_slots", "kv_row_w", "cmp_row_w", "rope_dim"):
        src = {"q_lora": "Q_LORA", "o_lora": "O_LORA", "o_groups": "O_GROUPS",
               "head_dim": "HEAD_DIM", "n_heads": "N_HEADS", "n_experts": "N_EXPERTS",
               "topk": "TOPK", "hc_mult": "HC_MULT", "index_topk": "INDEX_TOPK",
               "index_head_dim": "INDEX_HEAD_DIM", "index_n_heads": "INDEX_N_HEADS",
               "win_slots": "WIN_SLOTS", "kv_row_w": "KV_ROW_W",
               "cmp_row_w": "CMP_ROW_W", "rope_dim": "ROPE_DIM"}[k]
        env[k] = env[src]
    env.update(nblk=nblk, nblk_cmp=nblk, nblk_idx=nblk_idx, nblk_full=nblk_full)
    return env


def _recipe_ops(recipes) -> set[str]:
    """recipes.yaml 里出现的全部算子名（含占位/未融合链；不展开、不过滤）。"""
    out: set[str] = set()

    def walk(x):
        if isinstance(x, str):
            if not x.startswith("_"):
                out.add(x)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
    walk(recipes)
    return out


def _diagnostics(tbuild: TreeBuild, cfg, meta, cov_agg, recipes) -> dict:
    diag = dict(
        unresolved=[], optional=[], shadowed=[], phantom=[], overloads=[],
        coverage=dict(recipes_ops=0, opgraph_ops=len(opgraph.OPS),
                      missing_from_opgraph=[], declared_only=[], unfused=[]),
        weights=dict(unmapped=cov_agg["unmapped"], missing=cov_agg["missing"]),
        impl_missing=[], flags={}, templates={},
    )
    for tid, comp in tbuild.templates_raw.items():
        res = tbuild.resolutions[tid]
        for item in res.diagnostics.get("unresolved", []):
            diag["unresolved"].append(dict(template=tid, **item))
        for item in res.diagnostics.get("optional", []):
            diag["optional"].append(dict(template=tid, **item))
        for item in res.diagnostics.get("shadowed", []):
            diag["shadowed"].append(dict(template=tid, **item))
        for item in res.diagnostics.get("phantom", []):
            diag["phantom"].append(dict(template=tid, **item))
        diag["templates"][tid] = dict(unresolved=len(res.diagnostics.get("unresolved", [])),
                                      shadowed=len(res.diagnostics.get("shadowed", [])),
                                      phantom=len(res.diagnostics.get("phantom", [])))
    recipes_ops = _recipe_ops(recipes)
    diag["coverage"]["recipes_ops"] = len(recipes_ops)
    diag["coverage"]["missing_from_opgraph"] = sorted(recipes_ops - set(opgraph.OPS))
    diag["coverage"]["declared_only"] = sorted(set(opgraph.OPS) - recipes_ops)
    diag["coverage"]["unfused"] = ["gating", "gating_sort", "reroute", "dispatch",
                                   "gmm", "swiglu_quant", "combine", "finalize"]
    diag["impl_missing"] = sorted(
        op for op in recipes_ops
        if OP_IFACE[op].impl is None
        and not (OP_IFACE[op].flags & {"implicit", "trace_only", "placeholder"}))
    for op in sorted(OP_IFACE):
        if OP_IFACE[op].flags:
            diag["flags"][op] = sorted(OP_IFACE[op].flags)
    from modules import names
    diag["overloads"] = [dict(key=k, roles=list(v)) for k, v in names.OVERLOADS.items()]
    return diag


def _variants(tbuild: TreeBuild) -> dict:
    """层型分组 + 相邻层型的 delta（HTML 的"变体对比"用）。"""
    groups: dict[str, list[int]] = {}
    for ls in tbuild.tree.layers:
        key = ls["kind"] + ("+engram" if ls["engram"] else "")
        groups.setdefault(key, []).append(ls["idx"])
    seqs = {tid: [s.op for s in comp.steps]
            for tid, comp in tbuild.templates_raw.items()}
    deltas = []
    tids = list(tbuild.templates_raw)
    for a in tids:
        for b in tids:
            if a >= b:
                continue
            sa, sb = seqs[a], seqs[b]
            if sa == sb:
                continue
            deltas.append(dict(a=a, b=b, added=[x for x in sb if x not in sa],
                               removed=[x for x in sa if x not in sb]))
    return dict(groups=[dict(key=k, layers=v) for k, v in groups.items()],
                deltas=deltas[:12],
                templates={tid: dict(kind=comp.kind, view=comp.view,
                                     ops=[s.op for s in comp.steps])
                           for tid, comp in tbuild.templates_raw.items()})


def to_json(tbuild: TreeBuild) -> dict:
    """导出器最终 JSON（不含 HTML）。"""
    t = tbuild.tree
    usage = _usage(tbuild)
    catalog = {op: _op_json(op, usage.get(op)) for op in OP_IFACE}
    return dict(
        schema="ds41m.model_viewer/2",
        model=t.model,
        geometry_presets=t.presets,
        op_catalog=catalog,
        plugin_catalog=_plugin_catalog(catalog),
        templates={tid: dict(kind=v["kind"], view=v["view"], steps=v["steps"],
                             root=v["root"])
                   for tid, v in t.templates.items()},
        layers=t.layers,
        variants=t.variants,
        blueprint=getattr(t, "blueprint", {}),
        builder=getattr(t, "builder", {}),
        session_extras=_extras(),
        diagnostics=t.diagnostics,
    )


def _usage(tbuild) -> dict:
    """op → 用在哪（模板/路径/层）。未使用的插件也保留空记录。"""
    tpl_layers: dict[str, list[int]] = {}
    for row in tbuild.tree.layers:
        tpl_layers.setdefault(row["template"], []).append(int(row["idx"]))
        if row.get("split_template"):
            tpl_layers.setdefault(row["split_template"], []).append(int(row["idx"]))
    out: dict[str, dict] = {}
    for tid, comp in tbuild.templates_raw.items():
        for step in comp.steps:
            u = out.setdefault(step.op, dict(templates=[], paths=[], layers=[]))
            if tid not in u["templates"]:
                u["templates"].append(tid)
            if step.path not in u["paths"]:
                u["paths"].append(step.path)
    for op, u in out.items():
        layers = set()
        for tid in u["templates"]:
            layers.update(tpl_layers.get(tid, []))
        u["layers"] = sorted(layers)
    return out


def _plugin_catalog(catalog: dict) -> list[dict]:
    out = []
    for op, d in sorted(catalog.items(), key=lambda kv: (kv[1].get("category", ""), kv[0])):
        out.append(dict(
            op=op, category=d.get("category", "misc"),
            module=d.get("module"), kernel=d.get("kernel"),
            impl=d.get("impl"), flags=d.get("flags", []),
            used=bool(d.get("used")), used_in=d.get("used_in", {}),
            container=d.get("container"), after=d.get("after", []),
            requires=len(d.get("requires", [])),
            provides=len(d.get("provides", [])),
            weights=len(d.get("weights", [])),
            doc=d.get("doc", ""),
        ))
    return out


def _blueprint(recipes, stack, cfg) -> dict:
    """配方蓝图：每个段的有序积木（含 `_引用`）+ 层型到段的映射。"""
    seg = {}
    for key in ("common", "attn_core", "attn_tail", "engram", "extras"):
        if key in recipes:
            seg[key] = recipes[key]
    seg["attention"] = recipes.get("attention", {})
    seg["moe"] = recipes.get("moe", {})
    kinds = {}
    for layer in stack:
        kinds.setdefault(layer["attn"], []).append(int(layer["idx"]))
    return dict(segments=seg, kinds=kinds,
                moe_key=compose.moe_key(cfg))


def _builder(cfg, stack) -> dict:
    """构建器用的基线：层起点的可见提供者 + 积木分类。"""
    base = []
    if stack:
        for p in runner_iface.providers_for(stack[0], cfg):
            if p.key not in base:
                base.append(p.key)
    cats = sorted({i.category for i in OP_IFACE.values()})
    return dict(base_providers=base, categories=cats,
                containers=sorted({i.container for i in OP_IFACE.values() if i.container}))


def _op_json(op: str, usage: dict | None = None) -> dict:
    d = OP_IFACE[op].to_json()
    cls = MODULES.get(op)
    if cls is not None:
        d["module"] = cls.__name__
    g = opgraph.OPS.get(op)
    if not g and OP_IFACE[op].kernel:
        d["kernel"] = OP_IFACE[op].kernel
    if g:
        d["kernel"] = g.get("kernel")
        d["opgraph_shapes"] = g.get("shapes")
        if g.get("trace"):
            d["trace"] = g["trace"]
        if g.get("note"):
            d["note"] = g["note"]
        if g.get("affects"):
            d["affects"] = g["affects"]
    d["used"] = bool(usage)
    d["used_in"] = usage or dict(templates=[], paths=[], layers=[])
    return d


def _used_op(tree, op: str) -> bool:
    for comp in tree.templates.values():
        if op in comp["steps"]:
            return True
    return False


def _extras() -> dict:
    import yaml
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "recipes.yaml")) as f:
        recipes = yaml.safe_load(f)
    out = []
    for name, n in recipes.get("extras") or []:
        g = opgraph.OPS.get(name, {})
        out.append(dict(op=name, count=int(n), kernel=g.get("kernel"),
                        note="会话级 trace 余量（40 层循环之外）"))
    return dict(ops=out)

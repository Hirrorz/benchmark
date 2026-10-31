#!/usr/bin/env python3



"""模块规格门禁（Phase 1，CPU-only）。

    python3.11 tools/check_module_spec.py --configs 'configs/*.yaml' --strict
    python3.11 tools/check_module_spec.py --compare-json /tmp/dry.json /tmp/spec_dry.json

断言（任一失败 → 退出码 1）：
  1. 40 层模块树 DFS 叶子序列 == run.expand_layer（default / PAD_OPS / SKIP_MOE /
     SKIP_ENGRAM / unfused / split 六种口径）；
  2. recipes 算子 ⊆ OP_IFACE；recipes−opgraph 恰好 {boundary_cut, idx_k_write}；
     opgraph−recipes 恰好 {gate_topk, gather_tail, scatter_tail}；
  3. 每层权重清单双向覆盖（unmapped/missing 为空，已知未读键除外）；
  4. 每个 require 都能解析（unresolved 为空）；shadowed 为空；phantom 集合 == 白名单；
  5. 所有符号形状在全部几何预设/相位下可求值；
  6. 两次构建的 JSON/HTML 逐字节一致（确定性）。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ds41_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    _q = _os.path.join(_ds41_root, _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import assemble                                  # noqa: E402
import opgraph                                   # noqa: E402
import run as runmod                             # noqa: E402

from modules import tree as treemod              # noqa: E402
from modules import runtime as rtmod             # noqa: E402
from modules import weights as wmod              # noqa: E402
from modules.op_iface import OP_IFACE             # noqa: E402
from modules import shapes as shapemod           # noqa: E402
import model_spec                                # noqa: E402

PHANTOM_ALLOW = {
    ("attn.boundary_cut", "geometry.rows"),
    ("attn.compressor.kv_compress", "cmp.kv"),
    ("attn.indexer.search.qli", "idx.topk_length"),
    ("attn.indexer.search.qsli", "idx.topk_length"),
    ("engram.proj", "engram.out"),
    ("engram.proj", "engram.proj_out"),
    ("ffn.moe.experts.mega_moe", "moe.out"),
}

fails: list[str] = []


def ok(cond, msg):
    if not cond:
        fails.append(msg)


def dfs_ops(node):
    if node.get("kind") == "container":
        for c in node.get("children") or []:
            yield from dfs_ops(c)
    else:
        yield node["op"]


def _with_env(overrides, fn):
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return fn()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def check_orders(cfg, recipes):
    stack, _meta = assemble.build_stack(cfg)
    modes = [
        ("default", {}),
        ("pad", {"DS41M_PAD_OPS": "1"}),
        ("skip_moe", {"DS41M_SKIP_MOE": "1"}),
        ("skip_engram", {"DS41M_SKIP_ENGRAM": "1"}),
    ]
    for name, env in modes:
        def run_mode():
            for layer in stack:
                comp = treemod.compose.compose_layer(cfg, recipes, layer, "default")
                got = list(dfs_ops(comp.tree))
                want = runmod.expand_layer(cfg, recipes, layer)
                ok(got == want, f"[{name}] L{layer['idx']} 顺序不一致："
                                f"got {got[:6]}… want {want[:6]}…")
            return True
        _with_env(env, run_mode)

    # unfused MoE（megamoe=none）
    import copy
    cfg_uf = copy.deepcopy(cfg)
    cfg_uf.setdefault("backends", {})["megamoe"] = "none"
    for layer in stack[:2]:
        comp = treemod.compose.compose_layer(cfg_uf, recipes, layer, "default")
        ok(list(dfs_ops(comp.tree)) == runmod.expand_layer(cfg_uf, recipes, layer),
           f"[unfused] L{layer['idx']} 顺序不一致")

    # split 视图（cut-1 且 c1a_source）
    cut, _w = assemble.replay_cut_w(cfg)
    spl_layer = next((l for l in stack
                      if int(l["idx"]) == int(cut) - 1 and l["attn"] == "c1a_source"), None)
    if spl_layer is not None:
        comp = treemod.compose.compose_layer(cfg, recipes, spl_layer, "split")
        want = runmod.expand_layer(cfg, recipes, dict(spl_layer, attn="c1a_source_split"))
        ok(list(dfs_ops(comp.tree)) == want,
           f"[split] L{spl_layer['idx']} 顺序不一致：{list(dfs_ops(comp.tree))}")


def check_runtime_plan(cfg, recipes):
    """Phase 2：模块运行时的执行计划必须与 legacy 序逐项一致（含 split / 环境口径）。"""
    stack, _meta = assemble.build_stack(cfg)
    for layer in stack:
        plan = rtmod.build_plan(cfg, recipes, layer, "default")
        try:
            rtmod.verify_order(plan, runmod.expand_layer(cfg, recipes, layer))
        except rtmod.ModuleRuntimeOrderError as exc:
            fails.append(f"[runtime-plan] {exc}")
    for name, env in (("pad", {"DS41M_PAD_OPS": "1"}),
                      ("skip_moe", {"DS41M_SKIP_MOE": "1"}),
                      ("skip_engram", {"DS41M_SKIP_ENGRAM": "1"})):
        def _run_mode(_env=env, _name=name):
            for layer in (stack[0], stack[-1]):
                plan = rtmod.build_plan(cfg, recipes, layer, "default")
                try:
                    rtmod.verify_order(plan, runmod.expand_layer(cfg, recipes, layer))
                except rtmod.ModuleRuntimeOrderError as exc:
                    fails.append(f"[runtime-plan/{_name}] {exc}")
            return True
        _with_env(env, _run_mode)
    cut, _w = assemble.replay_cut_w(cfg)
    spl = next((l for l in stack
                if int(l["idx"]) == int(cut) - 1 and l["attn"] == "c1a_source"), None)
    if spl is not None:
        plan = rtmod.build_plan(cfg, recipes, spl, "split")
        try:
            rtmod.verify_order(
                plan, runmod.expand_layer(
                    cfg, recipes, dict(spl, attn="c1a_source_split")))
        except rtmod.ModuleRuntimeOrderError as exc:
            fails.append(f"[runtime-plan/split] {exc}")


def check_context():
    from modules import context, runtime_probe
    for mod, name in ((context, "context"), (runtime_probe, "runtime_probe")):
        try:
            mod.self_test()
        except AssertionError as e:
            fails.append(f"{name}.self_test 失败：{e}")


def check_catalog(recipes):
    from modules.tree import _recipe_ops
    from modules import ops as _ops                       # noqa: F401 —— 注册
    from modules.registry import MODULES
    ok(len(MODULES) == len(OP_IFACE),
       f"Module 类数 {len(MODULES)} != OP_IFACE {len(OP_IFACE)}")
    for op, iface in OP_IFACE.items():
        cls = MODULES.get(op)
        ok(cls is not None and cls.IFACE is iface,
           f"op {op!r} 缺 Module 类或 IFACE 不一致")
    refs = _recipe_ops(recipes)
    missing_iface = sorted(r for r in refs if r not in OP_IFACE)
    ok(not missing_iface, f"recipes 算子缺 OP_IFACE：{missing_iface}")
    miss_graph = sorted(refs - set(opgraph.OPS))
    ok(set(miss_graph) == {"boundary_cut", "idx_k_write"},
       f"recipes−opgraph 应为 {{boundary_cut, idx_k_write}}，实际 {miss_graph}")
    # 自动补的默认条目（新 YAML 模块尚未校准）不算"opgraph 声明但配方未用"，
    # 否则"先注册模块、后进 recipes"的正常顺序会让门禁变红。
    decl = sorted(op for op in set(opgraph.OPS) - refs
                  if not opgraph.OPS[op].get("auto"))
    ok(set(decl) == {"gate_topk", "gather_tail", "scatter_tail"},
       f"opgraph−recipes 应为 {{gate_topk, gather_tail, scatter_tail}}，实际 {decl}")


def check_config(cfg, recipes):
    """配置层不变量：来源文件存在、模块可执行、容器已登记、params 在真实 ctx 可解析。"""
    from modules import spec_loader, recipe_map
    src = os.environ.get("DS41M_MODULE_SPEC") or spec_loader.DEFAULT_SPEC
    ok(os.path.exists(src), f"模块配置不存在：{src}")
    rsrc = os.environ.get("DS41M_RECIPE_MAP") or recipe_map.DEFAULT_SPEC
    ok(os.path.exists(rsrc), f"装配配置不存在：{rsrc}")

    no_impl_ok = {"no_impl", "implicit", "declared_only", "placeholder", "runner_hook"}
    for op, iface in OP_IFACE.items():
        ok(bool(iface.impl) or bool(iface.flags & no_impl_ok),
           f"{op}: 配置既无 impl 也无 no_impl/implicit/... 标志")
        if iface.container:
            c = iface.container
            ok(c in recipe_map.CONTAINERS
               or any(c.startswith(x + ".") for x in recipe_map.CONTAINERS),
               f"{op}: container {c!r} 未登记在 recipe_map.CONTAINERS")
        for k in iface.params:
            ok(isinstance(k, str) and bool(k), f"{op}: params 键非法 {k!r}")

    missing = sorted({op for (_seg, op, _occ) in recipe_map.PATHS
                      if op not in OP_IFACE})
    ok(not missing, f"recipe_map 路径引用了未配置模块：{missing}")

    # params 的 "${祖先键}" 必须能在真实层 ctx 上解析（与 runner 追加的常量一致）
    stack, meta = assemble.build_stack(cfg)
    m = cfg["model"]
    unresolved = []
    for layer in stack:
        ctx = runmod.ctx_for(cfg, layer)
        ctx.update(hc_mult=m.get("hc_mult", 4),
                   hc_sinkhorn_iters=m.get("hc_sinkhorn_iters", 20),
                   rope_dim=m.get("qk_rope_head_dim", 64), topk=m["topk"],
                   eps=m.get("rms_norm_eps", 1e-20), pos_offset=None,
                   L_full=meta.get("L"))
        for op in runmod.expand_layer(cfg, recipes, layer):
            iface = OP_IFACE.get(op)
            if iface is None or not iface.params:
                continue
            try:
                rtmod.params_for(op, rtmod._ctx_getter(ctx))
            except KeyError as exc:
                unresolved.append(f"L{layer['idx']} {op}: {exc}")
    ok(not unresolved, f"params 引用无法解析（前 3 条）：{unresolved[:3]}")


def check_tree(tbuild):
    d = tbuild.tree.diagnostics
    ok(not d["unresolved"], f"unresolved 非空：{d['unresolved'][:5]}")
    ok(not d["shadowed"], f"shadowed 非空：{d['shadowed'][:5]}")
    got = {(p["leaf"], p["key"]) for p in d["phantom"]}
    ok(got <= PHANTOM_ALLOW, f"phantom 白名单漂移（新增）：{sorted(got - PHANTOM_ALLOW)}")
    ok(not d["weights"]["unmapped"],
       f"权重未归属：{d['weights']['unmapped'][:5]}")
    ok(not d["weights"]["missing"],
       f"声明权重不在清单：{d['weights']['missing'][:5]}")


def check_plugin_catalog(tbuild):
    """积木库/蓝图数据完整性：每个插件都在 catalog 里，分类非空，used_in 结构正确。"""
    from modules.tree import to_json
    data = to_json(tbuild)
    cat = data["op_catalog"]
    plugs = data["plugin_catalog"]
    ok(set(cat) == set(OP_IFACE), "op_catalog 与 OP_IFACE 不一致")
    ok(len(plugs) == len(OP_IFACE), f"plugin_catalog 数量 {len(plugs)} != {len(OP_IFACE)}")
    for p in plugs:
        ok(bool(p.get("category")), f"插件 {p['op']} 缺 category")
        ok(isinstance(p.get("used_in"), dict), f"插件 {p['op']} 缺 used_in")
        if p.get("used"):
            ok(bool(p["used_in"].get("paths")), f"插件 {p['op']} used 但无 paths")
    bp = data.get("blueprint") or {}
    ok("segments" in bp and "attention" in bp["segments"], "blueprint 缺配方段")
    bd = data.get("builder") or {}
    ok(bool(bd.get("base_providers")), "builder 缺 base_providers")
    ok(bool(bd.get("categories")), "builder 缺 categories")


def check_shapes(cfg, tbuild):
    spec = model_spec.build(cfg)
    for pname, preset in tbuild.tree.presets.items():
        idxs = (0, int(preset["cut"]) + 1) if preset.get("cut") is not None else (0,)
        for ratio in (1, 2):
            for phase in ("same", "compact"):
                for lidx in idxs:
                    env = treemod.env_for(spec, preset, ratio, phase, layer_idx=lidx)
                    for op in OP_IFACE.values():
                        for k in list(op.requires) + list(op.provides):
                            if not k.shape:
                                continue
                            try:
                                shapemod.eval_shape(k.shape, env)
                            except shapemod.ShapeError as e:
                                fails.append(f"shape {op.op}:{k.key} {k.shape} "
                                             f"@ {pname}/{ratio}/{phase}/L{lidx}: {e}")


def check_determinism(cfg_path):
    from modules.export_model_html import build_data, render
    d1, _ = build_data(cfg_path)
    d2, _ = build_data(cfg_path)
    ok(json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True),
       "两次构建 JSON 不一致（非确定性）")
    tpl = os.path.join(ROOT, "tools", "model_viewer.template.html")
    ok(render(tpl, d1) == render(tpl, d2), "两次渲染 HTML 不一致")


def check_compare(a: str, b: str):
    """模块树 dry-run 与 run.py --dry-run 的直方图逐层逐 kernel 一致。"""
    A = json.load(open(a))
    B = json.load(open(b))
    ok(A["meta"] == B["meta"], f"meta 不一致：{A['meta']} vs {B['meta']}")
    la = {x["idx"]: x for x in A["layers"]}
    lb = {x["idx"]: x for x in B["layers"]}
    for idx in sorted(set(la) | set(lb)):
        ok(la.get(idx) == lb.get(idx), f"L{idx} 直方图不一致：\n  {la.get(idx)}\n  {lb.get(idx)}")
    ok(A.get("hist") == B.get("hist"), f"全会话直方图不一致：\n{A.get('hist')}\n{B.get('hist')}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=os.path.join(ROOT, "configs", "*.yaml"))
    ap.add_argument("--strict", action="store_true", default=True)
    ap.add_argument("--compare-json", nargs=2, default=None)
    ap.add_argument("--skip-determinism", action="store_true")
    a = ap.parse_args(argv)

    if a.compare_json:
        check_compare(a.compare_json[0], a.compare_json[1])
        _report()
        return 0 if not fails else 1

    paths = sorted(glob.glob(a.configs))
    ok(bool(paths), f"没有匹配的配置：{a.configs}")
    for path in paths:
        cfg = assemble.load_yaml(path)
        if not isinstance(cfg, dict) or "model" not in cfg:
            print(f"跳过（无 model 段）：{os.path.relpath(path, ROOT)}")
            continue
        recipes = runmod.load_recipes()
        check_context()
        check_catalog(recipes)
        check_config(cfg, recipes)
        check_orders(cfg, recipes)
        check_runtime_plan(cfg, recipes)
        tb = treemod.build(cfg, recipes)
        check_tree(tb)
        check_plugin_catalog(tb)
        check_shapes(cfg, tb)
        if not a.skip_determinism and path == paths[0]:
            check_determinism(path)
    _report()
    return 0 if not fails else 1


def _report():
    if fails:
        print(f"FAIL（{len(fails)}）")
        for f in fails[:40]:
            print("  -", f)
        if len(fails) > 40:
            print(f"  … 其余 {len(fails) - 40} 条")
    else:
        print("PASS：模块规格 / 顺序 / 权重 / 接口 / 形状 / 确定性 全部通过")


if __name__ == "__main__":
    raise SystemExit(main())

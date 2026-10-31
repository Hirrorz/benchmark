#!/usr/bin/env python3



"""Phase 2 模块运行时单测（CPU-only，不碰卡）。

    python3.11 tools/test_module_runtime.py

断言：
  1. `modules.runtime.build_plan` 的叶子序 == legacy `run.expand_layer`
     （default / pad / skip_moe / skip_engram / unfused / split 六种口径）；
  2. `walk()` 的 DFS 访问序 == plan.ops，容器 enter/exit 成对，叶子数一致；
  3. 同一层两次构建的 ops/signature 完全一致（确定性）；
  4. 顺序不一致时 `verify_order` 抛 `ModuleRuntimeOrderError` 且能定位首个差异；
  5. `is_enabled()` 的开关解析符合预期。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ds41_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    _q = _os.path.join(_ds41_root, _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)

import copy
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import assemble                                    # noqa: E402
import run as runmod                               # noqa: E402
from modules import runtime as rt                  # noqa: E402

PASSED = 0
FAILED = 0


def ok(cond, msg):
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  ✗ {msg}")


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


def check_plan(cfg, recipes, layer, view="default", tag=""):
    plan = rt.build_plan(cfg, recipes, layer, view)
    legacy_layer = dict(layer, attn="c1a_source_split") if view == "split" else layer
    legacy = runmod.expand_layer(cfg, recipes, legacy_layer)
    try:
        rt.verify_order(plan, legacy)
        ok(True, "")
    except rt.ModuleRuntimeOrderError as exc:
        ok(False, f"{tag} L{layer['idx']} {view} 顺序不一致：{exc}")

    ok(list(plan.ops) == list(legacy),
       f"{tag} L{layer['idx']} {view} ops != legacy")
    ok(len(plan.ops) == len(plan.steps),
       f"{tag} L{layer['idx']} {view} ops/steps 长度不一致")

    seen = []
    depth = {"cur": 0, "max": 0, "enter": 0, "exit": 0}

    def _enter(node):
        depth["cur"] += 1
        depth["enter"] += 1
        depth["max"] = max(depth["max"], depth["cur"])

    def _exit(node):
        depth["cur"] -= 1
        depth["exit"] += 1

    n = rt.walk(plan, lambda node: seen.append(node["op"]), _enter, _exit)
    ok(seen == list(plan.ops), f"{tag} L{layer['idx']} {view} DFS 序 != ops")
    ok(n == len(plan.ops), f"{tag} L{layer['idx']} {view} walk 叶子数 {n}")
    ok(depth["cur"] == 0 and depth["enter"] == depth["exit"],
       f"{tag} L{layer['idx']} {view} 容器 enter/exit 不成对")
    ok(depth["max"] >= 1, f"{tag} L{layer['idx']} {view} 树没有容器层级")

    plan2 = rt.build_plan(cfg, recipes, layer, view)
    ok(plan.ops == plan2.ops and rt.signature(plan) == rt.signature(plan2),
       f"{tag} L{layer['idx']} {view} 两次构建不一致")
    return plan


def main():
    paths = sorted(glob.glob(os.path.join(ROOT, "configs", "*.yaml")))
    ok(bool(paths), "没有 configs/*.yaml")

    checked_cfgs = 0
    for path in paths:
        cfg = assemble.load_yaml(path)
        if not isinstance(cfg, dict) or "model" not in cfg:
            continue
        checked_cfgs += 1
        recipes = runmod.load_recipes()
        stack, _meta = assemble.build_stack(cfg)
        ok(bool(stack), f"{os.path.basename(path)} 层栈为空")

        for layer in stack:
            check_plan(cfg, recipes, layer, "default",
                       tag=os.path.basename(path))

        for name, env in (("pad", {"DS41M_PAD_OPS": "1"}),
                          ("skip_moe", {"DS41M_SKIP_MOE": "1"}),
                          ("skip_engram", {"DS41M_SKIP_ENGRAM": "1"})):
            _with_env(env, lambda: [
                check_plan(cfg, recipes, layer, "default",
                           tag=f"{os.path.basename(path)}[{name}]")
                for layer in (stack[0], stack[-1])])

        cfg_uf = copy.deepcopy(cfg)
        cfg_uf.setdefault("backends", {})["megamoe"] = "none"
        for layer in stack[:2]:
            check_plan(cfg_uf, recipes, layer, "default",
                       tag=f"{os.path.basename(path)}[unfused]")

        cut, _w = assemble.replay_cut_w(cfg)
        spl = next((l for l in stack
                    if int(l["idx"]) == int(cut) - 1 and l["attn"] == "c1a_source"), None)
        if spl is not None:
            check_plan(cfg, recipes, spl, "split",
                       tag=os.path.basename(path))

    ok(checked_cfgs > 0, "没有任何含 model 段的配置")

    # 错误路径：尾部少一个算子必须被判定为长度不一致
    cfg = assemble.load_yaml(paths[0])
    if "model" in cfg:
        recipes = runmod.load_recipes()
        stack, _ = assemble.build_stack(cfg)
        plan = rt.build_plan(cfg, recipes, stack[0], "default")
        try:
            rt.verify_order(plan, list(plan.ops)[:-1])
            ok(False, "verify_order 未拦住长度不一致")
        except rt.ModuleRuntimeOrderError as exc:
            ok("长度不一致" in str(exc), f"错误信息缺少长度提示：{exc}")

    # 开关解析（2026-10-24 起默认开；=0/off 回退）
    ok(rt.is_enabled({"DS41M_MODULE_RUNTIME": "1"}), "is_enabled(1) 失败")
    ok(rt.is_enabled({"DS41M_MODULE_RUNTIME": "on"}), "is_enabled(on) 失败")
    ok(rt.is_enabled({"DS41M_MODULE_RUNTIME": ""}), "is_enabled('') 应为 True")
    ok(rt.is_enabled({}), "is_enabled(缺省) 应为 True（默认开）")
    ok(not rt.is_enabled({"DS41M_MODULE_RUNTIME": "0"}), "is_enabled(0) 应为 False")
    ok(not rt.is_enabled({"DS41M_MODULE_RUNTIME": "off"}), "is_enabled(off) 应为 False")

    print(f"module_runtime: {PASSED} passed, {FAILED} failed")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

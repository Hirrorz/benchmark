#!/usr/bin/env python3



"""配置驱动模块规格的单测（CPU-only，不碰卡）。

    python3.11 tools/test_module_config.py

断言：
  1. `modules/specs/model_modules.yaml` 可解析，且与历史 Python 表逐字段一致
     （`params` 除外——配置新增字段）；
  2. 每个模块只声明自己的 I/O；父容器不需要汇总子节点需求，
     子节点缺键时沿祖先作用域解析（`modules.context.Scope`）；
  3. `params` 字面量 / `"${key}"` 引用按祖先作用域求值；
  4. 坏配置（重复 op / 未知字段 / 错 schema）明确报错，不静默吞。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ds41_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    _q = _os.path.join(_ds41_root, _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules import context, spec_loader          # noqa: E402
from modules.spec_loader import load_spec_file, resolve_params  # noqa: E402

PASSED = 0
FAILED = 0


def ok(cond, msg):
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  ✗ {msg}")


def _write_tmp(doc):
    import yaml
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    f.close()
    return f.name


def expect_error(doc, needle, what):
    import yaml
    p = _write_tmp(doc)
    try:
        load_spec_file(p)
    except Exception as exc:                          # noqa: BLE001
        ok(needle in str(exc), f"{what}: 错误信息缺少 {needle!r}（{exc}）")
    else:
        ok(False, f"{what}: 应当报错")
    finally:
        os.unlink(p)


def main():
    import yaml
    base = load_spec_file(overlay_dir="")      # 主配置：关闭默认 modules.d
    spec = load_spec_file()                    # 实际默认加载链：主配置 + modules.d
    by = {i.op: i for i in spec}
    ok(bool(base), "主配置没有任何模块")
    ok(set(i.op for i in base) <= set(by),
       "默认 modules.d 加载后丢失了主配置里的模块")

    # 1) 历史 Python 表是迁移基线：基线内 op 必须逐字段一致（params 除外）。
    #    基线外的新增 op 只要求通过 schema/字段校验（load_spec_file 已做）——
    #    新增模块不是回归，不能再断言"总数 == 41"（DX P0-1）。
    from modules import op_iface_builtin as b
    ba = {i.op: i.to_json() for i in base}
    py = {k: v.to_json() for k, v in b.OP_IFACE.items()}
    for d in (ba, py):
        for v in d.values():
            v.pop("params", None)
    missing = sorted(set(py) - set(ba))
    ok(not missing, f"主配置缺少历史 Python 表中的算子：{missing[:5]}")
    diff = [k for k in py if ba.get(k) != py[k]]
    ok(not diff, f"历史算子与 Python 表字段不一致：{diff[:5]}")

    # 2) 父容器不携带子节点 I/O：子节点沿祖先解析
    root = context.Scope("process", tier="const")
    root.define("model.hidden", 5120)
    root.define("model.rms_norm_eps", 1e-6)
    root.define("model.n_attention_heads", 64)
    root.define("model.head_dim", 512)
    layer = root.child("layer", tier="layer")
    layer.define("layer.ratio", 2)
    box = layer.child("attn", tier="container")       # 容器不声明任何键
    leaf = box.child("attn.q_proj.wq_a")
    ok("model.hidden" not in box.local and not box.local, "容器不应汇总子节点键")
    ok(leaf.get("H") == 5120, "子节点未沿祖先取到 model.hidden")
    ok(leaf.get("ratio") == 2, "子节点未沿祖先取到 layer.ratio")
    try:
        leaf.require("model.nope")
    except context.MissingContext:
        ok(True, "")
    else:
        ok(False, "缺失键应当抛 MissingContext")

    # 3) params 字面量 / 引用求值
    ok(by["qli"].params.get("candidate_block_size") == 8, "qli params 字面量丢失")
    ok(by["rmsnorm"].params.get("eps") == "${model.rms_norm_eps}",
       "rmsnorm params 引用丢失")
    got = resolve_params(by["rmsnorm"].params, leaf.get)
    ok(got["eps"] == 1e-6, f"params 引用解析错误：{got}")
    got2 = resolve_params(by["wq_b"].params, leaf.get)
    ok(got2["head_dim"] == 512, f"wq_b 参数解析错误：{got2}")

    # 3b) 运行时参数注入（配置 → ctx，执行后撤销；运行配置优先）
    from modules import runtime as rt
    ctx = {"H": 5120, "eps": 1e-7, "topk": 3, "ratio": 2,
           "n_attention_heads": 64, "head_dim": 512, "hidden": 5120,
           "n_routed_experts": 384, "o_groups": 8, "o_lora_rank": 1024,
           "hc_mult": 4, "sinkhorn_iters": 20, "index_topk": 512,
           "sliding_window": 128, "win_keep": 140}
    inj = rt.inject_params("qli", ctx)
    ok(ctx.get("candidate_block_size") == 8, "qli 未注入 candidate_block_size")
    ok("candidate_block_size" in inj and "candidate_topk_blocks" in inj,
       f"注入键记录不完整：{inj}")
    for k in inj:
        ctx.pop(k, None)
    ok("candidate_block_size" not in ctx, "注入的键未撤销")
    ok(rt.inject_params("topk_gate", ctx) == (), "运行配置应优先于 YAML 默认值")
    try:
        rt.inject_params("rmsnorm", {})
    except KeyError as exc:
        ok("model.rms_norm_eps" in str(exc), f"缺引用错误信息不明确：{exc}")
    else:
        ok(False, "缺引用应当报错（不静默用默认值）")

    # 3d) 真实层 ctx 上所有 params 引用必须可解析（防止配置写错键）
    import assemble
    import run as runmod
    cfg = assemble.load_yaml(os.path.join(ROOT, "configs", "mini_ds41_flash.yaml"))
    stack, meta = assemble.build_stack(cfg)
    recipes = runmod.load_recipes()
    m = cfg["model"]
    checked = 0
    for layer in (stack[0], stack[len(stack) // 2], stack[-1]):
        ctx0 = runmod.ctx_for(cfg, layer)
        ctx0.update(hc_mult=m.get("hc_mult", 4),
                    hc_sinkhorn_iters=m.get("hc_sinkhorn_iters", 20),
                    rope_dim=m.get("qk_rope_head_dim", 64), topk=m["topk"],
                    eps=m.get("rms_norm_eps", 1e-20), pos_offset=None,
                    L_full=meta.get("L"))
        for op in runmod.expand_layer(cfg, recipes, layer):
            iface = by.get(op)
            if iface is None or not iface.params:
                continue
            checked += 1
            try:
                rt.params_for(op, rt._ctx_getter(ctx0))
            except KeyError as exc:
                ok(False, f"L{layer['idx']} {op} 参数引用不可解析：{exc}")
    ok(checked > 0, "没有检查到任何模块 params")

    # 3c) 装配映射（容器/路径）同样来自配置，且与历史 Python 表一致
    from modules import recipe_map as rmap, recipe_map_builtin as rbuiltin
    ok(rmap.CONTAINERS == rbuiltin.CONTAINERS,
       "recipe_map 容器表与 Python 表不一致")
    ok(rmap.PATHS == rbuiltin.PATHS, "recipe_map 路径表与 Python 表不一致")
    ok(rmap.path_of("common", "hc_pre", 1) == "ffn.hc.pre", "path_of 查表错误")
    ok(rmap.path_of("common", "hc_pre", 9) is None, "未映射路径应为 None")

    # 3e) modules.d 小文件：新增模块 / 显式 override / 重复必须报错
    import shutil
    tmpd = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmpd, "zz.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(op="zz_demo", impl="exec_ops:_zz", flags=["no_impl"],
                                requires=[dict(key="hidden.x")]), f, sort_keys=False)
        spec2 = load_spec_file(overlay_dir=tmpd)
        # 隔离口径：overlay_dir=tmpd 不含真实 modules.d，基线必须用同口径的 base。
        ok(len(spec2) == len(base) + 1 and any(i.op == "zz_demo" for i in spec2),
           "modules.d 新增模块失败")
        with open(os.path.join(tmpd, "ov.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(op="rmsnorm", impl="exec_ops:_override",
                                override=True, requires=[]), f, sort_keys=False)
        spec3 = load_spec_file(overlay_dir=tmpd)
        rms = [i for i in spec3 if i.op == "rmsnorm"][0]
        ok(rms.impl == "exec_ops:_override", "override 未生效")
        with open(os.path.join(tmpd, "dup.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(op="rmsnorm", impl="dup"), f, sort_keys=False)
        try:
            load_spec_file(overlay_dir=tmpd)
        except Exception as exc:                      # noqa: BLE001
            ok("override" in str(exc), f"重复模块错误信息不明确：{exc}")
        else:
            ok(False, "重复模块未报错")
    finally:
        shutil.rmtree(tmpd)

    # 3f) 真实加载链（环境变量）：modules.d 覆盖 / 缺失回退 Python 表
    import subprocess
    tmpd2 = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmpd2, "ov.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(op="rmsnorm", override=True,
                                impl="exec_ops:_override", requires=[]), f,
                           sort_keys=False)
        env2 = os.environ.copy()
        env2["DS41M_MODULE_SPEC_DIR"] = tmpd2
        r = subprocess.run(
            [sys.executable, "-c",
             "import modules.op_iface as m; print(m.OP_IFACE['rmsnorm'].impl)"],
            cwd=ROOT, env=env2, capture_output=True, text=True)
        ok(r.stdout.strip() == "exec_ops:_override",
           f"环境变量覆盖未生效：{r.stdout.strip()!r} {r.stderr[-200:]}")
    finally:
        shutil.rmtree(tmpd2)
    env3 = os.environ.copy()
    env3["DS41M_MODULE_SPEC"] = "/nonexistent/spec.yaml"
    r = subprocess.run(
        [sys.executable, "-c",
         "import modules.op_iface as m; print(len(m.OP_IFACE))"],
        cwd=ROOT, env=env3, capture_output=True, text=True)
    ok(r.stdout.strip().splitlines()[-1:] == [str(len(b.OP_IFACE))],
       f"配置缺失时未回退 Python 表：{r.stdout.strip()!r} {r.stderr[-200:]}")

    # 3g) 主配置可由环境变量替换；模块定义不在 Python 文件里
    tmpd3 = tempfile.mkdtemp()
    try:
        doc = yaml.safe_load(open(spec_loader.DEFAULT_SPEC, encoding="utf-8"))
        for m in doc["modules"]:
            if m["op"] == "qli":
                m["params"] = {"candidate_block_size": 16}
        fp = os.path.join(tmpd3, "spec.yaml")
        with open(fp, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
        env4 = os.environ.copy()
        env4["DS41M_MODULE_SPEC"] = fp
        r = subprocess.run(
            [sys.executable, "-c",
             "from modules.op_iface import OP_IFACE as m; "
             "print(m['qli'].params['candidate_block_size'])"],
            cwd=ROOT, env=env4, capture_output=True, text=True)
        ok(r.stdout.strip().splitlines()[-1:] == ["16"],
           f"环境变量主配置替换未生效：{r.stdout.strip()!r}")
    finally:
        shutil.rmtree(tmpd3)
    op_src = open(os.path.join(ROOT, "modules", "op_iface.py"),
                  encoding="utf-8").read()
    ok("_add(OpInterface(" not in op_src,
       "op_iface.py 又出现了硬编码模块定义（应只做配置加载）")

    # 3i) 键命名空间也来自配置，且与历史表逐项/逐序一致
    from modules import names, names_builtin as nb
    ok(list(names.LEGACY.items()) == list(nb.LEGACY.items()),
       "key_names.legacy 与历史表不一致（含顺序）")
    ok(names.OVERLOADS == nb.OVERLOADS, "key_names.overloads 不一致")
    ok(names._TIER_BY_PREFIX == nb._TIER_BY_PREFIX, "key_names.tiers 不一致")
    ok(names.canonical("H") == "model.hidden", "canonical(H) 错误")
    ok("H" in names.aliases_for("model.hidden"), "反向别名索引缺 H")
    tmpd5 = tempfile.mkdtemp()
    try:
        doc5 = yaml.safe_load(open(names.DEFAULT_SPEC, encoding="utf-8"))
        doc5["legacy"]["H"] = "model.custom_hidden"
        fp5 = os.path.join(tmpd5, "keys.yaml")
        with open(fp5, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc5, f, sort_keys=False, allow_unicode=True)
        env5 = os.environ.copy()
        env5["DS41M_KEY_NAMES"] = fp5
        r = subprocess.run(
            [sys.executable, "-c",
             "from modules import names; print(names.canonical('H'))"],
            cwd=ROOT, env=env5, capture_output=True, text=True)
        ok(r.stdout.strip().splitlines()[-1:] == ["model.custom_hidden"],
           f"键名配置覆盖未生效：{r.stdout.strip()!r}")
    finally:
        shutil.rmtree(tmpd5)

    # 3h) new_module.py 默认生成 YAML 小文件，并能被加载器注册
    tmpd4 = tempfile.mkdtemp()
    try:
        r = subprocess.run(
            [sys.executable, "tools/new_module.py", "zz_gen",
             "--out-dir", tmpd4, "--impl", "exec_ops:_zz",
             "--requires", "hidden.x=[T,H]", "--param", "scale=1.5"],
            cwd=ROOT, capture_output=True, text=True)
        ok("注册 OK" in r.stdout, f"new_module 注册失败：{r.stdout[-200:]}{r.stderr[-200:]}")
        fp = os.path.join(tmpd4, "zz_gen.yaml")
        ok(os.path.exists(fp), "new_module 未生成 YAML")
        spec4 = load_spec_file(overlay_dir=tmpd4)
        zz = [i for i in spec4 if i.op == "zz_gen"]
        ok(bool(zz) and zz[0].params.get("scale") == 1.5,
           "new_module 生成的配置未被加载")

        # 回归（DX P0-2/P0-3）：新模块只写 YAML 也必须自动生成 Module 类，
        # 并自动获得 opgraph 默认条目（否则 gates 必红）。
        env6 = os.environ.copy()
        env6["DS41M_MODULE_SPEC_DIR"] = tmpd4
        r6 = subprocess.run(
            [sys.executable, "-c",
             "import modules.ops; from modules.registry import MODULES; "
             "import opgraph; "
             "print(int('zz_gen' in MODULES), int('zz_gen' in opgraph.OPS))"],
            cwd=ROOT, env=env6, capture_output=True, text=True)
        ok(r6.stdout.strip().splitlines()[-1:] == ["1 1"],
           f"新模块未自动注册 Module/opgraph：{r6.stdout.strip()!r} "
           f"{r6.stderr[-300:]}")
    finally:
        shutil.rmtree(tmpd4)

    # 3j) 回归（DX 沙盒假绿）：显式覆盖目录不存在时必须报错，不能静默退回主配置。
    env7 = os.environ.copy()
    env7["DS41M_MODULE_SPEC_DIR"] = "/nonexistent/dx_modules_d"
    r7 = subprocess.run(
        [sys.executable, "-c", "import modules.op_iface; print('loaded')"],
        cwd=ROOT, env=env7, capture_output=True, text=True)
    ok(r7.returncode != 0 and "不存在" in (r7.stdout + r7.stderr),
       f"缺失的 DS41M_MODULE_SPEC_DIR 未报错（假绿）：rc={r7.returncode} "
       f"{r7.stdout[-200:]}{r7.stderr[-200:]}")

    # 4) 坏配置必须报错
    bad_base = dict(schema=spec_loader.SCHEMA, modules=[
        dict(op="a", requires=[], provides=[], weights=[])])
    dup = dict(schema=spec_loader.SCHEMA, modules=[bad_base["modules"][0],
                                                    bad_base["modules"][0]])
    expect_error(dup, "重复模块", "重复 op")
    bad_field = dict(schema=spec_loader.SCHEMA, modules=[
        dict(op="a", requires=[dict(key="x", nope=1)])])
    expect_error(bad_field, "未知字段", "未知字段")
    bad_schema = dict(schema="wrong", modules=bad_base["modules"])
    expect_error(bad_schema, "schema", "错 schema")

    print(f"module_config: {PASSED} passed, {FAILED} failed")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

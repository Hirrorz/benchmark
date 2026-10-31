#!/usr/bin/env python3



"""导出“当前模型抽象”单文件 YAML（层 → 模板 → 模块树 + 模块目录）。

    python3.11 tools/export_model_yaml.py --config configs/mini_ds41_flash.yaml \
            --out modules/specs/model_abstract.yaml
    python3.11 tools/export_model_yaml.py --config ... --check   # 与已提交文件比对

数据来源（全部是配置/装配，不读运行期状态）：
  configs/*.yaml + recipes.yaml + modules/specs/{model_modules,recipe_map,key_names}.yaml。
纯 CPU；输出确定性（固定排序、无时间戳）。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ds41_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    _q = _os.path.join(_ds41_root, _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import assemble                                      # noqa: E402
import run as runmod                                 # noqa: E402
from modules import tree as treemod                  # noqa: E402
from modules.op_iface import OP_IFACE                # noqa: E402

SCHEMA = "ds41_mini/model-abstract@1"


def _node_name(node: dict) -> str:
    """容器名：优先层内路径末段（id 可能带 #k）。"""
    nid = str(node.get("id") or node.get("_path") or "?")
    return nid.rsplit(".", 1)[-1]


def _tree(node: dict):
    """容器 → 有序 list；叶子 → op 字符串。"""
    if node.get("kind") != "container":
        return node.get("op", "?")
    out = []
    for c in node.get("children") or []:
        if c.get("kind") == "container":
            out.append({_node_name(c): _tree(c)})
        else:
            out.append(c.get("op", "?"))
    return out


def _key(k) -> str:
    s = k.name
    if k.shape:
        s += f" {k.shape}"
    if k.optional:
        s += "?"
    return s


def _module(op: str) -> dict:
    i = OP_IFACE[op]
    d: dict = dict(category=i.category, impl=i.impl or "", kernel=i.kernel or "")
    if i.flags:
        d["flags"] = sorted(i.flags)
    if i.params:
        d["params"] = dict(i.params)
    d["requires"] = [_key(k) for k in i.requires]
    d["provides"] = [_key(k) for k in i.provides]
    if i.weights:
        d["weights"] = [w.suffix for w in i.weights]
    if i.doc:
        d["doc"] = i.doc
    return d


def build(cfg_path: str) -> dict:
    cfg = assemble.load_yaml(cfg_path)
    recipes = runmod.load_recipes()
    tb = treemod.build(cfg, recipes)
    tree = tb.tree

    templates = {}
    used_ops = set()
    for tid, t in tree.templates.items():
        templates[tid] = dict(kind=t["kind"], view=t["view"], steps=t["steps"],
                              tree=_tree(t["root"]))
        used_ops.update(t["steps"])

    layers = [dict(idx=l["idx"], kind=l["kind"], ratio=l["ratio"],
                   roles=l["roles"], engram=l["engram"],
                   template=l["template"], split=l.get("split", False))
              for l in tree.layers]

    modules = {op: _module(op) for op in sorted(used_ops)}

    return {
        "schema": SCHEMA,
        "config": os.path.relpath(os.path.abspath(cfg_path), ROOT),
        "model": tree.model,
        "layers": layers,
        "templates": templates,
        "modules": modules,
    }


def render(doc: dict) -> str:
    import yaml
    head = (
        "# DS4.1 mini 当前模型抽象（自动导出，勿手改）\n"
        "# 生成：python3.11 tools/export_model_yaml.py --config <cfg> --out <file>\n"
        "# 结构：model 常量 / layers 40 层 / templates 模板（steps + 模块树）/\n"
        "#       modules 用到的模块接口与参数（完整定义见 model_modules.yaml）。\n"
    )
    return head + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=110)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        ROOT, "configs", "mini_ds41_flash.yaml"))
    ap.add_argument("--out", default=os.path.join(
        ROOT, "modules", "specs", "model_abstract.yaml"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    doc = build(a.config)
    text = render(doc)
    if a.check:
        try:
            cur = open(a.out, encoding="utf-8").read()
        except OSError:
            cur = None
        if cur != text:
            print(f"--check 失败：{a.out} 与配置/装配不一致（请重新生成）")
            return 1
        print(f"--check OK：{a.out}")
        return 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"导出 {a.out}（{len(text) / 1024:.0f} KiB，"
          f"{len(doc['layers'])} 层 / {len(doc['templates'])} 模板 / "
          f"{len(doc['modules'])} 模块）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

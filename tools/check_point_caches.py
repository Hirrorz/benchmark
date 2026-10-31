#!/usr/bin/env python3
"""点位缓存登记检查（静态，不碰卡）—— `core/memlife.py` 的防漏网。

为什么需要：跨点位显存累积的老根因是**手写清表会漏**。现在释放走「模块自己声明
`POINT_CACHES` / `POINT_RELEASE()`」的登记表，本工具做两件事：

  1. 扫出每个模块的**模块级容器**（dict/list/set），按「是否登记」分类；
  2. 用启发式判断容器里**是否装了设备张量**（赋值右值里出现 torch.* / npu_* /
     empty/zeros/view/to( …），没登记的报 ⚠️。

白名单 `RESIDENT`（与几何无关、**必须**跨点位复用，不许登记）写死在下面；
新增这类缓存放白名单并写清理由。

用法：
    python3 tools/check_point_caches.py            # 报告
    python3 tools/check_point_caches.py --strict   # 有未登记的设备容器 → 退出码 1
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKGS = ("core", "meta", "layout", "ops", "cp", "trace", "weights", "modules")

# 白名单**唯一来源** = `core/memlife.RESIDENT`（运行时兜底清理也用同一份）
sys.path.insert(0, os.path.join(ROOT, "core"))
try:
    from memlife import RESIDENT
except Exception:                                    # noqa: BLE001 —— 单文件运行兜底
    RESIDENT = set()

TENSOR_HINTS = (
    "torch.", "torch_npu", "tensor", ".to(", ".view(", ".npu(", "empty(",
    "zeros(", "ones(", "full(", "rand", "buf", "pool", "_cache_buf", "index_select",
    "arange", "cat(", "merge_global", "all_gather", "quant",
)


def _is_container_node(node) -> bool:
    if isinstance(node, ast.Dict) and not node.keys:
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
            node.func.id in ("dict", "list", "set"):
        return True
    if isinstance(node, (ast.List, ast.Set)) and not node.elts:
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
            node.func.attr in ("OrderedDict",):
        return True
    return False


def _registered_names(tree) -> set:
    out = set()
    for n in tree.body:                       # 只看模块级
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in (
                        "POINT_CACHES", "POINT_RELEASE"):
                    pass
            if any(isinstance(t, ast.Name) and t.id == "POINT_CACHES"
                   for t in n.targets):
                for e in ast.walk(n.value):
                    if isinstance(e, ast.Name):
                        out.add(e.id)
    return out


def scan_file(path: str):
    src = open(path, encoding="utf-8").read()
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:                # 语法坏的文件直接报出来
        return None, f"SyntaxError: {exc}"
    containers = {}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and _is_container_node(n.value):
                    containers[t.id] = n.lineno
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and \
                n.value is not None and _is_container_node(n.value):
            containers[n.target.id] = n.lineno
    registered = _registered_names(tree)
    # 启发式：容器名出现在「下标赋值」的左值里，且右值像设备张量
    tensorish = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) \
                        and t.value.id in containers:
                    txt = ast.get_source_segment(src, n.value) or ""
                    if any(h in txt for h in TENSOR_HINTS):
                        tensorish.setdefault(t.value.id, []).append(n.lineno)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and \
                n.func.attr in ("append", "extend", "add") and \
                isinstance(n.func.value, ast.Name) and n.func.value.id in containers:
            txt = ast.get_source_segment(src, n) or ""
            if any(h in txt for h in TENSOR_HINTS):
                tensorish.setdefault(n.func.value.id, []).append(n.lineno)
    return (containers, registered, tensorish), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="有未登记的设备容器就退出 1")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    bad, rows, n_files = [], [], 0
    for pkg in PKGS:
        d = os.path.join(ROOT, pkg)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".py") or fn.endswith(".bak") or ".bak_" in fn:
                continue
            path = os.path.join(d, fn)
            n_files += 1
            res, err = scan_file(path)
            if err:
                bad.append((f"{pkg}/{fn}", err))
                continue
            containers, registered, tensorish = res
            for name, line in sorted(containers.items(), key=lambda kv: kv[1]):
                if not tensorish.get(name):
                    continue
                state = ("registered" if name in registered
                         else "RESIDENT" if name in RESIDENT else "UNREGISTERED")
                rows.append((state, f"{pkg}/{fn}:{line}", name,
                             len(tensorish[name])))
    print(f"扫了 {n_files} 个文件；疑似装设备张量的模块级容器 {len(rows)} 个：\n")
    for state, where, name, n in sorted(rows):
        mark = {"registered": "✅", "RESIDENT": "· ", "UNREGISTERED": "⚠️"}[state]
        print(f" {mark} {state:<12} {name:<28} {where}  （{n} 处写入）")
    if bad:
        print("\n解析失败：")
        for where, err in bad:
            print(f"  ✗ {where}: {err}")
    unreg = [r for r in rows if r[0] == "UNREGISTERED"]
    print(f"\n未登记且像设备张量：{len(unreg)} 个"
          + ("" if not unreg else " → 要么加进该模块 POINT_CACHES，要么写进本工具 RESIDENT 并注明理由"))
    if a.strict and (unreg or bad):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

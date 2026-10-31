#!/usr/bin/env python3



"""新增一个模块（**配置优先**，默认不生成 Python 类）。

    python3.11 tools/new_module.py my_attn_gate \
        --category attention --container attn.core --after mmsmla \
        --requires 'hidden.x=[T,H],model.hidden' \
        --provides 'attn.gate_out=[T,H]' \
        --weights 'attn.my_attn_gate.weight' \
        --param gate_scale=1.0 --param hidden='${model.hidden}' \
        --kernel MyGateKernel --impl 'exec_ops:_my_attn_gate' \
        --doc '一句话说明'

默认写 `modules/specs/modules.d/<op>.yaml`（单模块小文件，加载器自动合并）；
随后：
  1) 把 `<op>` 加进 `recipes.yaml` 对应段（存在性）；
  2) `python3.11 tools/test_module_config.py && tools/check_module_spec.py`；
  3) `python3.11 -m modules.export_model_html --config configs/mini_ds41_flash.yaml`。

`--python-stub` 仍可生成旧的动态 Python 插件（`modules/ops/plugins/<op>.py`）；
默认工作流是改配置，只有实现体确实要写在 Python 里才需要它。
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
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_OUT_DIR = os.path.join(ROOT, "modules", "specs", "modules.d")


def _camel(op: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in re.split(r"[_@]+", op) if p)


def _split_top(s: str) -> list[str]:
    """按**顶层**逗号切分（`[T,H]` 里的逗号不切）。"""
    out, depth, cur = [], 0, []
    for ch in s or "":
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return [x.strip() for x in out if x.strip()]


def _parse_keys(spec: str) -> list[dict]:
    """`"hidden.x=[T,H],model.hidden"` → [{key,type,shape?}]。"""
    out = []
    for item in _split_top(spec):
        if "=" in item:
            name, shape = item.split("=", 1)
            out.append(dict(key=name.strip(), type="Tensor", shape=shape.strip()))
        else:
            out.append(dict(key=item, type="int"))
    return out


def _parse_param(s: str) -> tuple[str, object]:
    key, sep, raw = s.partition("=")
    if not sep or not key.strip():
        raise SystemExit(f"--param 需要 key=value：{s!r}")
    raw = raw.strip()
    if re.fullmatch(r"-?\d+", raw):
        val: object = int(raw)
    elif re.fullmatch(r"-?\d+\.\d+", raw):
        val = float(raw)
    elif raw.lower() in ("true", "false"):
        val = raw.lower() == "true"
    else:
        val = raw
    return key.strip(), val


def _entry(a, requires, provides, weights, after, params):
    d: dict = dict(op=a.op, category=a.category)
    if a.container:
        d["container"] = a.container
    if a.impl:
        d["impl"] = a.impl
    if a.kernel:
        d["kernel"] = a.kernel
    if after:
        d["after"] = after
    d["requires"] = requires
    d["provides"] = provides
    if weights:
        d["weights"] = [dict(suffix=w) for w in weights]
    if params:
        d["params"] = params
    if a.doc:
        d["doc"] = a.doc
    return d


def _write_yaml(path: str, entry: dict) -> None:
    import yaml
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# 模块配置：{entry['op']}（由 tools/new_module.py 生成）\n")
        yaml.safe_dump(entry, f, sort_keys=False, allow_unicode=True, width=110)


def _key_lit(k: dict) -> str:
    if k.get("shape"):
        return f'K("{k["key"]}", "{k["type"]}", "{k["shape"]}")'
    if k.get("type", "int") != "int":
        return f'K("{k["key"]}", "{k["type"]}")'
    return f'K("{k["key"]}")'


def _tuple_lit(items: list[str]) -> str:
    """Python tuple 字面量：单元素必须带尾逗号，否则 `(K(...))` 不是 tuple。"""
    return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"


def _write_python_stub(path: str, a, requires, provides, weights, after) -> None:
    after_lit = _tuple_lit([f'"{x}"' for x in after])
    req = _tuple_lit([_key_lit(k) for k in requires])
    prov = _tuple_lit([_key_lit(k) for k in provides])
    ws = _tuple_lit([f'W("{w}")' for w in weights])
    impl = a.impl or f"exec_ops:_{a.op}"
    body = f'''"""插件：{a.op}（旧的动态 Python 入口；默认请用配置）。"""
from modules.plugin import K, W, module
from modules.registry import LeafModule


@module(
    op="{a.op}",
    category="{a.category}",
    container="{a.container}",
    after={after_lit},
    requires={req},
    provides={prov},
    weights={ws},
    impl="{impl}",
    kernel="{a.kernel}",
    doc="{a.doc}",
)
class {_camel(a.op)}Module(LeafModule):
    def execute(self, st, ctx, w):
        from exec_ops import OPS
        return OPS[self.OP](st, ctx, w)
'''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("op", help="算子名（小写下划线，如 my_attn_gate）")
    ap.add_argument("--category", default="misc",
                    help="积木库分类：attention/compressor/indexer/moe/engram/common/...")
    ap.add_argument("--container", default="", help="默认容器路径，如 attn.core")
    ap.add_argument("--after", default="", help="顺序约束（逗号分隔的算子名）")
    ap.add_argument("--kernel", default="", help="trace 里的内核名")
    ap.add_argument("--impl", default="", help="实现位置，如 exec_ops:_my_attn_gate")
    ap.add_argument("--requires", default="hidden.x=[T,H]",
                    help="逗号分隔，key 或 key=shape")
    ap.add_argument("--provides", default="", help="同上")
    ap.add_argument("--weights", default="", help="逗号分隔的权重键")
    ap.add_argument("--param", action="append", default=[],
                    help="模块参数 key=value（可重复；支持 ${祖先键}）")
    ap.add_argument("--doc", default="", help="一句话说明")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="输出目录（默认 modules/specs/modules.d）")
    ap.add_argument("--python-stub", action="store_true",
                    help="生成旧的动态 Python 插件，而不是 YAML")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    if not re.match(r"^[a-z_][a-z0-9_]*$", a.op):
        raise SystemExit(f"算子名必须小写下划线：{a.op!r}")
    requires = _parse_keys(a.requires)
    provides = _parse_keys(a.provides)
    weights = [w.strip() for w in a.weights.split(",") if w.strip()]
    after = [x.strip() for x in a.after.split(",") if x.strip()]
    params = dict(_parse_param(s) for s in a.param)

    if a.python_stub:
        path = os.path.join(ROOT, "modules", "ops", "plugins", a.op + ".py")
    else:
        path = os.path.join(a.out_dir, a.op + ".yaml")
    if os.path.exists(path) and not a.force:
        raise SystemExit(f"已存在：{os.path.relpath(path, ROOT)}（--force 覆盖）")

    if a.python_stub:
        _write_python_stub(path, a, requires, provides, weights, after)
    else:
        _write_yaml(path, _entry(a, requires, provides, weights, after, params))

    env = os.environ.copy()
    if not a.python_stub:
        env["DS41M_MODULE_SPEC_DIR"] = os.path.dirname(path)
    check = subprocess.run(
        [sys.executable, "-c",
         "import modules.ops; from modules.op_iface import OP_IFACE; "
         "from modules.registry import MODULES; "
         f"assert '{a.op}' in OP_IFACE, '未注册接口'; "
         f"assert '{a.op}' in MODULES, '未生成 Module 类'; print('注册 OK')"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    print(f"已生成 {os.path.relpath(path, ROOT)}")
    if check.returncode == 0:
        print(f"  {check.stdout.strip()}")
    else:
        print("  ⚠️ 注册校验失败：")
        print("  " + (check.stderr or check.stdout).strip().splitlines()[-1])

    # 沙盒目录不是默认 modules.d 时，门禁必须显式带 DS41M_MODULE_SPEC_DIR，
    # 否则模块根本没加载、门禁却 PASS（DX 假绿）。
    sandbox = (not a.python_stub
               and os.path.abspath(a.out_dir) != os.path.abspath(DEFAULT_OUT_DIR))
    gate_env = f"DS41M_MODULE_SPEC_DIR={a.out_dir} " if sandbox else ""
    print("\n下一步：")
    print(f"  1) 把 \"{a.op}\" 加进 recipes.yaml 的对应段")
    print(f"  2) {gate_env}python3.11 tools/test_module_config.py && "
          f"{gate_env}python3.11 tools/check_module_spec.py")
    if sandbox:
        print(f"     注意：不设 DS41M_MODULE_SPEC_DIR={a.out_dir} 时门禁看不到该模块")
    print("  3) python3.11 -m modules.export_model_html --config configs/mini_ds41_flash.yaml")
    if not a.python_stub:
        print(f"  实现：在 exec_ops.py 提供 {a.impl or f'exec_ops:_{a.op}'} 指向的函数")
        print("  可选：在 opgraph.py 的 OPS 补 shapes/count/trace（不补也会自动登记，"
              "只是直方图未校准）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

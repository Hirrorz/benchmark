#!/usr/bin/env python3



"""校验 mini 的配置结构（model_spec）。

用法：
  python3 tools/check_specs.py                 # 遍历 configs/*.yaml
  python3 tools/check_specs.py configs/xx.yaml # 只查一个
  python3 tools/check_specs.py --strict        # 有错时退出码非 0（CI/门禁用）

检查内容：
  1. ModelSpec 能从 cfg 构建（所有字段都有默认值，不依赖 yaml 写全）；
  2. validate() 的结构一致性规则（见 model_spec.validate）；
  3. [后续] 推导的 weights_manifest 层集合与字面表一致（默认配置）。
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
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import yaml                                   # noqa: E402

import model_spec                             # noqa: E402


def check_manifest(cfg) -> list[str]:
    """断言“按 cfg 推导的 COND 层表”与 weights_manifest 的字面表一致。"""
    import weights_manifest as M
    h = M.cond_holders(cfg)
    if h is None:
        return []
    # 只有“默认结构”的配置才要求推导结果 == 字面表；有意改层表的配置按推导走（正确行为）
    _def = os.path.join(HERE, "configs", "mini_ds41_flash_hit97_64k.yaml")
    try:
        with open(_def) as f:
            _dcfg = yaml.safe_load(f) or {}
        if M.cond_holders(_dcfg) != h:
            return []
    except Exception:                      # noqa: BLE001
        pass
    bad = []
    for suf, (_spec, holders) in M.COND.items():
        want = sorted({int(x) for x in holders})
        got = sorted({int(x) for x in h.get(suf, holders)})
        if want != got:
            bad.append(f"COND[{suf}] 字面 {want} ≠ 推导 {got}")
    return bad


def check_engram(cfg) -> str:
    """engram 布局跟随 cfg；默认配置应与 DEFAULT_L 完全一致（行为不变）。"""
    import engram_layout as el
    el.configure(cfg)
    same = (el.L == el.DEFAULT_L)
    return f"engram_layout: {'= DEFAULT_L ✓' if same else '非默认（按 config 覆盖）'}"


def check_one(path: str) -> tuple[bool, list[str], str]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    s = model_spec.build(cfg)
    errs = model_spec.validate(s, cfg) + check_manifest(cfg)
    return (not errs), errs, model_spec.describe(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", help="yaml 路径（缺省 = configs/*.yaml）")
    ap.add_argument("--strict", action="store_true", help="有错时返回非 0")
    args = ap.parse_args()

    paths = args.configs or sorted(glob.glob(os.path.join(HERE, "configs", "*.yaml")))
    bad = 0
    for p in paths:
        ok, errs, desc = check_one(p)
        name = os.path.basename(p)
        if ok:
            print(f"✅ {name}")
        else:
            bad += 1
            print(f"❌ {name}")
            for e in errs:
                print(f"     - {e}")
        print(f"   {desc}")
        if "model" in (yaml.safe_load(open(p)) or {}):
            print(f"   {check_engram(yaml.safe_load(open(p)) or {})}")
    print(f"\n{len(paths) - bad}/{len(paths)} 个配置通过")
    return 1 if (bad and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())

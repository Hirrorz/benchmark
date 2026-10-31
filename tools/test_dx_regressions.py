#!/usr/bin/env python3
"""DX P0 回归（CPU-only，不碰卡；2026-10-26 BOARD T-014）。

    python3.11 tools/test_dx_regressions.py

覆盖：
  · P0-2/P0-3：只写 YAML 的新模块 → 自动生成 Module 类 + opgraph 默认条目；
  · P0-4：`new_module.py --python-stub` 的单元素 tuple 必须可导入；
  · P0-5：viewer 监听 `modules/specs/*.yaml`（改 YAML 版本号必须变化）；
  · 附带：导出器 fingerprint 纳入 specs/*.yaml；显式覆盖目录缺失不再假绿。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest.mock as mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PASSED = 0
FAILED = 0


def ok(cond, msg):
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  ✗ {msg}")


def _touch(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def test_auto_register_and_opgraph():
    tmpd = tempfile.mkdtemp()
    try:
        _touch(os.path.join(tmpd, "zz_dx_probe.yaml"),
               "op: zz_dx_probe\nimpl: exec_ops:_zz_dx_probe\n"
               "requires:\n  - key: hidden.x\n")
        env = os.environ.copy()
        env["DS41M_MODULE_SPEC_DIR"] = tmpd
        r = subprocess.run(
            [sys.executable, "-c",
             "import modules.ops; from modules.registry import MODULES; "
             "from modules.op_iface import OP_IFACE; import opgraph; "
             "print(int('zz_dx_probe' in OP_IFACE), "
             "int('zz_dx_probe' in MODULES), "
             "int('zz_dx_probe' in opgraph.OPS))"],
            cwd=ROOT, env=env, capture_output=True, text=True)
        ok(r.stdout.strip().splitlines()[-1:] == ["1 1 1"],
           f"YAML 新模块未自动注册 MODULES/opgraph：{r.stdout.strip()!r} "
           f"{r.stderr[-300:]}")
    finally:
        shutil.rmtree(tmpd)


def test_python_stub_single_element():
    op = "dx_stub_probe"
    stub = os.path.join(ROOT, "modules", "ops", "plugins", op + ".py")
    try:
        if os.path.exists(stub):
            os.unlink(stub)
        r = subprocess.run(
            [sys.executable, "tools/new_module.py", op, "--python-stub",
             "--requires", "hidden.x=[T,H]", "--force"],
            cwd=ROOT, capture_output=True, text=True)
        ok(r.returncode == 0 and "注册 OK" in r.stdout,
           f"--python-stub 生成/注册失败：{r.stdout[-300:]}{r.stderr[-300:]}")
        src = open(stub, encoding="utf-8").read() if os.path.exists(stub) else ""
        ok('K("hidden.x", "Tensor", "[T,H]"),' in src,
           f"单元素 requires 缺少尾逗号：{src[:200]}")
        r2 = subprocess.run(
            [sys.executable, "-c",
             "import modules.ops; from modules.registry import MODULES; "
             f"print(int('{op}' in MODULES))"],
            cwd=ROOT, capture_output=True, text=True)
        ok(r2.stdout.strip().splitlines()[-1:] == ["1"],
           f"--python-stub 生成物导入即崩：{r2.stdout.strip()!r} "
           f"{r2.stderr[-300:]}")
    finally:
        if os.path.exists(stub):
            os.unlink(stub)
        pyc_dir = os.path.join(ROOT, "modules", "ops", "plugins", "__pycache__")
        if os.path.isdir(pyc_dir):
            for n in os.listdir(pyc_dir):
                if n.startswith(op + "."):
                    os.unlink(os.path.join(pyc_dir, n))


def test_viewer_watches_yaml():
    if os.path.join(ROOT, "tools") not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
    import serve_viewer as viewer
    tmpd = tempfile.mkdtemp()
    try:
        for rel in ("recipes.yaml", "opgraph.py", "model_spec.py",
                    "weights_manifest.py", "cfg.yaml", "tpl.html",
                    "modules/specs/model_modules.yaml"):
            _touch(os.path.join(tmpd, rel), "x")
        cfg = os.path.join(tmpd, "cfg.yaml")
        tpl = os.path.join(tmpd, "tpl.html")
        with mock.patch.object(viewer, "ROOT", tmpd):
            st = viewer.State(cfg, tpl)
            v1 = st.source_version()
            yml = os.path.join(tmpd, "modules", "specs", "model_modules.yaml")
            _touch(yml, "modules:\n  - op: zz\n")
            v2 = st.source_version()
            files = viewer.watch_files(cfg, tpl)
        ok(yml in files, "watch_files 未监听 modules/specs/*.yaml")
        ok(v1 != v2, "改 YAML 后 viewer source_version 没变化（P0-5 回归）")
    finally:
        shutil.rmtree(tmpd)


def test_export_fingerprint_specs():
    import modules.export_model_html as exp
    tmpd = tempfile.mkdtemp()
    try:
        _touch(os.path.join(tmpd, "specs", "model_modules.yaml"), "a: 1\n")
        with mock.patch.object(exp, "HERE", tmpd):
            h1 = exp._sha_specs()
            _touch(os.path.join(tmpd, "specs", "modules.d", "zz.yaml"), "op: zz\n")
            h2 = exp._sha_specs()
        ok(h1 != h2, "export fingerprint 未纳入 modules.d YAML")
        ok(h1 not in ("", "missing"), "export specs fingerprint 为空")
    finally:
        shutil.rmtree(tmpd)


def test_missing_overlay_dir_not_green():
    from modules.spec_loader import load_spec_file
    bad = "/nonexistent/dx_modules_d"
    try:
        load_spec_file(overlay_dir=bad)
    except Exception as exc:                       # noqa: BLE001
        ok("不存在" in str(exc), f"缺失覆盖目录错误信息不明确：{exc}")
    else:
        ok(False, "缺失的 overlay_dir 被静默忽略（沙盒假绿）")


def main():
    test_auto_register_and_opgraph()
    test_python_stub_single_element()
    test_viewer_watches_yaml()
    test_export_fingerprint_specs()
    test_missing_overlay_dir_not_green()
    print(f"dx_regressions: {PASSED} passed, {FAILED} failed")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

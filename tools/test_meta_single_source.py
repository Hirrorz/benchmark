#!/usr/bin/env python3
"""C5 结构性不变量：**布局/`_packed` 只能由 `meta/metadata.py` 与其登记点写入**。

为什么需要它：本项目出过"compact 改了 layout 没还原 → 漏到下一个 pass → moe gate 吃 nan"
（WORKPLAN P1-1c）。那种 bug 在运行期很难定位，但**在写入点上是可枚举的**。
本测试用静态扫描把"谁能写"钉成 allowlist：

  1. `cfg["_packed"] = ...`（含单引号/带空格）只能出现在登记文件里；
  2. `.set_layout(` 调用只能出现在登记文件里；
  3. 允许清单里的每一项都必须在下面的注释里写明**原因**；
  4. 扫描结果打印出来 —— 新增写入点会让本测试 FAIL，逼迫走 review（铁律 0）。

登记（2026-11-01）：
  · `meta/metadata.py` —— 布局唯一 owner（`set_layout/restore_layout/stage_view`）；
  · `core/points_runner.py` —— 点位的**构建期**写 `_packed`（建 cfg，不是改运行中布局）；
  · `core/runner.py` —— CP/compact 的 `set_layout`（2 处，C5 待改成 `stage_view`）。
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 相对路径 → 原因
ALLOW = {
    "meta/metadata.py": "布局唯一 owner",
    "core/points_runner.py": "点位构建期写 _packed（建 cfg）",
    "core/runner.py": "CP/compact 的 set_layout（C5 待改 stage_view）",
    # ★ 2026-11-01：本文件原先是**第二写入点**（`ensure_packed` 直写 `cfg["_packed"]`），
    #   已改成经 `metadata.attach + set_layout`；现在它只是 set_layout 的合法调用方
    #   （dry-run/工具补默认单序列布局），不再直写镜像。
    "core/assemble.py": "dry-run/工具补默认单序列布局（经 metadata.set_layout）",
}

# 不扫的目录（产物/备份/运行时/第三方）
SKIP_DIRS = {".git", "tmp", "rt", "deps_py", "logs", "real_prof", "__pycache__",
             "ds41_runtime", "thirdparty", "parity", "configs", "modules"}

W_PACKED = re.compile(r"""cfg\[["']_packed["']\]\s*=""")
W_SETLAYOUT = re.compile(r"\.set_layout\(")
# ★ C5 收尾：命中策略也不许被就地整块覆盖（写入必须走 metadata.set_hit(_dict)）
W_HIT = re.compile(r"""cfg\[["']hit["']\]\s*=""")

# `cfg["hit"]` 直写的允许清单（只有 owner 自己写镜像）
ALLOW_HIT = {"meta/metadata.py"}

FAIL = []


def chk(name, cond, detail=""):
    print(("PASS" if cond else "FAIL") + " · " + name + ("" if cond else f"  <<< {detail}"))
    if not cond:
        FAIL.append(name)


def scan():
    hits = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = os.path.relpath(dirpath, ROOT).replace("\\", "/")
        top = rel.split("/")[0]
        if top in SKIP_DIRS or any(d in SKIP_DIRS for d in rel.split("/")):
            dirnames[:] = []
            continue
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            r = os.path.relpath(p, ROOT).replace("\\", "/")
            if r.startswith("tools/") and ("test_" in fn or fn in ("sweep.py", "halo_report.py")):
                continue                      # 测试/报告自身不算写入点
            try:
                txt = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for i, line in enumerate(txt.splitlines(), 1):
                if W_PACKED.search(line):
                    hits.append((r, i, "packed", line.strip()[:90]))
                if W_SETLAYOUT.search(line):
                    hits.append((r, i, "set_layout", line.strip()[:90]))
                if W_HIT.search(line):
                    hits.append((r, i, "hit", line.strip()[:90]))
    return hits


def main():
    hits = scan()
    print(f"== 扫描到 {len(hits)} 处写入点 ==")
    for r, i, kind, line in hits:
        print(f"  {kind:10s} {r}:{i}  {line}")

    files = {r for r, _i, _k, _l in hits}
    unknown = sorted(files - set(ALLOW))
    chk("没有 allowlist 之外的写入文件", not unknown, f"新增：{unknown}")
    for f, why in ALLOW.items():
        chk(f"登记文件存在且原因已写明：{f}", os.path.exists(os.path.join(ROOT, f)) and bool(why), why)

    packed_files = {r for r, _i, k, _l in hits if k == "packed"}
    chk("`_packed` 直写只出现在登记文件",
        packed_files <= set(ALLOW), f"{sorted(packed_files - set(ALLOW))}")
    setl_files = {r for r, _i, k, _l in hits if k == "set_layout"}
    chk("`set_layout` 调用只出现在登记文件",
        setl_files <= set(ALLOW), f"{sorted(setl_files - set(ALLOW))}")
    hit_files = {r for r, _i, k, _l in hits if k == "hit"}
    chk("`cfg[\"hit\"]` 直写只出现在 metadata（写入走 set_hit/set_hit_dict）",
        hit_files <= ALLOW_HIT, f"{sorted(hit_files - ALLOW_HIT)}")

    print()
    if FAIL:
        print(f"✗ meta_single_source: {len(FAIL)} 项失败")
        for n in FAIL:
            print(f"   - {n}")
        return 1
    print("✓ meta_single_source: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
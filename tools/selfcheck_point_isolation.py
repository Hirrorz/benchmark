#!/usr/bin/env python3
"""点位失败隔离的 CPU 自检（不碰卡）。

覆盖 4 条（对应 `core/points_runner.py` 的 2026-10-29 改动）：

  1. **结构**：`run_points` 的 for 体是**单个 Try**，barrier（`pass_barrier_sync`）在
     try 体**最前面**，except 里有 `continue`/`return` ——
     barrier 一旦不在最前面，某个 rank 提前抛异常就会少调一次，全体错位 → 整轮挂死；
  2. **续跑**：`_load_done_tags()` 读 `results.tsv` 的 tag；`DS41M_POINT_REDO=1` 时返回空；
  3. **watchdog**：超时后确实 `os._exit(9)`（子进程验证，不污染本进程）；
  4. **脚本**：`tools/finish_cp2cp4.sh` 把 `DS41M_POINT_TIMEOUT`/`MAX_FAILS` 传下去了。

跑法： python3 tools/selfcheck_point_isolation.py
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

import points_runner as PR                                   # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("  ✅ " if cond else "  ✗ ") + name + (f"：{extra}" if extra else ""))
    if not cond:
        FAILS.append(name)


def check_structure():
    src = open(os.path.join(ROOT, "core", "points_runner.py"), encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "run_points")
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    check("for 体是单一 Try", len(loop.body) == 1 and isinstance(loop.body[0], ast.Try),
          type(loop.body[0]).__name__)
    t = loop.body[0]
    check("有 except 处理器", len(t.handlers) == 1)
    # barrier 必须是 try 体的最早几条语句之一（在解析/建输入之前）
    early = [i for i, st in enumerate(t.body[:8])
             if any(isinstance(x, ast.Attribute) and x.attr == "pass_barrier_sync"
                    for x in ast.walk(st))]
    check("barrier 在 try 体最前面（前 8 条内）", bool(early), f"位置 {early}")
    kinds = [type(x).__name__ for x in ast.walk(t.handlers[0])
             if isinstance(x, (ast.Continue, ast.Return))]
    check("except 里有 continue/return（跳过而非中断）",
          "Continue" in kinds and "Return" in kinds, str(kinds))


def check_resume():
    d = tempfile.mkdtemp(prefix="ptdone_")
    p = os.path.join(d, "results.tsv")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("tag\tsteady_s\tms_per_forward\terrors\tts\n")
        fh.write("(81920_0.97)_r1\t1.23\t10.0\t0\t2026-10-29T00:00:00\n")
        fh.write("(65536_0.9)_r1\t2.00\t20.0\t0\t2026-10-29T00:01:00\n")
    got = PR._load_done_tags(d)
    check("续跑读到 2 个已完成点位（且不含表头）",
          got == {"(81920_0.97)_r1", "(65536_0.9)_r1"}, str(sorted(got)))
    os.environ["DS41M_POINT_REDO"] = "1"
    try:
        check("DS41M_POINT_REDO=1 时强制重采", PR._load_done_tags(d) == set())
    finally:
        os.environ.pop("DS41M_POINT_REDO", None)
    check("目录不存在时返回空集", PR._load_done_tags(os.path.join(d, "nope")) == set())


def check_fail_counts():
    d = tempfile.mkdtemp(prefix="ptfail_")
    PR._note_fail(d, "(512_0)_r1", rank=0)
    PR._note_fail(d, "(512_0)_r1", rank=1)          # 非 rank0 不写
    PR._note_fail(d, "(512_0)_r1", rank=0)
    got = PR._load_fail_counts(d)
    check("失败次数按 rank0 累计（非 rank0 不写）", got == {"(512_0)_r1": 2}, str(got))
    check("缺失文件 → 空 dict", PR._load_fail_counts(os.path.join(d, "nope")) == {})


def check_watchdog():
    code = (
        "import sys, time\n"
        f"sys.path.insert(0, {os.path.join(ROOT, 'core')!r})\n"
        "import points_runner as PR\n"
        "st = {'t0': time.time() - 10, 'pi': 7}\n"
        "PR._watchdog(st, 0.2, 0)\n"          # 正常不会返回
    )
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=30)
    dt = time.time() - t0
    check("watchdog 超时后退出码 = 9", r.returncode == 9, f"rc={r.returncode} {dt:.1f}s")


def check_script():
    p = os.path.join(ROOT, "tools", "finish_cp2cp4.sh")
    s = open(p, encoding="utf-8").read()
    check("finish 脚本传 DS41M_POINT_TIMEOUT",
          s.count("DS41M_POINT_TIMEOUT=") >= 2, f"{s.count('DS41M_POINT_TIMEOUT=')} 处")
    check("finish 脚本传 DS41M_POINT_MAX_FAILS", "DS41M_POINT_MAX_FAILS=" in s)


def main():
    print("== 点位失败隔离 selfcheck")
    check_structure()
    check_resume()
    check_fail_counts()
    check_watchdog()
    check_script()
    print("== 结果：" + ("全过 ✅" if not FAILS else f"失败 {len(FAILS)} 项 ❌ {FAILS}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""`run_points` 点位循环的 CPU 端到端自检（不碰卡）。

用一个**假 runner / 假 assemble** 真跑 `points_runner.run_points`，验证 2026-10-29 的
失败隔离确实生效：

  · 点 2 的 forward 抛异常 → **不中断整轮**：点 2 记为失败，点 3 照跑；
  · 返回码 9（"有跳过"），summary 里列出失败点位；
  · 重跑同一条命令（同一个 out 目录）→ 已完成的点 1/3 被跳过，点 2 重试
    （失败计数写进 `points_failed.tsv`）；
  · 失败到 `DS41M_POINT_MAX_ATTEMPTS` 次后连点 2 也跳过。

跑法： python3 tools/selfcheck_points_loop.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import types as _t

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    sys.path.insert(0, os.path.join(ROOT, _p))

import torch                                                   # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("  ✅ " if cond else "  ✗ ") + name + (f"：{extra}" if extra else ""))
    if not cond:
        FAILS.append(name)


def _make_env(tmp):
    """假 assemble / 假 runner / 假 args。"""
    asm = types.ModuleType("assemble")
    asm.__file__ = os.path.join(ROOT, "core", "assemble.py")
    asm.build_stack = lambda cfg: (
        [dict(idx=i, ratio=0, attn="win", is_kv_src=False, is_idx_src=False,
              is_candidate=False, engram=False, T=8) for i in range(2)],
        dict(T_main=8, cut=None, T_pass=8))
    asm.hit_m = lambda cfg, L: int(L * (1 - float(cfg["hit"]["ratio"])))
    asm.replay_tail_of = lambda cfg: 0
    asm.replay_keep_tokens = lambda cfg: 0
    sys.modules["assemble"] = asm

    calls = {"run": [], "barrier": [], "boom_T": None}

    def run_prefill_auto(cfg, source, T, *a_, **kw):
        calls["run"].append(T)
        if calls["boom_T"] and int(T) == int(calls["boom_T"]):
            raise RuntimeError("模拟 OOM")
        return {"errors": [], "x": torch.zeros(1)}

    R = types.ModuleType("fakerunner")
    R.__file__ = os.path.join(ROOT, "core", "runner.py")
    R._RANK = 0
    R._CTX_CACHE, R._SEQ_CACHE, R._MODULE_PLAN_CACHE = {}, {}, {}
    R.POINT_RELEASE = lambda: None
    R.make_input = lambda cfg, rows, seed=0: torch.zeros(int(rows), 4)
    R.make_profiler = lambda pdir, alc: None
    R.run_prefill_auto = run_prefill_auto
    R.pass_barrier_sync = lambda pi: calls["barrier"].append(pi)
    R.start_line_sync = lambda: False
    R._calls = calls

    class _Args(dict):
        __getattr__ = dict.get          # 缺的属性读成 None（全是 falsy，正合 argparse 默认）
        __setattr__ = dict.__setitem__

    a = _Args(points=os.path.join(tmp, "points.json"), iters=1, warmup_iters=0,
              out=tmp, prof=False, prof_ranks="all", allow_placeholder=True,
              alop_config=None, world=1, input_seed=0)
    cfg = {"run": {"seq_len": 8, "device": "npu:0", "iters": 1,
                   "moe_chunk_max_len": 8192},
           "model": {"n_layers": 2}, "hit": {}, "features": {}, "parallel_config": {}}
    return asm, R, a, cfg


def _run(tmp, boom_t, max_attempts=3):
    asm, R, a, cfg = _make_env(tmp)
    R._calls["boom_T"] = boom_t
    import points_runner
    os.environ["DS41M_UNIFY_PACKED"] = "0"
    os.environ["DS41M_POINT_MAX_ATTEMPTS"] = str(max_attempts)
    os.environ.pop("DS41M_POINT_REDO", None)
    return points_runner.run_points(cfg, object(), a, "npu:0", runner_mod=R)


def _tag(L: int, h: float, r: int = 0, b: int = 1) -> str:
    """标准扁平名（假 runner 的 cfg 里 world=1/cp=1/ep=1）。"""
    for unit, div in (("m", 1 << 20), ("k", 1 << 10)):
        if L >= div and L % div == 0:
            lt = f"{L // div}{unit}"
            break
    else:
        lt = str(L)
    return f"cp1dp1ep1_L{lt}_hit{h:g}_batch{b}_replay{'on' if r else 'off'}_prefill"


def _results(tmp):
    p = os.path.join(tmp, "results.tsv")
    if not os.path.exists(p):
        return []
    return [ln.split("\t")[0] for ln in open(p).read().splitlines()[1:] if ln]


def main():
    print("== 点位循环失败隔离 selfcheck")
    tmp = tempfile.mkdtemp(prefix="ptloop_")
    with open(os.path.join(tmp, "points.json"), "w") as fh:
        json.dump([{"L": 1024, "hit": 0.0, "batches": 1, "replay": False},
                   {"L": 2048, "hit": 0.0, "batches": 1, "replay": False},
                   {"L": 4096, "hit": 0.0, "batches": 1, "replay": False}], fh)

    # torch.npu.* 在无卡环境里可能抛；这里只关心控制流，直接打桩。
    orig = (torch.npu.synchronize, torch.npu.empty_cache,
            torch.npu.reset_peak_memory_stats, torch.npu.max_memory_allocated)
    torch.npu.synchronize = lambda *a, **k: None
    torch.npu.empty_cache = lambda *a, **k: None
    torch.npu.reset_peak_memory_stats = lambda *a, **k: None
    torch.npu.max_memory_allocated = lambda *a, **k: 0
    try:
        # ① 点 2 抛异常（第 1 个 forward = 点 1；第 2 个 = 点 2；iters=1）
        rc = _run(tmp, boom_t=2048)
        done = _results(tmp)
        check("点 2 失败不中断：点 1/3 已完成", done == [_tag(1024, 0.0), _tag(4096, 0.0)],
              str(done))
        check("返回码 = 9（有跳过）", rc == 9, f"rc={rc}")
        check("失败计数落盘 points_failed.tsv",
              open(os.path.join(tmp, "points_failed.tsv")).read().strip()
              == _tag(2048, 0.0) + "\t1",
              open(os.path.join(tmp, "points_failed.tsv")).read().strip())

        # ② 重跑：已完成的两点跳过，点 2 重试（这次成功）
        rc = _run(tmp, boom_t=None)
        done = _results(tmp)
        check("重跑后三个点齐全（点 2 被补上）",
              sorted(done) == [_tag(1024, 0.0), _tag(2048, 0.0), _tag(4096, 0.0)], str(done))
        check("重跑 rc=0", rc == 0, f"rc={rc}")

        # ③ 一直失败：到 MAX_ATTEMPTS 后连失败点也跳过（不会再堵住后面的点）
        tmp2 = tempfile.mkdtemp(prefix="ptloop2_")
        with open(os.path.join(tmp2, "points.json"), "w") as fh:
            json.dump([{"L": 1024, "hit": 0.0, "batches": 1, "replay": False},
                       {"L": 2048, "hit": 0.0, "batches": 1, "replay": False}], fh)
        _run(tmp2, boom_t=1024, max_attempts=1)      # 第 1 次：点 1 失败、点 2 完成
        rc = _run(tmp2, boom_t=1024, max_attempts=1)  # 第 2 次：点 1 达上限 → 直接跳过
        check("达次数上限后跳过失败点（第 2 次不再尝试它）",
              _results(tmp2) == [_tag(2048, 0.0)], str(_results(tmp2)))
        check("第 2 次 rc=0（无新增失败）", rc == 0, f"rc={rc}")
    finally:
        (torch.npu.synchronize, torch.npu.empty_cache,
         torch.npu.reset_peak_memory_stats, torch.npu.max_memory_allocated) = orig
        for k in ("DS41M_UNIFY_PACKED", "DS41M_POINT_MAX_ATTEMPTS"):
            os.environ.pop(k, None)

    print("== 结果：" + ("全过 ✅" if not FAILS else f"失败 {len(FAILS)} 项 ❌ {FAILS}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())

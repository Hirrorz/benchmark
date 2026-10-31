# coding=utf-8
"""Profiling 工厂：ALOP 是唯一采集后端（PERF_RULES §0 铁律）。

runner 只依赖本模块的 `make_profiler(out_dir, alop_config)`，不再关心 ALOP 的构造细节。
默认档位用 `configs/alop_mindstudio.yaml`（纯 aclprof C 级，`python_profiling` 永久禁用）。
"""
from __future__ import annotations

import os

__all__ = ["make_profiler"]


def make_profiler(prof_dir: str, alop_config_path: str = ""):
    """创建 ALOP profiler（接口与 torch_npu.profiler 一致：start/step/stop）。

    ★ 用 ALOP 原生后端（aclprof C API），`python_profiling_config.enabled` 必须为 false
      （2026-10-26 用户规则：torch_npu.profiler 永久禁用，置 true 直接 raise）。
      产出 `PROF_*` 原始目录；MindStudio 用的 `mindstudio_profiler_output/` 由
      `msprof --export` 事后转出（见 PIPELINE.md / TRACE_GUIDE.md）。
    ★ 通过 cli_overrides 注入 out_dir：start() 内 `_apply_cli_overrides` 在
      `_reload_config_if_needed` 之后执行，保证 yaml 回读不会把输出路径冲掉。
    """
    from alop import ALOPProfiler, ALOPConfig  # 同目录，直接 import

    if alop_config_path and os.path.exists(alop_config_path):
        cfg = ALOPConfig.from_file(alop_config_path) or ALOPConfig()
        cfg_path = alop_config_path
    else:
        cfg = ALOPConfig()
        cfg_path = None
    # ★ 2026-10-26 用户规则：torch_npu.profiler **永久禁用**（PERF_RULES §0）。
    _pp = getattr(cfg, "python_profiling", None) or {}
    if isinstance(_pp, dict) and _pp.get("enabled"):
        raise RuntimeError(
            "torch_npu.profiler 永久禁用（PERF_RULES 2026-10-26）：请把 "
            "python_profiling_config.enabled 设为 false，改用 ALOP 的 ACL C 级采集")
    os.makedirs(prof_dir, exist_ok=True)
    return ALOPProfiler(cfg, config_path=cfg_path, cli_overrides={"out_dir": prof_dir})

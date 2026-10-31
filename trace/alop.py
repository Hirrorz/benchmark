#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALOP 性能采集工具 - 基于 aclprof API (CANN 8.5.0) — 独立解耦版

用户友好的简易化 Profiling 采集工具，支持：
1. YAML/字典 配置文件驱动，灵活更改采集配置
2. 一键 start/stop 启停采集
3. 上下文管理器 with 语法
4. 装饰器 @alop_profile 语法
5. 一次 init，多次修改配置（每次 start 前自动重新读取配置）
6. Python 级采集 (torch_npu.profiler)：启用后作为唯一采集引擎，
   同时完成 C 级（NPU 硬件指标）和 Python 级（调用栈、Module 层次、
   Tensor 形状、内存事件、FLOPs 估算）采集，避免双引擎并行开销

【两种采集模式】（由配置 python_profiling_config.enabled 自动选择）
  - alop:         aclprof C API 采集（C 级硬件指标，轻量级，默认）
  - python_prof:  torch_npu.profiler 采集（C+Python 级一体，调用栈模式）

【故障隔离保证】
ALOP 的任何失败（配置文件缺失/格式错误、CANN 库加载失败、API 调用返回非零、
torch_npu 不可用等）**仅记录日志，绝不抛异常**，确保被采集程序主流程不受影响。

【日志】
本独立版用 Python 标准库 logging 自建 logger，输出到 stdout；
设置环境变量 ALOP_LOG_FILE=/path/to/alop.log 可额外落盘（100MB×5 切片）。
不依赖任何框架。

【输出目录】
默认 <当前目录>/alop_output/alop/data/<namespace>/<ip>/<name>_<dev>_<pid>/alop_<时间戳>/
可通过环境变量 FINAL_DUMP_DEST_DIR 改根目录，通过配置 ip/name 标识本机/本服务。

使用示例:

    # 方式1: 一键启停
    profiler = ALOPProfiler.from_file("alop_config.yaml")
    profiler.start(device=0)
    # ... 运行推理 ...
    profiler.stop()
    profiler.finalize()

    # 方式2: 上下文管理器
    with ALOPProfiler.from_file("alop_config.yaml") as profiler:
        profiler.start(device=0)
        # ... 运行推理 ...

    # 方式3: 装饰器（profiling 失败时被装饰函数仍正常执行）
    @alop_profile("alop_config.yaml", device=0)
    def inference():
        # ... 推理代码 ...

    # 方式4: 启用 Python 级采集（torch_npu.profiler 作为唯一引擎）
    # 在 alop_config.yaml 中设置 python_profiling_config.enabled: true
    # 启用后自动替代 aclprof，同时完成 C 级 + Python 级采集
    profiler = ALOPProfiler.from_file("alop_config.yaml")
    profiler.start(device=0)
    # ... 运行推理 ...
    profiler.stop()

【CLI 直接调用】（shell 入口 alop / alop.sh）
    配置可通过环境变量 export 注入，命令行选项也可直接覆盖：

        export ALOP_OUT_DIR=/data/alop_results    # 采集数据落盘目录
        export ALOP_DEVICE=1                      # 目标 NPU 卡号
        export ALOP_METRIC=MemoryBandwidth        # AI Core 指标
        export ALOP_CONFIG=/path/alop_config.yaml # 配置文件
        alop --out-dir=/data/alop_results ops.py  # 直接采集运行 ops.py
        alop ops.py a b                           # 等价简写（首参即脚本）
"""

import os
import sys
import ctypes
import time
import yaml
import json
import shutil
import logging
import argparse
import datetime
import threading
import traceback
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from functools import wraps
from pathlib import Path
from enum import IntEnum
from logging.handlers import RotatingFileHandler


# ============================================================================
# 日志：标准库自建，零框架依赖
# ============================================================================

def _init_logger() -> logging.Logger:
    """创建 ALOP 专用 logger。

    - 始终输出到 stdout；
    - 若设置了环境变量 ALOP_LOG_FILE（绝对路径），额外落盘到该文件，100MB×5 切片。
    """
    _logger = logging.getLogger("alop")
    if _logger.handlers:
        return _logger
    _logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(levelname)s %(asctime)s.%(msecs)03d %(process)d %(filename)s:%(lineno)d] %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(fmt)
    _logger.addHandler(sh)

    log_file = os.environ.get("ALOP_LOG_FILE", "")
    if log_file:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            fh = RotatingFileHandler(
                log_file, maxBytes=100 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            _logger.addHandler(fh)
        except Exception as e:
            # 文件 handler 建失败不影响 stdout 输出
            print(f"[alop] ALOP_LOG_FILE handler 初始化失败，仅输出到 stdout: {e}", file=sys.stderr)

    _logger.propagate = False
    return _logger


logger = _init_logger()


def is_torch_npu_available() -> bool:
    """检测 torch_npu.profiler 是否可用（公开接口，委托 _TorchNpuProfiler.available()）。"""
    return _TorchNpuProfiler.available()


# ============================================================================
# 落盘根目录：读取环境变量 FINAL_DUMP_DEST_DIR，未设置则回退默认路径
# 所有 log/data/config 统一放在 $root/alop/{log,data,config} 下
# ============================================================================

# 独立版友好默认：当前工作目录下 alop_output，不再绑定机器特定路径
_ALOP_FALLBACK_ROOT = os.path.join(os.getcwd(), "alop_output")


def resolve_alop_root() -> str:
    """返回 ALOP 落盘根目录。

    优先读取环境变量 FINAL_DUMP_DEST_DIR；未设置时回退到
    <当前工作目录>/alop_output。调用方应在进程启动后尽早
    调用并缓存结果，避免每次 I/O 都读取环境变量。
    """
    return os.environ.get("FINAL_DUMP_DEST_DIR") or _ALOP_FALLBACK_ROOT


# data 多级目录的一级 namespace：读取环境变量 ALOP_NAMESPACE，未设置时回退 "default"。
# 最终 data 路径形如 $root/alop/data/<namespace>/<ip>/<name>_<dev_id>_<pid>/alop_<ts>/PROF*
_ALOP_FALLBACK_NAMESPACE = "default"


def resolve_alop_namespace() -> str:
    """返回 ALOP data 目录的一级 namespace。

    优先读取环境变量 ALOP_NAMESPACE；未设置时回退到 "default"。
    显式设为空字符串时返回 ""，build_output_dir 会跳过 namespace 层级
    （用于扁平化输出目录，如整网推理按 rank 落盘场景）。
    用于按服务/部署实例隔离不同采集的数据，避免多套服务混在同一目录下。
    """
    ns = os.environ.get("ALOP_NAMESPACE")
    if ns is None:
        return _ALOP_FALLBACK_NAMESPACE
    return ns


def _alop_subdir(name: str) -> str:
    """返回 $root/alop/<name> 子目录路径（name ∈ data/config/log）。"""
    return os.path.join(resolve_alop_root(), "alop", name)


def build_output_dir(
    root: Optional[str] = None,
    namespace: Optional[str] = None,
    ip: str = "unknown",
    name: str = "alop",
    dev_id: int = 0,
    pid: Optional[int] = None,
    timestamp: Optional[str] = None,
) -> str:
    """构建本次采集的多级输出目录。

    结构：$root/alop/data/$namespace/$ip/${name}_${dev_id}_${pid}/alop_${ts}
    namespace / ip 为空的段会被跳过（os.path.join 忽略空段），
    例如两者都为空时得到 $root/alop/data/${name}_${dev_id}_${pid}/alop_${ts}。

    扁平模式（环境变量 ALOP_FLAT_DEV_LAYOUT=true）：直接返回
    $root/alop/data/<dev_id>/<ts>，dev 单独一层、时间戳不带 alop_ 前缀，
    无 namespace/ip/name/pid 层。用于整网推理按 rank 落盘场景。

    Args:
        root: 落盘根目录，None 则用 resolve_alop_root()
        namespace: 一级 namespace，None 则用 resolve_alop_namespace()；显式传 "" 表示跳过该层级
        ip: 机器 IP 或标识（配置提供，缺省 "unknown"），传 "" 表示跳过该层级
        name: 服务/模型名（配置提供，缺省 "alop"）
        dev_id: 设备 ID
        pid: 进程 ID，None 则用 os.getpid()
        timestamp: 时间戳字符串 YYYYMMDD_HHMMSS，None 则取当前时间

    Returns:
        绝对路径字符串（尚未创建，调用方自行 makedirs）
    """
    root = root or resolve_alop_root()
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # rank 布局：$root/rank<dev>_<ts>（root 即 "alop_data" 目录），一张卡一个目录。
    # 最终形如 alop_data/rank0_20261014_155500/PROF_*，无 alop/data/ 中间层。
    if os.environ.get("ALOP_LAYOUT", "").lower() in ("rank", "1", "true", "yes"):
        return os.path.join(root, f"rank{dev_id}_{timestamp}")
    # 扁平 dev 布局：$root/alop/data/<dev>/<ts>，忽略 namespace/ip/name/pid
    if os.environ.get("ALOP_FLAT_DEV_LAYOUT", "").lower() in ("true", "1", "yes"):
        return os.path.join(root, "alop", "data", str(dev_id), timestamp)
    namespace = resolve_alop_namespace() if namespace is None else namespace
    if pid is None:
        pid = os.getpid()
    return os.path.join(
        root, "alop", "data", namespace, ip, f"{name}_{dev_id}_{pid}", f"alop_{timestamp}"
    )


# ============================================================================
# 环境变量 → CLI 默认值（shell 入口 `alop --out-dir=xxx ops.py` 场景）
# ============================================================================
# 优先级：命令行参数 > 环境变量 > 配置文件 > 内置默认。CLI 在解析时读取
# 这些环境变量作为 argparse default，配置加载后仅当命令行未显式给出时才
# 用环境变量覆盖配置文件（配置文件优先级高于环境变量）。

_ALOP_ENV_DEFAULTS = {
    "device": "ALOP_DEVICE",
    "metric": "ALOP_METRIC",
    "config": "ALOP_CONFIG",
    "out_dir": "ALOP_OUT_DIR",
}


def env_default(key: str, fallback: Any = None) -> Any:
    """取环境变量作为 argparse 默认值：var 未设置时回退 fallback。"""
    var = _ALOP_ENV_DEFAULTS.get(key)
    if not var:
        return fallback
    val = os.environ.get(var)
    return val if val else fallback


def _cmdline_opt_set(argv: Optional[List[str]], opt_name: str) -> bool:
    """判断命令行是否显式给出了 --opt_name（用于区分 CLI 显式值与默认值）。"""
    if argv is None:
        argv = sys.argv[1:]
    for a in argv:
        if a == f"--{opt_name}" or a.startswith(f"--{opt_name}="):
            return True
    return False


__all__ = [
    "ALOPConfig",
    "ALOPProfiler",
    "TorchNpuProfilerBackend",
    "AiCoreMetric",
    "DataTypeConfig",
    "SetConfigType",
    "alop_profile",
    "generate_default_config",
    "init_alop",
    "is_torch_npu_available",
    "resolve_alop_root",
    "resolve_alop_namespace",
    "build_output_dir",
]


# ============================================================================
# torch_npu.profiler 可选依赖 — 懒加载封装
# ============================================================================

class _TorchNpuProfiler:
    """封装 torch_npu.profiler 的懒加载与可用性检测。

    torch_npu 为可选依赖：未安装时 available=False，所有方法静默返回 None。
    安装后 available=True，提供与 torch_npu.profiler.profile 一致的创建接口。

    当 python_profiling.enabled=True 时，ALOP 使用 torch_npu.profiler
    作为**唯一采集引擎**（替代 aclprof C API），由它同时完成 C 级和
    Python 级采集，避免双引擎并行带来的额外开销。

    故障隔离：import 失败或 API 调用异常均不向调用方传播，
    仅记录日志，确保 ALOP 在无 torch_npu 环境下正常工作。
    """

    _available: Optional[bool] = None   # None=未检测, True/False=已检测
    _profiler_module = None             # torch_npu.profiler 模块引用

    @classmethod
    def available(cls) -> bool:
        """检测 torch_npu.profiler 是否可用（仅检测一次，结果缓存）。"""
        if cls._available is None:
            try:
                import torch_npu.profiler as _prof  # noqa: F401
                cls._profiler_module = _prof
                cls._available = True
                logger.info("[ALOP] torch_npu.profiler 可用")
            except ImportError:
                cls._available = False
                logger.warning(
                    "[ALOP] torch_npu.profiler 不可用"
                    "（torch_npu 未安装，将回退到 C 级 aclprof 采集）"
                )
            except Exception as e:
                cls._available = False
                logger.error(f"[ALOP] torch_npu.profiler 加载异常: {e}")
        return cls._available

    @classmethod
    def create_profile(
        cls,
        activities: Optional[list] = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        experimental_config=None,
        on_trace_ready=None,
    ):
        """创建 torch_npu.profiler.profile 实例。不可用或失败时返回 None。"""
        if not cls.available():
            return None
        try:
            kwargs = dict(
                activities=activities,
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack,
                with_flops=with_flops,
                with_modules=with_modules,
            )
            if experimental_config is not None:
                kwargs["experimental_config"] = experimental_config
            if on_trace_ready is not None:
                kwargs["on_trace_ready"] = on_trace_ready
            prof = cls._profiler_module.profile(**kwargs)
            return prof
        except Exception as e:
            logger.error(f"[ALOP] torch_npu.profiler.profile() 创建失败: {e}")
            return None

    @classmethod
    def profiler_activity(cls):
        """返回 ProfilerActivity 枚举类。不可用时返回 None。"""
        if not cls.available():
            return None
        return cls._profiler_module.ProfilerActivity

    @classmethod
    def build_experimental_config(
        cls,
        aicore_metric: str = "PipeUtilization",
        data_types: Optional[Dict[str, bool]] = None,
    ):
        """根据 ALOP 的 aicore_metric 和 data_types 构建 _ExperimentalConfig。

        将 ALOP 的 C 级配置映射到 torch_npu.profiler 的 _ExperimentalConfig，
        使 torch_npu.profiler 作为唯一引擎时仍能应用用户的 C 级采集设置。

        Args:
            aicore_metric: ALOP 配置中的 aicore_metric 名称
            data_types: ALOP 配置中的 data_types 字典

        Returns:
            _ExperimentalConfig 实例，或 None（如果 torch_npu 不可用）
        """
        if not cls.available():
            return None
        try:
            # 映射 ALOP aicore_metric → torch_npu AiCMetrics
            _METRIC_MAP = {
                "ArithmeticUtilization": "ACL_AICORE_ARITHMETIC_UTILIZATION",
                "PipeUtilization": "ACL_AICORE_PIPE_UTILIZATION",
                "MemoryBandwidth": "ACL_AICORE_MEMORY_BANDWIDTH",
                "L0BBandwidth": "ACL_AICORE_L0B_AND_WIDTH",
                "ResourceConflictRatio": "ACL_AICORE_RESOURCE_CONFLICT_RATIO",
                "MemoryUB": "ACL_AICORE_MEMORY_UB",
                "L2Cache": "ACL_AICORE_L2_CACHE",
                "PipeExecuteUtilization": "ACL_AICORE_PIPE_UTILIZATION",
                "MemoryAccess": "ACL_AICORE_MEMORY_ACCESS",
            }
            aic_metrics_str = _METRIC_MAP.get(
                aicore_metric, "ACL_AICORE_PIPE_UTILIZATION"
            )

            # 确定 profiler_level
            # l2cache 或 op_attr 开启 → Level2，aicore_metrics 开启 → Level1，否则 Level0
            dt = data_types or {}
            profiler_level_str = "Level0"
            if dt.get("l2cache") or dt.get("op_attr"):
                profiler_level_str = "Level2"
            elif dt.get("aicore_metrics"):
                profiler_level_str = "Level1"

            exp_cfg = cls._profiler_module._ExperimentalConfig(
                profiler_level=profiler_level_str,
                aic_metrics=aic_metrics_str,
                l2_cache=bool(dt.get("l2cache", False)),
                data_simplification=True,
                record_op_args=False,
                op_attr=bool(dt.get("op_attr", False)),
                export_type=cls._profiler_module.ExportType.Text,
            )
            return exp_cfg
        except Exception as e:
            logger.warning(f"[ALOP] 构建 _ExperimentalConfig 失败，使用默认配置: {e}")
            return None


# ============================================================================
# 日志存根：历史调用点兼容，独立版不再有自建文件 handler 概念
# ============================================================================


def _setup_file_logger(output_dir: str = None) -> None:
    """DEPRECATED 空操作。

    独立版 logger 由 _init_logger() 统一管理（stdout + 可选 ALOP_LOG_FILE），
    此处不再做任何事。保留签名仅为兼容既有调用点（传入的 output_dir 被忽略）。
    """
    return None


def _tf(value) -> str:
    """布尔值 → yaml 小写字符串，集中处理以降低调用方圈复杂度。"""
    return "true" if value else "false"


# ============================================================================
# 故障隔离装饰器：所有 ALOP 操作永不向调用方抛异常
# ============================================================================

def _safe(fn):
    """装饰器：将函数内所有异常转为 logger.error，返回 None 而非抛出。

    ALOP 的任何失败（配置错误、API 调用失败、库加载失败等）只记录日志，
    绝不中断被采集程序主流程。失败时在日志里**显式带上完整堆栈**（拼进
    消息正文，确保无论 formatter 行为如何堆栈都写入日志），同时传
    exc_info=True 拿到结构化异常信息。
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = f"[ALOP] {fn.__name__} 失败（不影响被采集程序）: {e}\n{traceback.format_exc()}"
            logger.error(msg, exc_info=True)
            return None
    return wrapper


# ============================================================================
# CANN 8.5.0 aclprof 枚举常量定义
# ============================================================================

class AiCoreMetric(IntEnum):
    ARITHMETIC_UTILIZATION = 0
    PIPE_UTILIZATION = 1
    MEMORY_BANDWIDTH = 2
    L0B_BANDWIDTH = 3
    RESOURCE_CONFLICT_RATIO = 4
    MEMORY_UB = 5
    L2_CACHE = 6
    PIPE_EXECUTE_UTILIZATION = 7
    MEMORY_ACCESS = 8
    NONE = 0xFF

    @classmethod
    def _names_map(cls) -> Dict[int, str]:
        return {
            cls.ARITHMETIC_UTILIZATION: "ArithmeticUtilization",
            cls.PIPE_UTILIZATION: "PipeUtilization",
            cls.MEMORY_BANDWIDTH: "MemoryBandwidth",
            cls.L0B_BANDWIDTH: "L0BBandwidth",
            cls.RESOURCE_CONFLICT_RATIO: "ResourceConflictRatio",
            cls.MEMORY_UB: "MemoryUB",
            cls.L2_CACHE: "L2Cache",
            cls.PIPE_EXECUTE_UTILIZATION: "PipeExecuteUtilization",
            cls.MEMORY_ACCESS: "MemoryAccess",
            cls.NONE: "None",
        }

    def display_name(self) -> str:
        return self._names_map().get(self.value, f"Unknown({self.value})")


class DataTypeConfig(IntEnum):
    ACL_API = 0x0001
    TASK_TIME = 0x0002
    AICORE_METRICS = 0x0004
    AICPU = 0x0008
    L2CACHE = 0x0010
    HCCL_TRACE = 0x0020
    TRAINING_TRACE = 0x0040
    MSPROFTX = 0x0080
    RUNTIME_API = 0x0100
    GE_API_L0 = 0x0200
    TASK_TIME_L0 = 0x0800
    TASK_MEMORY = 0x1000
    TASK_TIME_L2 = 0x2000
    OP_ATTR = 0x4000
    TASK_TIME_L3 = 0x8000

    @classmethod
    def _names_map(cls) -> Dict[int, str]:
        return {
            cls.ACL_API: "ACL_API",
            cls.TASK_TIME: "TaskTime",
            cls.AICORE_METRICS: "AiCoreMetrics",
            cls.AICPU: "AICPU",
            cls.L2CACHE: "L2Cache",
            cls.HCCL_TRACE: "HCCLTrace",
            cls.TRAINING_TRACE: "TrainingTrace",
            cls.MSPROFTX: "MsprofTX",
            cls.RUNTIME_API: "RuntimeAPI",
            cls.GE_API_L0: "GEAPI_L0",
            cls.TASK_TIME_L0: "TaskTime_L0",
            cls.TASK_MEMORY: "TaskMemory",
            cls.TASK_TIME_L2: "TaskTime_L2",
            cls.OP_ATTR: "OpAttr",
            cls.TASK_TIME_L3: "TaskTime_L3",
        }

    def display_name(self) -> str:
        return self._names_map().get(self.value, f"0x{self.value:X}")


class SetConfigType(IntEnum):
    STORAGE_LIMIT = 1
    AIV_METRICS = 2
    SYS_HARDWARE_MEM_FREQ = 3
    LLC_MODE = 4
    SYS_IO_FREQ = 5
    SYS_INTERCONNECTION_FREQ = 6
    DVPP_FREQ = 7
    HOST_SYS = 8
    HOST_SYS_USAGE = 9
    HOST_SYS_USAGE_FREQ = 10
    LOW_POWER_FREQ = 11
    SYS_MEM_SERVICEFLOW = 12
    SYS_CPU_FREQ = 13
    SCALE = 14


# 配置名称 → 枚举值 映射（用于 YAML/JSON 配置解析）
_DATA_TYPE_NAME_MAP: Dict[str, DataTypeConfig] = {
    "acl_api": DataTypeConfig.ACL_API,
    "task_time": DataTypeConfig.TASK_TIME,
    "aicore_metrics": DataTypeConfig.AICORE_METRICS,
    "aicpu": DataTypeConfig.AICPU,
    "l2cache": DataTypeConfig.L2CACHE,
    "hccl_trace": DataTypeConfig.HCCL_TRACE,
    "training_trace": DataTypeConfig.TRAINING_TRACE,
    "msproftx": DataTypeConfig.MSPROFTX,
    "runtime_api": DataTypeConfig.RUNTIME_API,
    "ge_api_l0": DataTypeConfig.GE_API_L0,
    "task_time_l0": DataTypeConfig.TASK_TIME_L0,
    "task_memory": DataTypeConfig.TASK_MEMORY,
    "task_time_l2": DataTypeConfig.TASK_TIME_L2,
    "op_attr": DataTypeConfig.OP_ATTR,
    "task_time_l3": DataTypeConfig.TASK_TIME_L3,
}

_AICORE_METRIC_NAME_MAP: Dict[str, AiCoreMetric] = {
    "ArithmeticUtilization": AiCoreMetric.ARITHMETIC_UTILIZATION,
    "PipeUtilization": AiCoreMetric.PIPE_UTILIZATION,
    "MemoryBandwidth": AiCoreMetric.MEMORY_BANDWIDTH,
    "L0BBandwidth": AiCoreMetric.L0B_BANDWIDTH,
    "ResourceConflictRatio": AiCoreMetric.RESOURCE_CONFLICT_RATIO,
    "MemoryUB": AiCoreMetric.MEMORY_UB,
    "L2Cache": AiCoreMetric.L2_CACHE,
    "PipeExecuteUtilization": AiCoreMetric.PIPE_EXECUTE_UTILIZATION,
    "MemoryAccess": AiCoreMetric.MEMORY_ACCESS,
}

# sampling 键名 → SetConfigType 映射
_SAMPLING_CONFIG_MAP: Dict[str, SetConfigType] = {
    "hardware_mem_freq": SetConfigType.SYS_HARDWARE_MEM_FREQ,
    "llc_mode": SetConfigType.LLC_MODE,
    "io_freq": SetConfigType.SYS_IO_FREQ,
    "interconnection_freq": SetConfigType.SYS_INTERCONNECTION_FREQ,
    "host_sys": SetConfigType.HOST_SYS,
    "host_sys_usage": SetConfigType.HOST_SYS_USAGE,
    "host_sys_usage_freq": SetConfigType.HOST_SYS_USAGE_FREQ,
}


# ============================================================================
# 采集配置数据类
# ============================================================================

@dataclass
class ALOPConfig:
    output_dir: str = field(default_factory=lambda: _alop_subdir("data"))
    # 多级目录的标识字段：ip/name 用于在 data 树里区分机器/服务
    ip: str = "unknown"
    name: str = "alop"
    aicore_metric: str = "PipeUtilization"
    data_types: Dict[str, bool] = field(default_factory=lambda: {
        "acl_api": False,
        "task_time": True,
        "aicore_metrics": True,
        "aicpu": False,
        "l2cache": False,
        "hccl_trace": False,
        "training_trace": False,
        "msproftx": False,
        "runtime_api": False,
        "task_time_l0": False,
        "task_memory": False,
        "task_time_l2": False,
        "op_attr": False,
        "task_time_l3": False,
    })
    sampling: Dict[str, Any] = field(default_factory=dict)
    dashcam: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "profiling_data_limit_gb": 500,   # profiling_data 总占用上限 GB，0=禁用
        "free_disk_threshold_gb": 200,    # 剩余可用空间阈值 GB，0=禁用
    })
    python_profiling: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,           # 启用后替代 aclprof，作为唯一采集引擎
        "record_shapes": False,     # 记录算子输入 Tensor 形状
        "profile_memory": False,    # 记录内存分配/释放事件
        "with_stack": False,        # 记录 Python/C++ 调用栈
        "with_flops": False,        # 估算算子 FLOPs
        "with_modules": False,      # 记录 nn.Module 层次结构
    })
    sleep_interval: float = 0.1

    @classmethod
    @_safe
    def from_file(cls, path: str) -> "ALOPConfig":
        """从 YAML/JSON 文件自动识别格式并加载。失败时返回 None。"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        if p.suffix in (".yaml", ".yml"):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        elif p.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError(f"不支持的配置格式: {p.suffix}，请使用 .yaml/.yml/.json")
        return cls.from_dict(data)

    @classmethod
    @_safe
    def from_dict(cls, data: Dict[str, Any]) -> "ALOPConfig":
        """从字典构建配置，仅接受已知字段。失败时返回 None。"""
        cfg = cls()
        if "output_dir" in data:
            cfg.output_dir = str(data["output_dir"])
        if "ip" in data:
            cfg.ip = str(data["ip"])
        if "name" in data:
            cfg.name = str(data["name"])
        if "sleep_interval" in data:
            cfg.sleep_interval = float(data["sleep_interval"])

        cls._parse_profiling_config(cfg, data.get("profiling_config", {}))
        cls._parse_sampling_config(cfg, data)
        cls._parse_dashcam_config(cfg, data.get("dashcam_config"))
        cls._parse_python_profiling_config(cfg, data.get("python_profiling_config"))

        return cfg

    @staticmethod
    def _parse_profiling_config(cfg, pc: Dict[str, Any]) -> None:
        """profiling_config 段 → aicore_metric / data_types。"""
        if "aicore_metric" in pc:
            cfg.aicore_metric = pc["aicore_metric"]
        if "data_types" in pc and isinstance(pc["data_types"], dict):
            for k, v in pc["data_types"].items():
                if k in _DATA_TYPE_NAME_MAP:
                    cfg.data_types[k] = bool(v)

    @staticmethod
    def _parse_sampling_config(cfg, data: Dict[str, Any]) -> None:
        """sampling_config 段 → cfg.sampling（aclprofSetConfig）。

        缺省保留默认；显式 null 清空；dict 仅取已知键。
        """
        if "sampling_config" not in data:
            return  # 未写 → 保留默认值
        if data["sampling_config"] is None:
            cfg.sampling = {}  # 显式 null → 清空，不调用 setConfig
            return
        sc = data["sampling_config"]
        if isinstance(sc, dict):
            cfg.sampling = {k: v for k, v in sc.items() if k in _SAMPLING_CONFIG_MAP}

    @staticmethod
    def _parse_dashcam_config(cfg, dc) -> None:
        """dashcam_config 段 → cfg.dashcam。"""
        if dc is None:
            cfg.dashcam = {"enabled": False}  # 显式 null 或未写 → 禁用
            return
        if isinstance(dc, dict):
            allowed = ("enabled", "profiling_data_limit_gb", "free_disk_threshold_gb")
            cfg.dashcam = {k: v for k, v in dc.items() if k in allowed}

    @staticmethod
    def _parse_python_profiling_config(cfg, pp) -> None:
        """python_profiling_config 段 → cfg.python_profiling（torch_npu.profiler）。"""
        if not isinstance(pp, dict):
            return
        if "enabled" in pp:
            cfg.python_profiling["enabled"] = bool(pp["enabled"])
        for k in ("record_shapes", "profile_memory",
                  "with_stack", "with_flops", "with_modules"):
            if k in pp:
                cfg.python_profiling[k] = bool(pp[k])

    @_safe
    def to_file(self, path: str) -> None:
        """将配置写入 YAML/JSON 文件（根据后缀自动识别）。失败时仅记日志。"""
        p = Path(path)
        # 确保父目录存在
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix in (".yaml", ".yml"):
            self._write_annotated_yaml(path)
        elif p.suffix == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的格式: {p.suffix}")
        logger.info(f"配置已写入: {path}")

    # ------------------------------------------------------------------
    # 带注释 YAML 生成
    # ------------------------------------------------------------------

    _YAML_TEMPLATE = """\
# ============================================================================
# ALOP Profiling 配置文件 (CANN 8.5.0) — 独立解耦版
# 修改后无需重启 profiler，每次 start() 会自动重新读取
# ============================================================================
#
# 配置分两层，对应不同的 CANN API 调用:
#   profiling_config → aclprofCreateConfig  (采集什么数据)
#   sampling_config  → aclprofSetConfig     (硬件/系统采样参数)

# ----------------------------- 输出目录 -------------------------------------
# profiling 数据落盘根路径。默认 $FINAL_DUMP_DEST_DIR/alop/data（环境变量
# FINAL_DUMP_DEST_DIR 未设置时回退到 <当前目录>/alop_output）。
# 每次 start() 会在其下自动创建子目录
#   $root/alop/data/<namespace>/<ip>/<name>_<dev>_<pid>/alop_<时间戳>/
# namespace 由环境变量 ALOP_NAMESPACE 控制（默认 default）；ip/name 见下方。
output_dir: "{output_dir}"

# --------------------------- 机器/服务标识 ----------------------------------
# 用于在 data 树里区分不同机器/服务，便于多机多服务汇总定位。
# ip: 本机 IP 或任意标识字符串，缺省 "unknown"
# name: 本服务/模型名，缺省 "alop"
ip: "{ip}"
name: "{name}"

# ============================================================================
# 采集配置 (aclprofCreateConfig)
# 控制 profiling 采集哪些类别的数据和指标
# ============================================================================
profiling_config:
  # --------------------------- AI Core 指标 --------------------------------
  # 指定 AiCore 采集的算子指标，每次采集只能选一种
  # 可选值:
  #   ArithmeticUtilization   - 算力利用率（计算单元占比）
  #   PipeUtilization         - 流水线利用率（各阶段并行度）
  #   MemoryBandwidth         - 内存带宽利用率
  #   L0BBandwidth            - L0B 带宽利用率
  #   ResourceConflictRatio   - 资源冲突比例（标量/矢量队列冲突）
  #   MemoryUB                - Unity Buffer 利用率
  #   L2Cache                 - L2 Cache 命中率
  #   PipeExecuteUtilization  - 流水执行利用率
  #   MemoryAccess            - 内存访问情况
  aicore_metric: "{aicore_metric}"

  # --------------------------- 采集数据类型 --------------------------------
  # 控制 profiling 采集哪些类别的数据，true=开启，false=关闭
  # 建议根据场景按需开启，开启越多开销越大
  data_types:
    acl_api:          {acl_api}          # ACL API 层调用追踪
    task_time:        {task_time}        # 算子任务耗时（Task Time）
    aicore_metrics:   {aicore_metrics}   # AiCore 算子指标（配合 aicore_metric 使用）
    aicpu:            {aicpu}            # AICPU 算子执行信息
    l2cache:          {l2cache}          # L2 Cache 读写数据
    hccl_trace:       {hccl_trace}       # HCCL 集合通信 trace
    training_trace:   {training_trace}   # 训练迭代 trace（训练场景开启，推理场景关闭）
    msproftx:         {msproftx}         # MsprofTX 自定义打点
    runtime_api:      {runtime_api}      # Runtime API 层调用追踪
    task_time_l0:     {task_time_l0}     # 算子耗时 L0 层（更细粒度，开销较大）
    task_memory:      {task_memory}      # 算子内存分配信息
    task_time_l2:     {task_time_l2}     # 算子耗时 L2 层
    op_attr:          {op_attr}          # 算子属性信息
    task_time_l3:     {task_time_l3}     # 算子耗时 L3 层（最细粒度，开销最大）

# ============================================================================
# 采样配置 (aclprofSetConfig)
# 控制硬件/系统级数据的采集频率和模式
# 频率值均为字符串，单位为 ms；"0" 表示不采集
# ============================================================================
{sampling_block}

# ----------------------------- 其他参数 ------------------------------------
# ACL API 调用之间的等待时间（秒），用于确保各步骤初始化完成
# 通常无需修改，默认 0.1s
sleep_interval: {sleep_interval}

# ============================================================================
# Python 级采集配置 (torch_npu.profiler)
# 启用后，torch_npu.profiler 将作为唯一采集引擎（替代 aclprof C API），
# 同时完成 C 级（NPU 硬件指标）和 Python 级（调用栈/Module/形状等）采集，
# 避免双引擎并行带来的额外开销。
# 上方的 profiling_config 中的 aicore_metric 和 data_types 会自动映射到
# torch_npu.profiler 的 _ExperimentalConfig，C 级设置仍然生效。
# 需安装 torch_npu，未安装时自动回退到纯 C 级 aclprof 采集。
# 开启越多选项开销越大，建议按需开启
# ============================================================================
python_profiling_config:
  enabled: {pp_enabled}                     # 启用后替代 aclprof，作为唯一采集引擎
  record_shapes:  {pp_record_shapes}   # 记录算子输入 Tensor 形状（有助于分析算子瓶颈）
  profile_memory: {pp_profile_memory}   # 记录内存分配/释放事件（有额外开销）
  with_stack:     {pp_with_stack}        # 记录 Python/C++ 调用栈（有助于定位代码位置，开销较大）
  with_flops:     {pp_with_flops}        # 估算算子 FLOPs（仅对部分算子有效）
  with_modules:   {pp_with_modules}      # 记录 nn.Module 层次结构（有助于模型结构分析）

# ============================================================================
# 滑窗存储配置（双约束淘汰，独立版仅供参考）
# 以极低系统开销防止 profiling 数据导致磁盘溢出，双约束任一被违反即淘汰最旧数据包：
#   1) profiling_data 上限（profiling_data_limit_gb）：data 目录下采集数据总占用
#      超过此值 → 淘汰最旧包；0 = 禁用此约束。
#   2) 剩余容量阈值（free_disk_threshold_gb）：分区可用空间低于此值 → 淘汰最旧包；
#      0 = 禁用此约束。
# 注意：独立解耦版暂未实现自动淘汰逻辑，此段仅供查阅/后续扩展。
# ============================================================================
dashcam_config:
  enabled: {dc_enabled}                            # 滑窗淘汰开关（独立版暂未生效）
  profiling_data_limit_gb: {dc_profiling_data_limit_gb}  # profiling_data 总占用上限 GB，超过则淘汰最旧包，0=禁用
  free_disk_threshold_gb: {dc_free_disk_threshold_gb}    # 剩余可用空间阈值 GB，低于则触发淘汰，0=禁用
"""

    def _write_annotated_yaml(self, path: str) -> None:
        """生成带注释的 YAML 配置文件，便于用户阅读和修改。"""
        dt = self.data_types
        sa = self.sampling

        # sampling_config 为空时输出 null（不调用 setConfig）
        if sa:
            sampling_block = (
                "sampling_config:\n"
                f'  hardware_mem_freq:    "{sa.get("hardware_mem_freq", "10")}"'
                "    # Device 内存采集频率（ms），建议 \"10\"\n"
                f'  llc_mode:             "{sa.get("llc_mode", "read")}"'
                "             # LLC 采集模式: \"read\" / \"write\" / \"readAndWrite\"\n"
                f'  io_freq:              "{sa.get("io_freq", "10")}"'
                "              # IO 利用率采集频率（ms）\n"
                f'  interconnection_freq: "{sa.get("interconnection_freq", "10")}"'
                " # 芯片互联采集频率（ms）\n"
                f'  host_sys:             "{sa.get("host_sys", "cpu,mem")}"'
                "             # Host 采集项: \"cpu,mem\" / \"cpu\" / \"mem\"\n"
                f'  host_sys_usage:       "{sa.get("host_sys_usage", "cpu,mem")}"'
                "       # Host 利用率采集项: \"cpu,mem\"\n"
                f'  host_sys_usage_freq:  "{sa.get("host_sys_usage_freq", "10")}"'
                "  # Host 利用率采集频率（ms）"
            )
        else:
            sampling_block = (
                "# 设为 null 可跳过 aclprofSetConfig，仅使用 profiling_config 采集\n"
                "sampling_config: null"
            )

        content = self._YAML_TEMPLATE.format(
            output_dir=self.output_dir.replace("\\", "/"),
            ip=self.ip,
            name=self.name,
            aicore_metric=self.aicore_metric,
            acl_api=_tf(dt.get("acl_api")),
            task_time=_tf(dt.get("task_time")),
            aicore_metrics=_tf(dt.get("aicore_metrics")),
            aicpu=_tf(dt.get("aicpu")),
            l2cache=_tf(dt.get("l2cache")),
            hccl_trace=_tf(dt.get("hccl_trace")),
            training_trace=_tf(dt.get("training_trace")),
            msproftx=_tf(dt.get("msproftx")),
            runtime_api=_tf(dt.get("runtime_api")),
            task_time_l0=_tf(dt.get("task_time_l0")),
            task_memory=_tf(dt.get("task_memory")),
            task_time_l2=_tf(dt.get("task_time_l2")),
            op_attr=_tf(dt.get("op_attr")),
            task_time_l3=_tf(dt.get("task_time_l3")),
            sampling_block=sampling_block,
            sleep_interval=self.sleep_interval,
            dc_enabled=_tf(self.dashcam.get("enabled")),
            dc_profiling_data_limit_gb=self.dashcam.get("profiling_data_limit_gb", 500),
            dc_free_disk_threshold_gb=self.dashcam.get("free_disk_threshold_gb", 200),
            pp_enabled=_tf(self.python_profiling.get("enabled")),
            pp_record_shapes=_tf(self.python_profiling.get("record_shapes")),
            pp_profile_memory=_tf(self.python_profiling.get("profile_memory")),
            pp_with_stack=_tf(self.python_profiling.get("with_stack")),
            pp_with_flops=_tf(self.python_profiling.get("with_flops")),
            pp_with_modules=_tf(self.python_profiling.get("with_modules")),
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def resolve_aicore_metric(self) -> int:
        """解析 aicore_metric 为整数值。"""
        if isinstance(self.aicore_metric, int):
            return self.aicore_metric
        return _AICORE_METRIC_NAME_MAP.get(
            self.aicore_metric, AiCoreMetric.PIPE_UTILIZATION
        )

    def build_data_type_mask(self) -> int:
        """将启用的 data_types 组合为位掩码。"""
        mask = 0
        for name, enabled in self.data_types.items():
            if enabled and name in _DATA_TYPE_NAME_MAP:
                mask |= _DATA_TYPE_NAME_MAP[name]
        return mask

    def summary(self) -> str:
        metric_name = self.aicore_metric
        if isinstance(self.aicore_metric, int):
            metric_name = AiCoreMetric(self.aicore_metric).display_name()
        enabled_dt = [k for k, v in self.data_types.items() if v]
        pp = self.python_profiling
        pp_status = "enabled" if pp.get("enabled") else "disabled"
        pp_opts = [k for k, v in pp.items() if k != "enabled" and v]
        return (
            f"ALOP Config Summary:\n"
            f"  output_dir      : {self.output_dir}\n"
            f"  ip/name         : {self.ip} / {self.name}\n"
            f"  --- profiling_config (aclprofCreateConfig) ---\n"
            f"  aicore_metric   : {metric_name}\n"
            f"  data_type_mask  : 0x{self.build_data_type_mask():X}\n"
            f"  enabled_modules : {enabled_dt}\n"
            f"  --- sampling_config (aclprofSetConfig) ---\n"
            f"  sampling        : {self.sampling}\n"
            f"  --- python_profiling_config (torch_npu.profiler 唯一引擎) ---\n"
            f"  python_profiling: {pp_status}\n"
            f"  active options  : {pp_opts}\n"
            f"  --- 容量双约束淘汰配置 ---\n"
            f"  dashcam         : {self.dashcam}\n"
            f"  sleep_interval  : {self.sleep_interval}s"
        )


# ============================================================================
# torch_npu.profiler 独立采集后端
# ============================================================================

class TorchNpuProfilerBackend:
    """torch_npu.profiler 独立采集后端（C级+Python级一体）。

    当 python_profiling.enabled=True 时，ALOPProfiler 自动委托给此后端，
    由 torch_npu.profiler 同时完成 C 级（NPU 硬件指标）和 Python 级
    （调用栈、Module 层次、Tensor 形状、内存事件、FLOPs 估算）采集，
    避免双引擎并行开销。

    故障隔离：所有公开方法 @_safe 装饰，失败仅记日志，不向调用方传播。
    """

    def __init__(self, config: Optional["ALOPConfig"] = None,
                 output_dir_override: Optional[str] = None):
        self._config = config or ALOPConfig()
        self._python_prof = None       # torch_npu.profiler.profile 实例
        self._started = False
        self._output_dir = self._config.output_dir
        # ★ 外部注入的输出目录（ALOPProfiler.set_output_dir_override 透传过来）：
        #   start() 里 _output_dir_override is None 时会调 build_output_dir() 覆写路径，
        #   导致 --out 指定的目录被丢弃。override 才能阻止覆写。
        self._output_dir_override: Optional[str] = output_dir_override
        # 不在构造时创建数据目录，避免拉起阶段生成误导性扁平目录；
        # 真正的数据目录由 start() 在每次采集时用 build_output_dir 创建。
        _setup_file_logger(self._output_dir)

    @property
    def config(self) -> "ALOPConfig":
        return self._config

    @property
    def is_started(self) -> bool:
        return self._started

    # ------------------------------------------------------------------
    # 公共接口：输出目录 / 配置回读控制
    # ------------------------------------------------------------------

    def set_output_dir(self, output_dir: str) -> None:
        """注入本次采集的输出目录。"""
        self._config.output_dir = output_dir
        self._output_dir = output_dir

    @property
    def output_dir(self) -> str:
        """本次采集真正落盘的目录（ALOPProfiler.start() 读完它回写 `_output_dir`，
        否则报 `'TorchNpuProfilerBackend' object has no attribute 'output_dir'` → 采集没起来、
        step() 全被忽略、trace 为空）。与 `set_output_dir` 对称。"""
        return self._output_dir

    def set_output_dir_override(self, output_dir: str) -> None:
        """接口对齐：torch_npu 后端无 override 概念，直接落到 _output_dir。"""
        self._output_dir = output_dir
        self._config.output_dir = output_dir

    def disable_config_reload(self) -> None:
        """接口对齐：torch_npu 后端无 yaml 回读路径，无操作。"""
        return None

    @_safe
    def start(self, device: int = 0) -> None:
        """启动 torch_npu.profiler 采集（C级+Python级一体）。

        将 ALOP 的 profiling_config（aicore_metric, data_types）映射到
        torch_npu.profiler 的 _ExperimentalConfig，使 C 级采集设置仍生效，
        同时叠加 Python 级能力（record_shapes, with_stack 等）。

        Args:
            device: 采集的目标设备 ID（同时用于构建输出目录的 dev_id 段）。
        """
        if self._started:
            logger.warning("TorchNpuProfilerBackend 已在运行，跳过 start")
            return

        if not _TorchNpuProfiler.available():
            logger.warning(
                "[Step] torch_npu.profiler 不可用（torch_npu 未安装），"
                "无法启动采集"
            )
            return

        try:
            # 构建多级输出目录并创建：--out-dir 覆盖时直接用覆盖值，否则派生
            if self._output_dir_override is None:
                self._output_dir = build_output_dir(
                    ip=self._config.ip,
                    name=self._config.name,
                    dev_id=device,
                )
            self._config.output_dir = self._output_dir
            os.makedirs(self._output_dir, exist_ok=True)
            _setup_file_logger(self._output_dir)

            import torch_npu
            pp = self._config.python_profiling

            # 通过 _TorchNpuProfiler 映射 ALOP 配置 → _ExperimentalConfig
            experimental_config = _TorchNpuProfiler.build_experimental_config(
                aicore_metric=self._config.aicore_metric,
                data_types=self._config.data_types,
            )
            if experimental_config is None:
                # 映射失败时使用默认配置
                experimental_config = torch_npu.profiler._ExperimentalConfig(
                    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
                    l2_cache=False,
                    data_simplification=False,
                )

            activities = [
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ]

            self._python_prof = _TorchNpuProfiler.create_profile(
                activities=activities,
                record_shapes=bool(pp.get("record_shapes", False)),
                profile_memory=bool(pp.get("profile_memory", False)),
                with_stack=bool(pp.get("with_stack", False)),
                with_flops=bool(pp.get("with_flops", False)),
                with_modules=bool(pp.get("with_modules", False)),
                experimental_config=experimental_config,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    self._output_dir
                ),
            )

            if self._python_prof is None:
                logger.error(
                    "[Step] torch_npu.profiler.profile() 创建失败，采集未启动"
                )
                return

            self._python_prof.start()
            self._started = True
            logger.info(
                f"[Step] TorchNpuProfilerBackend 启动成功（C级+Python级一体）— "
                f"output_dir={self._output_dir}, "
                f"aicore_metric={self._config.aicore_metric}, "
                f"record_shapes={pp.get('record_shapes')}, "
                f"profile_memory={pp.get('profile_memory')}, "
                f"with_stack={pp.get('with_stack')}, "
                f"with_flops={pp.get('with_flops')}, "
                f"with_modules={pp.get('with_modules')}"
            )
        except Exception as e:
            logger.error(f"[Step] TorchNpuProfilerBackend 启动失败: {e}")
            self._python_prof = None

    @_safe
    def stop(self) -> None:
        """停止 torch_npu.profiler 采集（唯一引擎模式）。

        故障隔离：任何异常仅记录日志，不影响主流程。
        """
        if self._python_prof is None:
            logger.debug("[Step] TorchNpuProfilerBackend 未运行，跳过 stop")
            return

        logger.info("[Step] TorchNpuProfilerBackend 停止开始")
        try:
            self._python_prof.stop()
            self._started = False
            logger.info(f"[Step] TorchNpuProfilerBackend 停止成功 — 数据落盘至: {self._output_dir}")
        except Exception as e:
            logger.error(f"[Step] TorchNpuProfilerBackend.stop() 失败: {e}")
            self._started = False
        finally:
            self._python_prof = None

    def step(self) -> None:
        """透传 step() 到 torch_npu.profiler（如果有原生 step 方法）。

        torch_npu.profiler.profile 对象有 step() 方法时直接调用，
        使 torch_npu 后端也能产生原生的 Step Id 列。
        若对象没有 step() 方法（低版本 torch_npu），静默跳过。
        """
        if self._python_prof is not None and hasattr(self._python_prof, "step"):
            try:
                self._python_prof.step()
            except Exception as e:
                logger.debug(f"[Step] torch_npu.profiler.step() 异常（已忽略）: {e}")

    @_safe
    def finalize(self) -> None:
        """释放采集资源（torch_npu.profiler 在 stop() 时已释放）。"""
        if self._started:
            logger.warning("TorchNpuProfilerBackend 仍在运行，先执行 stop")
            self.stop()
        logger.info("TorchNpuProfilerBackend finalize（资源已在 stop() 中释放）")

    def reload_config(self) -> "TorchNpuProfilerBackend":
        """torch_npu.profiler 不支持配置重载，返回 self 保持接口一致。"""
        return self


# ============================================================================
# ctypes 结构体与 ACL Profiler 库加载
# ============================================================================

class _AclProfConfig(ctypes.Structure):
    """对应 C 层 aclprofConfig 不透明结构体。"""
    pass


class _ACLProfilerLib:
    """单例：加载 ascendcl 共享库，提供函数绑定。"""

    _instance: Optional["_ACLProfilerLib"] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "_ACLProfilerLib":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._lib: Optional[ctypes.CDLL] = None
        self._load()

    def _load(self) -> None:
        """按优先级尝试加载 ascendcl 库。"""
        search_paths = [
            "ascendcl",
            "libascendcl.so",
            "/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so",
        ]
        # 从环境变量补充路径
        cann_home = os.environ.get("ASCEND_TOOLKIT_HOME", "")
        if cann_home:
            search_paths.append(os.path.join(cann_home, "lib64", "libascendcl.so"))

        for p in search_paths:
            try:
                self._lib = ctypes.CDLL(p)
                logger.info(f"ACL 库加载成功: {p}")
                return
            except OSError as e:
                logger.debug(f"ACL 库加载失败，尝试下一个候选路径: {p}, {e}")
                continue

        raise RuntimeError(
            "无法加载 ACL Profiler 库。请确保已安装 CANN 工具包(>=8.5.0)，"
            "并设置 ASCEND_TOOLKIT_HOME 环境变量。"
        )

    def func(self, name: str, argtypes=None, restype=None):
        """获取并绑定库函数签名。"""
        if self._lib is None:
            raise RuntimeError("ACL Profiler 库未加载")
        try:
            f = getattr(self._lib, name)
        except AttributeError:
            raise AttributeError(f"库中未找到函数: {name}") from None
        if argtypes:
            f.argtypes = argtypes
        if restype is not None:
            f.restype = restype
        return f


class _ACLProfilerAPI:
    """封装 aclprof* C API 的类型签名。"""

    def __init__(self):
        lib = _ACLProfilerLib.get()
        self.init = lib.func(
            "aclprofInit",
            argtypes=[ctypes.c_char_p, ctypes.c_size_t],
            restype=ctypes.c_int,
        )
        self.start = lib.func(
            "aclprofStart",
            argtypes=[ctypes.POINTER(_AclProfConfig)],
            restype=ctypes.c_int,
        )
        self.stop = lib.func(
            "aclprofStop",
            argtypes=[ctypes.POINTER(_AclProfConfig)],
            restype=ctypes.c_int,
        )
        self.finalize = lib.func(
            "aclprofFinalize",
            argtypes=[],
            restype=ctypes.c_int,
        )
        self.create_config = lib.func(
            "aclprofCreateConfig",
            argtypes=[
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_uint32,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_uint64,
            ],
            restype=ctypes.POINTER(_AclProfConfig),
        )
        self.destroy_config = lib.func(
            "aclprofDestroyConfig",
            argtypes=[ctypes.POINTER(_AclProfConfig)],
            restype=ctypes.c_int,
        )
        self.set_config = lib.func(
            "aclprofSetConfig",
            argtypes=[ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t],
            restype=ctypes.c_int,
        )


# ============================================================================
# ALOPProfiler 主类
# ============================================================================

class ALOPProfiler:
    """
    ALOP 性能采集工具 — 统一入口，自动选择后端

    - 默认（python_profiling_config.enabled=false）：使用 aclprof C API，
      仅采集 C 级硬件指标，轻量级。
    - python_profiling_config.enabled=true 且 torch_npu 可用：自动委托给
      TorchNpuProfilerBackend，由 torch_npu.profiler 同时完成 C 级 + Python 级采集。
    - enabled=true 但 torch_npu 不可用：自动回退到 aclprof C API 并告警。

    故障隔离：所有公开方法（start/stop/finalize/reload_config）均不会抛异常，
    失败时仅记录日志，被采集程序主流程不受影响。

    核心用法::

        profiler = ALOPProfiler.from_file("alop_config.yaml")
        profiler.start(device=0)
        # ... 计算代码 ...
        profiler.stop()
        profiler.finalize()
    """

    def __init__(self, config: Optional[ALOPConfig] = None,
                 config_path: Optional[str] = None,
                 cli_overrides: Optional[Dict[str, Any]] = None):
        self._config = config or ALOPConfig()
        self._config_path = config_path
        self._api: Optional[_ACLProfilerAPI] = None
        self._prof_config = None
        self._initialized = False
        self._started = False
        self._output_dir_override: Optional[str] = None  # 外部注入的输出目录
        self._cli_overrides = dict(cli_overrides or {})  # CLI 覆盖（--out-dir 等）
        self._cli_device: Optional[int] = None           # CLI --device
        self._output_dir = self._resolve_output_dir()
        # 委托后端：仅当 python_profiling.enabled 且 torch_npu 可用时激活
        self._python_backend: Optional[TorchNpuProfilerBackend] = None
        self._use_python_backend: bool = False
        # step() 内置计数：forward 循环计数器
        # 每次 start() 清零，step() 递增；不涉及数据后处理
        self._step_counter: int = 0
        # 不在构造时创建数据目录，避免拉起阶段生成误导性扁平目录；
        # 真正的数据目录由 start() 在每次采集时用 build_output_dir 创建。
        _setup_file_logger(self._output_dir)

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    @_safe
    def from_file(cls, config_path: str,
                  cli_overrides: Optional[Dict[str, Any]] = None) -> "ALOPProfiler":
        """从 YAML/JSON 配置文件创建 Profiler。失败时返回 None。

        Args:
            config_path: YAML/JSON 配置文件路径。
            cli_overrides: CLI 覆盖字典（--out-dir / --device / --metric /
                --python-prof），start() 内配置重载后统一应用。
        """
        cfg = ALOPConfig.from_file(config_path)
        if cfg is None:
            raise RuntimeError(f"配置文件加载失败: {config_path}")
        logger.info(f"从配置文件加载: {config_path}")
        logger.info(cfg.summary())
        return cls(cfg, config_path=config_path, cli_overrides=cli_overrides)

    @classmethod
    @_safe
    def from_dict(cls, d: Dict[str, Any]) -> "ALOPProfiler":
        """从字典创建 Profiler。失败时返回 None。"""
        cfg = ALOPConfig.from_dict(d)
        if cfg is None:
            raise RuntimeError("字典配置加载失败")
        return cls(cfg)

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------

    @property
    def config(self) -> ALOPConfig:
        return self._config

    @property
    def config_path(self) -> Optional[str]:
        return self._config_path

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def output_dir(self) -> str:
        return self._output_dir

    # ------------------------------------------------------------------
    # 公共接口：输出目录 / 配置回读控制
    # ------------------------------------------------------------------

    def set_output_dir(self, output_dir: str) -> None:
        """注入本次采集的输出目录。同时写入 _config.output_dir。"""
        self._config.output_dir = output_dir
        self._output_dir = output_dir

    def set_output_dir_override(self, output_dir: str) -> None:
        """通过 override 注入输出目录，阻止 _resolve_output_dir() 用其它来源覆盖。"""
        self._output_dir_override = output_dir
        self._output_dir = output_dir
        self._config.output_dir = output_dir

    def disable_config_reload(self) -> None:
        """清空 config_path，禁止 start() 前的 _reload_config_if_needed() 回读 yaml。"""
        self._config_path = None

    # ------------------------------------------------------------------
    # 配置重载
    # ------------------------------------------------------------------

    @_safe
    def reload_config(self) -> "ALOPProfiler":
        """
        重新从配置文件加载配置（如果存在 config_path）。
        每次 start() 前会自动调用，修改 yaml 后无需手动调用。
        失败时仅记日志，不影响当前配置。
        """
        if self._config_path is None:
            return self
        p = Path(self._config_path)
        if not p.exists():
            logger.warning(f"配置文件不存在: {self._config_path}，跳过 reload")
            return self
        cfg = ALOPConfig.from_file(self._config_path)
        if cfg is None:
            # from_file 失败（@_safe 返回 None），保留当前有效配置
            logger.warning(f"配置文件重新加载失败: {self._config_path}，保持当前配置不变")
            return self
        self._config = cfg
        logger.info(f"重新加载配置: {self._config_path}")
        logger.info(cfg.summary())
        return self

    # ------------------------------------------------------------------
    # 核心启停方法
    # ------------------------------------------------------------------

    @_safe
    def start(self, device: int = 0) -> None:
        """
        启动 profiling 采集。失败时仅记日志，不影响主流程。

        自动选择后端：
        - python_profiling_config.enabled=true 且 torch_npu 可用 → torch_npu.profiler
        - 否则 → aclprof C API

        Args:
            device: 采集的目标设备 ID，必须为非负整数。
        """
        if not isinstance(device, int) or device < 0:
            raise ValueError(f"device 必须为非负整数，收到: {device!r}")
        if self._started:
            logger.warning("Profiler 已在运行，跳过 start")
            return

        # 重置 step 计数器
        self._step_counter = 0

        try:
            self._cleanup_env()
            self._reload_config_if_needed()

            # CLI --out-dir 等覆盖在配置（重新）加载后生效
            self._apply_cli_overrides()

            # 决定是否走 torch_npu 后端
            want_python = bool(self._config.python_profiling.get("enabled", False))
            if want_python and _TorchNpuProfiler.available():
                self._use_python_backend = True
                self._python_backend = TorchNpuProfilerBackend(
                self._config, output_dir_override=self._output_dir_override)
                logger.info("[Step] 采集模式: torch_npu.profiler（C级+Python级一体）")
                self._python_backend.start(device=device)
                self._started = self._python_backend.is_started
                if self._started:
                    self._output_dir = self._python_backend.output_dir
                    logger.info(f"ALOP Profiler 启动成功 (device={device}, backend=torch_npu)")
                return
            elif want_python and not _TorchNpuProfiler.available():
                logger.warning(
                    "[Step] 配置启用 python_profiling 但 torch_npu 不可用，"
                    "自动回退到 aclprof C API 采集"
                )
                self._use_python_backend = False

            # --- aclprof C API 采集 ---
            logger.info("[Step] 采集模式: aclprof C API（仅 C 级）")

            # 构建多级输出目录：--out-dir 覆盖时直接用覆盖值，否则派生
            if self._output_dir_override is None:
                self._output_dir = build_output_dir(
                    ip=self._config.ip,
                    name=self._config.name,
                    dev_id=device,
                )
            os.makedirs(self._output_dir, exist_ok=True)
            _setup_file_logger(self._output_dir)

            self._load_api()
            if not self._initialized:
                self._prof_init()
                self._sleep()
            # --- aclprofCreateConfig: 采集什么数据 ---
            self._create_config(device)
            self._sleep()
            # --- aclprofSetConfig: 硬件/系统采样参数 ---
            self._set_sampling()
            # --- aclprofStart: 开始采集 ---
            self._prof_start()
            self._sleep()

            logger.info(f"ALOP Profiler 启动成功 (device={device}, backend=aclprof)")
        except Exception as e:
            logger.error(f"启动失败: {e}")
            self._started = False
            # 不抛异常——故障隔离

    @_safe
    def stop(self) -> None:
        """停止 profiling 采集。失败时仅记日志，不影响主流程。"""
        if not self._started:
            logger.warning("Profiler 未运行，跳过 stop")
            return
        try:
            if self._use_python_backend and self._python_backend is not None:
                self._python_backend.stop()
                self._python_backend = None
                self._use_python_backend = False
                # ★ 修（2026-10-16）：原来漏了这句 → `_started` 一直是 True → 随后的
                #   `finalize()` 见到"仍在运行"又 stop 一次 → 走 aclprof 路径在已释放的
                #   backend 上踩 None → 日志报 `停止失败: 'NoneType' object has no attribute
                #   'stop'`（数据其实已落盘 ✓，但日志脏 + 误判）。与 aclprof 分支同语义。
                self._started = False
                logger.info("ALOP Profiler 停止成功（torch_npu 后端）")
                return
            self._prof_stop()
            self._destroy_config()
            logger.info("ALOP Profiler 停止成功（aclprof 后端，init 资源保留，可再次 start）")
        except Exception as e:
            logger.error(f"停止失败: {e}")

    # ------------------------------------------------------------------
    # step() 内置计数：forward 循环计数，不涉及数据后处理
    # ------------------------------------------------------------------

    def step(self) -> None:
        """递增 step 计数器（替代 torch_npu.profiler 的 profiler.step()）。

        不调用任何 NPU API，零开销。只做 _step_counter += 1。
        torch_npu 后端同时透传原生 step()（产生原生 Step Id 列）。

        用法::

            profiler.start(device=0)
            for layer in range(40):
                run_one_layer(...)
                profiler.step()       # ← 每层一个 step
            profiler.stop()
            profiler.finalize()        # ← 采集数据落盘
        """
        if not self._started:
            logger.warning("step() 在 profiler 未运行时调用，已忽略")
            return
        # torch_npu 后端：透传原生 step()（产生原生 Step Id 列）
        if self._use_python_backend and self._python_backend is not None:
            self._python_backend.step()
        self._step_counter += 1

    @_safe
    def finalize(self) -> None:
        """释放采集资源。失败时仅记日志，不影响主流程。"""
        if self._started:
            logger.warning("Profiler 仍在运行，先执行 stop")
            self.stop()
        if self._use_python_backend and self._python_backend is not None:
            self._python_backend.finalize()
            self._python_backend = None
            self._use_python_backend = False
            return
        if not self._initialized:
            logger.debug("Profiler 未初始化，跳过 finalize")
            return
        try:
            self._prof_finalize()
            logger.info("ALOP Profiler finalize 成功，init 资源已释放")
        except Exception as e:
            logger.error(f"finalize 失败: {e}")

    # ------------------------------------------------------------------
    # 上下文管理器
    # ------------------------------------------------------------------

    def __enter__(self) -> "ALOPProfiler":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """退出上下文时安全清理，绝不向调用方传播异常。"""
        if self._started:
            self.stop()   # @_safe 兜底，不会抛异常
        self.finalize()    # @_safe 兜底，不会抛异常

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _resolve_output_dir(self) -> str:
        # 外部注入的输出目录优先；未注入时返回配置中的 output_dir（仅占位，不创建目录）。
        if self._output_dir_override is not None:
            return self._output_dir_override
        return self._config.output_dir

    def _reload_config_if_needed(self) -> None:
        if self._config_path is not None:
            logger.info(f"[Step] 重新加载配置: {self._config_path}")
            self.reload_config()

    def _sleep(self) -> None:
        time.sleep(self._config.sleep_interval)

    def _apply_cli_overrides(self) -> None:
        """应用 CLI 传入的覆盖（--out-dir / --device / --metric / --python-prof）。

        在 start() 完成配置重载（_reload_config_if_needed）之后调用，保证
        覆盖不会被 yaml 回读冲掉。使用 profiler.config 原样设置，两个后端
        （aclprof / torch_npu）共用这一份配置。
        """
        ov = getattr(self, "_cli_overrides", None)
        if not ov:
            return
        if "out_dir" in ov and ov["out_dir"]:
            # set_output_dir_override 同时写入 _config.output_dir 与
            # _output_dir_override，优先级高于 build_output_dir 的派生路径
            self.set_output_dir_override(str(ov["out_dir"]))
        if "device" in ov and isinstance(ov["device"], int) and ov["device"] >= 0:
            self._cli_device = ov["device"]
        if "metric" in ov and ov["metric"]:
            self.config.aicore_metric = ov["metric"]
        if "python_prof" in ov and ov["python_prof"]:
            self.config.python_profiling["enabled"] = True
            self.config.python_profiling.setdefault("record_shapes", False)

    def _cleanup_env(self) -> None:
        """清理可能冲突的 profiling 环境变量。"""
        cleaned = []
        for key in ("PROFILING_MODE", "PROFILING_OPTIONS"):
            if key in os.environ:
                cleaned.append(f"{key}={os.environ[key]}")
                del os.environ[key]
        if cleaned:
            logger.info(f"[Step] 清理环境变量: {', '.join(cleaned)}")
        else:
            logger.debug("[Step] 无需清理环境变量")

    def _load_api(self) -> None:
        if self._api is not None:
            logger.debug("[Step] ACL API 已加载，跳过")
            return
        logger.info("[Step] 加载 ACL Profiler API")
        self._api = _ACLProfilerAPI()
        logger.info("[Step] ACL Profiler API 加载成功")

    def _prof_init(self) -> None:
        logger.info("[Step] aclprofInit 开始")
        path_bytes = self._output_dir.encode("utf-8")
        ret = self._api.init(path_bytes, len(path_bytes))
        if ret != 0:
            logger.error(f"[Step] aclprofInit 失败, ret={ret}")
            raise RuntimeError(f"aclprofInit 失败, ret={ret}")
        self._initialized = True
        logger.info(f"[Step] aclprofInit 成功, 输出目录: {self._output_dir}")

    def _create_config(self, device: int) -> None:
        logger.info(f"[Step] aclprofCreateConfig 开始 (profiling_config), device={device}")
        metric_val = self._config.resolve_aicore_metric()
        dt_mask = self._config.build_data_type_mask()
        device_array = (ctypes.c_uint32 * 1)(device)
        self._prof_config = self._api.create_config(
            device_array, 1, metric_val, None, dt_mask
        )
        m = AiCoreMetric(metric_val)
        logger.info(
            f"[Step] aclprofCreateConfig 成功: "
            f"device={device}, metric={m.display_name()}, "
            f"dataType=0x{dt_mask:X}"
        )

    def _set_sampling(self) -> None:
        sampling = self._config.sampling
        if not sampling:
            logger.info("[Step] aclprofSetConfig 跳过 (sampling_config 为空，不调用 setConfig)")
            return
        logger.info(f"[Step] aclprofSetConfig 开始 (sampling_config, {len(sampling)} 项)")
        for key, config_type in _SAMPLING_CONFIG_MAP.items():
            if key not in sampling or sampling[key] is None:
                continue
            value_bytes = str(sampling[key]).encode("utf-8")
            ret = self._api.set_config(config_type, value_bytes, len(value_bytes))
            status = "OK" if ret == 0 else f"FAIL(ret={ret})"
            logger.debug(f"  [{status}] setConfig {key} = {sampling[key]}")
        logger.info("[Step] aclprofSetConfig 完成")

    def _prof_start(self) -> None:
        logger.info("[Step] aclprofStart 开始 (profiling 采集启动)")
        ret = self._api.start(self._prof_config)
        if ret != 0:
            logger.error(f"[Step] aclprofStart 失败, ret={ret}")
            raise RuntimeError(f"aclprofStart 失败, ret={ret}")
        self._started = True
        logger.info("[Step] aclprofStart 成功 — profiling 采集中")

    def _prof_stop(self) -> None:
        logger.info("[Step] aclprofStop 开始 (profiling 采集停止)")
        ret = self._api.stop(self._prof_config)
        if ret != 0:
            logger.warning(f"[Step] aclprofStop 返回非零, ret={ret}")
        self._started = False
        logger.info("[Step] aclprofStop 成功 — profiling 采集已停止")

    def _prof_finalize(self) -> None:
        logger.info("[Step] aclprofFinalize 开始 (释放 init 资源)")
        ret = self._api.finalize()
        if ret != 0:
            logger.warning(f"[Step] aclprofFinalize 返回非零, ret={ret}")
        self._initialized = False
        logger.info(f"[Step] aclprofFinalize 成功 — 数据落盘至: {self._output_dir}")

    def _destroy_config(self) -> None:
        if self._prof_config is None:
            return
        logger.debug("[Step] aclprofDestroyConfig 开始")
        ret = self._api.destroy_config(self._prof_config)
        if ret != 0:
            logger.warning(f"[Step] aclprofDestroyConfig 返回非零, ret={ret}")
        self._prof_config = None
        logger.debug("[Step] aclprofDestroyConfig 完成")


# ============================================================================
# 装饰器
# ============================================================================

def alop_profile(config_path: str, device: int = 0):
    """
    装饰器：对被装饰函数自动执行 profiling 采集。
    采集失败时仅记日志，被装饰函数仍正常执行。

    用法::

        @alop_profile("alop_config.yaml", device=0)
        def inference():
            # ... 推理代码 ...  # 即使 profiling 失败，inference 也会正常执行

    Args:
        config_path: YAML/JSON 配置文件路径。
        device: 采集的目标设备 ID。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = ALOPProfiler.from_file(config_path)
            if profiler is not None:
                profiler.start(device=device)
            try:
                return func(*args, **kwargs)
            finally:
                if profiler is not None:
                    profiler.stop()
                    profiler.finalize()
        return wrapper
    return decorator


# ============================================================================
# 便捷函数
# ============================================================================

def generate_default_config(path: str) -> None:
    """生成默认配置文件到指定路径（根据后缀自动识别格式）。"""
    ALOPConfig().to_file(path)


@_safe
def init_alop(config_path: str) -> ALOPProfiler:
    """从配置文件初始化 Profiler（便捷函数）。失败时返回 None。"""
    return ALOPProfiler.from_file(config_path)


# ============================================================================
# 命令行入口 (CLI)
# ============================================================================
#
# 子命令:
#   init [config]          生成默认配置文件（缺省 alop_config.yaml），已存在则跳过
#   check [config]         检查配置/环境/CANN 可用性（无副作用）
#   run [options] <cmd>    在 ALOP 进程内采集执行目标 Python 程序（-- 后接目标命令）
#
# 说明:
#   aclprof 采集是进程内的（alop.py 内通过 ctypes 加载 libascendcl 并调用
#   aclprofInit/Start/Stop），因此目标程序必须**在 ALOP 进程内执行**才能被采到。
#   run 子命令用 runpy 在当前进程加载执行目标脚本/模块/代码片段，
#   start 前发起、stop 后透传目标退出码。跨进程（如子进程里跑目标程序）
#   是采不到的，这与 msprof 不能直接采集 torchrun 子进程是同一个道理。
#
# 常见用法:
#   ./alop.sh run -- python example.py          # 采集运行 example.py
#   ./alop.sh run --device 1 -- python app.py a b   # 采集 1 号卡，透传参数
#   ./alop.sh run -m my_module -- --flag 1      # 模块方式
#   ./alop.sh run -c "run_inference()" -i infer.py  # 代码片段 + import 前置

_CMD_HELP = """\
alop - ALOP 轻量级 NPU 性能采集工具

子命令:
  init   生成默认配置文件（已存在则跳过，不覆盖已有配置）
  check  检查配置解析 / 采集环境 / CANN 可用性（无副作用，不启动采集）
  run    采集运行目标 Python 程序（必须在 ALOP 进程内执行才能被采集到）

不带子命令时按 run 处理：第一个非选项参数视为目标脚本/模块（直接定位）。

run 的目标程序通过 -- 分隔，支持三种形式：
  run -- python script.py [args...]       # 脚本文件（脚本所在目录加入 sys.path）
  run -- python -m package.module [args]  # 模块方式
  run -c "code" [-i mod.py [mod2.py]]     # 代码片段，-i 先 import 若干模块

运行示例:
  alop --out-dir=xxx ops.py                 # 直接调用：采集运行 ops.py
  alop ops.py a b                           # 等价简写（不带子命令）
  alop --device 1 --metric MemoryBandwidth ops.py   # 选项 + 脚本
  ./alop.sh init [my_config.yaml]
  ./alop.sh check [my_config.yaml]
  ./alop.sh run -- python your_script.py arg1 arg2

退出码:
  init/check: 0=成功, 1=失败
  run: 透传目标程序的退出码（采集失败不影响目标程序执行，此时退出码为目标程序的）
"""


def _default_config_path() -> str:
    """CLI 缺省配置文件路径：cwd 存在 alop_config.yaml 用 cwd 的，否则回退到
    alop.py 同目录（用户 cd 到别处直接 `alop ops.py` 时也能找到默认配置）。"""
    cwd_cfg = os.path.join(os.getcwd(), "alop_config.yaml")
    if os.path.exists(cwd_cfg):
        return cwd_cfg
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "alop_config.yaml")


def _normalize_command(argv: Optional[List[str]]) -> List[str]:
    """把“无子命令的裸调用”归一化为 run 形式，支持直接定位目标脚本。

    `alop --out-dir=xxx ops.py` / `alop ops.py a b` 这类调用没有子命令，
    第一个非选项参数就是目标脚本/模块。归一化规则：
      - 首参命中 {init,check,run} 且其后跟的是子命令参数 → 视为子命令；
      - 否则把首参当作目标脚本，其余原样透传（目标脚本作为 REMAINDER 首项）。
    处理 --out-dir=/--device= 等带值选项时只剥离选项本身，保留其值。
    """
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    if not argv:
        return argv
    head = argv[0]
    if head in ("init", "check", "run"):
        return argv
    if head.startswith("-") and head != "-":
        # 前导选项：可能是带值的 run 选项，逐项剥离直到遇到子命令或目标
        out: List[str] = []
        i = 0
        while i < len(argv):
            a = argv[i]
            if a in ("init", "check", "run"):
                out.extend(argv[i:])
                return out
            if a.startswith("-") and a != "-":
                out.append(a)
                if "=" not in a and a not in ("--python-prof", "--dry-run", "-h", "--help"):
                    if i + 1 < len(argv):
                        out.append(argv[i + 1])
                        i += 1
            else:
                # 第一个非选项参数：视为目标，补 "run" 前缀
                return ["run"] + out + argv[i:]
            i += 1
        return out
    # 首个参数就是目标脚本/模块：补 "run" 前缀
    return ["run"] + argv


def _parse_cli(argv: Optional[List[str]] = None) -> Any:
    """构建并解析命令行参数。返回 (args, parser)。"""
    argv = sys.argv[1:] if argv is None else argv
    argv = _normalize_command(argv)
    parser = argparse.ArgumentParser(
        prog="alop",
        description="ALOP 轻量级 NPU 性能采集工具（CLI）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CMD_HELP,
    )
    sub = parser.add_subparsers(dest="command", metavar="{init,check,run}")

    # ---- init ----
    p_init = sub.add_parser("init", help="生成默认配置文件（已存在则跳过）")
    p_init.add_argument("config", nargs="?", default="alop_config.yaml",
                        help="配置文件路径，缺省 alop_config.yaml")

    # ---- check ----
    p_check = sub.add_parser("check", help="检查配置/环境/CANN 可用性（无副作用）")
    p_check.add_argument("config", nargs="?", default="alop_config.yaml",
                         help="配置文件路径，缺省 alop_config.yaml")

    # ---- run ----
    p_run = sub.add_parser("run", help="采集运行目标 Python 程序（进程内）")
    p_run.add_argument("--config", default=env_default("config", _default_config_path()),
                       help="配置文件路径，缺省 alop_config.yaml（或 $ALOP_CONFIG）")
    p_run.add_argument("--device", type=int, default=env_default("device", 0),
                       help="目标 NPU 设备 ID，缺省 0（或 $ALOP_DEVICE）")
    p_run.add_argument("--metric",
                       choices=list(_AICORE_METRIC_NAME_MAP.keys()),
                       default=env_default("metric"),
                       help="覆盖配置中的 aicore_metric（如 PipeUtilization，或 $ALOP_METRIC）")
    p_run.add_argument("--out-dir", default=env_default("out_dir"),
                       help="采集数据落盘目录（替代配置 output_dir，或 $ALOP_OUT_DIR）")
    p_run.add_argument("--python-prof", action="store_true",
                       help="临时启用 python_profiling（torch_npu 采集，需已安装）")
    p_run.add_argument("-c", dest="code", default=None,
                       help="直接执行代码片段（如 \"run_inference()\"）")
    p_run.add_argument("-i", dest="imports", action="append", default=[],
                       metavar="MODULE_OR_FILE",
                       help="执行 -c 前先 import 的模块/脚本文件，可多次指定（仅配合 -c 使用）")
    p_run.add_argument("--dry-run", action="store_true",
                       help="仅打印将执行的命令与配置，不真正采集运行")
    p_run.add_argument("target", nargs=argparse.REMAINDER,
                       help="目标命令：-- python script.py [args...]（脚本/模块/解释器选项）")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        raise SystemExit(1)
    return args


def _check_config(args: Any) -> int:
    """init/check 共用的配置加载与检查。返回退出码。"""
    path = args.config
    p = Path(path)
    if not p.exists():
        logger.error(f"配置文件不存在: {path}")
        print("提示: 先执行 `alop init [config]` 生成默认配置。")
        return 1
    cfg = ALOPConfig.from_file(str(p))
    if cfg is None:
        logger.error(f"配置文件解析失败: {path}")
        return 1
    print(f"配置加载成功: {p.resolve()}")
    print(cfg.summary())
    if args.command == "check":
        # 环境检查（无副作用）：不启动采集
        root = resolve_alop_root()
        print(f"落盘根目录 (FINAL_DUMP_DEST_DIR): {root}")
        print(f"namespace (ALOP_NAMESPACE): {resolve_alop_namespace()}")
        cann_home = os.environ.get("ASCEND_TOOLKIT_HOME", "(未设置)")
        print(f"ASCEND_TOOLKIT_HOME: {cann_home}")
        try:
            _ACLProfilerLib.get()
            print("CANN aclprof 库: 可用 ✓")
        except Exception as e:
            print(f"CANN aclprof 库: 不可用（{e}）")
        print(f"torch_npu.profiler: {'可用' if _TorchNpuProfiler.available() else '不可用'}")
    return 0


def _cmd_run(args: Any) -> int:
    """run 子命令：在 ALOP 进程内采集运行目标程序。返回目标程序退出码。"""
    # ---- 解析目标命令（-- 之后的部分） ----
    target = args.target or []
    if not args.code and not target:
        logger.error(
            "缺少目标程序。用法: alop run -- python your_script.py [args...]，"
            "或用 -c 指定代码片段（--help 查看更多）。"
        )
        return 2
    if args.code and target:
        logger.error("不能同时使用 -c 与 -- 后的目标命令，二选一。")
        return 2
    # argparse REMAINDER 会把分隔符 "--" 本身也收进 target，这里剥掉前导 "--"
    if target and target[0] == "--":
        target = target[1:]

    runpy_code = args.code
    runpy_argv = []
    runpy_inits = list(args.imports)
    use_imports = False
    if target:
        # target 形如 [python, -m, mod, args...] / [python, script.py, args...]，
        # 或裸调用形式 [script.py, args...]（alop ops.py / alop --out-dir=xxx ops.py）
        t = list(target)
        if t and Path(t[0]).name.lower() in ("python", "python3"):
            t = t[1:]
        if not t:
            logger.error("-- 后需跟 python 脚本/模块（例如: -- python script.py）")
            return 2
        if t and t[0] == "-m":
            if len(t) < 2:
                logger.error("-m 后缺少模块名")
                return 2
            runpy_code = "run_module(sys.argv[0], run_name='__main__', alter_sys=True)"
            runpy_argv = t[1:]          # [mod, args...]
            use_imports = True          # 模块名交由 runpy 处理
        else:
            script = t[0]
            # Windows 下用 abspath 展开再校验：Git Bash 的 /tmp 实际是
            # C:\Users\...\AppData\Local\Temp，Path('/tmp/x.py') 会误判不存在
            script_abs = os.path.abspath(script)
            if not Path(script_abs).exists():
                logger.error(f"目标脚本不存在: {script}")
                return 2
            runpy_code = "run_path(sys.argv[0], run_name='__main__')"
            runpy_argv = t              # [script, args...]
    elif args.code:
        runpy_code = args.code
        runpy_argv = []

    if use_imports:
        runpy_inits = []                # 模块方式由 runpy 自己处理模块名

    # ---- dry-run：只打印，不执行 ----
    if args.dry_run:
        print("=== dry-run：以下命令将被执行（未真正采集） ===")
        if runpy_inits:
            print(f"import 前置: {runpy_inits}")
        print(f"执行代码  : {runpy_code or '<目标脚本>'}")
        if runpy_argv:
            print(f"目标参数  : {runpy_argv}")
        print(f"配置文件  : {args.config}")
        print(f"device    : {args.device}")
        print(f"aicore    : {args.metric or '(来自配置)'}")
        print(f"out_dir   : {args.out_dir or '(来自配置)'}")
        print(f"python_prof: {args.python_prof}")
        return 0

    # ---- 配置加载与启动采集 ----
    # 组装 CLI 覆盖：传给 profiler，start() 内配置重载后统一应用
    cli_overrides: Dict[str, Any] = {}
    if args.out_dir:
        cli_overrides["out_dir"] = args.out_dir
    if args.device is not None:
        cli_overrides["device"] = args.device
    if args.metric:
        cli_overrides["metric"] = args.metric
    if args.python_prof:
        cli_overrides["python_prof"] = True
    profiler = ALOPProfiler.from_file(args.config, cli_overrides=cli_overrides)
    if profiler is None:
        logger.error(f"配置加载失败: {args.config}")
        return 2

    # ---- 进程内采集运行目标程序 ----
    try:
        # 目标脚本所在目录加入 sys.path，与 python script.py 行为一致
        if runpy_argv and not runpy_code.startswith("run_module"):
            script_dir = str(Path(os.path.abspath(runpy_argv[0])).resolve().parent)
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
        # 模块方式（-m）：把当前工作目录加入 sys.path，与 python -m 的行为一致，
        # 保证用户在 cwd 下的包/模块可被发现（alop.py 自身所在目录不一定在 sys.path）。
        if runpy_code.startswith("run_module"):
            cwd = os.getcwd()
            if cwd not in sys.path:
                sys.path.insert(0, cwd)
        import runpy
        if runpy_inits:
            for init in runpy_inits:
                ip = Path(init)
                if ip.suffix.lower() in (".py", ".pyc"):
                    _exec_script_file(init, globals())
                else:
                    __import__(init)
        # 实际 device：CLI --device（含环境变量默认）优先，否则配置/0
        dev = getattr(profiler, "_cli_device", None)
        if dev is None:
            dev = args.device if args.device is not None else 0
        profiler.start(device=dev)
        try:
            _run_target(runpy_code, runpy_argv)
        finally:
            profiler.stop()
        profiler.finalize()
        return 0
    except SystemExit as e:
        # 目标程序主动 sys.exit(code)：结束采集并透传退出码
        profiler.stop()
        profiler.finalize()
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        return code
    except BaseException as e:
        logger.error(f"目标程序执行异常: {e}\n{traceback.format_exc()}")
        profiler.stop()
        profiler.finalize()
        return 1


def _run_target(code: str, argv: List[str]) -> None:
    """在当前进程内执行目标代码（脚本 / 模块 / 代码片段）。

    注意 sys.argv 会被置为目标程序的 argv（runpy 的 alter_sys 行为），
    因此目标程序里读 sys.argv 得到的是它自己的参数。
    """
    if code.startswith("run_module"):
        import runpy
        mod = argv[0]
        rest = argv[1:]
        sys.argv = [mod] + rest
        runpy.run_module(mod, run_name="__main__", alter_sys=True)
    elif code == "run_path(sys.argv[0], run_name='__main__')":
        import runpy
        sys.argv = list(argv)
        runpy.run_path(argv[0], run_name="__main__")
    else:
        sys.argv = list(argv)
        exec(code, globals())


def _exec_script_file(path: str, g: Dict[str, Any]) -> None:
    """把 -i 指定的脚本文件作为初始化代码执行（作用域为 ALOP 全局命名空间）。

    用 exec(compile(...)) 直接在当前全局命名空间执行，保证脚本里定义的
    名字（变量/函数）对后续 -c 代码片段可见；runpy.run_path 会新建独立
    命名空间导致名字不可见，故不使用。
    """
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    code = compile(src, str(path), "exec")
    exec(code, g)


def _cmd_main(argv: Optional[List[str]] = None) -> int:
    """CLI 总入口：解析子命令并分发。返回退出码。"""
    try:
        args = _parse_cli(argv)
        if args.command == "init":
            p = Path(args.config)
            if p.exists():
                print(f"配置文件已存在，跳过（不会覆盖）: {p.resolve()}")
                return 0
            ALOPConfig().to_file(args.config)
            print(f"已生成默认配置: {args.config}")
            return 0
        if args.command == "check":
            return _check_config(args)
        if args.command == "run":
            return _cmd_run(args)
        print(f"未知子命令: {args.command}")
        return 2
    except SystemExit:
        raise
    except BaseException as e:
        logger.error(f"CLI 执行失败: {e}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    # 裸跑（无任何参数）时保留独立版原行为：生成默认配置
    if len(sys.argv) == 1:
        generate_default_config("alop_config.yaml")
        print("已生成默认配置: alop_config.yaml")
    else:
        sys.exit(_cmd_main())

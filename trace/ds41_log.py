"""统一日志等级（`DS41M_LOG_LEVEL` / run_hit `--log-level`）。

等级（从低到高）：**pure(0) < error(1) < warn(2) < info(3) < debug(4)**，默认 `info`。

  · `pure`：**什么都不打印**。实现方式是把 `sys.stdout` 换成 /dev/null，所以现有代码里
    几百处 `print(...)` 不用改；进度条（`progress.py` 写 stderr、`progress_bar.py` 写
    专用 fd）不受影响 ⇒ 终端上只剩进度条。
  · `debug`：`ds41_log.debug()` 的输出可见（原来的 print 行为不变）。
  · 直接跑 runner / launch_mini 时用环境变量；走 `tools/run_hit.sh` 用 `--log-level`。

用法：
    import ds41_log
    ds41_log.install()                  # 进程启动最早处调用一次（幂等）
    ds41_log.info("...")                # 新增代码建议用分级接口
"""
from __future__ import annotations

import os
import re
import sys

LEVELS = {"pure": 0, "quiet": 0, "none": 0,
          "error": 1, "warn": 2, "warning": 2, "info": 3, "debug": 4}

# pure 模式仍要放行的行：进度条 + **测量结束后**的一条结果（驱动要靠它填汇总表；
#   它在 `torch.npu.synchronize()` / profiler.stop() 之后打印 → 不在测量/trace 区间内）。
#   `[ 2/40] layer 1 win T=3456` / `[iter 1/4]` / `== 真跑：L=...` / `输入缓冲已建`
_PROGRESS_RE = re.compile(
    r"^\s*(?:\[r\d+\]\s*)?(\[\s*\d+\s*/\s*\d+\]\s+layer\b"
    r"|\[iter\s+\d+\s*/\s*\d+\]"
    r"|==\s*(真跑|--points)"
    r"|────\s*\[point\s+\d+/\d+\]"
    r"|^\s*(?:\[r\d+\]\s*)?\[point\s+\d+/\d+\]"          # 每点开始/完成小结（pure 也要看到）
    r"|多轮耗时：|纯 (?:pass|forward) 吞吐："
    # ★ 2026-10-29：显存审计行（core/memlife.py 在点位边界打印）必须放行，否则
    #   `--pure-log` 下被 stdout 代理整行丢弃 —— 2026-10-29 的 CP1/CP4 两轮就是因此
    #   一条 `[mem] 留存=` 都没有（launch_mini 的白名单也补过，但源头在这里）。
    r"|\[mem\]"
    r"|输入缓冲已建)"
    r"|^\s*(?:\[r\d+\]\s*)?\[点\s*\d+/\d+\]"
    # ★ 错误/告警**永不静默**：pure 只丢"计算无关的正常打印"，诊断信息必须留下
    #   （否则一次跑挂了、算子错又不足 3 处时，rc 仍是 0，问题会被彻底吞掉）
    r"|⚠️|^\s*(?:\[r\d+\]\s*)?层\d+\s+\S+:"
    r"|\[meta-miss\]"
    # ★ 2026-10-22：进度条快照写在 **stdout**（progress_bar.snapshot → print），
    #   pure 过滤时必须放行，否则非 tty（docker exec / `| tee`）下就完全看不到进度。
    #   `@@bar-est` 是驱动 ETA 校准行，也不能丢。
    r"|^\s*(?:\[r\d+\]\s*)?\[bar\s+\d{2}:\d{2}:\d{2}\]"
    r"|@@bar-est\b"
    # ★ 2026-10-27：launcher 的**收尾/失败**行属于点位边界报告（采集已结束），pure 必须放行，
    #   否则跑挂了连哪个 rank、什么退出码都看不到（2026-10-27 06:14 那次就是这么瞎的）。
    r"|^\s*==\s*全部结束"
    r"|非零退出|超时被杀")


class _PureStdout:
    """只放行进度行、其余整行丢弃的 stdout 代理（按行判定，兼容 print 的分次 write）。"""

    encoding = "utf-8"
    errors = "strict"

    def __init__(self, real):
        self._real = real
        self._buf = ""

    def write(self, s):
        if not isinstance(s, str):
            return self._real.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            # search 而非 match：告警/诊断类（⚠️/层…/meta-miss）允许出现在行中任何位置；
            # 进度类自带 `^\s*` 锚点，语义不变。
            if _PROGRESS_RE.search(line):
                self._real.write(line + "\n")
        return len(s)

    def flush(self):
        self._real.flush()

    def isatty(self):
        return False

    def writable(self):
        return True

    def fileno(self):
        return self._real.fileno()

_NAME = "info"
_LEVEL = 3
_INSTALLED = False


def _read_level() -> str:
    raw = (os.environ.get("DS41M_LOG_LEVEL")
           or os.environ.get("DS41M_LOG")
           or "info").strip().lower()
    if os.environ.get("DS41M_PURE_LOG", "").strip().lower() in ("1", "on", "true", "yes"):
        raw = "pure"
    return raw if raw in LEVELS else "info"


def install() -> str:
    """按环境变量安装日志过滤；返回最终等级名。幂等。"""
    global _NAME, _LEVEL, _INSTALLED
    _NAME = _read_level()
    _LEVEL = LEVELS[_NAME]
    if _INSTALLED:
        return _NAME
    _INSTALLED = True
    if _LEVEL <= 0:
        try:
            sys.stdout.flush()
        except Exception:                         # noqa: BLE001
            pass
        # 只放行进度行（progress_bar 需要），其余整行丢弃 ⇒ 终端/日志只剩进度条。
        sys.stdout = _PureStdout(sys.stdout)
    return _NAME


def level_name() -> str:
    return _NAME


def pure() -> bool:
    """pure(0)：只留计算 + 进度条；所有打印/统计/诊断关掉。"""
    return _LEVEL <= 0


def enabled(level: str) -> bool:
    return _LEVEL >= LEVELS.get(level, 3)


def _emit(level: str, *a, **k):
    if _LEVEL >= LEVELS.get(level, 3):
        print(*a, **k)


def info(*a, **k):
    _emit("info", *a, **k)


def warn(*a, **k):
    _emit("warn", *a, **k)


def error(*a, **k):
    _emit("error", *a, **k)


def debug(*a, **k):
    _emit("debug", *a, **k)

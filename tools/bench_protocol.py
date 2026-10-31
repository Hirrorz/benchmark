#!/usr/bin/env python3.11
# coding=utf-8
"""bench_protocol.py — 常驻 benchmark 的最小协议 + 简单进度条（无 TUI）。

设计口径（用户 2026-10-19）：
  · 不做交互式界面，终端里一行文本进度条即可；
  · rank 继续打印**人类可读**的进度行，server/client 用正则识别，不侵入热路径；
  · 需要跨进程传的结构化信息只有命令、结果、错误三种，用一行 JSON（`@@RESULT {...}`）。

本文件只依赖标准库，python3.11 / 系统 python 都能 import。
"""
from __future__ import annotations

import json
import os
import re
import socket
import sys
import time

SENTINEL = "@@RESULT "
EVT_SENTINEL = "@@EVT "
READY_LINE = "[bench] rank %d serve ready"

# ─────────────────────────── 行解析 ───────────────────────────
# warmup / 转换阶段（由 runner/decode_layout 打印）
RE_STAGE = re.compile(
    r"^\s*\[(W1|W2|W3|P→D|D→P|D→P-AC)\s+(\d+)\s*/\s*(\d+)\]\s*(.*)$")
# run_forward 的逐层行："  [ 12/40] layer 11 win T=512 算子 24"
RE_LAYER = re.compile(r"^\s*\[\s*(\d+)\s*/\s*(\d+)\]\s*layer\s+(\d+)\s*(.*)$")
# decode 步："  [inc 3/32] 预测位置 ..."
RE_INC = re.compile(r"^\s*\[inc\s+(\d+)\s*/\s*(\d+)\]\s*(.*)$")
# host cache："[moe-host-cache] 2000/92160  12.3/268.9 GiB  1500 MiB/s"
RE_HC = re.compile(
    r"^\[moe-host-cache\]\s+(\d+)\s*/\s*(\d+)\s+"
    r"([\d.]+)\s*/\s*([\d.]+)\s*GiB\s+(\d+)\s*MiB/s")
# 结构化结果
RE_RESULT = re.compile(r"^\s*@@RESULT\s+(\{.*\})\s*$")
RE_EVT = re.compile(r"^\s*@@EVT\s+(\{.*\})\s*$")


def parse_line(line: str):
    """返回 dict(stage, cur, total, extra) 或 None。"""
    m = RE_HC.match(line)
    if m:
        cur, total = int(m.group(1)), int(m.group(2))
        return {"stage": "hostcache", "cur": cur, "total": total,
                "extra": f"{m.group(3)}/{m.group(4)} GiB  {m.group(5)} MiB/s"}
    m = RE_STAGE.match(line)
    if m:
        return {"stage": m.group(1), "cur": int(m.group(2)),
                "total": int(m.group(3)), "extra": m.group(4).strip()}
    m = RE_INC.match(line)
    if m:
        return {"stage": "decode", "cur": int(m.group(1)),
                "total": int(m.group(2)), "extra": m.group(3).strip()}
    m = RE_LAYER.match(line)
    if m:
        return {"stage": "prefill", "cur": int(m.group(1)),
                "total": int(m.group(2)),
                "extra": f"layer {int(m.group(3))} {m.group(4).strip()}"}
    return None


def parse_result(line: str):
    m = RE_RESULT.match(line)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def parse_evt(line: str):
    m = RE_EVT.match(line)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def emit_result(**kw) -> None:
    print(SENTINEL + json.dumps(kw, ensure_ascii=False, default=str), flush=True)


def emit_evt(**kw) -> None:
    print(EVT_SENTINEL + json.dumps(kw, ensure_ascii=False, default=str), flush=True)


# ─────────────────────────── 简单进度条 ───────────────────────────
def _fmt_sec(s: float) -> str:
    s = max(0.0, float(s))
    if s >= 3600:
        return f"{int(s // 3600)}h{int(s % 3600 // 60):02d}m"
    if s >= 60:
        return f"{int(s // 60)}m{int(s % 60):02d}s"
    return f"{s:.1f}s"


class StageBar:
    """一行进度条：`[prefill] [██████░░░░] 12/40 30% 0.4s ETA 0.9s`。

    - `feed(line)` 识别 runner 现有的人类可读行；
    - `render()` 返回当前字符串；
    - `draw()` 写 stderr（tty 用 `\\r` 原地刷新，非 tty 只在完成/每 `every` 秒打一行）；
    - 阶段切换时自动打一行完成摘要。
    """

    LABELS = {
        "hostcache": "hostcache", "W1": "W1 AC", "W2": "W2 decode",
        "W3": "W3 shapes", "prefill": "prefill", "P→D": "P→D",
        "decode": "decode", "D→P": "D→P", "D→P-AC": "D→P AC",
    }

    def __init__(self, stream=None, enabled: bool = True, width: int = 14,
                 every: float = 0.5, color: bool = True):
        self.stream = stream or sys.stderr
        self.enabled = enabled
        self.width = max(6, int(width))
        self.every = max(0.1, float(every))
        self.color = bool(color) and self._is_tty()
        self.stage = None
        self.cur = 0
        self.total = 0
        self.extra = ""
        self.t0 = time.time()
        self.stage_t0 = self.t0
        self.last_draw = 0.0
        self.last_line = ""
        self.done = False

    def _is_tty(self) -> bool:
        try:
            return bool(self.stream.isatty())
        except Exception:                               # noqa: BLE001
            return False

    def _bar(self, frac: float) -> str:
        frac = min(1.0, max(0.0, frac))
        n = int(round(frac * self.width))
        return "█" * n + "░" * (self.width - n)

    def _c(self, text: str, code: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.color else text

    def render(self) -> str:
        if not self.total:
            return ""
        frac = self.cur / max(1, self.total)
        label = self.LABELS.get(self.stage, self.stage or "")
        elapsed = time.time() - self.stage_t0
        eta = ""
        if 0 < self.cur < self.total:
            eta = f" ETA {_fmt_sec(elapsed / self.cur * (self.total - self.cur))}"
        elif self.cur >= self.total:
            eta = " done"
        extra = f"  {self.extra}" if self.extra else ""
        return (f"[{self._c(label.ljust(10), '36')}] "
                f"[{self._c(self._bar(frac), '32')}] "
                f"{self.cur}/{self.total} {frac * 100:3.0f}% "
                f"{_fmt_sec(elapsed)}{eta}{extra}")

    def feed(self, line: str, now: float | None = None) -> bool:
        info = parse_line(line)
        if not info:
            return False
        stage = info["stage"]
        if self.stage is not None and stage != self.stage:
            self.finish()
        if self.stage != stage:
            self.stage = stage
            self.stage_t0 = now or time.time()
            self.last_draw = 0.0
        self.cur, self.total = int(info["cur"]), int(info["total"])
        self.extra = info.get("extra", "")
        self.last_line = line
        if self.cur >= self.total:
            self.finish(force=True)
        else:
            self.draw(now)
        return True

    def draw(self, now: float | None = None) -> None:
        if not self.enabled or not self.stage:
            return
        now = now or time.time()
        if self._is_tty():
            if now - self.last_draw < self.every:
                return
            self.last_draw = now
            text = self.render()
            try:
                self.stream.write("\r\x1b[2K" + text)
                self.stream.flush()
            except Exception:                           # noqa: BLE001
                pass
        else:
            if now - self.last_draw < max(5.0, self.every):
                return
            self.last_draw = now
            try:
                self.stream.write(self.render() + "\n")
                self.stream.flush()
            except Exception:                           # noqa: BLE001
                pass

    def finish(self, force: bool = False) -> None:
        if not self.stage:
            return
        try:
            if self._is_tty():
                if force:
                    self.stream.write("\r\x1b[2K" + self.render() + "\n")
                else:
                    self.stream.write("\r\x1b[2K")
            elif force:
                self.stream.write(self.render() + "\n")
            self.stream.flush()
        except Exception:                               # noqa: BLE001
            pass
        self.stage = None
        self.cur = self.total = 0
        self.extra = ""

    def close(self) -> None:
        if self.stage:
            self.finish(force=True)


# ─────────────────────────── socket JSONL ───────────────────────────
def send_json(sock: socket.socket, obj: dict) -> None:
    data = (json.dumps(obj, ensure_ascii=False, default=str) + "\n").encode()
    sock.sendall(data)


class JsonLines:
    """把 socket 包成逐行 JSON 迭代器（服务端/客户端同一套）。"""

    def __init__(self, sock: socket.socket, timeout: float | None = None):
        self.sock = sock
        if timeout is not None:
            sock.settimeout(timeout)
        self.f = sock.makefile("r", encoding="utf-8", errors="replace", newline="\n")

    def recv(self):
        line = self.f.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            return {}
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return {"type": "raw", "line": line}

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:                               # noqa: BLE001
            pass


# ─────────────────────────── /tmp 作业目录 ───────────────────────────
def job_root(job: str = "") -> str:
    base = os.environ.get("DS41_BENCH_ROOT", "/tmp/ds41_bench")
    return os.path.join(base, job) if job else base


def default_server_sock(job: str) -> str:
    return os.path.join(job_root(job), "server.sock")


def ensure_job_dir(job: str) -> str:
    d = job_root(job)
    os.makedirs(d, exist_ok=True)
    return d


def wait_socket(path: str, timeout: float = 30.0, interval: float = 0.1) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(path):
            return True
        time.sleep(interval)
    return False

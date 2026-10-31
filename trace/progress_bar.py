"""采集进度条（launch_mini 侧）：解析 rank0 的输出 → 单调进度 + ETA → 终端原地刷新 + 日志快照。

为什么不复用 runner 的 `progress.py`（2026-10-15 实测）：
  · 它只在 S2 层循环里 `tick()`（S0/S1/S3 一个字都不打）；
  · 百分比按阶段表加权，S2 走完只有 80.95%，而且**每个 pass 从 0 重来**（会倒退）；
  · 首 pass 32.9s vs 后续 pass 0.178s（差 180×），首 pass 里前 4 层又占 85%（JIT）——
    所以进度必须"按 pass + 层"自己算，且首 pass 单独加非线性权重。
这里只做显示：**不改 runner、不改任何既有日志行格式**，纯靠锚点行 + 正则解析。

两种输出，物理隔离（不可能把控制字符写进日志）：
  · 终端：`Bar.paint()` 往真终端 fd 写 `\\r\\x1b[K` + 一行（原地刷新）；
  · 日志：`Bar.snapshot()` 纯文本一行，走正常 stdout → 驱动层的 tee。

自测（不碰卡、不需要终端，用真实 rank0.log 当 fixture）：
    python3.11 progress_bar.py --selftest
    python3.11 progress_bar.py --replay <rank0.log> --label "点3/8 L=65536 prof轮" --speed 8
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time

# ── 解析（纯函数）──────────────────────────────────────────────────────────────
RE_LAYER = re.compile(r"^\s*\[\s*(\d+)\s*/\s*(\d+)\]\s+layer\s+(\d+)\s+(\S+)\s+T=(\d+)")
RE_ITER = re.compile(r"^\s*\[iter\s+(\d+)\s*/\s*(\d+)\]\s+([\d.]+)s")
RE_RUN = re.compile(r"^==\s*真跑：L=(\d+)\s+layers=(\S+)\s+iters=(\d+).*?prof=(True|False)")
RE_BUF = re.compile(r"^\s*输入缓冲已建：\((\d+),\s*(\d+)\)")

# 锚点行 → 阶段名（S0/S1/S3 没有 tick，只能靠这些行识别；每条实测恰好命中一次）
ANCHORS = (
    ("== 加载 vendor 算子命名空间", "S0 环境与算子注册"),
    ("  进程组就绪：", "S1 进程组"),
    ("  输入缓冲已建：", "S2 输入准备"),
    ("  口径：", "S2 输入准备"),
    ("  [起跑线]", "S2 采集"),
    ("profiler.py:", "S3 trace 导出"),
    ("  多轮耗时：", "S3 收尾"),
    ("  显存：峰值", "S3 收尾"),
    ("  trace → ", "S3 收尾"),
)

# 时间模型（秒）：全部来自 hit97_fromB/L65536 那次真实采集的实测值，运行时会被逐段替换
EST_DEFAULT = dict(start=19.0,    # launch_mini 起 → 第一层行（导入/算子注册/进程组/输入缓冲）
                   cold=33.0,     # 首 pass（含 JIT）：实测 32.887s
                   steady=0.18,   # 后续 pass：实测 0.177~0.179s
                   trace=20.0,    # prof 轮最后一面里的 trace 导出+解析：实测 12~25s
                   tail=3.0,      # 多轮耗时/显存/阶段实测 收尾
                   cold_n=4,      # 首 pass 里前几层吃掉 85%（JIT）
                   cold_share=0.85)

BAR_W = 12          # 终端进度条格子数
MIN_SNAP_MAX = 5.0  # 快照最小间隔的上限（阶段变化/pass 边界也受它约束，防刷屏）


def fmt_sec(s: float) -> str:
    """90 → 1m30s；3661 → 1h01m。"""
    s = max(0, int(round(s)))
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


class BarState:
    """状态机：吃一行 → 更新内部状态；进度/ETA/渲染都是它的纯函数。"""

    def __init__(self, est: dict | None = None, t0: float | None = None):
        e = dict(EST_DEFAULT)
        e.update(est or {})
        self.est = e
        self.t0 = time.monotonic() if t0 is None else t0
        self.stage = "启动"
        self.iters: int | None = None
        self.prof: bool | None = None
        self.R: int | None = None          # 输入缓冲的 T（= R = M + replay_tail）
        self.n = 0
        self.n_total: int | None = None
        self.pass_cur = 0                  # 当前 pass 序号（1-based；进 S2 后为 1）
        self.pass_done = 0                 # 已完成的 pass 数
        self.idx = self.attn = self.cur_T = None
        self.meas: dict = {}               # 实测替换估计：cold/steady/trace
        self.armed = False                 # 见到 [起跑线] 之后，[iter] 才算 pass 边界
        self.f_prev = 0.0
        self.last_out_t = time.monotonic()  # rank0 最后一次有输出（卡死可见性）
        self.t_last_iter: float | None = None   # 最后一个 pass 结束的时刻（S3 阶段用）
        self.dead: tuple[int, int] | None = None   # (rank, rc) rank0 退出了？

    # ── 喂行 ──
    def feed(self, line: str, rank: int, now: float) -> dict:
        """返回本次变化的键（stage/pass/run/layer），供渲染层决定要不要打快照。"""
        ch: dict = {}
        if rank != 0:
            return ch
        self.last_out_t = now
        m = RE_RUN.match(line)
        if m:
            self.iters, self.prof = int(m.group(3)), (m.group(4) == "True")
            self.pass_cur = max(1, self.pass_cur)
            self.stage = "S2 预备"
            ch["run"] = True
            return ch
        m = RE_BUF.match(line)
        if m:
            self.R = int(m.group(1))
        m = RE_LAYER.match(line)
        if m:
            if self.stage != "S2 前向":
                self.stage = "S2 前向"
                ch["stage"] = True
            self.n_total = int(m.group(2))
            self.n = int(m.group(1))
            self.idx, self.attn, self.cur_T = int(m.group(3)), m.group(4), int(m.group(5))
            self.pass_cur = max(1, self.pass_cur)
            return ch
        m = RE_ITER.match(line)
        if m:
            k, n, sec = int(m.group(1)), int(m.group(2)), float(m.group(3))
            self.iters = n
            self.pass_done = k
            self.meas["cold" if k == 1 else "steady"] = sec
            if self.prof and k == n:                      # prof 轮：最后一面里含 trace 导出
                self.meas["trace"] = max(self.est["trace"], sec - self.est["steady"])
            self.pass_cur = min(k + 1, n)
            self.n = 0
            ch["pass"] = True
            if k >= n:
                self.stage = "S3 收尾"
                self.t_last_iter = now
                ch["stage"] = True
            return ch
        for pref, st in ANCHORS:
            if line.startswith(pref) or pref in line:
                if st != self.stage:
                    self.stage = st
                    ch["stage"] = True
                if pref == "  [起跑线]":
                    self.armed = True
                break
        return ch

    def note_rank_exit(self, rank: int, rc: int, now: float) -> None:
        if rank == 0:
            self.dead = (rank, rc)

    # ── 进度 / ETA ──
    def _weights(self) -> list[float]:
        """[start, pass1..passN, tail]（秒）。实测优先于估计。"""
        e = self.est
        npass = self.iters or 1
        p1 = self.meas.get("cold", e["cold"])
        st = self.meas.get("steady", e["steady"])
        prof = bool(self.prof)
        w = [e["start"]]
        for k in range(1, npass + 1):
            d = p1 if k == 1 else st
            if prof and k == npass:
                d += self.meas.get("trace", e["trace"])
            w.append(d)
        w.append(e["tail"])
        return w

    def frac(self, now: float) -> tuple[float, float]:
        """返回 (0~1 单调进度, 估计总时长秒)。"""
        e, w = self.est, self._weights()
        W = sum(w)
        npass = self.iters or 1
        if self.pass_done >= npass:                      # S3
            tail_el = max(0.0, now - self.t_last_iter) if self.t_last_iter else 0.0
            g = min(1.0, tail_el / max(1e-6, w[-1]))
            raw = (sum(w[:-1]) + w[-1] * g) / W
        elif self.pass_cur <= 1 and self.n == 0:         # 还没进层循环（S0/S1/输入准备）
            el = max(0.0, now - self.t0)
            raw = (w[0] * min(1.0, el / max(1e-6, e["start"]))) / W
        else:                                            # 层循环里
            k = self.pass_cur
            nt = self.n_total or 40
            if k <= 1:                                   # 首 pass：JIT 非线性
                g = e["cold_share"] * min(1.0, self.n / max(1, e["cold_n"])) \
                    + (1 - e["cold_share"]) * min(1.0, self.n / nt)
            else:
                g = min(1.0, self.n / nt)
            raw = (sum(w[:k]) + w[k] * g) / W
        raw = min(1.0, max(0.0, raw))
        self.f_prev = max(self.f_prev, raw)              # ★ 单调不倒退
        return self.f_prev, W

    def eta(self, now: float) -> tuple[float, bool]:
        """(剩余秒, 是否含估计成分)。

        ★ 用**实测速率**外推（`已用/进度 × 剩余进度`），不用"模型总时长 − 已用"：后者在实测比
        模型快时不收敛（实测 98% 时还报"再 2m28s"）。进度太小（<3%）或跑了不到 3s 才退回模型。
        """
        f, W = self.frac(now)
        el = max(0.0, now - self.t0)
        rough = (self.meas.get("cold") is None) or (self.iters is None) or \
                (bool(self.prof) and self.meas.get("trace") is None)
        if f > 0.03 and el > 3.0:
            return max(0.0, (1.0 - f) * el / f), rough
        return max(0.0, W - el), True

    # ── 渲染 ──
    def _glyph(self, f: float) -> str:
        k = int(round(max(0.0, min(1.0, f)) * BAR_W))
        return "█" * k + "░" * (BAR_W - k)

    def _phase_str(self) -> str:
        if self.stage == "S2 前向" and self.n_total:
            return f"S2 前向 pass{self.pass_cur}/{self.iters or '?'} 层{self.n}/{self.n_total}"
        return self.stage

    def snapshot(self, now: float, label: str, note: str = "") -> str:
        """日志用：纯文本一行（无 \\r / 无 ANSI）。"""
        f, _ = self.frac(now)
        eta, rough = self.eta(now)
        el = max(0.0, now - self.t0)
        parts = [f"[bar {time.strftime('%H:%M:%S')}] {label}".rstrip(),
                 self._phase_str()]
        if self.idx is not None and self.stage == "S2 前向":
            parts.append(f"idx={self.idx} {self.attn} T={self.cur_T}")
        parts.append(f"{self._glyph(f)} {f * 100:.0f}%")
        parts.append(f"已用{fmt_sec(el)} 预计再~{fmt_sec(eta)}" + ("（估）" if rough else ""))
        if note:
            parts.append(note)
        if self.dead:
            parts.append(f"[!] rank0 已退出 rc={self.dead[1]}（状态冻结）")
        else:
            gap = now - self.last_out_t
            if gap > 30:
                parts.append(f"⚠ rank0 已 {fmt_sec(gap)} 无输出")
        return " │ ".join(parts)

    def render(self, now: float, label: str, width: int, note: str = "") -> str:
        """终端用：一行，长度严格 ≤ width（超出会让 \\r 失去意义）。"""
        f, _ = self.frac(now)
        eta, rough = self.eta(now)
        el = max(0.0, now - self.t0)
        # 段优先级（窄终端按这个顺序丢尾）：层/pass > 百分比 > 标签(哪个点/哪轮) > 已用/ETA > 明细
        #   —— 标签必须排在"已用/ETA"前面：80 列终端下先丢的是用时这类细节，不是"这是哪个点"。
        seg_layer = self._phase_str()
        seg_pct = f"{self._glyph(f)} {f * 100:.0f}%"
        seg_time = f"已用{fmt_sec(el)}·再~{fmt_sec(eta)}{'（估）' if rough else ''}"
        seg_det = f"idx={self.idx} {self.attn} T={self.cur_T}" if (self.idx is not None and self.stage == "S2 前向") else ""
        seg_dead = f"[!] rank0 rc={self.dead[1]}" if self.dead else ""
        out = ""
        for seg, sep in ((seg_layer, ""), (seg_pct, " │ "), (label, " │ "), (seg_time, " │ "),
                         (seg_det, " │ "), (seg_dead, " │ ")):
            if not seg:
                continue
            cand = out + (sep if out else "") + seg
            if len(cand) > width - 1:
                break
            out = cand
        if not out:                                  # 极窄：至少给个百分比
            out = f"{f * 100:.0f}%"[: max(1, width - 1)]
        return out[: max(1, width - 1)]


class Bar:
    """渲染线程 + 真终端 fd 的持有者。**所有公开方法都自己在共享锁里跑。**

    与 launch_mini 的约定：
      · `note_line(rank, line)` —— pump 每读到一行就调（解析与回显无关，`--console off` 也要调）；
      · `after_line()` —— 回显日志行**之前**调：擦掉进度条，避免和 tee 转发过来的日志行抢同一行；
      · `write_line(text)` —— 所有 stdout 输出的唯一入口。
    """

    def __init__(self, label: str = "", fd: int = 9, every: float = 30.0, redraw: float = 0.25,
                 est: dict | None = None, lock: threading.RLock | None = None,
                 stream=None, tty_off: bool = False, t0: float | None = None, note: str = ""):
        self.label = label
        self.note = note          # 全 run 级信息（总已用/剩余）：最低优先级，窄终端先丢它
        self.every = max(1.0, every)
        self.redraw = max(0.05, redraw)
        self.state = BarState(est, t0=t0 or time.monotonic())
        self.lock = lock or threading.RLock()
        self.stream = stream or sys.stdout
        self.enabled = bool(label)
        self._tty_fd = None if tty_off else self._open_tty(fd)
        self._tty_file = None
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self._pending = False          # 有日志行刚写出去 → 该重画了
        self._visible = False          # 当前终端上有没有进度条
        self._last_paint = 0.0
        self._last_snap = 0.0
        self._min_snap = min(MIN_SNAP_MAX, max(1.0, every / 6))   # 快照最小间隔（--bar-every 小时同步变小）
        self._stage_seen = None
        self._pass_seen = 0
        self._est_out = None

    # ── 终端 ──
    @staticmethod
    def _open_tty(fd: int):
        """降级链：显式 fd（驱动 `exec 9>&1` 给的）→ 标准流 → /dev/tty → 关掉（只留日志快照）。"""
        try:
            if fd is not None and os.isatty(fd):
                return fd
        except OSError:
            pass
        try:
            if any(os.isatty(i) for i in (0, 1, 2)):
                return os.open("/dev/tty", os.O_WRONLY)
        except OSError:
            pass
        return None

    def _tty_write(self, s: str) -> None:
        if self._tty_fd is None:
            return
        try:
            os.write(self._tty_fd, s.encode("utf-8", "replace"))
        except OSError:
            self._tty_fd = None          # 终端没了（被 kill / 关闭）→ 静默降级

    def _tty_cols(self) -> int:
        try:
            return max(40, os.get_terminal_size(self._tty_fd).columns)
        except OSError:
            return 100

    def _erase(self) -> None:
        if self._tty_fd is not None and self._visible:
            self._tty_write("\r\x1b[K")
            self._visible = False

    def _paint(self, now: float) -> None:
        if self._tty_fd is None:
            return
        cols = self._tty_cols()
        line = self.state.render(now, self.label, cols, self.note)
        self._tty_write("\r\x1b[K" + line)     # 帧自带清行：最坏只是位置偏一格，不污染历史行
        self._visible = True
        self._last_paint = now

    # ── 快照 ──
    def _write_stdout(self, text: str) -> None:
        try:
            with self.lock:
                self.stream.write(text + "\n")
                self.stream.flush()
        except (BrokenPipeError, ValueError, OSError):
            pass

    def _snapshot(self, now: float, force: bool = False) -> None:
        if now - self._last_snap < (0.0 if force else self._min_snap):
            return
        self._last_snap = now
        self._erase()
        self._write_stdout(self.state.snapshot(now, self.label, self.note))

    # ── 生命周期 ──
    def start(self) -> "Bar":
        # ★ 渲染线程**无条件**起（没终端也起）：日志快照就靠它，nohup/重定向时终端那一路
        #   自然降级成 no-op（`_paint`/`_erase` 里 `_tty_fd is None` 直接返回）。
        if self.enabled:
            self._th = threading.Thread(target=self._loop, name="bar", daemon=True)
            self._th.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2.0)
            self._th = None
        with self.lock:
            self._erase()
        self._snapshot(time.monotonic(), force=True)

    def _loop(self) -> None:
        while not self._stop.wait(min(self.redraw, 0.25)):
            try:
                now = time.monotonic()
                with self.lock:
                    st = self.state
                    if now - self._last_snap >= self.every or \
                       st.stage != self._stage_seen or st.pass_done != self._pass_seen:
                        self._stage_seen, self._pass_seen = st.stage, st.pass_done
                        self._snapshot(now)
                    if self._tty_fd is not None and (self._pending or
                                                     now - self._last_paint >= self.redraw):
                        self._erase()
                        self._paint(now)
                        self._pending = False
            except Exception:                      # noqa: BLE001 —— 条坏了也不能影响采集
                self._tty_fd = None

    # ── 被 launch_mini 调用 ──
    def note_line(self, rank: int, line: str) -> None:
        if not self.enabled:
            return
        try:
            with self.lock:
                self.state.feed(line, rank, time.monotonic())
        except Exception:                          # noqa: BLE001
            pass

    def note_rank_exit(self, rank: int, rc: int) -> None:
        if not self.enabled:
            return
        with self.lock:
            self.state.note_rank_exit(rank, rc, time.monotonic())

    def after_line(self) -> None:
        """回显一条日志行之前调：先把条擦掉。"""
        if not self.enabled:
            return
        try:
            with self.lock:
                self._erase()
                self._pending = True
        except Exception:                          # noqa: BLE001
            pass

    def write_line(self, text: str = "") -> None:
        """唯一 stdout 入口（launch_mini 的 say() 用它）。"""
        if not self.enabled:
            print(text, flush=True)
            return
        try:
            with self.lock:
                self._erase()
                self.stream.write(text + "\n")
                self.stream.flush()
                self._pending = True
        except (BrokenPipeError, ValueError, OSError):
            pass

    def emit_est(self) -> None:
        """退出时把本轮实测回标（供驱动侧 ETA 模型离线校准）。"""
        e, m = self.state.est, self.state.meas
        self.write_line("@@bar-est " + ",".join(
            f"{k}={m.get(k, e[k])}" for k in ("cold", "steady", "trace")) +
            f",R={self.state.R or 0},iters={self.state.iters or 0},prof={bool(self.state.prof)}")


# ── 自测 / 回放（不碰卡）──────────────────────────────────────────────────────
FIXTURES = [
    "/mnt/xds/sfs/l00972766/hit97_fromB/L65536/logs2/rank0.log",   # 无 prof 轮
    "/mnt/xds/sfs/l00972766/hit97_fromB/L65536/logs/rank0.log",    # prof 轮（含 trace 解析）
    "/mnt/xds/sfs/l00972766/hit97_fromB/L65536/logs2/all.log",     # 8 rank 混合（假阳性检查）
]


def _selftest() -> int:
    bad = 0
    now = 1000.0

    def chk(cond: bool, what: str) -> None:
        nonlocal bad
        if not cond:
            bad += 1
            print(f"  ✗ {what}")

    # ① 行级解析
    st = BarState(t0=now)
    st.feed("== 加载 vendor 算子命名空间", 0, now); chk(st.stage.startswith("S0"), "锚点 S0")
    st.feed("  进程组就绪：rank 0/8", 0, now);      chk(st.stage.startswith("S1"), "锚点 S1")
    st.feed("== 真跑：L=65536 layers=全部 iters=4 卡=npu:0 megamoe=ascendc prof=False", 0, now)
    chk(st.iters == 4 and st.prof is False, "真跑行给 iters/prof")
    st.feed("  输入缓冲已建：(2048, 5120) torch.bfloat16（各轮复用）", 0, now)
    chk(st.R == 2048, "输入缓冲给 R")
    st.feed("  [起跑线] 跨 rank barrier ×1 完成 → 现在开 profiler 开采", 0, now)
    chk(st.armed, "起跑线 armed")
    st.feed("  [ 1/40] layer  0 win              T=2048 算子 24", 0, now)
    chk((st.n, st.n_total, st.idx, st.attn, st.cur_T) == (1, 40, 0, "win", 2048), "层行字段")
    st.feed("  [40/40] layer 39 c1a_consumer     T=2048 算子 24", 0, now)
    chk(st.n == 40, "末层行")
    st.feed("  [iter 1/4] 32.887s（822.18 ms/层）", 0, now)
    chk(st.pass_done == 1 and "cold" in st.meas, "iter 行 → 首 pass 实测")
    st.feed("  [iter 4/4] 0.177s（4.41 ms/层）", 0, now)
    chk(st.pass_done == 4 and st.stage.startswith("S3"), "最后一 pass → S3")
    st.feed("[15:57:43]  69.3% │ S2 逐层：前向执行 层 25/39 c1a_consumer │ 已用 41s", 0, now)
    chk(True, "进度行不炸（不参与解析）")

    # ② 单调性 + 渲染宽度 + 快照纯净度（用真实日志逐行喂）
    for path in FIXTURES[:2]:
        if not os.path.exists(path):
            print(f"  （fixture 不在，跳过：{path}）")
            continue
        st = BarState(t0=now)
        prev = 0.0
        w = now
        with open(path, errors="ignore") as fh:
            for i, line in enumerate(fh):
                w += 0.05                                   # 合成时钟：只验单调性，不验绝对 ETA
                st.feed(line.rstrip("\n"), 0, w)
                f, _ = st.frac(w)
                chk(0.0 <= f <= 1.0, f"{os.path.basename(path)}:{i} frac 越界 {f}")
                chk(f >= prev - 1e-9, f"{os.path.basename(path)}:{i} 进度倒退 {prev}→{f}")
                prev = f
                for width in (20, 40, 80, 200):
                    chk(len(st.render(w, "点3/8 L=65536 prof轮", width)) <= width - 1,
                        f"render({width}) 超宽")
                s = st.snapshot(w, "点3/8 L=65536 prof轮")
                chk(("\r" not in s) and ("\x1b" not in s) and ("\x08" not in s),
                    "snapshot 含控制字符")
        print(f"  ✓ {os.path.basename(path)}：{i + 1} 行喂完，单调 + 宽度 + 快照纯净")

    # ③ 假阳性：8 rank 混合日志里不应多出 layer/iter 命中
    mix = FIXTURES[2]
    if os.path.exists(mix):
        rx_pref = re.compile(r"^\[r\d\] ")
        n_layer = n_iter = n_fp = 0
        with open(mix, errors="ignore") as fh:
            for line in fh:
                line = rx_pref.sub("", line.rstrip("\n"))     # all.log 带 [rN] 前缀；真实路径喂裸行
                if RE_LAYER.match(line):
                    n_layer += 1
                elif RE_ITER.match(line):
                    n_iter += 1
                elif ("layer" in line or "[iter" in line) and not RE_RUN.match(line):
                    n_fp += 1       # 含关键字却没被任何正则/锚点吃掉 → 疑似漏解析
                    if n_fp <= 3:
                        print(f"    ? {line[:100]}")
        print(f"  ✓ all.log：layer 命中 {n_layer}（期望 1280=8×40×4）、iter 命中 {n_iter}（期望 32）、"
              f"疑似漏解析 {n_fp}")
        chk(n_layer == 1280, f"all.log layer 命中数应为 1280，实为 {n_layer}")
        chk(n_iter == 32, f"all.log iter 命中数应为 32，实为 {n_iter}")
        chk(n_fp == 0, f"疑似漏解析 {n_fp} 行")
    else:
        print(f"  （fixture 不在，跳过假阳性检查：{mix}）")

    print("自测" + ("通过 ✓" if bad == 0 else f"失败 ✗（{bad} 项）"))
    return 0 if bad == 0 else 1


def _replay(path: str, label: str, speed: float, every: float) -> int:
    """按真实日志回放：走的是真的 Bar（真渲染线程、真 fd 逻辑），只是没有子进程。"""
    lock = threading.RLock()
    bar = Bar(label=label, every=every, lock=lock)
    print(f"[replay] tty={bar._tty_fd} speed={speed} label={label!r}")
    bar.start()
    try:
        with open(path, errors="ignore") as fh:
            for line in fh:
                line = line.rstrip("\n")
                bar.after_line()                    # 模拟 pump：先擦条，再"回显"
                bar.note_line(0, line)
                with lock:
                    bar.stream.write("[r0] " + line + "\n")
                    bar.stream.flush()
                time.sleep(min(0.2, 0.2 / max(1.0, speed)))
    finally:
        bar.stop()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="采集进度条（自测/回放）")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--replay", default="")
    ap.add_argument("--label", default="回放")
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--bar-every", type=float, default=5.0)
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.replay:
        return _replay(a.replay, a.label, a.speed, a.bar_every)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

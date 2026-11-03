#!/usr/bin/env python3



"""多 rank 启动器：每个 rank 一个进程，**每个进程只看到自己那张物理卡**。

为什么要这样：这台机器上"逻辑卡号 ≥1"会踩设备索引坑（`hc_pre` aicore fault，见 memory
`npu-logical-device-index-bug`）。给每个子进程设 `ASCEND_RT_VISIBLE_DEVICES=<物理卡>`
之后，子进程里那张卡永远是逻辑 `npu:0` ✓ 既真实用到指定物理卡、又绕开坑。

用法（ids 文件仓库不自带，可用 `--prompt` 走 tokenizer，或自备 .pt/.json）：
    python3 launch_mini.py --config configs/mini_real.yaml --world 2 --cards 1,4 \
        --ids-file <ids.pt> --collect-layers <输出目录> --progress off

日志（2026-10-15 起，用户口径「所有日志打印到控制台并同时落盘」）：每个 rank 的输出**边跑边**
写三处 —— 控制台（带 `[rN]` 前缀）、`<logdir>/rankN.log`、合并的 `<logdir>/all.log`
（行内加锁，整行不会撕开）。`--console rank0|off` 收窄控制台回显；某 rank 静默超过
`--silent-warn`（默认 300s）会打一行告警（长跑卡死时不用去翻文件才知道）。

进度条（2026-10-15 起，`progress_bar.py`）：`--bar "<标签>"` 打开。解析 **rank0** 的输出
（不新增埋点）算进度/ETA —— 终端上一行**原地刷新**，同时每 `--bar-every` 秒往 stdout
（→ 日志）追加一行**纯文本快照**。终端那一路写 `--bar-fd`（驱动用 `exec 9>&1` 把"真终端"
递进来），**不进 tee**，所以日志里永远不会有 `\r`。详见 `progress_bar.py` 顶部注释。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ds41_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    _q = _os.path.join(_ds41_root, _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)

import argparse
import atexit
import os
import signal
import socket
import subprocess
import sys
import re
import tempfile
import threading
import time

import progress_bar                                    # noqa: E402

def _no_log_files(arg) -> bool:
    """`--logdir none|off|null|/dev/null`：**不写任何日志文件**（rank/all 落 /dev/null），
    只为 CANN 的 ASCEND_PROCESS_LOG_PATH 建一个临时目录。--pure-log 默认走这条。"""
    return str(arg or "").strip().lower() in ("none", "off", "null", "devnull", "/dev/null", "0")


_IO_LOCK = threading.RLock()     # 三个 sink（控制台 / rank 日志 / all.log）+ 进度条共用
_LAST_SEEN: dict = {}            # rank → 最后一次有输出的时刻（主循环用它做静默告警）
_BAR: progress_bar.Bar | None = None   # main() 里建；say()/pump() 都走它


def say(text: str = "") -> None:
    """**唯一**的 stdout 出口：有进度条时先擦条再写，写完标脏让渲染线程重画。"""
    if _BAR is not None:
        _BAR.write_line(text)
    else:
        print(text, flush=True)


_SUMMARY_RX = re.compile(
    r"(\[point |\[mem\] |多轮耗时|纯 (pass|forward) 吞吐|barrier timeout"
    # ★ 2026-10-27：崩溃证据**永不吞**（这些只在出事时出现；正常跑一条都不会多）
    r"|Traceback|RuntimeError|EE9999|EZ9999|ERROR|Segmentation|core dumped"
    r"|Aborted|out of memory|OOM|Signal)")


def pump(rank, stream, rank_fh, all_fh, echo, bar=None, summary_only=False):
    """把某个 rank 的 stdout/stderr 边读边分发。

    ★ 2026-10-27 用户规则：无日志模式（rank_fh/all_fh 皆为 None）下**不做任何 write**
    （连 /dev/null 都不写）——只读管道 + 内存里解析；回显也只放行汇总行（summary_only）。
    解析进度用的是**裸行**（不带 `[rN]` 前缀）——`all.log` 里那种带前缀的行正则匹配不上。
    """
    tag = f"[r{rank}] "
    _dump = 0                     # >0：异常块放行计数（Traceback 之后的 File/xxxError 行）
    try:
        for raw in stream:
            line = raw.rstrip("\n")
            _LAST_SEEN[rank] = time.time()
            if summary_only and "Traceback" in line:
                _dump = 40            # 一次异常块最多 40 行，防止万一刷屏
            _summ = bool(_SUMMARY_RX.search(line))
            if _dump > 0:
                _summ = True
                _dump -= 1
            with _IO_LOCK:
                if rank_fh is not None:
                    rank_fh.write(line + "\n")
                    rank_fh.flush()
                if all_fh is not None:
                    all_fh.write(tag + line + "\n")
                    all_fh.flush()
                if bar is not None:
                    bar.note_line(rank, line)      # ★ 与 echo 无关：--console off 也要解析
                if echo and (not summary_only or _summ):
                    if bar is not None:
                        bar.after_line()           # ★ 先擦条，再回显（避免和 tee 抢同一行）
                    sys.stdout.write(tag + line + "\n")
                    sys.stdout.flush()
    except Exception:                     # noqa: BLE001 —— 回显线程不许把主流程带崩
        pass
    finally:
        try:
            stream.close()
        except Exception:                 # noqa: BLE001
            pass


def free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ── 自测钩子：都复用**真的** Bar / pump，只是不启 NPU 子进程 ──────────────────────
_SYNTH_FIXTURE = [                       # 没有真日志时的兜底剧本（格式照抄 runner 真实输出）
    "== 加载 vendor 算子命名空间",
    "  [exec_ops] torch.ops.custom: 32 个算子",
    "  进程组就绪：rank 0/8",
    "== 真跑：L=65536 layers=全部 iters=4 卡=npu:0 megamoe=ascendc prof=False",
    "  输入缓冲已建：(2048, 5120) torch.bfloat16（各轮复用）",
    "  口径：elb/强均路由=ON；跨 rank barrier=仅起跑线一次；weights=synthetic",
    "  [起跑线] 跨 rank barrier ×1 完成 → 现在开 profiler 开采",
    "[elb/balroute] 强均路由 ON：topk_idx=(t*6+k)%384，T=2048",
] + [f"  [{n:>2}/40] layer {n - 1:>2} {'win' if n < 3 else 'c1a_consumer':<16} T=2048 算子 24"
     for n in range(1, 41)] + [
    "  [iter 1/4] 32.887s（822.18 ms/层）",
] + [f"  [{n:>2}/40] layer {n - 1:>2} c1a_consumer     T=2048 算子 24" for n in range(1, 41)] + [
    "  [iter 2/4] 0.179s（4.48 ms/层）",
    "  多轮耗时：['32.887', '0.179', '0.179', '0.179']s  均(除首轮)=0.179s  最快=0.179s",
    "  显存：峰值 6.90 GiB / 保留 10.41 / 总 91.8 GiB",
]


class _FixtureStream:
    """把一份日志当"子进程的输出流"喂给 pump（带节奏 + 每个 rank 略微错开，模拟真实交错）。"""

    def __init__(self, lines, lps: float, rank: int, repeat: int = 1):
        self.lines = list(lines) * max(1, repeat)
        self.delay = 1.0 / max(1.0, lps)
        self.rank = rank

    def __iter__(self):
        for line in self.lines:
            time.sleep(self.delay * (1.0 + 0.03 * self.rank))
            yield line + "\n"

    def close(self) -> None:
        pass


def _run_replay(args) -> int:
    """`--bar-replay <rank0.log>`：单路回放，人肉看条/快照。"""
    global _BAR
    _BAR = progress_bar.Bar(label=args.bar or "回放", note=args.bar_note, fd=args.bar_fd, every=args.bar_every,
                            redraw=args.bar_redraw, lock=_IO_LOCK,
                            tty_off=(args.bar_tty == "off"))
    _BAR.start()
    say(f"[回放] {args.bar_replay}  速度 {args.bar_replay_speed:g} 行/秒  "
        f"终端={'fd %d' % _BAR._tty_fd if _BAR._tty_fd is not None else '无（只落日志快照）'}")
    try:
        with open(args.bar_replay, errors="ignore") as fh:
            pump(0, _FixtureStream([l.rstrip("\n") for l in fh], args.bar_replay_speed, 0),
                 open(os.devnull, "w"), open(os.devnull, "w"), True, _BAR)
    finally:
        _BAR.stop()
    return 0


def _run_demo(args) -> int:
    """`--bar-demo N`：N 路假 rank **走真实 pump 路径**（锁/回显/擦条/快照全一样）。"""
    global _BAR
    n = max(1, args.bar_demo)
    logdir = os.path.abspath(args.logdir)
    os.makedirs(logdir, exist_ok=True)
    all_fh = open(os.path.join(logdir, "all.log"), "w")
    lines = _SYNTH_FIXTURE
    if args.bar_replay and os.path.exists(args.bar_replay):
        with open(args.bar_replay, errors="ignore") as fh:
            lines = [l.rstrip("\n") for l in fh]
    _BAR = progress_bar.Bar(label=args.bar or f"{n} 路假 rank（demo）", fd=args.bar_fd,
                            every=args.bar_every, redraw=args.bar_redraw,
                            lock=_IO_LOCK, tty_off=(args.bar_tty == "off"))
    _BAR.start()
    say(f"[demo] {n} 路假 rank · {len(lines)} 行/路 · 速度 {args.bar_replay_speed:g} 行/秒 · "
        f"控制台回显={args.console} · "
        f"终端={'fd %d' % _BAR._tty_fd if _BAR._tty_fd is not None else '无（只落日志快照）'}")
    ths = []
    for r in range(n):                       # rank0 用真 fixture 的"骨架"，其余 rank 复制同一份（模拟同构输出）
        fh = open(os.path.join(logdir, f"rank{r}.log"), "w")
        th = threading.Thread(target=pump, name=f"pumpr{r}", daemon=True,
                              args=(r, _FixtureStream(lines, args.bar_replay_speed, r),
                                    fh, all_fh, args.console == "all" or (args.console == "rank0" and r == 0),
                                    _BAR))
        th.start()
        ths.append((th, fh))
    for th, _fh in ths:
        th.join()
        _fh.close()
    all_fh.close()
    _BAR.stop()
    say("== demo 结束（8 路日志在 %s/rank*.log，逐行比对应完全一致）" % logdir)
    return 0


def main():
    global _BAR
    # ★ 2026-10-21：日志等级；pure 时驱动/rank 的 print 全部丢弃，只留进度条（Bar 走专用 fd）。
    import ds41_log
    ds41_log.install()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--cards", required=True, help="物理卡号，逗号分隔，数量必须 = world（如 1,4）")
    ap.add_argument("--logdir", default=".tmp")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--console", default="all", choices=["all", "rank0", "off", "summary"],
                    help="控制台回显：all=所有 rank 带 [rN] 前缀 / rank0=只看 rank0 / off=只落盘")
    ap.add_argument("--silent-warn", type=float, default=300.0,
                    help="某 rank 静默超过这么多秒就打一行告警（0=关）")
    g = ap.add_argument_group("进度条")
    g.add_argument("--bar", default="", help="进度条标签（如 '点3/8 L=65536 prof轮'）；空=关")
    g.add_argument("--bar-note", default="", help="进度条尾注（全 run 级信息，如 '总已用12m 剩余~9m'）")
    g.add_argument("--bar-every", type=float, default=30.0, help="日志快照间隔（秒）")
    g.add_argument("--bar-redraw", type=float, default=0.25, help="终端重画间隔（秒）")
    g.add_argument("--bar-fd", type=int, default=9,
                   help="真终端 fd（驱动用 exec 9>&1 递进来；默认 9，没开就退回 /dev/tty）")
    g.add_argument("--bar-tty", default="auto", choices=["auto", "off"],
                   help="off=不画终端条，只留日志快照")
    g.add_argument("--bar-replay", default="", help="自测：回放一份 rank0.log（不启子进程）")
    g.add_argument("--bar-replay-speed", type=float, default=30.0, help="回放速度（行/秒）")
    g.add_argument("--bar-demo", type=int, default=0, help="自测：N 路假 rank 走真实 pump 路径")
    # 其余参数原样透传给 runner.py
    args, passthru = ap.parse_known_args()
    if args.bar_replay:
        return _run_replay(args)
    if args.bar_demo:
        return _run_demo(args)
    cards = [c.strip() for c in args.cards.split(",") if c.strip()]
    if len(cards) != args.world:
        sys.exit(f"卡数({len(cards)}) 必须等于 world({args.world})")

    port = free_port()
    master = f"127.0.0.1:{port}"
    _no_logs = _no_log_files(args.logdir)
    if _no_logs:
        # ★ 2026-10-27（用户规则）：--pure-log 默认**不写日志文件**（rank/all 全丢 /dev/null），
        #   否则 8 rank × NFS 写盘会加重 host-bound。CANN 的进程日志目录放 **tmpfs**：
        #   ASCEND_SLOG_PRINT_TO_STDOUT=1 只干掉 plog（实测 33,683 B → 0），但每个进程
        #   仍会写 run/device-0/device-<pid>_*.log（实测 8,830 B，ASCEND_GLOBAL_LOG_LEVEL=3/4
        #   都去不掉）。指到 /dev/shm → 纯内存，零磁盘/NFS 写（2026-10-27 实测验证）。
        _tmpfs = ("/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)
                  else (os.environ.get("TMPDIR") or "/tmp"))
        logdir = tempfile.mkdtemp(prefix="ds41_nolog_", dir=_tmpfs)
        all_log = "（关闭）"
        all_fh = None                    # 不写任何文件（含 /dev/null）
    else:
        logdir = os.path.abspath(args.logdir)
        os.makedirs(logdir, exist_ok=True)
        all_log = os.path.join(logdir, "all.log")
        all_fh = open(all_log, "w")
    # CANN 运行日志默认落在本次 --logdir 下（换机器不再依赖共享盘固定路径；env 显式给则优先）
    ascend_log = os.environ.get("ASCEND_PROCESS_LOG_PATH", os.path.join(logdir, "ascend_log"))
    os.makedirs(ascend_log, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    # ★ 进度条：在起子进程**之前**建（它的时钟就是"这一轮从 launch_mini 起算"）
    _BAR = progress_bar.Bar(label=args.bar, fd=args.bar_fd, every=args.bar_every,
                            redraw=args.bar_redraw, lock=_IO_LOCK, note=args.bar_note,
                            tty_off=(args.bar_tty == "off"))
    if args.bar:
        atexit.register(_BAR.stop)
        for _sig in (signal.SIGINT, signal.SIGTERM):          # 被 Ctrl-C / kill：先擦干净再退
            try:
                signal.signal(_sig, lambda *_a: (_BAR.stop(), sys.exit(130)))
            except (ValueError, OSError):
                pass
        _BAR.start()
    say(f"== 启动 {args.world} 个 rank（物理卡 {cards}，master {master}）")
    say(f"   透传给 runner 的参数：{passthru}")
    say(f"   日志：{logdir}/rank*.log（每 rank）+ {all_log}（合并，带 [rN] 前缀）"
        f"；控制台回显 = {args.console}")
    if args.bar:
        say(f"   进度条：{args.bar!r}  终端={'有（fd %d）' % _BAR._tty_fd if _BAR._tty_fd is not None else '无（只落日志快照）'}"
            f"  快照每 {args.bar_every:g}s")
    procs = []
    # DS41M_NO_VISIBLE=1：不做"每进程只见一张卡（逻辑恒 0）"的可见性收缩，改成
    #   「8 卡全可见 + --device npu:<物理卡>」（= 厂商自测/官方引擎的启动方式）。
    #   排查用：mega_moe ascendc 的 topo/rank 映射要看设备拓扑，逻辑号恒 0 可能让它算错。
    no_visible = bool(os.environ.get("DS41M_NO_VISIBLE"))
    for r in range(args.world):
        env = os.environ.copy()
        if no_visible:
            env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
        else:
            env["ASCEND_RT_VISIBLE_DEVICES"] = cards[r]  # ← 每个进程只见自己那张卡（逻辑 0）
        env["MASTER_ADDR"], env["MASTER_PORT"] = master.split(":")
        if _no_logs:
            env["DS41M_SILENT"] = "1"    # runner 不打印逐层进度（无日志也无人解析）
            # CANN 的 slog 默认写 plog 文件（1.4–2MB/次）；重定向到 stdout（本进程管道→内存丢弃）
            env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
            env["ASCEND_GLOBAL_LOG_LEVEL"] = env.get("ASCEND_GLOBAL_LOG_LEVEL", "3")
        env.setdefault("ASCEND_PROCESS_LOG_PATH", ascend_log)
        env.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "3")
        log = "（关闭）" if _no_logs else os.path.join(logdir, f"rank{r}.log")
        cmd = [sys.executable, "-u", os.path.join(here, "runner.py"), args.config,
               "--rank", str(r), "--world", str(args.world),
               "--device", (f"npu:{cards[r]}" if no_visible else "npu:0"),
               "--master", master] + passthru
        say(f"   rank {r} ← 卡 {cards[r]}  日志 {log}" if not _no_logs else f"   rank {r} ← 卡 {cards[r]}  日志 关闭")
        rank_fh = None if _no_logs else open(log, "w")
        # close_fds 用默认（True）：8 个 rank **拿不到** fd 9/终端，进度条那一份是 launch_mini 独有的
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1,
                             errors="replace")
        _echo = (args.console == "all" or args.console == "summary"
                 or (args.console == "rank0" and r == 0))
        th = threading.Thread(target=pump, name=f"pump{r}", daemon=True,
                              args=(r, p.stdout, rank_fh, all_fh, _echo, _BAR,
                                    args.console == "summary"))
        th.start()
        procs.append({"r": r, "p": p, "log": log, "fh": rank_fh, "th": th})

    t0 = time.time()
    codes: dict = {}
    warns = {r: 0.0 for r in range(args.world)}
    while len(codes) < len(procs) and time.time() - t0 < args.timeout:
        now = time.time()
        for pr in procs:
            r = pr["r"]
            if r in codes:
                continue
            if pr["p"].poll() is not None:
                codes[r] = pr["p"].returncode
                if _BAR is not None:
                    _BAR.note_rank_exit(r, pr["p"].returncode)   # 条上标"rank0 已退出"（rank0 才有意义）
                # ★ 2026-11-03：任一等 rank 非零退出 = 整组已失败。设备 fatal（507015/
                #   aicore trap）常常是 C++ terminate，走不到 Python 的组内标记；这里
                #   直接把其余 rank 杀掉，避免干等 HCCL watchdog（实测 ~60s）。
                if codes[r] != 0:
                    _rest = [q for q in procs if q["r"] not in codes]
                    if _rest:
                        say(f"  ✗ rank{r} 非零退出({codes[r]}) → 立即终止其余 rank "
                            f"（不等 HCCL watchdog）")
                    for q in _rest:
                        q["p"].kill()
                        codes[q["r"]] = -9
                continue
            seen = _LAST_SEEN.get(r, t0)
            if (args.silent_warn > 0 and now - seen > args.silent_warn
                    and now - warns[r] > args.silent_warn):
                warns[r] = now
                say(f"   ⏳ rank{r} 已静默 {now - seen:.0f}s（仍在跑，pid {pr['p'].pid}）")
        time.sleep(1)
    killed = [pr["r"] for pr in procs if pr["r"] not in codes]
    for pr in procs:
        if pr["r"] in killed:
            pr["p"].kill()
            codes[pr["r"]] = -9
    for pr in procs:                      # 子进程已退出 → pipe EOF → 回显线程收尾
        pr["th"].join(timeout=5)
        if pr["fh"] is not None:          # ★ 无日志模式：句柄为 None（不是 /dev/null）
            pr["fh"].close()
    if all_fh is not None:
        all_fh.close()
    if _no_logs:
        import shutil as _sh
        try:
            _sh.rmtree(logdir, ignore_errors=True)   # 无日志模式的临时目录不留垃圾
        except Exception:                            # noqa: BLE001
            pass
    if _BAR is not None:
        _BAR.stop()                       # ★ 先擦条、停渲染线程，再打收尾（否则收尾行会被贴条）
    say(f"== 全部结束（{time.time()-t0:.0f}s）: " +
        ", ".join(f"rank{r}={c}" for r, c in sorted(codes.items())))
    if killed:
        say(f"   ⚠️ 超时被杀（>{args.timeout}s）：{killed}")
    for pr in procs:
        if codes[pr["r"]] != 0:
            say(f"  rank{pr['r']} 非零退出（日志已关闭，无尾部）"
                if _no_logs else f"  rank{pr['r']} 非零退出，日志尾部：")
            if not _no_logs:
                with open(pr["log"], errors="ignore") as fh:
                    for line in fh.read().splitlines()[-8:]:
                        say(f"    {line[:160]}")
    if _BAR is not None:
        _BAR.emit_est()                   # 供驱动侧 ETA 模型校准（@@bar-est …）
    sys.exit(0 if all(c == 0 for c in codes.values()) else 1)


if __name__ == "__main__":
    main()

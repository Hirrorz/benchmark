#!/usr/bin/env python3.11
# coding=utf-8
"""bench_server.py — 常驻 benchmark supervisor（8 rank + 简单进度条，无 TUI）。

用法（命令由 tools/bench_command_builder.html 生成）：
  python3.11 tools/bench_server.py start \
    --config configs/mini_real.yaml --world 8 --cards 0,1,2,3,4,5,6,7 \
    --logdir /mnt/xds/sfs/l00972766/tmp/bench_1019 --mode auto --host-cache auto
  python3.11 tools/bench_server.py status
  python3.11 tools/bench_server.py stop

职责：
  · 按 launch_mini 的卡隔离方式拉起 8 个 `runner.py --serve`；
  · 等 8 rank ready，建 server.sock 接受 bench_cli 命令；
  · 把命令同时发给 8 rank（ctl socket），等 ACK/result；
  · rank0 的进度行实时转发给 client（client 画一行进度条）；
  · 任一 rank 退出/超时 → 整组 kill（HCCL 不能单 rank 重启）。
不负责：host cache 的 start/stop（仍用 tools/moe_host_cache.py 手动控制）。
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bench_protocol as P


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _job_dir(job: str) -> str:
    d = os.path.join(P.job_root(), job)
    os.makedirs(d, exist_ok=True)
    return d


def _resolve_job(args) -> str:
    if args.job:
        return args.job
    return "bench_" + time.strftime("%m%d_%H%M%S")


def _latest_job() -> str:
    root = P.job_root()
    try:
        ds = [d for d in os.listdir(root)
              if os.path.isdir(os.path.join(root, d))]
        if not ds:
            return ""
        return max(ds, key=lambda d: os.path.getmtime(os.path.join(root, d)))
    except OSError:
        return ""


def _sock_of(args) -> str:
    if getattr(args, "socket", ""):
        return args.socket
    job = getattr(args, "job", "") or _latest_job()
    return P.default_server_sock(job) if job else ""


class RankConn:
    """一个 rank 的 ctl 连接：后台读线程 → queue，dispatch 按 cmd_id 取消息。"""

    def __init__(self, rank: int, path: str, timeout: float = 30.0):
        self.rank = rank
        self.path = path
        self.q: queue.Queue = queue.Queue()
        self.dead = False
        self.sock = None
        self.jl = None
        self._connect(timeout)

    def _connect(self, timeout):
        t0 = time.time()
        last = None
        while time.time() - t0 < timeout:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(5.0)
                s.connect(self.path)
                s.settimeout(None)
                self.sock = s
                self.jl = P.JsonLines(s)
                threading.Thread(target=self._reader, name=f"r{self.rank}ctl",
                                 daemon=True).start()
                return
            except OSError as exc:                      # noqa: BLE001
                last = exc
                time.sleep(0.1)
        raise RuntimeError(f"rank{self.rank} ctl 连接失败 {self.path}: {last}")

    def _reader(self):
        try:
            while True:
                msg = self.jl.recv()
                if msg is None:
                    break
                self.q.put(msg)
        except Exception:                               # noqa: BLE001
            pass
        finally:
            self.dead = True
            self.q.put({"type": "closed"})

    def send(self, obj):
        if self.dead or self.sock is None:
            raise RuntimeError(f"rank{self.rank} 控制连接已断开")
        P.send_json(self.sock, obj)

    def try_get(self, cmd_id, types):
        """非阻塞取一条匹配消息；其余消息放回队列（命令串行，顺序影响可忽略）。"""
        stash = []
        got = None
        while True:
            try:
                msg = self.q.get_nowait()
            except queue.Empty:
                break
            if msg.get("type") == "closed":
                self.dead = True
                continue
            if msg.get("cmd_id") not in (None, cmd_id):
                stash.append(msg)
                continue
            if got is None and msg.get("type") in types:
                got = msg
            else:
                stash.append(msg)
        for s in stash:
            self.q.put(s)
        return got

    def wait_for(self, cmd_id, types, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self.try_get(cmd_id, types)
            if msg is not None:
                return msg
            if self.dead:
                return None
            time.sleep(0.02)
        return None

    def close(self):
        try:
            if self.jl:
                self.jl.close()
        except Exception:                               # noqa: BLE001
            pass
        try:
            if self.sock:
                self.sock.close()
        except Exception:                               # noqa: BLE001
            pass


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.job = _resolve_job(args)
        self.job_dir = _job_dir(self.job)
        self.logdir = os.path.abspath(args.logdir)
        os.makedirs(self.logdir, exist_ok=True)
        self.server_sock_path = os.path.join(self.job_dir, "server.sock")
        self.ranks: list[RankConn] = []
        self.procs = []
        self.ready = set()
        self.ready_evt = threading.Event()
        self.all_fh = None
        self.rank_fhs = {}
        self.client = None
        self.client_lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.busy = threading.Lock()
        # 服务端自己画条只用于 --warmup-on-start（没有 client 可转发时）
        self.bar = P.StageBar(stream=sys.stderr, enabled=bool(getattr(args, "bar", True)),
                              width=16, every=0.5)
        self.bar_lock = threading.Lock()
        self.bar_active = False
        self.mode = args.mode
        self.shutdown_flag = False
        self._shutdown_done = False
        self.state_path = os.path.join(self.job_dir, "state.json")
        self.started = time.time()

    # ───────────── env / spawn ─────────────
    def _resolve_host_cache(self) -> str:
        hc = (self.args.host_cache or "auto").strip()
        default_sock = "/tmp/ds41_moe_host_cache.sock"
        if hc.lower() in ("1", "on", "true", "yes", "auto"):
            st = default_sock + ".status.json"
            if os.path.exists(default_sock) and os.path.exists(st):
                try:
                    info = json.load(open(st))
                    os.kill(int(info["pid"]), 0)
                    return default_sock
                except Exception:                       # noqa: BLE001
                    return ""
            return ""
        if hc.lower() in ("0", "off", "false", "no", ""):
            return ""
        return hc

    def _rank_env(self, rank: int, card: str, master: str) -> dict:
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(card)
        env["MASTER_ADDR"], env["MASTER_PORT"] = master.split(":")
        env["ASCEND_PROCESS_LOG_PATH"] = os.path.join(self.logdir, "ascend_log")
        env.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "3")
        env["DS41_BENCH_LOGDIR"] = self.logdir
        env["TMPDIR"] = os.path.join(self.job_dir, "tmp")
        env["ASCEND_WORK_PATH"] = os.path.join(self.logdir, "_ascend_work")
        # ★ 2026-10-19 实测：这个开关影响 host-NZ decode 的**稳定性**。
        #   ep8（NZ 在 pinned host，每层 H2D 的临时张量立刻释放）在**不设**它时，
        #   第 1 个 decode forward 就偶发 topk gate NaN（两条 A/B：不设 → 2/2 崩；
        #   设 4096 → 连续通过）。max_split 让 >4 GiB 块不被切碎，临时权重块的
        #   复用节奏更保守，避开 vendor 异步算子读已释放块的窗口。
        #   tp8 的 W3 曾因它 + 10.05 GiB vendor 缓冲假 OOM；现在 chunk=1024（1.31 GiB）
        #   已避开，可继续用 4096。
        env.setdefault("PYTORCH_NPU_ALLOC_CONF", "max_split_size_mb:4096")
        # PERF_RULES 默认开（显式 env 可覆盖）
        env.setdefault("DS41M_DECODER_REPLAY", "1")
        # 入图（ACL Graph）已永久删除（2026-10-27 用户规则）：无 DS41M_GRAPH_TAIL 开关，
        # 也不再需要为"图池 ~10 GiB"登记降级原因。
        # MoE 块：官方默认 8192。常驻池 ≤1024 时用 1024 —— token 数不超过块长，
        # 每层仍是一整块调用（不是切块），**逐位数值已验证一致**（同 prompt ids
        # 与 8192 完全相同：...24463/32487/40555/301）。代价从 10.05 GiB 降到 1.26 GiB
        # 的 vendor 对称缓冲，正好让 tp8 的 AC+TP NZ 共存（8192 时 W3 会 10.05 GiB OOM）。
        _lmax = int(getattr(self.args, "serve_l_max", 1024) or 1024)
        env.setdefault("DS41M_MOE_CHUNK", "1024" if _lmax <= 1024 else "8192")
        env.setdefault("DS41M_MOEMODE", "decode")
        env.setdefault("DS41M_AC_CACHE_N", "40")
        env.setdefault("DS41M_AC_RELEASE_RAW", "1")
        # AC/NZ 已常驻，逐 rank pin 专家权重只会白占 host；需要时显式覆盖为 1
        env.setdefault("DS41M_WEIGHT_HOST_CACHE", "0")
        # ep8 双布局：decode NZ 放 pinned host，每层 H2D。
        # ★ 实测（2026-10-19）：NZ 常驻设备时 W2 后 allocated 80.4 GiB，而 W3/prefill 还要
        #   ~10.05 GiB（ascendc 对称缓冲按 chunk=8192 常驻）→ 反复 OOM-retry，W3 128 从
        #   0.16 s 变 13.9 s。放 host 后 W2 峰值 54.8 GiB、prefill 正常；代价 decode
        #   ~2.2 s/token（单跑 device-NZ 约 0.69 s/token）。
        env.setdefault("DS41M_DECODE_HOST_NZ", "1")
        # ★ 常驻模式默认 engram 走主机取表：厂商设备侧窗口 12.9 GiB/rank 是 AC+decode NZ
        #   之外最大的一块；主机路径每层多 ~33 ms（两层 ~66 ms），仍远低于 3 s 目标。
        #   显存宽裕时可设 DS41M_ENGRAM_HOST=0 强制厂商路径。
        env.setdefault("DS41M_ENGRAM_HOST", "1")
        env["DS41M_SERVER_WARMUP"] = "1"
        env.setdefault("DS41M_CMPSTATE_RESET", "1")
        hc = self._resolve_host_cache()
        if hc:
            env["DS41M_MOE_HOST_CACHE"] = hc
        else:
            env.pop("DS41M_MOE_HOST_CACHE", None)
        if self.mode == "ep8":
            env.pop("DS41M_TP_DECODE_ONLY", None)
            env.pop("DS41M_PD_MERGE", None)
            env.pop("DS41M_DECODE_LAYOUT", None)
        elif self.mode == "ep8-fidelity":
            # EP 同布局：P→D 在设备上建 EP decode NZ；D→P 从 host cache 回 raw。
            # 不走 all2all，也不需要 TP decode 布局。
            if not hc:
                raise SystemExit(
                    "--mode ep8-fidelity 需要 host cache（D→P 从内存回载 EP raw）；"
                    "先启动 tools/moe_host_cache.py，或改用 --mode ep8")
            env["DS41M_PD_MERGE"] = "1"
            env["DS41M_DECODE_LAYOUT"] = "ep"
            env["DS41M_DECODE_HOST_NZ"] = "0"
            env.pop("DS41M_TP_DECODE_ONLY", None)
            env.pop("DS41M_PD_TP_XFER", None)
        elif self.mode.startswith("tp8"):
            env["DS41M_TP_DECODE_ONLY"] = "1"
            env["DS41M_PD_MERGE"] = "1"
            env["DS41M_DECODE_LAYOUT"] = "tp"
        os.makedirs(env["TMPDIR"], exist_ok=True)
        os.makedirs(env["ASCEND_WORK_PATH"], exist_ok=True)
        os.makedirs(env["ASCEND_PROCESS_LOG_PATH"], exist_ok=True)
        return env

    def _rank_cmd(self, rank: int, master: str, ctl: str) -> list[str]:
        if getattr(self.args, "fixture", False):
            return [sys.executable, "-u", os.path.join(HERE, "bench_fake_rank.py"),
                    "--rank", str(rank), "--world", str(self.args.world),
                    "--control", ctl, "--serve-l-max", str(self.args.serve_l_max)]
        cmd = [
            sys.executable, "-u", os.path.join(BASE, "runner.py"),
            self.args.config,
            "--rank", str(rank), "--world", str(self.args.world),
            "--device", "npu:0", "--master", master,
            "--serve", "--control", ctl,
            "--serve-l-max", str(self.args.serve_l_max),
            "--serve-mode", self.args.mode,
            "--progress", "off",
        ]
        if self.args.passthru:
            cmd += shlex.split(self.args.passthru)
        return cmd

    def _check_perf(self, env: dict) -> None:
        checker = "/mnt/xds/sfs/l00972766/tools/check_perf_defaults.py"
        if not os.path.exists(checker):
            checker = os.path.join(os.path.dirname(BASE), "tools/check_perf_defaults.py")
        if not os.path.exists(checker):
            return
        r = subprocess.run([sys.executable, checker, "--runtime"],
                           env=env, capture_output=True, text=True)
        if r.returncode == 0:
            return
        msg = (r.stdout or "") + (r.stderr or "")
        if env.get("DS41_ALLOW_SUBOPTIMAL", "").strip():
            print(f"⚠️ PERF_RULES 检查未通过（已用 DS41_ALLOW_SUBOPTIMAL 放行）：\n{msg[-1200:]}")
            return
        raise SystemExit(
            "✗ PERF_RULES 检查未通过，拒绝启动。\n"
            f"{msg[-2000:]}\n"
            "A/B 降级请设 DS41_ALLOW_SUBOPTIMAL=\"原因\"。")

    def spawn(self):
        cards = [c.strip() for c in self.args.cards.split(",") if c.strip()]
        if len(cards) != self.args.world:
            raise SystemExit(f"卡数({len(cards)}) 必须等于 world({self.args.world})")
        master = f"127.0.0.1:{_free_port()}"
        env0 = self._rank_env(0, cards[0], master)
        if not getattr(self.args, "fixture", False):
            self._check_perf(env0)
        self.all_fh = open(os.path.join(self.logdir, "all.log"), "w")
        print(f"== 启动 {self.args.world} 个常驻 rank（卡 {','.join(cards)}，master {master}）")
        print(f"   日志：{self.logdir}/rank*.log  +  {self.logdir}/all.log")
        print(f"   mode={self.mode} l_max={self.args.serve_l_max} "
              f"host_cache={env0.get('DS41M_MOE_HOST_CACHE', 'off')}")
        for r in range(self.args.world):
            env = self._rank_env(r, cards[r], master)
            ctl = os.path.join(self.job_dir, f"rank{r}.ctl")
            rank_log = os.path.join(self.logdir, f"rank{r}.log")
            fh = open(rank_log, "w")
            self.rank_fhs[r] = fh
            cmd = self._rank_cmd(r, master, ctl)
            p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1,
                                 errors="replace")
            self.procs.append({"r": r, "p": p, "log": rank_log, "fh": fh})
            threading.Thread(target=self._pump, name=f"pump{r}", daemon=True,
                             args=(r, p.stdout, fh)).start()
        t0 = time.time()
        last_boot = 0.0
        while len(self.ready) < self.args.world:
            if self.ready_evt.wait(0.5):
                self.ready_evt.clear()
            if any(pr["p"].poll() is not None for pr in self.procs):
                break
            if time.time() - t0 > self.args.boot_timeout:
                print(f"✗ 等待 rank ready 超时（{self.ready}）")
                self.kill_all()
                raise SystemExit(1)
            if time.time() - last_boot >= 15.0:
                last_boot = time.time()
                print(f"  [boot {len(self.ready)}/{self.args.world}] "
                      f"已等 {int(time.time() - t0)}s …", flush=True)
        if len(self.ready) < self.args.world:
            for pr in self.procs:
                if pr["p"].poll() is not None:
                    print(f"✗ rank{pr['r']} 在 ready 前退出（rc={pr['p'].returncode}）；"
                          f"日志尾部：")
                    try:
                        with open(pr["log"], errors="ignore") as fh:
                            for ln in fh.readlines()[-20:]:
                                print("   " + ln.rstrip())
                    except OSError:
                        pass
            self.kill_all()
            raise SystemExit("✗ 有 rank 在 ready 前退出；见日志")
        print(f"== 8 rank ready（{time.time() - t0:.1f}s）")
        # 连接 ctl
        for r in range(self.args.world):
            self.ranks.append(RankConn(r, os.path.join(self.job_dir, f"rank{r}.ctl"),
                                       timeout=60.0))
        print("== rank 控制通道已连接")

    def _pump(self, r, stream, fh):
        try:
            for raw in stream:
                line = raw.rstrip("\n")
                with self.log_lock:
                    fh.write(line + "\n")
                    fh.flush()
                    if self.all_fh:
                        self.all_fh.write(f"[r{r}] " + line + "\n")
                        self.all_fh.flush()
                if "serve ready" in line and "[bench] rank" in line:
                    self.ready.add(r)
                    self.ready_evt.set()
                if r == 0:
                    prog = P.parse_line(line)
                    with self.client_lock:
                        cli = self.client
                    if cli is not None and prog:
                        with self.send_lock:
                            try:
                                P.send_json(cli, {"type": "line", "line": line})
                            except Exception:           # noqa: BLE001
                                with self.client_lock:
                                    self.client = None
                    elif cli is None and prog and self.bar_active:
                        with self.bar_lock:
                            self.bar.feed(line)
        except Exception:                               # noqa: BLE001
            pass

    # ───────────── dispatch ─────────────
    def _dispatch(self, op: str, payload: dict, timeout: float | None = None):
        if not self.ranks:
            raise RuntimeError("rank 未就绪")
        cmd = {"op": op, "cmd_id": f"{op}-{time.time_ns()}", **payload}
        cid = cmd["cmd_id"]
        for rc in self.ranks:
            rc.send(cmd)
        timeout = float(timeout or self.args.timeout)
        # ACK：所有 rank 收到命令才放行（收不到 → 集合通信必然缺人）
        pending = set(self.ranks)
        deadline = time.time() + 15.0
        while pending and time.time() < deadline:
            for rc in list(pending):
                msg = rc.try_get(cid, {"ack", "error"})
                if msg is None:
                    continue
                pending.discard(rc)
                if msg.get("type") == "error":
                    return {"ok": False, "error": msg.get("error", "unknown"),
                            "recoverable": msg.get("recoverable", False),
                            "rank": rc.rank}
            if pending:
                time.sleep(0.02)
        if pending:
            names = sorted(rc.rank for rc in pending)
            self.kill_all()
            raise RuntimeError(f"rank {names} 未 ACK（集合通信失步风险）→ 整组已杀")
        # 结果：轮询所有 rank；一旦见到错误，给其余 rank 10s 收尾，避免长等
        pending = set(self.ranks)
        results, errors = {}, {}
        deadline = time.time() + timeout
        grace = None
        while pending and time.time() < deadline:
            for rc in list(pending):
                msg = rc.try_get(cid, {"result", "error"})
                if msg is None:
                    continue
                pending.discard(rc)
                if msg.get("type") == "error":
                    errors[rc.rank] = msg
                else:
                    results[rc.rank] = msg.get("result") or {}
            if errors and grace is None:
                grace = time.time() + 10.0
            if grace is not None and time.time() > grace and pending:
                break
            if pending:
                time.sleep(0.02)
        if pending:
            names = sorted(rc.rank for rc in pending)
            self.kill_all()
            raise RuntimeError(f"rank {names} 等 {op} 结果超时 → 整组已杀")
        if errors:
            recoverable = (len(errors) == len(self.ranks)
                           and all(e.get("recoverable") for e in errors.values()))
            first = next(iter(errors.values()))
            if not recoverable:
                self.kill_all()
            return {"ok": False, "error": first.get("error", "unknown"),
                    "recoverable": recoverable, "rank": next(iter(errors))}
        return {"ok": True, "results": results, "cmd_id": cid}

    @staticmethod
    def _agg_run(dispatch_res, logdir):
        if not dispatch_res.get("ok"):
            return dispatch_res
        results = dispatch_res["results"]
        r0 = results.get(0) or {}
        runs = r0.get("runs") or []
        ranks = {r: {"prefill_s": v.get("prefill_s"), "p2d_s": v.get("p2d_s"),
                     "d2p_s": v.get("d2p_s"),
                     "decode_ms_per_token": v.get("decode_ms_per_token"),
                     "max_allocated_gib": v.get("max_allocated_gib")}
                 for r, v in results.items()}
        if runs:
            # 每个 rank 的每次 run 做 max（端到端取最慢 rank）
            n = min(len(v.get("runs") or []) for v in results.values())
            merged = []
            for i in range(n):
                merged.append({
                    "prefill_s": max(float(v["runs"][i].get("prefill_s") or 0)
                                     for v in results.values()),
                    "p2d_s": max(float(v["runs"][i].get("p2d_s") or 0)
                                 for v in results.values()),
                    "d2p_s": max(float(v["runs"][i].get("d2p_s") or 0)
                                 for v in results.values()),
                    "decode_ms_per_token": max(float(v["runs"][i].get(
                        "decode_ms_per_token") or 0) for v in results.values()),
                    "first_token": r0["runs"][i].get("first_token"),
                    "text": r0["runs"][i].get("text", ""),
                })
            steady = merged[1:] if len(merged) > 1 else merged
            def med(key):
                xs = sorted(float(x.get(key) or 0) for x in steady)
                if not xs:
                    return 0.0
                k = len(xs) // 2
                return xs[k] if len(xs) % 2 else (xs[k - 1] + xs[k]) / 2
            return {
                "ok": True, "mode": r0.get("mode"), "runs": merged,
                "summary": {k: med(k) for k in
                            ("prefill_s", "p2d_s", "d2p_s", "decode_ms_per_token")},
                "n_new": r0.get("n_new"), "ids": r0.get("ids"),
                "text": r0.get("text"), "ranks": ranks, "logdir": logdir,
            }
        agg = dict(r0)
        agg["ranks"] = ranks
        agg["logdir"] = logdir
        return agg

    @staticmethod
    def _agg_warmup(dispatch_res, logdir):
        if not dispatch_res.get("ok"):
            return dispatch_res
        results = dispatch_res["results"]
        r0 = results.get(0) or {}
        out = {
            "ok": True, "mode": r0.get("mode"),
            "w1_s": max(float(v.get("w1_s") or 0) for v in results.values()),
            "w2_s": max(float(v.get("w2_s") or 0) for v in results.values()),
            "p2d_s": max(float(v.get("p2d_s") or 0) for v in results.values()),
            "d2p_s": max(float(v.get("d2p_s") or 0) for v in results.values()),
            # ★ 用 is not None 而不是 `or`：0 层是合法值（ep8 全量 decode 没有 NZ），
            #   以前 `0 or -1` 会把它显示成 -1。
            "ac_layers": max(int(v["ac_layers"]) if v.get("ac_layers") is not None else -1
                             for v in results.values()),
            "nz_layers": max(int(v["nz_layers"]) if v.get("nz_layers") is not None else -1
                             for v in results.values()),
            "dropped_raw": max(int(v.get("dropped_raw") or 0) for v in results.values()),
            "max_allocated_gib": max(float(v.get("max_allocated_gib") or 0)
                                     for v in results.values()),
            "buckets": r0.get("buckets") or [],
            "logdir": logdir,
        }
        return out

    # ───────────── client ─────────────
    def _welcome(self, conn):
        return {"type": "welcome", "job": self.job, "logdir": self.logdir,
                "mode": self.mode, "server_pid": os.getpid(),
                "started": self.started}

    def serve_client(self, conn):
        jl = P.JsonLines(conn)

        def _send(obj):
            # rank0 的进度转发线程也会往同一 socket 写；必须串行化 JSONL
            with self.send_lock:
                try:
                    P.send_json(conn, obj)
                except Exception:                       # noqa: BLE001
                    pass

        try:
            _send(self._welcome(conn))
            cmd = jl.recv()
            if not cmd:
                return
            op = cmd.get("op")
            if op == "status":
                if not self.busy.acquire(blocking=False):
                    _send({"type": "error", "error": "服务忙（有命令在跑）",
                                       "recoverable": True})
                    return
                try:
                    res = self._dispatch("status", {}, timeout=30)
                    if res.get("ok"):
                        st = (res["results"].get(0)
                              or next(iter(res["results"].values())))
                        st["server_pid"] = os.getpid()
                        st["job"] = self.job
                        st["logdir"] = self.logdir
                        _send({"type": "status", "status": st})
                    else:
                        _send({"type": "error", "error": res.get("error")})
                finally:
                    self.busy.release()
                return
            if op == "stop":
                _send({"type": "bye", "cmd_id": cmd.get("cmd_id")})
                self.shutdown()
                return
            if op == "cancel":
                for rc in self.ranks:
                    try:
                        rc.send({"op": "cancel", "cmd_id": cmd.get("cmd_id")})
                    except Exception:                   # noqa: BLE001
                        pass
                _send({"type": "ack", "op": "cancel"})
                return
            if not self.busy.acquire(blocking=False):
                _send({"type": "error", "error": "服务忙（有命令在跑）",
                                   "recoverable": True})
                return
            with self.client_lock:
                self.client = conn
            try:
                if op == "warmup":
                    _send({"type": "stage", "stage": "warmup"})
                    res = self._dispatch("warmup", {
                        "lengths": cmd.get("lengths") or [128],
                        "decode": int(cmd.get("decode", 1) or 1),
                    }, timeout=self.args.timeout)
                    if not res.get("ok"):
                        _send({"type": "error", "error": res.get("error", "unknown"),
                               "recoverable": res.get("recoverable", False)})
                    else:
                        agg = self._agg_warmup(res, self.logdir)
                        _send({"type": "result", "result": agg})
                elif op == "run":
                    _send({"type": "stage", "stage": "run"})
                    res = self._dispatch("run", {
                        "prompt": cmd.get("prompt", ""),
                        "decode": int(cmd.get("decode", 0) or 0),
                        "repeat": int(cmd.get("repeat", 1) or 1),
                        "mode": cmd.get("mode", ""),
                        "chat": bool(cmd.get("chat")),
                        "thinking": cmd.get("thinking", "chat"),
                    }, timeout=self.args.timeout)
                    if not res.get("ok"):
                        _send({"type": "error", "error": res.get("error", "unknown"),
                               "recoverable": res.get("recoverable", False)})
                    else:
                        agg = self._agg_run(res, self.logdir)
                        _send({"type": "result", "result": agg})
                elif op == "hostcache":
                    res = self._dispatch("hostcache", {
                        "action": cmd.get("action", "apply"),
                        "path": cmd.get("path", ""),
                    }, timeout=60)
                    if res.get("ok"):
                        results = res["results"]
                        _send({"type": "result", "result": {
                            "ok": True, "hostcache": results.get(0, {}).get("hostcache"),
                            "ranks": {r: v.get("hostcache") for r, v in results.items()},
                            "logdir": self.logdir}})
                    else:
                        _send({"type": "error", "error": res.get("error")})
                else:
                    _send({"type": "error",
                                       "error": f"未知 op {op!r}", "recoverable": True})
            except Exception as exc:                    # noqa: BLE001
                traceback.print_exc()
                _send({"type": "error",
                                   "error": f"{type(exc).__name__}: {exc}"})
            finally:
                with self.client_lock:
                    self.client = None
                self.busy.release()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self.client_lock:
                if self.client is conn:
                    self.client = None
            jl.close()
            try:
                conn.close()
            except Exception:                           # noqa: BLE001
                pass

    # ───────────── lifecycle ─────────────
    def _watchdog(self):
        """任一 rank 在 idle/运行中退出 → 整组 kill（HCCL 不能单 rank 重启）。"""
        while not self.shutdown_flag:
            for pr in self.procs:
                rc = pr["p"].poll()
                if rc is None:
                    continue
                if self.shutdown_flag:
                    return
                print(f"\n✗ rank{pr['r']} 意外退出（rc={rc}）→ 整组停止；日志尾部：")
                try:
                    with open(pr["log"], errors="ignore") as fh:
                        for ln in fh.readlines()[-20:]:
                            print("   " + ln.rstrip())
                except OSError:
                    pass
                self.kill_all()
                self.shutdown_flag = True
                try:
                    self.write_state("fatal")
                except Exception:                       # noqa: BLE001
                    pass
                try:
                    if getattr(self, "server_sock", None) is not None:
                        self.server_sock.close()
                except Exception:                       # noqa: BLE001
                    pass
                return
            time.sleep(2.0)

    def write_state(self, status="running"):
        st = {"job": self.job, "pid": os.getpid(), "logdir": self.logdir,
              "socket": self.server_sock_path, "mode": self.mode,
              "started": self.started, "status": status,
              "ranks": [{"rank": pr["r"], "pid": pr["p"].pid} for pr in self.procs]}
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(st, fh, indent=2)
        os.replace(tmp, self.state_path)

    def kill_all(self):
        for pr in self.procs:
            try:
                if pr["p"].poll() is None:
                    pr["p"].terminate()
            except Exception:                           # noqa: BLE001
                pass
        t0 = time.time()
        while time.time() - t0 < 10:
            if all(pr["p"].poll() is not None for pr in self.procs):
                break
            time.sleep(0.1)
        for pr in self.procs:
            try:
                if pr["p"].poll() is None:
                    pr["p"].kill()
            except Exception:                           # noqa: BLE001
                pass

    def shutdown(self):
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self.shutdown_flag = True
        try:
            if getattr(self, "server_sock", None) is not None:
                self.server_sock.close()
        except Exception:                               # noqa: BLE001
            pass
        print("\n== 停止常驻服务 …")
        for rc in self.ranks:
            try:
                rc.send({"op": "stop", "cmd_id": f"stop-{time.time_ns()}"})
            except Exception:                           # noqa: BLE001
                pass
        t0 = time.time()
        while time.time() - t0 < 10:
            if all(pr["p"].poll() is not None for pr in self.procs):
                break
            time.sleep(0.1)
        self.kill_all()
        for rc in self.ranks:
            rc.close()
        for pr in self.procs:
            try:
                pr["fh"].close()
            except Exception:                           # noqa: BLE001
                pass
        if self.all_fh:
            try:
                self.all_fh.close()
            except Exception:                           # noqa: BLE001
                pass
        for p in [self.server_sock_path] + [
                os.path.join(self.job_dir, f"rank{r}.ctl")
                for r in range(self.args.world)]:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        try:
            self.write_state("stopped")
        except Exception:                               # noqa: BLE001
            pass
        print("== 已停止（host cache 未动；要释放请 python3.11 tools/moe_host_cache.py stop）")

    def run(self):
        signal.signal(signal.SIGINT, lambda *_a: (self.shutdown(), sys.exit(130)))
        signal.signal(signal.SIGTERM, lambda *_a: (self.shutdown(), sys.exit(143)))
        try:
            self.spawn()
            threading.Thread(target=self._watchdog, name="watchdog",
                             daemon=True).start()
            if self.args.warmup_on_start:
                lengths = [int(x) for x in self.args.warmup_on_start.split(",")
                           if x.strip()]
                print(f"== 启动预热 {lengths} …")
                self.bar_active = True
                try:
                    res = self._dispatch("warmup", {"lengths": lengths, "decode": 1},
                                         timeout=self.args.timeout)
                    if not res.get("ok"):
                        print(f"✗ 启动预热失败：{res.get('error')}")
                    else:
                        agg = self._agg_warmup(res, self.logdir)
                        print(f"   W1 {agg.get('w1_s', 0):.2f}s  "
                              f"W2 {agg.get('w2_s', 0):.2f}s  "
                              f"P→D {agg.get('p2d_s', 0):.2f}s  "
                              f"D→P {agg.get('d2p_s', 0):.2f}s  "
                              f"AC={agg.get('ac_layers')} NZ={agg.get('nz_layers')}  "
                              f"max_alloc={agg.get('max_allocated_gib')} GiB")
                finally:
                    with self.bar_lock:
                        self.bar.finish(force=True)
                    self.bar_active = False
            try:
                os.unlink(self.server_sock_path)
            except FileNotFoundError:
                pass
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(self.server_sock_path)
            srv.listen(4)
            srv.settimeout(1.0)
            os.chmod(self.server_sock_path, 0o600)
            self.server_sock = srv
            self.write_state()
            print("\n== 常驻服务就绪 ==")
            print(f"   logdir : {self.logdir}")
            print(f"   socket : {self.server_sock_path}")
            print(f"   日志   : tail -f {self.logdir}/all.log")
            print(f"   等待   : python3.11 tools/bench_cli.py warmup/run/status/stop")
            while not self.shutdown_flag:
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                threading.Thread(target=self.serve_client, args=(conn,),
                                 name="client", daemon=True).start()
            srv.close()
        finally:
            self.shutdown()


# ─────────────────────────── CLI ───────────────────────────
def _cmd_start(args):
    # rank 崩溃时不要产生 core（几百 MB~GB，根盘常年贴满；CANN 的 extra-info 已走 logdir）
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_MEMLOCK,
                           (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except Exception:                                   # noqa: BLE001
        pass
    if args.dry_run:
        sup = Supervisor(args)
        cards = [c.strip() for c in args.cards.split(",") if c.strip()]
        print(f"job={sup.job}  logdir={sup.logdir}")
        print("rank0 命令：")
        print("  " + " ".join(shlex.quote(x) for x in
                             sup._rank_cmd(0, "127.0.0.1:<free>",
                                           os.path.join(sup.job_dir, "rank0.ctl"))))
        print(f"env: DS41M_MOE_HOST_CACHE={sup._resolve_host_cache() or '(off)'}  "
              f"mode={sup.mode}  l_max={args.serve_l_max}")
        return 0
    Supervisor(args).run()
    return 0


def _cmd_stop(args):
    path = _sock_of(args)
    if not path or not os.path.exists(path):
        job = getattr(args, "job", "") or _latest_job()
        st = os.path.join(P.job_root(), job, "state.json") if job else ""
        if st and os.path.exists(st):
            info = json.load(open(st))
            try:
                os.kill(int(info["pid"]), signal.SIGTERM)
                print(f"已向 server pid={info['pid']} 发 SIGTERM")
            except ProcessLookupError:
                print("服务已不在")
            return 0
        raise SystemExit("找不到 server.sock / state.json；服务未运行？")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(30)
    s.connect(path)
    jl = P.JsonLines(s)
    jl.recv()                                          # welcome
    P.send_json(s, {"op": "stop", "cmd_id": f"stop-{time.time_ns()}"})
    while True:
        msg = jl.recv()
        if msg is None:
            break
        if msg.get("type") == "bye":
            print("已请求停止")
            break
    jl.close()
    return 0


def _cmd_status(args):
    path = _sock_of(args)
    if not path or not os.path.exists(path):
        raise SystemExit("找不到 server.sock；服务未运行？")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(30)
    s.connect(path)
    jl = P.JsonLines(s)
    jl.recv()
    P.send_json(s, {"op": "status", "cmd_id": f"status-{time.time_ns()}"})
    while True:
        msg = jl.recv()
        if msg is None:
            break
        if msg.get("type") == "status":
            print(json.dumps(msg["status"], ensure_ascii=False, indent=2))
            break
        if msg.get("type") == "error":
            raise SystemExit(msg.get("error"))
    jl.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description="常驻 benchmark supervisor（无 TUI）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start", help="启动 8 rank 常驻服务（前台阻塞）")
    s.add_argument("--config", default="configs/mini_real.yaml")
    s.add_argument("--world", type=int, default=8)
    s.add_argument("--cards", default="0,1,2,3,4,5,6,7")
    s.add_argument("--logdir", default="")
    s.add_argument("--job", default="", help="作业名（/tmp/ds41_bench/<job>）")
    s.add_argument("--mode", default="auto",
                   choices=["auto", "ep8", "ep8-fidelity", "tp8-fidelity", "tp8-dual"])
    s.add_argument("--host-cache", default="auto",
                   help="auto|off|<socket>；只用于 rank 的 env，不启停 daemon")
    s.add_argument("--serve-l-max", type=int, default=1024,
                   help="常驻池容量（prompt+decode 最大 token 数；默认 1024 以留显存给入图）")
    s.add_argument("--warmup-on-start", default="",
                   help="启动后自动预热（如 128,256,512；空=手动）")
    s.add_argument("--timeout", type=float, default=3600.0)
    s.add_argument("--boot-timeout", type=float, default=1200.0)
    s.add_argument("--passthru", default="", help="透传给 runner 的额外参数")
    s.add_argument("--fixture", action="store_true",
                   help="自测：用 bench_fake_rank.py 代替真 rank（不占 NPU）")
    s.add_argument("--bar", action=argparse.BooleanOptionalAction, default=True,
                   help="服务端画启动预热进度条（run 的进度条在 client 终端）")
    s.add_argument("--dry-run", action="store_true")
    for name in ("stop", "status"):
        p = sub.add_parser(name)
        p.add_argument("--socket", default="")
        p.add_argument("--job", default="")
    args = ap.parse_args()
    if args.cmd == "start":
        if not args.logdir:
            args.logdir = os.path.join("/mnt/xds/sfs/l00972766/tmp",
                                       "bench_" + time.strftime("%m%d_%H%M%S"))
        return _cmd_start(args)
    if args.cmd == "stop":
        return _cmd_stop(args)
    return _cmd_status(args)


if __name__ == "__main__":
    sys.exit(main())

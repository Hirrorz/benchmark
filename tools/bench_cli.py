#!/usr/bin/env python3.11
# coding=utf-8
"""bench_cli.py — 常驻 benchmark 的命令行客户端（无 TUI，只有简单进度条）。

用法（命令由 tools/bench_command_builder.html 生成）：
  python3.11 tools/bench_cli.py warmup --lengths 128,256,512 --decode 1
  python3.11 tools/bench_cli.py run --prompt "你好" --decode 32 --repeat 3
  python3.11 tools/bench_cli.py status
  python3.11 tools/bench_cli.py stop
  python3.11 tools/bench_cli.py cancel
  python3.11 tools/bench_cli.py hostcache-apply
  python3.11 tools/bench_cli.py hostcache-detach

socket 默认取 $DS41_BENCH_SOCK，否则 /tmp/ds41_bench/<job>/server.sock；用 --job/--socket 覆盖。
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_protocol as P


def _default_sock(args) -> str:
    if args.socket:
        return args.socket
    env = os.environ.get("DS41_BENCH_SOCK", "")
    if env:
        return env
    if args.job:
        return P.default_server_sock(args.job)
    # 最近一次作业：/tmp/ds41_bench/<job>/server.sock
    root = P.job_root()
    try:
        cands = [os.path.join(root, d, "server.sock") for d in os.listdir(root)]
        cands = [c for c in cands if os.path.exists(c)]
        if cands:
            return max(cands, key=os.path.getmtime)
    except OSError:
        pass
    return P.default_server_sock("default")


def _connect(args, timeout: float = 120.0) -> socket.socket:
    path = _default_sock(args)
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        if not os.path.exists(path):
            last = f"控制 socket 不存在：{path}"
            time.sleep(0.2)
            continue
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(max(1.0, args.timeout))
            s.connect(path)
            return s
        except OSError as exc:                        # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(0.2)
    raise SystemExit(f"连接常驻服务失败（{last}）；先启动 tools/bench_server.py start")


def _fmt_sec(v) -> str:
    if v is None:
        return "-"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if v >= 3600:
        return f"{v / 3600:.2f}h"
    if v >= 60:
        return f"{v / 60:.2f}m"
    if v >= 1:
        return f"{v:.3f}s"
    return f"{v * 1000:.1f}ms"


def _print_result(res: dict) -> None:
    if not res:
        return
    if "w1_s" in res:
        print("\n== 预热完成 ==")
        print(f"  W1 AC      {_fmt_sec(res.get('w1_s'))}  "
              f"({res.get('ac_layers')} 层)")
        print(f"  W2 decode  {_fmt_sec(res.get('w2_s'))}  "
              f"({res.get('nz_layers')} 层 · 释放 raw {res.get('dropped_raw')} 项)")
        if res.get("p2d_s") or res.get("d2p_s"):
            print(f"  转换        P→D {_fmt_sec(res.get('p2d_s'))} · "
                  f"D→P {_fmt_sec(res.get('d2p_s'))}")
        for b in res.get("buckets") or []:
            print(f"  W3 L={str(b.get('L')):<7} {_fmt_sec(b.get('prefill_s'))}")
        if res.get("max_allocated_gib"):
            print(f"  显存峰值   {res.get('max_allocated_gib')} GiB")
        logdir = res.get("logdir")
        if logdir:
            print(f"\n日志：{logdir}/rank*.log  +  {logdir}/all.log")
            print(f"tail -f {logdir}/all.log")
        return
    if res.get("runs"):
        runs = res["runs"]
        print("\n== 结果（%d 次）==" % len(runs))
        print("  #   prefill      P→D       decode/tok   D→P       首 token  文本")
        for i, r in enumerate(runs, 1):
            print(f"  {i:<2}  {_fmt_sec(r.get('prefill_s')):<10} "
                  f"{_fmt_sec(r.get('p2d_s')):<9} "
                  f"{_fmt_sec((r.get('decode_ms_per_token') or 0) / 1000):<12} "
                  f"{_fmt_sec(r.get('d2p_s')):<9} "
                  f"{str(r.get('first_token', '-')):<9} {str(r.get('text', ''))[:40]}")
        # 表格里的文本列只放 40 字预览；完整文本另起一行（EOS 可能是正常句末，
        # 之前被截断成"…它最有"让人误以为没生成完）。
        for i, r in enumerate(runs, 1):
            _txt = str(r.get("text") or "")
            if _txt:
                print(f"  文本 #{i}（{len(_txt)} 字）：{_txt}")
        sm = res.get("summary") or {}
        if sm:
            print(f"  稳态中位：prefill {_fmt_sec(sm.get('prefill_s'))} · "
                  f"P→D {_fmt_sec(sm.get('p2d_s'))} · "
                  f"decode {_fmt_sec((sm.get('decode_ms_per_token') or 0) / 1000)}/tok · "
                  f"D→P {_fmt_sec(sm.get('d2p_s'))}")
        if len(runs) > 1:
            _pos = [float(r.get("prefill_s") or 0) for r in runs[1:]]
            _pos = [x for x in _pos if x > 0]
            _p0 = float(runs[0].get("prefill_s") or 0)
            if _pos and _p0 > 1.5 * min(_pos):
                print(f"  [first-touch] 首次形状预热 {_fmt_sec(_p0)}；"
                      f"后续同形状稳态 {_fmt_sec(sm.get('prefill_s'))}")
        if res.get("n_new"):
            print(f"  生成长度：{res['n_new']} token")
        if res.get("ids"):
            print(f"  ids：{res['ids'][:24]}{'…' if len(res['ids']) > 24 else ''}")
    else:
        print("\n== 结果 ==")
        for k in ("prefill_s", "p2d_s", "decode_ms_per_token", "decode_total_s",
                  "d2p_s", "first_token", "text", "ids"):
            if k in res:
                print(f"  {k:<20} {res[k]}")
    logdir = res.get("logdir")
    if logdir:
        print(f"\n日志：{logdir}/rank*.log  +  {logdir}/all.log")
        print(f"tail -f {logdir}/all.log")


def _stream(sock: socket.socket, cmd: dict, args) -> dict | None:
    P.send_json(sock, cmd)
    jl = P.JsonLines(sock)
    bar = P.StageBar(enabled=not args.no_bar, width=args.bar_width,
                     every=args.bar_every,
                     color=not args.no_color)
    result, error, status = None, None, None
    try:
        while True:
            ev = jl.recv()
            if ev is None:
                break
            t = ev.get("type")
            if t == "welcome":
                if args.verbose:
                    print(f"已连接：job={ev.get('job')} logdir={ev.get('logdir')} "
                          f"mode={ev.get('mode')} pid={ev.get('server_pid')}")
            elif t == "line":
                bar.feed(ev.get("line", ""))
            elif t == "result":
                result = ev.get("result")
                bar.close()
                _print_result(result or {})
            elif t == "status":
                status = ev.get("status")
                bar.close()
                print(json.dumps(status, ensure_ascii=False, indent=2))
            elif t == "error":
                error = ev.get("error") or "unknown error"
                bar.close()
                print(f"✗ {error}", file=sys.stderr)
            elif t == "bye":
                break
            elif t == "raw" and args.verbose:
                print(ev.get("line", ""))
    except KeyboardInterrupt:
        bar.close()
        print("\n[Ctrl-C] 发送 cancel（常驻服务不会退出）", file=sys.stderr)
        try:
            P.send_json(sock, {"op": "cancel", "cmd_id": cmd.get("cmd_id")})
        except Exception:                               # noqa: BLE001
            pass
        return None
    finally:
        bar.close()
        jl.close()
    if error:
        return None
    return result if result is not None else status


def _send_simple(args, obj: dict):
    sock = _connect(args)
    try:
        return _stream(sock, obj, args)
    finally:
        try:
            sock.close()
        except Exception:                           # noqa: BLE001
            pass


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--socket", default="", help="server.sock 路径")
    common.add_argument("--job", default="", help="作业名（/tmp/ds41_bench/<job>）")
    common.add_argument("--timeout", type=float, default=3600.0, help="等待结果的秒数")
    common.add_argument("--no-bar", action="store_true", help="关掉进度条")
    common.add_argument("--no-color", action="store_true")
    common.add_argument("--bar-width", type=int, default=14)
    common.add_argument("--bar-every", type=float, default=0.5)
    common.add_argument("--verbose", "-v", action="store_true")

    ap = argparse.ArgumentParser(description="常驻 benchmark 命令行客户端")
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("warmup", parents=[common],
                       help="手动预热 W1/W2/W3（显示进度条）")
    w.add_argument("--lengths", default="16,32,64,128,256,512,1024",
                   help="形状阶梯，逗号分隔（含 16/32：短 prompt 的首触/拆分形状）")
    w.add_argument("--decode", type=int, default=1, help="预热 decode 步数")
    w.add_argument("--dry-run", action="store_true")

    r = sub.add_parser("run", parents=[common], help="跑一个任意 prompt")
    r.add_argument("--prompt", required=True)
    r.add_argument("--decode", type=int, default=32)
    r.add_argument("--repeat", type=int, default=1, help="重复次数（稳态取中位）")
    r.add_argument("--mode", default="", choices=["", "auto", "ep8", "ep8-fidelity",
                                                  "tp8-fidelity", "tp8-dual"])
    r.add_argument("--chat", action="store_true", help="用 chat 模板包装 prompt")
    r.add_argument("--thinking", default="chat", choices=["chat", "thinking"])
    r.add_argument("--max-new", type=int, default=32, help="decode 上限（同 --decode）")
    r.add_argument("--dry-run", action="store_true")

    sub.add_parser("status", parents=[common], help="查看服务状态")
    sub.add_parser("stop", parents=[common], help="停止常驻服务")
    sub.add_parser("cancel", parents=[common], help="取消当前请求（服务继续常驻）")
    sub.add_parser("hostcache-apply", parents=[common],
                   help="在 idle 安全点挂接 host cache")
    sub.add_parser("hostcache-detach", parents=[common],
                   help="解绑 host cache（释放 mmap）")
    args = ap.parse_args()

    if args.cmd == "warmup":
        lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
        obj = {"op": "warmup", "cmd_id": f"warmup-{time.time_ns()}",
               "lengths": lengths, "decode": int(args.decode)}
    elif args.cmd == "run":
        obj = {"op": "run", "cmd_id": f"run-{time.time_ns()}",
               "prompt": args.prompt, "decode": int(args.decode),
               "repeat": max(1, int(args.repeat)), "mode": args.mode,
               "chat": bool(args.chat), "thinking": args.thinking}
    elif args.cmd == "status":
        obj = {"op": "status", "cmd_id": f"status-{time.time_ns()}"}
    elif args.cmd == "stop":
        obj = {"op": "stop", "cmd_id": f"stop-{time.time_ns()}"}
    elif args.cmd == "cancel":
        obj = {"op": "cancel", "cmd_id": f"cancel-{time.time_ns()}"}
    elif args.cmd == "hostcache-apply":
        obj = {"op": "hostcache", "action": "apply", "cmd_id": f"hc-{time.time_ns()}"}
    elif args.cmd == "hostcache-detach":
        obj = {"op": "hostcache", "action": "detach", "cmd_id": f"hc-{time.time_ns()}"}
    else:
        raise SystemExit(2)

    if getattr(args, "dry_run", False):
        print(json.dumps(obj, ensure_ascii=False, indent=2))
        return 0

    res = _send_simple(args, obj)
    if args.cmd == "stop":
        print("已请求停止常驻服务")
    return 0 if res is not None or args.cmd in ("stop", "cancel") else 1


if __name__ == "__main__":
    sys.exit(main())

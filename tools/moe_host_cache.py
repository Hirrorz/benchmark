#!/usr/bin/env python3.11
# coding=utf-8



"""moe_host_cache.py — 把 MoE 专家权重常驻在**主机内存**的独立程序（手动开/关）。

用法（先 source env_pack.sh）：
  python3.11 tools/moe_host_cache.py start --config configs/mini_real.yaml
  python3.11 tools/moe_host_cache.py status
  python3.11 tools/moe_host_cache.py stop

  # 只缓存部分层/专家（联调用，加载快）：
  ... start --config configs/mini_real.yaml --layers 0-2 --experts 0-383

原理：
  · 用 `memfd_create` 建一块**匿名共享内存**（不受 /dev/shm 64MB 限制），把真实 MoE 专家
    权重（`ffn.experts.*` 的 weight+scale）按清单顺序写进去；
  · daemon 通过 Unix socket 把 memfd 的 fd（SCM_RIGHTS）发给 bench 进程；
  · bench 侧 `RealSource.get` 发现缺专家权重时，**优先**从这块 mmap 读（零拷贝视图 +
    按需 TP 切片/上卡），完全不碰共享盘；
  · 手动 `stop` 杀掉 daemon（客户端已 mmap 的页在客户端退出后释放）。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ds41_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    _q = _os.path.join(_ds41_root, _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)

import argparse
import array
import json
import mmap
import os
import signal
import socket
import struct
import sys
import time
import zlib

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SOCK = "/tmp/ds41_moe_host_cache.sock"


def _write_progress(path, **kw):
    """原子写进度 JSON（TUI 不做后，server/client/start 都读这一份）。"""
    if not path:
        return
    try:
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(kw, fh)
        os.replace(tmp, path)
    except OSError:
        pass


# ─────────────────────────── 权重清单 ───────────────────────────
def parse_range(s, lo, hi):
    if not s:
        return list(range(lo, hi))
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def expert_keys(cfg, layers, e0, e1):
    import weights_manifest as M
    E = int(cfg["model"]["n_routed_experts"])
    for li in layers:
        for suf, dt, shp in M.layer_manifest(li, E, e_range=(e0, e1), cfg=cfg):
            if "ffn.experts." in suf:
                yield f"layers.{li}.{suf}", dt, tuple(int(x) for x in shp)


def tensor_bytes(dt, shape):
    import weights as W
    n = 1
    for x in shape:
        n *= int(x)
    return n * int(W.ELEM[str(dt)])


# ─────────────────────────── daemon ───────────────────────────
def _load_into_memfd(cfg, layers, e0, e1, total_hint=0, progress_json=""):
    import torch
    from safetensors import safe_open
    import weights as W

    root = cfg["weights"]["path"]
    idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))
    wmap = idx["weight_map"]
    handles = {}

    def handle(f):
        if f not in handles:
            handles[f] = safe_open(os.path.join(root, f), framework="pt", device="cpu")
        return handles[f]

    keys = list(expert_keys(cfg, layers, e0, e1))
    total = sum(tensor_bytes(dt, shp) for _n, dt, shp in keys)
    try:
        with open("/proc/meminfo") as fh:
            avail = next(int(line.split()[1]) for line in fh
                         if line.startswith("MemAvailable:")) * 1024
        if total > max(0, avail - 32 * 2**30):
            raise SystemExit(
                f"缓存需要 {total / 2**30:.1f} GiB，但 MemAvailable 只有 "
                f"{avail / 2**30:.1f} GiB（预留 32 GiB）→ 拒绝启动；"
                f"可用 --layers/--experts 缩小范围")
        print(f"[moe-host-cache] MemAvailable {avail / 2**30:.1f} GiB，"
              f"本次缓存 {total / 2**30:.1f} GiB", flush=True)
    except OSError:
        pass
    print(f"[moe-host-cache] {len(keys)} 个张量，合计 {total / 2**30:.1f} GiB；"
          f"layers={layers[0]}..{layers[-1]} experts={e0}..{e1 - 1}", flush=True)
    _write_progress(progress_json, phase="loading", pid=os.getpid(),
                    started=time.time(), keys_done=0, keys_total=len(keys),
                    loaded=0, total=total, rate_mib_s=0, eta_s=0)

    fd = os.memfd_create("ds41_moe_host_cache", os.MFD_CLOEXEC)
    os.ftruncate(fd, total)
    mm = mmap.mmap(fd, total)
    man = {}
    off = 0
    t0 = time.time()
    done = 0
    for i, (name, dt, shape) in enumerate(keys):
        f = wmap.get(name)
        if f is None:
            raise SystemExit(f"真权重里没有 {name}")
        t = handle(f).get_tensor(name)
        if t.dtype == getattr(torch, "float8_e8m0fnu", None):
            t = t.view(torch.uint8)
        raw = t.reshape(-1).view(torch.uint8).numpy().tobytes()
        n = len(raw)
        mm[off:off + n] = raw
        man[name] = [str(dt), list(shape), off, n]
        off += n
        done += n
        if (i + 1) % 2000 == 0 or i + 1 == len(keys):
            sec = time.time() - t0
            rate = done / 2**20 / max(sec, 1e-6)
            eta = (total - done) / max(rate * 2**20, 1e-9)
            print(f"[moe-host-cache] {i + 1}/{len(keys)}  "
                  f"{done / 2**30:.1f}/{total / 2**30:.1f} GiB  "
                  f"{rate:.0f} MiB/s", flush=True)
            _write_progress(progress_json, phase="loading", pid=os.getpid(),
                            started=t0, keys_done=i + 1, keys_total=len(keys),
                            loaded=done, total=total, rate_mib_s=round(rate, 1),
                            eta_s=round(eta, 1))
    mm.flush()
    assert off == total, (off, total)
    _write_progress(progress_json, phase="ready", pid=os.getpid(),
                    started=t0, keys_done=len(keys), keys_total=len(keys),
                    loaded=total, total=total,
                    rate_mib_s=round(total / 2**20 / max(time.time() - t0, 1e-6), 1),
                    eta_s=0)
    return fd, mm, man, total, root


def serve(args):
    sys.path.insert(0, HERE)
    import assemble
    cfg = assemble.load_yaml(args.config)
    layers = parse_range(args.layers, 0, int(cfg["model"]["n_layers"]))
    E = int(cfg["model"]["n_routed_experts"])
    e0, e1 = 0, E
    if args.experts:
        ex = parse_range(args.experts, 0, E)
        e0, e1 = min(ex), max(ex) + 1
    progress_json = args.progress_json or (args.socket + ".progress.json")
    fd, mm, man, total, root = _load_into_memfd(cfg, layers, e0, e1,
                                                progress_json=progress_json)

    sockpath = args.socket
    try:
        os.unlink(sockpath)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sockpath)
    srv.listen(16)
    os.chmod(sockpath, 0o600)

    meta = {"version": 1, "model": root, "total": total,
            "layers": [layers[0], layers[-1]], "experts": [e0, e1 - 1],
            "keys": man}
    blob = zlib.compress(json.dumps(meta).encode(), 6)
    status = {"pid": os.getpid(), "socket": sockpath, "model": root,
              "total": total, "n_keys": len(man), "started": time.time()}
    with open(sockpath + ".status.json", "w") as fh:
        json.dump(status, fh)
    print(f"[moe-host-cache] daemon ready: {sockpath}  "
          f"{total / 2**30:.1f} GiB / {len(man)} keys", flush=True)

    stop = {"v": False}

    def _sig(_a, _b):
        stop["v"] = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    # ★ accept 必须带超时：否则 SIGTERM 只会置标志，accept 仍阻塞 → stop 假成功、
    #   daemon 一直占着 269 GiB（实测要再 SIGKILL 才退，munmap 大页表还要 ~30s）。
    srv.settimeout(1.0)
    while not stop["v"]:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        try:
            req = conn.recv(16).strip()
            if req.startswith(b"GET"):
                conn.sendall(struct.pack("<Q", len(blob)))
                conn.sendall(blob)
                fds = array.array("i", [fd])
                conn.sendmsg([b"F"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)])
            elif req.startswith(b"STATUS"):
                conn.sendall(json.dumps(status).encode())
        except Exception as exc:                       # noqa: BLE001
            print(f"[moe-host-cache] client error: {type(exc).__name__}: {exc}", flush=True)
        finally:
            try:
                conn.close()
            except Exception:                          # noqa: BLE001
                pass
    try:
        mm.close()
        os.close(fd)
        os.unlink(sockpath)
    except Exception:                                  # noqa: BLE001
        pass
    _write_progress(progress_json, phase="stopped", pid=os.getpid(),
                    loaded=0, total=total, keys_done=0, keys_total=len(man))
    print("[moe-host-cache] stopped", flush=True)


def daemonize(logpath, statuspath, timeout=1800, wait=True):
    """双 fork。返回 True = 第一父进程（wait=False 时继续画进度条）。"""
    pid = os.fork()
    if pid > 0:
        if not wait:
            return True
        # 父进程等 daemon 就绪（status.json 出现且 pid 存活）后再返回，
        # 这样 `start` 命令退出时缓存已经可用，用户不用猜加载到哪了。
        t0 = time.time()
        while time.time() - t0 < timeout:
            if os.path.exists(statuspath):
                try:
                    st = json.load(open(statuspath))
                    os.kill(int(st["pid"]), 0)
                    os._exit(0)
                except Exception:                      # noqa: BLE001
                    pass
            time.sleep(0.5)
        os._exit(1)
    os.setsid()
    pid = os.fork()
    if pid > 0:
        os._exit(0)
    sys.stdout.flush()
    sys.stderr.flush()
    with open(logpath, "ab") as fh:
        os.dup2(fh.fileno(), 1)
        os.dup2(fh.fileno(), 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False


def _wait_with_bar(sockpath, statuspath, args, timeout=1800):
    """start --progress-bar：父进程读 progress.json，在终端画一行加载条。"""
    import bench_protocol as P
    prog = args.progress_json or (sockpath + ".progress.json")
    bar = P.StageBar(stream=sys.stderr, width=16, every=0.3)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(statuspath):
            bar.finish(force=True)
            return True
        try:
            with open(prog) as fh:
                p = json.load(fh)
            if p.get("phase") == "stopped":
                bar.close()
                print("✗ host cache 子进程已停止", file=sys.stderr)
                return False
            pid = p.get("pid")
            if pid and not _pid_alive(pid):
                bar.close()
                print(f"✗ host cache 子进程退出（pid={pid}）；日志 {sockpath}.log",
                      file=sys.stderr)
                try:
                    with open(sockpath + ".log", errors="ignore") as fh:
                        sys.stderr.writelines(fh.readlines()[-15:])
                except OSError:
                    pass
                return False
            line = (f"[moe-host-cache] {p.get('keys_done', 0)}/"
                    f"{p.get('keys_total', 1)}  "
                    f"{p.get('loaded', 0) / 2**30:.1f}/"
                    f"{p.get('total', 1) / 2**30:.1f} GiB  "
                    f"{int(p.get('rate_mib_s', 0))} MiB/s")
            bar.feed(line)
        except (OSError, ValueError):
            pass
        time.sleep(0.3)
    bar.close()
    print(f"✗ 等待 host cache 就绪超时（{timeout}s）；日志 {sockpath}.log",
          file=sys.stderr)
    return False


def cmd_start(args):
    sockpath = args.socket
    status = sockpath + ".status.json"
    if os.path.exists(status):
        try:
            pid = json.load(open(status))["pid"]
            os.kill(pid, 0)
            raise SystemExit(f"已有 daemon 在跑（pid={pid}）；先 stop")
        except ProcessLookupError:
            os.unlink(status)
    # 清掉上一次的进度文件：否则 --progress-bar 的父进程会先读到 phase=stopped
    prog = args.progress_json or (sockpath + ".progress.json")
    for p in (prog, prog + ".tmp"):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass
    if not args.foreground:
        parent = daemonize(sockpath + ".log", status,
                           wait=not args.progress_bar)
        if parent:
            ok = _wait_with_bar(sockpath, status, args)
            os._exit(0 if ok else 1)
    serve(args)


def cmd_stop(args):
    status = args.socket + ".status.json"
    if not os.path.exists(status):
        print("没有在跑的 moe_host_cache")
        return
    pid = json.load(open(status))["pid"]

    def alive():
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(80):                                # 最多等 8 s
        if not alive():
            break
        time.sleep(0.1)
    if alive():                                        # 兜底：大缓存的 munmap 可能较慢
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for _ in range(900):                           # 269 GiB 的页表回收要 ~30 s
            if not alive():
                break
            time.sleep(0.1)
    for p in (status, args.socket):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass
    print(f"已停止（pid={pid}），内存释放{'完成' if not alive() else '中（请稍候）'}")


def cmd_status(args):
    status = args.socket + ".status.json"
    if not os.path.exists(status):
        print("未运行")
        return
    st = json.load(open(status))
    try:
        os.kill(st["pid"], 0)
        alive = True
    except ProcessLookupError:
        alive = False
    print(f"pid={st['pid']} alive={alive} {st['total'] / 2**30:.1f} GiB "
          f"{st['n_keys']} keys model={st['model']}")
    if alive:
        try:
            c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            c.settimeout(2)
            c.connect(args.socket)
            c.sendall(b"STATUS\n")
            print(c.recv(4096).decode())
            c.close()
        except Exception as exc:                       # noqa: BLE001
            print(f"连接失败：{type(exc).__name__}: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["start", "stop", "status"])
    ap.add_argument("--config", default=os.path.join(HERE, "configs/mini_real.yaml"))
    ap.add_argument("--socket", default=DEFAULT_SOCK)
    ap.add_argument("--layers", default="", help="如 0-39 或 0,1,2")
    ap.add_argument("--experts", default="", help="如 0-383 或 0-47")
    ap.add_argument("--foreground", action="store_true")
    ap.add_argument("--progress-bar", action="store_true",
                    help="start 时在终端画加载进度条（daemon 仍后台运行）")
    ap.add_argument("--progress-json", default="",
                    help="进度 JSON 路径（默认 <socket>.progress.json）")
    a = ap.parse_args()
    {"start": cmd_start, "stop": cmd_stop, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""MindStudio `trace_view.json` 多 forward 时延：**N 个 forward 的总时间 ÷ N**。

口径（用户 2026-10-22 规则；术语 2026-10-23 起统一为 forward）：
  · 多 forward 时延 = 覆盖这 N 个 forward 的总设备时间 / N；
  · 不用单个 forward，也不对逐 forward 采样取中位数；
  · trace 里用设备事件包络：`Computing`（内核外包络）或 `HcPre`（每层两次，
    每 forward=80 次；用它分组可校验 forward 数）。

用法：
  python3 tools/trace_multipass_time.py <trace_view.json> [--forwards N]
  （`--passes` 保留为兼容别名）
  N 缺省时用 HcPre 次数 / 80 推断（DSA4.1 mini 40 层 × 2 次 hc_pre）。
输出：total_ms、forwards、ms_per_forward（Computing 包络 / HcPre 包络各一份）。

文件名保留 `trace_multipass_time.py` 以兼容旧脚本；对外描述统一用 forward。
"""
import argparse
import json
import sys


def load_events(path):
    d = json.load(open(path))
    return d if isinstance(d, list) else d.get("traceEvents", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--forwards", "--passes", dest="n_fwd", type=int, default=0)
    ap.add_argument("--hcpre-per-forward", "--hcpre-per-pass",
                    dest="hcpre_per_fwd", type=int, default=80)
    a = ap.parse_args()
    ev = load_events(a.trace)
    xs = [e for e in ev if isinstance(e, dict) and e.get("ph") == "X" and "dur" in e]

    def ts(e):
        return float(e.get("ts", 0)) / 1000.0

    def end(e):
        return ts(e) + float(e["dur"]) / 1000.0

    hc = sorted([e for e in xs if str(e.get("name", "")) == "HcPre"], key=ts)
    comp = [e for e in xs if str(e.get("name", "")) == "Computing"]
    n_fwd = int(a.n_fwd)
    if n_fwd <= 0:
        if len(hc) % a.hcpre_per_fwd:
            sys.exit(f"HcPre 次数 {len(hc)} 不是 {a.hcpre_per_fwd} 的整数倍；"
                     f"请显式 --forwards")
        n_fwd = len(hc) // a.hcpre_per_fwd
    if n_fwd <= 0:
        sys.exit("无法推断 forward 数")
    print(f"trace={a.trace}")
    print(f"forwards={n_fwd}  HcPre={len(hc)}  Computing={len(comp)}")
    for name, es in (("Computing", comp), ("HcPre", hc)):
        if not es:
            continue
        t0 = min(ts(e) for e in es)
        t1 = max(end(e) for e in es)
        total = t1 - t0
        print(f"  {name:10s} total={total:9.2f} ms  /{n_fwd} = {total / n_fwd:9.2f} ms/forward")
    return 0


if __name__ == "__main__":
    sys.exit(main())

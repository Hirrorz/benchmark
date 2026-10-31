#!/usr/bin/env python3
"""MindStudio trace 的**空泡**分析：设备空闲 / 通信暴露 / 每层分摊。

口径（PERF_RULES §0 第三条：只看稳态）：
  · 一个 forward 的窗口 = HcPre 内核事件（每层 2 个，40 层 = 80 个/forward）；
  · `--forward last`（默认）只分析**最后一个** forward，避开捕获/首轮回放；
  · 设备"忙" = 该窗口内所有 kernel 区间的并集（跨 stream）；空泡 = 并集的补集；
  · 通信暴露 = 只有通信 kernel 在跑的区间（compute 侧没活干，等通信）；
  · 空泡/暴露都按层切分（HcPre 每 2 个一层），方便对到代码。

输入：`--trace <out 目录 | .../ASCEND_PROFILER_OUTPUT | kernel_details.csv>`。
      同目录有 trace_view.json 时额外统计 host 侧热点（aten::to / empty_tensor 等）。

用法：
  python3 tools/trace_gaps.py --trace <out>/trace --forward last
  python3 tools/trace_gaps.py --trace .../kernel_details.csv --json /tmp/cp2.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys

COMM_RE = re.compile(
    r"hccl|allgather|all_gather|alltoall|all_to_all|reducescatter|reduce_scatter"
    r"|broadcast|sendrecv|batchsendrecv", re.I)
HOST_HOT = ("aten::to", "aten::empty", "empty_tensor", "aten::copy_",
            "aten::index_select", "aten::zero_", "aten::fill_")


def find_output(path: str) -> str:
    """把任意入口统一成 ASCEND_PROFILER_OUTPUT 目录。"""
    if os.path.isfile(path):
        return os.path.dirname(os.path.abspath(path))
    p = os.path.abspath(path)
    if os.path.basename(p) == "ASCEND_PROFILER_OUTPUT":
        return p
    hits = []
    for root, _dirs, files in os.walk(p):
        if os.path.basename(root) == "ASCEND_PROFILER_OUTPUT":
            hits.append(root)
    if not hits:
        sys.exit(f"没找到 ASCEND_PROFILER_OUTPUT：{path}")
    return max(hits, key=os.path.getmtime)


def load_kernels(outdir: str):
    kd = os.path.join(outdir, "kernel_details.csv")
    if not os.path.exists(kd):
        sys.exit(f"缺 kernel_details.csv：{kd}")
    rows = []
    with open(kd, newline="", errors="ignore") as fh:
        for r in csv.DictReader(fh):
            try:
                st = float(str(r["Start Time(us)"]).strip())
                du = float(str(r["Duration(us)"]).strip())
            except (KeyError, ValueError):
                continue
            rows.append({
                "start": st, "end": st + du, "dur": du,
                "name": r.get("Name", ""), "type": r.get("Type", ""),
                "stream": str(r.get("Stream ID", "")).strip(),
                "wait": _f(r.get("Wait Time(us)")),
            })
    rows.sort(key=lambda x: x["start"])
    return rows


def _f(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return 0.0


def is_comm(k) -> bool:
    return bool(COMM_RE.search(f"{k['name']} {k['type']}"))


def union(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def union_len(iv):
    return sum(e - s for s, e in union(iv))


def subtract(a, b):
    """a、b 都是已合并区间；返回 a \\ b。"""
    out, j = [], 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            bs, be = b[k]
            if bs > cur:
                out.append((cur, min(bs, e)))
            cur = max(cur, be)
            if cur >= e:
                break
            k += 1
        if cur < e:
            out.append((cur, e))
    return out


def intersect(a, b):
    return subtract(a, subtract(a, b))


def gaps_of(busy, t0, t1):
    out, cur = [], t0
    for s, e in busy:
        if s > cur:
            out.append((cur, min(s, t1)))
        cur = max(cur, e)
        if cur >= t1:
            break
    if cur < t1:
        out.append((cur, t1))
    return [(s, e) for s, e in out if e > s]


def kernel_at(rows, t, side):
    """离 t 最近的前一个（side='prev'）或后一个 kernel。"""
    if side == "prev":
        best = None
        for k in rows:
            if k["end"] <= t and (best is None or k["end"] > best["end"]):
                best = k
    else:
        best = None
        for k in rows:
            if k["start"] >= t and (best is None or k["start"] < best["start"]):
                best = k
    return best


def fmt_k(k):
    if not k:
        return "-"
    return f"{k['type']}/{k['name'][:38]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--forward", default="last",
                    help="last（默认）| all | 0 起的序号。注意：最后一个 forward 的窗口"
                         "含 profiler.stop() 尾巴（实测 2.6s→27.8s），稳态分析请用 1")
    ap.add_argument("--json", default="")
    ap.add_argument("--min-gap-ms", type=float, default=0.5,
                    help="列出的空泡下限（ms）")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()

    outdir = find_output(a.trace)
    rows = load_kernels(outdir)
    hc = [k for k in rows if k["type"] == "HcPre" or k["name"] == "HcPre"]
    hc.sort(key=lambda k: k["start"])
    per_fwd = 80
    if len(hc) < per_fwd:
        sys.exit(f"HcPre 只有 {len(hc)} 个，不足一个 forward（{per_fwd}）")
    n_fwd = len(hc) // per_fwd
    if a.forward == "last":
        fwd = n_fwd - 1
    elif a.forward == "all":
        fwd = None
    else:
        fwd = int(a.forward)

    print(f"trace={outdir}")
    print(f"kernels={len(rows)}  HcPre={len(hc)}  forwards={n_fwd}")
    fwds = range(n_fwd) if fwd is None else [fwd]
    summary = {"trace": outdir, "forwards": {}}
    for f in fwds:
        t0 = hc[f * per_fwd]["start"]
        t1 = (hc[(f + 1) * per_fwd]["start"]
              if (f + 1) * per_fwd < len(hc) else max(k["end"] for k in rows))
        res = analyze(rows, hc, f, t0, t1, a, outdir)
        summary["forwards"][f] = res
        print_report(f, res, a)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(summary, fh, indent=1, ensure_ascii=False)
        print(f"json → {a.json}")


def analyze(rows, hc, f, t0, t1, a, outdir):
    per_fwd = 80
    win = [k for k in rows if k["end"] > t0 and k["start"] < t1]
    busy = union([(max(k["start"], t0), min(k["end"], t1)) for k in win])
    idle = gaps_of(busy, t0, t1)
    comm = [k for k in win if is_comm(k)]
    comp = [k for k in win if not is_comm(k)]
    u_comm = union([(max(k["start"], t0), min(k["end"], t1)) for k in comm])
    u_comp = union([(max(k["start"], t0), min(k["end"], t1)) for k in comp])
    comm_only = subtract(u_comm, u_comp)
    other_only = subtract(subtract(busy, u_comp), u_comm)
    comp_idle = subtract([(t0, t1)], u_comp)
    comp_idle_comm = intersect(comp_idle, u_comm)
    comp_idle_none = subtract(comp_idle, busy)
    comp_idle_other = subtract(subtract(comp_idle, u_comm), comp_idle_none)

    types = collections.defaultdict(lambda: [0, 0.0])
    for k in win:
        ty = k["type"] or k["name"]
        types[ty][0] += 1
        types[ty][1] += min(k["end"], t1) - max(k["start"], t0)

    # ── 每层 ──
    layers = []
    for li in range(40):
        i0 = f * per_fwd + 2 * li
        ls = hc[i0]["start"]
        le = hc[i0 + 2]["start"] if (li < 39 and i0 + 2 < len(hc)) else t1
        le = min(le, t1)
        if le <= ls:
            continue
        lwin = [k for k in win if k["end"] > ls and k["start"] < le]
        lbusy = union([(max(k["start"], ls), min(k["end"], le)) for k in lwin])
        lidle = gaps_of(lbusy, ls, le)
        lcomm = union([(max(k["start"], ls), min(k["end"], le))
                       for k in lwin if is_comm(k)])
        lcomp = union([(max(k["start"], ls), min(k["end"], le))
                       for k in lwin if not is_comm(k)])
        layers.append({
            "layer": li,
            "ms": (le - ls) / 1000.0,
            "idle_ms": union_len(lidle) / 1000.0,
            "comm_exposed_ms": union_len(subtract(lcomm, lcomp)) / 1000.0,
            "comm_ms": union_len(lcomm) / 1000.0,
            "max_idle_ms": max([e - s for s, e in lidle], default=0.0) / 1000.0,
        })

    # ── 空泡明细 ──
    gaps = []
    for s, e in sorted(idle, key=lambda x: -(x[1] - x[0]))[: a.top]:
        if (e - s) / 1000.0 < a.min_gap_ms:
            break
        p = kernel_at(win, s, "prev")
        n = kernel_at(win, e, "next")
        gaps.append({
            "start_ms": (s - t0) / 1000.0,
            "ms": (e - s) / 1000.0,
            "prev": fmt_k(p), "prev_stream": p["stream"] if p else "",
            "next": fmt_k(n), "next_stream": n["stream"] if n else "",
        })

    # ── 通信暴露明细 ──
    exp = []
    for s, e in sorted(comm_only, key=lambda x: -(x[1] - x[0]))[: a.top]:
        if (e - s) / 1000.0 < a.min_gap_ms:
            break
        inside = [k for k in comm if k["start"] < e and k["end"] > s]
        exp.append({
            "start_ms": (s - t0) / 1000.0,
            "ms": (e - s) / 1000.0,
            "kernels": collections.Counter(fmt_k(k) for k in inside).most_common(4),
        })

    long_k = sorted(win, key=lambda k: -k["dur"])[: a.top]

    host = host_profile(outdir, t0, t1)

    return {
        "window_ms": (t1 - t0) / 1000.0,
        "kernels": len(win),
        "busy_ms": union_len(busy) / 1000.0,
        "idle_ms": union_len(idle) / 1000.0,
        "idle_pct": 100.0 * union_len(idle) / max(1e-9, t1 - t0),
        "comm_kernels": len(comm),
        "comm_busy_ms": union_len(u_comm) / 1000.0,
        "comm_exposed_ms": union_len(comm_only) / 1000.0,
        "other_only_ms": union_len(other_only) / 1000.0,
        "compute_idle_ms": union_len(comp_idle) / 1000.0,
        "compute_idle_comm_ms": union_len(comp_idle_comm) / 1000.0,
        "compute_idle_none_ms": union_len(comp_idle_none) / 1000.0,
        "compute_idle_other_ms": union_len(comp_idle_other) / 1000.0,
        "gaps_over_1ms": sum(1 for s, e in idle if e - s >= 1000),
        "gaps_over_5ms": sum(1 for s, e in idle if e - s >= 5000),
        "max_gap_ms": max([e - s for s, e in idle], default=0.0) / 1000.0,
        "types": {k: {"n": v[0], "ms": v[1] / 1000.0}
                  for k, v in sorted(types.items(), key=lambda kv: -kv[1][1])[:14]},
        "layers": layers,
        "gaps": gaps,
        "comm_exposed": exp,
        "long_kernels": [{"ms": k["dur"] / 1000.0, "k": fmt_k(k),
                          "stream": k["stream"]} for k in long_k],
        "host": host,
    }


def host_profile(outdir, t0, t1):
    tv = os.path.join(outdir, "trace_view.json")
    if not os.path.exists(tv):
        return {}
    try:
        ev = json.load(open(tv))
    except (OSError, ValueError):
        return {}
    if not isinstance(ev, list):
        ev = ev.get("traceEvents", [])
    acc = collections.defaultdict(lambda: [0, 0.0])
    for e in ev:
        if not isinstance(e, dict) or e.get("ph") != "X" or "dur" not in e:
            continue
        try:
            ts = float(e.get("ts", 0))
        except (TypeError, ValueError):
            continue
        if not (t0 <= ts < t1):
            continue
        nm = str(e.get("name", ""))
        cat = str(e.get("cat", ""))
        if cat == "cpu_op" or "aten::" in nm or "aclrt" in nm or "AscendCL" in nm:
            du = float(e["dur"])
            acc[nm][0] += 1
            acc[nm][1] += du
    hot = {k: {"n": v[0], "ms": v[1] / 1000.0}
           for k, v in acc.items() if k in HOST_HOT}
    top = dict(sorted(acc.items(), key=lambda kv: -kv[1][1])[:12])
    top = {k: {"n": v[0], "ms": v[1] / 1000.0} for k, v in top.items()}
    return {"hot": hot, "top": top}


def print_report(f, r, a):
    print(f"\n== forward {f}（窗口 {r['window_ms']:.1f} ms，kernel {r['kernels']}）==")
    print(f"设备忙 {r['busy_ms']:.1f} ms；空泡 {r['idle_ms']:.1f} ms"
          f"（{r['idle_pct']:.1f}%；>1ms {r['gaps_over_1ms']} 个 / >5ms "
          f"{r['gaps_over_5ms']} 个；最大 {r['max_gap_ms']:.2f} ms）")
    print(f"通信：kernel {r['comm_kernels']} 个，忙 {r['comm_busy_ms']:.1f} ms；"
          f"**暴露 {r['comm_exposed_ms']:.1f} ms**（只有通信在跑，compute 空闲）")
    print(f"compute 空闲 {r['compute_idle_ms']:.1f} ms = 等通信 "
          f"{r['compute_idle_comm_ms']:.1f} + 全设备空泡 {r['compute_idle_none_ms']:.1f}"
          f" + 其他流 {r['compute_idle_other_ms']:.1f}")
    print("类型（n / 区间合计 ms）：")
    for k, v in list(r["types"].items())[:10]:
        print(f"  {v['n']:>6}  {v['ms']:>9.1f}  {k[:60]}")
    if r["gaps"]:
        print("最长空泡：")
        for g in r["gaps"]:
            print(f"  +{g['start_ms']:8.1f}ms  {g['ms']:7.2f}ms  "
                  f"前={g['prev']}(s{g['prev_stream']}) → 后={g['next']}(s{g['next_stream']})")
    if r["comm_exposed"]:
        print("最长通信暴露：")
        for g in r["comm_exposed"]:
            ks = ", ".join(f"{n}×{c}" for n, c in g["kernels"])
            print(f"  +{g['start_ms']:8.1f}ms  {g['ms']:7.2f}ms  {ks}")
    worst = sorted(r["layers"], key=lambda x: -(x["idle_ms"] + x["comm_exposed_ms"]))[: a.top]
    print("每层空泡/暴露 top：")
    for l in worst:
        print(f"  L{l['layer']:>2}  层 {l['ms']:7.1f}ms  空泡 {l['idle_ms']:6.2f}ms"
              f"  通信暴露 {l['comm_exposed_ms']:6.2f}ms  最长空泡 {l['max_idle_ms']:.2f}ms")
    if r["host"]:
        print("host 热点（n / 合计 ms）：")
        for k, v in sorted(r["host"]["hot"].items(), key=lambda kv: -kv[1]["ms"]):
            print(f"  {v['n']:>6}  {v['ms']:>8.2f}  {k}")
        print("host 总耗时 top3：" + "  ".join(
            f"{k} {v['ms']:.1f}" for k, v in list(r["host"]["top"].items())[:3]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""从 `mindstudio_profiler_output/` 出「逐 forward 时延 + 算子 Top」报表。

用法（在 ds41_mini 根目录）：
  python3.11 tools/trace_report.py <路径> [-o 输出目录] [--layers 40] [--export] [--all-ranks]

`<路径>` 可以是：
  · 采集根目录（下面有 `h*/L*/prefill/PROF_*`）
  · 点位目录（`.../h0.97/L262144`，下面有 `prefill/PROF_*`）
  · 一个 `PROF_*` 目录，或它的 `mindstudio_profiler_output/`

产物（默认写到 `<路径>/PARSED/`，`-o` 可改）：
  forward_times.csv   每个点一行：forward 数 / 稳态 ms / 总 span / busy / idle
  forward_detail.csv  每个 forward 一行：点、序号、span/busy/idle
  op_top.csv          按 OP Type 聚合的总时长与次数
  api_top.csv         按 ACL API 聚合的总时长与次数
  SUMMARY.md          人读

口径（与 `PERF_RULES.md` §0 一致）：
  · **forward = 80 个 `HcPre` 事件**（每层 2 次 × 40 层；`--layers N` 可改 → 2N）。
  · **稳态 ms/forward = 剔除第 1 个 traced forward 后取中位数**——第 1 个含 profiler
    启动/残余冷启，直接平均会虚高（实测 256k CP2：含首轮 270ms vs 剔除后 243ms）。
  · **busy** = `task_time_*.csv` 里 task 区间的并集；`idle = span − busy`。
  · 这是 rank0 的 trace（**带 msprof 开销**）；性能数仍以 `launch.log` 的
    `[r1] 多轮耗时` 为准（trace 只用于看结构/空泡/算子占比）。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import shutil
import subprocess
import sys
from collections import defaultdict

MSPROF_CANDIDATES = [
    os.environ.get("MSPROF", ""),
    "/mnt/xds/sfs/l00972766/ds41_runtime/ascend_full/usr/local/Ascend/cann-9.2.0/aarch64-linux/bin/msprof",
]
PREFIXES = ["op_summary", "op_statistic", "task_time", "api_statistic",
            "communication_statistic", "step_trace", "msprof"]


# ── 点位命名（用户口径 2026-10-28：扁平名，不套多层目录）────────────────────
#   规范：cp<CP>dp<DP>ep<EP>_L<Lk>_hit<h>_batch<b>_replay<on|off>_prefill
#   例：  cp4dp4ep8_L256k_hit0.97_batch1_replayon_prefill
def _fmt_L(L: int) -> str:
    """序列长 = **具体数字**（2026-10-29 用户口径：262144 就写 262144，不用 256k/1m）。"""
    return str(int(L))


def _fmt_h(h) -> str:
    try:
        return f"{float(h):g}"
    except (TypeError, ValueError):
        return "0"


def _read_simple_yaml(path):
    """只读顶层 section 下的 `  key: value`（本仓 config_point.yaml 够用）。"""
    d, sec = {}, None
    try:
        for ln in open(path, encoding="utf-8", errors="replace"):
            if not ln.strip() or ln.lstrip().startswith("#") or ":" not in ln:
                continue
            if not ln[0].isspace() and ln.rstrip().endswith(":"):
                sec = ln.strip()[:-1]
                continue
            if sec and ln.startswith("  ") and not ln.startswith("   "):
                k, _, v = ln.strip().partition(":")
                if k and v.strip():
                    d[f"{sec}.{k}"] = v.strip()
    except OSError:
        pass
    return d


# 标准扁平名（2026-10-29 用户口径 = `_batch<b>_replay<on|off>`；也认过渡期 `_b<b>_r<0|1>`）
_FLAT_RX = re.compile(
    r"^cp\d+dp\d+ep\d+_L\d+(?:\+\d+)*_hit[0-9.]+(?:\+[0-9.]+)*"
    r"(?:_batch\d+_replay(?:on|off)|_b\d+_r[01])_prefill$")


def flat_name(point_dir: str, ov: dict | None = None) -> str:
    """给出该点位的扁平名。优先级：`<点>/NAME` > config_point.yaml+PARAMS 推导 > 目录名兜底。"""
    ov = ov or {}
    # ★ 2026-10-29：采集时目录名就已经是标准扁平名（points_runner 生成）→ 直接用
    _base = os.path.basename(os.path.normpath(point_dir))
    if _FLAT_RX.match(_base):
        return _base
    name_file = os.path.join(point_dir, "NAME")
    if os.path.isfile(name_file):
        v = open(name_file, encoding="utf-8", errors="replace").read().strip()
        if v:
            return v
    cfg = _read_simple_yaml(os.path.join(point_dir, "config_point.yaml"))
    params = {}
    try:
        for tok in open(os.path.join(point_dir, "PARAMS"), encoding="utf-8",
                        errors="replace").read().split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                params[k] = v
    except OSError:
        pass
    def pick(o, k, d):
        return ov.get(k, o.get(k, d))
    world = int(float(cfg.get("parallel_config.world_size", params.get("world_size", 8)) or 8))
    cp = int(ov["cp"]) if "cp" in ov else int(float(cfg.get("parallel_config.cp_size", 1) or 1))
    ep = int(ov["ep"]) if "ep" in ov else int(float(cfg.get("parallel_config.moe_ep_size", world) or world))
    dp = int(ov["dp"]) if "dp" in ov else (world // cp if cp > 0 else world)
    L = int(float(params.get("L") or cfg.get("run.seq_len") or 0))
    h = params.get("h", cfg.get("hit.ratio", 0))
    b = int(float(params.get("batches") or cfg.get("run.batch") or 1))
    replay = str(params.get("replay", cfg.get("features.decoder_replay", "1"))).lower() in ("1", "true", "on", "yes")
    if L <= 0:                                     # 连 L 都读不到 → 退回目录名
        # points 路径的点位目录名形如 `(262144_0.6)_r1`：直接解析成扁平名
        m = re.fullmatch(r"\((\d+)_([\d.]+)\)_r([01])", os.path.basename(point_dir))
        if m:
            _L, _h, _r = int(m.group(1)), m.group(2), m.group(3) == "1"
            return (f"cp{cp}dp{dp}ep{ep}_L{_fmt_L(_L)}_hit{_fmt_h(_h)}_batch{b}"
                    f"_replay{'on' if _r else 'off'}_prefill")
        return os.path.basename(point_dir)
    return (f"cp{cp}dp{dp}ep{ep}_L{_fmt_L(L)}_hit{_fmt_h(h)}_batch{b}"
            f"_replay{'on' if replay else 'off'}_prefill")


def point_dir_of(pdir: str) -> str:
    """`.../<point>/prefill/PROF_*` → `.../<point>`；`.../<point>/PROF_*` → `.../<point>`。"""
    d = os.path.dirname(pdir)
    return os.path.dirname(d) if os.path.basename(d) in ("prefill", "profile") else d


def find_traces(path: str, ov: dict | None = None):
    """返回 [(point_tag, PROF 目录)]。"""
    path = os.path.abspath(path)
    if os.path.basename(path).startswith("PROF_"):
        return [(flat_name(point_dir_of(path), ov), path)]
    if os.path.basename(path) == "mindstudio_profiler_output":
        pdir = os.path.dirname(path)
        return [(flat_name(point_dir_of(pdir), ov), pdir)]
    out = []
    for pdir in sorted(glob.glob(os.path.join(path, "**", "prefill", "PROF_*"), recursive=True)):
        out.append((flat_name(point_dir_of(pdir), ov), pdir))
    if not out:                                   # 直接给点位目录
        for pdir in sorted(glob.glob(os.path.join(path, "prefill", "PROF_*"))):
            out.append((flat_name(point_dir_of(pdir), ov), pdir))
    if not out:                                   # 扁平交付包：<包>/<扁平名>/mindstudio_profiler_output
        for od in sorted(glob.glob(os.path.join(path, "**", "mindstudio_profiler_output"), recursive=True)):
            out.append((os.path.basename(os.path.dirname(od)), os.path.dirname(od)))
    return out


def pick_device0(traces, all_ranks):
    if all_ranks:
        return traces
    picked = [t for t in traces if os.path.isdir(os.path.join(t[1], "device_0"))]
    return picked or traces


# ── 导出 / 去重 ─────────────────────────────────────────────────────────────
def find_msprof():
    for p in MSPROF_CANDIDATES:
        if p and os.path.exists(p) and os.access(p, os.X_OK):
            return p
    return None


def _has_export(out):
    """★ 2026-10-31：目录存在**不等于**导出成功——实测 `mindstudio_profiler_output/`
    里可能只有 README.txt（msprof 未导出 summary）。旧代码只要看到目录就复用 ⇒
    整套 PARSED 全空、用户拿不到算子级数据（这就是"打包是空包"的真根因）。
    判据：至少有一个非 README 的 json/csv。"""
    if not os.path.isdir(out):
        return False
    for f in os.listdir(out):
        if f.startswith("README"):
            continue
        if f.endswith((".json", ".csv")):
            return True
    return False


def ensure_export(pdir, do_export):
    out = os.path.join(pdir, "mindstudio_profiler_output")
    if _has_export(out):
        return out
    if not do_export:
        return None
    msprof = find_msprof()
    if not msprof:
        return None
    try:
        subprocess.run([msprof, "--export=on", f"--output={pdir}", "--type=text",
                        "--task-time=l0", "--aicore-shape=on"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
    except Exception:                                        # noqa: BLE001
        return None
    return out if _has_export(out) else None


def dedupe(outdir):
    """`msprof --export=on` 非幂等：同前缀多份只留最新时间戳。"""
    for pre in PREFIXES:
        cand = [f for f in glob.glob(os.path.join(outdir, pre + "_*")) if os.path.isfile(f)]
        if len(cand) > 1:
            for f in sorted(cand, key=os.path.getmtime)[:-1]:
                try:
                    os.remove(f)
                except OSError:
                    pass


def newest(outdir, pattern):
    c = sorted(glob.glob(os.path.join(outdir, pattern)))
    return c[-1] if c else None


# ── 解析 ────────────────────────────────────────────────────────────────────
def load_events(path):
    ev = json.load(open(path))
    if isinstance(ev, dict):
        ev = ev.get("traceEvents") or ev.get("events") or []
    return [e for e in ev if isinstance(e, dict) and e.get("ph") == "X" and "dur" in e]


def forward_bounds(events, layers):
    """用 HcPre 分组切 forward：每 forward = 2×层数 个 HcPre。返回 (nf, [起点...])。"""
    per = 2 * layers
    ts = sorted(float(e["ts"]) for e in events if str(e.get("name")) == "HcPre")
    nf = len(ts) // per if per else 0
    if nf < 1:
        return 0, [], len(ts)
    return nf, [ts[per * k] for k in range(nf)], len(ts)


def load_intervals(path):
    iv = []
    if not path:
        return iv
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                s = float((r.get("task_start(us)") or "").strip())
                d = float((r.get("task_time(us)") or "").strip())
            except ValueError:
                continue
            if d >= 0:
                iv.append((s, s + d))
    iv.sort()
    merged = []
    for s, e in iv:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def busy_between(merged, a, b):
    return sum(max(0.0, min(e, b) - max(s, a)) for s, e in merged)


def top_by(csv_path, name_keys, time_key, count_key="Count"):
    agg = defaultdict(lambda: [0.0, 0])
    if not csv_path:
        return agg
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            name = next((r.get(k) for k in name_keys if r.get(k)), None)
            if not name:
                continue
            try:
                t = float((r.get(time_key) or "0").strip())
            except ValueError:
                continue
            try:
                n = int(float((r.get(count_key) or "1").strip()))
            except ValueError:
                n = 1
            cur = agg[name]
            cur[0] += t
            cur[1] += n
    return agg


# ── 主流程 ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-o", "--out", default="")
    ap.add_argument("--layers", type=int, default=40, help="层数（HcPre 每 forward = 2×层数）")
    ap.add_argument("--export", action="store_true", help="缺 mindstudio_profiler_output 时调 msprof 导出")
    ap.add_argument("--all-ranks", action="store_true", help="每个 PROF_* 都解析（默认只取 device_0）")
    ap.add_argument("--tag", default="", help="点位名显式覆盖（只给单点时用）")
    ap.add_argument("--pack", default="", metavar="DIR",
                    help="扁平交付：把每个点的 mindstudio_profiler_output/ 复制到 <DIR>/<扁平名>/ 下")
    ap.add_argument("--cp", type=int, default=None, help="覆盖 cp（config_point.yaml 里的 cp 可能被 env 覆盖而过期）")
    ap.add_argument("--dp", type=int, default=None, help="覆盖 dp")
    ap.add_argument("--ep", type=int, default=None, help="覆盖 ep")
    a = ap.parse_args()

    ov = {k: v for k, v in (("cp", a.cp), ("dp", a.dp), ("ep", a.ep)) if v is not None}
    traces = pick_device0(find_traces(a.path, ov), a.all_ranks)
    if a.tag:
        traces = [(a.tag, p) for _, p in traces]
    if not traces:
        sys.exit(f"✗ 没找到 prefill/PROF_*：{a.path}")
    outdir = a.out or os.path.join(os.path.abspath(a.path), "PARSED")
    os.makedirs(outdir, exist_ok=True)

    rows, details = [], []
    op_agg, api_agg = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
    for point, pdir in traces:
        od = ensure_export(pdir, a.export)
        if not od:
            rows.append([point, "", "", "", "", "", "无 mindstudio_profiler_output（加 --export）"])
            continue
        dedupe(od)
        if a.pack:                                # 扁平交付：<pack>/<扁平名>/<七件套>
            # ★ 2026-10-29 用户口径：文件夹名 = 扁平点位名，文件**直接**放它下面
            #   （不再套一层 mindstudio_profiler_output/）。
            dst = os.path.join(a.pack, point)
            os.makedirs(dst, exist_ok=True)
            for _f in glob.glob(os.path.join(od, "*")):
                if os.path.isfile(_f):
                    shutil.copy2(_f, os.path.join(dst, os.path.basename(_f)))
        js, tt = newest(od, "msprof_*.json"), newest(od, "task_time_*.csv")
        if not js:
            rows.append([point, "", "", "", "", "", "缺 msprof_*.json"])
            continue
        events = load_events(js)
        nf, starts, hc_n = forward_bounds(events, a.layers)
        if nf < 1:
            rows.append([point, 0, "", "", "", "", f"HcPre={hc_n}，切不出 forward（--layers 对吗？）"])
            continue
        merged = load_intervals(tt)
        if not merged:
            rows.append([point, nf, "", "", "", "", "缺 task_time_*.csv"])
            continue
        end_all = merged[-1][1]
        spans, busies = [], []
        for k in range(nf):
            s = starts[k]
            e = starts[k + 1] if k + 1 < nf else end_all
            b = busy_between(merged, s, e)
            spans.append(e - s)
            busies.append(b)
            details.append([point, k, f"{(e - s) / 1000:.1f}", f"{b / 1000:.1f}",
                            f"{(e - s - b) / 1000:.1f}"])
        warm = spans[1:] if nf > 1 else spans
        steady = statistics.median(warm)
        span_all = end_all - merged[0][0]
        busy_all = sum(m[1] - m[0] for m in merged)
        rows.append([point, nf, f"{steady / 1000:.1f}", f"{span_all / 1000:.1f}",
                     f"{busy_all / 1000:.1f}", f"{(span_all - busy_all) / span_all * 100:.1f}" if span_all else "",
                     ""])
        for k, v in top_by(newest(od, "op_statistic_*.csv"), ("OP Type", "Op Type", "Name"),
                           "Total Time(us)").items():
            op_agg[k][0] += v[0]
            op_agg[k][1] += v[1]
        for k, v in top_by(newest(od, "api_statistic_*.csv"), ("API Name", "api_name", "Name"),
                           "Time(us)").items():
            api_agg[k][0] += v[0]
            api_agg[k][1] += v[1]

    def dump(name, header, data):
        p = os.path.join(outdir, name)
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(data)
        return p

    dump("forward_times.csv",
         ["point", "forwards", "steady_ms_per_forward", "span_ms", "busy_ms", "idle_pct", "note"], rows)
    dump("forward_detail.csv", ["point", "fwd", "span_ms", "busy_ms", "idle_ms"], details)
    dump("op_top.csv", ["op", "total_us", "count"],
         [[k, f"{v[0]:.1f}", v[1]] for k, v in sorted(op_agg.items(), key=lambda kv: -kv[1][0])])
    dump("api_top.csv", ["api", "total_us", "count"],
         [[k, f"{v[0]:.1f}", v[1]] for k, v in sorted(api_agg.items(), key=lambda kv: -kv[1][0])])

    with open(os.path.join(outdir, "SUMMARY.md"), "w") as f:
        f.write("# trace 报表（trace_report.py）\n\n")
        f.write(f"- 输入：`{os.path.abspath(a.path)}`\n- forward = {2 * a.layers} 个 HcPre（层数 {a.layers}）\n")
        f.write("- steady = 剔除第 1 个 traced forward 后的中位数（rank0 trace，带 msprof 开销）\n\n")
        f.write("| 点 | forwards | steady ms/fwd | span ms | busy ms | idle % | 备注 |\n|---|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
        f.write("\n## 算子 Top10（Total Time）\n\n| op | ms | count |\n|---|---:|---:|\n")
        for k, v in sorted(op_agg.items(), key=lambda kv: -kv[1][0])[:10]:
            f.write(f"| {k[:60]} | {v[0] / 1000:.1f} | {v[1]} |\n")
        f.write("\n## Host API Top10\n\n| api | ms | count |\n|---|---:|---:|\n")
        for k, v in sorted(api_agg.items(), key=lambda kv: -kv[1][0])[:10]:
            f.write(f"| {k} | {v[0] / 1000:.1f} | {v[1]} |\n")
    print(f"✅ {outdir}/SUMMARY.md（{len(rows)} 个点）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

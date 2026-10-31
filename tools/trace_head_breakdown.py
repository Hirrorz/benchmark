#!/usr/bin/env python3
"""头段/尾段拆分器（CPU-only，只读 trace，不跑 NPU、不改 ds41_mini）。

回答一个问题：**一次 forward 的墙钟里，尾段（graph replay）占多少，剩下的头段里
设备忙多少、空泡多少、忙在哪些算子族**。

用法：
  python3 tools/trace_head_breakdown.py --trace-dir <点位目录或单个 *_ascend_pt>
  python3 tools/trace_head_breakdown.py --trace-dir <点位>/prefill --json out.json
  python3 tools/trace_head_breakdown.py --trace-dir ... --ranks 0,3,7 --passes 1,2,3

输入：
  · 点位目录（含若干 `*_ascend_pt` 子目录，如 `(65536_0.97)_r1/prefill`），或
    单个 `*_ascend_pt` 目录，或一个直接含 `ASCEND_PROFILER_OUTPUT/` 的目录。
  · 每个 rank 目录下优先用 `ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_*.db`
    （SQLite，**带 streamId**，能分出尾段），否则退回 `kernel_details.csv`
    （无 stream 列 → 尾段不可分，本工具会显式告警并把 tail 记 0）。

口径（与 TEAM/notes/tooling-profiler/2026-10-26-optimal-trace-report.md 一致）：
  · 「真实 kernel」= TASK 表里能 join 到 COMPUTE_TASK_INFO 名字的那些；EVENT/NOTIFY/
    MEMCPY/MODEL_WAIT_COMPLETE 这类同步标记另算，不记入算子族时长。
  · 「设备忙 / 空泡」取 `OVERLAP_ANALYSIS`（COMPUTE / FREE）与窗口求交——这是
    profiler 自己算的设备级并集，比自算 union 更可信；没有该表时退回自算 union。
  · **窗口**：以「尾段 replay 结束」为 forward 边界（尾段是每次 forward 末尾在侧流上
    的一个连续簇）。边界处不设额外余量 ⇒ forward 边界（profiler sync 补偿
    4.2–5.9ms，见 runner.py:1435-1443 与 creative-lead 2026-10-26 创意 3）
    天然被排除在窗口外，只在 `boundary` 一栏单独标注。
  · **尾段识别**：主算流 = 真实 kernel 时长最大的流；尾段 = 某条非主流上
    「前面静默 ≥ --tail-idle-ms、≥ --tail-min-kernels 个 kernel、位于窗口后 60%」的
    最大连续簇。找不到就认为这条 trace 没有 graph replay（tail=0）。
  · **稳态**：默认剔掉第 1 个 forward（捕获后首轮回放），见 PERF_RULES §0。
  · **多 rank**：逐 rank 算完取中位（--stat median|mean|min|max）。

输出：stderr 打进度，stdout 打 markdown 表（尾段/头段/算子族/空泡 top），
      --json 另存结构化结果。

已知限制：
  · trace 里的 kernel 时长是 **profiling 口径**；本工具的结论是「占比与结构」，
    绝对值请用 noprof 墙钟（如 tmp/MATRIX_28_V4_FULL.tsv）。
  · 只有 4 个 forward 时，中位 forward 与均值差 <1%，但别拿单 forward 下结论。
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import statistics
import sys

# ── 算子族（顺序 = 优先级，先匹配先赢）──────────────────────────────────────
FAMILIES = [
    ("MoE",          r"mega_moe|moe_topk_gate|MoeInitRouting|MoeReRouting|MoeFinalize|alltoall"),
    ("Attn",         r"MqsmlaKernel|mixed_quant_sparse_flash_mla|KvCompressEpilog|CompressorV2"
                     r"|FusedPostScatter"),
    ("RoPE",         r"RotaryMul"),
    ("Indexer",      r"LightningIndexer|indexer_prolog"),
    ("HcPre",        r"^HcPre$"),
    ("MhcPost",      r"MhcPost"),
    ("DenseGEMM",    r"QuantBatchMatmul|TransposeQuantBatchMatMul|FusedMatMul|MatMulV3"),
    ("EngramDeq",    r"LeftShift|RightShift"),          # 融合 QMM 之前的 engram 解量化
    ("Engram",       r"[Ee]ngram"),
    ("Cast/Move",    r"Cast|TensorMove|BroadcastTo|ViewCopy"),
    ("RmNorm",       r"^RmsNorm$"),
    ("MxQuant",      r"DynamicMxQuant"),
    ("Gather/Sel",   r"GatherV2|IndexSelect|SelectV2|SWhere|RepeatInterleave|^Index"),
    ("Elementwise",  r".*"),
]
FAM_RX = [(f, re.compile(rx)) for f, rx in FAMILIES]

# 同步/标记类任务名（不是真算子）：不参与算子族统计
MARKER_RX = re.compile(
    r"^(EVENT RECORD|EVENT WAIT|EVENT RESET|NOTIFY RECORD|NOTIFY WAIT"
    r"|MEMCPY_ASYNC|MEMSET_ASYNC|MODEL_WAIT_COMPLETE|PROFILER_|AI_CORE)$")

# 一次 forward 里逐层算子（用来把时间点映射回层号）：MoE 每层一次
LAYER_MARKERS = [re.compile(r"mega_moe"), re.compile(r"^HcPre$")]


def family_of(name: str) -> str:
    for fam, rx in FAM_RX:
        if rx.search(name):
            return fam
    return "Elementwise"


def is_marker(name: str) -> bool:
    return bool(MARKER_RX.match(name))


# ── 数据加载 ────────────────────────────────────────────────────────────────
def find_rank_dirs(root: str):
    """→ [(label, dir)]，dir 里含 ASCEND_PROFILER_OUTPUT/。"""
    root = os.path.abspath(root)
    if os.path.isdir(os.path.join(root, "ASCEND_PROFILER_OUTPUT")):
        return [(os.path.basename(root.rstrip("/")), root)]
    hits = []
    for depth in range(1, 5):          # 逐层加深：<pt>/prefill/<host>_ascend_pt/…
        pat = os.path.join(root, *(["*"] * depth), "ASCEND_PROFILER_OUTPUT")
        hits = sorted(glob.glob(pat))
        if hits:
            break
    out = []
    for h in hits:
        d = os.path.dirname(h)
        base = os.path.basename(d)
        m = re.search(r"_(\d{5,})_\d{14,}_ascend_pt$", base)
        out.append((m.group(1) if m else base, d))
    return out


def _db_path(d):
    cands = sorted(glob.glob(os.path.join(d, "ASCEND_PROFILER_OUTPUT",
                                          "ascend_pytorch_profiler_*.db")))
    return cands[0] if cands else None


def load_from_db(path):
    """→ (tasks, overlap, meta)。时间单位统一成 µs，起点归零。

    tasks:  [{'start','end','stream','name','fam','marker'}]
    overlap: {'compute': [(s,e)], 'free': [(s,e)]} 或 None
    """
    import sqlite3
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cur = con.execute(
        "SELECT t.startNs, t.endNs, t.streamId, s.value "
        "FROM TASK t "
        "JOIN COMPUTE_TASK_INFO ci ON t.globalTaskId = ci.globalTaskId "
        "JOIN STRING_IDS s ON s.id = ci.name")
    raw = cur.fetchall()
    if not raw:
        con.close()
        return None, None, {"error": "no named tasks"}
    # 同步标记类任务（不在 COMPUTE_TASK_INFO 里）单独捞一次，用来标边界
    marker_rows = []
    try:
        cur = con.execute(
            "SELECT t.startNs, t.endNs, t.streamId, s.value FROM TASK t "
            "JOIN STRING_IDS s ON s.id = t.taskType "
            "WHERE s.value LIKE 'MODEL_WAIT%' OR s.value LIKE 'EVENT%' "
            "OR s.value LIKE 'NOTIFY%' OR s.value LIKE 'MEM%ASYNC'")
        marker_rows = cur.fetchall()
    except Exception:
        marker_rows = []
    # host 侧 forward 边界停顿（profiler sync 补偿那一类）：PYTORCH_API 的相邻间隔
    host_raw = []          # [(startNs, gapNs)]
    try:
        cur = con.execute(
            "SELECT startNs, endNs FROM PYTORCH_API ORDER BY startNs")
        prev_end = None
        for s, e in cur:
            s, e = float(s), float(e)          # PYTORCH_API 的时间列是 TEXT
            if prev_end is not None and (s - prev_end) > 2000 * 1000:   # >2ms
                host_raw.append((prev_end, s - prev_end))
            prev_end = max(prev_end, e) if prev_end is not None else e
    except Exception:
        host_raw = []
    overlap = None
    try:
        c = con.execute("SELECT startNs, endNs, type FROM OVERLAP_ANALYSIS")
        comp, free = [], []
        for s, e, ty in c:
            if ty == 0:
                comp.append((s, e))
            elif ty == 3:
                free.append((s, e))
        if comp:
            overlap = {"compute": comp, "free": free}
    except Exception:
        overlap = None
    con.close()
    rows = raw + marker_rows
    t0 = min(r[0] for r in rows)
    tasks = []
    for s, e, st, name in rows:
        tasks.append({"start": (s - t0) / 1000.0, "end": (e - t0) / 1000.0,
                      "stream": int(st), "name": name, "fam": family_of(name),
                      "marker": is_marker(name)})
    if overlap:
        overlap = {k: [((s - t0) / 1000.0, (e - t0) / 1000.0) for s, e in v]
                   for k, v in overlap.items()}
    tasks.sort(key=lambda x: x["start"])
    return tasks, overlap, {"source": os.path.basename(path), "n_task": len(raw),
                            "host_gaps_ms": [round(g / 1e6, 3) for _, g in host_raw]}


def load_from_csv(path):
    rows = []
    with open(path, newline="", errors="ignore") as fh:
        for r in csv.DictReader(fh):
            try:
                st = float(str(r["Start Time(us)"]).strip())
                du = float(str(r["Duration(us)"]).strip())
            except (KeyError, ValueError):
                continue
            rows.append((st, st + du, 0, r.get("Name", "")))
    if not rows:
        return None, None, {"error": "empty csv"}
    t0 = min(r[0] for r in rows)
    tasks = [{"start": s - t0, "end": e - t0, "stream": 0, "name": n,
              "fam": family_of(n), "marker": is_marker(n)} for s, e, _, n in rows]
    tasks.sort(key=lambda x: x["start"])
    return tasks, None, {"source": os.path.basename(path), "n_task": len(tasks),
                         "warn": "kernel_details.csv 无 streamId，尾段不可分"}


# ── 区间工具 ────────────────────────────────────────────────────────────────
def union_len(iv):
    if not iv:
        return 0.0
    iv = sorted(iv)
    tot, cs, ce = 0.0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > ce:
            tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + (ce - cs)


def clip(iv, a, b):
    return [(max(s, a), min(e, b)) for s, e in iv if min(e, b) > max(s, a)]


def gaps_in(a, b, busy):
    """[a,b] 内被 busy 覆盖之外的区间 → [(s,e)]"""
    iv = sorted(clip(busy, a, b))
    out, cur = [], a
    for s, e in iv:
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < b:
        out.append((cur, b))
    return out


# ── 尾段 / forward 切分 ─────────────────────────────────────────────────────
def candidate_tails(tasks, idle_ms, min_k, min_tail_ms):
    """扫出所有「侧流 replay 簇」→ [(start, end, stream)]，按 end 升序。

    判定（全部满足）：
      · 在**非主算流**上（主算流 = 真实 kernel 总时长最大的那条流）；
      · 是该流上的一个连续簇（簇内相邻 kernel 间隔 < idle_ms）；
      · 前面在该流上静默 ≥ idle_ms（⇒ 不是常驻并行流）；
      · kernel 数 ≥ min_k、簇时长 ≥ min_tail_ms；
      · **簇里含层标记算子**（`mega_moe` / `HcPre`）——graph replay 放的是「层 20..39」，
        所以必然带层标记；这条把「engram 侧流在 forward 开头的取表突发」排除掉
        （那种簇只有 EngramFetch/Cast，没有层标记）。若全 trace 找不到带层标记的簇，
        退回不要求层标记（不认识层结构的模型）。
    """
    real = [t for t in tasks if not t["marker"]]
    if not real:
        return []
    by_stream = collections.defaultdict(list)
    for t in real:
        by_stream[t["stream"]].append(t)
    if len(by_stream) < 2:
        return []
    tot = {s: sum(t["end"] - t["start"] for t in v) for s, v in by_stream.items()}
    main = max(tot, key=tot.get)
    gap = idle_ms * 1000.0

    def n_layers(c):
        return sum(1 for t in c if any(rx.search(t["name"]) for rx in LAYER_MARKERS))

    cands = []
    for st, v in by_stream.items():
        if st == main:
            continue
        v.sort(key=lambda x: x["start"])
        clusters, cur = [], [v[0]]
        for prev, nxt in zip(v, v[1:]):
            if nxt["start"] - prev["end"] > gap:
                clusters.append(cur)
                cur = [nxt]
            else:
                cur.append(nxt)
        clusters.append(cur)
        prev_end = None
        for c in clusters:
            s, e = c[0]["start"], max(x["end"] for x in c)
            if (len(c) >= min_k and (e - s) >= min_tail_ms * 1000.0
                    and (prev_end is None or s - prev_end >= gap)):
                cands.append((s, e, st, len(c), n_layers(c)))
            prev_end = e
    if any(c[4] > 0 for c in cands):
        cands = [c for c in cands if c[4] > 0]
    cands.sort(key=lambda x: x[1])
    # 同一结束时刻的多条流只留 kernel 最多的那条
    merged = []
    for c in cands:
        if merged and abs(c[1] - merged[-1][1]) < 100.0:
            if c[3] > merged[-1][3]:
                merged[-1] = c
        else:
            merged.append(c)
    return [(s, e, st) for s, e, st, _, _ in merged]


def build_passes(tasks, idle_ms, min_k, min_tail_ms=1.0):
    """→ [{'start','end','tail','head_end'}...]；**以尾段结束为 forward 边界**。"""
    real = [t for t in tasks if not t["marker"]]
    if not real:
        return []
    lo = min(t["start"] for t in real)
    hi = max(t["end"] for t in real)
    tails = candidate_tails(tasks, idle_ms, min_k, min_tail_ms)
    if not tails:
        # 没有 graph replay（纯 eager）：整条 trace 当一个窗口，不硬切
        return [{"start": lo, "end": hi, "tail": None, "head_end": hi}]
    bounds = [lo] + [t[1] for t in tails]
    out = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        tl = tails[i]                      # 该窗口末尾的尾段
        out.append({"start": a, "end": b, "tail": tl, "head_end": tl[0]})
    return out


# ── 单个 rank 的完整分析 ────────────────────────────────────────────────────
def analyse_rank(dirpath, args):
    db = _db_path(dirpath)
    if db:
        tasks, overlap, meta = load_from_db(db)
    else:
        csvp = os.path.join(dirpath, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
        tasks, overlap, meta = load_from_csv(csvp) if os.path.exists(csvp) else (None, None, {})
    if not tasks:
        return None
    passes = build_passes(tasks, args.tail_idle_ms, args.tail_min_kernels,
                          args.tail_min_ms)
    real = [t for t in tasks if not t["marker"]]
    comp_iv = overlap["compute"] if overlap else [(t["start"], t["end"]) for t in tasks]

    res = {"dir": dirpath, "meta": meta, "passes": []}
    for i, p in enumerate(passes):
        a, b = p["start"], p["end"]
        if b - a < args.min_pass_ms * 1000.0:
            continue
        tail = p["tail"]
        head_end = tail[0] if tail else b
        head_span = head_end - a
        tail_span = (tail[1] - tail[0]) if tail else 0.0
        head_busy = union_len(clip(comp_iv, a, head_end))
        head_free = gaps_in(a, head_end, comp_iv)

        # 算子族：head 内、按 kernel 裁剪求和
        fam_t = collections.Counter()
        fam_n = collections.Counter()
        for t in real:
            if t["start"] >= head_end or t["end"] <= a:
                continue
            d = min(t["end"], head_end) - max(t["start"], a)
            if d <= 0:
                continue
            fam_t[t["fam"]] += d
            fam_n[t["fam"]] += 1

        # 逐 kernel 名（head 内）——用来找「同一个算子被调太多次/搬太多」
        name_t = collections.Counter()
        name_n = collections.Counter()
        for t in real:
            if t["start"] >= head_end or t["end"] <= a:
                continue
            d = min(t["end"], head_end) - max(t["start"], a)
            if d > 0:
                name_t[t["name"]] += d
                name_n[t["name"]] += 1

        # 尾段算子族（同一口径，窗口 = [tail_start, tail_end]）
        tail_fam = collections.Counter()
        tail_n = collections.Counter()
        tail_name_t = collections.Counter()
        tail_name_n = collections.Counter()
        if tail:
            ts, te = tail[0], tail[1]
            for t in real:
                if t["start"] >= te or t["end"] <= ts:
                    continue
                d = min(t["end"], te) - max(t["start"], ts)
                if d > 0:
                    tail_fam[t["fam"]] += d
                    tail_n[t["fam"]] += 1
                    tail_name_t[t["name"]] += d
                    tail_name_n[t["name"]] += 1

        # 空泡 top：>50µs
        big = [(s, e) for s, e in head_free if (e - s) >= 50.0]
        big.sort(key=lambda x: x[1] - x[0], reverse=True)
        bubbles = []
        for s, e in big[:args.top_bubbles]:
            before = [t for t in real if t["end"] <= s + 1e-6]
            after = [t for t in real if t["start"] >= e - 1e-6]
            before.sort(key=lambda x: x["end"])
            after.sort(key=lambda x: x["start"])
            moe_before = [t for t in real if LAYER_MARKERS[0].search(t["name"])
                          and a <= t["start"] < s]      # 只数本 forward 内的层
            bubbles.append({
                "start_ms": (s - a) / 1000.0, "dur_ms": (e - s) / 1000.0,
                "layer": len(moe_before),
                "after": before[-1]["name"][:48] if before else "-",
                "after_fam": before[-1]["fam"] if before else "-",
                "before": after[0]["name"][:48] if after else "-",
                "before_fam": after[0]["fam"] if after else "-",
            })

        res["passes"].append({
            "idx": i, "start_ms": a / 1000.0, "end_ms": b / 1000.0,
            "span_ms": (b - a) / 1000.0,
            "tail_ms": tail_span / 1000.0,
            "tail_stream": tail[2] if tail else None,
            "head_ms": head_span / 1000.0,
            "head_busy_ms": head_busy / 1000.0,
            "head_idle_ms": (head_span - head_busy) / 1000.0,
            "no_tail": tail is None,
            "head_sum_kernel_ms": sum(fam_t.values()) / 1000.0,
            "n_kernel": sum(fam_n.values()),
            "families": {k: {"ms": v / 1000.0, "n": fam_n[k]} for k, v in fam_t.items()},
            "tail_families": {k: {"ms": v / 1000.0, "n": tail_n[k]}
                              for k, v in tail_fam.items()},
            "by_name": {k: {"ms": v / 1000.0, "n": name_n[k]}
                        for k, v in name_t.items()},
            "tail_by_name": {k: {"ms": v / 1000.0, "n": tail_name_n[k]}
                             for k, v in tail_name_t.items()},
            "bubbles": bubbles,
            "n_free_gt50us": len(big),
            "free_gt50us_ms": sum(e - s for s, e in big) / 1000.0,
        })
    if not res["passes"]:
        return None
    return res


# ── 汇总（跨 rank / 跨 forward 取中位）──────────────────────────────────────
def agg(values, stat):
    if not values:
        return float("nan")
    if stat == "mean":
        return statistics.fmean(values)
    if stat == "min":
        return min(values)
    if stat == "max":
        return max(values)
    return statistics.median(values)


def summarise(per_rank, args):
    # 每个 rank：剔除首个 forward（捕获后首回放），取稳态 forward 的中位
    steady = {}
    for label, r in per_rank.items():
        ps = r["passes"]
        use = ps[1:] if (args.settle and len(ps) > 1) else ps
        if args.passes:
            want = set(args.passes)
            use = [p for p in use if p["idx"] in want] or use
        steady[label] = use
    if not steady:
        return None
    rank_ids = sorted(steady)
    n = len(rank_ids)

    def med(key, idx=None):
        vals = []
        for lbl in rank_ids:
            ps = steady[lbl]
            if idx is not None:
                ps = [p for p in ps if p["idx"] == idx]
            if ps:
                vals.append(statistics.median([p[key] for p in ps]))
        return agg(vals, args.stat)

    def fam_agg(key):
        fams = collections.defaultdict(list)
        fam_n = collections.defaultdict(list)
        for lbl in rank_ids:
            per_fam = collections.defaultdict(list)
            per_n = collections.defaultdict(list)
            for p in steady[lbl]:
                for f, d in p[key].items():
                    per_fam[f].append(d["ms"])
                    per_n[f].append(d["n"])
            for f, v in per_fam.items():
                fams[f].append(statistics.median(v))
                fam_n[f].append(statistics.median(per_n[f]))
        return {f: {"ms": agg(v, "median"), "n": agg(fam_n[f], "median")}
                for f, v in fams.items()}


    # 空泡：逐 rank 取「最接近中位窗口」的那个 forward，再按位置聚类求中位
    raw = []
    for lbl in rank_ids:
        ps = sorted(steady[lbl], key=lambda p: p["span_ms"])
        if ps:
            raw.extend(ps[len(ps) // 2]["bubbles"])
    groups = collections.defaultdict(list)
    for b in raw:
        key = (round(b["start_ms"] / 2.0) * 2, b["after_fam"], b["before_fam"])
        groups[key].append(b)
    bubs = []
    for (pos, af, bf), v in groups.items():
        bubs.append({
            "start_ms": statistics.median([x["start_ms"] for x in v]),
            "dur_ms": statistics.median([x["dur_ms"] for x in v]),
            "layer": int(statistics.median([x["layer"] for x in v])),
            "after": v[0]["after"], "after_fam": af,
            "before": v[0]["before"], "before_fam": bf,
            "n_rank": len(v),
        })
    bubs.sort(key=lambda x: -x["dur_ms"])

    hg = [v["meta"].get("host_gaps_ms") or [] for v in per_rank.values()]
    host = {
        "n": statistics.median([len(x) for x in hg]) if hg else 0,
        "max_ms": statistics.median([max(x) if x else 0.0 for x in hg]) if hg else 0.0,
        "sum_ms": statistics.median([sum(x) for x in hg]) if hg else 0.0,
    }
    return {
        "n_rank": n, "ranks": rank_ids, "host_boundary": host,
        "span_ms": med("span_ms"), "tail_ms": med("tail_ms"),
        "head_ms": med("head_ms"), "head_busy_ms": med("head_busy_ms"),
        "head_idle_ms": med("head_idle_ms"),
        "head_sum_kernel_ms": med("head_sum_kernel_ms"),
        "n_kernel": med("n_kernel"),
        "n_free_gt50us": med("n_free_gt50us"),
        "free_gt50us_ms": med("free_gt50us_ms"),
        "families": fam_agg("families"),
        "tail_families": fam_agg("tail_families"),
        "by_name": fam_agg("by_name"),
        "bubbles": bubs[:args.top_bubbles],
        "per_rank": {k: [{"idx": p["idx"], "span_ms": p["span_ms"],
                          "tail_ms": p["tail_ms"], "head_ms": p["head_ms"],
                          "head_busy_ms": p["head_busy_ms"],
                          "head_idle_ms": p["head_idle_ms"]}
                         for p in v] for k, v in steady.items()},
    }


# ── 输出 ────────────────────────────────────────────────────────────────────
def render(s, title, args):
    L = []
    pct = lambda a, b: (100.0 * a / b) if b else 0.0
    L.append(f"### {title}")
    L.append("")
    L.append(f"稳态 forward（{s['n_rank']} rank {args.stat}，剔除首个 forward）")
    L.append("")
    L.append("| 项 | ms | 占窗口 |")
    L.append("|---|---:|---:|")
    L.append(f"| **窗口（forward）** | **{s['span_ms']:.2f}** | 100.0% |")
    L.append(f"| 尾段 graph replay | {s['tail_ms']:.2f} | {pct(s['tail_ms'], s['span_ms']):.1f}% |")
    L.append(f"| 头段（eager） | {s['head_ms']:.2f} | {pct(s['head_ms'], s['span_ms']):.1f}% |")
    L.append(f"| ├ 头段设备忙 | {s['head_busy_ms']:.2f} | "
             f"{pct(s['head_busy_ms'], s['head_ms']):.1f}% of 头段 |")
    L.append(f"| └ 头段空泡 | {s['head_idle_ms']:.2f} | "
             f"{pct(s['head_idle_ms'], s['head_ms']):.1f}% of 头段 |")
    L.append("")
    L.append("头段算子族（按 kernel 时长求和；>头段长 = 有跨流重叠）")
    L.append("")
    L.append("| 族 | ms | 占头段 | 占头段忙 | 调用数 |")
    L.append("|---|---:|---:|---:|---:|")
    tot = s["head_sum_kernel_ms"] or 1.0
    for f, d in sorted(s["families"].items(), key=lambda kv: -kv[1]["ms"]):
        v = d["ms"]
        L.append(f"| {f} | {v:.2f} | {pct(v, s['head_ms']):.1f}% | {pct(v, tot):.1f}% "
                 f"| {d['n']:.0f} |")
    L.append(f"| **合计（含跨流重叠）** | **{tot:.2f}** | {pct(tot, s['head_ms']):.1f}% "
             f"| 100% | {s['n_kernel']:.0f} |")
    L.append("")
    ov = tot - s["head_busy_ms"]
    if ov > 0.05:
        L.append(f"> 合计 − 头段忙 = **{ov:.2f}ms** 是跨流重叠（engram 侧流等并行部分）；"
                 f"分族占比请按「占头段忙」列读，别按 ms 直接相加。")
        L.append("")
    if s.get("tail_families"):
        L.append("尾段（graph replay）算子族")
        L.append("")
        L.append("| 族 | ms | 占尾段 | 调用数 |")
        L.append("|---|---:|---:|---:|")
        tt = sum(v["ms"] for v in s["tail_families"].values()) or 1.0
        for f, d in sorted(s["tail_families"].items(), key=lambda kv: -kv[1]["ms"])[:8]:
            L.append(f"| {f} | {d['ms']:.2f} | {pct(d['ms'], s['tail_ms']):.1f}% "
                     f"| {d['n']:.0f} |")
        L.append(f"| **合计（含重叠）** | **{tt:.2f}** | {pct(tt, s['tail_ms']):.1f}% | — |")
        L.append("")
    L.append(f"头段 >50µs 空泡：{s['n_free_gt50us']:.0f} 个 / 共 {s['free_gt50us_ms']:.2f}ms")
    L.append("")
    hb = s.get("host_boundary") or {}
    if hb.get("n"):
        L.append(f"> **forward 边界（已剔除，不计入上表）**：host 侧 >2ms 停顿 "
                 f"{hb['n']:.0f} 个 / 合计 {hb['sum_ms']:.2f}ms / 最大 {hb['max_ms']:.2f}ms。"
                 f"这是 profiler 轮的 sync 补偿（`runner.py:1435-1443` 的 "
                 f"`_gt_sync_default`；creative-lead 2026-10-26 创意 3 已定性），"
                 f"生产口径没有这块。设备时间线在窗口内连续，故它不进入上面的分解。")
        L.append("")
    if args.by_name and s.get("by_name"):
        L.append(f"头段单算子 top{args.by_name}")
        L.append("")
        L.append("| kernel | ms | 占头段忙 | 调用数 | 单次 µs |")
        L.append("|---|---:|---:|---:|---:|")
        hb = s["head_busy_ms"] or 1.0
        for nm, d in sorted(s["by_name"].items(), key=lambda kv: -kv[1]["ms"])[:args.by_name]:
            n = d["n"] or 1
            L.append(f"| `{nm[:64]}` | {d['ms']:.2f} | {pct(d['ms'], hb):.1f}% "
                     f"| {d['n']:.0f} | {1000.0 * d['ms'] / n:.1f} |")
        L.append("")
    if s.get("per_rank"):
        L.append("逐 rank（稳态各 forward 取中位；剔除首个 forward）")
        L.append("")
        L.append("| rank | forward | 窗口 ms | 尾段 ms | 头段 ms | 头段忙 ms | 头段空泡 ms |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for r, ps in s["per_rank"].items():
            if not ps:
                continue
            m = lambda k: statistics.median([p[k] for p in ps])
            L.append(f"| {r} | {len(ps)} | {m('span_ms'):.2f} | {m('tail_ms'):.2f} "
                     f"| {m('head_ms'):.2f} | {m('head_busy_ms'):.2f} "
                     f"| {m('head_idle_ms'):.2f} |")
        L.append("")
    if s["bubbles"]:
        L.append(f"头段空泡 top{len(s['bubbles'])}（跨 {s['n_rank']} rank 聚类，层号按 MoE 调用序号推）")
        L.append("")
        L.append("| # | 层 | 起(ms) | 时长(ms) | rank 数 | 前一个算子 | 后一个算子 |")
        L.append("|---:|---:|---:|---:|---:|---|---|")
        for i, b in enumerate(s["bubbles"], 1):
            L.append(f"| {i} | {b['layer']} | {b['start_ms']:.2f} | {b['dur_ms']:.3f} "
                     f"| {b['n_rank']} | {b['after_fam']}: `{b['after']}` "
                     f"| {b['before_fam']}: `{b['before']}` |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--ranks", default="", help="逗号分隔的 rank 序号（默认全部）")
    ap.add_argument("--passes", default="", help="只看这些 forward 序号（默认稳态全部）")
    ap.add_argument("--settle", type=int, default=1,
                    help="剔除前 N 个 forward（默认 1 = 捕获后首回放）")
    ap.add_argument("--stat", default="median", choices=["median", "mean", "min", "max"])
    ap.add_argument("--tail-idle-ms", type=float, default=5.0,
                    help="侧流判定尾段所需的「前面静默」阈值（默认 5ms）")
    ap.add_argument("--tail-min-kernels", type=int, default=10)
    ap.add_argument("--tail-min-ms", type=float, default=1.0,
                    help="尾段簇的最短时长（默认 1ms）")
    ap.add_argument("--min-pass-ms", type=float, default=1.0)
    ap.add_argument("--top-bubbles", type=int, default=6)
    ap.add_argument("--by-name", type=int, default=0, metavar="N",
                    help="额外打印头段内耗时前 N 的单个 kernel 名（0=不打印）")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    dirs = find_rank_dirs(args.trace_dir)
    if not dirs:
        sys.exit(f"[err] 在 {args.trace_dir} 下没找到 *_ascend_pt/ASCEND_PROFILER_OUTPUT")
    if args.ranks:
        want = {x.strip() for x in args.ranks.split(",") if x.strip()}
        sel = [d for d in dirs if d[0] in want or d[0].split("_")[-1] in want]
        dirs = sel or dirs
    per_rank = {}
    for i, (label, d) in enumerate(dirs):
        r = analyse_rank(d, args)
        if r is None:
            print(f"[warn] {label}: 无可用数据", file=sys.stderr)
            continue
        if r["passes"] and r["passes"][0]["no_tail"]:
            print(f"[warn] {label}: 未识别到 graph replay 尾段（纯 eager trace？）"
                  f"→ tail 记 0，整条 trace 当一个窗口", file=sys.stderr)
        per_rank[label] = r
        print(f"[ok] r{i} (pid {label}): {len(r['passes'])} forward, "
              f"src={r['meta'].get('source')}", file=sys.stderr)
    if not per_rank:
        sys.exit("[err] 没有任何 rank 分析成功")
    args.passes = [int(x) for x in args.passes.split(",") if x.strip()] if args.passes else None
    s = summarise(per_rank, args)
    print(render(s, os.path.basename(os.path.abspath(args.trace_dir)), args))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": s,
                       "raw": {k: v["passes"] for k, v in per_rank.items()},
                       "meta": {k: v["meta"] for k, v in per_rank.items()}},
                      f, ensure_ascii=False, indent=1)
        print(f"\n[json] {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()

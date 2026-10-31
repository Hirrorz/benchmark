#!/usr/bin/env python3
"""两个 trace 的算子集合/次数差异表 —— 直接回答"这两个 run 差在哪"。

用法：
  python3 tools/diff_traces.py <traceA> <traceB> [--passes N | --passes-a N --passes-b N] [--top 40]

输入：
  · trace 目录（`<host>_<pid>_<ts>_ascend_pt`，自动找 ASCEND_PROFILER_OUTPUT/kernel_details.csv）；
  · 该目录的上一层（若下面只有一个 trace 目录也能自己找到）；
  · 或直接给 kernel_details.csv 的路径。

归一（`--passes`，兼容旧名；对外描述用 forward）：
  trace 里有多个完整前向时，按次数除以"前向数"再比。例：bench 一次 prof 里 4 个 Step
  → `--passes 4`。★ 注意：runner（mini）**每层**调一次 profiler.step()，所以 mini 的
  trace 里有 40 个 Step，但它只跑了一个前向 → forward 数=1，别拿 Step 数当 forward 数。
  `--passes-a/--passes-b` 可分别指定两侧（默认都用 --passes，再默认 1）。

算子名归一（照 tools/align_ops.py 的 ALIAS 思路）：
  mangled C++ 名（`_ZN9deep_gemm…mega_moe_impl…`、`_Z30kernel_MqsmlaKernel…`、
  `_Z44kernel_indexer_prologue_qw_kernel…`）与短名归到同一规范名；
  **不同实现的内核保持分开**（例如 `MhcPost` vs `mhc_post_fwd_asc_kernel`）——
  那正是差异表要暴露的东西。

输出：
  ① 两侧各多少行内核 / 多少种算子 / 各几个 Step；
  ② 只在 A 有 / 只在 B 有（集合差）；
  ③ 共有算子按 |A−B| 降序（次数差表的最上一行就是"差在哪"的答案）；
  ④ 汇总行数差。
"""
import argparse
import collections
import csv
import glob
import os
import re
import sys

# ── 规范名表：(规范名, 匹配正则)。与 align_ops.py 同思路，但这里面对的是
#    kernel_details.csv 的 Type 列（两侧都是 trace），所以顺序敏感的模式放前面。──
ALIAS = [
    ("mega_moe_impl",                 r"deep_gemm.*mega_moe|mega_moe_impl"),
    ("MqsmlaKernel",                  r"MqsmlaKernel"),
    ("mqsmla_metadata_kernel",        r"mqsmla_metadata_kernel"),
    ("metadata_kernel",               r"^metadata_kernel"),
    ("indexer_prologue_qw_kernel",    r"indexer_prologue_qw_kernel"),
    ("QuantSparseLightningIndexer",   r"QuantSparseLightningIndexer"),
    ("QuantLightningIndexer",         r"QuantLightningIndexer"),
    ("_FusedPostScatterKernel",       r"_FusedPostScatterKernel"),
    ("hcom_alltoallv_",               r"hcom_alltoallv_"),          # 必须先于 alltoall
    ("hcom_alltoall_",                r"hcom_alltoall_$"),
    ("moe_topk_gate_forward_kernel",  r"moe_topk_gate"),
    ("EngramFetchWait",               r"^EngramFetchWait$"),
    ("EngramFetch",                   r"^EngramFetch$"),
    ("engram_fused_weight_kernel",    r"engram_fused_weight_kernel"),
    ("engram_gate_fwd_kernel",        r"engram_gate_fwd_kernel"),
    ("MoeInitRoutingV3",              r"MoeInitRoutingV3"),
    ("MoeReRouting",                  r"MoeReRouting"),
    ("MoeFinalizeRoutingV2",          r"MoeFinalizeRoutingV2"),
]


def demangle_short(s):
    """从 mangled C++ 名里抠出可读标识符（不是完整 demangle，够归类用）。

    `_Z30kernel_MqsmlaKernel___call__0…` → MqsmlaKernel
    `_ZN9deep_gemm8mega_moe13mega_moe_impl…` → mega_moe_impl
    匹配不出来的原样返回。
    """
    if not s.startswith("_Z"):
        return s
    i = 2
    if i < len(s) and s[i] == "N":                    # 嵌套名：末段 = 函数名
        i += 1
        parts = []
        while i < len(s) and s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            n = int(s[i:j])
            parts.append(s[j:j + n])
            i = j + n
        name = parts[-1] if parts else ""
    else:                                             # _Z<len><name>
        j = i
        while j < len(s) and s[j].isdigit():
            j += 1
        n = int(s[i:j]) if j > i else 0
        name = s[j:j + n] if n else ""
    name = re.sub(r"^kernel_", "", name)              # kernel_XxxKernel → XxxKernel
    name = name.split("__", 1)[0]                     # 砍掉 __call__ 及参数尾巴
    return name or s


def canon(raw):
    if not raw or raw == "-":
        return None
    for name, pat in ALIAS:
        if re.search(pat, raw):
            return name
    return demangle_short(raw)


def find_csv(path):
    """trace 目录 / 其父目录 / csv 文件 → kernel_details.csv 的路径。"""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        cand = os.path.join(path, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
        if os.path.isfile(cand):
            return cand
        subs = sorted(glob.glob(os.path.join(path, "*", "ASCEND_PROFILER_OUTPUT",
                                             "kernel_details.csv")))
        if len(subs) == 1:
            return subs[0]
        if len(subs) > 1:
            sys.exit(f"{path} 下有 {len(subs)} 个 trace，请直接指到某一个：\n  "
                     + "\n  ".join(subs[:6]) + ("\n  …" if len(subs) > 6 else ""))
        sys.exit(f"{path} 里找不到 ASCEND_PROFILER_OUTPUT/kernel_details.csv")
    sys.exit(f"路径不存在：{path}")


def load(csv_path, passes):
    """→ (归一后的 {算子: 次数}, 原始行数, Step 数)。"""
    rows = list(csv.DictReader(open(csv_path, errors="ignore")))
    if rows and "Type" not in rows[0] and "Name" not in rows[0]:
        sys.exit(f"{csv_path} 不像 kernel_details.csv（没有 Type/Name 列）")
    c = collections.Counter()
    for r in rows:
        t = (r.get("Type") or "").strip()
        if t in ("", "-"):
            t = (r.get("Name") or "").strip()          # 个别算子 Type 为空，退回 Name
        k = canon(t)
        if k:
            c[k] += 1
    steps = len({r.get("Step Id", "") for r in rows})
    if passes > 1:
        c = collections.Counter({k: v / passes for k, v in c.items()})
    return c, len(rows), steps


def fnum(v):
    return f"{int(round(v))}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"


def show_table(rows, top):
    for k, va, vb, d in rows[:top]:
        print(f"  {k:<44}{fnum(va):>10}{fnum(vb):>10}{('+' if d > 0 else '') + fnum(d):>10}")
    if len(rows) > top:
        print(f"  …另有 {len(rows) - top} 种未显示（--top 调整）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="trace A（目录或 kernel_details.csv）")
    ap.add_argument("b", help="trace B（目录或 kernel_details.csv）")
    ap.add_argument("--passes", type=int, default=1, help="两侧共同的前向数（默认 1）")
    ap.add_argument("--passes-a", type=int, default=0, help="只覆盖 A 的前向数")
    ap.add_argument("--passes-b", type=int, default=0, help="只覆盖 B 的前向数")
    ap.add_argument("--top", type=int, default=40, help="每张表最多显示多少行")
    a = ap.parse_args()

    pa = a.passes_a or a.passes
    pb = a.passes_b or a.passes
    ca, ra, sa = load(find_csv(a.a), pa)
    cb, rb, sb = load(find_csv(a.b), pb)

    print("== diff_traces：两个 run 的算子集合/次数差异 ==")
    print(f"A = {os.path.basename(os.path.normpath(a.a)):<60} {ra} 行内核 / {len(ca)} 种算子"
          f" / {sa} Step（按 {pa} pass 归一）")
    print(f"B = {os.path.basename(os.path.normpath(a.b)):<60} {rb} 行内核 / {len(cb)} 种算子"
          f" / {sb} Step（按 {pb} pass 归一）")
    print("（注：Step≠pass —— runner 每层调一次 profiler.step()，mini 的 40 Step 仍是 1 个前向）")

    only_a = [(k, v) for k, v in ca.items() if k not in cb]
    only_b = [(k, v) for k, v in cb.items() if k not in ca]
    shared = [(k, ca[k], cb[k], ca[k] - cb[k]) for k in ca if k in cb]
    shared.sort(key=lambda x: (-abs(x[3]), -max(x[1], x[2])))

    print(f"\n── 只在 A 有（{len(only_a)} 种）" + (" —— 无" if not only_a else ""))
    for k, v in sorted(only_a, key=lambda x: -x[1])[:a.top]:
        print(f"  {k:<44}{fnum(v):>10}")
    if len(only_a) > a.top:
        print(f"  …另有 {len(only_a) - a.top} 种未显示（--top 调整）")

    print(f"\n── 只在 B 有（{len(only_b)} 种）" + (" —— 无" if not only_b else ""))
    for k, v in sorted(only_b, key=lambda x: -x[1])[:a.top]:
        print(f"  {k:<44}{fnum(v):>10}")
    if len(only_b) > a.top:
        print(f"  …另有 {len(only_b) - a.top} 种未显示（--top 调整）")

    ndiff = sum(1 for _, _, _, d in shared if abs(d) > 1e-9)
    print(f"\n── 共有算子（{len(shared)} 种，其中 {ndiff} 种次数不一致；按 |A−B| 降序）"
          + (" —— 全部为 0 ✓" if ndiff == 0 else ""))
    print(f"  {'算子':<44}{'A':>10}{'B':>10}{'A−B':>10}")
    show_table(shared, a.top)

    ta, tb = sum(ca.values()), sum(cb.values())
    print(f"\n汇总：A {len(ca)} 种 / {fnum(ta)} 个内核（归一后）｜"
          f"B {len(cb)} 种 / {fnum(tb)} 个内核（归一后）｜"
          f"只在 A {len(only_a)} 种、只在 B {len(only_b)} 种、共有次数差 {ndiff} 种")


if __name__ == "__main__":
    main()

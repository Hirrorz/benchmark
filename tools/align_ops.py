"""把 dry-run 的算子直方图与**真实 trace** 对齐（M1 的核心验收工具）。

用法：
  python3 tools/align_ops.py --ours /tmp/dry.json \
      --ref  <...>/mindstudio_profiler_output/op_statistic_*.csv --ref-passes 4 [--top 30]
  python3 tools/align_ops.py --ours /tmp/dry.json \
      --ref  <...>/ASCEND_PROFILER_OUTPUT/kernel_details.csv --ref-passes 4   # 旧版导出
说明：--ref-passes = 参考 trace 里包含几个完整前向（bench 的 --hit-inner-reps/loops）；
工具会把参考侧除以它，得到"单 pass 计数"再与我们的比对。

两种参考格式（2026-11-01）：
  · 新版 msprof 导出（现行口径，`msprof --export=on`）只有
    `mindstudio_profiler_output/op_statistic_*.csv`，列是 `OP Type` + `Count`
    （每行已按算子聚合，`Count` 是该算子的次数）——**没有** `kernel_details.csv`。
  · 旧版导出是 `ASCEND_PROFILER_OUTPUT/kernel_details.csv`，列是 `Type`，一行一次调用。
  以前只认 `Type` 列 ⇒ 喂新版导出时**每行都取到空串**，静默得出"参考全 0"，
  看起来像"我们的算子比官方多 250 倍"。现在两种都认，且认不出列名时**报错**而不是静默。
"""
import argparse
import collections
import csv
import json
import re
import sys


def our_hist(path):
    d = json.load(open(path))
    return {k: int(v) for k, v in d["hist"].items()}, d["meta"]


def ref_hist(csv_path, passes):
    """→ {算子名: 单 pass 次数}。兼容 op_statistic（OP Type+Count）与 kernel_details（Type）。"""
    c = collections.Counter()
    with open(csv_path, errors="ignore") as fh:
        rd = csv.DictReader(fh)
        cols = set(rd.fieldnames or [])
        # 新版：op_statistic_*.csv（每行一个算子，Count 是次数）
        name_col = next((x for x in ("OP Type", "Op Type", "Type") if x in cols), None)
        if name_col is None:
            raise SystemExit(
                f"✗ 认不出参考 CSV 的算子名列（{csv_path}）：列 = {sorted(cols)[:12]}\n"
                f"  期望 op_statistic_*.csv（OP Type/Count）或 kernel_details.csv（Type）")
        cnt_col = "Count" if "Count" in cols else None
        n_rows = 0
        for r in rd:
            t = (r.get(name_col) or "").strip()
            if not t or t == "-":
                continue
            n = 1
            if cnt_col:
                try:
                    n = int(float(r.get(cnt_col) or 1))
                except (TypeError, ValueError):
                    n = 1
            c[t] += n
            n_rows += 1
    if n_rows == 0:
        raise SystemExit(f"✗ 参考 CSV 里没有有效算子行（{csv_path}）：列 = {sorted(cols)[:12]}")
    # 归一到单 pass；类型名做模糊化（trace 里是 mangled 名，我们用的是短名）
    return {k: v / max(1, passes) for k, v in c.items()}


ALIAS = [  # (我们的短名, trace 名匹配正则)
    ("mega_moe_impl", r"mega_moe_impl|_ZN9deep_gemm8mega_moe"),
    ("MqsmlaKernel", r"MqsmlaKernel"),
    ("QuantSparseLightningIndexer", r"QuantSparseLightningIndexer"),
    ("QuantLightningIndexer", r"QuantLightningIndexer"),
    ("indexer_prologue_qw_kernel", r"indexer_prolog"),
    # 2026-11-01：厂商 trace 里的名字是 `MhcPost`（大写、无下划线），旧正则 r"mhc_post"
    # 匹配不到 ⇒ 对齐表把 80 次报成 "参考 0 / 比值 inf"（假差异）。两种写法都认。
    ("mhc_post_fwd_asc_kernel", r"mhc_post|MhcPost"),
    ("HcPre", r"^HcPre$"),
    ("RmsNorm", r"^RmsNorm$"),
    ("QuantBatchMatmulV3", r"^QuantBatchMatmulV3$"),
    ("GroupedMatmul", r"^GroupedMatmul$"),
    ("SwigluGroupQuant", r"^SwigluGroupQuant$"),
    ("KvCompressEpilogV2", r"^KvCompressEpilogV2$"),
    ("InplacePartialRotaryMul", r"InplacePartialRotaryMul"),
    ("MoeInitRoutingV3", r"MoeInitRoutingV3"),
    ("MoeReRouting", r"MoeReRouting"),
    ("MoeFinalizeRoutingV2", r"MoeFinalizeRoutingV2"),
    ("hcom_alltoallv_", r"hcom_alltoallv_"),
    ("hcom_alltoall_", r"hcom_alltoall_$"),
    ("DynamicMxQuant", r"^DynamicMxQuant$"),
    ("FusedMatMul", r"^FusedMatMul$"),
    ("TransposeQuantBatchMatMul", r"^TransposeQuantBatchMatMul$"),
    ("moe_topk_gate_forward_kernel", r"moe_topk_gate"),
    ("TensorMove", r"^TensorMove$"),
    ("Mul", r"^Mul$"),
    ("Add", r"^Add$"),
    ("EngramFetch", r"^EngramFetch$"),
    ("EngramFetchWait", r"^EngramFetchWait$"),
    ("Cast", r"^Cast$"),
    ("Sort", r"^Sort$"),
]


def resolve(name, ref):
    for ours, pat in ALIAS:
        if ours == name:
            return sum(v for k, v in ref.items() if re.search(pat, k))   # ★ 同族名求和
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-passes", type=int, default=1)
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()

    ours, meta = our_hist(a.ours)
    ref = ref_hist(a.ref, a.ref_passes)
    print(f"参考 trace: {a.ref}（{a.ref_passes} pass 归一到单 pass）")
    print(f"我们的: L={meta['L']} T_main={meta['T_main']}\n")
    print(f"{'算子':<32}{'我们':>8}{'参考':>9}{'比值':>8}")
    rows = []
    for k, v in sorted(ours.items(), key=lambda x: -x[1])[: a.top]:
        r = resolve(k, ref)
        if r is None:
            rows.append((k, v, None, "无对应"))
            continue
        ratio = (v / r) if r else float("inf")
        flag = "" if 0.85 <= ratio <= 1.15 else ("← 偏多" if ratio > 1 else "← 偏少")
        rows.append((k, v, r, f"{ratio:.2f} {flag}"))
    for k, v, r, note in rows:
        rs = "—" if r is None else f"{r:.1f}"
        print(f"{k:<32}{v:>8}{rs:>9}   {note}")
    miss = [k for k in ref if not any(re.search(p, k) for _, p in ALIAS) and ref[k] > 2]
    if miss:
        print("\n参考里有、我们没建模的算子（前 10）：")
        for k in sorted(miss, key=lambda x: -ref[x])[:10]:
            print(f"  {ref[k]:>7.1f}  {k[:60]}")


if __name__ == "__main__":
    main()

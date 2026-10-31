"""把 dry-run 的算子直方图与**真实 trace** 对齐（M1 的核心验收工具）。

用法：
  python3 tools/align_ops.py --ours /tmp/dry.json \
      --ref  <...>/ASCEND_PROFILER_OUTPUT/kernel_details.csv --ref-passes 4 [--top 30]
说明：--ref-passes = 参考 trace 里包含几个完整前向（bench 的 --hit-inner-reps/loops）；
工具会把参考侧除以它，得到"单 pass 计数"再与我们的比对。
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
    c = collections.Counter()
    for r in csv.DictReader(open(csv_path, errors="ignore")):
        t = r.get("Type", "")
        if not t or t in ("-",):
            continue
        c[t] += 1
    # 归一到单 pass；类型名做模糊化（trace 里是 mangled 名，我们用的是短名）
    return {k: v / max(1, passes) for k, v in c.items()}


ALIAS = [  # (我们的短名, trace 名匹配正则)
    ("mega_moe_impl", r"mega_moe_impl|_ZN9deep_gemm8mega_moe"),
    ("MqsmlaKernel", r"MqsmlaKernel"),
    ("QuantSparseLightningIndexer", r"QuantSparseLightningIndexer"),
    ("QuantLightningIndexer", r"QuantLightningIndexer"),
    ("indexer_prologue_qw_kernel", r"indexer_prolog"),
    ("mhc_post_fwd_asc_kernel", r"mhc_post"),
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

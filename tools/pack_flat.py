#!/usr/bin/env python3
"""把每个点位的 `mindstudio_profiler_output/` 摊成**扁平命名的点位文件夹**（交付/解析包）。

目录规则（2026-10-29 用户口径，写入 PERF_RULES §5 / TRACE_GUIDE §1.3）：

    <PACK>/<并行策略>_L<Lk>_hit<h>_b<批>_r<0|1>_prefill/
        README.txt
        msprof_<ts>.json
        task_time_<ts>.csv
        op_summary_<ts>.csv
        op_statistic_<ts>.csv
        api_statistic_<ts>.csv
        communication_statistic_<ts>.csv
        step_trace_<ts>.json

即：**文件夹名 = 扁平点位名**，文件**直接放在它下面**（不再套 `mindstudio_profiler_output/`，
也不套 `PROF_*/`、`h0.97/L262144/`）。文件内容 = msprof 导出原样，一份不动。
`msprof --export=on` 非幂等 ⇒ 每个前缀只取**最新时间戳**那一份。

点位名的推导复用 `tools/trace_report.py:flat_name()`（唯一来源），
所以 `<点位目录>/NAME` 优先、其次 config_point.yaml + PARAMS、最后目录名兜底。

用法：
    python3 tools/pack_flat.py <采集根目录> -o <PACK 目录> [--cp 2] [--dp 4] [--ep 8]
                              [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trace_report import flat_name, point_dir_of  # noqa: E402  命名唯一来源

PREFIXES = ("api_statistic", "op_statistic", "op_summary", "task_time",
            "msprof", "step_trace", "communication_statistic")


def newest_files(od: str):
    """每个前缀只取最新一份（msprof export 非幂等）。README.txt 单独带上。"""
    out = []
    for pre in PREFIXES:
        c = [f for f in glob.glob(os.path.join(od, pre + "_*")) if os.path.isfile(f)]
        if c:
            out.append(max(c, key=lambda p: os.path.basename(p)))
    rd = os.path.join(od, "README.txt")
    if os.path.isfile(rd):
        out.append(rd)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--cp", type=int, default=None)
    ap.add_argument("--dp", type=int, default=None)
    ap.add_argument("--ep", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ov = {k: v for k, v in (("cp", a.cp), ("dp", a.dp), ("ep", a.ep)) if v is not None}
    ods = sorted(glob.glob(os.path.join(a.root, "**", "mindstudio_profiler_output"),
                           recursive=True))
    if not ods:
        sys.exit(f"✗ {a.root} 下没有 mindstudio_profiler_output（先跑 trace_report --export）")
    if not a.dry_run:
        os.makedirs(a.out, exist_ok=True)

    n, skipped, seen = 0, 0, set()
    rows = []
    for od in ods:
        pdir = os.path.dirname(od)
        flat = flat_name(point_dir_of(pdir), ov)
        if flat in seen:
            print(f"  ⚠️ 重名，跳过：{flat}  ← {od}", flush=True)
            skipped += 1
            continue
        seen.add(flat)
        files = newest_files(od)
        rows.append((flat, [os.path.basename(f) for f in files]))
        if a.dry_run:
            continue
        dst = os.path.join(a.out, flat)
        os.makedirs(dst, exist_ok=True)
        for f in files:
            shutil.copy2(f, os.path.join(dst, os.path.basename(f)))
        n += 1

    # 清单（人读 + 机器读）
    man = os.path.join(a.out, "MANIFEST.txt")
    if not a.dry_run:
        with open(man, "w", encoding="utf-8") as fh:
            fh.write(f"# 扁平点位包  {len(rows)} 个点位\n")
            fh.write("# 布局：<扁平名>/ = README.txt + 七件套（msprof 导出原样，每前缀取最新）\n")
            fh.write("# 命名：cp<CP>dp<DP>ep<EP>_L<Lk>_hit<h>_b<批>_r<0|1>_prefill\n")
            for flat, files in rows:
                fh.write(f"{flat}\t{len(files)}\t{','.join(files)}\n")
    print(f"{'[dry-run] ' if a.dry_run else ''}点位 {len(rows)} 个"
          + (f"，重名跳过 {skipped} 个" if skipped else ""))
    for flat, files in rows[:5]:
        print(f"  {flat}/  ({len(files)} 个文件)")
    if len(rows) > 5:
        print(f"  … 共 {len(rows)} 个")
    if not a.dry_run:
        print(f"→ {a.out}\n→ {man}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

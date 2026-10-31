#!/usr/bin/env python3
"""ACL 采集产物：导出 + 解析（采集完成后自动跑）。

输入：一个采集输出目录（含 `<point>/prefill/PROF_*` 或已导出的 `mindstudio_profiler_output`）。
动作：
  1) 对每个 PROF_* 目录跑 `msprof --export=on --output=<dir> --type=text`（默认只导出 rank0，
     `--all-ranks` 导出全部；已导出过的跳过）；
  2) 解析 `op_statistic_*.csv` / `api_statistic_*.csv`，聚合出：
       <root>/PARSED/op_top.csv      每算子总时长（跨 rank/点）
       <root>/PARSED/api_top.csv     ACL/host API 总时长（看同步/下发成本）
       <root>/PARSED/SUMMARY.md      人读汇总
用法：
  python3 tools/parse_acl_export.py <run_out_dir> [--all-ranks] [--no-export]
"""
from __future__ import annotations
import argparse, csv, glob, os, subprocess, sys

MSPROF_CANDIDATES = [
    os.environ.get("MSPROF", ""),
    "/mnt/xds/sfs/l00972766/ds41_runtime/ascend_full/usr/local/Ascend/cann-9.2.0/aarch64-linux/bin/msprof",
]


def find_msprof():
    for p in MSPROF_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return None


def prof_dirs(root):
    """按点位分组返回：默认每个点位只看第一份（rank0），--all-ranks 时返回全部。"""
    out = []
    for pat in ("prefill/PROF_*", "*/prefill/PROF_*", "*/L*/prefill/PROF_*", "*/h*/L*/prefill/PROF_*"):
        for d in sorted(glob.glob(os.path.join(root, pat))):
            out.append(d)
        if out: break
    return out


def export_one(msprof, pdir):
    if os.path.isdir(os.path.join(pdir, "mindstudio_profiler_output")):
        return True
    try:
        subprocess.run([msprof, "--export=on", f"--output={pdir}", "--type=text"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    except Exception:
        return False
    return os.path.isdir(os.path.join(pdir, "mindstudio_profiler_output"))


def read_stats(path, key_fields, time_field):
    agg = {}
    if not glob.glob(path):
        return agg
    for f in glob.glob(path):
        try:
            for r in csv.DictReader(open(f)):
                name = next((r.get(k) for k in key_fields if r.get(k)), "?")
                try:
                    t = float(r.get(time_field) or 0)
                except ValueError:
                    continue
                n = int(float(r.get("Count") or 1))
                cur = agg.setdefault(name, [0.0, 0])
                cur[0] += t; cur[1] += n
        except Exception:
            continue
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--all-ranks", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    dirs = prof_dirs(root)
    if not dirs:
        print(f"✗ 没找到 PROF_* 采集目录：{root}")
        return 2
    msprof = find_msprof()
    exported = 0
    if a.all_ranks:
        todo = dirs
    else:                                   # 每个点位取第一份（rank0）
        seen, todo = set(), []
        for d in dirs:
            point = os.path.dirname(os.path.dirname(d))
            if point not in seen:
                seen.add(point); todo.append(d)
    if not a.no_export:
        if not msprof:
            print("⚠️ 找不到 msprof，跳过导出（只解析已有 mindstudio_profiler_output）")
        else:
            for d in todo:
                if export_one(msprof, d):
                    exported += 1
    ops = read_stats(os.path.join(root, "prefill", "PROF_*", "mindstudio_profiler_output", "op_statistic_*.csv"),
                     ("OP Type", "Op Type", "Name"), "Total Time(us)")
    apis = read_stats(os.path.join(root, "prefill", "PROF_*", "mindstudio_profiler_output", "api_statistic_*.csv"),
                      ("API Name", "api_name", "Name"), "Time(us)")
    outdir = os.path.join(root, "PARSED")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "op_top.csv"), "w") as f:
        f.write("op,total_us,count\n")
        for k, (t, n) in sorted(ops.items(), key=lambda kv: -kv[1][0]):
            f.write(f"{k},{t:.1f},{n}\n")
    with open(os.path.join(outdir, "api_top.csv"), "w") as f:
        f.write("api,total_us,count\n")
        for k, (t, n) in sorted(apis.items(), key=lambda kv: -kv[1][0]):
            f.write(f"{k},{t:.1f},{n}\n")
    op_tot = sum(t for t, _ in ops.values())
    api_tot = sum(t for t, _ in apis.values())
    with open(os.path.join(outdir, "SUMMARY.md"), "w") as f:
        f.write(f"# ACL 采集解析汇总\n\n- 采集目录：`{root}`\n- PROF 目录数：{len(dirs)}（本导出 {exported} 个）\n")
        f.write(f"- 算子总时长：**{op_tot/1000:.1f} ms**；host/API 总时长：**{api_tot/1000:.1f} ms**\n\n")
        f.write("## 算子 Top10（总时长）\n\n| 算子 | ms | 次数 |\n|---|---:|---:|\n")
        for k, (t, n) in sorted(ops.items(), key=lambda kv: -kv[1][0])[:10]:
            f.write(f"| {k[:60]} | {t/1000:.1f} | {n} |\n")
        f.write("\n## Host/API Top10（总时长）\n\n| API | ms | 次数 |\n|---|---:|---:|\n")
        for k, (t, n) in sorted(apis.items(), key=lambda kv: -kv[1][0])[:10]:
            f.write(f"| {k} | {t/1000:.1f} | {n} |\n")
    print(f"✅ 解析完成：{outdir}/SUMMARY.md（算子 {len(ops)} 类 / API {len(apis)} 类；导出 {exported}/{len(todo)}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

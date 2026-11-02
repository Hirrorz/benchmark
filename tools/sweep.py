#!/usr/bin/env python3
"""B6 吸收（GPT §10 修改方案C「自动执行」/ §14 tools 行）：**dry-run 矩阵枚举 + 定点复跑**。

设计口径（与 GPT 的 autotune 引擎不同，见 `GPT设计验收与吸收计划.md` §4.2-B6）：
- 只做**枚举 + 定点复跑**，不做黑盒搜索：矩阵是笛卡尔积，点位可解释、可复现；
- 复用现成的 `tools/run_hit.sh`（一次进程跑多点位，权重/进程组只加载一次）；
- `cp`（`DS41M_CP_SIZE`）是**进程级**环境变量 ⇒ 按 cp 分组，每组一个 run_hit 调用；
- 每个 (cp, 点位) 的结果由 run_hit 自己写进 `<out>/cp<N>/results.tsv`（含 L/M/R/H/S/plan_hash/后端/…）；
- `--dry` 只打印**将要执行的命令**（与真实执行同一份 argv 构造，避免"dry 与实际不一致"）。

用法：
  python3.11 tools/sweep.py --matrix 'L=8192,16384;hit=0.5;b=1,2;cp=1,2' --dry
  python3.11 tools/sweep.py --matrix 'L=8192;hit=0.5;b=2;cp=1' --out /tmp/sw1 --iters 4 --run
"""
import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse_matrix(spec):
    """'L=8192,16384;hit=0.5;b=1,2;cp=1,2' → dict(有序) 的轴名→取值列表。"""
    axes = {}
    for part in str(spec).split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            sys.exit(f"✗ matrix 段要写成 key=v1,v2：{part!r}")
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in ("L", "hit", "b", "cp", "replay"):
            sys.exit(f"✗ 未知矩阵轴 {k!r}（支持 L/hit/b/cp/replay）")
        vals = [x.strip() for x in v.split(",") if x.strip()]
        if not vals:
            sys.exit(f"✗ 轴 {k} 没有取值")
        axes[k] = vals
    if not axes:
        sys.exit("✗ matrix 为空")
    return axes


def _norm_L(v):
    m = {"k": 1024, "m": 1024 * 1024}
    s = str(v).strip().lower()
    if s and s[-1] in m:
        return int(s[:-1]) * m[s[-1]]
    return int(s)


def _norm_hit(v):
    s = str(v).strip().rstrip("%")
    f = float(s)
    f = f / 100.0 if f > 1 else f
    if not (0.0 <= f < 1.0):
        sys.exit(f"✗ hit 必须 ∈ [0,1)：{v!r}")
    return f


def enumerate_points(axes):
    """笛卡尔积 → [{tag, L, hit, batches(=b), replay}] + 每个点的 cp。"""
    keys = list(axes.keys())
    out = []
    for combo in itertools.product(*[axes[k] for k in keys]):
        d = dict(zip(keys, combo))
        cp = int(d.get("cp", 1))
        L = _norm_L(d.get("L", 8192))
        if L % 128:
            sys.exit(f"✗ L 必须是 128 的倍数：{L}")
        pt = {"L": L, "hit": _norm_hit(d.get("hit", 0.5)),
              "batches": int(d.get("b", 1)),
              "replay": str(d.get("replay", "1")).strip().lower() not in ("0", "off", "false", "no")}
        out.append((cp, pt))
    return out


def group_by_cp(points):
    g = {}
    for cp, pt in points:
        g.setdefault(cp, []).append(pt)
    return g


def build_cmd(cp, pts, out, cards, iters, extra):
    """与真实执行同一份 argv 构造（dry 与实际不会漂）。"""
    pj = json.dumps(pts, ensure_ascii=False, separators=(",", ":"))
    cmd = ["bash", os.path.join(ROOT, "tools", "run_hit.sh"),
           "--points-json", pj, "--cards", cards,
           "--iters", str(iters), "-o", os.path.join(out, f"cp{cp}")]
    cmd += list(extra)
    env = dict(os.environ, DS41M_CP_SIZE=str(cp))
    # ★ 2026-11-01：CP>1 且点位含命中 ⇒ 必须显式 DS41M_CP_HIT=1。
    #   runner 里有正确性守卫（D2）：不设会把命中段当 plain prefill（缓存写错行，静默算错）。
    #   以前 sweep 不设 ⇒ 一跑 cp2/4 的命中点位就整组失败（作业 239 的 cp2 组就是这样）。
    if cp > 1 and any(float(pt.get("hit", 0) or 0) > 0 for pt in pts):
        env["DS41M_CP_HIT"] = "1"
    return cmd, env


def main():
    ap = argparse.ArgumentParser(description="B6：矩阵枚举 + 定点复跑（dry-run 优先）")
    ap.add_argument("--matrix", required=True,
                    help="'L=8192,16384;hit=0.5,0.97;b=1,2;cp=1,2'（轴：L/hit/b/cp/replay）")
    ap.add_argument("--out", default="/tmp/sweep", help="输出根（每个 cp 一个子目录）")
    ap.add_argument("--cards", default="0,1", help="物理卡，透传 run_hit（默认 0,1）")
    ap.add_argument("--iters", type=int, default=4, help="每点位采集轮（≥4，PERF_RULES）")
    ap.add_argument("--noprof", action="store_true", default=True,
                    help="只跑无 prof 稳态（默认开；run_hit 侧 --noprof-only）")
    ap.add_argument("--run", action="store_true", help="真的执行（默认只 dry-run）")
    ap.add_argument("--dry", action="store_true",
                    help="只打印命令（= 不给 --run 的默认行为；显式写出来便于脚本化）")
    ap.add_argument("--allow-suboptimal", default="B6 sweep：矩阵枚举",
                    help="透传 DS41_ALLOW_SUBOPTIMAL 的原因（关闭开关的臂需要）")
    a = ap.parse_args()
    if a.dry:
        a.run = False

    axes = _parse_matrix(a.matrix)
    points = enumerate_points(axes)
    groups = group_by_cp(points)

    total_runs = len(groups)
    print(f"[sweep] 矩阵：{dict((k, v) for k, v in axes.items())}")
    print(f"[sweep] 点位 {len(points)} 个 × {os.cpu_count() and ''}按 cp 分 {total_runs} 组：")
    for cp, pts in groups.items():
        print(f"  cp{cp}: {len(pts)} 点")
    os.makedirs(a.out, exist_ok=True)

    extra = ["--noprof-only"] if a.noprof else []
    rc_all = 0
    for cp, pts in groups.items():
        cmd, env = build_cmd(cp, pts, a.out, a.cards, a.iters, extra)
        env = dict(env, DS41_ALLOW_SUBOPTIMAL=a.allow_suboptimal)
        print(f"[sweep] cp{cp} 命令：")
        _disp = list(cmd)
        _disp[3] = f"<points:{len(pts)} 点>"
        print("  " + " ".join(shlex.quote(c) for c in _disp))
        if not a.run:
            continue
        print(f"[sweep] cp{cp} 执行中 …", flush=True)
        rc = subprocess.call(cmd, cwd=ROOT, env=env)
        print(f"[sweep] cp{cp} rc={rc}", flush=True)
        rc_all = rc_all or rc

    if not a.run:
        print("[sweep] （dry-run）以上命令用 --run 才执行。")
        return 0
    print(f"[sweep] 全部结束 rc={rc_all}；结果在各 <out>/cp*/results.tsv")
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
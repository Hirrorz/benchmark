#!/bin/bash
# ============================================================================
# run_cases.sh — 按并行策略（CP）跑一组 **case**，只报 forward 时间
#
# 口径（2026-10-29 用户口径）：
#   · **无日志**：`--pure-log`（不落任何 rank 日志；只有很小的 driver 汇总日志）
#   · **无 profiler**：`--noprof-only`（不产生 trace）⇒ forward 时间取**纯 forward 吞吐**
#     （连续下发 N 个 forward + 一次 sync，墙钟 / N；没有 msprof stop 的污染）
#   · **内循环 10 次 = 2 轮预热 + 8 轮计时**，8 轮全算（`DS41M_STEADY_KEEP_FIRST` 已按预热
#     情况自动处理），报**中位数**
#   · **同一并行策略内所有 case 一个进程一次性跑完**（一次预热，case 之间靠 memlife 释放）
#
# 用法：
#   bash tools/run_cases.sh <输出目录> [选项]
#
#     --cp "1 2 4 8"          要跑的并行策略（默认 1 2 4 8；空格或逗号分隔）
#     --cases "L@hit ..."      case 列表（默认 = 标准 205 点集：几何闸 + T_pass≤104998）
#                             支持 `1024`/`64k`/`1m` 三种写法；可带 /bN /r0|1 后缀
#     --iters 8               计时轮数（默认 8）
#     --warmup 2              预热轮数（默认 2）
#     --batch 1               每 case 的批数（默认 1）
#     --summary-only          不跑采集，只用已有 results.tsv 重出 FORWARD_TIMES 表
#
# 例：
#   bash tools/run_cases.sh /mnt/xds/sfs/l00972766/tmp/CASES_1029_1400
#   bash tools/run_cases.sh /tmp/cases --cp "1 4" --cases "1024@0.97 65536@0.97 262144@0.9"
#
# 产物：
#   <OUT>/cp<N>/results.tsv         每 case 一行（steady_s / ms_per_forward / errors）
#   <OUT>/FORWARD_TIMES.md          对比表：行 = case，列 = CP
#   <OUT>/FORWARD_TIMES.csv         长表（便于画图）
# 控制台最后打印同一张表。
#
# 断点续跑（2026-10-29）：
#   · **重跑同一条命令**即可 —— 每个 CP 的 `results.tsv` 里已有的 case 会被跳过
#     （points_runner 的续跑机制）；某个 CP 的 case **全做完**时连进程都不起（整组跳过）。
#   · 强制重跑：`DS41M_POINT_REDO=1`。
#   · 只想拿表不想跑：`--summary-only`（用已有 results.tsv 重出 FORWARD_TIMES.md/csv）。
#   · 中断后重跑：已跑完的 CP 跳过（秒级），没跑完的 CP 起一次进程继续补。
# ============================================================================
set -u
BASE=/mnt/xds/sfs/l00972766
DS=$BASE/ds41_mini

OUT_ROOT=""; CPS="1 2 4"; CASES=""; ITERS=8; WARM=2; BATCH=1; SUM_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --cp)      CPS=${2:-}; shift ;;
    --cp=*)    CPS=${1#--cp=} ;;
    --cases)   CASES=${2:-}; shift ;;
    --cases=*) CASES=${1#--cases=} ;;
    --iters)   ITERS=${2:-8}; shift ;;
    --warmup)  WARM=${2:-2}; shift ;;
    --batch)   BATCH=${2:-1}; shift ;;
    --summary-only) SUM_ONLY=1 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    -*)        echo "✗ 未知参数 $1" >&2; exit 2 ;;
    *)         OUT_ROOT=$1 ;;
  esac
  shift
done
[ -n "$OUT_ROOT" ] || { echo "用法: $0 <输出目录> [--cp \"1 2 4 8\"] [--cases \"…\"]" >&2; exit 2; }
CPS=$(echo "$CPS" | tr ',' ' ')

cd "$DS" || { echo "✗ 进不去 $DS" >&2; exit 2; }
source env_pack.sh >/dev/null 2>&1
mkdir -p "$OUT_ROOT" || exit 2
export TMPDIR=${DS41M_TMPDIR:-$OUT_ROOT/tmp}; mkdir -p "$TMPDIR"
LOG=$OUT_ROOT/run_cases_$(date -u +%m%d_%H%M).log
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ── case 列表：默认 = 标准 205 点集（几何闸 + T_pass ≤ 104998）────────────
if [ -z "$CASES" ]; then
  CASES=$(MATRIX_TPASS_CAP=${MATRIX_TPASS_CAP:-104998} python3 - <<'PY'
import os, re
CAP = int(os.environ.get("MATRIX_TPASS_CAP", "104998") or 0)
LENS = "1m".split()
HITS = "0.991 0.992 0.993 0.994 0.995 0.996 0.997 0.998 0.998047 0.999 0.999023 0.999512 0.999756".split()
def plen(s):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([km]?)", s.lower())
    return int(float(m.group(1)) * {"": 1, "k": 1024, "m": 1048576}[m.group(2)])
out = []
for Ls in LENS:
    L = plen(Ls)
    for h in HITS:
        hf = float(h)
        t = L if hf <= 0 else int(round((1 - hf) * L)) + 140
        if hf > 0 and t > L:
            continue
        if CAP > 0 and t > CAP:
            continue
        out.append(f"{Ls}@{h}")
print(" ".join(out))
PY
)
fi
NCASE=$(wc -w <<< "$CASES")
if [ "$SUM_ONLY" = 1 ]; then
  say "== summary-only：从已有 results.tsv 出表   并行策略：$CPS   输出：$OUT_ROOT"
else
  say "== case 数：$NCASE   并行策略：$CPS   预热 $WARM + 计时 $ITERS（全算）  输出：$OUT_ROOT"
fi
say "   口径：--noprof-only（无 trace）+ --pure-log（无 rank 日志）；forward 时间 = 稳态中位"

# 覆盖率：results.tsv 里已有多少个本次要求的 case（用于整组跳过）
_coverage(){  # $1=results.tsv $2..=cases → "done total"
python3 - "$@" <<'PY2'
import os, re, sys
p, cases = sys.argv[1], sys.argv[2:]
MUL = {"": 1, "k": 1024, "m": 1048576}
def plen(x):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([km]?)", x.lower())
    return int(float(m.group(1)) * MUL[m.group(2)])
want = set()
for c in cases:
    L, h = c.split("@")
    want.add((plen(L), float(h.split("/")[0])))
have = set()
rx = re.compile(r"_L(\d+)_hit([\d.]+)_batch\d+_replay(?:on|off)_prefill$")
if os.path.isfile(p):
    for ln in open(p, encoding="utf-8"):
        m = rx.search(ln.split("\t")[0])
        if m:
            have.add((int(m.group(1)), float(m.group(2))))
print(len(want & have), len(want))
PY2
}

rc_all=0
if [ "$SUM_ONLY" = 1 ]; then
  say "== --summary-only：跳过采集，只用已有 results.tsv 出表（CP：$CPS）"
fi

for cp in $CPS; do
  [ "$SUM_ONLY" = 1 ] && break
  OUT=$OUT_ROOT/cp$cp
  SP=""
  for c in $CASES; do
    case "$c" in
      */*) SP="$SP $c" ;;                      # 已带 /bN/rX 后缀
      *)   SP="$SP ${c}/b${BATCH}/r1" ;;
    esac
  done
  SP=${SP# }
  if [ "${DS41M_POINT_REDO:-0}" != "1" ]; then
    read -r _done _tot <<< "$(_coverage "$OUT/results.tsv" $CASES)"
    if [ "$_tot" -gt 0 ] && [ "$_done" = "$_tot" ]; then
      say "── CP$cp：已有 $_done/$_tot 个 case → **整组跳过**（DS41M_POINT_REDO=1 可强制重跑）"
      continue
    fi
    [ "$_done" -gt 0 ] && say "── CP$cp：已有 $_done/$_tot，续跑剩余 $((_tot - _done)) 个（进程会先起 5–6 min）"
  fi
  say "── CP$cp：$NCASE 个 case，一个进程跑完  → $OUT"
  DS41M_TMPDIR="$TMPDIR" \
    nice -n 5 env DS41M_CP_SIZE="$cp" DS41M_CP_HIT=1 \
      DS41M_MEM_AUDIT=${DS41M_MEM_AUDIT:-0} DS41M_POINT_TIMEOUT=${DS41M_POINT_TIMEOUT:-3600} \
      timeout ${DS41M_CASES_TIMEOUT:-21600} \
    bash tools/run_hit.sh --points-spec "$SP" --noprof-only --pure-log \
         --iters "$ITERS" --prof-warmup "$WARM" -o "$OUT" \
    2>&1 | tee -a "$LOG" | grep --line-buffered -E '\[point |\[mem\]|✗|↷|全部完成|多点位总用时'
  rc=${PIPESTATUS[0]}
  say "   CP$cp rc=$rc"
  [ "$rc" = 0 ] || rc_all=1
done

# ── 汇总：每个 case 的 forward 时间（中位），按 CP 列开 ──────────────
say "== 汇总"
python3 - "$OUT_ROOT" "$CPS" <<'PY' 2>&1 | tee -a "$LOG"
import csv, os, re, sys
root, cps = sys.argv[1], sys.argv[2].split()
rx = re.compile(r"_L(\d+)_hit([\d.]+)_batch(\d+)_replay(on|off)_prefill$")
data, cases = {}, {}
for cp in cps:
    p = os.path.join(root, f"cp{cp}", "results.tsv")
    if not os.path.isfile(p):
        continue
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            m = rx.search((r.get("tag") or "").strip())
            if not m:
                continue
            L, h = int(m.group(1)), float(m.group(2))
            key = (L, h)
            cases[key] = True
            data[(cp, key)] = (float(r.get("ms_per_forward") or "nan"),
                               float(r.get("steady_s") or "nan"),
                               int(float(r.get("errors") or 0)))
if not data:
    print("✗ 没有结果（results.tsv 缺失或 tag 不匹配）"); sys.exit(0)
ks = sorted(cases)
cps_have = [c for c in cps if any((c, k) in data for k in ks)]
def tpass(L, h):
    return L if h <= 0 else int(round((1 - h) * L)) + 140
md = ["# forward 时间（无 prof / 无日志；每格 = 计时轮中位，ms/forward）", "",
      "| case | T_pass | " + " | ".join(f"CP{c}" for c in cps_have) + " |",
      "|---|---:|" + "---:|" * len(cps_have)]
rows = []
for k in ks:
    cells = []
    for c in cps_have:
        v = data.get((c, k))
        cells.append("—" if v is None else f"{v[0]:.1f}")
    md.append(f"| L{k[0]}@hit{k[1]:g} | {tpass(*k)} | " + " | ".join(cells) + " |")
    rows.append([k, cps_have, [data.get((c, k)) for c in cps_have]])
out_md = os.path.join(root, "FORWARD_TIMES.md")
open(out_md, "w", encoding="utf-8").write("\n".join(md) + "\n")
out_csv = os.path.join(root, "FORWARD_TIMES.csv")
with open(out_csv, "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["L", "hit", "T_pass"] + [f"cp{c}_ms" for c in cps_have]
               + [f"cp{c}_errors" for c in cps_have])
    for k, cs, vals in rows:
        w.writerow([k[0], f"{k[1]:g}", tpass(*k)]
                   + [("" if v is None else f"{v[0]:.2f}") for v in vals]
                   + [("" if v is None else v[2]) for v in vals])
print(f"case 数 {len(ks)}，并行策略 {cps_have}")
print("\n".join(md[2:]))
print(f"\n→ {out_md}\n→ {out_csv}")
PY

say "== 完成（日志：$LOG）"
[ "$rc_all" = 0 ] || { say "⚠ 有 CP 非 0 退出，看日志"; exit 1; }
exit 0

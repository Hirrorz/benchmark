#!/bin/bash
# ============================================================================
# DP8CP1EP8 / DP4CP2EP8 / DP2CP4EP8 矩阵采集（forward 时延 + 单卡 profiling）
#
# 口径（用户 2026-10-29）：
#   · 长度：每个 DP 的输入序列长度 L（1k … 1m，共 20 档）
#   · 命中率：0/10/60/70/80/90/95/96/97/98/99 %（共 11 档）
#   · 并行：world=8；CP=1→DP8、CP=2→DP4、CP=4→DP2；MoE EP=world=8
#   · 采集：**默认采集策略** —— prof 轮、只采 rank0、`--pure-log`（无日志）；
#           forward 时延取 launch.log 的 `[r1] 多轮耗时`（默认口径）
#
# 用法：
#   bash tools/collect_dp_cp_matrix.sh <OUT_ROOT> [CP列表] [L列表] [hit列表]
#   例：bash tools/collect_dp_cp_matrix.sh /workspace/tmp/collect "2 4"
#       bash tools/collect_dp_cp_matrix.sh /workspace/tmp/collect "2" "64k 256k" "0.97 0"
#
# 进程分组（MATRIX_L_GROUP，每个进程里连跑多少个长度档）：
#   1（默认）= 每个长度一个进程（11 个命中率连跑），断点粒度最细；
#   N>1      = N 个长度档合并进一个进程（如 7 → 64k 80k 96k 112k 128k … 一起跑）；
#   0        = 该 CP 的全部长度档一个进程（等价"28 点矩阵"那种长 SPEC 跑法）。
#   命中率始终跟着长度一起进同一个进程。
#
# 单个进程跑多久：MATRIX_RUN_TIMEOUT（默认 14400s=4h，传给 run_hit --timeout）。
#   一整段 205 点带 prof 可能要 40–70 分钟，驱动默认 3600s 会中途杀掉。
#
# 每 CP 的实际点数：由几何闸 + `MATRIX_TPASS_CAP`（默认 104998，0=不设限）**现算**，不写死。
#   切 CP 后可行性不同可**按 CP 分别设上限**：`MATRIX_TPASS_CAP_<CP>` 优先于全局值，
#   例：MATRIX_TPASS_CAP_2=104998 MATRIX_TPASS_CAP_4=80000 …
#   只看点集不跑：MATRIX_DRY_RUN=1（打印各 CP 的点数与完整点位串后跳过）。
#
# 断点续跑：每块成功后写 `.done`；重跑自动跳过。日志在 <OUT>/cpX/L<首>-<末>/collect.log。
# 终端实时显示逐点行：**默认开**（`[r0] [point i/N] 开始/完成 … 每forward=…ms`）；
#   静默只落块日志：MATRIX_VERBOSE=0。
# 磁盘：输出目录所在盘需 ≥10G 可用（共享盘满时用 /workspace/tmp）。
# ============================================================================
set -u
OUT_ROOT=${1:?用法: collect_dp_cp_matrix.sh OUT_ROOT [CP列表] [L列表] [hit列表]}
CPS=${2:-"2 4"}
LENS=${3:-"1k 2k 4k 8k 12k 16k 20k 24k 32k 40k 48k 56k 64k 80k 96k 112k 128k 256k 512k 1m"}
HITS=${4:-"0 0.1 0.6 0.7 0.8 0.9 0.95 0.96 0.97 0.98 0.99"}
GROUP=${MATRIX_L_GROUP:-1}
RUN_TMO=${MATRIX_RUN_TIMEOUT:-14400}

BASE=/mnt/xds/sfs/l00972766
cd "$BASE/ds41_mini" || exit 2
source env_pack.sh >/dev/null 2>&1
export TMPDIR=${DS41M_TMPDIR:-/workspace/tmp}
mkdir -p "$OUT_ROOT" "$TMPDIR" || exit 2

# 磁盘闸（输出盘可用 <10G 直接拒绝，避免半途写满）
_avail=$(df -Pk "$OUT_ROOT" | awk 'NR==2{print $4}')
if [ "${_avail:-0}" -lt $((10*1024*1024)) ]; then
  echo "✗ $OUT_ROOT 可用 $(( _avail/1024/1024 ))G < 10G，先把输出盘腾出空间" >&2
  exit 3
fi

read -r -a LA <<< "$LENS"
NG=${#LA[@]}
if [ "$GROUP" -le 0 ] || [ "$GROUP" -gt "$NG" ]; then STEP=$NG; else STEP=$GROUP; fi
echo "[分组] 每个进程 ${STEP} 个长度档（共 ${NG} 档；MATRIX_L_GROUP=$GROUP）"

for CP in $CPS; do
  case "$CP" in 1|2|4|8) ;; *) echo "✗ CP=$CP 不支持（1/2/4/8）" >&2; exit 2 ;; esac
  # 每个 CP 可用独立上限：MATRIX_TPASS_CAP_<CP> 优先，其次 MATRIX_TPASS_CAP，默认 104998。
  _cv="MATRIX_TPASS_CAP_${CP}"
  CAP=${!_cv:-${MATRIX_TPASS_CAP:-104998}}
  echo "[点集] CP=$CP T_pass 上限=$CAP（0=不设限）"
  gi=0
  while [ "$gi" -lt "$NG" ]; do
    GRP=("${LA[@]:gi:STEP}")
    gi=$((gi + STEP))
    if [ ${#GRP[@]} -eq 1 ]; then
      LNAME="L${GRP[0]}"
    else
      LNAME="L${GRP[0]}-${GRP[${#GRP[@]}-1]}"
    fi
    OUT=$OUT_ROOT/cp${CP}/${LNAME}
    if [ -f "$OUT/.done" ]; then echo "[skip] $OUT"; continue; fi
    mkdir -p "$OUT"
    # 几何闸（逐长度）：hit>0 时 R=M+KEEP 必须 ≤ L（KEEP=140 不能被 /t 调小：
    #   `assemble.replay_tail_of` 取 max(显式, keep_tokens)）。已知唯一非法点：1k@10%。
    SPEC=""
    for L in "${GRP[@]}"; do
      S=$(MATRIX_TPASS_CAP="$CAP" python3.11 - "$L" "$HITS" <<'PY'
import os, re, sys

def plen(s):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([km]?)", s.lower())
    return int(float(m.group(1)) * {"": 1, "k": 1024, "m": 1048576}[m.group(2)])

CAP = int(os.environ.get("MATRIX_TPASS_CAP", "104998") or 0)   # 0 = 不设上限
L = plen(sys.argv[1])
toks = []
for h in sys.argv[2].split():
    hf = float(h)
    m = int(round((1 - hf) * L))
    tpass = L if hf <= 0 else m + 140          # hit=0 是满 prefill，无 KEEP
    if hf > 0 and tpass > L:
        print(f"  [skip] {sys.argv[1]}@{h}: M={m}+KEEP140 > L={L}（几何非法）",
              file=sys.stderr)
        continue
    if CAP > 0 and tpass > CAP:
        print(f"  [skip] {sys.argv[1]}@{h}: T_pass={tpass} > {CAP}（与 SWEEP205 CP1 口径对齐）",
              file=sys.stderr)
        continue
    toks.append(f"{sys.argv[1]}@{h}/b1/r1")
print(" ".join(toks))
PY
)
      if [ -z "$S" ]; then echo "[skip] L=$L 无可跑点"; continue; fi
      SPEC="$SPEC $S"
    done
    SPEC=${SPEC# }
    if [ -z "$SPEC" ]; then echo "[skip] CP=$CP $LNAME 无可跑点"; continue; fi
    NPT=$(wc -w <<< "$SPEC")
    echo "===== CP=$CP $LNAME（${NPT} 点）$(date -u +%H:%M:%S) ====="
    if [ "${MATRIX_DRY_RUN:-0}" = 1 ]; then
      # 只算点位、不跑：用于切 CP 前确认各 CP 的实际点数与点集
      echo "  [dry-run] ${NPT} 点：$SPEC"
      continue
    fi
    # 等空闲窗口：本机 8 卡与另一容器共用（邻居任务 ~19.7GB/卡，会周期性起停）。
    # 默认等 30 分钟拿干净窗口；等不到就照跑但打 `.contended` 标记，便于事后筛除。
    WAIT_FREE=${MATRIX_WAIT_FREE_SEC:-1800}
    if bash "$BASE/tools/wait_npu_idle.sh" "$WAIT_FREE" > "$OUT/wait_idle.log" 2>&1; then
      :
    else
      touch "$OUT/.contended"
      echo "  ⚠ 未等到空闲窗口（${WAIT_FREE}s），照跑并标记 .contended" >> "$OUT/wait_idle.log"
    fi
    rc=1
    for ATT in 1 2 3; do
      # 第 2 轮起保留 rank 日志（--pure-log 默认连报错栈都不留，见 2026-10-29 L1k 事故）
      KEEP=0; [ "$ATT" -gt 1 ] && KEEP=1
      if [ "${MATRIX_VERBOSE:-1}" = 1 ]; then
        # 逐点行同时打到终端（tee），块日志照留
        DS41M_KEEP_LOG_FILES=$KEEP DS41M_CP_SIZE=$CP DS41M_CP_HIT=1 \
          DS41M_MEM_AUDIT=1 DS41M_MEM_GUARD_GB=${DS41M_MEM_GUARD_GB:-4} \
          DS41M_POINT_TIMEOUT=${DS41M_POINT_TIMEOUT:-1800} \
          DS41M_POINT_MAX_FAILS=${DS41M_POINT_MAX_FAILS:-2} \
          timeout $((RUN_TMO + 600)) \
          bash tools/run_hit.sh --points-spec "$SPEC" --pure-log --timeout "$RUN_TMO" -o "$OUT" \
          2>&1 | tee "$OUT/collect.log"
        rc=${PIPESTATUS[0]}
      else
        DS41M_KEEP_LOG_FILES=$KEEP DS41M_CP_SIZE=$CP DS41M_CP_HIT=1 \
          DS41M_MEM_AUDIT=1 DS41M_MEM_GUARD_GB=${DS41M_MEM_GUARD_GB:-4} \
          DS41M_POINT_TIMEOUT=${DS41M_POINT_TIMEOUT:-1800} \
          DS41M_POINT_MAX_FAILS=${DS41M_POINT_MAX_FAILS:-2} \
          timeout $((RUN_TMO + 600)) \
          bash tools/run_hit.sh --points-spec "$SPEC" --pure-log --timeout "$RUN_TMO" -o "$OUT" \
          > "$OUT/collect.log" 2>&1
        rc=$?
      fi
      echo "cp$CP $LNAME attempt=$ATT rc=$rc"
      [ "$rc" = 0 ] && break
      # ★ 2026-10-29：**不再把 $OUT 整个搬走**。点位连跑现在支持续跑
      #   （`$OUT/results.tsv` 里已完成的点位直接跳过），而且 trace 也在 $OUT 里 ——
      #   搬目录等于：①重试从头再跑 205 点；②已采的 trace 变成孤儿。
      #   失败现场只归档本轮的日志（rank 日志在 --pure-log 下本来就只有 KEEP=1 时才有）。
      cp -f "$OUT/collect.log" "$OUT/collect.attempt${ATT}.$(date -u +%H%M%S).log" 2>/dev/null
      for _f in "$OUT"/rank*.log "$OUT"/*_rank*.log; do
        [ -f "$_f" ] && cp -f "$_f" "${_f}.attempt${ATT}.$(date -u +%H%M%S)" 2>/dev/null
      done
    done
    if [ "$rc" = 0 ]; then
      echo "points=$NPT cap=$CAP $(date -u +%FT%TZ)" > "$OUT/.done"
    else
      echo "rc=$rc $(date -u +%FT%TZ) attempts=3" > "$OUT/.failed"
    fi
  done
done

# ── 汇总 forward 时延 ──
# 口径（PERF_RULES §0）：只采 rank0；时延取**未采集的 rank1**（`[r1] 纯 forward 吞吐`），
# 不用 rank0 的 `每forward=`（含 msprof 开销）。★8 个 rank 都会打印 `──── [point i/N]` 头，
# 交错时按"出现顺序"配对不可靠 ⇒ 改成**第 k 个点 ↔ 第 k 条 [r1] 行**（每 rank 严格顺序跑点）。
python3.11 - "$OUT_ROOT" "$CPS" <<'PY'
import os, re, sys
root, cps = sys.argv[1], sys.argv[2].split()
hdr = re.compile(r"──── \[point \d+/\d+\] \((\d+)_([0-9.]+)\)_r1")
r1 = re.compile(r"\[r1\]\s+纯 forward 吞吐：([0-9.]+) ms/forward")
rows = []
for cp in cps:
    cpdir = os.path.join(root, f"cp{cp}")
    if not os.path.isdir(cpdir):
        continue
    for name in sorted(os.listdir(cpdir)):
        if not name.startswith("L") or "." in name:   # 跳过 .attempt/.failed 旁路目录
            continue
        d = os.path.join(cpdir, name)
        if not os.path.isdir(d):
            continue
        p = os.path.join(d, "driver_latest.log")
        if not os.path.exists(p):
            p = os.path.join(d, "collect.log")
        if not os.path.exists(p):
            continue
        pts, seen, r1s = [], set(), []
        for line in open(p, errors="ignore"):
            m = hdr.search(line)
            if m:
                key = (int(m.group(1)), float(m.group(2)))
                if key not in seen:
                    seen.add(key)
                    pts.append(key)
                continue
            m = r1.search(line)
            if m:
                r1s.append(float(m.group(1)))
        if len(pts) != len(r1s):
            print(f"  [warn] {cp}/{name}: 点 {len(pts)} 个、[r1] 行 {len(r1s)} 条（不对齐，按前 {min(len(pts), len(r1s))} 个配对）")
        for (L, hit), ms in zip(pts, r1s):
            rows.append((cp, L, hit, ms))
rows = sorted(set(rows), key=lambda r: (r[0], r[1], r[2]))
out = os.path.join(root, "forward_times.tsv")
with open(out, "w") as fh:
    fh.write("cp\tL\thit\tr1_ms_per_forward\n")
    for cp, L, hit, ms in rows:
        fh.write(f"{cp}\t{L}\t{hit}\t{ms:.1f}\n")
print(f"forward_times → {out}（{len(rows)} 行；口径 = [r1] 未采集卡，不含 prof 开销）")
PY

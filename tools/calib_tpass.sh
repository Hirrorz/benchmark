#!/bin/bash
# ============================================================================
# calib_tpass.sh — 实测某个 CP 下"可行 T_pass 上限"（用于给采集器定 MATRIX_TPASS_CAP_<CP>）
#
# 用法：bash tools/calib_tpass.sh <CP> <OUT_DIR>
#   按 T_pass 升序逐个点单独起进程（--noprof-only，无 prof 开销），
#   遇到第一个失败点即停 —— 上一个成功点的 T_pass 就是该 CP 的实测上限。
#   每点后台采样 npu-smi 的 HBM 峰值，写入 <OUT>/<L>_<hit>/peak_mb 与 summary.tsv。
# ============================================================================
set -u
CP=${1:?用法: calib_tpass.sh CP OUT_DIR}
OUT=${2:?用法: calib_tpass.sh CP OUT_DIR}
CAND=${MATRIX_CALIB_POINTS:-"112k@0 128k@0 256k@0.1 256k@0 512k@0 1m@0"}

BASE=/mnt/xds/sfs/l00972766
cd "$BASE/ds41_mini" || exit 2
source env_pack.sh >/dev/null 2>&1
export TMPDIR=${DS41M_TMPDIR:-/workspace/tmp}
export DS41M_CP_SIZE=$CP DS41M_CP_HIT=1
mkdir -p "$OUT" "$TMPDIR"
: > "$OUT/summary.tsv"
echo "[calib] CP=$CP  候选: $CAND" | tee -a "$OUT/summary.tsv"

last_ok=""
for spec in $CAND; do
  L=${spec%@*}; H=${spec#*@}
  d="$OUT/${L}_${H}"
  rm -rf "$d"; mkdir -p "$d"
  echo "[calib] $(date -u +%H:%M:%S) 测 $spec ..."
  bash "$BASE/tools/wait_npu_idle.sh" 600 > "$d/wait_idle.log" 2>&1 \
    || echo "  [warn] 未等到空闲基线，照跑" >> "$d/wait_idle.log"
  # 后台采样 HBM 峰值
  (
    peak=0
    while :; do
      v=$(npu-smi info 2>/dev/null | grep -oE "[0-9]+ +/ +98304" | awk '{print $1}' | sort -n | tail -1)
      if [ -n "$v" ] && [ "$v" -gt "$peak" ]; then peak=$v; echo "$peak" > "$d/peak_mb"; fi
      sleep 2
    done
  ) &
  smp=$!
  timeout 2400 bash tools/run_hit.sh --points-spec "${L}@${H}/b1/r1" \
    --noprof-only --pure-log --timeout 1800 -o "$d" > "$d/run.log" 2>&1
  rc=$?
  kill "$smp" 2>/dev/null; wait "$smp" 2>/dev/null
  pk=$(cat "$d/peak_mb" 2>/dev/null || echo NA)
  printf '%s\trc=%s\tpeak_mb=%s\n' "$spec" "$rc" "$pk" | tee -a "$OUT/summary.tsv"
  if [ "$rc" = 0 ]; then
    last_ok=$spec
    sleep 20          # 等幽灵内存回落
  else
    echo "✗ $spec 失败（rc=$rc）→ 该 CP 的可行上限 = ${last_ok:-104998（未超过 CP1 口径）}" | tee -a "$OUT/summary.tsv"
    # MATRIX_CALIB_ALL=1：失败也继续跑后面的候选（用于摸 L 维边界）
    [ "${MATRIX_CALIB_ALL:-0}" = 1 ] || break
  fi
done

if [ -z "$last_ok" ]; then
  echo "结论: CP=$CP 连 $CAND 的第一档都跑不过 ⇒ 维持 MATRIX_TPASS_CAP=104998" | tee -a "$OUT/summary.tsv"
else
  tp=$(python3.11 - "$last_ok" <<'PY'
import re, sys
s = sys.argv[1]; L, h = s.split("@")
n = int(float(L[:-1]) * (1024 if L[-1] == "k" else 1048576))
L = n
hf = float(h); m = int(round((1 - hf) * L))
print(L if hf <= 0 else m + 140)
PY
)
  echo "结论: CP=$CP 实测可行上限 ≥ T_pass=$tp（最大成功点 $last_ok）；建议 MATRIX_TPASS_CAP_${CP}=$tp" | tee -a "$OUT/summary.tsv"
fi

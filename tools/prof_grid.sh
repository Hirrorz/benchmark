#!/bin/bash
# ============================================================================
# 采 prof 轮 MindStudio trace（每个点位只采 rank0）
#
#   用法：
#     bash tools/prof_grid.sh              # 跑 CP1 / CP2 / CP4
#     bash tools/prof_grid.sh 1            # 只跑 CP1
#     bash tools/prof_grid.sh 2 4          # 跑 CP2 和 CP4
#     GRID=/path/to/grid bash tools/prof_grid.sh    # 换网格目录
#
#   行为：
#     · 每个点位：warmup 2 轮（不进 trace）+ 8 轮采集，只采 rank0
#     · 进程若崩（aicore error / OOM / barrier timeout），自动重跑同一条命令
#     · `results.tsv` 有记录的点位跳过 → 崩在哪里就从哪里续
#     · 确定性失败的点位（`points_failed.tsv` 里累计到 `DS41M_POINT_MAX_ATTEMPTS`）
#       自动跳过，不会永远堵在最前面
#     · trace 落在 <GRID>/prof_cp<N>/<tag>/prefill/PROF_*/
# ============================================================================
set -u
BASE=/mnt/xds/sfs/l00972766
GRID=${GRID:-$BASE/grid_1103_0646}
ITERS=${ITERS:-8}
WARM=${WARM:-2}
MAXROUND=${MAXROUND:-6}          # 每 CP 最多重跑几轮
MINFREE_GB=${MINFREE_GB:-60}     # 低于这个可用空间就停（trace 约 96 MB/点位）

cd "$BASE/ds41_mini" || exit 2
source env_pack.sh >/dev/null 2>&1
unset LD_PRELOAD
export DS41M_CP_BUCKET=0         # DeepEP 与 ALOP profiler 不兼容，采集必须 HCCL
export DS41M_CP_HIT=1
export DS41M_MOE_CAST_INTO_SB=1
export DS41M_POINT_MAX_ATTEMPTS=2   # 坏点最多试 2 次就跳过
export DS41M_POINT_TIMEOUT=900      # 单点位超过 15 分钟强制退出

CPS="${*:-1 2 4}"

free_gb() { df -Pk "$BASE" 2>/dev/null | awk 'NR==2{printf "%d", $4/1048576}'; }
kill_leaked() {
  for p in $(pgrep -f 'launch_mini' 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  sleep 4
}
done_count() { awk 'END{print NR-1}' "$1" 2>/dev/null || echo 0; }

echo "=============================================================="
echo "  采 prof 轮 trace"
echo "  网格    : $GRID"
echo "  CP      : $CPS"
echo "  每点位  : warmup $WARM + 采 $ITERS 轮，只采 rank0"
echo "  可用空间: $(free_gb) GB"
echo "=============================================================="

for CP in $CPS; do
  PTS="$GRID/points_cp${CP}.json"
  PROFDIR="$GRID/prof_cp${CP}"
  if [ ! -f "$PTS" ]; then
    echo ""
    echo "✗ CP${CP}: 找不到 $PTS（先跑 tools/gen_grid.py --cp $CP）"
    continue
  fi
  NP=$(python3.11 -c "import json;print(len(json.load(open('$PTS'))))" 2>/dev/null || echo 0)
  echo ""
  echo "================= CP${CP}: ${NP} 个点位 ================="

  for round in $(seq 1 "$MAXROUND"); do
    DN=$(done_count "$PROFDIR/results.tsv")
    if [ "${DN:-0}" -ge "$NP" ]; then
      echo "  ✓ CP${CP} 已全部采完 (${DN}/${NP})"
      break
    fi
    FREE=$(free_gb)
    if [ "$FREE" -lt "$MINFREE_GB" ]; then
      echo "  ✗ 可用空间只剩 ${FREE} GB < ${MINFREE_GB} GB，停止 CP${CP}"
      break
    fi

    kill_leaked
    echo ""
    echo "  ── 第 ${round} 轮：已完成 ${DN}/${NP}，可用 ${FREE} GB ──"
    env DS41M_CP_SIZE="$CP" bash tools/run_hit.sh \
        --points "$PTS" \
        --prof-ranks 0 --prof-warmup "$WARM" --iters "$ITERS" \
        --cards 0,1,2,3,4,5,6,7 \
        -o "$PROFDIR" \
        > "$GRID/prof_cp${CP}_r${round}.log" 2>&1
    RC=$?
    DN2=$(done_count "$PROFDIR/results.tsv")
    echo "  rc=$RC  完成=${DN2:-0}/${NP}  (+$(( ${DN2:-0} - ${DN:-0} )))"
    if [ "${DN2:-0}" -ge "$NP" ]; then
      echo "  ✓ CP${CP} 完成"
      break
    fi
    if [ "${DN2:-0}" -le "${DN:-0}" ]; then
      echo "  ⚠ 本轮零进展，检查 $GRID/prof_cp${CP}_r${round}.log"
    fi
  done

  echo ""
  echo "  ── CP${CP} 结果 ──"
  awk -F'\t' 'NR>1{printf "    %-56s %ss err=%s\n",$1,$2,$4}' \
      "$PROFDIR/results.tsv" 2>/dev/null | tail -3
  TR=$(du -sh "$PROFDIR" 2>/dev/null | cut -f1)
  NPROF=$(find "$PROFDIR" -maxdepth 4 -name 'PROF_*' -type d 2>/dev/null | wc -l)
  echo "    trace 目录 $NPROF 个，占用 $TR"
done

echo ""
echo "=============================================================="
echo "  完成汇总"
for CP in $CPS; do
  NP=$(python3.11 -c "import json;print(len(json.load(open('$GRID/points_cp${CP}.json'))))" 2>/dev/null || echo 0)
  DN=$(done_count "$GRID/prof_cp${CP}/results.tsv")
  echo "  CP${CP}: ${DN}/${NP}   $(du -sh "$GRID/prof_cp${CP}" 2>/dev/null | cut -f1)"
done
echo "  可用空间: $(free_gb) GB"
echo "=============================================================="

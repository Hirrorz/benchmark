#!/bin/bash
# 网格总控 v3：每轮 attempt 前后强制清干净 + 验证 8 卡都空 + 校验 rank 数量
set -u
BASE=/mnt/xds/sfs/l00972766
ROOT=$1
cd "$BASE/ds41_mini" || exit 2
source env_pack.sh >/dev/null 2>&1
export DS41M_CP_BUCKET=0 DS41M_CP_HIT=1 DS41M_MOE_CAST_INTO_SB=1
unset LD_PRELOAD

# 彻底清：grid_all 自己以外的 launch_mini/runner 全杀，等设备回落
hard_clean() {
  for round in 1 2 3; do
    pids=$(pgrep -f 'launch_mini|core/runner\.py' 2>/dev/null)
    [ -z "$pids" ] && break
    for p in $pids; do kill -9 "$p" 2>/dev/null; done
    sleep 4
  done
  sleep 3
  # 等 8 卡 HBM 回落到基线（<8GB）
  for wait in $(seq 1 12); do
    busy=$(npu-smi info 2>/dev/null | grep -oE '[0-9]+ */ *98304' | awk -F'/' '{gsub(/ /,"",$1); if ($1+0 > 8000) n++} END{print n+0}')
    [ "${busy:-0}" -eq 0 ] && break
    sleep 5
  done
  echo "     [clean] 空闲卡检查完成（busy=${busy:-0}）"
}

# 校验本轮真的起了 8 个 rank
check_ranks() {
  n=$(pgrep -f 'core/runner\.py.*--rank' 2>/dev/null | wc -l)
  echo "     [check] runner 进程数 = $n（应 8）"
  [ "$n" -ge 8 ] && return 0 || return 1
}

echo "########## GRID v3 START $(date +%H:%M:%S) ##########"
for CP in 1 2 4; do
  PTS="$ROOT/points_cp${CP}.json"
  [ -f "$PTS" ] || python3.11 tools/gen_grid.py --out "$PTS" --cp "$CP" --report-skipped
  NP=$(python3.11 -c "import json;print(len(json.load(open('$PTS'))))")
  OUT="$ROOT/noprof_cp${CP}"
  echo ""
  echo "================= CP${CP}: ${NP} 点位 $(date +%H:%M:%S) ================="
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    DONE0=$(awk 'END{print NR-1}' "$OUT/results.tsv" 2>/dev/null || echo 0)
    if [ "${DONE0:-0}" -ge "$NP" ]; then echo "  ✓ 已全部完成"; break; fi
    echo "  --- attempt ${attempt}（已完成 ${DONE0:-0}/${NP}）$(date +%H:%M:%S)"
    hard_clean
    env DS41M_CP_SIZE="$CP" timeout 7200 bash tools/run_hit.sh --points "$PTS" \
        --noprof-only --prof-warmup 2 --iters 8 \
        --cards 0,1,2,3,4,5,6,7 -o "$OUT" > "$ROOT/noprof_cp${CP}_a${attempt}.log" 2>&1
    RC=$?
    DONE=$(awk 'END{print NR-1}' "$OUT/results.tsv" 2>/dev/null || echo 0)
    echo "  CP${CP} a${attempt}: rc=$RC 完成=${DONE:-0}/${NP} $(date +%H:%M:%S)"
    [ "${DONE:-0}" -ge "$NP" ] && break
    # 崩溃后必须彻底清，否则下一轮只起部分 rank（实测只起 rank 6）
    echo "     （崩溃，强制清理后重试）"
    hard_clean
  done
done

echo ""
echo "########## 汇总 $(date +%H:%M:%S) ##########"
python3.11 tools/gen_table.py --root "$ROOT" \
    --out "$ROOT/grid_noprof.tsv" --md "$ROOT/grid_noprof.md" 2>&1 | tail -4
echo "GRID_ALL_DONE $(date +%H:%M:%S)"

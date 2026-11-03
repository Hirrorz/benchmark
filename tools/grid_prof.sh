#!/bin/bash
# prof 轮网格：3 CP 全部采 rank0，采完一个 CP 立即 export→打包→删原始，最后统一出表。
# 用法: grid_prof.sh <ROOT>
set -u
BASE=/mnt/xds/sfs/l00972766
ROOT=$1
cd "$BASE/ds41_mini" || exit 2
source env_pack.sh >/dev/null 2>&1
export DS41M_CP_BUCKET=0          # ★ 采集必须 0（DeepEP 与 ALOP 不兼容）
export DS41M_CP_HIT=1
export DS41M_MOE_CAST_INTO_SB=1
unset LD_PRELOAD

hard_clean() {
  for round in 1 2 3; do
    PIDS=$(pgrep -f 'launch_mini|core/runner\.py' 2>/dev/null)
    [ -z "$PIDS" ] && break
    for p in $PIDS; do kill -9 "$p" 2>/dev/null; done
    sleep 4
  done
  sleep 3
  for w in $(seq 1 12); do
    B=$(npu-smi info 2>/dev/null | grep -oE '[0-9]+ */ *98304' | sed 's|/.*||' | tr -d ' ' \
        | awk '$1+0>8000' | wc -l)
    [ "${B:-0}" -eq 0 ] && break
    sleep 5
  done
  echo "     [clean] 高占用卡=${B:-0}"
}

echo "########## PROF GRID START $(date '+%m-%d %H:%M:%S') ##########"
for CP in 1 2 4; do
  PTS="$ROOT/points_cp${CP}.json"
  [ -f "$PTS" ] || python3.11 tools/gen_grid.py --out "$PTS" --cp "$CP" --report-skipped
  NP=$(python3.11 -c "import json;print(len(json.load(open('$PTS'))))")
  OUT="$ROOT/prof_cp${CP}"
  echo ""
  echo "================= CP${CP} PROF: ${NP} 点位 $(date '+%H:%M:%S') ================="
  for attempt in 1 2 3 4 5 6 7 8; do
    DONE0=$(awk 'END{print NR-1}' "$OUT/results.tsv" 2>/dev/null || echo 0)
    [ "${DONE0:-0}" -ge "$NP" ] && { echo "  ✓ 已全部完成"; break; }
    # 空间守卫：低于 20 GB 就先 export 清理
    AVAIL=$(df -BG /mnt/xds/sfs 2>/dev/null | tail -1 | grep -oE '[0-9]+G' | head -1 | tr -d G)
    if [ "${AVAIL:-999}" -lt 20 ]; then
      echo "  ⚠ 可用 ${AVAIL}G < 20G → 先导出并清理"
      bash tools/prof_pack.sh "$OUT" "$ROOT" "$CP"
    fi
    echo "  --- attempt ${attempt}（已完成 ${DONE0:-0}/${NP}）$(date '+%H:%M:%S')"
    hard_clean
    env DS41M_CP_SIZE="$CP" timeout 21600 bash tools/run_hit.sh --points "$PTS" \
        --prof-ranks 0 --prof-warmup 2 --iters 8 \
        --cards 0,1,2,3,4,5,6,7 -o "$OUT" > "$ROOT/prof_cp${CP}_a${attempt}.log" 2>&1
    RC=$?
    DONE=$(awk 'END{print NR-1}' "$OUT/results.tsv" 2>/dev/null || echo 0)
    echo "  CP${CP} PROF a${attempt}: rc=$RC 完成=${DONE:-0}/${NP} $(date '+%H:%M:%S')"
    [ "${DONE:-0}" -ge "$NP" ] && break
    hard_clean
  done
  # 一个 CP 采完 → 立即导出并删原始
  echo "  --- CP${CP} 采完，导出 trace ---"
  bash tools/prof_pack.sh "$OUT" "$ROOT" "$CP"
done

echo ""
echo "########## 统一解析 + 出表 $(date '+%H:%M:%S') ##########"
python3.11 tools/gen_table.py --root "$ROOT" \
    --out "$ROOT/grid_prof.tsv" --md "$ROOT/grid_prof.md" 2>&1 | tail -6
echo "PROF_GRID_ALL_DONE $(date '+%m-%d %H:%M:%S')"

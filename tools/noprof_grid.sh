#!/bin/bash
# 只跑 noprof 轮（真实性能数据，几乎不占空间），各 CP 顺序执行。
set -u
BASE=/mnt/xds/sfs/l00972766
ROOT=$1
cd "$BASE/ds41_mini" || exit 2
source env_pack.sh >/dev/null 2>&1
export DS41M_CP_BUCKET=0 DS41M_CP_HIT=1 DS41M_MOE_CAST_INTO_SB=1
unset LD_PRELOAD

for CP in 1 2 4; do
  PTS="$ROOT/points_cp${CP}.json"
  python3.11 tools/gen_grid.py --out "$PTS" --cp "$CP" --report-skipped 2>&1 | tail -2
  NP=$(python3.11 -c "import json;print(len(json.load(open('$PTS'))))")
  echo ""
  echo "================= CP${CP}: ${NP} 点位 ================="
  for attempt in 1 2 3 4 5; do
    for p in $(pgrep -f "launch_mini" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
    sleep 5
    env DS41M_CP_SIZE="$CP" bash tools/run_hit.sh --points "$PTS" \
        --noprof-only --prof-warmup 2 --iters 8 \
        --cards 0,1,2,3,4,5,6,7 -o "$ROOT/noprof_cp${CP}" \
        > "$ROOT/noprof_cp${CP}_a${attempt}.log" 2>&1
    RC=$?
    DONE=$(awk 'END{print NR-1}' "$ROOT/noprof_cp${CP}/results.tsv" 2>/dev/null || echo 0)
    echo "  CP${CP} noprof a${attempt}: rc=$RC 完成=${DONE:-0}/${NP}"
    [ "${DONE:-0}" -ge "$NP" ] && break
  done
done
echo ""
echo "NOPROF_GRID_ALL_DONE"
for CP in 1 2 4; do
  echo "--- CP${CP} ---"
  awk -F'\t' 'NR>1{printf "  %-56s steady=%ss err=%s\n",$1,$2,$4}' \
      "$ROOT/noprof_cp${CP}/results.tsv" 2>/dev/null | head -6
done

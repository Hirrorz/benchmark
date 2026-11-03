#!/bin/bash
# 把一个 CP 目录下所有 PROF_* 目录批量 export 成 CSV → 打包 → 删原始。
# 用法: prof_pack.sh <CP_OUT_DIR> <ROOT> <CP>
set -u
OUT=$1
ROOT=$2
CP=$3
CSVDIR="$ROOT/trace_csv_cp${CP}"
mkdir -p "$CSVDIR"

echo "     [pack] CP${CP}: 扫描 PROF 目录…"
mapfile -t PROFS < <(find "$OUT" -maxdepth 6 -name 'PROF_*' -type d 2>/dev/null | sort)
echo "     [pack] 找到 ${#PROFS[@]} 个 PROF 目录"
[ "${#PROFS[@]}" -eq 0 ] && { echo "     [pack] 无 trace，跳过"; return 0 2>/dev/null || exit 0; }

OK=0; FAIL=0
for P in "${PROFS[@]}"; do
  # 用点位 tag 作 CSV 文件名
  tag=$(echo "$P" | sed -E 's|.*/(cp[0-9]dp[0-9]ep[0-9]_L[0-9]+_hit[0-9.]+)_batch.*|\1|')
  [ "$tag" = "$P" ] && tag=$(basename "$(dirname "$(dirname "$P")")")
  DST="$CSVDIR/$tag"
  if [ -f "$DST/.done" ]; then OK=$((OK+1)); continue; fi
  mkdir -p "$DST"
  bash -lc "msprof --export=on --type=text --output='$P'" >/dev/null 2>&1
  if [ -d "$P/mindstudio_profiler_output" ]; then
    cp -a "$P/mindstudio_profiler_output/." "$DST/" 2>/dev/null
    rm -rf "$P/mindstudio_profiler_output"
    touch "$DST/.done"
    OK=$((OK+1))
  else
    FAIL=$((FAIL+1))
  fi
  # 单个点位导出后立刻删原始目录，控制空间
  rm -rf "$P" 2>/dev/null
done
echo "     [pack] export 成功=$OK 失败=$FAIL"

# 打包 CSV
TGZ="$ROOT/trace_csv_cp${CP}.tar.gz"
tar czf "$TGZ" -C "$ROOT" "trace_csv_cp${CP}" 2>/dev/null && \
  echo "     [pack] ✓ $TGZ ($(du -sh "$TGZ" 2>/dev/null | cut -f1))"
df -BG /mnt/xds/sfs 2>/dev/null | tail -1 | awk '{print "     [pack] 可用 "$4}'

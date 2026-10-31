#!/bin/bash
# ============================================================================
# pack_trace_outputs.sh — 打包 trace 解析产物（含清单 + sha256）
#
# 目录规则（2026-10-29 用户口径，见 PERF_RULES §5 / TRACE_GUIDE §1.3）：
#   <PACK>/<并行策略>_L<Lk>_hit<h>_b<批>_r<0|1>_prefill/
#       README.txt  msprof_<ts>.json  task_time_<ts>.csv  op_summary_<ts>.csv
#       op_statistic_<ts>.csv  api_statistic_<ts>.csv
#       communication_statistic_<ts>.csv  step_trace_<ts>.json
#   —— 文件夹名 = 扁平点位名，文件**直接**放在它下面（不再套 mindstudio_profiler_output/）。
#   `msprof --export=on` 非幂等 ⇒ 每个前缀只取**最新时间戳**那一份（tools/pack_flat.py）。
#
# 用法：
#   bash tools/pack_trace_outputs.sh <点位根目录> [CP]
#     <点位根目录>  形如 .../cp2/L1k-1m（下面有 */prefill/PROF_*/mindstudio_profiler_output）
#     [CP]          并行策略里的 CP（默认 1；点位目录里没有 config 时必须给，例：2）
#
# 产物：<根目录>/PACK_<UTC时间戳>/
#     TRACE_FLAT_<ts>.tar.gz      扁平点位包（上面那个布局，主体）
#     TRACE_REPORTS_<ts>.tar.gz   顶层报表（forward_times/detail、op_top、api_top、
#                                 SUMMARY、TRACE_MATRIX/TRACE_MODEL/TRACE_POINTS）
#     MANIFEST.txt                点位 → 文件夹名 → 文件清单
#     SHA256SUMS
# ============================================================================
set -u
ROOT=${1:?用法: pack_trace_outputs.sh <点位根目录> [CP]}
CP=${2:-1}
[ -d "$ROOT" ] || { echo "✗ 目录不存在：$ROOT" >&2; exit 2; }

HERE=$(cd "$(dirname "$0")" && pwd)
TS=$(date -u +%Y%m%d_%H%M%S)
OUT="$ROOT/PACK_$TS"
STAGE="$OUT/STAGE"
mkdir -p "$STAGE"

# ── ① 扁平点位包（主体）──────────────────────────────────────────────
if ! python3 "$HERE/pack_flat.py" "$ROOT" -o "$STAGE" --cp "$CP"; then
  echo "✗ 扁平打包失败" >&2; exit 3
fi
cp -f "$STAGE/MANIFEST.txt" "$OUT/MANIFEST.txt"
rm -f "$STAGE/MANIFEST.txt"          # 清单不进包（否则和点位文件夹混在一起）
tar -czf "$OUT/TRACE_FLAT_$TS.tar.gz" -C "$STAGE" .
rm -rf "$STAGE"
echo "✓ 扁平点位包：$OUT/TRACE_FLAT_$TS.tar.gz"

# ── ② 顶层报表（附件）──────────────────────────────────────────────
RSUB=""
for cand in "" "PARSED/"; do
  [ -f "$ROOT/${cand}forward_times.csv" ] && { RSUB="$cand"; break; }
done
REPORTS=()
for f in forward_times.csv forward_detail.csv op_top.csv api_top.csv SUMMARY.md \
         TRACE_MATRIX.md TRACE_POINTS.csv TRACE_MODEL.md; do
  [ -f "$ROOT/${RSUB}$f" ] && REPORTS+=("${RSUB}$f")
done
if [ ${#REPORTS[@]} -gt 0 ]; then
  tar -czf "$OUT/TRACE_REPORTS_$TS.tar.gz" -C "$ROOT" "${REPORTS[@]}"
  echo "✓ 报表：$OUT/TRACE_REPORTS_$TS.tar.gz（${#REPORTS[@]} 个文件）"
else
  echo "[warn] 没有顶层报表（先跑 tools/trace_report.py）"
fi

# ── ③ 清单 + 校验和 ────────────────────────────────────────────────
{
  echo "# trace 打包清单  $(date -u +%FT%TZ)"
  echo "# 根目录：$ROOT"
  echo "# 布局：<扁平名>/ = README.txt + 七件套（msprof 导出原样，每前缀取最新）"
  echo "# 命名：cp<CP>dp<DP>ep<EP>_L<Lk>_hit<h>_b<批>_r<0|1>_prefill"
  echo "# 报表：${REPORTS[*]:-（无）}"
  echo "# 口径：trace 设备时间线（相邻 forward 起点差的中位、剔除首个）；墙钟 results.tsv 不可用"
  echo "#"
  tail -n +4 "$OUT/MANIFEST.txt" 2>/dev/null
} > "$OUT/MANIFEST.txt.new" && mv "$OUT/MANIFEST.txt.new" "$OUT/MANIFEST.txt"

( cd "$OUT" && sha256sum ./*.tar.gz > SHA256SUMS 2>/dev/null )
ls -la "$OUT"
du -sh "$OUT"/*.tar.gz 2>/dev/null

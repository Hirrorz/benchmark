#!/bin/bash
# ============================================================================
# run_cp1_matrix_pipeline.sh — CP1 全量矩阵：**采集 → 解析 → 汇总 → 打包** 一条龙
#
# 为什么重跑：CP1 上一版（SWEEP205 / 2026-10-27）是**旧代码**跑的数，之后做了
#   decoder-replay、尾段入图、绑定直发、MoE 后端等优化 ⇒ 数不可比，需要用当前代码重采。
#
# 口径（与 CP2/CP4 完全一致，可直接对照）：
#   · 点集：几何闸 + `T_pass ≤ MATRIX_TPASS_CAP`（默认 104998）→ 205 点
#   · **一个进程跑完全部点**（一次预热；点位间靠 core/memlife.py 释放，不重启）
#   · 采集：`--pure-log` + prof rank0（trace 口径）；失败隔离/watchdog/续跑默认开
#   · 性能数：**trace 设备时间线**（相邻 forward 起点差的中位、剔除首个）；
#     `results.tsv` 是墙钟（含 msprof stop，小 T 虚高 1.4–25×），**只看不动**
#
# 用法：
#   bash tools/run_cp1_matrix_pipeline.sh <OUT_ROOT> [--no-collect] [--no-parse]
#                                                [--no-pack] [--cp N]
#   例（推荐：与 CP2/CP4 用**同一个 OUT_ROOT**，跑完会自动出 CP2/CP4 vs CP1 对照）：
#     bash tools/run_cp1_matrix_pipeline.sh \
#       /mnt/xds/sfs/l00972766/tmp/COLLECT_CP2CP4_1PROC_1029_0447 --cp 1
#   或独立目录（只出 CP1 自己的矩阵/模型，不做对照）：
#     bash tools/run_cp1_matrix_pipeline.sh /mnt/xds/sfs/l00972766/tmp/COLLECT_CP1_1029_0900
#
#   断点续跑：重跑同一条命令即可 —— 已完成的点位（results.tsv）会跳过，
#   已导出的 point 也不会重导出（trace_report 复用 mindstudio_profiler_output）。
#
# 产物：
#   <OUT_ROOT>/cp<N>/L1k-1m/                      采集（每点一个 `(L_hit)_r1/prefill/PROF_*`）
#   <OUT_ROOT>/cp<N>/L1k-1m/PARSED/               解析（forward_times/detail、op_top、api_top、SUMMARY）
#   <OUT_ROOT>/cp<N>/L1k-1m/PARSED/TRACE_MATRIX.md  L×hit 矩阵
#   <OUT_ROOT>/cp<N>/L1k-1m/PARSED/TRACE_MODEL.md   时间模型（同 T 一致性/拟合/残差/idle）
#   <OUT_ROOT>/cp<N>/L1k-1m/PACK_<ts>/            TRACE_FLAT + TRACE_REPORTS + MANIFEST + SHA256SUMS
# ============================================================================
set -u
BASE=/mnt/xds/sfs/l00972766
DS=$BASE/ds41_mini

OUT_ROOT=""; CP=1; DO_COLLECT=1; DO_PARSE=1; DO_PACK=1
while [ $# -gt 0 ]; do
  case "$1" in
    --no-collect) DO_COLLECT=0 ;;
    --no-parse)   DO_PARSE=0 ;;
    --no-pack)    DO_PACK=0 ;;
    --cp)         CP=${2:-1}; shift ;;
    --cp=*)       CP=${1#--cp=} ;;
    -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
    -*)           echo "✗ 未知参数 $1" >&2; exit 2 ;;
    *)            OUT_ROOT=$1 ;;
  esac
  shift
done
[ -n "$OUT_ROOT" ] || { echo "用法: $0 <OUT_ROOT> [--no-collect|--no-parse|--no-pack] [--cp N]" >&2; exit 2; }
case "$CP" in ''|*[!0-9]*) echo "✗ --cp 要正整数" >&2; exit 2 ;; esac

cd "$DS" || { echo "✗ 进不去 $DS" >&2; exit 2; }
source env_pack.sh >/dev/null 2>&1
mkdir -p "$OUT_ROOT" || exit 2
export TMPDIR=${DS41M_TMPDIR:-$OUT_ROOT/tmp}; mkdir -p "$TMPDIR"

LOGDIR=$OUT_ROOT/cp$CP/L1k-1m
mkdir -p "$LOGDIR"
LOG=$OUT_ROOT/pipeline_cp${CP}_$(date -u +%m%d_%H%M).log
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "== CP$CP 一条龙：OUT=$OUT_ROOT"
say "   collect=$DO_COLLECT parse=$DO_PARSE pack=$DO_PACK   TMPDIR=$TMPDIR"
say "   代码版本检查："
{ grep -c "_load_done_tags" core/points_runner.py | xargs -I{} echo "   points_runner 失败隔离/续跑: {} 处";
  grep -c "POINT_CACHES" core/points_runner.py core/memlife.py | sed 's/^/   /';
  grep -c "\[mem\] " core/launch_mini.py | xargs -I{} echo "   pure-log 放行 [mem]: {}";
} 2>&1 | tee -a "$LOG"

# ── ① 采集（一进程 205 点）─────────────────────────────────────────
if [ "$DO_COLLECT" = 1 ]; then
  say "== ① 采集：CP=$CP，几何闸 + T_pass ≤ ${MATRIX_TPASS_CAP:-104998}，一个进程跑完"
  MATRIX_L_GROUP=0 DS41M_TMPDIR="$TMPDIR" \
    bash tools/collect_dp_cp_matrix.sh "$OUT_ROOT" "$CP" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  say "   采集 rc=$rc（0=全过；9=有跳过的点位，重跑本脚本可续跑）"
  [ "$rc" = 0 ] || say "   ⚠ 采集非 0 退出，继续后面的解析（已完成的点位仍然可解析）"
else
  say "== ① 采集：跳过（--no-collect）"
fi

ROOT=$OUT_ROOT/cp$CP/L1k-1m
NP=$(ls -d "$ROOT"/*_r1 2>/dev/null | wc -l)
say "   点位目录：$NP 个"
[ "$NP" -gt 0 ] || { say "✗ 没有点位目录，停"; exit 3; }

# ── ② 解析（msprof 导出 + forward 切分）────────────────────────────
if [ "$DO_PARSE" = 1 ]; then
  say "== ② 解析：msprof --export=on → PARSED/（已导出的点自动复用）"
  TMPDIR="$TMPDIR" nice -n 19 python3.11 tools/trace_report.py "$ROOT" \
      --export --cp "$CP" -o "$ROOT/PARSED" 2>&1 | tee -a "$LOG"
  say "   解析 rc=${PIPESTATUS[0]}"
else
  say "== ② 解析：跳过（--no-parse）"
fi

# ── ③ 汇总（矩阵 + 模型 [+ 与其他 CP 对照]）────────────────────────
if [ -f "$ROOT/PARSED/forward_times.csv" ]; then
  say "== ③ 汇总"
  python3 tools/trace_matrix.py "$ROOT/PARSED" 2>&1 | tee -a "$LOG" >/dev/null
  python3 tools/trace_model.py "$ROOT/PARSED/TRACE_POINTS.csv" 2>&1 | tee -a "$LOG" >/dev/null
  say "   → $ROOT/PARSED/TRACE_MATRIX.md、TRACE_MODEL.md"
  # 和同 OUT_ROOT 下其它 CP 的解析结果对拍（有就做）
  for other in "$OUT_ROOT"/cp*/L1k-1m/PARSED/TRACE_POINTS.csv; do
    [ -f "$other" ] || continue
    [ "$other" = "$ROOT/PARSED/TRACE_POINTS.csv" ] && continue
    ocp=$(echo "$other" | sed -n 's#.*/cp\([0-9]*\)/L1k-1m/.*#\1#p')
    say "   对照：CP$CP vs CP$ocp"
    python3 tools/compare_cp.py "$other" "$ROOT/PARSED/TRACE_POINTS.csv" \
        -o "$ROOT/PARSED/COMPARE_CP${ocp}_vs_CP${CP}.md" \
        --labels "CP$ocp" "CP$CP" 2>&1 | tail -3 | tee -a "$LOG"
  done
else
  say "== ③ 汇总：跳过（没有 forward_times.csv）"
fi

# ── ④ 打包 ────────────────────────────────────────────────────────
if [ "$DO_PACK" = 1 ]; then
  say "== ④ 打包（扁平点位包 + 报表 + 清单 + sha256）"
  bash tools/pack_trace_outputs.sh "$ROOT" "$CP" 2>&1 | tee -a "$LOG" | tail -8
  PK=$(ls -d "$ROOT"/PACK_* 2>/dev/null | tail -1)
  say "   包：${PK:-（未生成）}"
else
  say "== ④ 打包：跳过（--no-pack）"
fi

say "== 完成。日志：$LOG"
[ -f "$ROOT/PARSED/TRACE_MATRIX.md" ] && say "   矩阵：$ROOT/PARSED/TRACE_MATRIX.md"
[ -f "$ROOT/PARSED/TRACE_MODEL.md" ] && say "   模型：$ROOT/PARSED/TRACE_MODEL.md"
say "   提醒：性能口径看 TRACE_*（trace 设备时间线）；results.tsv 是墙钟，含 msprof stop，别用"

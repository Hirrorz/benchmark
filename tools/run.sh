#!/bin/bash
# ============================================================================
# 唯一跑测入口（2026-10-26 合并）：一个命令，三种模式
#
#   bash tools/run.sh --weights synth  <run_hit 参数>    # 合成权重（默认）
#   bash tools/run.sh --weights real   <run_real 参数>   # 真权重 + 精度口径
#   bash tools/run.sh --dry-run --config <yaml>          # 只打印算子序列（不占卡）
#
# 说明：`--weights synth|real` 之后的所有参数**原样透传**给对应后端
#   （后端仍是 tools/run_hit.sh / tools/run_real.sh，CLI 各自见 --help）。
#   这是唯一对外入口；两个后端只作为内部实现，不再直接对外宣传。
# ============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd -P)
ROOT=$(cd "$HERE/.." && pwd -P)

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
  echo
  echo "后端参数：--weights synth 看 tools/run_hit.sh --help；--weights real 看 tools/run_real.sh --help"
  exit "${1:-0}"
}

WEIGHTS=synth
WEIGHTS_SET=0
MODE=run
DRY_CFG=""
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    # --help 不在这里吞掉：原样透传给后端（只有单独 `run.sh --help` 才显示本页）
    --weights) WEIGHTS=${2:-synth}; WEIGHTS_SET=1; shift 2 ;;
    --weights=*) WEIGHTS=${1#*=}; WEIGHTS_SET=1; shift ;;
    --dry-run) MODE=dry; shift ;;
    --config) DRY_CFG=${2:-}; shift 2 ;;
    --config=*) DRY_CFG=${1#*=}; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [ "$MODE" = run ] && [ "$WEIGHTS_SET" = 0 ] && [ ${#ARGS[@]} -eq 1 ] \
   && { [ "${ARGS[0]}" = "-h" ] || [ "${ARGS[0]}" = "--help" ]; }; then
  usage 0
fi

if [ "$MODE" = dry ]; then
  [ -n "$DRY_CFG" ] || { echo "✗ --dry-run 需要 --config <yaml>" >&2; exit 2; }
  exec python3.11 "$ROOT/core/run.py" "$DRY_CFG" --dry-run "${ARGS[@]+"${ARGS[@]}"}"
fi

case "$WEIGHTS" in
  synth|fake|synthetic) echo "[run.sh] 后端 = tools/run_hit.sh（合成权重）" >&2
                        exec bash "$HERE/run_hit.sh" "${ARGS[@]+"${ARGS[@]}"}" ;;
  real|true|weights)    echo "[run.sh] 后端 = tools/run_real.sh（真权重）" >&2
                        exec bash "$HERE/run_real.sh" "${ARGS[@]+"${ARGS[@]}"}" ;;
  *) echo "✗ --weights 只支持 synth|real（收到：$WEIGHTS）" >&2; exit 2 ;;
esac

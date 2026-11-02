#!/bin/bash
# 一键验收（CPU 侧）：把今天所有"逐条采纳"的判据跑一遍并汇总。
#
#   bash tools/verify_all.sh          # 只跑 CPU（不需要卡，任何机器都能跑）
#   bash tools/verify_all.sh --npu    # 额外提交一次 NPU 矩阵（走 tmp/npu_exec 执行器）
#
# 退出码：0 = 全绿；非 0 = 有 FAIL（并打印失败的套件名）。
# 设计要点：所有套件都**不依赖 NPU**（torch 只在 CPU 侧用，且关掉 torch_npu 自动加载），
#          所以 099/208/办公机只要 `env_pack.sh` 能 source 就能跑。
set -u
cd "$(dirname "$0")/.." || exit 2
BASE=$(pwd)
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="/tmp/deps311:$BASE/core:$BASE/meta:$BASE/layout:$BASE/ops:$BASE/cp:$BASE/trace:$BASE/weights"

SUITES="test_hit_geometry test_cp_layout test_cp_state test_batch_replay
        test_cp_plan test_metadata_layout test_meta_single_source test_cp_state_offsets"

echo "===== 一键验收（CPU）  $(date '+%F %T') ====="
echo "tree=$BASE"
echo
printf "%-26s %-4s %-6s %-6s\n" 套件 rc PASS FAIL
printf "%-26s %-4s %-6s %-6s\n" "--------------------------" -- ---- ----
total_fail=0
failed=""
for t in $SUITES; do
  if [ ! -f "tools/$t.py" ]; then
    printf "%-26s %-4s %-6s %-6s\n" "$t" "-" "-" "-"
    failed="$failed $t(缺失)"
    total_fail=$((total_fail + 1))
    continue
  fi
  python3.11 "tools/$t.py" > "/tmp/va_$t.log" 2>&1
  rc=$?
  p=$(grep -c '^PASS' "/tmp/va_$t.log" 2>/dev/null || true)
  f=$(grep -c '^FAIL' "/tmp/va_$t.log" 2>/dev/null || true)
  printf "%-26s %-4s %-6s %-6s\n" "$t" "$rc" "${p:-0}" "${f:-0}"
  if [ "$rc" != "0" ] || [ "${f:-0}" != "0" ]; then
    failed="$failed $t"
    total_fail=$((total_fail + ${f:-1}))
  fi
done

echo
echo "合计 FAIL=$total_fail"
if [ -n "$failed" ]; then
  echo "✗ 失败套件：$failed"
  for t in $failed; do
    t=${t%%(*}
    echo "--- $t 末尾 ---"
    tail -4 "/tmp/va_$t.log" 2>/dev/null
  done
  exit 1
fi
echo "✓ 全部 CPU 判据通过"

# 报告关键不变量（把"今天的结论"变成可复核输出）
python3.11 - <<'PY'
import os, sys
BASE = os.getcwd()
for p in ("cp", "layout", "core", "meta", "ops", "trace"):
    sys.path.insert(0, os.path.join(BASE, p))
try:
    import cp_plan
    v = cp_plan.build_plan([8192], 2, 0)
    s = cp_plan.gather_stats(v, mode="replay")
    print(f"[不变量] cp2/L=8192 尾部窗口需 {s['rows_needed']} 行 / 块 "
          f"{s['blocks_needed']}/{s['blocks_total']}（B5 已撤回'可省 87%'，见 WORKPLAN）")
except Exception as e:
    print(f"[不变量] cp_plan 跳过：{type(e).__name__}: {e}")
PY

if [ "${1:-}" = "--npu" ]; then
  echo
  echo "===== 额外：提交 NPU 矩阵（执行器） ====="
  STAGE=/mnt/xds/sfs/l00972766/tmp/npu_exec/stage
  if [ -d "$STAGE" ]; then
    NAME="verify_$(date +%H%M%S).sh"
    cp "/mnt/xds/sfs/l00972766/tmp/npu_exec/done/239_reg_matrix.sh" "$STAGE/$NAME" 2>/dev/null \
      && chmod +x "$STAGE/$NAME" \
      && mv "$STAGE/$NAME" "/mnt/xds/sfs/l00972766/tmp/npu_exec/queue/$NAME" \
      && echo "已提交：$NAME（结果看 /mnt/xds/sfs/l00972766/tmp/REG_*）" \
      || echo "⚠️ 提交失败（执行器/模板不可用）"
  else
    echo "⚠️ 没有执行器目录，跳过 NPU"
  fi
fi
exit 0
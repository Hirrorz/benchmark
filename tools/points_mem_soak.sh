#!/bin/bash
# ============================================================================
# points_mem_soak.sh — 点位显存累积回归（一个进程连跑多个点位）
#
# 目的：验证「一次预热跑多点位」**不再留残**。每个点位边界由 core/memlife.py
#   打印一行 `[mem] … allocated=… 留存=…`；留存（相对低水位）超过
#   `DS41M_MEM_GUARD_GB`（默认 4 GiB）时，自动按
#   「模块容器（谁钉住的） / 张量形状（多大）」归因。
#
# 背景：099 实测 CP2 单进程跑到 point 192（L=256k、T=2761）峰值 66.9 GiB，
#   同点全新进程只要 ~2 GiB ⇒ 65 GiB 是前面点位留下的**活张量**
#   （OOM 现场 allocated == active，不是碎片）。见 WORKPLAN 2026-10-29 与
#   NOTES/2026-10-29-points-memory-retention.md。
#
# 用法：
#   bash tools/points_mem_soak.sh [OUT] [CP] [SPECS…]
#   默认：OUT=/tmp/memsoak_<ts>  CP=2
#         SPECS = 1k@0.99 32k@0.98 64k@0.98 128k@0.98 256k@0.98 512k@0.9 256k@0.98 1k@0.99
#   （长度递增再回小点：泄漏会把大 L 的占用带进小点，正是 099 的现场）
#
# 产物：$OUT/soak.log（含每点 [mem] 行）、$OUT/results.tsv
# 退出码：0 = 全部点位完成且末点留存 < 阈值；1 = 有泄漏/失败（看 soak.log）
# ============================================================================
set -u
BASE=/mnt/xds/sfs/l00972766
cd "$BASE/ds41_mini" || exit 2
source env_pack.sh >/dev/null 2>&1

TS=$(date -u +%m%d_%H%M)
OUT=${1:-$BASE/tmp/memsoak_$TS}
CP=${2:-2}
shift $(( $# >= 2 ? 2 : $# ))
SPECS=${*:-"1k@0.99 32k@0.98 64k@0.98 128k@0.98 256k@0.98 512k@0.9 256k@0.98 1k@0.99"}
GUARD=${DS41M_MEM_GUARD_GB:-4}
mkdir -p "$OUT"
export TMPDIR=${DS41M_TMPDIR:-$OUT/tmp}; mkdir -p "$TMPDIR"

# 点位表："256k@0.98/b1/r1"
PLIST=""
for s in $SPECS; do PLIST="$PLIST ${s}/b1/r1"; done

echo "== mem soak：CP$CP · ${#} 个规格：$SPECS"
echo "   out=$OUT  guard=${GUARD}GiB  $(date -u +%FT%TZ)"

bash "$BASE/tools/wait_npu_idle.sh" 600 > "$OUT/wait_idle.log" 2>&1 || \
  echo "  [warn] 未等到空闲基线，照跑（见 $OUT/wait_idle.log）"

DS41M_MEM_AUDIT=1 \
DS41M_MEM_GUARD_GB="$GUARD" \
DS41M_CP_SIZE="$CP" \
DS41M_CP_HIT=1 \
  bash tools/run_hit.sh --points-spec "$PLIST" --pure-log --timeout 5400 -o "$OUT" \
  2>&1 | tee "$OUT/soak.log"
rc=${PIPESTATUS[0]}

echo
echo "== 每点留存（末点最小/最大）："
grep -o '\[mem\].*' "$OUT/soak.log" | tail -20
LAST=$(grep -o '留存=[-+0-9.]*' "$OUT/soak.log" | tail -1 | cut -d= -f2)
echo "== 末点留存 = ${LAST:-?} GiB（阈值 $GUARD）"
if [ "$rc" != 0 ]; then
  echo "✗ 点位运行失败 rc=$rc（见 $OUT/soak.log）"; exit 1
fi
awk -v v="$LAST" -v g="$GUARD" 'BEGIN{exit !(v>g || v<-g)}' && {
  echo "✗ 留存超阈值 ⇒ 仍有跨点位占用（看 soak.log 的 ⚠️ 归因行）"; exit 1; }
echo "✅ 留存 ${LAST} GiB ≤ ${GUARD} GiB：点位边界已无累积"
exit 0

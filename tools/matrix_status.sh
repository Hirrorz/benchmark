#!/bin/bash
# 采集矩阵进度速查
# 用法: bash tools/matrix_status.sh [OUT_ROOT]   （默认取最新的 /workspace/tmp/COLLECT_DPCP_*）
set -u
D=${1:-$(ls -dt /workspace/tmp/COLLECT_DPCP_*/ 2>/dev/null | head -1)}
D=${D%/}
[ -d "$D" ] || { echo "找不到输出目录（给个参数: OUT_ROOT）" >&2; exit 2; }
echo "== $D =="
echo "总目录启动: $(stat -c %y "$D" | cut -d. -f1)   现在: $(date '+%F %T')"
echo
tot=0; done_n=0; fail_n=0
for cp in "$D"/cp*; do
  [ -d "$cp" ] || continue
  for L in "$cp"/L*; do
    [ -d "$L" ] || continue
    case "$(basename "$L")" in *.*) continue ;; esac   # 跳过 .attempt/.failed/.aborted 等旁路目录
    tot=$((tot+1)); st="运行中"
    [ -f "$L/.done" ] && { st="完成"; done_n=$((done_n+1)); }
    [ -f "$L/.failed" ] && { st="失败"; fail_n=$((fail_n+1)); }
    n=$(grep -oE '\[point [0-9]+/[0-9]+\]' "$L/collect.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+/[0-9]+')
    # 时延口径：未采集的 rank1（`[r1] 纯 forward 吞吐`），不用 rank0 的 每forward（带 prof）
    src="$L/driver_latest.log"; [ -f "$src" ] || src="$L/collect.log"
    last=$(grep -oE '\[r1\]   纯 forward 吞吐：[0-9.]+ ms/forward' "$src" 2>/dev/null | tail -1 \
             | sed -E 's/.*：([0-9.]+) ms.*/r1=\1ms/')
    [ -f "$L/.contended" ] && st="$st/争用"
    printf '%-22s %-12s %-10s %s\n' "$(basename "$cp")/$(basename "$L")" "$st" "${n:+点 $n}" "$last"
  done
done
echo
echo "长度块: $done_n 完成 / $fail_n 失败 / $tot 已建 (块数 = 长度档数/分组；默认 20+20=40)"
echo "失败项: $(ls "$D"/cp*/L*/.failed 2>/dev/null | grep -Ev '/L[^/]*\.' | tr '\n' ' ')"

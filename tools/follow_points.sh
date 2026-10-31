#!/bin/bash
# follow_points.sh — 实时跟踪采集矩阵的"逐点行"，跨块自动切换
#
# 用法：bash tools/follow_points.sh [OUT_ROOT]
#   默认取最新的 /workspace/tmp/COLLECT_DPCP_* 目录。
# 说明：采集器的块级输出在 run.out；run_hit 的逐点行写在
#       <OUT_ROOT>/cpX/LY/collect.log（每个 (CP,L) 一个文件）。
#       本脚本始终跟随**最新修改**的那个 collect.log，块切换时自动换文件。
set -u
ROOT=${1:-$(ls -dt /workspace/tmp/COLLECT_DPCP_*/ 2>/dev/null | head -1)}
ROOT=${ROOT%/}
[ -d "$ROOT" ] || { echo "找不到输出目录: $ROOT" >&2; exit 2; }

cur=""; off=0
echo "== 跟随 $ROOT（Ctrl-C 退出）=="
while true; do
  f=$(ls -dt "$ROOT"/cp*/L*/collect.log 2>/dev/null | head -1)
  if [ -n "$f" ] && [ "$f" != "$cur" ]; then
    cur=$f; off=0
    echo "──── $(date '+%H:%M:%S') 切换块: ${cur#"$ROOT"/} ────"
  fi
  if [ -n "$cur" ]; then
    sz=$(stat -c %s "$cur" 2>/dev/null || echo 0)
    if [ "$sz" -gt "$off" ]; then
      tail -c +$((off + 1)) "$cur" 2>/dev/null \
        | grep --line-buffered -E '\[point |每forward|全部完成|有失败|trace 不齐|❌|✅ 全部'
      off=$sz
    fi
  fi
  sleep 2
done

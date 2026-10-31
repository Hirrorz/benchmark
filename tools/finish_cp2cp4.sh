#!/bin/bash
# ============================================================================
# finish_cp2cp4.sh — CP2 补跑（剩余点）+ CP4 全量，一次跑完
#
# 背景：099 上 CP2 单进程跑到 point 193（524288@0.8，L=512k、T=104998）时 8 个 rank
#   全部 OOM（84.9/91.75 GiB）——根因是**跨点位累积的活张量**（point 192 仍钉着 65 GiB），
#   不是点位本身不可行。修复在 `core/memlife.py`（点位边界统一释放 + 审计）。
#   本脚本分两段，**两段都是一个进程跑完**（不再分块重启）：
#
#   ① CP2 补跑：剩余 13 点（512k@0.8–0.99、1m@0.9–0.99），一进程跑完；
#      已有点位目录自动跳过（前面 192 点不动）。
#   ② CP4 全量：先**预检**两个 T_pass 最大的点（512k@0.8、1m@0.9），挂了的从点集里剔除；
#      其余点按采集器同一规则（几何闸 + T_pass ≤ MATRIX_TPASS_CAP，默认 104998）
#      一个进程跑完（SWEEP205 形式），最多重试 3 轮。
#   两段都带 `DS41M_MEM_AUDIT=1`：每个点位边界打印留存，超阈值自动归因。
#
# 失败隔离（2026-10-29）：点位之间**不再互相拖死** ——
#   · 单点异常 → 打 traceback、清显存、跳过，继续下一个点（返回码 9 表示"有跳过"）；
#   · 设备卡死 → watchdog（`DS41M_POINT_TIMEOUT`，默认 1800s）强制退出进程；
#   · 重跑同一条命令会自动跳过 `results.tsv` 里已完成的点位（`DS41M_POINT_REDO=1` 强制重采），
#     所以上面"最多重试 3 轮"只补没跑完的点，不再从头再跑 200 点。
#
# 用法：
#   bash tools/finish_cp2cp4.sh <OUT_ROOT>
#   <OUT_ROOT> 就用 CP2 那轮的根目录（例如
#   /mnt/xds/sfs/l00972766/tmp/COLLECT_CP2CP4_1PROC_1029_0447）
#
# 产物：
#   $OUT_ROOT/cp2/L1k-1m/         CP2 补跑的点位目录（与前面 192 点同目录）
#   $OUT_ROOT/cp4/L1k-1m/         CP4 全量
#   $OUT_ROOT/FINISH_SUMMARY.tsv  每点/每段的结果
# ============================================================================
set -u
OUT_ROOT=${1:?用法: finish_cp2cp4.sh OUT_ROOT}
BASE=/mnt/xds/sfs/l00972766
cd "$BASE/ds41_mini" || exit 2
source env_pack.sh >/dev/null 2>&1
export TMPDIR=${DS41M_TMPDIR:-$OUT_ROOT/tmp}
mkdir -p "$TMPDIR" "$OUT_ROOT"

CAP=${MATRIX_TPASS_CAP:-104998}
RUN_TMO=${MATRIX_RUN_TIMEOUT:-3600}

# 起跑前空气检查：还卡着的 CP2 进程必须先 Ctrl-C 掉，否则两个 8 卡任务会互相踩
if pgrep -f 'launch_mini.py' >/dev/null 2>&1; then
  echo "✗ 本容器还有 launch_mini.py 进程在跑（大概率是卡住的 CP2）。" >&2
  echo "  先在前台 Ctrl-C 掉那个任务，再运行本脚本。" >&2
  exit 3
fi
CP2_DIR=$OUT_ROOT/cp2/L1k-1m
CP4_DIR=$OUT_ROOT/cp4/L1k-1m
SUM=$OUT_ROOT/FINISH_SUMMARY.tsv
[ -f "$SUM" ] || printf 'phase\tcp\tpoint\trc\tnote\n' > "$SUM"

num_of(){ python3.11 - "$1" <<'PY'
import re, sys
s = sys.argv[1].lower()
m = re.fullmatch(r"(\d+(?:\.\d+)?)([km]?)", s)
print(int(float(m.group(1)) * {"":1, "k":1024, "m":1048576}[m.group(2)]))
PY
}

# 该点是否已有完整点位目录（有 PROF_* 即视为已采）
point_done(){ [ -d "$1" ] && compgen -G "$1/prefill/PROF_*" >/dev/null 2>&1; }

# 单点跑（独立进程）
run_one(){   # $1=CP $2=spec(L@hit) $3=OUTDIR $4=phase [$5=超时秒]
  local cp="$1" spec="$2" out="$3" ph="$4" tmo="${5:-$RUN_TMO}"
  local L="${spec%@*}" H="${spec#*@}" n dir
  n=$(num_of "$L")
  # 2026-10-29 起点位目录名 = 标准扁平名 `…_batch<b>_replay<on|off>_prefill`
  local _dp=$(( 8 / cp )); [ "$_dp" -gt 0 ] || _dp=1
  dir="$out/cp${cp}dp${_dp}ep8_L${L}_hit${H}_batch1_replayon_prefill"
  if point_done "$dir"; then
    echo "  [skip] 已有点位: $dir"
    printf '%s\t%s\t%s\t0\tskip-existing\n' "$ph" "$cp" "$spec" >> "$SUM"
    return 0
  fi
  bash "$BASE/tools/wait_npu_idle.sh" 900 > "$out/wait_idle_${L}_${H}.log" 2>&1 \
    || echo "  [warn] 未等到空闲基线，照跑（见 wait_idle_${L}_${H}.log）"
  echo "  → $(date -u +%H:%M:%S) 跑 $spec"
  DS41M_CP_SIZE=$cp DS41M_CP_HIT=1 timeout "$tmo" \
    bash tools/run_hit.sh --points-spec "${spec}/b1/r1" --pure-log \
      --timeout "$(( tmo - 300 ))" -o "$out" \
      > "$out/point_${L}_${H}.log" 2>&1
  local rc=$?
  printf '%s\t%s\t%s\t%s\t%s\n' "$ph" "$cp" "$spec" "$rc" \
    "$( [ $rc = 0 ] && echo ok || echo "见 point_${L}_${H}.log" )" >> "$SUM"
  echo "  ← $spec rc=$rc"
  [ "$rc" = 0 ] && sleep 20      # 等幽灵内存回落
  return $rc
}

# 多点一进程跑（用户口径：**不靠重启回避**累积；释放靠 core/memlife.py）
# $1=CP $2=OUTDIR $3=PHASE $4=spec 列表(空格分隔) [$5=超时秒]
run_multi(){
  local cp="$1" out="$2" ph="$3" specs="$4" tmo="${5:-$RUN_TMO}"
  mkdir -p "$out"
  bash "$BASE/tools/wait_npu_idle.sh" 900 > "$out/wait_idle_${ph}.log" 2>&1 \
    || echo "  [warn] 未等到空闲基线，照跑（见 wait_idle_${ph}.log）"
  local rc=1 att
  for att in 1 2 3; do
    echo "  → $(date -u +%H:%M:%S) 一进程跑 $(wc -w <<< "$specs") 点（attempt=$att）"
    DS41M_MEM_AUDIT=1 DS41M_MEM_GUARD_GB=${DS41M_MEM_GUARD_GB:-4} \
      DS41M_KEEP_LOG_FILES=$(( att > 1 )) \
      DS41M_POINT_TIMEOUT=${DS41M_POINT_TIMEOUT:-1800} \
      DS41M_POINT_MAX_FAILS=${DS41M_POINT_MAX_FAILS:-2} \
      DS41M_CP_SIZE=$cp DS41M_CP_HIT=1 timeout "$tmo" \
      bash tools/run_hit.sh --points-spec "$specs" --pure-log \
        --timeout "$(( tmo - 300 ))" -o "$out" \
        2>&1 | tee "$out/multi_${ph}_att${att}.log" | \
        grep --line-buffered -E '\[point |\[mem\]|✗|跳过|每forward|全部完成|trace 不齐|❌|✅'
    rc=${PIPESTATUS[0]}
    printf '%s\t%s\t%s点\t%s\tattempt=%s\n' "$ph" "$cp" "$(wc -w <<< "$specs")" "$rc" "$att" >> "$SUM"
    [ "$rc" = 0 ] && break
    echo "  ⚠ $ph attempt=$att rc=$rc"
  done
  return $rc
}

# 段尾：合成 .done（点位目录里已有 PROF_* 的计数）
mark_done(){  # $1=outdir $2=phase $3=cp
  local n; n=$(compgen -G "$1/(*)_r1/prefill/PROF_*" 2>/dev/null | wc -l)
  echo "points=$n cap=$CAP $(date -u +%FT%TZ) phase=$2" > "$1/.done"
  echo "  [$2] $1 现有 $n 个点位目录 → 已写 .done"
}

echo "=================================================================="
echo "== ① CP2 补跑：剩余 13 点（**一个进程**跑完）  $(date -u +%H:%M:%S)"
echo "=================================================================="
# ★ 2026-10-29（二改）：以前是"逐点独立进程"——那是**回避**显存累积，不是修。
#   现在释放交给 core/memlife.py 的点位边界 sweep；这里按原始口径一进程跑完。
#   顺序：小 L 先、最大 T 的 512k@0.8 最后（它最费时间）。
CP2_LEFT="512k@0.9 512k@0.95 512k@0.96 512k@0.97 512k@0.98 512k@0.99"
CP2_LEFT="$CP2_LEFT 1m@0.9 1m@0.95 1m@0.96 1m@0.97 1m@0.98 1m@0.99"
CP2_LEFT="$CP2_LEFT 512k@0.8"
_cp2_ok=0
if run_multi 2 "$CP2_DIR" cp2 "$(for s in $CP2_LEFT; do printf '%s/b1/r1 ' "$s"; done)" 14400; then
  _cp2_ok=1
fi
if [ "$_cp2_ok" = 1 ]; then echo "== CP2 补跑结束：✅ 13 点全过"; else echo "== CP2 补跑结束：❌ 有失败（见 $CP2_DIR/multi_cp2_att*.log）"; fi
mark_done "$CP2_DIR" cp2 2

echo
echo "=================================================================="
echo "== ② CP4 全量  $(date -u +%H:%M:%S)"
echo "=================================================================="
mkdir -p "$CP4_DIR"
# 点集：采集器同规则（几何闸 + T_pass≤CAP），再减去预检挂掉的点
SPEC4=$(MATRIX_TPASS_CAP="$CAP" python3.11 - <<'PY'
import os, re, sys
CAP = int(os.environ.get("MATRIX_TPASS_CAP", "104998") or 0)
LENS = "1k 2k 4k 8k 12k 16k 20k 24k 32k 40k 48k 56k 64k 80k 96k 112k 128k 256k 512k 1m".split()
HITS = "0 0.1 0.6 0.7 0.8 0.9 0.95 0.96 0.97 0.98 0.99".split()
def plen(s):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([km]?)", s.lower())
    return int(float(m.group(1)) * {"":1, "k":1024, "m":1048576}[m.group(2)])
toks = []
for Ls in LENS:
    L = plen(Ls)
    for h in HITS:
        hf = float(h); m = int(round((1 - hf) * L))
        tpass = L if hf <= 0 else m + 140
        if hf > 0 and tpass > L:      # 几何非法（1k@0.1）
            continue
        if CAP > 0 and tpass > CAP:   # 与 CP1 SWEEP205 口径对齐
            continue
        toks.append(f"{Ls}@{h}/b1/r1")
print(" ".join(toks))
PY
)
# 预检点单独跑（不管成败，都不再进批量，避免重复采集）
PRECHECK="512k@0.8 1m@0.9"
for risky in $PRECHECK; do run_one 4 "$risky" "$CP4_DIR" cp4_pre 1800 || true; done
for e in $PRECHECK; do
  SPEC4=$(echo "$SPEC4" | tr ' ' '\n' | grep -v "^${e}/b1/r1$" | tr '\n' ' ')
done
NPT=$(wc -w <<< "$SPEC4")
echo "== CP4 正式跑：${NPT} 点（预检点 ${PRECHECK} 已单独跑） $(date -u +%H:%M:%S)"

# ★ 2026-10-29（二改，用户口径）：**不再靠分块重启回避**显存累积。
#   一个进程连跑 200+ 点时，形状相关的进程级缓存会累积（099 实测 point 192 仍钉着
#   65 GiB 活张量；OOM 现场 allocated == active = 真活张量，不是碎片）。现在由
#   `core/memlife.py` 的登记表在**每个点位边界**统一释放，并由同模块审计
#   （`DS41M_MEM_AUDIT=1`）打印留存 + 归因 ⇒ 单进程跑完全部点位。
#   逃生舱（仅诊断）：`MATRIX_CHUNK_POINTS=N`（默认 0 = 不分块）切成 N 点一块。
read -r -a T4 <<< "$SPEC4"
N4=${#T4[@]}
CHUNK=${MATRIX_CHUNK_POINTS:-0}
case "$CHUNK" in ''|*[!0-9]*) CHUNK=0;; esac
[ "$CHUNK" -le 0 ] && CHUNK=$N4
i=0; ci=1; rc_all=0
while [ "$i" -lt "$N4" ]; do
  CS=("${T4[@]:i:CHUNK}"); i=$((i + CHUNK))
  cs="${CS[*]}"; cn=$(wc -w <<< "$cs")
  mk="$CP4_DIR/.chunk${ci}.done"
  if [ -f "$mk" ]; then echo "  [skip] CP4 第$ci 段已完成"; ci=$((ci+1)); continue; fi
  _end=$(( i < N4 ? i : N4 ))
  echo "  ── CP4 第$ci 段：${cn} 点（第 $((i - CHUNK + 1))–${_end} / $N4）$(date -u +%H:%M:%S)"
  rc=1
  for ATT in 1 2 3; do
    KEEP=0; [ "$ATT" -gt 1 ] && KEEP=1
    bash "$BASE/tools/wait_npu_idle.sh" 900 > "$CP4_DIR/wait_idle_chunk${ci}.log" 2>&1 || true
    DS41M_KEEP_LOG_FILES=$KEEP DS41M_MEM_AUDIT=1 DS41M_MEM_GUARD_GB=${DS41M_MEM_GUARD_GB:-4} \
      DS41M_POINT_TIMEOUT=${DS41M_POINT_TIMEOUT:-1800} \
      DS41M_POINT_MAX_FAILS=${DS41M_POINT_MAX_FAILS:-2} \
      DS41M_CP_SIZE=4 DS41M_CP_HIT=1 timeout 14400 \
      bash tools/run_hit.sh --points-spec "$cs" --pure-log --timeout 12600 -o "$CP4_DIR" \
      2>&1 | tee "$CP4_DIR/collect_chunk${ci}_att${ATT}.log" | \
      grep --line-buffered -E '\[point |\[mem\]|✗|跳过|每forward|全部完成|trace 不齐|❌|✅'
    rc=${PIPESTATUS[0]}
    echo "  cp4 第$ci 段 attempt=$ATT rc=$rc"
    printf 'cp4_chunk\t4\tchunk%s(%s点)\t%s\tattempt=%s\n' "$ci" "$cn" "$rc" "$ATT" >> "$SUM"
    [ "$rc" = 0 ] && break
  done
  if [ "$rc" = 0 ]; then
    echo "points=$cn $(date -u +%FT%TZ)" > "$mk"
  else
    rc_all=1; echo "rc=$rc $(date -u +%FT%TZ)" > "$CP4_DIR/.chunk${ci}.failed"
    echo "  ⚠ chunk$ci 失败（重跑本脚本会自动补这一块）"
  fi
  ci=$((ci+1))
done
if [ "$rc_all" = 0 ]; then
  mark_done "$CP4_DIR" cp4 4
else
  echo "rc=1 $(date -u +%FT%TZ) 有 chunk 失败" > "$CP4_DIR/.failed"
fi

echo
echo "=================================================================="
echo "== 全部结束 $(date -u +%H:%M:%S)"
echo "== 汇总：$SUM"
echo "=================================================================="
column -t -s$'\t' "$SUM" 2>/dev/null || cat "$SUM"

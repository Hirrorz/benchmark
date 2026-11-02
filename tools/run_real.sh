#!/bin/bash
# 真权重 + EP8 的 profiling 采集入口（"带精度"那套）：**自然路由、不负载均衡**、数值有效（能吐字）。
#
# 与合成入口 `tools/run_hit.sh` 的关系：CLI/日志/rounds/进度条**同约定**，但口径不同、互不影响，
# 两份脚本各自独立（合成那套是已验证产物，这里一个字都不改它）。
#
# 口径（写进 PARAMS 指纹）：
#   dp8/ep8/cp1 · 真权重(DeepSeek-V4.1-Flash-cann-mxfp8-mxfp4) · engram_offload=true + **engram_tp_size=8**
#   · **自然路由（无 elb/强均）** · weights.resident=true · host_cache=false · ulimit -l unlimited
#   · ids = parity/ids_16k.pt（**必给**；不给会退回强均路由 + 合成 id 取表）· DS41M_MOEMODE=deepgemm
#
# 用法：
#   bash tools/run_real.sh -L 16k                    # 默认 --mode both：prefill + hit 两个口径各一轮
#   bash tools/run_real.sh -L 16k --mode prefill     # 只普通 prefill（全长 16k，"能吐字"那条）
#   bash tools/run_real.sh -L 16k --mode hit --hit 0.96875   # 命中口径（stage A 真前缀 + stage B）
#   bash tools/run_real.sh -L 16k --parity           # 额外一轮 --collect-layers(rank0) + 逐层对拍
#   bash tools/run_real.sh -L 16k --iters 3 --warmup-iters 1 # trace 里跳过含装载的首 pass
#   bash tools/run_real.sh -L 16k --layers 15 --no-precision # 冒烟（★★ 必须 15：engram 层是 1 和 14）
#   bash tools/run_real.sh -L 16k --engram-host      # 逃生口：厂商路径炸了时退回主机取表（丢那两个算子）
#   bash tools/run_real.sh -L 16k --no-moe-chunk     # 关掉官方同款 MoE 分块（默认开：4096 token/块）
#   bash tools/run_real.sh -L 16k --alop-config <yaml>  # 换 ALOP 采集档位（默认 alop_mindstudio.yaml）
#   bash tools/run_real.sh -L 16k --no-alop-config   # 用 ALOP 内置默认（纯 C 级，只有 PROF_* 原始目录）
#   bash tools/run_real.sh -L 16k --dry-run / --help
#
# 采集引擎：**ALOP**（2026-10-16 用户口径：以后都用 ALOP）。`--prof` 即 ALOP；默认配置
# `configs/alop_mindstudio.yaml` = 纯 aclprof C 级（python_profiling 永久禁用）→ 产出 PROF_* 原始目录；
# MindStudio 用的 mindstudio_profiler_output/ 由 msprof --export 事后转出（见 PIPELINE.md）。
#
# 验收（每条都会自动打）：
#   ① 自然路由：**不该出现** `[elb/balroute]` 行；② 厂商 ElasticBuffer 就绪**恰好两次**（层 1、层 14 各一次；
#      之后每个 pass 都命中缓存 —— 每层一个专用 HCCL 组，见 `exec_engram._vendor_group`）；
#   ③ trace 里 `EngramFetch = EngramFetchWait = engram层数×iters×卡数`（层 1/14 → ×2）、
#      `MegaMoe = 层数×iters×卡数×块数`（块数 = ceil(每轮 token/moe_chunk_max_len)，官方同口径），
#      且 engram 在**另一条流**（侧流）；
#   ④ **精度 gate**：末位 token 与官方参考一致（`tools/check_next_token.py`，基准 id=396 ' that'）。
set -u

usage() { awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"; exit "${1:-0}"; }

# ── 默认值 ──
L_DEFAULT=16384; MODE=both; HIT=0.96875; ITERS=6; ITERS_SET=0; PROF_ITERS=2; WARMUP=0; LAYERS=0
IDS=/mnt/xds/sfs/l00972766/parity/ids_16k.pt
REF=/mnt/xds/sfs/l00972766/parity/ref
WEIGHTS=/mnt/xds/sfs/DeepSeek-V4.1-Flash-cann-mxfp8-mxfp4
CARDS=0,1,2,3,4,5,6,7; TIMEOUT=3600; CONSOLE=all
DO_PROF=1; DO_NOPROF=1; PRECISION=1; PARITY=0; CCOLLECT_RANKS=0; FORCE=0
RESIDENT=true; HOST_CACHE=false; MOEMODE=deepgemm; ENGRAM_HOST=0; DRY=0; LIST=0; MOE_CHUNK=1
AC_CACHE=1   # ★ ascendc 融合 MoE 权重预转换（只存一份，prefill 专用）：默认开，--no-ac-cache 关
# 采集引擎 = ALOP（runner 的 --prof 现在就是 ALOP；这张 yaml 决定"出 MindStudio trace"还是纯 C 级）
ALOP_CFG=configs/alop_mindstudio.yaml
BAR=1; BAR_FD=9; BAR_EVERY=30
KEEP=${DS41M_REPLAY_KEEP:-140}       # DeepInfer V4.1 KEEP_TOKENS 默认 140
OUT_DIR=${OUT_DIR:-}; PROF_DIR=${PROF_DIR:-}; SEQ=""

while [ $# -gt 0 ]; do
  case "$1" in
    -L|--seqs) shift
      while [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; do SEQ=${SEQ:-$1}; shift; done
      continue ;;
    -L=*|--seqs=*) SEQ=${1#*=} ;;
    --mode) MODE=${2:-both}; shift ;;
    --hit) HIT=${2:-0.96875}; shift ;;
    --iters) ITERS=${2:-6}; ITERS_SET=1; shift ;;
    --log-level) LOG_LEVEL=${2:-info}; shift ;;   # pure|error|warn|info|debug
    --pure-log) LOG_LEVEL=pure ;;
    --warmup-iters) WARMUP=${2:-0}; shift ;;
    --layers) LAYERS=${2:-0}; shift ;;
    --ids-file) IDS=${2:-}; shift ;;
    --ref) REF=${2:-}; shift ;;
    --weights-path) WEIGHTS=${2:-}; shift ;;
    --parity) PARITY=1 ;;
    --collect-ranks) CCOLLECT_RANKS=${2:-0}; shift ;;
    --resident) RESIDENT=true ;;
    --lazy) RESIDENT=false ;;
    --host-cache) HOST_CACHE=${2:-false}; shift ;;
    --moemode) MOEMODE=${2:-deepgemm}; shift ;;
    --engram-host) ENGRAM_HOST=1 ;;
    --no-moe-chunk) MOE_CHUNK=0 ;;
    --no-ac-cache) AC_CACHE=0 ;;          # 关 ascendc 融合权重预转换/raw 释放（A/B 用）
    --alop-config) ALOP_CFG=${2:-}; shift ;;
    --no-alop-config) ALOP_CFG="" ;;
    --no-precision) PRECISION=0 ;;
    --force) FORCE=1 ;;
    --prof-only) DO_NOPROF=0 ;;
    --noprof-only) DO_PROF=0 ;;
    --console) CONSOLE=${2:-all}; shift ;;
    --cards) CARDS=${2:-0,1,2,3,4,5,6,7}; shift ;;
    --timeout) TIMEOUT=${2:-3600}; shift ;;
    --bar-every) BAR_EVERY=${2:-30}; shift ;;
    --bar-fd) BAR_FD=${2:-9}; shift ;;
    --no-bar) BAR=0 ;;
    -o|--out) OUT_DIR=${2:-}; shift ;;
    -P|--prof-dir) PROF_DIR=${2:-}; shift ;;
    --dry-run) DRY=1 ;;
    --list) LIST=1 ;;
    -h|--help) usage 0 ;;
    *) echo "未知参数：$1（--help 看用法）"; exit 2 ;;
  esac
  shift
done
[ -n "$SEQ" ] || SEQ=$L_DEFAULT
case "$MODE" in prefill|hit|both) ;; *) echo "--mode 只能是 prefill|hit|both"; exit 2 ;; esac

# 仓根 = 本脚本所在目录的上一级（换机器/换目录都不用改脚本）
cd "$(cd "$(dirname "$0")/.." && pwd -P)" || exit 1
source ./env_pack.sh >/dev/null 2>&1 || { echo "⚠️ env_pack.sh 引导失败（先手动 source 看看）"; exit 2; }
# ★ PERF_RULES 门禁（2026-10-18）：性能开关默认值不对就拒绝启动。
_PERF_BASE=$(cd .. && pwd -P)
_PERF_CHECK="$_PERF_BASE/tools/check_perf_defaults.py"
[ -f "$_PERF_CHECK" ] || _PERF_CHECK=/mnt/xds/sfs/l00972766/tools/check_perf_defaults.py
if [ -f "$_PERF_CHECK" ] && ! DS41_BASE="$_PERF_BASE" python3 "$_PERF_CHECK" --runtime; then
  echo "✗ PERF_RULES 检查未通过，拒绝启动（见 PERF_RULES.md；A/B 用 DS41_ALLOW_SUBOPTIMAL=原因 放行）" >&2
  exit 2
fi

# 默认输出根：本机共享盘优先，别的机器落仓内 _out/
if [ -z "$OUT_DIR" ]; then
  for d in /mnt/xds/sfs/l00972766 /mnt/l00972766; do
    [ -d "$d" ] && { OUT_DIR="$d/real_prof"; break; }
  done
  [ -n "$OUT_DIR" ] || OUT_DIR="$PWD/_out/real_prof"
fi
OUT=$OUT_DIR
pdir_for() {  # $1=L  $2=mode
  if [ -z "${PROF_DIR:-}" ]; then printf '%s/L%s/prefill_%s\n' "$OUT" "$1" "$2"; return; fi
  case "$PROF_DIR" in
    *"{mode}"*) printf '%s\n' "${PROF_DIR//\{mode\}/$2}" ;;
    *"{L}"*)    printf '%s\n' "${PROF_DIR//\{L\}/$1}_$2" ;;   # 两个口径不能落同一目录
    *)          printf '%s/L%s/prefill_%s\n' "$PROF_DIR" "$1" "$2" ;;
  esac
}

if [ "$LIST" = 1 ]; then
  echo "本入口只做真权重 EP8（先支持 16k）；清单："
  echo "  ids（必给）: $IDS      官方参考: $REF"
  echo "  权重:        $WEIGHTS"
  echo "  口径:        dp8/ep8/cp1 · engram_tp_size=8 · 自然路由 · resident · host_cache=off"
  echo "  模式:        --mode prefill（全长）| hit（h 默认 0.96875=官方 16k 点）| both"
  echo "  基准:        官方真权重 EP8 16k = 2394.7 ms / 6841.7 tok/s；next-token = id 396 ' that'（37.750）"
  exit 0
fi

# ── 依赖自检（不碰卡）──
[ -f "$IDS" ] || { echo "⚠️ 缺 ids 文件：$IDS（真权重必须给真 ids，否则会退回强均路由）"; exit 2; }
if [ "$PRECISION" = 1 ] && [ ! -f "$REF/dump_rank0_call1.pt" ]; then
  echo "⚠️ 缺官方参考 $REF/dump_rank0_call1.pt（--ref 可换；--no-precision 可跳过精度 gate）"; exit 2
fi
[ -d "$WEIGHTS" ] || { echo "⚠️ 缺真权重目录：$WEIGHTS（--weights-path 可换）"; exit 2; }
if [ "$LAYERS" != 0 ] && [ "$MODE" = hit ]; then
  echo "⚠️ --layers 与命中口径互斥（runner 要求 hit.source=real 时跑全层）→ 改用 --mode prefill"; exit 2
fi
if [ "$LAYERS" != 0 ] && [ "$MODE" = both ]; then
  echo "   （--layers 冒烟 → 只跑 prefill 口径）"; MODE=prefill
fi
[ "$LAYERS" = 0 ] || [ "$LAYERS" -ge 15 ] || echo "   ⚠️ 冒烟建议 --layers 15（engram 层是 1 和 14，少了测不到第二个厂商池）"

case "$SEQ" in
  16k|16384) L=16384 ;;
  *) L=$(python3 - "$SEQ" <<'PY'
import re, sys
m = re.fullmatch(r"(\d+(?:\.\d+)?)([km]?)", sys.argv[1].strip().lower())
sys.exit(f"真权重入口目前只支持 16k（现成 ids + 官方参考）") if not m else None
print(int(float(m.group(1)) * {"": 1, "k": 1024, "m": 1048576}[m.group(2)]))
PY
) || exit 2 ;;
esac
[ "$L" = 16384 ] || [ "$LAYERS" != 0 ] || {
  echo "⚠️ 真权重目前只做 16k（$L 需要自备同长度 ids 与参考）；要冒烟用 --layers N + --mode prefill"; exit 2; }

MODES=""
case "$MODE" in prefill) MODES="prefill" ;; hit) MODES="hit" ;; both) MODES="prefill hit" ;; esac

if [ "$DRY" = 1 ]; then
  echo "（--dry-run）真权重 EP8 采集计划：L=$L  modes=[$MODES]  iters=$ITERS  warmup=$WARMUP  layers=${LAYERS:-全部}"
  for m in $MODES; do
    if [ "$m" = hit ]; then R=$(python3 -c "print(int(round((1-$HIT)*$L)) + $KEEP)"); else R=$L; fi
    printf "  mode=%-8s R=%-8s trace → %s\n" "$m" "$R" "$(pdir_for "$L" "$m")"
  done
  echo "  ids=$IDS"; echo "  参考=$REF（精度 gate 基准 id=396）"; echo "  输出根=$OUT"
  echo "  口径：engram_tp_size=8 · resident=$RESIDENT · host_cache=$HOST_CACHE · moemode=$MOEMODE · 自然路由（给了 ids）"
  [ "$ENGRAM_HOST" = 1 ] && echo "  ⚠️ --engram-host：退回主机取表，trace 里**没有** EngramFetch/Wait"
  exit 0
fi

mkdir -p "$OUT"
[ -d /mnt/l00972766 ] && ln -sfn "$OUT" "/mnt/l00972766/$(basename "$OUT")"

# ★ 真权重跑法的硬前提：pinned 主机内存（厂商 engrma 窗口 ~12.9 GiB/卡 + sf 表）
ulimit -l unlimited 2>/dev/null && echo "[env] ulimit -l → $(ulimit -l)" || echo "   ⚠️ ulimit -l 提不上去（当前 $(ulimit -l)）"

# ── 进度条要"终端原地刷新"：先把真终端存到 fd 9（必须在 tee 之前）──
exec 9>&1
BAR_TTY=0; [ -t 9 ] && BAR_TTY=1

DRIVER="$OUT/driver_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$DRIVER") 2>&1
ln -sfn "$(basename "$DRIVER")" "$OUT/driver_latest.log"
RUN_ID=$(basename "$DRIVER" .log | sed 's/^driver_//')
ROUNDS_TSV="$OUT/rounds.tsv"
RUN_T0=$(date +%s)

# 采集前清掉会改图/改数的调试开关（真权重口径同样不该带这些）
for v in DS41M_MOE_BARRIER DS41M_NO_BALROUTE DS41M_START_NOBARRIER DS41M_NO_CMP DS41M_SKIP_KV \
         DS41M_FAKE_Q DS41M_NO_INVROPE DS41M_MOE_NOSHARED DS41M_MOE_FAKEIN DS41M_PIPELINE \
         DS41M_MOE_ONLY DS41M_SYNC DS41M_SYNC_BEFORE DS41M_NO_VISIBLE DS41M_ENGRAM_WIN_MB \
         DS41M_NO_ENGRAM DS41M_ENGRAM_NOSIDE DS41M_ENGRAM_HOST \
         DS41M_PD_MERGE DS41M_TP_DECODE_ONLY DS41M_DECODE_LAYOUT \
         DS41M_DECODE_PREBUILD DS41M_PD_TP_XFER DS41M_DECODE_BARRIER; do
  [ -n "${!v:-}" ] && { echo "[env] 清掉 $v=${!v}（不在真权重采集口径里）"; unset "$v"; }
done
[ "$ENGRAM_HOST" = 1 ] && export DS41M_ENGRAM_HOST=1
export DS41M_LOG_LEVEL="${LOG_LEVEL:-info}"
export DS41M_MOEMODE=$MOEMODE
# ★ 2026-10-21：deepgemm 的等价性能开关（本脚本是 prefill-only）。
#   transform_weights_for_mega_moe 是 1:1 实体拷贝（40 层 ≈34 GB/卡）；按层缓存 +
#   释放本层 raw 后，峰值不涨、后续 pass 不再重做（否则每 pass 多 ~400 ms）。
if [ "$MOEMODE" = "deepgemm" ]; then
  export DS41M_PREP_CACHE_N=${DS41M_PREP_CACHE_N:-40}
  export DS41M_DG_RELEASE_RAW=${DS41M_DG_RELEASE_RAW:-1}
fi

# ★ ascendc 融合 MoE 权重**预转换 + 释放 raw**（只存一份；仅 prefill）：真权重 16k hit
#   684 → 453（去掉白算的 deepgemm transform）→ **188 ms**（融合缓存；合成同口径 148）。
#   显存净变化：ac ~18 GiB/40 层 < raw ~32 GiB ⇒ **净省 ~14 GiB**。精度 gate 仍 PASS（id=396）。
if [ "$AC_CACHE" = 1 ]; then
  export DS41M_AC_CACHE_N=${DS41M_AC_CACHE_N:-40}
  export DS41M_AC_RELEASE_RAW=${DS41M_AC_RELEASE_RAW:-1}
else
  export DS41M_AC_CACHE_N=0
  unset DS41M_AC_RELEASE_RAW
fi

# ★ MoE 分块（官方 cann-recipes `moe_chunk_max_len` 同口径，默认开）：厂商融合算子的内部
#   workspace 按每 call 的 token 数走（T=16384 单次 ≈20 GiB 瞬时）→ 40 层常驻权重 + profiler
#   下 16k 顶穿 91.8 GiB。分块后 ÷4；MoE 逐 token 独立 ⇒ 数值等价。`--no-moe-chunk` 关掉。
if [ "$MOE_CHUNK" = 1 ]; then export DS41M_MOE_CHUNK=auto; else unset DS41M_MOE_CHUNK; fi

# ★ PERF_RULES §0：SWA bounded replay + 尾段入图**默认开，且不区分合成/真权重**
#   （显式 DS41M_DECODER_REPLAY=0 才关；入图已永久删除，无图相关开关。）
export DS41M_DECODER_REPLAY=${DS41M_DECODER_REPLAY:-1}
# 2026-10-27 用户规则：入图（ACL Graph）已永久删除，无任何图开关
# ★ Phase 2（2026-10-24 用户拍板）：模块运行时默认开；=0/off 回退 legacy。
export DS41M_MODULE_RUNTIME=${DS41M_MODULE_RUNTIME:-1}

# ★ ALOP 采集配置（runner 的 `--prof` 走 ALOP；这张 yaml 决定采集档位）：
#   默认 configs/alop_mindstudio.yaml = 纯 aclprof C 级（python_profiling 永久禁用）→ PROF_* 原始目录，
#   MindStudio trace 需 msprof --export 转出。`--no-alop-config` 用 ALOP 内置默认（更轻，
#   只有 PROF_* 原始目录 → trace 次数断言会自动跳过并提示）。
ALOP_ARG=""
[ -n "$ALOP_CFG" ] && { [ -f "$ALOP_CFG" ] && ALOP_ARG="--alop-config $ALOP_CFG" \
  || echo "⚠️ 找不到 ALOP 配置 $ALOP_CFG（改用 ALOP 内置默认）"; }

# ★ 设备异常时 CANN 会往 **cwd** 写 `extra-info/data-dump`（单次 20+ GB；2026-10-16 实测把
#   仓库 cwd 撑到 351 GB，还会拖慢/撑爆共享盘）→ 统一挪到本次输出目录下。
export ASCEND_WORK_PATH="${ASCEND_WORK_PATH:-$OUT/_ascend_work}"
mkdir -p "$ASCEND_WORK_PATH"

# ★ 分配器别切大块（≤4 GiB 的块不再被拆）：40 层常驻权重 + 厂商 MoE 算子每次 ~20 GiB 的瞬时
#   workspace，总账 68 < 91.8 是够的，但默认分配器把大块切成碎片 → "Tried to allocate 20.12 GiB
#   (… 15.45 GiB free … 72.61 GiB reserved)" 的假 OOM（紧接着 flush-retry 风暴 → 慢/挂死）。
#   CANN 的提示原文就是这句（NPUCachingAllocator.cpp:1386）。
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:4096}"


# ── 工具函数（与 run_hit.sh 同约定）──
fmt_sec() { local s=${1:-0}
  if [ "$s" -ge 3600 ]; then printf '%dh%02dm' $((s/3600)) $(((s%3600)/60))
  elif [ "$s" -ge 60 ]; then printf '%dm%02ds' $((s/60)) $((s%60)); else printf '%ds' "$s"; fi; }
# rounds.tsv：1=run_id 2=idx 3=L 4=R 5=round 6=rc 7=sec 8=trace_n
record_round() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$RUN_ID" "$1" "$2" "$3" "$4" "$5" "${6:-}" "${7:-}" >> "$ROUNDS_TSV"; }
round_sec_of() { awk -F'\t' -v r="$1" -v L="$2" '$5==r && $3==L && $7+0>0 {s=$7} END{if(s!="")print s}' "$ROUNDS_TSV" 2>/dev/null; }
# ETA 先验按**真权重**给（合成那套是 90/60；这里是实测口径：prof ~240s / noprof ~150s）
ROUNDS_WANT=0
_avg_of() { local a n
  read -r a n < <(awk -F'\t' -v r="$1" 'index($5,r)==1 && $7+0>10 {v[k++]=$7}
      END{ if(k==0){print "0 0"; exit}
           for(i=1;i<k;i++){x=v[i]; j=i-1; while(j>=0 && v[j]>x){v[j+1]=v[j]; j--} v[j+1]=x}
           printf "%d %d", v[int(k/2)], k }' "$ROUNDS_TSV" 2>/dev/null || echo "0 0")
  if [ "${a:-0}" -lt 10 ]; then
    case "$1" in noprof*) echo "150 0" ;; *) echo "240 0" ;; esac   # 先验按真权重实测口径
    return; fi
  echo "$a $n"; }
eta_rest() { local pa na np_ nn rest=0
  read -r pa np_ < <(_avg_of prof); read -r na nn < <(_avg_of noprof)
  for m in $MODES; do
    [ "$DO_PROF" = 1 ] && [ -z "$(round_sec_of "prof-$m" "$L")" ] && rest=$((rest + pa))
    [ "$DO_NOPROF" = 1 ] && [ -z "$(round_sec_of "noprof-$m" "$L")" ] && rest=$((rest + na))
  done
  echo "$rest $pa $na $((np_ + nn))"; }
print_points_done() { local rest_line; rest_line=$(eta_rest); set -- $rest_line
  echo
  echo "──── 总进度 · 已用 $(fmt_sec $(( $(date +%s) - RUN_T0 ))) · 剩余粗估 ~$(fmt_sec "$1")"
  printf '  %-9s %-10s %-9s %-9s %s\n' 轮次 稳态 用时 trace份 状态
  for m in $MODES; do
    local ps ns tr
    ps=$(round_sec_of "prof-$m" "$L"); ns=$(round_sec_of "noprof-$m" "$L")
    tr=$(ls "$(pdir_for "$L" "$m")" 2>/dev/null | wc -l)
    printf '  %-9s %-10s %-9s %-9s %s\n' "prof-$m" "${ps:+$(fmt_sec "$ps")}" "" "$tr" "$([ -d "$(pdir_for "$L" "$m")" ] && echo ✓)"
    [ "$DO_NOPROF" = 1 ] && printf '  %-9s %-10s %-9s %-9s %s\n' "noprof-$m" "${ns:+$(fmt_sec "$ns")}" "" "" "$([ -n "$ns" ] && echo ✓)"
  done; }

# 真权重版口径自检（与合成**相反**：elb 必须缺席）
chk_lines() {  # $1=点位目录 $2=mode
  local D=$1 m=$2 L0="$D/logs_$m/rank0.log" fail=0
  if grep -qm1 'elb/balroute' "$L0"; then
    echo "   [FAIL] 出现 [elb/balroute]（强均路由被启用 → 不是自然路由；检查 --ids-file）"; fail=1
  else echo "   [OK] 自然路由（无 elb/balroute 行）"; fi
  local nready; nready=$(grep -c '厂商 ElasticBuffer 就绪' "$L0" 2>/dev/null || echo 0)
  grep -hm1 'engram] 厂商 ElasticBuffer 就绪' "$L0" 2>/dev/null | sed 's/^/   /'
  # 应**恰好** NENGRAM_RUN 次（每层一次；之后每个 pass 都该命中缓存）。多出来 = 缓存被顶掉，
  # 每 pass 重建 11.44 GiB 主机窗口（曾经 27 s/pass 的元凶，见 exec_engram._vendor_group）。
  echo "   [$( [ "${nready:-0}" = "$NENGRAM_RUN" ] && echo OK || echo FAIL )] 厂商路径就绪 $nready 次（应恰好 $NENGRAM_RUN = engram 层数）"
  [ "${nready:-0}" = "$NENGRAM_RUN" ] || fail=1
  local ngrp; ngrp=$(grep -c '专用进程组就绪' "$L0" 2>/dev/null || echo 0)
  echo "   [$( [ "${ngrp:-0}" = "$NENGRAM_RUN" ] && echo OK || echo FAIL )] 每层专用进程组 $ngrp 个（应 $NENGRAM_RUN：厂商窗口按通信域池化，同组=同一块窗口）"
  [ "${ngrp:-0}" = "$NENGRAM_RUN" ] || fail=1
  if grep -qm1 '厂商 ElasticBuffer 不可用' "$L0"; then
    echo "   [FAIL] 厂商路径不可用 → 退回主机取表（trace 会缺 EngramFetch/Wait）"; fail=1; fi
  if grep -qm1 '侧流发起失败' "$L0"; then
    echo "   [FAIL] 侧流发起失败 → 主干会重跑整条链 → trace 里 EngramFetch 双份"; fail=1; fi
  grep -hm1 'engram] 侧流已启用' "$L0" 2>/dev/null | sed 's/^/   /'
  grep -hm1 '\[起跑线\]' "$L0" 2>/dev/null | sed 's/^/   /'
  grep -hm1 '口径：' "$L0" 2>/dev/null | sed 's/^/   /'
  grep -hm1 'mega_moe\] 后端' "$L0" 2>/dev/null | sed 's/^/   /'
  [ "$fail" = 0 ] && echo "   [OK] 口径自检通过" || echo "   [FAIL] 口径自检有红项（见上）"
  return $fail; }

# trace 断言：把 8 rank 的 op_statistic.csv 求和
trace_ops() { awk -F, 'FNR>1{print $2","$4}' "$1"/*/ASCEND_PROFILER_OUTPUT/op_statistic.csv 2>/dev/null \
              | awk -F, '{c[$1]+=$2} END{for(k in c) printf "%s\t%d\n", k, c[k]}' | sort; }
trace_lint() {  # $1=trace 根 $2=iters  $3=实际跑的层数  $4=其中 engram 层数  $5=每层 MoE 块数
  local it=$2 nl=$3 ne=$4 nr ch=${5:-1} want want_moe got got_moe
  nr=$(ls -d "$1"/*/ 2>/dev/null | wc -l)
  if [ "$(ls "$1"/*/ASCEND_PROFILER_OUTPUT/op_statistic.csv 2>/dev/null | wc -l)" = 0 ]; then
    echo "   [trace] 没有 ASCEND_PROFILER_OUTPUT/op_statistic.csv（ALOP 纯 C 级模式？）→ 跳过次数断言"
    echo "         要出 MindStudio trace 就别用 --no-alop-config（默认 configs/alop_mindstudio.yaml）"
    return 0
  fi
  # 期望 = 每 rank 每次数 × rank 数：EngramFetch 每 engram 层 1 次（侧流那次；主干侧的同名
  # 算子会被早退吃掉 → 不应出现第二份）；MegaMoe 每层**每块** 1 次（官方同款 MoE 分块：
  # ceil(每轮 token / moe_chunk_max_len) 块，DS41M_MOE_CHUNK=auto 时生效）。
  want=$((ne * it * nr)); want_moe=$((nl * it * nr * ch))
  got=$(trace_ops "$1" | awk -F'\t' '$1=="EngramFetch"{print $2}')
  got_moe=$(trace_ops "$1" | awk -F'\t' '$1=="MegaMoe"{print $2}')
  printf '   [trace] EngramFetch=%s（期望 %s = %s层×%s轮×%s卡）  MegaMoe=%s（期望 %s = %s层×%s轮×%s卡×%s块）\n' \
         "${got:-0}" "$want" "$ne" "$it" "$nr" "${got_moe:-0}" "$want_moe" "$nl" "$it" "$nr" "$ch"
  [ "${got:-0}" = "$want" ] && [ "${got_moe:-0}" = "$want_moe" ] \
    && echo "   [OK] trace 算子次数对上" || echo "   [FAIL] trace 算子次数不对（engram 没走厂商路径？mega_moe 回退了？分块口径变了？）"; }

# ── 点位目录 / 指纹 ──
PIDX=1; _nmodes=$(echo $MODES | wc -w); POINT_TOTAL=$(( _nmodes * (DO_PROF + DO_NOPROF) ))
D=$OUT/L$L; mkdir -p "$D"
WANT="L=$L modes=$MODE h=$HIT iters=$ITERS warmup=$WARMUP layers=$LAYERS weights=real engram_tp=8 \
resident=$RESIDENT host_cache=$HOST_CACHE moemode=$MOEMODE engram_host=$ENGRAM_HOST moe_chunk=$MOE_CHUNK \
alop=${ALOP_CFG:-builtin} ids=$(sha256sum "$IDS" | cut -c1-8)"

echo "===== 真权重 EP8 profiling 采集（mini：dp8/ep8/cp1 + 自然路由（不负载均衡） + 真表 engram + 精度 gate）====="
echo "开始 $(date -u +'%Y-%m-%d %H:%M:%SZ')    驱动日志 $DRIVER"
echo "点位：L=$L  模式=[$MODES]  iters=$ITERS  warmup=$WARMUP  layers=${LAYERS:-全部}  卡=$CARDS"
echo "口径：真权重 $WEIGHTS"
echo "      engram_offload=true + engram_tp_size=8（每卡 ~12.9 GiB 分片）· resident=$RESIDENT · host_cache=$HOST_CACHE · moemode=$MOEMODE"
echo "      ids=$IDS（真 ids → 自然路由，不做负载均衡）· host=$(hostname) · $(nproc) 核"
df -h "$OUT" | tail -1 | sed 's/^/磁盘 /'

FAILS=""
# 实际跑的层数 / 其中 engram 层数（`--layers K` = 只跑前 K 层；层号从 config 读，不硬编码）
read -r NLAYERS_RUN NENGRAM_RUN <<<"$(python3 - "$LAYERS" <<'PY'
import sys, yaml
L = int(sys.argv[1]); m = yaml.safe_load(open('configs/mini_real.yaml'))['model']
nl = int(m['n_layers']); n = nl if L <= 0 else min(L, nl)
print(n, sum(1 for i in (m.get('engram_layers') or []) if int(i) < n))
PY
)"
echo "层数口径：本轮跑 $NLAYERS_RUN 层（其中 engram 层 $NENGRAM_RUN 个）→ trace 期望按此算"
# MoE 分块长度（官方 moe_chunk_max_len；trace 期望里每层的 MegaMoe 次数 = ceil(每轮 token/它)）
MOE_CHUNK_LEN=$(python3 -c "import yaml;print(int((yaml.safe_load(open('configs/mini_real.yaml')).get('run') or {}).get('moe_chunk_max_len',4096)))")
[ "$MOE_CHUNK" = 1 ] && echo "MoE 分块：$MOE_CHUNK_LEN token/块（官方 get_moe_num_chunks 同口径；每层块数 = ceil(每轮 token/$MOE_CHUNK_LEN)）" \
                   || echo "MoE 分块：关（--no-moe-chunk；每层 1 次 MegaMoe）"
[ "$AC_CACHE" = 1 ] && echo "MoE 权重：ascendc 融合预转换缓存 ${DS41M_AC_CACHE_N:-40} 层 + 释放 raw（只存一份；仅 prefill）" \
                    || echo "MoE 权重：不预转换（--no-ac-cache；每 pass 重新融合）"
for m in $MODES; do
  if [ "$m" = hit ]; then
    HR=$HIT; SRC=real; R=$(python3 -c "print(int(round((1-$HIT)*$L)) + $KEEP)")
  else
    HR=0; SRC=fake; R=$L
  fi
  CFG="$D/config_$m.yaml"
  python3 - "$CFG" "$L" "$HR" "$SRC" "$ITERS" "$RESIDENT" "$HOST_CACHE" "$WEIGHTS" <<'PY' || { echo "配置生成失败"; exit 2; }
import sys, yaml
p, L, h, src, iters, resident, host_cache, wpath = sys.argv[1:9]
cfg = yaml.safe_load(open('configs/mini_real.yaml'))
cfg['run']['seq_len'] = int(L); cfg['run']['iters'] = int(iters)
cfg['hit']['ratio'] = float(h); cfg['hit']['source'] = src
# DeepInfer 口径：non_cached 直接写进配置（不做 128 对齐）
# ★ 2026-11-01（P0 几何收口后的口径对齐 —— P5 首跑 rc=1 的真因）：
#   **只有命中开启（h>0）才写 non_cached_tokens**。原来无条件写 `round((1-h)*L)`，
#   prefill（h=0）时 = L ⇒ `hit_is_on()` 为真、M=L、R=L+KEEP>L ⇒
#   `assemble.check_hit_geometry` 按铁律 0 **显式拒绝**（没有可测的 prefill 语义）。
#   正确写法：prefill 用 ratio=0 + nc=0（命中关闭），几何即全长 L。
cfg['hit']['non_cached_tokens'] = (
    int(round((1 - float(h)) * int(L))) if float(h) > 0 else 0)
cfg['weights']['mode'] = 'real'; cfg['weights']['path'] = wpath
cfg['weights']['resident'] = (resident == 'true'); cfg['weights']['host_cache'] = (host_cache == 'true')
cfg['model']['engram_offload'] = True; cfg['model']['engram_tp_size'] = 8       # ★ P0：不设=1 会吃 1.6 TB
yaml.safe_dump(cfg, open(p, 'w'), allow_unicode=True, sort_keys=False)
# 起跑前硬断言（P0）——写错层级/写漏 = 直接 1.4 TB pinned
assert cfg['model']['engram_tp_size'] == 8, 'engram_tp_size 必须是 8'
assert cfg['weights']['mode'] == 'real' and 'mxfp8' in cfg['weights']['path']
print(f"   配置就绪 {p}（engram_tp_size=8 ✓ host_cache={host_cache} resident={resident}）")
PY

  if [ "$DO_PROF" = 1 ]; then
    PDIR=$(pdir_for "$L" "$m"); mkdir -p "$(dirname "$PDIR")"
    NPRE=$(ls "$PDIR" 2>/dev/null | wc -l)
    OLD=$(cat "$D/PARAMS_$m" 2>/dev/null || echo "无记录")
    RUN=1
    if [ -d "$PDIR" ]; then
      if [ "$NPRE" -ge 8 ] && [ "$FORCE" != 1 ] && [ "$OLD" = "$WANT" ]; then
        echo "   [skip] 已有 $PDIR（$NPRE 份 trace，口径一致）"; RUN=0
      else
        TS=$(date -u +%m%d_%H%M%S); mv "$PDIR" "$PDIR.bak_$TS" 2>/dev/null && echo "   （旧 trace → .bak_$TS）"
        rm -f "$D/PARAMS_$m" "$D/DONE_$m"; [ -d "$D/logs_$m" ] && mv "$D/logs_$m" "$D/logs_$m.bak_$TS"
      fi
    fi
    if [ "$RUN" = 1 ]; then
      BAR_ARGS=(--bar "真权重 L=$L 口径$m prof轮（点 $PIDX/$POINT_TOTAL）"
                --bar-note "总已用$(fmt_sec $(( $(date +%s) - RUN_T0 ))) 剩余粗估~$(fmt_sec "$(eta_rest | cut -d' ' -f1)")"
                --bar-every "$BAR_EVERY")
      [ "$BAR_TTY" = 1 ] && BAR_ARGS+=(--bar-fd "$BAR_FD")
      [ "$BAR" = 1 ] || BAR_ARGS=(--bar "" --bar-every "$BAR_EVERY")
      T0=$(date +%s)
      echo
      echo "──── [$(date -u +%H:%M:%SZ)] 真权重 L=$L 口径=$m h=$HR R=$R prof 轮开始（trace → $PDIR）"
      python3.11 core/launch_mini.py --config "$CFG" --world 8 --cards "$CARDS" \
          --logdir "$D/logs_$m" --timeout "$TIMEOUT" --console "$CONSOLE" \
          --seq "$L" --iters "$PROF_ITERS" --prof --out "$PDIR" ${ALOP_ARG:+$ALOP_ARG} \
          --ids-file "$IDS" --warmup-iters "$WARMUP" ${LAYERS:+--layers "$LAYERS"} \
          ${BAR_ARGS[@]+"${BAR_ARGS[@]}"} 2>&1 | tee "$D/launch_${m}_prof.log"
      RC=${PIPESTATUS[0]}
      SEC=$(( $(date +%s) - T0 ))
      record_round "$PIDX" "$L" "$R" "prof-$m" "$RC" "$SEC" "$(ls "$PDIR" 2>/dev/null | wc -l)"
      echo "   [prof-$m] rc=$RC  用时 ${SEC}s（$(fmt_sec "$SEC")）  trace 份数=$(ls "$PDIR" 2>/dev/null | wc -l)/8  → $PDIR"
      chk_lines "$D" "$m" || FAILS="$FAILS ${L}/${m}(口径自检)"
      _ch=1
      if [ "$MOE_CHUNK" = 1 ] && [ "$MOE_CHUNK_LEN" -gt 0 ]; then
        _ch=$(( (R + MOE_CHUNK_LEN - 1) / MOE_CHUNK_LEN ))
      fi
      [ "$RC" = 0 ] && trace_lint "$PDIR" "$PROF_ITERS" "$NLAYERS_RUN" "$NENGRAM_RUN" "$_ch"
      # 精度 gate
      if [ "$PRECISION" = 1 ]; then
        python3.11 tools/check_next_token.py --log "$D/logs_$m/rank0.log" --ref-dump "$REF/dump_rank0_call1.pt" \
            --json "$D/next_token_$m.json" 2>&1 | sed 's/^/   /'
        [ "${PIPESTATUS[0]}" = 0 ] || FAILS="$FAILS ${L}/${m}(精度 gate)"
      fi
      [ "$RC" != 0 ] && FAILS="$FAILS ${L}/${m}(prof rc=$RC)"
      PIDX=$((PIDX + 1))
    fi
  fi

  if [ "$DO_NOPROF" = 1 ]; then
    BAR_ARGS=(--bar "真权重 L=$L 口径$m 无prof轮（点 $PIDX/$POINT_TOTAL）"
              --bar-note "总已用$(fmt_sec $(( $(date +%s) - RUN_T0 ))) 剩余粗估~$(fmt_sec "$(eta_rest | cut -d' ' -f1)")"
              --bar-every "$BAR_EVERY")
    [ "$BAR_TTY" = 1 ] && BAR_ARGS+=(--bar-fd "$BAR_FD")
    [ "$BAR" = 1 ] || BAR_ARGS=(--bar "" --bar-every "$BAR_EVERY")
    T0=$(date +%s)
    echo
    echo "──── [$(date -u +%H:%M:%SZ)] 真权重 L=$L 口径=$m h=$HR R=$R 无 prof 轮（干净稳态）"
    python3.11 core/launch_mini.py --config "$CFG" --world 8 --cards "$CARDS" \
        --logdir "$D/logs_${m}2" --timeout "$TIMEOUT" --console "$CONSOLE" \
        --seq "$L" --iters "$ITERS" --ids-file "$IDS" --warmup-iters "$WARMUP" \
        ${LAYERS:+--layers "$LAYERS"} ${BAR_ARGS[@]+"${BAR_ARGS[@]}"} 2>&1 | tee "$D/launch_${m}_noprof.log"
    RC=${PIPESTATUS[0]}
    SEC=$(( $(date +%s) - T0 ))
    record_round "$PIDX" "$L" "$R" "noprof-$m" "$RC" "$SEC" ""
    echo "   [noprof-$m] rc=$RC  用时 ${SEC}s（$(fmt_sec "$SEC")）"
    grep -h '多轮耗时' "$D/logs_${m}2/rank0.log" 2>/dev/null | tail -1 | sed 's/^/   /'
    [ "$RC" != 0 ] && FAILS="$FAILS ${L}/${m}(noprof rc=$RC)"
    PIDX=$((PIDX + 1))
  fi

  # ── 可选：逐层对拍（--parity，只收 rank0，对照官方 parity/ref/collect_rank0.pt）──
  if [ "$PARITY" = 1 ]; then
    echo
    echo "──── [$(date -u +%H:%M:%SZ)] 逐层对拍轮（--collect-layers，rank=$CCOLLECT_RANKS，普通 prefill）"
    CFG0="$D/config_parity.yaml"
    python3 - "$CFG0" "$L" "$RESIDENT" "$HOST_CACHE" "$WEIGHTS" <<'PYCFG'
import sys, yaml
p, L, resident, host_cache, wpath = sys.argv[1:6]
cfg = yaml.safe_load(open('configs/mini_real.yaml'))
cfg['run']['seq_len'] = int(L); cfg['run']['iters'] = 1
cfg['hit']['ratio'] = 0.0; cfg['hit']['source'] = 'fake'
cfg['weights']['mode'] = 'real'; cfg['weights']['path'] = wpath
cfg['weights']['resident'] = (resident == 'true'); cfg['weights']['host_cache'] = (host_cache == 'true')
cfg['model']['engram_offload'] = True; cfg['model']['engram_tp_size'] = 8
yaml.safe_dump(cfg, open(p, 'w'), allow_unicode=True, sort_keys=False)
PYCFG
    python3.11 core/launch_mini.py --config "$CFG0" --world 8 --cards "$CARDS" \
        --logdir "$D/logs_parity" --timeout "$TIMEOUT" --console "$CONSOLE" \
        --seq "$L" --iters 1 --ids-file "$IDS" --collect-layers "$D/collect" \
        --collect-ranks "$CCOLLECT_RANKS" 2>&1 | tee "$D/launch_parity.log" | tail -20
    RC=${PIPESTATUS[0]}
    record_round "$PIDX" "$L" "$L" parity "$RC" "0" ""
    echo "   [parity] rc=$RC  → $D/collect"
    [ "$RC" != 0 ] && FAILS="$FAILS ${L}(parity rc=$RC)"
    PIDX=$((PIDX + 1))
  fi
done

if [ -z "$FAILS" ]; then touch "$D/DONE"; for m in $MODES; do [ "$DO_PROF" = 1 ] && printf '%s\n' "$WANT" > "$D/PARAMS_$m"; done; fi
print_points_done

# ── 汇总（同时落盘 SUMMARY.md）──
{
  echo
  echo "===== 汇总 $(date -u +'%Y-%m-%d %H:%M:%SZ')  真权重 EP8 ====="
  printf "%-8s %-9s %-10s %-8s %-9s %-9s %-22s %s\n" 口径 h R trace份 prof用时 noprof用时 "稳态(均,除首轮)" "精度gate"
  for m in $MODES; do
    if [ "$m" = hit ]; then HR=$HIT; else HR=0; fi
    TR=$(ls "$(pdir_for "$L" "$m")" 2>/dev/null | wc -l)
    PS=$(round_sec_of "prof-$m" "$L"); NS=$(round_sec_of "noprof-$m" "$L")
    ST=$(grep -h '多轮耗时' "$D/logs_${m}2/rank0.log" 2>/dev/null | tail -1 | sed -n 's/.*均(除首轮)=\([0-9.]*\)s.*/\1/p')
    ST=$(awk -v s="${ST:-}" 'BEGIN{ if (s=="") print "—"; else printf "%.0f ms", s*1000 }')
    NT=$(python3 -c "
import json,sys
try:
    d=json.load(open('$D/next_token_$m.json')); print(('✅' if d['ok'] else '❌')+f\" id={d['mini_top1']}/{d['ref_top1']} 重合{d['top8_overlap']}/8\")
except Exception: print('—')")
    printf "%-8s %-9s %-10s %-8s %-9s %-9s %-22s %s\n" "$m" "$HR" "$R" "$TR" \
        "${PS:+$(fmt_sec "$PS")}" "${NS:+$(fmt_sec "$NS")}" "${ST:-—}" "$NT"
  done
  echo
  echo "官方基准（同权重/同 ids/同 EP8）：prefill 2394.7 ms · 6841.7 tok/s；next-token id=396 ' that'（37.750）"
  echo "总用时：$(fmt_sec $(( $(date +%s) - RUN_T0 )))   轮次明细：$ROUNDS_TSV"
  [ -n "$FAILS" ] && { echo; echo "⚠️ 失败/@自检红项：$FAILS"; } || { echo; echo "✅ 全部完成（口径自检 + trace 断言 + 精度 gate 全过）"; }
  echo "驱动日志：$DRIVER"
} | tee "$OUT/SUMMARY.md"

[ -z "$FAILS" ] || exit 1

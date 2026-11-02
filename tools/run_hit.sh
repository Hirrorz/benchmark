#!/bin/bash
# mini 前缀命中（prefix-cache hit）采集入口：**一次启动跑多个序列**，L 与命中率都传参。
# 口径：dp8 / ep8 / cp1 + elb 强均路由 + 合成假权重 + fake 前缀 + 起跑线 barrier×1 + engram 侧流。
# 默认点位 = 官方 hit97 矩阵那 8 个（16k…1m，其 h 各异）；显式给了 --hit / @h 就按给的算。
#
# 用法：
#   bash tools/run_hit.sh -L 16k 32k 64k 128k 256k     # 一次跑这 5 个点（顺序照给）
#   bash tools/run_hit.sh -L=16k,32k,64k               # 逗号写法也行
#   bash tools/run_hit.sh                              # 不写 -L = 官方 8 点全跑（16k…1m）
#   bash tools/run_hit.sh -L 64k --force               # 重采（旧 prefill/logs 挪成 *.bak_*）
#   （★ 2026-10-28 用户规则：只跑 prof 轮。只采一张卡时，其余 rank 本身就是"无 prof"，
#     它们的耗时（launch.log 的 [r1] 多轮耗时）就是真实稳态值 —— 两轮口径已取消，
#     --prof-only 已是默认，保留兼容；--noprof-only 仍可只跑无 prof 轮）
#
# 命中率（默认取官方 hit97 矩阵每点的值；**显式给了就按给的算**）：
#   --hit 0.97                # 全局命中率（也认 97% / 97）→ M = round((1-h)·L)，R = M + replay_tail
#   -L 64k@0.99 128k@0.97     # 逐点给（优先级最高）
#   -L 100k:3072              # 直接给 M（= 官方口径里"新增 token 数"，h 反推为 1-M/L）
#   --replay-tail 140         # R = M + replay_tail（默认 KEEP=140，DeepInfer 口径）
#   --moemode ascendc         # 显式 MoE 后端 deepgemm(默认)|ascendc(海思厂商 mega_moe)；
#                             #   ★ 必须用这个参数：脚本开头会 unset DS41M_MOEMODE（防环境残留），
#                             #     直接 `DS41M_MOEMODE=… bash run_hit.sh` 会被清掉、静默回落 ascendc
#   （2026-10-26 用户规则：入图已删除，无相关开关）
#                             #   前半 eager，prof 轮自动走 eager 以保留逐算子 trace）
#   --cards 0,1,...           # 物理卡号；**个数 = world（1..8）**。1 个卡号 = 单卡，
#                             #   多点/逐 L 都支持；单卡通常配 DS41M_SKIP_MOE=1 DS41M_SKIP_ENGRAM=1
#                             #   （不跑 MoE/engram，也不加载对应权重；数值无数值意义）
#   --alop-config <yaml>      # 换 ALOP 采集配置（默认 configs/alop_mindstudio.yaml；
#                             #   可用 aicore_metrics:false 的配置验证图+prof 稳定性）
#   -o/--out <dir>            # 换**输出根目录**（日志/汇总/默认 trace 都在这下面）
#   -P/--prof-dir <dir>       # 单独指定 **trace 输出目录**（大文件换盘、或单开一个实验目录）
#                             #   默认 <out>/L<L>/prefill；给 -P 后 = <dir>/L<L>/prefill
#                             #   路径里含 {L} 就按点替换、不再追加 L<L>/prefill：
#                             #     -P '/mnt/xds/sfs/prof_{L}' → /mnt/xds/sfs/prof_65536
#   bash tools/run_hit.sh --list   /   ... --dry-run    # 看可用点位 / 只解析不跑
#
# ★ 多点位预热跑（一次进程，不重复重启）：
#   bash tools/run_hit.sh --points /tmp/points.json \
#        --prof-ranks 0 --prof-warmup 2 --iters 4 -o <out根>
#   points.json = [{"L":81920,"hit":0.96,"batches":1,"replay":true}, …]
#   ★ 一键粘贴（免建文件，内联点位表会落到 <out>/points_inline.json 留档）：
#     bash tools/run_hit.sh --points-json '[{"L":81920,"hit":0.96,"batches":1,"replay":true}]' …
#     bash tools/run_hit.sh --points-spec '80k@0.96/b1/r1 80k@0.96/b2/r1 80k@0.96/b3/r0' …
#       spec 语法：L@hit[/bN][/r0|1][/tTAIL]，空白/逗号分隔；L 支持 80k/1m；hit 支持 96/96%/0.96
#   每个点位各自「预热 K 轮（不进 trace）→ 采集 iters 轮」，trace 落
#   <out>/L<L>/h<hhh>_b<N>_r<0|1>/prefill（N=batches，r=replay）；权重/进程组只加载一次。
#
# 每个点跑两轮：`--prof`（MindStudio trace → trace 目录；★ 默认只采 rank0，1 份/点）
#              无 prof（干净单 pass 稳态耗时，用来对官方 ms）
# ★ 2026-10-27 用户规则：prof 默认只采 rank0（--prof-ranks 0）；其余卡不采，它们的
#   forward 时间就是无 prof 的真实值 → 同一进程既有 trace 又有真实 forward 时间。
#   稳态读数（完成行/汇总）自动取"未被采集的第一个 rank"（默认 rank1），不取 rank0。
#
# 日志（口径：全部打印到控制台，并同时落盘）：
#   控制台 ← 本脚本 + 8 个 rank 的实时输出（launch_mini.py --console，带 [rN] 前缀）
#            ★ --pure-log 时：rank 侧静默、不写任何日志文件（连 /dev/null 都不写），
#              控制台只回显汇总行；要留完整日志用 DS41M_KEEP_LOG_FILES=1
#   落盘   ← $OUT/driver_<时间戳>.log            本脚本全过程（+ driver_latest.log 软链）
#            $OUT/L<L>/launch.log                该点 prof 轮全过程
#            $OUT/L<L>/launch_noprof.log         该点无 prof 稳态轮全过程
#            $OUT/L<L>/logs{,2}/rankN.log        每 rank 一份；all.log = 8 rank 合并（带前缀）
#            $OUT/L<L>/logs{,2}/ascend_log/      CANN plog
#            $OUT/SUMMARY.md                     末尾汇总表
#
# ★ 旧口径（M 128 对齐 / KEEP=128 / 旧后端）已删除：历史数字见 NOTES/2026-10-18-*，
#   不得与当前 DeepInfer 口径（M=round((1-h)L)、KEEP=140）混比。
set -u

usage() { awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"; exit "${1:-0}"; }

SEQS=(); LIST=0; DRY=0; FORCE=0; DO_PROF=1; DO_NOPROF=0; CONSOLE=all
# ★ 2026-10-28 用户规则（PERF_RULES §0 采集规则 5）：**只跑 prof 轮，两轮口径已取消**。
#   只采一张卡时，其余 rank 本来就不带 prof → 同一次跑里同时拿到 trace 和真实时延：
#   稳态数取 prof 轮里未被采集的 rank（REPORT_RANK，默认 rank1）的「多轮耗时」，不再单跑一轮。
ITERS=4; ITERS_SET=0; TIMEOUT=3600; CARDS=0,1,2,3,4,5,6,7; HIT=""
KEEP=${DS41M_REPLAY_KEEP:-140}       # DeepInfer V4.1 KEEP_TOKENS 默认 140
RTAIL=$KEEP; PROF_DIR=${PROF_DIR:-}
MOEMODE=""                           # --moemode ascendc|deepgemm：显式后端（清理循环之后再写回）
REPLAY=1; REPLAY_W=0; REPLAY_CUT=0   # ★ 2026-10-18 用户口径：性能开关默认开（--no-replay 可关）
                                     # 入图（ACL Graph）已永久删除（2026-10-27 用户规则）：无 GRAPH 变量/开关
PROF_RANKS=0; PROF_WARMUP=""         # ★ 2026-10-27 用户规则：prof 默认**只采 rank0**（其余卡不采，
                                     #   它们的 forward 时间才是无 prof 的真实值）；"all" 仍可显式给
BATCHES=1                            # ★ 每 DP 的 batch 数（--batches N；规则：>1 自动 packed）
BATCH_MODE=auto                      # auto|packed|sequential；sequential 仅历史 A/B
PLAIN=0                              # --plain：整段 L 全算（hit.ratio=0），不做前缀命中
POINTS_FILE=""                       # --points <json>：一次进程跑多个点位（不重复重启）
POINTS_JSON=""                       # --points-json '<json>'：内联 JSON（免建文件）
POINTS_SPEC=""                       # --points-spec 'L@hit[/bN][/r0|1][/tTAIL] …'：内联简写
BAR=1; BAR_FD=9; BAR_EVERY=30; ETA_FROM=""
LOG_LEVEL=${DS41M_LOG_LEVEL:-info}   # pure|error|warn|info|debug；pure=什么都不打印，只留进度条
LOG_LEVEL_SET=0                      # ★ 2026-10-28：显式给过 --log-level/--pure-log 就置 1
ALOP_CFG=${DS41M_ALOP_CONFIG:-configs/alop_mindstudio.yaml}   # --alop-config：换采集配置（如关 aicore_metrics）
while [ $# -gt 0 ]; do
  case "$1" in
    -L|--seqs)                                   # 后面跟一串（到下一个 - 开头为止）
      shift
      while [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; do
        IFS=',' read -r -a _parts <<< "$1"; SEQS+=("${_parts[@]}"); shift
      done
      continue ;;
    -L=*|--seqs=*)                               # -L=16k,32k,64k 写法
      IFS=',' read -r -a _parts <<< "${1#*=}"; SEQS+=("${_parts[@]}") ;;
    --list) LIST=1 ;;
    --batches) BATCHES=${2:-1}; shift ;;         # ★ 每 DP 的 batch 数（规则：>1 自动 packed）
    --batch-mode) BATCH_MODE=${2:-auto}; shift ;;  # auto（默认）/packed/sequential
    --plain) PLAIN=1 ;;                          # 整段全算（hit.ratio=0），不做命中
    --points) POINTS_FILE=${2:-}; shift ;;       # ★ 多点位预热跑（JSON list；一个进程跑完全部）
    --points-json) POINTS_JSON=${2:-}; shift ;;  # ★ 内联 JSON：免建文件，一条命令直接跑
    --points-spec) POINTS_SPEC=${2:-}; shift ;;  # ★ 内联简写：'80k@0.96/b1/r1 80k@0.96/b2/r0 …'
    --dry-run) DRY=1 ;;                          # 只解析点位并打印，不碰卡
    --hit) HIT=${2:-}; shift ;;                  # 全局命中率（0.97 / 97% / 97）
    --replay-tail) RTAIL=${2:-$KEEP}; shift ;;      # R = M + replay_tail（默认 KEEP=140）
    --moemode) MOEMODE=${2:-}; shift ;;             # ascendc|deepgemm：显式后端（覆盖清理循环）
    --replay) REPLAY=1 ;;                        # ★ decoder-side SWA bounded replay（官方同款，已默认开）
    --no-replay) REPLAY=0 ;;                     # 关掉 replay（A/B 对照用）
    --replay-w) REPLAY_W=${2:-0}; shift ;;       # 0 = sliding_window；>=序列长 = 不裁（等价回归用）
    --replay-cut) REPLAY_CUT=${2:-0}; shift ;;   # 0 = 最后一个 kv_source+1
    --prof-ranks) PROF_RANKS=${2:-0}; shift ;;  # 只落指定 rank 的 trace（默认 0；all=全采）
    --prof-warmup) PROF_WARMUP=${2:-0}; shift ;; # prof 轮先跑 K 个不计入 trace 的 forward
    --force) FORCE=1 ;;
    --prof-only) DO_NOPROF=0 ;;                  # 已是默认（PERF_RULES §0 采集规则 5），保留兼容
    --with-noprof)                               # ★ 2026-10-28 用户规则：两轮口径已取消
      echo "✗ --with-noprof 已取消：只采一张卡时其余 rank 本来就是无 prof，性能数取" >&2
      echo "  launch.log 里 [r1] 多轮耗时即可，不需要单独再跑一轮。" >&2
      exit 2 ;;
    --noprof-only) DO_PROF=0; DO_NOPROF=1 ;;     # 只跑无prof轮（DO_NOPROF 默认已是 0，这里必须置 1）
    --console) CONSOLE=${2:-all}; shift ;;
    --iters) ITERS=${2:-4}; ITERS_SET=1; shift ;;
    --timeout) TIMEOUT=${2:-3600}; shift ;;
    --alop-config) ALOP_CFG=${2:-configs/alop_mindstudio.yaml}; shift ;;
    --cards) CARDS=${2:-0,1,2,3,4,5,6,7}; shift ;;
    -o|--out) OUT_DIR=${2:-}; shift ;;           # 输出根目录（默认 hit97_8points）
    -P|--prof-dir) PROF_DIR=${2:-}; shift ;;     # trace 输出目录（默认 <out>/L<L>/prefill）
    --log-level) LOG_LEVEL=${2:-info}; LOG_LEVEL_SET=1; shift ;;  # pure|error|warn|info|debug（pure=只留进度条）
    --pure-log) LOG_LEVEL=pure; LOG_LEVEL_SET=1 ;;  # 等价 --log-level pure
    --no-bar) BAR=0 ;;                           # 关进度条（只留常规日志）
    --bar-fd) BAR_FD=${2:-9}; shift ;;           # 真终端 fd（默认 9；非终端会自动降级）
    --bar-every) BAR_EVERY=${2:-30}; shift ;;    # 日志快照间隔（秒）
    --eta-from) ETA_FROM=${2:-}; shift ;;        # 只算不改：用一份 rounds.tsv 验 ETA 模型
    -h|--help) usage 0 ;;
    *) echo "未知参数：$1（--help 看用法）"; exit 2 ;;
  esac
  shift
done

# ── ★ 2026-10-22 用户规则：每个点位至少 4 个 pass（多 pass 取平均/中位的口径）──
if [ "$ITERS" -lt 4 ]; then
  echo "[iters] --iters=$ITERS < 4 → 规则要求至少 4 个 forward，改为 4"
  ITERS=4
fi

# ── 卡号 → world（run_hit 的三种入口共用同一个 world）────────────────────────
#   ★ 2026-10-22 用户口径：多点测试也要能单卡跑（1 个卡号 = world 1）。
#   8 卡是默认口径；1..8 都按卡号个数推导，launch_mini 里 moe_ep_size=world。
CARDS=$(printf '%s' "$CARDS" | tr -d ' ')
if ! [[ "$CARDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "✗ --cards 格式不对：只允许数字和逗号（收到 '$CARDS'）" >&2; exit 2
fi
IFS=',' read -r -a _CARD_ARR <<< "$CARDS"
WORLD=${#_CARD_ARR[@]}
if [ "$WORLD" -lt 1 ] || [ "$WORLD" -gt 8 ]; then
  echo "✗ --cards 需要 1..8 个卡号（收到 $WORLD 个：$CARDS）" >&2; exit 2
fi
if [ "$(printf '%s\n' "${_CARD_ARR[@]}" | sort -u | wc -l)" != "$WORLD" ]; then
  echo "✗ --cards 有重复卡号：$CARDS" >&2; exit 2
fi
# --prof-ranks：all 或 rank 列表；列表里的 rank 必须 < world
if [ "$PROF_RANKS" != "all" ] && [ -n "$PROF_RANKS" ]; then
  for _pr in $(printf '%s' "$PROF_RANKS" | tr ',' ' '); do
    if ! [[ "$_pr" =~ ^[0-9]+$ ]] || [ "$_pr" -ge "$WORLD" ]; then
      echo "✗ --prof-ranks=$PROF_RANKS 超出 world=$WORLD（rank 从 0 开始）" >&2; exit 2
    fi
  done
fi
# ★ 2026-10-27：稳态读数必须取**未被采集**的 rank（默认 rank1）——prof 轮里 rank0 被 msprof
#   拖慢，不能拿它的 forward 时间当性能数。PROF_RANKS=all 时退化为 rank0（全采口径）。
REPORT_RANK=0
if [ "$PROF_RANKS" != "all" ] && [ -n "$PROF_RANKS" ]; then
  # ★ 2026-10-30（用户口径修正）：**必须排除 prof rank 所在 CP 组的全部成员**。
  #   只排除 rank0 是不够的：cp_size=2 时 (0,1) 同组，rank0 带 msprof 插桩后
  #   rank1 在集合通信里必须等它 ⇒ 实测 r1=3.643s vs r2-r7=0.164~0.187s，
  #   慢 20×。若 REPORT_RANK=1 会把"被 prof 污染的 rank"当成稳态基准。
  #   规则：先按 CP 组算出被污染集合（prof_rank 所在组的全部成员），再取剩余最小值。
  _cpn=${DS41M_CP_SIZE:-1}
  case "$_cpn" in ''|*[!0-9]*) _cpn=1 ;; esac
  [ "$_cpn" -gt 0 ] || _cpn=1
  REPORT_RANK=""
  for _r in $(seq 0 $((WORLD - 1))); do
    _grp_start=$(( (_r / _cpn) * _cpn ))
    _polluted=0
    for _p in $(printf '%s' "$PROF_RANKS" | tr ',' ' '); do
      case "$_p" in ''|*[!0-9]*) continue ;; esac
      _pg=$(( (_p / _cpn) * _cpn ))
      [ "$_pg" = "$_grp_start" ] && { _polluted=1; break; }
    done
    if [ "$_polluted" = "0" ]; then REPORT_RANK=$_r; break; fi
  done
  if [ -z "$REPORT_RANK" ]; then
    REPORT_RANK=0
    echo "⚠ 所有 CP 组都含 prof rank（cp=$_cpn world=$WORLD prof=$PROF_RANKS）：" >&2
    echo "  稳态数字全部被 msprof 污染，**不可作性能数**（PERF_RULES §0）" >&2
  fi
fi
[ "$WORLD" = 1 ] && echo "[world] 单卡口径（--cards $CARDS → world=1）；MoE/engram 是否跳过由 DS41M_SKIP_* 决定"

# 默认输出根：本机共享盘优先；别的机器上没有就落到仓内 _out/（正式跑前用 -o 指到你的大盘）
_default_out() {
  for d in /mnt/xds/sfs/l00972766 /mnt/l00972766; do
    [ -d "$d" ] && { printf '%s/hit97_8points\n' "$d"; return; }
  done
  printf '%s/_out/hit97_8points\n' "$PWD"
}
OUT=${OUT_DIR:-$(_default_out)}

# ── 每点的 trace 目录：默认 <out>/L<L>/prefill；-P 给了就用它（含 {L} 则替换、不再追加）──
pdir_for() {  # $1 = L
  if [ -z "${PROF_DIR:-}" ]; then printf '%s/L%s/prefill\n' "$OUT" "$1"; return; fi
  case "$PROF_DIR" in
    *"{L}"*) printf '%s\n' "${PROF_DIR//\{L\}/$1}" ;;
    *)       printf '%s/L%s/prefill\n' "$PROF_DIR" "$1" ;;
  esac
}

# ── 进度/ETA：每轮耗时记进 rounds.tsv，ETA 用"已完成轮次均值"（实测与 R 基本无关：
#    hit97_fromB 那次 prof ≈ 88s / noprof ≈ 60s 横跨 R=1152..7936 都是平的，固定开销占大头；
#    仅**全 run 第一轮**因冷编译要 ~246s，显示时单独提示）──
fmt_sec() {  # 90 → 1m30s
  local s=${1:-0}
  if [ "$s" -ge 3600 ]; then printf '%dh%02dm' $((s/3600)) $(((s%3600)/60))
  elif [ "$s" -ge 60 ]; then printf '%dm%02ds' $((s/60)) $((s%60))
  else printf '%ds' "$s"; fi
}

# rounds.tsv 列：1=run_id 2=idx 3=L 4=R 5=round 6=rc 7=sec 8=trace_n
record_round() {  # idx L R round rc sec trace_n
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$RUN_ID" "$1" "$2" "$3" "$4" "$5" "${6:-}" "${7:-}" >> "$ROUNDS_TSV"
}

round_sec_of() {  # $1=round(prof|noprof) $2=L → 该轮用时（秒，空=没有）
  awk -F'\t' -v r="$1" -v L="$2" '$5==r && $3==L && $7+0>0 {s=$7} END{ if (s!="") print s }' "$ROUNDS_TSV" 2>/dev/null
}

chk_lines() {  # 口径自检三行（prof 轮在 logs/，noprof-only 时只有 logs2/）
  local D=$1 tag=$2
  if [ ! -f "$D/logs/rank0.log" ] && [ ! -f "$D/logs2/rank0.log" ]; then
    echo "   ${tag}口径自检：日志文件已关闭（--pure-log），跳过"
    return
  fi
  echo "   ${tag}口径自检：$(grep -hm1 'elb/balroute' "$D/logs/rank0.log" "$D/logs2/rank0.log" 2>/dev/null | head -1 || true)"
  echo "             $(grep -hm1 '\[起跑线\]' "$D/logs/rank0.log" "$D/logs2/rank0.log" 2>/dev/null | head -1 || true)"
  echo "             $(grep -hm1 'engram\] 侧流已启用' "$D/logs/rank0.log" "$D/logs2/rank0.log" 2>/dev/null | head -1 || echo '（未启用 engram 侧流）')"
  [ -z "$(grep -hm1 'elb/balroute' "$D/logs/rank0.log" "$D/logs2/rank0.log" 2>/dev/null)" ] && \
    echo "             ⚠️ 没查到 elb/起跑线 行（日志目录：$D/logs{,2}）"
}

_avg_of() {  # $1=round → "均值 轮数"（先验：prof 90s / noprof 60s）
  local a n
  # 中位数（不是均值）：全 run 第一轮含冷编译、比后续慢 ~3×，均值会被它带偏
  read -r a n < <(awk -F'\t' -v r="$1" '$5==r && $7+0>10 {v[k++]=$7}
      END{ if(k==0){print "0 0"; exit}
           for(i=1;i<k;i++){x=v[i]; j=i-1; while(j>=0 && v[j]>x){v[j+1]=v[j]; j--} v[j+1]=x}
           printf "%d %d", v[int(k/2)], k }' "$ROUNDS_TSV" 2>/dev/null || echo "0 0")
  if [ "${a:-0}" -lt 10 ]; then          # 没测过 / 测出来是 0 → 用先验
    if [ "$1" = prof ]; then echo "90 0"; else echo "60 0"; fi
    return
  fi
  echo "$a $n"
}

eta_rest() {  # 剩余秒 + 依据（prof均值 noprof均值 已测轮数）
  read -r pa np_ < <(_avg_of prof); read -r na nn < <(_avg_of noprof)
  local rest=0
  while IFS=: read -r _l _m _h; do
    [ -z "${_l:-}" ] && continue
    [ "$DO_PROF" = 1 ] && [ -z "$(round_sec_of prof "$_l")" ] && rest=$((rest + pa))
    [ "$DO_NOPROF" = 1 ] && [ -z "$(round_sec_of noprof "$_l")" ] && rest=$((rest + na))
  done <<< "$POINTS"
  echo "$rest $pa $na $((np_ + nn))"
}

print_progress_point() {  # $1=已跑完点数
  local done_n=$1 rest_line rest pa na measured basis
  rest_line=$(eta_rest); set -- $rest_line
  rest=$1; pa=${2:-90}; na=${3:-60}; measured=${4:-0}
  if [ "${measured:-0}" -ge 1 ]; then
    basis="按已完成 ${measured} 轮中位数 prof~${pa}s/noprof~${na}s"
  else basis="先验 prof~${pa}s/noprof~${na}s"; fi
  echo
  echo "──── 总进度 ${done_n}/${POINT_TOTAL} 点 · 已用 $(fmt_sec $(( $(date +%s) - RUN_T0 ))) · 剩余粗估 ~$(fmt_sec "$rest")（$basis）"
  printf '  %-8s %-7s %-9s %-9s %s\n' L R prof noprof 状态
  while IFS=: read -r L M H; do
    [ -z "${L:-}" ] && continue
    local ps ns st
    ps=$(round_sec_of prof "$L"); ns=$(round_sec_of noprof "$L")
    if [ -d "$(pdir_for "$L")" ] && [ "$(ls "$(pdir_for "$L")" 2>/dev/null | wc -l)" -ge 8 ]; then st="✓"
    elif [ -n "$ps$ns" ]; then st="…"
    else st="待跑"; fi
    printf '  %-8s %-7s %-9s %-9s %s\n' "$L" "$((M + (RTAIL>KEEP?RTAIL:KEEP)))" \
      "${ps:+$(fmt_sec "$ps")}" "${ns:+$(fmt_sec "$ns")}" "$st"
  done <<< "$POINTS"
  [ "$done_n" = 1 ] && echo "  （首点含冷编译：实测第一轮比后续慢 ~3×，之后估计里已按中位数剔除）"
}

# ── 官方矩阵：L → M（R = M + KEEP(140)；h = 1 - M/L）──
# 默认点位（不带 -L/--points 时）：DeepInfer 口径 h=0.97，M=round((1-h)L)、无 128 对齐
DEFAULT_LS="16384 32768 65536 81920 131072 262144 524288 1048576"

if [ "$LIST" = 1 ]; then
  echo "DeepInfer 口径默认点（h=0.97，M=round(0.03L) 无 128 对齐，R=M+KEEP=140）："
  echo "  16k→M492/R632  32k→M983/R1123  64k→M1966/R2106  80k→M2458/R2598"
  echo "  128k→M3932/R4072  256k→M7864/R8004  512k→M15729/R15869  1m→M31457/R31597"
  echo "改命中率：--hit 0.97（全局）/ -L 64k@0.99（逐点）/ -L 100k:3072（直接给 M）"
  exit 0
fi

# ── 点位解析（Python 干这活更稳）：token 形如 16k / 16k@0.97 / 100k:3072 / 100k:3072@0.99 / all
#    优先级：逐点 @h > 逐点 :M > 全局 --hit > 官方矩阵默认
#    输出：每行 "L:M:h"（h 是写进配置 hit.ratio 的值；M 只用于显示与实际落盘的 ratio 一致）──
POINTS=$(python3 - "$HIT" "$PLAIN" "${SEQS[@]+"${SEQS[@]}"}" <<'PY'
import re, sys

DEFAULT_LS = [16384, 32768, 65536, 81920, 131072, 262144, 524288, 1048576]

def plen(s):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([km]?)", s.strip().lower().replace("_", ""))
    if not m:
        sys.exit(f"无法解析的序列长度：{s!r}（写 16k / 1m / 16384 这样）")
    return int(float(m.group(1)) * {"": 1, "k": 1024, "m": 1048576}[m.group(2)])

def phit(s):
    s = s.strip()
    try:
        v = float(s[:-1]) / 100.0 if s.endswith("%") else float(s)
    except ValueError:
        sys.exit(f"命中率写法不认：{s!r}（用 0.97 / 97% / 97；一个点上不能同时给 @h 和 :M）")
    if v > 1.0:
        v = v / 100.0                      # 97 → 0.97
    if not 0.0 < v < 1.0:
        sys.exit(f"命中率要在 (0,1)：{s!r}")
    return v

hit_g, plain = (sys.argv[1] or None), (sys.argv[2] == "1")
toks, out, seen = [t for t in sys.argv[3:] if t], [], set()
if not toks:
    toks = ["all"]
# all → 展开成矩阵里的 L（再走下面统一逻辑，于是 --hit 对 all 一样生效）
expanded = []
for t in toks:
    expanded.extend(f"{L}@0.97" for L in DEFAULT_LS) if t.lower() == "all" else expanded.append(t)
for tok in expanded:
    spec, _, hspec = tok.partition("@")
    lspec, _, mspec = spec.partition(":")
    L = plen(lspec)
    if L in seen:
        continue                           # 同一个 L 给了两次：用第一次
    if plain:                              # --plain：整段 L 全算（hit.ratio=0）
        out.append((L, L, 0.0)); seen.add(L); continue
    if hspec:
        h = phit(hspec)
    elif mspec:
        h = round(1 - int(mspec) / L, 7)
    elif hit_g is not None:
        h = phit(hit_g)
    else:
        # DeepInfer 口径默认 h=0.97（不再用旧的 128 对齐 M 表）
        h = 0.97
    seen.add(L)
    # DeepInfer 口径：non_cached 直接给定（mspec）或 round((1-h)*L)，**不做 128 对齐**。
    out.append((L, int(mspec) if (mspec and not hspec)
                else int(round((1 - h) * L)), h))
for L, M, h in out:
    print(f"{L}:{M}:{h}")
PY
) || exit 2
[ -n "$POINTS" ] || { echo "没有要跑的点"; exit 2; }

# ETA 模型离线验证入口：--dry-run --eta-from <rounds.tsv> 只算不改
if [ -n "$ETA_FROM" ]; then
  echo "（--eta-from $ETA_FROM）每点预测（模型：固定开销为主、与 R 基本无关）"
  ROUNDS_TSV=$ETA_FROM
  set -- $(eta_rest)
  echo "  剩余合计 ~$(fmt_sec "$1")   依据：$( [ "${4:-0}" -ge 2 ] && echo "已完成 ${4} 轮中位数" || echo 先验 )" \
       "prof~${2}s / noprof~${3}s"
  exit 0
fi

if [ "$DRY" = 1 ]; then
  # --points <file>：dry-run 也必须按文件里的点位打印（此前忽略了它、回落到旧默认 8 点）
  if [ -n "$POINTS_FILE" ]; then
    python3 - "$POINTS_FILE" "$RTAIL" "$KEEP" <<'PYDRY' || exit 2
import json, sys
pts = json.load(open(sys.argv[1])); rtail = int(sys.argv[2]); keep = int(sys.argv[3])
def _one(p):
    bs = p.get("batches")
    if isinstance(bs, list) and bs and isinstance(bs[0], dict):
        return [(int(b.get("L", 0)), float(b.get("hit", 0))) for b in bs]
    return [(int(p.get("L", 0)), float(p.get("hit", 0)))]
items = [x for p in pts for x in _one(p)]
rr_tail = max(rtail, keep)
print(f"（--dry-run）将跑 {len(items)} 个点（replay_tail={rtail}，R = M + {rr_tail}；"
      f"DeepInfer 口径 M=round((1-h)L)，无 128 对齐）")
for L, h in items:
    M = int(round((1 - h) * L))
    print(f"  L={L:<9} M={M:<6} R={M + rr_tail:<7} h={h:<10}")
PYDRY
    exit 0
  fi
  if [ "$PLAIN" = 1 ]; then
    echo "（--dry-run）将跑 $(echo "$POINTS" | wc -l) 个点（--plain：整段全算，R = L；batches=$BATCHES）："
  else
    echo "（--dry-run）将跑 $(echo "$POINTS" | wc -l) 个点（replay_tail=$RTAIL，R = M + $RTAIL；batches=$BATCHES）："
  fi
  while IFS=: read -r L M H; do
    if [ "$PLAIN" = 1 ]; then RR=$L; else RR=$((M + (RTAIL>KEEP?RTAIL:KEEP))); fi
    printf "  L=%-9s M=%-6s R=%-7s h=%-10s → %s\n" "$L" "$M" "$RR" "$H" "$(pdir_for "$L")"
  done <<< "$POINTS"
  exit 0
fi

# 仓根 = 本脚本所在目录的上一级：换机器/换目录都不用改脚本（只认相对位置）
cd "$(cd "$(dirname "$0")/.." && pwd -P)" || exit 1
source ./env_pack.sh >/dev/null 2>&1
# ★ PERF_RULES 门禁（2026-10-18）：性能开关默认值不对就拒绝启动。
_PERF_BASE=$(cd .. && pwd -P)
_PERF_CHECK="$_PERF_BASE/tools/check_perf_defaults.py"
[ -f "$_PERF_CHECK" ] || _PERF_CHECK=/mnt/xds/sfs/l00972766/tools/check_perf_defaults.py
if [ -f "$_PERF_CHECK" ] && ! DS41_BASE="$_PERF_BASE" python3 "$_PERF_CHECK" --runtime; then
  echo "✗ PERF_RULES 检查未通过，拒绝启动（见 PERF_RULES.md；A/B 用 DS41_ALLOW_SUBOPTIMAL=原因 放行）" >&2
  exit 2
fi
mkdir -p "$OUT"
# 顺带在 /mnt/l00972766/ 下也挂一份（本机约定；目录不存在就跳过，别影响别的机器）
[ -d /mnt/l00972766 ] && ln -sfn "$OUT" "/mnt/l00972766/$(basename "$OUT")"

# ★ 进度条要"终端原地刷新、日志只追加快照"：先把**真终端**存到 fd 9。必须在下面那句
#   `exec > >(tee …)` **之前**——之后 stdout 就变成管道了，tee 也看不见 fd 9，
#   所以进度条不可能把 \r 写进日志。nohup/重定向时 [ -t 9 ] 为假 → 只留日志快照。
exec 9>&1
BAR_TTY=0; [ -t 9 ] && BAR_TTY=1

# ── 全程 tee：控制台 + driver 日志（含每个 rank 的实时输出都汇到这里）──
#    ★ pure 模式：驱动自身的 echo **只落日志、不上控制台**（终端只剩 fd 9 的进度条；
#      各 rank 的 stdout 由 ds41_log 丢弃，stderr 里的进度行仍被 launch_mini 的 Bar 解析）。
DRIVER="$OUT/driver_$(date -u +%Y%m%d_%H%M%S).log"
if [ "$LOG_LEVEL" = pure ] || [ "$LOG_LEVEL" = quiet ] || [ "$LOG_LEVEL" = none ]; then
  # ★ 2026-10-22：非 tty（executor / nohup / `| tee`）时没有 fd 9，进度条只剩 stdout 快照；
  #   以前这里把 stdout 全吞进 DRIVER → 终端“什么都没有”。现在把**进度类行**过滤到终端，
  #   其余仍只落 DRIVER（pure 的语义不变：终端只剩进度）。
  if [ "$BAR_TTY" = 1 ]; then
    exec >>"$DRIVER" 2>&1
  else
    exec > >(tee -a "$DRIVER" | grep --line-buffered -E \
      '^\[bar |^@@bar-est|^[[:space:]]*(\[r0\] )?(\[point |──── \[point|多轮耗时|纯 (pass|forward)|多点位总用时|✅|❌|rc=)') 2>&1
  fi
else
  exec > >(tee -a "$DRIVER") 2>&1
fi
ln -sfn "$(basename "$DRIVER")" "$OUT/driver_latest.log"
RUN_ID=$(basename "$DRIVER" .log | sed 's/^driver_//')
ROUNDS_TSV="$OUT/rounds.tsv"
POINT_TOTAL=$(printf '%s\n' "$POINTS" | sed '/^$/d' | wc -l)
PIDX=0; RUN_T0=$(date +%s)

# ── 日志等级（2026-10-21）：pure = 各进程 stdout 全部丢弃，终端只剩进度条 ──
case "$LOG_LEVEL" in
  pure|quiet|none|error|warn|warning|info|debug) ;;
  *) echo "✗ --log-level 只支持 pure|error|warn|info|debug（收到 $LOG_LEVEL）" >&2; exit 2 ;;
esac
export DS41M_LOG_LEVEL="$LOG_LEVEL"

# ★ 2026-10-27 用户规则：--pure-log（DS41M_LOG_LEVEL=pure）默认**不写任何日志文件**——
#   rankN.log/all.log 全落 /dev/null，只有控制台回显（launch_*.log 一份）供解析。
#   要保留每 rank 日志：DS41M_KEEP_LOG_FILES=1。
_logdir_or_none(){ if [ "${LOG_LEVEL:-info}" = "pure" ] && [ "${DS41M_KEEP_LOG_FILES:-0}" != "1" ]; then echo none; else echo "$1"; fi; }
# 无日志模式：控制台只放行汇总行（每点几行），不逐行回显 rank 输出
_console_or_summary(){ if [ "${LOG_LEVEL:-info}" = "pure" ] && [ "${DS41M_KEEP_LOG_FILES:-0}" != "1" ]; then echo summary; else echo "$1"; fi; }

# ★ 2026-10-28 用户规则（PERF_RULES §0 采集规则第 4 条）：**开采集时默认无日志**。
#   只要本轮带 prof（DO_PROF=1）且用户没显式指定日志等级 → 自动 pure（不落 rankN.log/all.log）。
if [ "$DO_PROF" = 1 ] && [ "${LOG_LEVEL_SET:-0}" = 0 ] && [ "${DS41M_LOG_LEVEL:-}" = "" ]; then
  LOG_LEVEL=pure
  echo "[log] 采集轮默认 --pure-log（PERF_RULES §0 采集规则；要留全量日志用 DS41M_KEEP_LOG_FILES=1 或 --log-level info）"
fi

# ── 采集前清掉会改图/改数的调试开关（清之前先报出来，免得"我明明设过"）──
# ★ 2026-10-26：移除了 DS41M_ENGRAM_NOSIDE —— 它只换流、不改数，被清掉会让
#   side/noside A/B 两支都跑同一臂（实测污染了 v2 基线与“侧流无收益”的结论）。
for v in DS41M_MOE_BARRIER DS41M_NO_BALROUTE DS41M_START_NOBARRIER \
         DS41M_NO_CMP DS41M_SKIP_KV DS41M_FAKE_Q DS41M_NO_INVROPE \
         DS41M_MOE_NOSHARED DS41M_MOE_FAKEIN DS41M_MOEMODE DS41M_PIPELINE \
         DS41M_MOE_ONLY DS41M_SYNC DS41M_SYNC_BEFORE DS41M_NO_VISIBLE \
         DS41M_ENGRAM_HOST DS41M_ENGRAM_WIN_MB DS41M_NO_ENGRAM \
         DS41M_PD_MERGE DS41M_TP_DECODE_ONLY DS41M_DECODE_LAYOUT \
         DS41M_DECODE_PREBUILD DS41M_PD_TP_XFER DS41M_DECODE_BARRIER; do
  if [ -n "${!v:-}" ]; then echo "[env] 清掉 $v=${!v}（不在本次采集口径里）"; unset "$v"; fi
done

# ── 索引器内核选择（DS41M_IDX_KERNEL=vendor|dg，2026-10-30 阶段 1a）──
#   dg 走 DeepInfer 内核（deep_gemm_di + deep_select），需要：
#     · PYTHONPATH 加 rt/deepgemm-di-pkg（包名 deep_gemm_di，与生产 deep_gemm 隔离）
#     · LD_PRELOAD conda libstdc++（.so 用 gcc14 编，系统 gcc12 缺 CXXABI_1.3.15）
#   ★ 由脚本自动注入，不给人工设置的机会（208 会话要求）。
if [ "${DS41M_IDX_KERNEL:-dg}" = "dg" ]; then
  _DG_PKG=/mnt/xds/sfs/l00972766/rt/deepgemm-di-pkg
  _DG_DS=/mnt/xds/sfs/l00972766/rt/deepselect-pkg
  if [ ! -d "$_DG_PKG/deep_gemm_di" ]; then
    echo "✗ DS41M_IDX_KERNEL=dg 但找不到 $_DG_PKG/deep_gemm_di" >&2; exit 2
  fi
  export PYTHONPATH="$_DG_PKG:$_DG_DS:${PYTHONPATH:-}"
  # ★★ 2026-10-31（主控，去掉容器本地依赖这道"草盖"）：libstdc++ 优先用**共享盘**
  #   `rt/ds-env-lib`（从 anaconda3 libstdc++.so.6.0.34 拷来，含 CXXABI_1.3.15），
  #   其次 /opt/ds-env/lib（208 旧 conda env）。系统 libstdc++ 若已满足 1.3.15 则不必 preload。
  #   背景：原实现硬编码 /opt/ds-env，导致 dg 只能在 208 那个容器跑，099 执行器直接 exit 2。
  _DG_LIB=""
  for _c in /mnt/xds/sfs/l00972766/rt/ds-env-lib /opt/ds-env/lib; do
    if [ -f "$_c/libstdc++.so.6" ]; then _DG_LIB="$_c"; break; fi
  done
  if strings /lib64/libstdc++.so.6 2>/dev/null | grep -q "CXXABI_1.3.15"; then
    echo "[env] 索引器内核 = dg（deep_gemm_di + deep_select，系统 libstdc++ 已含 CXXABI_1.3.15，不注入 LD_PRELOAD）"
  else
    if [ -z "$_DG_LIB" ]; then
      echo "✗ DS41M_IDX_KERNEL=dg：系统 libstdc++ 缺 CXXABI_1.3.15，且共享盘/本地都没有可用的 libstdc++.so.6" >&2
      exit 2
    fi
    export LD_PRELOAD="$_DG_LIB/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "[env] 索引器内核 = dg（deep_gemm_di + deep_select，LD_PRELOAD=$_DG_LIB/libstdc++.so.6）"
  fi
else
  echo "[env] 索引器内核 = vendor（显式回退，融合 quant_lightning_indexer；默认是 dg）"
fi

# ── 显式后端（--moemode）：清理循环只防"环境残留"，不挡显式请求 → 这里写回 ──
#   背景（2026-10-21）：以前 `DS41M_MOEMODE=deepgemm bash tools/run_hit.sh …` 会被上面的
#   清理循环 unset，后端静默回落 auto→ascendc（用户实测 16 点 trace 里仍是 npu_mega_moe）。
if [ -n "${MOEMODE:-}" ]; then
  case "$MOEMODE" in
    ascendc|deepgemm) export DS41M_MOEMODE="$MOEMODE" ;;
    *) echo "✗ --moemode 只支持 ascendc|deepgemm（收到 $MOEMODE）" >&2; exit 2 ;;
  esac
  echo "[env] 后端显式指定：DS41M_MOEMODE=$DS41M_MOEMODE（--moemode）"
fi

# ── decoder-side SWA bounded replay（官方 ds41_light/adaptlib/replay.py 同款，vLLM#58132）──
#   只在**全量 prefill**（非 hit）的 cut 之后各层生效：只跑最后 W 个 token；hit 口径本来就
#   是尾段语义 → runner 会自动跳过（与官方"replay 是 prefill 特性"一致）。
if [ "$REPLAY" = 1 ]; then
  export DS41M_DECODER_REPLAY=1
  [ "$REPLAY_W" != 0 ] && export DS41M_DECODER_REPLAY="{\"on\":true,\"w\":$REPLAY_W,\"cut\":$REPLAY_CUT}"
  echo "  replay：开（W=${REPLAY_W:-0→sliding_window}，cut=${REPLAY_CUT:-0→最后一个 kv_source+1}）"
else
  export DS41M_DECODER_REPLAY=0    # 显式关（否则 runner 的默认开会把它重新打开）
  echo "  replay：关（--no-replay）"
fi

# ── 模块运行时（Phase 2，2026-10-24 用户拍板默认开；=0/off 回退 legacy）──
export DS41M_MODULE_RUNTIME="${DS41M_MODULE_RUNTIME:-1}"

# ── 入图代码已于 2026-10-26 彻底删除（恒 eager）：不再设置任何入图开关 ──
[ -n "$PROF_WARMUP" ] || PROF_WARMUP=1  # eager 版也先跑 1 轮，别把权重装载写进 trace

# ── 多点位预热跑（--points FILE）：一个进程（8 rank 各一）跑完全部点位 ──────────────
#   ★ 用户口径（2026-10-20）："不同点位不需要重复重启" —— 权重/进程组/算子注册只做一次；
#   每个点位内部：重建层栈/池/输入 → 预热 K 轮（不进 trace）→ 采集 iters 轮。
#   点位间由 runner 清掉几何相关缓存（图/池/ctx/META），见 points_runner.py 顶部注释。
#   点位 JSON：[{"L":81920,"hit":0.96,"batches":1,"replay":true}, …]
#   trace 落 <out>/L<L>/h<hhh>_b<N>_r<0|1>/prefill（N=batches，r=replay 开关）。
if [ -n "${POINTS_FILE:-}" ] || [ -n "${POINTS_JSON:-}" ] || [ -n "${POINTS_SPEC:-}" ]; then
  if [ ${#SEQS[@]} -gt 0 ]; then
    echo "✗ --points / --points-json / --points-spec 与 -L 不能同用（点位全部由点位表给）"; exit 2
  fi
  if [ -n "${POINTS_JSON:-}" ] && [ -n "${POINTS_SPEC:-}" ]; then
    echo "✗ --points-json 与 --points-spec 只能给一个"; exit 2
  fi
  if [ -n "${POINTS_JSON:-}" ] || [ -n "${POINTS_SPEC:-}" ]; then
    if [ -n "${POINTS_FILE:-}" ]; then
      echo "✗ --points 与 --points-json/--points-spec 不能同用"; exit 2
    fi
    # ★ 一键粘贴：把内联点位表落到本次 out 目录（随产物留档，可复盘）
    POINTS_FILE="$OUT/points_inline.json"
    if [ -n "${POINTS_JSON:-}" ]; then
      python3 - "$POINTS_FILE" "$POINTS_JSON" <<'PY' || exit 2
import json, sys
raw = sys.argv[2]
try:
    pts = json.loads(raw)
except Exception as e:                                   # noqa: BLE001
    sys.exit(f"✗ --points-json 不是合法 JSON：{e}")
if not isinstance(pts, list) or not pts:
    sys.exit("✗ --points-json 要非空数组")
def _norm_hit(hv, where):
    if isinstance(hv, str):
        hv = hv.strip().rstrip("%")
    h = float(hv)
    h = h / 100.0 if h > 1 else h
    if not (0.0 <= h < 1.0):
        sys.exit(f"✗ {where} hit 必须 ∈ [0,1)（也接受 96/96%）：{hv!r}")
    return h

def _norm_L(Lv, where):
    L = int(Lv)
    if L <= 0 or L % 128:
        sys.exit(f"✗ {where} L 必须是 128 的正倍数：{L}") 
    return L

for i, p in enumerate(pts):
    if not isinstance(p, dict):
        sys.exit(f"✗ --points-json 第 {i + 1} 项必须是对象：{p}")
    where0 = f"--points-json 第 {i + 1} 项"
    # ★ v2：[{tag, batches:[{L,hit}, …]}]；旧格式 [{L,hit,batches:N}] 也认
    if isinstance(p.get("batches"), list):
        if not p["batches"]:
            sys.exit(f"✗ {where0} batches 数组不能为空")
        for bi, b in enumerate(p["batches"], 1):
            if not isinstance(b, dict):
                sys.exit(f"✗ {where0} 第 {bi} 批必须是 {{L, hit}}：{b}")
            w = f"{where0} 第 {bi} 批"
            if "L" not in b:
                sys.exit(f"✗ {w} 缺 L")
            b["L"] = _norm_L(b["L"], w)
            if b.get("hit") is not None:
                b["hit"] = _norm_hit(b["hit"], w)
        continue
    # 旧格式
    if "L" not in p:
        sys.exit(f"✗ {where0} 缺 L（或改用 v2：batches:[{{L,hit}}]）：{p}")
    p["L"] = _norm_L(p["L"], where0)
    if p.get("hit") is not None:
        p["hit"] = _norm_hit(p["hit"], where0)
    if "batches" in p:
        b = int(p["batches"])
        if b < 1:
            sys.exit(f"✗ {where0} batches 必须 ≥1：{b}")
        p["batches"] = b
    if "replay_tail" in p:
        t = int(p["replay_tail"])
        if t < 1:
            sys.exit(f"✗ {where0} replay_tail 必须 ≥1：{t}")
        p["replay_tail"] = t
json.dump(pts, open(sys.argv[1], "w"), ensure_ascii=False, indent=2)
print(f"[points] 内联 JSON：{len(pts)} 个点位 → {sys.argv[1]}")
PY
    else
      python3 - "$POINTS_FILE" "$POINTS_SPEC" <<'PY' || exit 2
import json, re, sys


def parse_l(s):
    m = re.fullmatch(r"(\d+)([km]?)", s.strip().lower())
    if not m:
        sys.exit(f"✗ 点位 spec 里 L 不合法：{s!r}")
    return int(m.group(1)) * {"": 1, "k": 1024, "m": 1024 * 1024}[m.group(2)]


def parse_hit(s):
    t = s.strip().rstrip("%")
    try:
        v = float(t)
    except ValueError:
        sys.exit(f"✗ 点位 spec 里 hit 不合法：{s!r}")
    v = v / 100.0 if v > 1 else v
    if not (0.0 <= v < 1.0):
        sys.exit(f"✗ hit 必须 ∈ [0,1)：{s!r}")
    return v


out = []
for tok in re.split(r"[,\s]+", sys.argv[2].strip()):
    if not tok:
        continue
    parts = tok.split("@")
    if len(parts) != 2:
        sys.exit(f"✗ 点位 token 要写成 L@hit[/bN][/r0|1][/tTAIL]：{tok!r}")
    L = parse_l(parts[0])
    rest = parts[1].split("/")
    d = {"L": L, "hit": parse_hit(rest[0]), "batches": 1, "replay": True}
    for kv in rest[1:]:
        k = kv.strip().lower()
        m = re.fullmatch(r"b(\d+)", k)
        if m:
            d["batches"] = int(m.group(1)); continue
        m = re.fullmatch(r"r([01])", k)
        if m:
            d["replay"] = bool(int(m.group(1))); continue
        m = re.fullmatch(r"t(\d+)", k)
        if m:
            d["replay_tail"] = int(m.group(1)); continue
        sys.exit(f"✗ 点位 token 里未知修饰 {kv!r}（{tok!r}）")
    if L % 128:
        sys.exit(f"✗ L 必须是 128 的倍数：{tok!r}")
    if d["batches"] < 1:
        sys.exit(f"✗ batches 必须 ≥1：{tok!r}")
    out.append(d)
if not out:
    sys.exit("✗ --points-spec 为空")
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=2)
print(f"[points] 内联简写：{len(out)} 个点位 → {sys.argv[1]}")
PY
    fi
    # 只生成点位文件、不跑（校验 spec / 拿一份可复用的 points.json）
    if [ "${DS41_POINTS_ONLY:-0}" = "1" ]; then
      echo "[points] 已生成：$POINTS_FILE"; cat "$POINTS_FILE"; exit 0
    fi
  fi
  [ -f "$POINTS_FILE" ] || { echo "✗ 找不到 --points 文件：$POINTS_FILE"; exit 2; }
  read -r NPOINTS FIRST_L < <(python3 - "$POINTS_FILE" <<'PY'
import json, sys
pts = json.load(open(sys.argv[1]))
if not isinstance(pts, list) or not pts:
    sys.exit(f"✗ points 文件要非空 list：{sys.argv[1]}")
def _first_L(p):
    """v2: [{tag, batches:[{L,hit}]}]; legacy: [{L,hit,batches:N}]"""
    bs = p.get("batches")
    if isinstance(bs, list) and bs and isinstance(bs[0], dict):
        return int(bs[0].get("L", 0) or 0)
    return int(p.get("L", 0) or 0)
for i, p in enumerate(pts):
    if _first_L(p) <= 0:
        sys.exit(f"✗ 第 {i + 1} 个点位没有可用的 L（v2 要 batches[0].L）：{p}")
print(len(pts), _first_L(pts[0]))
PY
) || exit 2
  CFG=$OUT/config_points.yaml
  # 基础配置：合成权重 + fake 前缀；L/hit/batches/replay 由 runner 按点位写回 cfg
  python3 - "$CFG" "$POINTS_FILE" "$RTAIL" <<'PY'
import json, sys, yaml
cfg = yaml.safe_load(open('configs/mini_ds41_flash_hit97_64k.yaml'))
pts = json.load(open(sys.argv[2]))
p0 = pts[0]
_b0 = p0.get("batches")
_b0 = _b0[0] if isinstance(_b0, list) and _b0 and isinstance(_b0[0], dict) else p0
cfg['run']['seq_len'] = int(_b0.get('L', p0.get('L', 0)) or 0)
cfg['hit'] = {'ratio': float(_b0.get('hit', p0.get('hit', 0)) or 0),
              'replay_tail': int(p0.get('replay_tail', sys.argv[3])),
              'source': 'fake'}
cfg['weights']['mode'] = 'synthetic'
yaml.safe_dump(cfg, open(sys.argv[1], 'w'), allow_unicode=True, sort_keys=False)
PY
  PROF_ARGS=(); [ "$DO_PROF" = 1 ] && PROF_ARGS=(--prof)
  echo "===== 多点位预热跑（mini：dp8/ep8/cp1 + elb 强均路由 + 合成假权重 + engram 侧流）====="
  echo "开始 $(date -u +'%Y-%m-%d %H:%M:%SZ')    驱动日志 $DRIVER"
  P_SUM=$(python3 - "$POINTS_FILE" <<'PY'
import json, sys
pts = json.load(open(sys.argv[1]))
def _desc(p):
    bs = p.get("batches")
    if isinstance(bs, list) and bs and isinstance(bs[0], dict):
        return "+".join(f"{int(b.get('L', 0))}@{b.get('hit', 0)}" for b in bs)
    return f"{p.get('L')}@{p.get('hit', 0)}/b{p.get('batches', 1)}"
print(" ".join(_desc(p) for p in pts))
PY
)
  echo "点位 $NPOINTS 个（一次进程）：$P_SUM"
  echo "轮次：prof=$DO_PROF  预热=$PROF_WARMUP 采集=$ITERS  卡=$CARDS  回显=$CONSOLE"
  df -h "$OUT" | tail -1 | sed 's/^/磁盘 /'
  T0=$(date +%s)
  python3.11 core/launch_mini.py --config "$CFG" --world "$WORLD" --cards "$CARDS" \
      --logdir "$(_logdir_or_none "$OUT/logs")" --timeout "$TIMEOUT" --console "$(_console_or_summary "$CONSOLE")" \
      --seq "$FIRST_L" --iters "$ITERS" --batches 1 --batch-mode auto \
      --out "$OUT" --warmup-iters "$PROF_WARMUP" --prof-ranks "$PROF_RANKS" \
      --points "$POINTS_FILE" ${PROF_ARGS[@]+"${PROF_ARGS[@]}"} \
      --alop-config "$ALOP_CFG" 2>&1 | tee "$OUT/launch_points.log"
  RC=${PIPESTATUS[0]}
  SEC=$(( $(date +%s) - T0 ))
  echo
  echo "──── 多点位稳态（rank${REPORT_RANK}，未采 prof 的 rank）"
  if [ -f "$OUT/logs/rank${REPORT_RANK}.log" ]; then
    _pts_src=$(cat "$OUT/logs/rank${REPORT_RANK}.log" 2>/dev/null)
  else
    _pts_src=$(grep -h "^\[r${REPORT_RANK}\] " "$OUT/launch_points.log" 2>/dev/null | sed "s/^\[r${REPORT_RANK}\] //")
  fi
  printf '%s\n' "$_pts_src" | grep -h '\[point ' | sed 's/^/  /'
  printf '%s\n' "$_pts_src" | grep -h '多轮耗时' | sed 's/^/  /'
  if [ "$DO_PROF" = 1 ]; then
    echo
    echo "──── trace 核对"
    python3 - "$POINTS_FILE" "$OUT" "$PROF_RANKS" "$WORLD" <<'PY'
import json, os, sys
pts = json.load(open(sys.argv[1])); out = sys.argv[2]
pr = sys.argv[3].strip().lower()
world = int(sys.argv[4])
want = world if pr in ("", "all") else len([x for x in pr.split(",") if x.strip()])
bad = 0

def tag_of(p):
    """与 points_runner 的产物目录名逐字一致：
       v2 batches=[{L,hit},…] → (L1_h1&L2_h2)_r1；旧 batches:N → (L_h&…)_r1；
       per-DP（dps）的 tag 由点位自带 tag，实际目录 <tag>_r1。"""
    rep = 1 if p.get("replay", True) else 0
    if isinstance(p.get("dps"), list) and p.get("dps"):
        return f"{p.get('tag') or 'perdp'}_r{rep}"
    bs = p.get("batches")
    if isinstance(bs, list) and bs and isinstance(bs[0], dict):
        specs = [(int(b.get("L", 0)), float(b.get("hit", 0) or 0)) for b in bs]
    else:
        L = int(p.get("L", 0)); h = float(p.get("hit", 0) or 0)
        nb = max(1, int(p.get("batches", 1) or 1))
        specs = [(L, h)] * nb
    name = "&".join(f"{L}_{h:.4g}" for L, h in specs)
    return f"({name})_r{rep}"

import glob
for i, p in enumerate(pts, 1):
    tag = tag_of(p)
    d = os.path.join(out, tag, "prefill")
    # ★ 2026-10-31：`--points-spec` 路径的产物目录是**扁平名**
    #   （cp1dp1ep1_L8192_hit0.5_batch1_replayon_prefill，用户口径），
    #   tag_of() 算的是 `(8192_0.5)_r1` → 旧代码 100% 判 "trace 0/1 ✗"，
    #   而 PROF_* 其实已经落盘（125_prof_cp1 实测）。目录不存在就按扁平名 glob 兜底。
    if not os.path.isdir(d):
        _bs = p.get("batches")
        if isinstance(_bs, list) and _bs and isinstance(_bs[0], dict):
            _L = int(_bs[0].get("L", 0)); _h = float(_bs[0].get("hit", 0) or 0)
        else:
            _L = int(p.get("L", 0)); _h = float(p.get("hit", 0) or 0)
        _hit = sorted(glob.glob(os.path.join(out, f"*_L{_L}_hit{_h:g}_batch*", "prefill")))
        if _hit:
            d = _hit[0]
    n = len(os.listdir(d)) if os.path.isdir(d) else 0
    ok = n >= want
    bad += 0 if ok else 1
    print(f"  {tag:<44} trace {n}/{want}  {'✓' if ok else '✗'}")
print(f"  {'✓ 全部点位 trace 齐' if not bad else f'✗ {bad} 个点位 trace 不齐'}")
sys.exit(1 if bad else 0)
PY
    TRC=$?; [ "$TRC" != 0 ] && RC=1
  fi
  echo
  echo "多点位总用时 $(fmt_sec "$SEC")    rc=$RC   驱动日志：$DRIVER"
  [ "$RC" = 0 ] && echo "✅ 全部完成" || echo "❌ 有失败（rc=$RC）"
  exit "$RC"
fi

echo "===== hit97 采集（mini：dp8/ep8/cp1 + elb 强均路由 + 合成假权重 + 起跑线 barrier×1 + engram 侧流）====="
echo "开始 $(date -u +'%Y-%m-%d %H:%M:%SZ')    驱动日志 $DRIVER"
echo "点位（$(echo "$POINTS" | wc -l) 个，replay_tail=$RTAIL）：$(echo "$POINTS" | tr '\n' ' ')"
echo "轮次：prof=$DO_PROF noprof=$DO_NOPROF   iters=$ITERS   卡=$CARDS   控制台回显=$CONSOLE   force=$FORCE"
echo "trace 目录：${PROF_DIR:-$OUT/L<L>/prefill（默认）}"
df -h "$OUT" | tail -1 | sed 's/^/磁盘 /'

FAILS=""; NPOINTS=0
while IFS=: read -r L M H; do
  [ -z "${L:-}" ] && continue
  D=$OUT/L$L; mkdir -p "$D"
  # ★ 2026-10-28 用户口径：点位**扁平名**（交付/解析统一用它，不套多层目录）
  #   cp<CP>dp<DP>ep<EP>_L<Lk>_hit<h>_batch<b>_replay<on|off>_prefill
  #   例 cp4dp4ep8_L256k_hit0.97_batch1_replayon_prefill
  #   写到 $D/NAME，trace_report.py 会优先读它（config_point.yaml 里的 cp 可能被 env 覆盖而过期）。
  _cp=${DS41M_CP_SIZE:-1}; case "$_cp" in ''|*[!0-9]*) _cp=1 ;; esac; [ "$_cp" -gt 0 ] || _cp=1
  _dp=$(( WORLD / _cp )); [ "$_dp" -gt 0 ] || _dp=1
  _ltag=$(printf '%d' "$L")          # 序列长写**具体数字**（262144，不用 256k/1m）
  _htag=$(awk -v h="${H:-0}" 'BEGIN{ printf "%g", h+0 }')
  _rname=$([ "${REPLAY:-1}" = "1" ] && echo on || echo off)
  printf 'cp%sdp%sep%s_L%s_hit%s_batch%s_replay%s_prefill\n' \
    "$_cp" "$_dp" "$WORLD" "$_ltag" "$_htag" "${BATCHES:-1}" "$_rname" > "$D/NAME"
  # ★★ 点位配置必须写进**本次输出目录**（不要写共享的 configs/point_L*.yaml）：
  #   2026-10-16 实测，两个人同时跑本脚本时会**并发写同一个 configs/point_L<L>.yaml** →
  #   撕裂写入（生成出 `…decoder_replay: false` 后面接半行 `se`）→ runner 读 YAML 直接崩
  #   （`ScannerError: could not find expected ':'`，prof 轮 16s 就 rc=1、trace 0/8）。
  CFG=$D/config_point.yaml
  # 点位配置：从 baseline 派生，显式钉死本次口径（假权重 + fake 前缀 + 本次的 h/replay_tail）
  python3 - "$CFG" "$L" "$H" "$RTAIL" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open('configs/mini_ds41_flash_hit97_64k.yaml'))
cfg['run']['seq_len'] = int(sys.argv[2])
cfg['hit']['ratio'] = float(sys.argv[3])
# DeepInfer 口径：non_cached 直接写进配置（不做 128 对齐）
cfg['hit']['non_cached_tokens'] = int(round((1 - float(sys.argv[3])) * int(sys.argv[2])))
cfg['hit']['replay_tail'] = int(sys.argv[4])
cfg['hit']['source'] = 'fake'
cfg['weights']['mode'] = 'synthetic'
yaml.safe_dump(cfg, open(sys.argv[1], 'w'), allow_unicode=True, sort_keys=False)
PY
  PIDX=$((PIDX + 1))
  echo
  if [ "$PLAIN" = 1 ]; then RR=$L; else RR=$((M + (RTAIL>KEEP?RTAIL:KEEP))); fi
echo "──── [$(date -u +%H:%M:%SZ)] 点 ${PIDX}/${POINT_TOTAL}  L=$L  M=$M  h=$H  R=$RR  配置 $CFG"
  PT0=$(date +%s)                                            # 本点起点（纯模式小结用）
  echo "[point ${PIDX}/${POINT_TOTAL}] 开始 $(date -u +%H:%M:%S)  L=$L M=$M h=$H R=$RR"
  NF0=$(echo $FAILS | wc -w)                                 # 本点开始前的失败数（结束时比对）

  # 旧 trace 的处理：采全了（8 份）且没 --force → 跳过；否则（没采完/被杀/--force）先挪走，
  # 绝不和新采的混在一个目录里（profiler 是往目录里**追加**新子目录的）。
  # ★ 口径指纹（PARAMS）：换过 --hit / --replay-tail 之后，"已有 8 份 trace"不能当"采过了"，
  #   否则默认跑会把别的命中率采出来的 trace 当本点的结果用。
  WANT="L=$L M=$M h=$H replay_tail=$RTAIL iters=$ITERS replay=$REPLAY w=$REPLAY_W cut=$REPLAY_CUT batches=$BATCHES batch_mode=$BATCH_MODE moemode=${MOEMODE:-auto}"
  RUN_PROF=$DO_PROF
  PDIR=$(pdir_for "$L")                      # trace 落哪（默认 <out>/L<L>/prefill）
  mkdir -p "$(dirname "$PDIR")"
  NPRE=$(ls "$PDIR" 2>/dev/null | wc -l)
  OLD=$(cat "$D/PARAMS" 2>/dev/null || echo "无记录")
  if [ "$DO_PROF" = 1 ] && [ -d "$PDIR" ]; then
    if [ "$NPRE" -ge 8 ] && [ "$FORCE" != 1 ] && [ "$OLD" = "$WANT" ]; then
      echo "   [skip] 已有 $PDIR（$NPRE 份 trace，口径一致：$WANT）"
      RUN_PROF=0
    else
      [ "$NPRE" -ge 8 ] && [ "$FORCE" != 1 ] && \
        echo "   ⚠️ 已有 trace 口径与本次不同（旧：$OLD｜新：$WANT）→ 重采（旧目录挪成 .bak_*）"
      TS=$(date -u +%m%d_%H%M%S)
      mv "$PDIR" "$PDIR.bak_$TS" && echo "   （旧 trace $NPRE 份 → $(basename "$PDIR").bak_$TS）"
      rm -f "$D/PARAMS" "$D/DONE"        # ★ 先摘口径指纹：这次若半途被杀（哪怕恰好写了 8 份
                                         #   trace），下次也绝不会被当成"已采过"跳过
      [ -d "$D/logs" ] && mv "$D/logs" "$D/logs.bak_$TS"
    fi
  fi

  # 进度条参数在每一轮里现算（标签含"总已用/剩余粗估"，每轮开始时是最新的）
  # ── prof 轮：出 trace ──
  if [ "$RUN_PROF" = 1 ]; then
    if [ "$BAR" = 1 ]; then
      BAR_ARGS=("--bar" "点${PIDX}/${POINT_TOTAL} L=$L prof轮"
                "--bar-note" "总已用$(fmt_sec $(( $(date +%s) - RUN_T0 ))) 剩余粗估~$(fmt_sec "$(eta_rest | cut -d' ' -f1)")"
                "--bar-every" "$BAR_EVERY")
      [ "$BAR_TTY" = 1 ] && BAR_ARGS+=(--bar-fd "$BAR_FD")
    fi
    T0=$(date +%s)
    python3.11 core/launch_mini.py --config "$CFG" --world "$WORLD" --cards "$CARDS" \
        --logdir "$(_logdir_or_none "$D/logs")" --timeout "$TIMEOUT" --console "$(_console_or_summary "$CONSOLE")" \
        --seq "$L" --iters "$ITERS" --batches "$BATCHES" --batch-mode "$BATCH_MODE" --prof --out "$PDIR" \
        --warmup-iters "$PROF_WARMUP" --prof-ranks "$PROF_RANKS" \
        --alop-config "$ALOP_CFG" ${BAR_ARGS[@]+"${BAR_ARGS[@]}"} 2>&1 | tee "$D/launch.log"
    RC=${PIPESTATUS[0]}
    SEC=$(( $(date +%s) - T0 ))
    record_round "$PIDX" "$L" "$RR" prof "$RC" "$SEC" "$(ls "$PDIR" 2>/dev/null | wc -l)"
    echo "   [prof] rc=$RC  用时 ${SEC}s（$(fmt_sec "$SEC")）  trace 份数=$(ls "$PDIR" 2>/dev/null | wc -l)/8  → $PDIR"
    chk_lines "$D" ""
    [ "$RC" != 0 ] && FAILS="$FAILS ${L}(prof rc=$RC)"
  fi

  # ── 无 prof 轮：干净单 pass 稳态耗时（和官方表对齐用）──
  if [ "$DO_NOPROF" = 1 ]; then
    if [ "$BAR" = 1 ]; then
      BAR_ARGS=("--bar" "点${PIDX}/${POINT_TOTAL} L=$L 无prof轮"
                "--bar-note" "总已用$(fmt_sec $(( $(date +%s) - RUN_T0 ))) 剩余粗估~$(fmt_sec "$(eta_rest | cut -d' ' -f1)")"
                "--bar-every" "$BAR_EVERY")
      [ "$BAR_TTY" = 1 ] && BAR_ARGS+=(--bar-fd "$BAR_FD")
    fi
    T0=$(date +%s)
    python3.11 core/launch_mini.py --config "$CFG" --world "$WORLD" --cards "$CARDS" \
        --logdir "$(_logdir_or_none "$D/logs2")" --timeout "$TIMEOUT" --console "$(_console_or_summary "$CONSOLE")" \
        --seq "$L" --iters "$ITERS" --batches "$BATCHES" --batch-mode "$BATCH_MODE" ${BAR_ARGS[@]+"${BAR_ARGS[@]}"} 2>&1 | tee "$D/launch_noprof.log"
    RC=${PIPESTATUS[0]}
    SEC=$(( $(date +%s) - T0 ))
    record_round "$PIDX" "$L" "$RR" noprof "$RC" "$SEC" ""
    echo "   [noprof] rc=$RC  用时 ${SEC}s（$(fmt_sec "$SEC")）"
    grep -h '多轮耗时' "$D/logs/rank${REPORT_RANK}.log" 2>/dev/null | tail -1 | sed 's/^/   /'
    [ "$RC" != 0 ] && FAILS="$FAILS ${L}(noprof rc=$RC)"
  fi

  if [ "$(echo $FAILS | wc -w)" = "${NF0:-0}" ]; then      # 本点没新增失败
    touch "$D/DONE"
    printf '%s\n' "$WANT" > "$D/PARAMS"                    # 记下本次口径，供下次 skip 判据
  fi
  # ── 本点小结：第几个点 / 点位 / 用时 / 每 pass（纯模式终端也能看到）──
  #   每 pass 口径 = runner 的「均(除首轮)」（预热与首轮不计）；没有时回退「纯 pass 吞吐」。
  PTSEC=$(( $(date +%s) - PT0 ))
  PER=""
  for _lg in "$D/logs2/rank${REPORT_RANK}.log" "$D/logs/rank${REPORT_RANK}.log" \
             "$D/launch_noprof.log" "$D/launch.log"; do
    case "$_lg" in
      *launch*) _src=$(grep -h "^\[r${REPORT_RANK}\] " "$_lg" 2>/dev/null | sed "s/^\[r${REPORT_RANK}\] //") ;;
      *)        _src=$(cat "$_lg" 2>/dev/null) ;;
    esac
    [ -z "$_src" ] && continue
    PER=$(printf '%s\n' "$_src" | grep -h '多轮耗时' | tail -1 | sed -n 's/.*均(除首轮)=\([0-9.]*\)s.*/\1/p')
    [ -z "$PER" ] && PER=$(printf '%s\n' "$_src" | grep -hE '纯 (pass|forward) 吞吐' | tail -1 | sed -n 's/.*均=\([0-9.]*\)s\/轮.*/\1/p')
    [ -n "$PER" ] && break
  done
  if [ -n "$PER" ]; then
    echo "[point ${PIDX}/${POINT_TOTAL}] 完成 $(date -u +%H:%M:%S)  L=$L h=$H R=$RR  用时 $(fmt_sec "$PTSEC")  每forward=${PER}s（iters=$ITERS，均除首轮）"
  else
    echo "[point ${PIDX}/${POINT_TOTAL}] 完成 $(date -u +%H:%M:%S)  L=$L h=$H R=$RR  用时 $(fmt_sec "$PTSEC")"
  fi
  NPOINTS=$((NPOINTS + 1))
  print_progress_point "$PIDX"                             # 每点一块总进度（含每轮耗时表）
done <<< "$POINTS"
[ "$NPOINTS" -gt 0 ] || { echo "⚠️ 一个点都没跑（点位解析或循环有问题）"; exit 2; }

# ── 汇总（同时落盘 $OUT/SUMMARY.md）──
{
  echo
  echo "===== 汇总 $(date -u +'%Y-%m-%d %H:%M:%SZ') ====="
  printf "%-9s %-9s %-6s %-8s %-22s %-9s %-9s %s\n" L "R=M+KEEP" h trace份 "稳态ms(未采rank)" prof用时 noprof用时 elb/起跑线
  while IFS=: read -r L M H; do
    [ -z "${L:-}" ] && continue
    D=$OUT/L$L
    TR=$(ls "$(pdir_for "$L")" 2>/dev/null | wc -l)
    ST=$(grep -h '多轮耗时' "$D/logs2/rank${REPORT_RANK}.log" 2>/dev/null | tail -1 \
         | sed -n 's/.*均(除首轮)=\([0-9.]*\)s.*/\1/p')
    [ -z "$ST" ] && ST=$(grep -h "^\[r${REPORT_RANK}\] .*多轮耗时" "$D/launch_noprof.log" 2>/dev/null | tail -1 \
         | sed -n 's/.*均(除首轮)=\([0-9.]*\)s.*/\1/p')
    # ★ 2026-10-28：默认 prof-only（无 noprof 轮）时回落到 prof 轮 launch.log 里
    #   **未被采集 rank** 的「多轮耗时」——pure-log 下汇总行仍会回显（§0 采集规则 4），
    #   这就是真实稳态时延（rank0 带 msprof 开销，不能取）。
    [ -z "$ST" ] && ST=$(grep -h "^\[r${REPORT_RANK}\] .*多轮耗时" "$D/launch.log" 2>/dev/null | tail -1 \
         | sed -n 's/.*均(除首轮)=\([0-9.]*\)s.*/\1/p')
    ST=$(awk -v s="${ST:-}" 'BEGIN{ if (s=="") print "—"; else printf "%.1f ms", s*1000 }')
    E=$(grep -chm1 'elb/balroute' "$D/logs/rank0.log" 2>/dev/null || echo 0)
    S=$(grep -chm1 '\[起跑线\]' "$D/logs/rank0.log" 2>/dev/null || echo 0)
    PS=$(round_sec_of prof "$L"); NS=$(round_sec_of noprof "$L")
    CK="elb=$E 起跑线=$S"
    printf "%-9s %-9s %-6s %-8s %-22s %-9s %-9s %s\n" "$L" "$((M + (RTAIL>KEEP?RTAIL:KEEP)))" "$H" "$TR" "${ST:-—}" \
        "${PS:+$(fmt_sec "$PS")}" "${NS:+$(fmt_sec "$NS")}" "$CK"
  done <<< "$POINTS"
  echo
  echo "总用时：$(fmt_sec $(( $(date +%s) - RUN_T0 )))   轮次明细：$ROUNDS_TSV"
  if [ -n "$FAILS" ]; then echo; echo "⚠️ 失败：$FAILS"; else echo; echo "✅ 全部完成"; fi
  echo "驱动日志：$DRIVER"
} | tee "$OUT/SUMMARY.md"

[ -z "$FAILS" ] || exit 1

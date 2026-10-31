#!/usr/bin/env bash
# 生成可迁移的 ds41_mini 便携包：把「不依赖共享盘」所需的 runtime 一起打进去。
#
# ★ 分层打包（默认）：runtime（~16 GB，基本不变）只打一次，源码每次都打（秒级）。
#   bash tools/make_portable_kit.sh                 # 首次：打 runtime base + code；之后：只重打 code
#   bash tools/make_portable_kit.sh --rebuild-base  # runtime 变更后才需要重打 base（约 10+ 分钟）
#   bash tools/make_portable_kit.sh --code-only     # 只打 code（要求 base 已存在）
#   bash tools/make_portable_kit.sh --single        # 旧行为：单个整包 tar.gz（慢）
#   bash tools/make_portable_kit.sh -o /data/kit    # 输出目录（默认共享盘 tmp/ds41_mini_kit）
#   --no-torch / --no-cann / --no-tar 同旧版
#
# 目标机（两个包解到同一父目录，先 base 后 code）：
#   tar -xzf ds41_mini_runtime_base_<tag>.tar.gz
#   tar -xzf ds41_mini_code_<时间>.tar.gz
#   cd ds41_mini && source ./env_pack.sh
#
# 包内布局（env_pack.sh 会优先用包内 runtime/，完全不碰共享盘）：
#   ds41_mini/
#     env_pack.sh
#     runtime/
#       python/                 # 厂商算子 py 包
#       ascend/usr/local/Ascend/custom/   # 厂商算子二进制
#       py213/                  # python3.13 侧依赖
#       deps/                   # yaml/sympy/...（来自 /root/py311deps）
#       shim/
#       tilelang/tilelang-merged2/tilelang + TileKernels-v2
#       deepgemm-pkg/
#       torch_ext/ py311_include/（有就带）
#       pysite_full/…           # --with-torch 才有
#       ascend_full/…           # --with-cann 才有
#
# 目标机仍需要：NPU 驱动 + python3.11；镜像自带 CANN/torch 时不用 --with-*。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
SHARE="${DS41_SHARE:-/mnt/xds/sfs/l00972766}"
OUT_DIR=""
WITH_TORCH=1          # 默认带上 torch/torch_npu/transformers（目标镜像不一定有）
WITH_CANN=1          # 默认带 CANN overlay（与验证环境一致；--no-cann 可省 12 GB）
NO_TAR=0
REBUILD_BASE=0
CODE_ONLY=0
SINGLE=0

usage() { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    -o|--out) OUT_DIR=${2:-}; shift ;;
    --share) SHARE=${2:-}; shift ;;
    --with-torch) WITH_TORCH=1 ;;
    --no-torch) WITH_TORCH=0 ;;
    --with-cann) WITH_CANN=1 ;;
    --no-cann) WITH_CANN=0 ;;
    --no-tar) NO_TAR=1 ;;
    --rebuild-base) REBUILD_BASE=1 ;;
    --code-only) CODE_ONLY=1 ;;
    --single) SINGLE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1（--help 看用法）" >&2; exit 2 ;;
  esac
  shift
done

[ -f "$SRC/env_pack.sh" ] || { echo "✗ 找不到 $SRC/env_pack.sh" >&2; exit 2; }
[ -d "$SHARE/ds41_runtime/python" ] || { echo "✗ 共享盘 runtime 缺失：$SHARE/ds41_runtime/python" >&2; exit 2; }
[ -d "$SHARE/rt/tilelang-merged2/tilelang" ] || { echo "✗ 共享盘 tilelang 缺失：$SHARE/rt/tilelang-merged2" >&2; exit 2; }
[ -d "$SHARE/rt/deepgemm-pkg/deep_gemm" ] || { echo "✗ 共享盘 deep_gemm 缺失：$SHARE/rt/deepgemm-pkg" >&2; exit 2; }
[ -d /root/py311deps/yaml ] || { echo "✗ 缺 /root/py311deps（纯 py 依赖）" >&2; exit 2; }
[ "$CODE_ONLY" = 1 ] && [ "$REBUILD_BASE" = 1 ] && {
  echo "✗ --code-only 与 --rebuild-base 互斥" >&2; exit 2; }

# base 的变体标签：不同 --with-* 组合不能混用同一个 base
if [ "$WITH_TORCH" = 1 ] && [ "$WITH_CANN" = 1 ]; then BASE_TAG=torch_cann
elif [ "$WITH_TORCH" = 1 ]; then BASE_TAG=torch
elif [ "$WITH_CANN" = 1 ]; then BASE_TAG=cann
else BASE_TAG=min; fi

if [ -z "$OUT_DIR" ]; then
  OUT_DIR="$SHARE/tmp/ds41_mini_kit"      # 固定目录：base 才能跨次复用
fi
STAGE="$OUT_DIR/ds41_mini"                # runtime 持久目录（base 从这里打）
CODE_ROOT="$OUT_DIR/.code_build"          # 每次重建，避免陈旧源码混入
TS="$(date +%Y%m%d_%H%M%S)"
BASE="$OUT_DIR/ds41_mini_runtime_base_${BASE_TAG}.tar.gz"
CODE_TAR="$OUT_DIR/ds41_mini_code_${TS}.tar.gz"
MONO_TAR="$OUT_DIR/ds41_mini_portable.tar.gz"
mkdir -p "$OUT_DIR"
echo "== 便携包输出：$OUT_DIR"
df -h "$OUT_DIR" | tail -1 | sed 's/^/   磁盘 /'

# ── 源码（排除日志 / dump / 构建产物）──────────────────────────────────────────
stage_source() {
  local dest=$1
  mkdir -p "$dest"
  echo "== 复制源码（排除 logs/extra-info/dist/__pycache__）"
  tar -C "$SRC" -cf - \
    --exclude='./.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='./logs' --exclude='./logs.*' --exclude='./extra-info' \
    --exclude='./dist' --exclude='./_out' --exclude='./tmp' \
    --exclude='./env_local.sh' --exclude='./PROF_*' --exclude='*.bak*' --exclude='*.new_*' \
    --exclude='./tools/test_*.js' --exclude='./tools/gen_layout_variants.py' \
    --exclude='./tools/make_portable_kit.sh' --exclude='./tools/bench_command_builder.layouts.html' \
    . | tar -C "$dest" -xf -
}

# ── runtime（env_pack.sh 的包内布局）──────────────────────────────────────────
copy_runtime() {
  local R="$STAGE/runtime"
  rm -rf "$R"
  mkdir -p "$R"
  echo "== 复制 runtime（厂商算子 / deps / tilelang / deep_gemm）"
  cp -a "$SHARE/ds41_runtime/python" "$R/"
  cp -a "$SHARE/ds41_runtime/py213" "$R/"
  mkdir -p "$R/ascend/usr/local/Ascend"
  cp -a "$SHARE/ds41_runtime/ascend/usr/local/Ascend/custom" "$R/ascend/usr/local/Ascend/"
  cp -aL /root/py311deps "$R/deps"    # -L：/root/py311deps 里有指向 python3.12 site 的绝对链接
  [ -d "$SHARE/ds41_bench/shim" ] && cp -a "$SHARE/ds41_bench/shim" "$R/shim"
  mkdir -p "$R/tilelang/tilelang-merged2"
  # ★ 必须 -L：共享盘的 tilelang-merged2 是一堆绝对符号链接（指向 tilelang-e77 / 3rdparty）
  cp -aL "$SHARE/rt/tilelang-merged2/tilelang" "$R/tilelang/tilelang-merged2/"
  # ★ 3rdparty 被 -L 实体化后，env.py 里的「3rdparty/../src」落在包内；必须带上真 src，
  #   否则 TILELANG_TEMPLATE_PATH=None → Ascend 后端 "-I"+None → TypeError（topk_gate/engram_hash 全挂）
  TL_3P="$(readlink -f "$SHARE/rt/tilelang-merged2/tilelang/3rdparty")"
  [ -d "$TL_3P/../src" ] || { echo "✗ 找不到 tilelang 模板源码：$TL_3P/../src" >&2; exit 2; }
  cp -a "$TL_3P/../src" "$R/tilelang/tilelang-merged2/tilelang/src"
  cp -a "$SHARE/rt/TileKernels-v2" "$R/tilelang/"
  cp -a "$SHARE/rt/deepgemm-pkg" "$R/"
  # ninja：tilelang/ascendc JIT 需要；目标镜像没有时用包内这份
  mkdir -p "$R/bin"
  NINJA_BIN=$(command -v ninja 2>/dev/null || true)
  [ -z "$NINJA_BIN" ] && [ -x /usr/local/python3.12.13/bin/ninja ] && NINJA_BIN=/usr/local/python3.12.13/bin/ninja
  [ -n "$NINJA_BIN" ] && cp -L "$NINJA_BIN" "$R/bin/ninja" && chmod +x "$R/bin/ninja"
  # JIT 缓存（tilelang / cannbotdsl）：能少一次冷编译
  [ -d "$SHARE/rt/tilelang-cache" ] && cp -a "$SHARE/rt/tilelang-cache" "$R/tilelang/"
  [ -d "$SHARE/rt/cannbotdsl-cache" ] && cp -a "$SHARE/rt/cannbotdsl-cache" "$R/tilelang/"
  [ -d "$SHARE/ds41_runtime/torch_ext" ] && cp -a "$SHARE/ds41_runtime/torch_ext" "$R/"
  [ -d "$SHARE/ds41_runtime/py311_include" ] && cp -a "$SHARE/ds41_runtime/py311_include" "$R/"

  # 去掉 runtime 里的 .git 元数据（不是运行所需；也避免被误认为日志）
  find "$R" -type d -name .git -prune -exec rm -rf {} + 2>/dev/null || true

  if [ "$WITH_TORCH" = 1 ]; then
    echo "== 复制 torch/torch_npu/transformers 站点包（+2 GB；--no-torch 可跳过）"
    cp -a "$SHARE/ds41_runtime/pysite_full" "$R/"
  fi
  if [ "$WITH_CANN" = 1 ]; then
    echo "== --with-cann：复制 CANN overlay（+12 GB）"
    cp -a "$SHARE/ds41_runtime/ascend_full" "$R/"
  fi
}

# 附带 ids（小文件；ref 太大不带）
copy_extras() {
  local dest=$1
  if [ -f "$SHARE/parity/ids_16k.pt" ]; then
    mkdir -p "$dest/parity"
    cp -a "$SHARE/parity/ids_16k.pt" "$SHARE/parity/ids_16k.meta.json" "$dest/parity/" 2>/dev/null || true
  fi
  cp "$SRC/PORTABLE.md" "$dest/PORTABLE.md"
  cp "$SRC/PORTABLE_TROUBLESHOOTING.md" "$dest/PORTABLE_TROUBLESHOOTING.md"
}

if [ "$SINGLE" = 1 ]; then
  # ── 旧行为：单个整包 ──
  rm -rf "$STAGE"
  stage_source "$STAGE"
  copy_runtime
  copy_extras "$STAGE"
  echo "== 目录大小：$(du -sh "$STAGE" | cut -f1)"
  if [ "$NO_TAR" = 1 ]; then
    echo "✅ 已生成目录（未打包）：$STAGE"; exit 0
  fi
  echo "== 打整包 tar.gz（大文件，稍等）"
  tar -C "$OUT_DIR" -czf "$MONO_TAR" ds41_mini
  echo "✅ 便携整包：$MONO_TAR（$(du -h "$MONO_TAR" | cut -f1)）"
  echo "   目标机：tar -xzf $(basename "$MONO_TAR") && cd ds41_mini && source ./env_pack.sh"
  exit 0
fi

# ── 分层：runtime base（只打一次）────────────────────────────────────────────
if [ "$CODE_ONLY" = 1 ] && [ ! -f "$BASE" ]; then
  echo "✗ --code-only 要求 base 已存在：$BASE" >&2
  echo "  先跑一次不带 --code-only 的打包（或 --rebuild-base）" >&2
  exit 2
fi
NEED_BUILD=0
if [ "$CODE_ONLY" = 0 ] && [ "$REBUILD_BASE" = 0 ] && [ -f "$BASE" ]; then
  echo "== runtime base 命中（跳过 16G 拷贝）：$BASE"
elif [ "$CODE_ONLY" = 0 ]; then
  NEED_BUILD=1
fi
runtime_complete() {
  local R="$STAGE/runtime"
  [ -d "$R/python" ] && [ -d "$R/deps/yaml" ] \
    && [ -d "$R/tilelang/tilelang-merged2/tilelang/src" ] \
    && { [ "$WITH_TORCH" != 1 ] || [ -d "$R/pysite_full" ]; } \
    && { [ "$WITH_CANN" != 1 ] || [ -d "$R/ascend_full" ]; }
}
if [ "$NEED_BUILD" = 1 ]; then
  if [ "$REBUILD_BASE" = 0 ] && runtime_complete; then
    echo "== 复用现有 runtime 目录，直接打 base tar（不再拷贝 16G）"
  else
    copy_runtime
  fi
  echo "== 打 runtime base（$(du -sh "$STAGE/runtime" 2>/dev/null | cut -f1)）"
  tar -C "$OUT_DIR" -czf "$BASE" ds41_mini/runtime
  echo "✅ runtime base：$BASE（$(du -h "$BASE" | cut -f1)）"
fi

# ── 分层：code（每次重打，秒级）──────────────────────────────────────────────
rm -rf "$CODE_ROOT"
stage_source "$CODE_ROOT/ds41_mini"
copy_extras "$CODE_ROOT/ds41_mini"
echo "== code 目录大小：$(du -sh "$CODE_ROOT/ds41_mini" | cut -f1)"
if [ "$NO_TAR" = 1 ]; then
  echo "✅ 未打包目录：runtime=$STAGE/runtime  code=$CODE_ROOT/ds41_mini"
  exit 0
fi
tar -C "$CODE_ROOT" -czf "$CODE_TAR" ds41_mini
echo "✅ code 包：$CODE_TAR（$(du -h "$CODE_TAR" | cut -f1)）"
echo "   目标机（先 base 后 code）："
echo "     tar -xzf $(basename "$BASE")"
echo "     tar -xzf $(basename "$CODE_TAR")"
echo "     cd ds41_mini && source ./env_pack.sh"

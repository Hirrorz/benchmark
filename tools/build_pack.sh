#!/bin/bash
# 打包 ds41_mini：代码（git archive，只出跟踪文件）+ 轻量 runtime（≈380 MB）→ tar.gz + sha256
#
#   bash tools/build_pack.sh                 # 产物落 dist/
#   DIST_SHARE=/mnt/xds/sfs/... bash tools/build_pack.sh   # 额外拷一份到共享盘
#
# 轻量 runtime 的边界：**不含** CANN 本体(12 G)与四厂商算子包(1.4 G)——
# 那两样按「目标机镜像自带」处理（env_pack.sh 会自动找，找不到可回退共享盘覆盖层）。
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SHARE=${DS41_SHARE:-/mnt/xds/sfs/l00972766}
RT_SRC=${DS41_RUNTIME_SRC:-$SHARE/ds41_runtime}     # 厂商算子 python 包 + custom_ops
TL_SRC=${TILELANG_RT:-$SHARE/rt}                    # tilelang / TileKernels / JIT 缓存
DEPS_SRC=${DS41_DEPS:-/root/py311deps}              # 纯 python 依赖（软链，拷贝时解引用）
STAMP=${STAMP:-$(date +%m%d)}
NAME=ds41_mini_pack_$STAMP

echo "== 打包 $NAME"
[ -d "$RT_SRC/python" ] || { echo "缺 $RT_SRC/python（厂商算子 py 包）"; exit 1; }
[ -d "$TL_SRC/tilelang-merged2/tilelang" ] || { echo "缺 $TL_SRC/tilelang-merged2"; exit 1; }

STAGE=$(mktemp -d /tmp/ds41_mini_pack.XXXXXX)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$ROOT/dist"

echo "-- ① 代码（git archive HEAD = 只出跟踪文件）"
git -C "$ROOT" archive --format=tar HEAD | tar -x -C "$STAGE"
echo "   $(git -C "$ROOT" ls-files | wc -l) 个跟踪文件；HEAD=$(git -C "$ROOT" rev-parse --short HEAD)"

echo "-- ② runtime（轻量依赖）"
mkdir -p "$STAGE/runtime"
cp -a "$RT_SRC/python" "$STAGE/runtime/python"
cp -a "$RT_SRC/py213"  "$STAGE/runtime/py213"
mkdir -p "$STAGE/runtime/tilelang"
cp -a "$TL_SRC/tilelang-merged2" "$TL_SRC/TileKernels-v2" "$STAGE/runtime/tilelang/"
[ -d "$TL_SRC/tilelang-cache" ]    && cp -a "$TL_SRC/tilelang-cache"    "$STAGE/runtime/tilelang/"
[ -d "$TL_SRC/cannbotdsl-cache" ]  && cp -a "$TL_SRC/cannbotdsl-cache"  "$STAGE/runtime/tilelang/"
[ -d "$SHARE/ds41_bench/shim" ]    && cp -a "$SHARE/ds41_bench/shim"    "$STAGE/runtime/shim"
[ -d "$SHARE/torch_ext_py311" ]    && cp -a "$SHARE/torch_ext_py311"    "$STAGE/runtime/torch_ext"
[ -d "$SHARE/py311_include" ]      && cp -a "$SHARE/py311_include"      "$STAGE/runtime/py311_include"
if [ -d "$DEPS_SRC/yaml" ]; then                    # -L：把软链解成真文件（目标机没有 py3.12 也能用）
    mkdir -p "$STAGE/runtime/deps"
    (cd "$DEPS_SRC" && find . -maxdepth 1 -mindepth 1 -exec cp -rL {} "$STAGE/runtime/deps/" \;)
fi

echo "-- ③ 打 tar.gz + 校验和"
tar -czf "$ROOT/dist/$NAME.tar.gz" -C "$STAGE" .
(cd "$ROOT/dist" && sha256sum "$NAME.tar.gz" > SHA256SUMS)
du -sh "$STAGE" "$ROOT/dist/$NAME.tar.gz" | sed 's/^/   /'
cat "$ROOT/dist/SHA256SUMS"

if [ -n "${DIST_SHARE:-}" ]; then
    mkdir -p "$DIST_SHARE"
    cp "$ROOT/dist/$NAME.tar.gz" "$ROOT/dist/SHA256SUMS" "$DIST_SHARE/"
    echo "-- ④ 已另存一份到 $DIST_SHARE"
fi
echo "== 完成：$ROOT/dist/$NAME.tar.gz"
echo "   目标机：tar -xzf $NAME.tar.gz && cd <目录> && source env_pack.sh && python3.11 tools/selfcheck.py --card N"

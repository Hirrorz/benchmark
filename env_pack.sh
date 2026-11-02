#!/bin/bash
# ds41_mini 环境引导 —— 在任何机器上把「能跑算子」的环境拼出来。
#
#   source env_pack.sh            # 用法（必须 source，环境才留在当前 shell）
#   python3.11 tools/selfcheck.py # 验收：算子自检
#
# 设计口径：**不锁 CANN 版本**——本机有哪套 CANN 就用哪套，只要算子能跑通
# （以 tools/selfcheck.py 的算子自检为准）。
#
# 依赖探测优先序（逐组件，先命中先用）：
#   ① 显式环境变量（DS41_PACK_RUNTIME / DS41_CANN / DS41_DEPS / TILELANG_RT / DS41_PYTHON）
#   ② 包内 runtime/（打包时随包分发，≈380 MB：厂商算子 py 包 + custom_ops .so + py311 依赖
#      + tilelang/TileKernels + 预编译 torch 扩展）
#   ③ 共享盘老路径 /mnt/xds/sfs/l00972766/{ds41_runtime,rt}（本机/同网机器兼容，存在才用）
#
# CANN 本体(12 G)与四厂商算子包(1.4 G)不打进包：按「目标机镜像自带」处理，找不到再回退共享盘。
# 末尾会生成 env_local.sh（下次直接 `source env_local.sh` 即可）。

# ★ 本脚本要在**任意调用方 shell** 里 source（容器里常见 `set -u`）——脚本内部有大量
#   `[ -n "$DS41_PACK_RUNTIME" ]` 这类裸引用，在 nounset 下会**直接把调用方 shell 干掉**；
#   错误再被 `>/dev/null` 吞掉时表现成"日志空白、静默退出"（实测驱动脚本就这么死了两次）。
#   这里临时关 nounset，末尾恢复调用方原来的设置。
case $- in *u*) _ep_u_on=1; set +u;; *) _ep_u_on=0;; esac

_pack_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARE=${DS41_SHARE:-/mnt/xds/sfs/l00972766}

# ★ Ninja 必须在 PATH 上：tilelang / MegaMoE **ascendc** 走 torch C++ 扩展加载，检测不到
#   `ninja --version` 时会**静默回退 deepgemm**（违反 PERF_RULES「一律 ascendc」）。
#   本机 ninja 在 python3.12 的 bin 下；`env -i` 启动的 run_hit 曾经漏掉它 → 全体回退。
if ! command -v ninja >/dev/null 2>&1; then
    if [ -x "$_pack_root/runtime/bin/ninja" ]; then
        export PATH="$_pack_root/runtime/bin:$PATH"          # 便携包自带
    elif [ -x /mnt/xds/sfs/l00972766/rt/bin/ninja ]; then
        # ★ 2026-10-31（主控）：共享盘兜底。099 执行器容器里既没有 python3.12 的 ninja、
        #   也没有便携包 runtime → engram 厂商路径**静默退回主机取表**（丢 EngramFetch 两算子）。
        export PATH="/mnt/xds/sfs/l00972766/rt/bin:$PATH"
    elif [ -x /usr/local/python3.12.13/bin/ninja ]; then
        export PATH="/usr/local/python3.12.13/bin:$PATH"
    fi
fi

say() { echo "[env_pack] $*"; }

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    say "⚠️  直接执行不会把环境留在你的 shell 里 —— 请用：source ${BASH_SOURCE[0]}"
fi

# ── ① runtime 根（厂商算子 py 包 / custom_ops / deps / tilelang / 缓存）────────────
if [ -n "$DS41_PACK_RUNTIME" ]; then RT=$DS41_PACK_RUNTIME
elif [ -d "$_pack_root/runtime/python" ]; then RT=$_pack_root/runtime      # 包内
elif [ -d "$SHARE/ds41_runtime/python" ]; then RT=$SHARE/ds41_runtime     # 共享盘回退
else RT=""; fi

# ── ② tilelang / TileKernels（镜像自带的 editable stub 是坏的，必须用可用版本）────
if [ -d "$RT/tilelang/tilelang-merged2/tilelang" ]; then TL_RT=$RT/tilelang          # 包内布局
elif [ -n "$TILELANG_RT" ] && [ -d "$TILELANG_RT/tilelang-merged2/tilelang" ]; then TL_RT=$TILELANG_RT
elif [ -d "$SHARE/rt/tilelang-merged2/tilelang" ]; then TL_RT=$SHARE/rt              # 共享盘
else TL_RT=""; fi

# ── ③ python3.11（容器默认 python3=3.12 装不了这套 torch）───────────────────────
PY=${DS41_PYTHON:-$(command -v python3.11 2>/dev/null || true)}
[ -z "$PY" ] && [ -x /usr/bin/python3.11 ] && PY=/usr/bin/python3.11
if [ -z "$PY" ]; then
    say "❌ 找不到 python3.11（这套 torch/torch_npu 只有 cp311 包）。"
    say "   装法示例：apt-get install -y python3.11 python3.11-dev，或指向镜像自带解释器后重试。"
else
    say "python: $PY ($($PY -V 2>&1))"
fi

# ── ④ CANN：多候选探测，找到哪套用哪套（不判版本）─────────────────────────────
# 「装好可用」的判据：有 aarch64-linux/（老脚本同款判据）或有 libascendcl.so——光有 lib64/ 空目录不算
_cann_ok() {
    [ -n "$1" ] && [ -d "$1" ] && { [ -d "$1/aarch64-linux" ] || [ -f "$1/lib64/libascendcl.so" ]; }
}
CANN=""
if _cann_ok "$DS41_CANN"; then CANN=$DS41_CANN
else
    # 先 overlay 的镜像 CANN（与跑通过的配方一致），再本机自带（版本不限）
    for c in "$RT/ascend_full/usr/local/Ascend/cann-9.2.0" \
             /usr/local/Ascend/ascend-toolkit/latest \
             $(ls -d /usr/local/Ascend/cann-* 2>/dev/null | sort -r); do
        if _cann_ok "$c"; then CANN=$c; break; fi
    done
fi
if [ -n "$CANN" ]; then
    cannv=$(grep -m1 -oE 'version *= *[0-9][^ ]*' "$CANN/version.cfg" 2>/dev/null | head -1)
    say "CANN: $CANN ${cannv:+($cannv)}  ← 不校验版本，只要求算子能跑"
else
    say "⚠️  没找到 CANN（找过 /usr/local/Ascend/*）。算子会 import 失败 —— 见自检输出。"
fi

# ── ⑤ 厂商算子包：扫目录拼 ASCEND_CUSTOM_OPP_PATH（不再写死厂商名字）───────────
_custom_roots="/usr/local/Ascend/custom"
[ -d "$RT/ascend/usr/local/Ascend/custom" ] && _custom_roots="$RT/ascend/usr/local/Ascend/custom:$_custom_roots"
VEND_OPP=""   # ASCEND_CUSTOM_OPP_PATH 用（vendors/<name> 目录）
VEND_LIB=""   # LD_LIBRARY_PATH 用（各 vendor 的 op_api/lib）
_n_vendor=0
for root in ${_custom_roots//:/ }; do
    for v in "$root"/*/vendors/*; do
        if [ -d "$v" ]; then
            VEND_OPP="$VEND_OPP$v:"
            [ -d "$v/op_api/lib" ] && VEND_LIB="$VEND_LIB$v/op_api/lib:"
            _n_vendor=$((_n_vendor + 1))
        fi
    done
done
if [ "$_n_vendor" -gt 0 ]; then say "厂商算子: $_n_vendor 个 vendor 已接入"; else
    say "⚠️  没扫到厂商算子包（看 /usr/local/Ascend/custom/*/vendors/*）——算子会注册失败"
fi

# ── ⑥ 纯 python 依赖（yaml/sympy/requests…；overlay 里没有，PyPI 常不可达）──────
if [ -n "$DS41_DEPS" ]; then DEPS=$DS41_DEPS
elif [ -d "$RT/deps/yaml" ]; then DEPS=$RT/deps                    # 包内实体拷贝
elif [ -d "/root/py311deps/yaml" ]; then DEPS=/root/py311deps      # 本机老路径
elif [ -d "$_pack_root/deps/yaml" ]; then DEPS=$_pack_root/deps
# ★ 2026-11-01：**共享盘**上的公共依赖（099 已装、208 缺；拷贝自 099 的站点包）。
#   位置与三台共享的 `/mnt/xds/sfs` 同盘，只走 PYTHONPATH 前置，不动系统 site-packages。
elif [ -d "/mnt/xds/sfs/l00972766/deps_py/yaml" ]; then DEPS=/mnt/xds/sfs/l00972766/deps_py
else DEPS=""; fi
[ -n "$DEPS" ] && say "deps: $DEPS" || say "⚠️  没有 yaml 等纯 py 依赖 → 请 pip install pyyaml sympy requests（或拷到 /mnt/xds/sfs/l00972766/deps_py）"

# ── ⑦ torch/torch_npu 站点包：镜像自带优先，其次覆盖层（轻量包不打这 2 GB）──────
_py_sp=/usr/local/python3.11.16/lib/python3.11/site-packages     # 950-DT 镜像自带位置
if [ -n "$DS41_TORCH_SITE" ]; then TS=$DS41_TORCH_SITE
elif [ -d "$RT/pysite_full/usr/local/python3.11.16/lib/python3.11/site-packages/torch" ]; then
    TS=$RT/pysite_full/usr/local/python3.11.16/lib/python3.11/site-packages      # 包内/覆盖层
elif [ -d "$_py_sp/torch" ]; then TS=$_py_sp                                     # 镜像自带
elif [ -d "$SHARE/ds41_runtime/pysite_full/usr/local/python3.11.16/lib/python3.11/site-packages/torch" ]; then
    TS=$SHARE/ds41_runtime/pysite_full/usr/local/python3.11.16/lib/python3.11/site-packages   # 共享盘兜底
else TS=""; fi
if [ -n "$TS" ]; then say "torch 站点包: $TS"
else say "⚠️  找不到 torch/torch_npu 站点包（镜像里应有 /usr/local/python3.11.16/lib/python3.11/site-packages）——"; \
     say "   可用 DS41_TORCH_SITE=<site-packages 路径> 指定；自检会在 Tier 0 报缺件"; fi

# ── ⑧ 拼环境 ───────────────────────────────────────────────────────────────
# PYTHONPATH 顺序**照抄验证过的老配方**：deps/tilelang 在前，厂商包（python/）在 shim **之前**
# —— 顺序反了会让 ds41 命名空间注册 0 个算子（cann_ops_transformer_experimental stub 抢先）。
_PP=""
[ -n "$TL_RT" ] && _PP="$_PP$TL_RT/tilelang-merged2:$TL_RT/TileKernels-v2:"
[ -n "$DEPS" ] && _PP="$_PP$DEPS:"
[ -d "$RT/deepgemm-pkg" ] && _PP="$_PP$RT/deepgemm-pkg:"   # 便携包自带 deep_gemm（镜像没装时用）
[ -n "$TS" ] && _PP="$_PP$TS:"
[ -d "$RT/py213" ] && _PP="$_PP$RT/py213:"
[ -d "$RT/python" ] && _PP="$_PP$RT/python:"
# shim（空壳 cann_ops_transformer_experimental）：只在镜像自带 stub 时才需要，放厂商包**之后**即可
for _s in "$RT/shim" "$_pack_root/runtime/shim" "$SHARE/ds41_bench/shim"; do
    [ -d "$_s" ] && { _PP="$_PP$_s:"; break; }
done
[ -n "$CANN" ] && [ -d "$CANN/python/site-packages" ] && _PP="$_PP$CANN/python/site-packages:"
export PYTHONPATH="$_PP${PYTHONPATH:-}"

if [ -n "$CANN" ]; then
    export DS41_CANN=$CANN DS41_CANN_PATH=$CANN
    export ASCEND_HOME_PATH=$CANN ASCEND_TOOLKIT_HOME=$CANN
    export ASCEND_OPP_PATH=$CANN/opp ASCEND_AICPU_PATH=$CANN TOOLCHAIN_HOME=$CANN/toolkit
    export LD_LIBRARY_PATH=$CANN/lib64:$CANN/lib64/plugin/opskernel:$CANN/lib64/plugin/nnengine:\
$CANN/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/Ascend/driver/lib64:\
/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}
fi
export ASCEND_CUSTOM_OPP_PATH="${VEND_OPP}${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${VEND_LIB}${LD_LIBRARY_PATH:-}"
[ -n "$TL_RT" ] && export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-$TL_RT/tilelang-cache} \
                && export CANNBOTDSL_CACHE_DIR=${CANNBOTDSL_CACHE_DIR:-$TL_RT/cannbotdsl-cache}
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-65400}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-24000-24100}
export DS41_RT="$RT" DS41_PACK_ROOT="$_pack_root"
# 大块显存分配（10GiB 级对称缓冲/图池）的碎片：官方建议；run_real.sh 已用同一默认，
# 合成 benchmark 也统一，避免 reserved 很大但连续块不够的假 OOM。
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:4096}"


# ── ⑦b 索引器内核（DeepInfer 口径；2026-10-31 起**默认 dg**）──────────────────
#   DeepInfer 的索引打分 = `deep_gemm.fp8_fp4_mqa_logits` + `deep_select.topk`。
#   mini 用**独立包**（deep_gemm_di / deep_select）与生产 deep_gemm 隔离（pybind 类型会撞）。
#   放在 env_pack 里而不是 run_hit.sh：`tools/run_real.sh` 走的是 launch_mini，
#   原来只有 run_hit 注入 ⇒ 切默认后 run_real 会直接 ModuleNotFound。
#   ★ deep-jit 编译还需要 `/opt/ds-env/.../include/c++` 头文件树（_C.so 里硬编码了
#     `-nostdinc++ -isystem` 到该路径）——缺了就是 command.hpp:35 编译失败；
#     修复脚本：`rt/setup_ds_env_headers.sh`。
export DS41M_IDX_KERNEL=${DS41M_IDX_KERNEL:-dg}
# ★★ 包注入与**内核选择解耦**（2026-10-31 实测事故）：`DS41M_QSLI_KERNEL` 也能独立选 dg，
#   而以前 PYTHONPATH 只在 `IDX_KERNEL=dg` 时注入 ⇒ 设 `IDX_KERNEL=vendor` 后
#   L24 的 qsli 直接 `ModuleNotFoundError: deep_gemm_di`（CP4 隔离实验就是被这个坑掉的）。
#   现在：包存在就注入，与选哪套内核无关。
_dg_pp=""
for _p in "$SHARE/rt/deepgemm-di-pkg" "$SHARE/rt/deepselect-pkg"; do
    [ -d "$_p" ] && _dg_pp="$_dg_pp$_p:"
done
if [ -n "$_dg_pp" ]; then
    export PYTHONPATH="${_dg_pp}${PYTHONPATH:-}"
    # _C.so 用 gcc14 编：系统 libstdc++ 缺 CXXABI_1.3.15 时必须注入共享盘那份
    if ! strings /lib64/libstdc++.so.6 2>/dev/null | grep -q "CXXABI_1.3.15"; then
        for _c in "$SHARE/rt/ds-env-lib" /opt/ds-env/lib; do
            [ -f "$_c/libstdc++.so.6" ] && {
                export LD_PRELOAD="$_c/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"; break; }
        done
    fi
fi
case "$DS41M_IDX_KERNEL" in
  dg)
    if [ -n "$_dg_pp" ]; then
        say "索引器内核: dg（DeepInfer：deep_gemm_di + deep_select；PYTHONPATH 已注入）"
    else
        say "⚠️  DS41M_IDX_KERNEL=dg 但共享盘没有 rt/deepgemm-di-pkg —— 索引器 import 会失败；"
        say "    便携包/无共享盘的机器请显式 export DS41M_IDX_KERNEL=vendor DS41M_QSLI_KERNEL=vendor"
    fi
    ;;
  vendor) say "索引器内核: vendor（融合 quant_lightning_indexer；DS41M_IDX_KERNEL 显式指定）" ;;
  *) say "⚠️  DS41M_IDX_KERNEL=$DS41M_IDX_KERNEL 不认识（只支持 dg|vendor）" ;;
esac

# TMPDIR 必须是本地盘：指到共享盘(SFS)上会让 CANN JIT 收尾的 rmtree 报 "Directory not empty"，
# 表现为编译完成后进程卡住/报错。这里直接 unset（落回 /tmp）。
unset TMPDIR

# ── ⑨ 可选补件（缺了才补，幂等）──────────────────────────────────────────────
# torch_ext/ 里是各扩展的构建目录（comm_context / compressor_v2 / …），要落到 torch 的缓存位置
if [ -d "$RT/torch_ext/compressor_v2" ] && [ ! -d "$HOME/.cache/torch_extensions/py311_cpu" ]; then
    mkdir -p "$HOME/.cache/torch_extensions/py311_cpu"
    cp -r "$RT/torch_ext/." "$HOME/.cache/torch_extensions/py311_cpu/" 2>/dev/null && \
        say "预编译 torch 扩展已放入 ~/.cache/torch_extensions/py311_cpu（免 JIT 重编）"
fi
if [ -d "$RT/py311_include" ] && [ ! -e /usr/include/python3.11/Python.h ]; then
    cp -r "$RT/py311_include/." /usr/include/python3.11/ 2>/dev/null && \
        say "python3.11 头文件已补到 /usr/include/python3.11（JIT 编译需要）"
fi

# ── ⑩ 落一份 env_local.sh（下次直接 source，不依赖本脚本）────────────────────
_env_out="$_pack_root/env_local.sh"
{
    echo "# 由 env_pack.sh 生成（$(date -Is)）—— 可直接 source 复用"
    echo "export DS41_RT='$RT'"
    echo "export DS41_TORCH_SITE='${TS}'"
    echo "export DS41_CANN='${CANN}'"
    echo "export DS41_DEPS='${DEPS}'"
    echo "export TILELANG_RT='$TL_RT'"
    echo "export PYTHONPATH='$PYTHONPATH'"
    echo "export LD_LIBRARY_PATH='$LD_LIBRARY_PATH'"
    echo "export ASCEND_CUSTOM_OPP_PATH='$ASCEND_CUSTOM_OPP_PATH'"
    echo "export ASCEND_HOME_PATH='${ASCEND_HOME_PATH:-}' ASCEND_TOOLKIT_HOME='${ASCEND_TOOLKIT_HOME:-}'"
    echo "export ASCEND_OPP_PATH='${ASCEND_OPP_PATH:-}' ASCEND_AICPU_PATH='${ASCEND_AICPU_PATH:-}'"
    echo "export TILELANG_CACHE_DIR='${TILELANG_CACHE_DIR:-}' CANNBOTDSL_CACHE_DIR='${CANNBOTDSL_CACHE_DIR:-}'"
    echo "export HCCL_IF_BASE_PORT='$HCCL_IF_BASE_PORT' HCCL_NPU_SOCKET_PORT_RANGE='$HCCL_NPU_SOCKET_PORT_RANGE'"
    echo "unset TMPDIR"
} > "$_env_out" 2>/dev/null && say "已写 $_env_out（下次可 source 它）"

say "就绪。下一步（算子自检）："
say "    ${PY:-python3.11} tools/selfcheck.py            # 环境 + 逐算子最小调用"
say "    ${PY:-python3.11} tools/selfcheck.py --full      # 再跑一遍 40 层合成权重（慢）"

# 恢复调用方原来的 nounset 设置（见文件开头）
[ "${_ep_u_on:-0}" = 1 ] && set -u
unset _ep_u_on

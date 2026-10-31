# coding=utf-8
"""DeepInfer Ascend TileLang 压缩 kernel 的 mini 封装（**可选**路径）。

来源：`thirdparty/deepinfer_compress_asc.py`（Apache-2.0，见 thirdparty/NOTICE）。
只在 `DS41M_CP_CMP_KERNEL=deepinfer` 时被 `exec_ops._kv_compress` 的 CP 分支使用：
本地先投影出 (score=Gate, latent=A) fp32 → CP all-gather → 本 kernel 在全序列上
做组内 softmax 加权压缩 + state ring 更新。

★ 为什么不默认开：与厂商融合算子 `ds41.compressor` **不逐位**（实测 bf16 ULP 级：
pre-norm KV max|Δ|=4.9e-4、相对 5.4e-3；对 fp64 参考本 kernel 更准），
开它就失去 CP 与 CP1 的逐位对拍；而 64K 实测 all-gather 只占层耗时 0.36%
（hidden 13.8ms + SWA 37.4ms / 17.3s），换 projected-KV 收益可忽略。
留作 1M 大序列 profiling 的备选（通信 10KB→4KB/token）。
"""
from __future__ import annotations

import importlib.util
import os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 树根（分包后）
_SRC = os.path.join(_HERE, "thirdparty", "deepinfer_compress_asc.py")
_MOD = None
_KERNELS: dict = {}


def _mod():
    global _MOD
    if _MOD is None:
        spec = importlib.util.spec_from_file_location("deepinfer_compress_asc", _SRC)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)          # tilelang/tile_kernels 在本机可用（CANN 9.2）
        _MOD = m
    return _MOD


def kernel(dim: int, ratio: int):
    key = (int(dim), int(ratio))
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = _mod().compress_and_update_state_prefill_tl(
            DIM=int(dim), COMPRESS_RATIO=int(ratio), HAS_INDEX=False)
    return k


def compress_prefill(score, latent, state_cache, out, cu_q, cu_k, cu_c, cbm, sbt,
                     ratio: int):
    """就地写 `out`（[total_q//ratio, DIM] bf16，pre-norm）。

    score/latent: [total_q, DIM] fp32（全局序；CP 下由调用方 all-gather 得到）
    state_cache: [num_slots, ring, 2, DIM] fp32（与厂商路径同一块缓冲）
    其余为 [num_seqs+1]/[num_seqs]/[num_seqs+1]/[total_c] 的 int32 元数据。
    """
    k = kernel(int(score.shape[-1]), int(ratio))
    k(score, latent, state_cache, sbt, cbm, cu_q, cu_k, cu_c, None, out)
    return out

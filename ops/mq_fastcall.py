"""mq_fastcall.py —— mqsmla 的"薄下发"通道（对齐 DeepInfer 的 host 成本）。

# 0. 先纠正一件事（本文件第一版写错的层次）

`torch.ops.cann_ops_transformer.ds41.mixed_quant_sparse_flash_mla` **不是**
`cann_ops_transformer/csrc/attention/mixed_quant_sparse_flash_mla.cpp` 那颗 pybind+aclnn 算子。
`ds41` 子命名空间由
`cann_ops_transformer_experimental/ops/attention/mixed_quant_sparse_flash_mla_dsl/__init__.py`
注册，真正跑的是 **cannbotdsl JIT 算子**：
`ds41_runtime/python/ops/mixed_quant_sparse_flash_mla.py`（2339 行，
`@kernel`/`class MqsmlaKernel` 在 919-920 行），host 侧经过
`cannbotdsl/core/compiler/runtime_bridge.py` 的 provider 桥。
所以 `ACLNN_CMD`/`aclCreateTensor`/`GetWorkspaceSize` 那条成本模型**不适用**（本文件第一版据此写的
ctypes/aclnn 直调分支已删除）。

# 1. 实测事实（全部有出处）

- mqsmla **主调用** = **0.68 ms/次**（`tmp/HOSTAB_T3_ROPETBL_20261026/SUMMARY.txt:9`
  "主调用 0.027s / 40 次"；`HOSTAB_T2_GIDS_20261026` 0.67 ms、`HOSTAB_T1_WINIDX_20261026` 同量级）。
- 我们自己的 Python 胶水只有 0.05 ms/次（同文件 "前置 0.007s | glue 0.002s"，/40）。
- DeepInfer 的 3.4–6.2 µs 是 **ACL 级**（api_statistic），口径不同，不能直接和 0.68 ms 比；
  同口径下我们 ACL 级 16.2 µs/kernel（见 `NOTES/2026-10-26-a4-deepinfer-host.md`）。

# 2. 这 0.68 ms 花在哪（能 CPU 量的都量了）

纯 CPU 实测（`tmp/a1_bench/_a1_val.log`，不碰 NPU）：
    `_validate_inputs`       7.1 µs/次
    `_addr_vec_eligible`     1.3 µs/次（vendor 一调用两次 → 2.7 µs）
    `torch.empty(workspace)` 1.3 µs/次
    torch 派发（同级 schema）14.8 µs/次（`tmp/a1_bench/bench_dispatch.py`）
⇒ 这些加起来 ~26 µs，**解释不了 680 µs**。其余在
`compiled(...)` = cannbotdsl 的 provider 桥：
`ParameterLowering.bind_physical`（`core/compiler/parameter_lowering.py:177`，循环体里访问
`self.abi_arguments[index]`(201 行) 触发 O(n²) 属性重建）+ `Function.__call__`
（`core/compiler/function.py:305`）+ `resolve_provider_call`
（`core/compiler/runtime_bridge.py:347`：preflight / `inspect_storage` / `validate_storage` /
`resolve_stream` / `_export_records`）+ `TorchNPUIntegration.prepare_submit`
（`core/compiler/torch_npu_integration.py:720-737`：`torch.npu.device()` 守卫 +
对 ~17 个张量各调一次 `tensor.record_stream()` + `_npu_getCurrentRawStream`）。
这一段的细分**必须上卡 profile**，本文件不猜。

# 3. 三级通道（默认 L0，行为与本文件引入前逐字节相同）

  DS41M_MQ_FAST=0（默认）  L0  torch.ops.cann_ops_transformer.ds41.mixed_quant_sparse_flash_mla(q, **kw)
  DS41M_MQ_FAST=1          L1  直接调 ds41 的 PrivateUse1 impl 函数（跳过 torch 派发）
                               预期 −15 µs/次 ⇒ 40 层 −0.6 ms/forward，零数值风险
                               （下行代码与 L0 完全相同，只是不经过 dispatcher）
  DS41M_MQ_FAST=2          L2  进一步跳过 impl 里的 `_validate_inputs` / out,lse 复查 /
                               `construct_workspace` 的 `torch.npu.get_device_properties`
                               （`torch_npu/npu/__init__.py:542-548` 每次调用都进 C++ 查一次）。
                               复刻 vendor 尾部 `ops/mixed_quant_sparse_flash_mla.py:2246-2265`。
                               预期再 −10~20 µs/次；**代价是丢掉入参自检**，只在自测臂里开。

# 4. 诊断臂 DS41M_MQ_NOPREP=1（这条才是可能的大头）

把 `TorchNPUIntegration.prepare_submit` 换成 no-op，用一次 A/B 就能判定
"`record_stream`×17 + device 守卫 + raw-stream 取句柄" 到底吃掉多少。
安全性：只在**长生命周期张量**（我们的池/索引/元数据都是常驻）下等价；
显存复用激进的场景不要开。

纪律：只读 vendor 已导出的名字，不改 vendor；main 树不受影响。
"""
from __future__ import annotations

import math
import os
import sys

import torch

# ─────────────────────────── level ───────────────────────────
_LVL = None


def level() -> int:
    global _LVL
    if _LVL is None:
        v = os.environ.get("DS41M_MQ_FAST", "0").strip()
        _LVL = int(v) if v in ("0", "1", "2") else 0
    return _LVL


_STATS = {"l0": 0, "l1": 0, "l2": 0, "fallback": 0, "noprep": 0}


def stats():
    return dict(_STATS, level=level())


# ─────────────────────────── vendor 入口定位 ───────────────────────────
# 不能按点号 import（`cann_ops_transformer/ops/attention/` 没有 __init__.py），
# 按 __file__ 后缀在 sys.modules 里找，找到后缓存。
_DSL = None          # (attn_fn, meta_fn)  ← .../mixed_quant_sparse_flash_mla_dsl/mixed_quant_sparse_flash_mla.py
_TAIL = None         # module             ← ds41_runtime/python/ops/mixed_quant_sparse_flash_mla.py


def _scan(suffix):
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None) or ""
        if f.endswith(suffix):
            return m
    return None


def _dsl():
    global _DSL
    if _DSL is None:
        m = _scan("/mixed_quant_sparse_flash_mla_dsl/mixed_quant_sparse_flash_mla.py")
        if m is None:
            return None
        attn = getattr(m, "mixed_quant_sparse_flash_mla", None)
        meta = getattr(m, "mixed_quant_sparse_flash_mla_metadata", None)
        if not callable(attn) or not callable(meta):
            return None
        _DSL = (attn, meta)
    return _DSL


def _tail():
    global _TAIL
    if _TAIL is None:
        m = _scan("/python/ops/mixed_quant_sparse_flash_mla.py")
        if m is None:
            return None
        for name in ("_validate_inputs", "_addr_vec_eligible", "_addr_tab_w",
                     "construct_workspace", "_get_compiled_kernel",
                     "_batch_consistency_enabled"):
            if not hasattr(m, name):
                return None
        _TAIL = m
    return _TAIL


# ─────────────────────────── NOPREP 诊断臂 ───────────────────────────
_NOPREP_DONE = False


def install_noprep() -> bool:
    """把 cannbotdsl 的 prepare_submit 换成 no-op（仅诊断用）。幂等、失败静默。"""
    global _NOPREP_DONE
    if _NOPREP_DONE:
        return True
    try:
        from cannbotdsl.core.compiler import torch_npu_integration as _ti

        if getattr(_ti.TorchNPUIntegration.prepare_submit, "_a1_noprep", False):
            _NOPREP_DONE = True
            return True

        def _noop(self, tensors, *, device_id: int, handle: int, owner) -> None:  # noqa: ANN001
            return None

        _noop._a1_noprep = True
        _ti.TorchNPUIntegration.prepare_submit = _noop
        _NOPREP_DONE = True
    except Exception:                                     # noqa: BLE001
        return False
    return True


def _maybe_noprep():
    if os.environ.get("DS41M_MQ_NOPREP", "").strip().lower() in ("1", "on", "true", "yes"):
        if install_noprep():
            _STATS["noprep"] += 1


# ─────────────────────────── L0 ───────────────────────────
def _l0(q, kw):
    d = torch.ops.cann_ops_transformer.ds41
    return d.mixed_quant_sparse_flash_mla(q, **kw)


# ─────────────────────────── L1 ───────────────────────────
def _l1(q, kw):
    fn = _dsl()[0]
    return fn(q, **kw)


# ─────────────────────────── L2（复刻 vendor 尾部）───────────────────────────
def _l2(q, kw):
    V = _tail()
    ori_kv = kw.get("ori_kv")
    idx = kw.get("ori_sparse_indices")
    bt = kw.get("ori_block_table")
    ln = kw.get("ori_topk_length")
    cmp_kv = kw.get("cmp_kv")
    ci = kw.get("cmp_sparse_indices")
    cbt = kw.get("cmp_block_table")
    cl = kw.get("cmp_topk_length")
    # 与 vendor impl 一致：out 每次新分配（要作为本层输出被后续算子消费），lse 空。
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((0,), dtype=torch.float32, device=q.device)
    addr_ws = None
    addr_bytes = 0
    if V._addr_vec_eligible(ori_kv, bt, idx, cmp_kv, cbt, ci):
        cols = V._addr_tab_w(idx.shape[2])
        if cmp_kv is not None:
            cols += V._addr_tab_w(ci.shape[2])
        addr_bytes = q.shape[0] * cols * 4
    ws_full = V.construct_workspace(kw["metadata"], addr_bytes)
    if addr_bytes:
        workspace = ws_full[:-addr_bytes]
        addr_ws = ws_full[-addr_bytes:]
    else:
        workspace = ws_full
    compiled = V._get_compiled_kernel(ori_kv, idx, bt, ln, cmp_kv, ci, cbt, cl, None)
    tensors = (q, ori_kv, idx, bt, ln, out, kw["sinks"], kw["cu_seqlens_q"],
               kw["metadata"], workspace, cmp_kv, ci, cbt, cl)
    s = kw.get("softmax_scale")
    scale = float(s) if s is not None else 1.0 / math.sqrt(q.shape[-1])
    bc = 1 if V._batch_consistency_enabled() else 0
    compiled(*(t for t in tensors if t is not None), scale, bc,
             *([addr_ws] if addr_ws is not None else []))
    return out, lse


# ─────────────────────────── 公共入口 ───────────────────────────
def mqsmla(q, kw):
    """mqsmla 主调用。`kw` 是 exec_ops 组好的 kwargs dict（与 torch.ops 同形）。"""
    lv = level()
    if lv:
        _maybe_noprep()
    if lv >= 2:
        try:
            r = _l2(q, kw)
            _STATS["l2"] += 1
            return r
        except Exception as e:                            # noqa: BLE001
            _STATS["fallback"] += 1
            print(f"  [mq-fast] L2 失败回退 L1: {type(e).__name__}: {e}", flush=True)
            lv = 1
    if lv == 1:
        try:
            r = _l1(q, kw)
            _STATS["l1"] += 1
            return r
        except Exception as e:                            # noqa: BLE001
            _STATS["fallback"] += 1
            print(f"  [mq-fast] L1 失败回退 L0: {type(e).__name__}: {e}", flush=True)
    _STATS["l0"] += 1
    return _l0(q, kw)


def mqsmla_metadata(*args, **kw):
    """metadata 生成（exec_ops 里已按 ctx 做值缓存，稳态 0 次/forward，这里只是顺手剥派发）。"""
    if level():
        _maybe_noprep()
        fn = _dsl()[1]
        return fn(*args, **kw)
    d = torch.ops.cann_ops_transformer.ds41
    return d.mixed_quant_sparse_flash_mla_metadata(*args, **kw)

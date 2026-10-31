"""mq_outbuf —— mqsmla 输出/workspace 缓冲复用（2026-10-29，原型默认关）。

背景：vendor DSL impl 每次 `torch.empty` 分配 `attn_out`/`softmax_lse`，runtime tail 每次
`construct_workspace()` 再分配 FD workspace ⇒ `ProviderCallable` 的实参地址每调用都变，
vendor `_benchmark_bind` 的绑定无法跨调用复用（`bind_fast` 的前提）。

做法（**确定性挂载**）：vendor 模块由 torch.ops 的自定义加载器加载，且 tail 模块是
在 DSL 入口**函数体内**才 import 的。所以这里包装 `mq_fastcall2._impl`：
  · `_impl()` 第一次成功返回 `(vendor_fn, meta)` 时（此刻 DSL 模块必然已在 sys.modules），
    立刻把 DSL 模块的入口换成"输出/workspace 从常驻缓冲取"的实现；
  · tail 函数在**调用期**惰性 import（与 vendor 自己同一写法），并顺手把它的
    `construct_workspace` 也换成按 `(addr_bytes, device)` 常驻；
  · 任何一步失败 → 直接调原入口（不改变厂商行为）。
**不修改共享盘上的 vendor 文件。**

安全性：mini 主流水线单流顺序执行；第 N 层消费者先于第 N+1 层写入入队，复用安全。
数值由 8k/16k/256k final-row 对拍验证（NOTES/2026-10-29-bind-fast-persistent-buffers.md）。

开关：`DS41M_MQ_OUTBUF=1`；`DS41M_MQ_OUTBUF_STATS=1` 退出打印缓冲统计。
"""
from __future__ import annotations

import atexit
import importlib
import os
import sys

_ON = os.environ.get("DS41M_MQ_OUTBUF", "1").strip().lower() in ("1", "on", "true", "yes")
_STATS_ON = os.environ.get("DS41M_MQ_OUTBUF_STATS", "").strip().lower() in (
    "1", "on", "true", "yes")
_BUFS: dict = {}
_WSBUFS: dict = {}
_TAIL = [None]
_PATCHED = [False]


def install():
    if not _ON or _PATCHED[0]:
        return
    if _STATS_ON:
        atexit.register(_dump)
    try:
        import mq_fastcall2
    except Exception:                                    # noqa: BLE001
        return
    _orig_impl = getattr(mq_fastcall2, "_impl", None)
    if _orig_impl is None or getattr(_orig_impl, "_outbuf_wrap", False):
        return

    def _impl():
        r = _orig_impl()
        if r is not None and not _PATCHED[0]:
            try:
                _patch(r[0])
            except Exception as e:                       # noqa: BLE001 绝不改变厂商路径
                print(f"  [mq-outbuf] 挂载失败: {type(e).__name__}: {e}", flush=True)
        return r

    _impl._outbuf_wrap = True
    mq_fastcall2._impl = _impl


def _tail():
    if _TAIL[0] is None:
        mod = importlib.import_module("ops.mixed_quant_sparse_flash_mla")
        _patch_ws(mod)
        _TAIL[0] = getattr(mod, "mixed_quant_sparse_flash_mla")
    return _TAIL[0]


def _patch_ws(mod):
    orig_ws = getattr(mod, "construct_workspace", None)
    if orig_ws is None or getattr(orig_ws, "_wsbuf", False):
        return

    def construct_workspace(metadata, addr_bytes=0):
        key = (int(addr_bytes), str(metadata.device))
        b = _WSBUFS.get(key)
        if b is None:
            b = orig_ws(metadata, addr_bytes)
            _WSBUFS[key] = b
        return b

    construct_workspace._wsbuf = True
    mod.construct_workspace = construct_workspace


def _patch(orig_fn):
    _PATCHED[0] = True
    if getattr(orig_fn, "_outbuf", False):
        return
    mod = sys.modules.get(getattr(orig_fn, "__module__", ""))
    if mod is None:
        return
    import torch

    def _buf(kind, shape, dtype, device):
        key = (kind, tuple(shape), dtype, str(device))
        b = _BUFS.get(key)
        if b is None:
            b = torch.empty(shape, dtype=dtype, device=device)
            _BUFS[key] = b
        return b

    def mixed_quant_sparse_flash_mla(
        q,
        *,
        ori_kv=None,
        cmp_kv=None,
        ori_sparse_indices=None,
        cmp_sparse_indices=None,
        ori_block_table=None,
        cmp_block_table=None,
        cu_seqlens_q=None,
        seqused_q=None,
        seqused_ori_kv=None,
        seqused_cmp_kv=None,
        ori_topk_length=None,
        cmp_topk_length=None,
        sinks=None,
        metadata=None,
        quant_mode,
        softmax_scale=None,
        layout_q="TND",
        layout_kv="PA_BBND",
        return_softmax_lse=False,
    ):
        try:
            tail = _tail()
        except Exception:                                # noqa: BLE001
            return orig_fn(
                q, ori_kv=ori_kv, cmp_kv=cmp_kv,
                ori_sparse_indices=ori_sparse_indices,
                cmp_sparse_indices=cmp_sparse_indices,
                ori_block_table=ori_block_table,
                cmp_block_table=cmp_block_table,
                cu_seqlens_q=cu_seqlens_q, seqused_q=seqused_q,
                seqused_ori_kv=seqused_ori_kv, seqused_cmp_kv=seqused_cmp_kv,
                ori_topk_length=ori_topk_length,
                cmp_topk_length=cmp_topk_length, sinks=sinks,
                metadata=metadata, quant_mode=quant_mode,
                softmax_scale=softmax_scale, layout_q=layout_q,
                layout_kv=layout_kv,
                return_softmax_lse=return_softmax_lse)
        attn_out = _buf("out", q.shape, torch.bfloat16, q.device)
        if return_softmax_lse:
            n2 = ori_kv.shape[2]
            if layout_q == "TND":
                lse_shape = (n2, q.shape[0], q.shape[1] // n2)
            else:
                lse_shape = (q.shape[0], n2, q.shape[1], q.shape[2] // n2)
        else:
            lse_shape = (0,)
        softmax_lse = _buf("lse", lse_shape, torch.float32, q.device)
        tail(
            q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            ori_sparse_indices=ori_sparse_indices,
            cmp_sparse_indices=cmp_sparse_indices,
            ori_block_table=ori_block_table,
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            seqused_ori_kv=seqused_ori_kv,
            seqused_cmp_kv=seqused_cmp_kv,
            ori_topk_length=ori_topk_length,
            cmp_topk_length=cmp_topk_length,
            sinks=sinks,
            metadata=metadata,
            quant_mode=quant_mode,
            softmax_scale=softmax_scale,
            layout_q=layout_q,
            layout_kv=layout_kv,
            return_softmax_lse=return_softmax_lse,
            out=attn_out,
            lse=softmax_lse,
        )
        return attn_out, softmax_lse

    mixed_quant_sparse_flash_mla.__name__ = orig_fn.__name__
    mixed_quant_sparse_flash_mla.__module__ = orig_fn.__module__
    mixed_quant_sparse_flash_mla._outbuf = True
    mod.mixed_quant_sparse_flash_mla = mixed_quant_sparse_flash_mla
    try:
        import mq_fastcall2
        if (getattr(mq_fastcall2, "_IMPL_STATE", "") == "found"
                and getattr(mq_fastcall2, "_IMPL", None)):
            mq_fastcall2._IMPL = (mixed_quant_sparse_flash_mla,
                                  mq_fastcall2._IMPL[1])
    except Exception:                                    # noqa: BLE001
        pass
    print("  [mq-outbuf] 已挂载（确定性包装 _impl）", flush=True)


def _dump():
    if not _BUFS:
        return
    tot = sum(int(b.numel()) * b.element_size() for b in _BUFS.values())
    wst = sum(int(b.numel()) * b.element_size() for b in _WSBUFS.values())
    print(f"  [mq-outbuf] out/lse={len(_BUFS)} ({tot / 2**20:.1f}MiB) "
          f"ws={len(_WSBUFS)} ({wst / 2**20:.1f}MiB)", flush=True)

# ── 点位边界（core/memlife.py）：out/lse 与 workspace 按形状缓存，**随点位几何变** ──
# 这是跨点位累积的典型来源：以前没有任何地方清它，每个新 (T,H,heads) 都新增一块常驻。
POINT_CACHES = (_BUFS, _WSBUFS)

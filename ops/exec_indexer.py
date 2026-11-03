"""exec_indexer.py —— 索引器三件套（`idx_prolog` / `qli`，`qsli` 留下一轮）。

插件接缝见 exec_ops.py 的 `_load_plugins` docstring：本文件只被 exec_ops 在收尾时
`register(OPS, SKIP, core)` 一次，`core` = exec_ops 模块本身（借用 `_hold` / `_cache_buf`）。

算子口径（官方调用点，逐字对齐）：
  ① idx_prolog → `torch.ops.cann_ops_transformer.ds41.indexer_prologue_qw`
       models/modules/op_impls/indexer_prolog.py:46-73（ascendc 融合版）
       吃 x[T,H] + qr=量化后的 q_norm(wq_a(x))，出 q[T,32,64] / descale_q[T,32,2,2] / w[T,32]
  ② qli → `quant_lightning_indexer`（+ 其前置 `quant_lightning_indexer_metadata`）
       models/modules/indexer.py:331-393
  ③ qsli → `quant_sparse_lightning_indexer`（**下一轮**：要先有 candidate 块）

与 exec_ops 里 mqsmla 的关系：mqsmla 现在的 cmp 索引是"全 -1 + topk_length=0"的空索引
（官方支持的退化配置）。`qli` 把真索引按**约定**写进激活表，mqsmla 那侧有就用真的：
  · `st["cmp_sparse_indices"]` int32 [T,1,512] —— QLI 的 sparse_indices（-1 = 无效列）
  · `st["cmp_topk_length"]`    int32 [T,1]     —— 每行有效长度 = (sparse_indices >= 0).sum(-1)
（另留 `st["idx_q"] / st["idx_descale_q"] / st["idx_w"] / st["idx_sparse"]` 备用。）

硬约束（均为算子 host 侧自己报的，实测踩过）：
  · wqb/ww 必须是 FRACTAL_NZ（否则 dma_gm2l1 identical layout map kinds 的裸报错）
  · descale_qr/descale_wqb 末维按 2 配对：qr 20 对 = 1280/64，wq_b 4096×20×2
  · K 池内轴 stride 必须恰为 (64,64,1)，descale_k 为 (4,4,2,1)（只允许 axis-0 有 padding）
  · seqused_k 是**压缩后**的 K 长度（T/ratio），mask_mode=3 且 cmp_ratio≠1 时必须给 cmp_residual_k
"""
from __future__ import annotations

import os
import time
import sys

import torch
import batch as batch_layout
import torch_npu
# A6：索引器三族厂商算子的「薄下发」通道（DS41M_IDX_FAST=0|1|2，默认 0 = 原行为）
import idx_fastcall as _ifc

# ★ 默认值 = 改造前的硬编码值；启动时由 model_spec 刷新（见 `_apply_spec`）。
N_HEADS = 32             # index_n_heads（≠ 主注意力的 64）
HEAD_DIM = 128           # index_head_dim（≠ 主注意力的 512）
ROPE_DIM = 64            # qk_rope_head_dim（同时是 rope 表宽度）
PACKED_D = HEAD_DIM // 2                 # 64：MXFP4 两值一字节
SCALE_PAIR = 64                          # 每 64 个 K 元素一组配对 scale
NORM_EPS = 1e-20                         # rms_norm_eps（model_spec）
ROWS_PER_BLOCK = 128                     # 压缩行块（= model_spec.block_size）
CAND_BLOCK = 8                           # 候选块宽（model_spec.candidate_block_size）
CAND_TOPK = 2048                         # 候选块数（model_spec.candidate_topk_blocks）
SOFTMAX_SCALE = HEAD_DIM ** -0.5 * N_HEADS ** -0.5      # = 0.015625（官方同式）
FRACTAL_NZ = 29

# ★ 2026-10-28（P0-3）：QSLI metadata 进程内复用（同一 forward 里 4 个索引源层的
#   T/off/ratio 完全相同 → 只构建一次）。`DS41M_QS_MD_CACHE=0` 回退到每次重算（A/B 归因）。
_QS_MD_CACHE = os.environ.get("DS41M_QS_MD_CACHE", "1").strip().lower() in (
    "1", "on", "true", "yes")

# ★ 2026-10-28（P0-5，**已否决**）：CP 的 QSLI 双块合并（两块当 2 个请求一次调用）
#   在 8k@0.90 CP2 上**设备挂死**（rank0 无输出、rc=124；共享块表 repeat + batch=2 姿势）。
#   代码已删，勿复用；结论见 NOTES/2026-10-28-tail-bubble-host-opt-plan.md。


def _apply_spec(cfg=None):
    """按 model_spec 刷新模块常量（register 时一次；运行期零开销）。"""
    import model_spec as _ms
    s = _ms.set_spec(cfg, strict=False) if cfg is not None else _ms.get()
    global N_HEADS, HEAD_DIM, ROPE_DIM, PACKED_D, SCALE_PAIR, SOFTMAX_SCALE
    global NORM_EPS, ROWS_PER_BLOCK, CAND_BLOCK, CAND_TOPK
    N_HEADS, HEAD_DIM, ROPE_DIM = int(s.index_n_heads), int(s.index_head_dim), int(s.qk_rope_head_dim)
    PACKED_D = HEAD_DIM // 2
    SCALE_PAIR = int(s.scale_pair)
    NORM_EPS = float(s.rms_norm_eps)
    ROWS_PER_BLOCK = int(s.block_size)
    CAND_BLOCK, CAND_TOPK = int(s.candidate_block_size), int(s.candidate_topk_blocks)
    SOFTMAX_SCALE = HEAD_DIM ** -0.5 * N_HEADS ** -0.5
    return s

_CORE = None             # register 时注入（exec_ops 模块）
ADDED = "idx_prolog+qli"  # exec_ops 插件加载时打印用（qsli 留下一轮）


# ─────────────────────────── 小工具 ───────────────────────────
# 走 NZ 布局的 indexer 权重：短名 → 权重全名。
# 调用侧只写 `_nz("wqb")`，不用再抄长路径。
_NZ_ALIAS = {
    "wqb": "attn.indexer.wq_b.weight",          # [4096,1280] fp8
    "wk":  "attn.indexer.wk.weight",
    "ww":  "attn.indexer.weights_proj.weight",  # [32,5120] bf16
}

# 权重 NZ 布局缓存：按**权重名**索引（不再绑 data_ptr），只持有转换产物。
_NZ_BY_NAME: dict = {}

_NZ_WEIGHTS = tuple(_NZ_ALIAS.values())
_IDXK_LOG: list = []          # 调试（DS41M_QLI_DUMP）：所有 K 写者的记录（模块级，跨 st/hold）


def _qli_dbg(msg: str) -> None:
    """分支级打点（DS41M_QLI_BRANCH_DBG）：只打 rank0，定位 A/B 走了哪条 K 路径。

    ★ 同时落 `<DS41M_QLI_DUMP>/branch.log` —— pure-log 模式下 stdout 会被
      launch_mini 的汇总过滤器吞掉（2026-10-30 实测 A/B 打点 0 行就是这原因）。
    """
    if not os.environ.get("DS41M_QLI_BRANCH_DBG"):
        return
    try:
        import torch.distributed as _d
        if _d.is_initialized() and _d.get_rank() != 0:
            return
    except Exception:                     # noqa: BLE001
        pass
    line = f"[qli-dbg] [mod={__name__} log={id(_IDXK_LOG):#x}] {msg}"
    print(line, flush=True)
    _dd = os.environ.get("DS41M_QLI_DUMP", "").strip()
    if _dd:
        try:
            os.makedirs(_dd, exist_ok=True)
            with open(os.path.join(_dd, "branch.log"), "a") as _fh:
                _fh.write(line + "\n")
        except Exception:                 # noqa: BLE001
            pass


def _cand_dbg_check(ctx, ci, cl, topk, cand_size, ratio):
    """DS41M_CAND_DBG：候选池语义核对（对照 DeepInfer `candidate_blocks.py`）。

    ① 块号是否**升序**（sparse metadata 内核硬要求：sorted/unique/in-range）；
    ② `cl` 是否等于 `get_candidate_compact_lens_ref`：选中的块里有多少个有效 token
       = Σ clamp(valid_len − block*block_size, 0, block_size)，无效块计 0。
    两处产候选的地方（`_qli_chunk` 与 `_qli_pk`）共用。
    """
    try:
        # ★ ci 是 [T,1,K]，先压成 [T,K]（否则下面的广播会报 2048 vs 512）
        _b = ci.detach().reshape(ci.shape[0], -1).to(torch.int64)
        _neg = _b < 0
        if _b.shape[0] > 1:
            _mono = (_b[1:] >= _b[:-1]) | _neg[1:] | _neg[:-1]
            _srt = bool(_mono.all().item())
        else:
            _srt = True
        _pb = _b.clamp_min(0)
        _offK = int(ctx.get("pos_offset", 0) or 0)
        _rows = torch.arange(_b.shape[0], device=_b.device, dtype=torch.int64)
        _valid = ((_offK + _rows + 1) // max(1, ratio)).clamp_max(int(topk))
        _tok = (_valid.unsqueeze(1) - _pb * int(cand_size)).clamp_(0, int(cand_size))
        _tok = torch.where(_neg, torch.zeros_like(_tok), _tok)
        _lens_ref = _tok.sum(1, dtype=torch.int32)
        _cl_l = cl.detach().reshape(-1).to(torch.int32)
        _neq = int((_lens_ref != _cl_l).sum())
        print(f"  [cand-dbg] L{int(ctx.get('idx'))} ratio={ratio} "
              f"blocks={tuple(ci.shape)} sorted={_srt} neg={int(_neg.sum())} "
              f"cl_vs_ref_neq={_neq}/{int(_cl_l.numel())} "
              f"cl_max={int(_cl_l.max())} ref_max={int(_lens_ref.max())}", flush=True)
    except Exception as _e:               # noqa: BLE001
        print(f"  [cand-dbg] 失败 {type(_e).__name__}: {_e}", flush=True)


def _nz(name: str):
    """取已预转的 NZ 权重。`name` 可用短名（wqb/wk/ww）或完整权重名。"""
    name = _NZ_ALIAS.get(name, name)
    nz = _NZ_BY_NAME.get(name)
    if nz is None:
        raise KeyError(f"权重 {name} 尚未预转 NZ（加载期应先调用 preload_nz）")
    return nz


def preload_nz(w) -> int:
    """加载期把所有走 NZ 的 indexer 权重转一次，返回条数。

    转换是这里唯一的职责；之后 `_nz(name)` 只查缓存，完全不依赖 raw ND 张量，
    于是 raw 可以被 `_release_packed_raw()` 释放。
    """
    n = 0
    for _name in _NZ_WEIGHTS:
        t = w.get(_name)
        if torch.is_tensor(t) and _name not in _NZ_BY_NAME:
            _NZ_BY_NAME[_name] = torch_npu.npu_format_cast(t.contiguous(), FRACTAL_NZ)
            n += 1
    return n


_CS_CACHE: dict = {}


def _rope_comp(T, ctx, device, start=0):
    """压缩分支的 rope 表 [T,64] fp32（theta=compress_rope_theta=160000，interleave）。

    官方传的是 attn_metadata["cos_sin"]["comp"].reshape(-1, 64)，即 RotaryEmbedding 的输出。
    只跟 T、start 有关 → 缓存复用，避免每层重算进 trace；
    **命中时 start = L-T_main**（块绝对位置，索引器 q 与压缩器共用同一张表）。
    """
    key = (T, str(device), int(start))
    if _pk(ctx) is not None:
        # packed：每 token 的 rope 位置 = 序列内局部位置（官方 prefill 的 position_ids 口径）
        # ★ CP：位置来自 `cp_positions`（head-tail 非连续）；同一个本地 T 可对应不同 L，
        #   缓存键必须带 `cp_total_q`，否则跨点位会复用错表。
        # ★★ 2026-10-27 修（多点位数值错）：旧键 ("pk", T, dev[, cp_total_q]) **不含位置几何**
        #   —— 同一进程跑多个点位时，尾段 T 恒为 140、pos_offset 却不同 ⇒ 第二个点位起
        #   复用上一点的 rope 表。实测：64k@0.97 单点 digest 08443723284dd9ed，作为第二个
        #   点位跑出 d0400a2e304cbb07。现在键里带 pos 对象身份（`local_positions` 已按
        #   几何 cu/T/off/offs 缓存，几何相同才返回同一对象），值也存 pos 引用做身份校验
        #   （防地址复用误命中）。
        pos, _ = _CORE._packed_pos(ctx, int(T), device)
        _pid = int(pos.data_ptr()) if pos is not None else 0
        _cpk = ("pk", int(T), str(device), _pid,
                int(ctx.get("cp_total_q") or 0) if ctx.get("cp_positions") is not None else 0)
        hit = _CS_CACHE.get(_cpk)
        if hit is not None and hit[0] is pos:
            return hit[1], hit[2]
        _nn = (int(ctx.get("cp_total_q") or 0)
               if ctx.get("cp_positions") is not None else 0)
        cos_t, sin_t = _CORE._rope_at_pos(pos, ctx, device, n=(_nn or None))
        _sin_r = sin_t.reshape(-1, ROPE_DIM).contiguous()
        _cos_r = cos_t.reshape(-1, ROPE_DIM).contiguous()
        _CS_CACHE[_cpk] = (pos, _sin_r, _cos_r)
        return _sin_r, _cos_r
    if _CORE.dg_on():
        # decode 图：`_comp_rope_tables` 返回固定缓冲 → reshape 只是同存储的视图，
        # 图里地址不变；每步 dg_update 原地换内容。key 不带上 start（缓冲与 pos 无关）。
        _dk = ("dg", int(T), str(device))
        hit = _CS_CACHE.get(_dk)
        if hit is None:
            cos_t, sin_t = _CORE._comp_rope_tables(T, 1, ctx, device, row_off=int(start))
            hit = _CS_CACHE[_dk] = (sin_t.reshape(-1, ROPE_DIM),
                                    cos_t.reshape(-1, ROPE_DIM))
        return hit
    hit = _CS_CACHE.get(key)
    if hit is not None:
        return hit
    # ★ 官方索引器 q 用的是 **同一张 comp 表**（`attn_metadata["cos_sin"]["comp"]`），
    #   与压缩器那张 `_comp_rope_tables` 同源：**交错排列 + YaRN(factor16/beta32,1/orig65536)**。
    #   曾经这里自建了 "cat(freqs,freqs) 半半 + 无 YaRN" 的表 ✗ —— 那是 _rope_cos_sin 文档里
    #   明确标记过的错误形态，会让 q 的 rope 整体偏掉 → 分数边界处翻元素（token 1023 那一个）。
    cos_t, sin_t = _CORE._comp_rope_tables(T, 1, ctx, device, row_off=int(start))   # 位置 = p×1
    hit = _CS_CACHE[key] = (sin_t.reshape(-1, ROPE_DIM).contiguous(),
                            cos_t.reshape(-1, ROPE_DIM).contiguous())
    return hit


def _descale_pairs(s, rows, n_pair):
    """E8M0 scale → uint8 视图 + 末维按 2 配对（官方 `view(uint8).view(rows, n, 2)`）。"""
    if s.dtype != torch.uint8:
        s = s.view(torch.uint8)
    return s.reshape(rows, n_pair, 2).contiguous()


def _idx_prolog(st, ctx, w):
    """融合 Q 前段：qr 量化 + wq_b + 部分 rope + MXFP4 量化 + weights_proj。

    官方 op_impls/indexer_prolog.py:46（indexer_prolog_qw_ascendc）→
      models/modules/indexer.py:238 `OpKernel.indexer_prolog_qw(self, x, qr, cos, sin)`
    在这里拆成：wq_a+q_norm 已由配方上游算子算好（st["qr"]），本算子从 qr 起。
    """
    d = torch.ops.cann_ops_transformer.ds41
    x = st["x"]
    if x.dim() == 3:                                   # hc_post 后的 [T,hc_mult,H]（保险）
        x = x.reshape(x.shape[0], -1)
    qr = st.get("qr")
    if qr is None:
        raise RuntimeError("idx_prolog 需要 st['qr']：请确认本层先跑了 wq_a + q_norm")
    T = qr.shape[0]
    q_lora = qr.shape[-1]
    # ★ CP：`indexer_prologue_qw` 的 **w 输出按 M 分块**（q/descale_q 不敏感，
    #   NOTES/2026-10-22-qmm-shape-sensitivity.md）：M≤2048 与全长尾部不同，
    #   补到 4096/8192 才能匹配 ≥12288 的参考。做法同 `_cp_ref_pad`：本地行放
    #   参考 M 缓冲的**尾部**、前端零补；rope 表按"前 ref-T 行位置 0 + 本地全局位置"
    #   现造（本地行是 head-tail，位置不连续），算完取尾部。
    _padT = 0
    _sc = None
    _cp_i = _CORE._cp_of(ctx)
    if _cp_i is not None:
        _seq_i = int(getattr(_cp_i, "seq_len", 0) or _cp_i.plan.total_q)
        # 参考 M 与 CP1 非 CP 拆分一致：**当前计划全局行数**（compact 段 = 512/140）
        _ref_i = min(8192, int(_cp_i.plan.total_q))
        if int(T) < _ref_i:
            x, _padT = _CORE._cp_ref_pad(x, ctx, min_pad=4096)
            if _padT:
                qr, _ = _CORE._cp_ref_pad(qr, ctx, min_pad=4096)
                # compact 段：位置是绝对位置（局部 0..W-1 + pos_offset）
                _cpos = _cp_i.abs_positions.to(torch.long).reshape(-1)
                _pos = torch.cat([torch.zeros(_ref_i - int(T), dtype=torch.long,
                                              device=qr.device), _cpos])
                # 表宽按**原序列长**开（绝对位置最大到 seq_len-1）
                _cos_t, _sin_t = _CORE._rope_at_pos(
                    _pos, ctx, qr.device, n=_seq_i)
                _sc = (_sin_t.reshape(-1, ROPE_DIM).contiguous(),
                       _cos_t.reshape(-1, ROPE_DIM).contiguous())
                T = _ref_i

    # qr：官方 npu_dynamic_mx_quant(qr.view(-1, q_lora), dst_type=float8_e4m3fn)
    qr8, qr_scale = torch_npu.npu_dynamic_mx_quant(
        qr.reshape(-1, q_lora), dst_type=torch.float8_e4m3fn)
    n_pair = (q_lora + SCALE_PAIR - 1) // SCALE_PAIR                 # 20
    qr_u8 = qr8.view(torch.uint8)
    descale_qr = _descale_pairs(qr_scale, T, n_pair)

    wqb = _nz("wqb")                          # [4096,1280] 1B + NZ
    ww = _nz("ww")                   # [32,5120] BF16 + NZ
    _pk = w.get("_packed") or {}
    descale_wqb = _pk.get("attn.indexer.wq_b.scale.pairs")
    if descale_wqb is None:                     # 权重侧配对：加载期备好（以前每次现做）
        descale_wqb = _descale_pairs(w["attn.indexer.wq_b.scale"], N_HEADS * HEAD_DIM, n_pair)
    if _sc is not None:
        sin, cos = _sc
    else:
        sin, cos = _rope_comp(T, ctx, qr.device, int(ctx.get("pos_offset", 0)))

    # A6 薄下发：L0 = 原样 d.indexer_prologue_qw(...)；L1/L2 见 ops/idx_fastcall.py
    q, descale_q, wi = _ifc.prolog_qw(
        (x, qr_u8, wqb, ww, descale_qr, descale_wqb, sin, cos),
        {"softmax_scale": SOFTMAX_SCALE})
    if _padT:
        q, descale_q, wi = q[-_padT:], descale_q[-_padT:], wi[-_padT:]
        T = _padT
    st["idx_q"] = q                                                  # [T,32,64] uint8
    st["idx_descale_q"] = descale_q                                  # [T,32,2,2] uint8
    st["idx_w"] = wi                                                 # [T,32] fp32
    if os.environ.get("DS41M_DEBUG"):
        print(f"      [idx_prolog] T={T} q={tuple(q.shape)} "
              f"dq={tuple(descale_q.shape)} w={tuple(wi.shape)}", flush=True)


def _pk(ctx):
    """packed → cu_seqlens 列表；无 cu → None。

    ★ 2026-10-21 统一实现：n_seqs==1 且带 cu 也走 packed（与 batch.cu_of 同口径）。
    """
    cu = ctx.get("cu_seqlens")
    n = int(ctx.get("n_seqs", 1) or 1)
    if not cu or n < 1:
        return None
    return [int(x) for x in cu]


# ─────────────────────────── ② metadata（qli 的前置）───────────────────────────
_MD_CACHE: dict = {}
# ★ T-042：packed QSLI 的 candidate_block_length 缓存（按 loc 对象身份，见 _qsli_pk）
_QS_CL_CACHE: dict = {}


def _qli_metadata(T, ratio, topk, device, cand_topk=-1, cand_size=-1, seq_k_len=None,
                  residual_k=None):
    """`quant_lightning_indexer_metadata` —— 官方 modeling_deepseek.py:3128-3147 逐字。

    同 device / int32 / [1024] / contiguous 是硬要求（QLI 侧会校验）。

    `seq_k_len`：**可见的压缩 K 长度**。默认 = T//ratio（普通 prefill）；命中时传 L//ratio
    —— 官方 hit stage B 的 `actual_seq_k = L`（compressed_seq_lens = L//ratio）：query 只有
    尾部 R 个，但索引器要看到**整条 L 的压缩 K**（前缀那几个压缩块已缓存）。

    `residual_k`：官方 `compressed_seq_remainders[ratio] = actual_seq_k % ratio`
    （actual_seq_k = 绝对 KV 长度 = off+T，**不是** compressed_len）。
    ★★ 2026-10-20（DSpark T=2 verify 崩因）：以前传 `k_full % ratio` ✗。
      例：off=19/T=2/ratio=2 → actual=21, lens=10: 正确 remainder=1，旧值=10%2=0。
      错的 remainder 让 QLI 少给一个有效索引，而 `cmp_topk_length` 仍按公式声明 10
      → MQSMLA 在"有效前缀"里读到 -1 → `kernel_MqsmlaKernel` UB 越界（aicore errcode 341）。
    """
    cand_topk = int(cand_topk)
    cand_size = int(cand_size)
    k_full = int(T // ratio if seq_k_len is None else seq_k_len)
    r = int(k_full % ratio if residual_k is None else int(residual_k) % max(1, int(ratio)))
    i32 = torch.int32
    if _CORE.dg_on():
        # decode 图：seqused_k / cmp_residual_k 住固定缓冲，每步按 pos 刷新；
        # ★ metadata **不能跨步缓存**：它编码了 seq_k，缓存会让所有步都用首步的 md。
        #   每次现调 → 捕获期被录进图，replay 时用刷新后的输入重算。
        cu_q = _CORE.dg_const(("qli_cu", int(T)), [0, int(T)], i32, device)
        seq_k = _CORE.dg_meta(
            ("qli_seqk_step", int(T), int(ratio), str(device)),
            lambda: torch.tensor([k_full], dtype=i32, device=device),
            lambda b, p: b.fill_((int(p) + int(T)) // max(1, int(ratio))))
        residual = None
        if ratio != 1:
            # ★ key **不能带 r**：r 随 pos 奇偶交替（(p+T)%ratio），带进 key 会让捕获步
            #   遇到首个 eager 两步没出现过的 r → dg_meta 捕获期新建张量 → H2D →
            #   aclrtMemcpy 107030（P3 实测：round1 r=1、round2 r=1、capture r=0）。
            #   值由 fill 每步原地刷新，key 只需几何常量。
            residual = _CORE.dg_meta(
                ("qli_res_step", int(T), int(ratio), str(device)),
                lambda: torch.tensor([r], dtype=i32, device=device),
                lambda b, p: b.fill_((int(p) + int(T)) % max(1, int(ratio))))
        md = torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer_metadata(
            cu_seqlens_q=cu_q, seqused_q=None, seqused_k=seq_k, cmp_residual_k=residual,
            batch_size=1, max_seqlen_q=-1, max_seqlen_k=-1,
            num_heads_q=N_HEADS, num_heads_k=1, head_dim=HEAD_DIM, topk=topk,
            mask_mode=3, cmp_ratio=ratio, layout_q="TND", layout_k="PA_BBND",
            candidate_topk_blocks=cand_topk, candidate_block_size=cand_size)
        return (md, cu_q, seq_k, residual)
    key = (T, ratio, topk, str(device), cand_topk, cand_size, k_full, r)
    hit = _MD_CACHE.get(key)
    if hit is not None:
        return hit
    cu_q = torch.tensor([0, T], dtype=i32, device=device)
    seq_k = torch.tensor([k_full], dtype=i32, device=device)
    residual = None if ratio == 1 else torch.tensor([r], dtype=i32, device=device)
    md = torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer_metadata(
        cu_seqlens_q=cu_q, seqused_q=None, seqused_k=seq_k, cmp_residual_k=residual,
        batch_size=1, max_seqlen_q=-1, max_seqlen_k=-1,
        num_heads_q=N_HEADS, num_heads_k=1, head_dim=HEAD_DIM, topk=topk,
        mask_mode=3, cmp_ratio=ratio, layout_q="TND", layout_k="PA_BBND",
        candidate_topk_blocks=cand_topk, candidate_block_size=cand_size)
    if len(_MD_CACHE) >= 32:
        _MD_CACHE.clear()
    _MD_CACHE[key] = (md, cu_q, seq_k, residual)        # 顺带把手搭的序列张量一起留着
    return _MD_CACHE[key]


# ─────────────────────────── ② qli ───────────────────────────
def _idx_k_cache(st, ctx, w, n_rows, device, who="?"):
    """PA 索引 K 池 + scale 池；**真写入** = 官方 `Indexer.write_prologue_k`
    （indexer.py:265-274 调用点 / 122-140 实现）。

        indexer_prologue_k(latent=compressor 的 latent, wk, norm_weight,
                           rope_sin/cos = c{ratio}a 表（与压缩器同一张）, k_cache,
                           k_scale_cache, cache_index, norm_eps, storage_mode=0,
                           combined_block_size=-1)

    形状/stride 契约（实测报错驱动）：K 内轴 stride (64,64,1)；descale_k 内轴 (4,4,2,1)。
    cache_index 用**恒等槽位**（行连续），与 qli 侧 block_table=None 的物理顺序读一致。
    池跨层复用（官方 kv_cache.indexer_cache 也是持久的）→ written 标记只写一次。
    """
    hold = _CORE._hold(st)
    ratio = max(1, int(ctx["ratio"]))
    # ★ 每块 **128 个压缩行**（与官方一致；实测官方 L2 拆分缓存 = (66,128,1,64)、
    #   L20 组合缓存 = (1026,16,1,544) → 块行数 = 128*ratio/ratio = 128，与 ratio 无关）。
    #   曾经用 128*ratio（c2a 层 256 行/块 → 33 块 vs 官方 66 块）：块数不同会让内核的
    #   分块/归并顺序不同 → top-k 并列时的取舍顺序不同（正是"仅 token 1023 有 59/262144
    #   槽位顺序不同"的来源假设）。
    rows_per_block = ROWS_PER_BLOCK
    # ★ 命中：本层只写尾部压缩行，但**绝对行号**从 row0 = off//ratio 起，池子按全序列开
    #   （前缀行保持池初值 = fake 缓存；官方 hit 仿真里前缀也是构造出来的缓存内容）
    row0 = int(ctx.get("pos_offset", 0)) // ratio
    n_full = max(n_rows + row0, int(ctx.get("L_full", ctx.get("T", 0))) // ratio)
    # ★ 官方池给的是 ceil((n_rows+1)/128) 块（实测：层 2 n_rows=8193 → 65 ✓；层 20 n_rows=16384
    #   → 129 ✓（官方块表宽 129、缓存张量 130 = 1 空块 + 129））。块表**宽度**参与
    #   logical_k_capacity → 影响内核分块/归并 → top-k 边界并列的取舍（层 2 本来就 65 一致
    #   所以几乎全对；层 20 差一块 → 每行都有 1~2 个元素不同）。
    nblk = max(1, -(-(n_full + 1) // rows_per_block))
    kk, sk = f"_idx_k_r{ratio}", f"_idx_ks_r{ratio}"
    k, s = hold.get(kk), hold.get(sk)
    if k is None or k.shape[0] < 1 + nblk:
        # ★ 与 KV 池同一约定：**块 0 恒空**、数据占块 1..nblk（官方索引器缓存也由调度器
        #   按同一约定分配）→ 让内核遍历顺序与官方一致（曾用 0 基紧凑布局，
        #   导致并列 top-k 的取舍顺序不同）
        if _CORE._KV_MGR is not None:
            k, s = _CORE._KV_MGR.idx_k(hold, ratio, nblk, device)
        else:
            k = torch.zeros(1 + nblk, rows_per_block, 1, PACKED_D, dtype=torch.uint8, device=device)
            s = torch.full((1 + nblk, rows_per_block, 1, 2, 2), 127, dtype=torch.uint8, device=device)
            hold[kk], hold[sk] = k, s
    # ★ 每个"压缩源层"都要用**本层 latent** 重写 K（官方 kv_cache.indexer_cache 逐层更新）：
    #   latent 随该层输入变化 → 不能只写一次（曾加 done 标记，导致层 8 及其后全错）
    latent = st.get("cmp_latent")
    if latent is None:
        raise RuntimeError("索引 K 需要 st['cmp_latent']（本层要先跑 kv_compress）")
    _qli_dbg(f"WRITE-cache who={who} idx={ctx.get('idx')} lat={latent.shape[0]} "
             f"n_rows={int(n_rows)} row0={int(row0)} T={ctx.get('T')}")
    if os.environ.get("DS41M_QLI_DUMP"):
        try:
            import hashlib as _hl
            _rec = dict(who=who, L=int(ctx.get("idx", -1) or -1)) | dict(
                row0=int(row0), n_full=int(n_full), nblk=int(nblk),
                n_rows=int(n_rows), ctx_T=int(ctx.get("T", -1) or -1),
                L_full=int(ctx.get("L_full", -1) or -1),
                pos=int(ctx.get("pos_offset", 0) or 0),
                lat_rows=latent.shape[0],
                lat_sha=_hl.sha1(latent.detach().float().cpu().numpy().tobytes()).hexdigest()[:16])
            _IDXK_LOG.append(_rec)
            try:
                import json as _js
                with open(os.path.join(os.environ["DS41M_QLI_DUMP"], "writers.log"), "a") as _fh:
                    _fh.write(_js.dumps(_rec) + "\n")
            except Exception as _e2:          # noqa: BLE001
                print(f"[idxk-log] writers.log 写失败: {type(_e2).__name__}: {_e2}", flush=True)
        except Exception as _e:               # noqa: BLE001
            # ★ 2026-10-30：这里原来是裸 `pass` —— B 臂的写者记录因此**静默丢失**，
            #   让 A/B 排查误以为"没有写者"。调试块失败必须可见。
            print(f"[idxk-log] 记录失败 idx={ctx.get('idx')} who={who}: "
                  f"{type(_e).__name__}: {_e}", flush=True)
    # 与压缩器同一张表：位置 = p×ratio（组首 token；c1a 二者相同）
    cos, sin = _CORE._comp_rope_tables(n_rows, max(1, int(ctx["ratio"])), ctx, device, row_off=row0)
    d = torch.ops.cann_ops_transformer.ds41

    def _ci_build(p):
        # ★ 2026-10-30 修（对方会话复核指出 + A/B 实证）：`p` 是**token 位置**，
        #   这里除 ratio 得到压缩行号。以前调用点传的是 `row0 = pos_offset//ratio`
        #   （**已经是压缩行号**），于是被二次除 ratio ⇒ 写区整体前移 2 倍
        #   （CP2/ratio2/off=7233：应写 3616..4095，实际写 1808..2287，
        #   实测分块臂 top-512 最大索引恰为 2287、完全不触及 3616..4095）。
        #   现在调用点统一传 **token 位置**（pos_offset），口径与 dg replay 分支一致。
        r0 = int(p) // max(1, int(ctx["ratio"]))
        q = torch.arange(n_rows, dtype=torch.int64, device=device) + r0
        return ((_qslitab(nblk + 1, device)[q // rows_per_block].long()
                 * rows_per_block) + q % rows_per_block)

    # ★ 2026-10-30：`_ci_build` 的入参语义是 **token 位置**（内部再除 ratio）。
    #   `row0` 已是压缩行号，只能给 `_comp_rope_tables(row_off=)`；槽位构造要用 `_pos0`。
    _pos0 = int(ctx.get("pos_offset", 0) or 0)
    if _CORE.dg_on():
        _ci = _CORE.dg_meta(
            ("idx_ci_step", int(n_rows), int(nblk), rows_per_block, str(device)),
            lambda: _ci_build(_pos0), lambda b, p: b.copy_(_ci_build(p)))
    else:
        _ci = _CORE._meta(("idx_ci", int(n_rows), int(_pos0), int(nblk), str(device)),
                          lambda: _ci_build(_pos0))
    _ifc.prolog_k((), dict(
        latent=latent.reshape(-1, latent.shape[-1]),
        wk=_nz("wk"),
        # 同 compressor.norm：官方把它 cast 成 fp32（modeling_deepseek.py:2648-2650）
        norm_weight=(w.get("_packed") or {}).get("attn.indexer.k_norm.weight.f32")
        if (w.get("_packed") or {}).get("attn.indexer.k_norm.weight.f32") is not None
        else w["attn.indexer.k_norm.weight"].to(torch.float32).contiguous(),
        rope_sin=sin.reshape(-1, ROPE_DIM),
        rope_cos=cos.reshape(-1, ROPE_DIM),
        k_cache=k,
        k_scale_cache=s.view(s.shape[0], rows_per_block, 1, 4),  # 末维 4 = 2×2 展平（算子的形状守卫）
        # ★ 槽位 = 旋转块表[p//128]*128 + p%128（官方实测 [129,1,2,…,128]，等价于"首块是池里
        #   最后一块、其余恒等"）。**必须与 reader 用同一张表**；且内核按块序归并 →
        #   表不同会让 top-k 并列/整排时的输出**顺序**不同（实测：层 2 token 1023 一行
        #   512 个索引集合相同、顺序不同 → 注意力求和顺序差 4e-7 → 逐层放大）。
        cache_index=_ci,
        norm_eps=NORM_EPS,
        storage_mode=0,
        combined_block_size=-1,
    ))
    # ★ 2026-10-31：池内容变了 → dense-K 缓存（`_dg_paged_to_dense`）失效。
    _IDX_K_GEN[0] += 1
    if os.environ.get("DS41M_DEBUG"):
        print(f"      [idx_k] ratio={ratio} 写入 {n_rows} 行（池 {nblk}×{rows_per_block}）",
              flush=True)

    # ── 候选源层（层 20）再写一份**组合**缓存：官方 `update_indexer_cache` 对同一份 K 调了
    #    两次 write_prologue_k，第二次给 QSLI 消费者用（indexer.py:286-288）：
    #        storage_mode=1, combined_block_size=self.candidate_block_size
    #    组合行内容 = cbs 个 K 的打包键(64) 紧接它们的 scale(4)（官方 repack_qsli_cache：
    #        torch.cat([keys.view(rows,-1), scales.view(rows,-1)], dim=-1) → 宽 8*(64+4)=544）
    #    布局实测（official_all5/ops/L20_idx_k_cache.pt）：(1026, 16, 1, 544) @ ratio=1, cbs=8，
    #    块 0 空、每块 rows_blk = 128*ratio//cbs 个组合行。
    #    ★ 槽位 = **块表[p//16]*16 + p%16**，块表用官方实测的旋转表 [1025,1,2,…,1024]
    #      （官方 all8 的 `idx_slots_m1` 前 8 个 = 16400..16407 ✓，等价于"除首块外恒等"）。
    #      写侧与 QSLI reader 必须用**同一张表**，否则写/读错位（曾用 p+16 写 + 1 基恒等表读
    #      → 错位 14 行 = 112 个位置；在"可见位置≈topk"的行上看不出来，一到真选择就全错）。
    #      拆分缓存那边是 p + 128 + 恒等块表（官方 [129,1,…] 亦自洽）→ 语义等价、无需改。
    if ctx.get("is_candidate"):
        cbs = int(ctx.get("candidate_block_size", CAND_BLOCK))
        width = cbs * (PACKED_D + 4)
        rows_blk = max(1, rows_per_block // cbs)
        kk2, sk2 = f"_idx_qsli_r{ratio}", f"_idx_qslis_r{ratio}"
        # ★ 池规模镜像官方：组合缓存条目 block_size=16 **行**（拆分缓存 128 行）→ 池给 8 倍块数。
        #   官方实测 (1026, 16, 1, 544) @ n_rows=16384：1026 = 1(空块) + ceil(16384/16) + 1。
        # ★ 2026-10-22 QSLI 定长环（用户口径）：QSLI 每次最多 gather
        #   candidate_topk_blocks×candidate_block_size = 2048×8 = 16384 个候选 K
        #   （这是**候选个数**上限，不是候选在序列里的跨度；候选块号仍是全局
        #   0..L/8-1）。组合缓存是 QSLI 寻址的 **K 镜像**：按 L 开时池/块表随 L 长；
        #   按环长开则绝对压缩行取模后复用固定块表。**默认关**（跨 L 复用已放弃，
        #   按 L 开 = 每个全局候选块读真实 K，精确）；`DS41M_QSLI_CAP=16384` 可开环
        #   做固定几何/显存实验。L=16k 时环长==实际长度 ⇒ 槽位/池形状与旧实现逐位相同；
        #   L>16k 时环会别名（top-k 选择有界变化，只作性能口径）。
        _qcap = int(os.environ.get("DS41M_QSLI_CAP", "0") or 0)
        if _qcap > 0:
            nblk2 = 1 + max(1, -(-int(_qcap) // rows_blk)) + 1
        elif _CORE.dg_on():
            # decode 图：池几何必须与 pos 无关 → 按 L_full 的最大 row0 一次开够
            _r0max = int(ctx.get("L_full", 0) or 0) // max(1, int(ctx["ratio"]))
            nblk2 = max(1 + max(1, -(-(n_rows + _r0max) // rows_blk)) + 1, 1 + nblk)
        else:
            nblk2 = max(1 + max(1, -(-(n_rows + row0) // rows_blk)) + 1, 1 + nblk)
        if os.environ.get("DS41M_QSLI_BLK"):
            nblk2 = int(os.environ["DS41M_QSLI_BLK"])
            if _qcap <= 0:
                nblk2 = max(nblk2, 1 + nblk)
        kc = hold.get(kk2)
        if kc is None or kc.shape[0] < nblk2:
            if _CORE._KV_MGR is not None and _qcap <= 0:
                # 池 = (1+nblk2) 块、块 0 恒空（与 inline 的 nblk2 已含空块口径一致）
                kc = _CORE._KV_MGR.qsli(hold, ratio, nblk2, rows_blk, device)
            else:
                kc = torch.zeros(nblk2, rows_blk, 1, width, dtype=torch.uint8, device=device)
                hold[kk2], hold[sk2] = kc, (rows_blk, nblk2)
        hold[f"_idx_qsli_cap_r{ratio}"] = int(_qcap)
        latent = st.get("cmp_latent")
        cos, sin = _CORE._comp_rope_tables(n_rows, max(1, int(ctx["ratio"])), ctx, device, row_off=row0)
        # ★ 官方 `_prepare_indexer_slot_mapping` 的分母是 (block_size*ratio)//candidate_block_size
        #   = 128//8 = **16**（即组合缓存的 rows_blk），不是 candidate_block_size(8)。
        #   用 8 会让前 16 个压缩位置落到别处 → 每个查询的候选块 0..15 读到错数据。
        slots = _qsli_slots(n_rows, rows_blk, nblk2, device, row_off=row0,
                            ratio=max(1, int(ctx["ratio"])), ring=_qcap)
        # ★ 2026-10-31（主控接手）：组合缓存槽位越界断言。
        #   顺序 K 池的同类断言在命中四档从未触发，但 hit0.5/hit0.7 仍在 L20→L21 崩
        #   507015，而 L20 是唯一走本分支的 ratio=1 候选源层 ⇒ 这里也必须钉住。
        #   ★ 2026-10-31：两个设备 reduce + 同步只跟 (slots, cap) 有关 → 每几何判一次
        #   （`slots` 本身来自 `_qsli_slots` 的几何缓存，指针稳定）。
        _cap_rows = int(nblk2) * int(rows_blk)
        _ckey = ("qsli_slot_cap", int(slots.data_ptr()), int(_cap_rows), str(device))
        if _CORE.META.get(_ckey) is None:
            _sl_max = int(slots.max()) if slots.numel() else 0
            _sl_min = int(slots.min()) if slots.numel() else 0
            if _sl_min < 0 or _sl_max >= _cap_rows:
                raise RuntimeError(
                    f"qsli slot OOB idx={ctx.get('idx')} slots=[{_sl_min},{_sl_max}] "
                    f"cap={_cap_rows}(nblk2={nblk2}x{rows_blk}) kc={tuple(kc.shape)} "
                    f"row0={row0} n_rows={n_rows} nblk={nblk} off={ctx.get('pos_offset')} "
                    f"L={ctx.get('L_full')} ratio={ratio}")
            _CORE.META[_ckey] = True
        # ★★ 2026-10-16（官方 c1a_mask 的**第二半**）：重放段的 `a_qsli_combined_kv` 写
        #   **也要抑制**（slot=-1）。官方 `adaptlib/hit.py:190-200`：
        #       for key, slots in slot_mapping.items():
        #           if key.startswith("c") and key.endswith(("a_cmp_kv",
        #                                                     "a_qsli_combined_kv")):
        #               slots[replay_mask] = -1        # replay_mask = 绝对位置 < H
        #   索引器的 K 组合缓存决定后续**所有**行的候选块/top-k 选择；重放行用"截断窗口"
        #   算出的 K 若覆盖 stage A 已写对的值，会经选择路径一路传到末 token
        #   （实测：只抑制 a_cmp_kv 时 top1 已翻正，但 ' that' 仅领先 ',' 0.6、
        #    8 rank 里 2 个仍翻 ✗）。
        # ⚠️ 2026-10-16 实测存疑：加上这段后（1828Z 4/8 ' that'，1835Z 更在 stage B 的 L2
        #   触发 NaN 断言）**结果反而变差且有新崩溃** ✗ —— 本 mini 用的 vendor 算子
        #   `indexer_prologue_k` 是否像官方那样认 slot=-1 **未经验证**（官方那段改的是它自己的
        #   slot map，未必同一语义）✗。故**默认关闭**，仅在 `DS41M_QSLI_SUPPRESS=1` 时启用。
        _rep = int(ctx.get("hit_replay_tail") or 0) if os.environ.get("DS41M_QSLI_SUPPRESS") == "1" else 0
        if _rep > 0 and int(ctx.get("pos_offset", 0)) > 0:
            _cut = row0 + max(1, _rep // max(1, int(ctx["ratio"])))
            _rows = torch.arange(n_rows, dtype=torch.int64, device=device) + row0
            slots = torch.where(_rows < _cut, torch.full_like(slots, -1), slots)
        _ifc.prolog_k((), dict(
            latent=latent.reshape(-1, latent.shape[-1]),
            wk=_nz("wk"),
            norm_weight=(w.get("_packed") or {}).get("attn.indexer.k_norm.weight.f32")
            if (w.get("_packed") or {}).get("attn.indexer.k_norm.weight.f32") is not None
            else w["attn.indexer.k_norm.weight"].to(torch.float32).contiguous(),
            rope_sin=sin.reshape(-1, ROPE_DIM),
            rope_cos=cos.reshape(-1, ROPE_DIM),
            k_cache=kc,
            k_scale_cache=None,
            cache_index=slots,
            norm_eps=NORM_EPS,
            storage_mode=1,
            combined_block_size=cbs,
        ))
        # ★ 归因实验（默认关）：把官方 dump 的组合缓存里 block 1025（reader 对候选块 0..15
        #   实际读的那 16 行）的字节灌进来。若灌入后层 24 输出与官方逐位一致，
        #   则"官方读未初始化显存"的归因被钉死。
        _inj = os.environ.get("DS41M_INJECT_QSLI")
        if _inj and os.path.exists(_inj):
            _o = torch.load(_inj, map_location="cpu", weights_only=False)["t"].numpy().astype("uint8")
            _o = _o.reshape(-1, kc.shape[1], 1, kc.shape[3])
            ncopy = min(_o.shape[0] - kc.shape[0], 0) if _o.shape[0] > kc.shape[0] else 0
            # 只灌 mini 张量里"官方更大"的那一段尾巴（官方 1026 块 vs mini 1026 块时灌 block 1025）
            if _o.shape[0] >= kc.shape[0]:
                src = _o[_o.shape[0] - 1: _o.shape[0]]
                kc[kc.shape[0] - 1: kc.shape[0]].copy_(torch.from_numpy(src).to(kc.device))
                if os.environ.get("DS41M_DEBUG"):
                    print("      [idx_k] 已注入官方 block 1025 的 16 行字节", flush=True)
        if os.environ.get("DS41M_DEBUG"):
            print(f"      [idx_k] 组合缓存 {tuple(kc.shape)}（块 {nblk}×行 {rows_blk}，"
                  f"宽 {width}）", flush=True)
    _CORE._hold(st)[f"_idx_nblk_r{ratio}"] = int(nblk)
    return k, s, nblk


def _idx_k_write(st, ctx, w):
    """KV 路径上的索引 K 缓存写入（官方 `update_indexer_cache` / `write_prologue_k`）。

    DeepInfer 的层 20 顺序：**全长 KV 路径**里写索引 K（含候选源层的组合缓存），
    之后 compact 段只算 query 侧（idx_prolog + qli），不再重写 K。若 compact 段
    才写，`row0 = pos_offset//ratio` + 全量 latent 会把整段 latent 写到错行
    （实测 idx_sparse 与全长基线不同；QLI 本身对 M 不敏感，见
    NOTES/2026-10-22-qmm-shape-sensitivity.md）。
    """
    ratio = max(1, int(ctx["ratio"]))
    if batch_layout.is_packed(ctx) and _CORE._cp_of(ctx) is None:
        return _idx_k_write_pk(st, ctx, w)      # 2026-10-26：varlen 是唯一口径
    lat = st.get("cmp_latent")
    if lat is None:
        raise RuntimeError("idx_k_write 需要 st['cmp_latent']（本层要先跑 kv_compress）")
    _idx_k_cache(st, ctx, w, lat.shape[0], lat.device, who="op:idx_k_write")


def _qli(st, ctx, w):
    """CP 分块包装：本地行 = head/tail 两块（+pad），按块各跑一次标准 prefill。

    每块 [gs, gs+nv) 都是**某前缀的尾部** → 与 QLI 的单 batch 右对齐语义一致
    （seq_k=(gs+nv)//ratio、residual=(gs+nv)%ratio）；块的输出按本地行序拼回。
    索引 K 缓存必须用**全长 latent**在 off=0 几何下先写一次（块几何写会错行）。
    """
    _cp = _CORE._cp_of(ctx)
    if os.environ.get("DS41M_CMP_ARGDBG") and int(ctx.get("idx", -1)) == 2 and os.environ.get("RANK", "0") == "0":
        print(f"  [qli-in] T={int(ctx.get('T') or -1)} cp={'yes' if _cp is not None else 'no'} "
              f"cpfull={ctx.get('_cp_full')} off={ctx.get('pos_offset')} "
              f"q_rows={tuple(st['idx_q'].shape)}", flush=True)
    # ★ 2026-10-30（阶段 1a）：内核选择开关。默认 vendor；`dg` 走 DeepInfer 内核。
    _kern = os.environ.get("DS41M_IDX_KERNEL", "dg").strip().lower()
    if os.environ.get("DS41M_CAND_DBG"):
        # ★ 2026-10-31（主控）：L20 是否带 candidate 进 QLI（缺了 QSLI 会退回全量 K）
        print(f"  [cand-dbg] qli idx={ctx.get('idx')} ratio={ctx.get('ratio')} "
              f"is_candidate={ctx.get('is_candidate')} off={ctx.get('pos_offset')} "
              f"T={ctx.get('T')}", flush=True)
    if _kern == "dg":
        if _cp is None or ctx.get("_cp_full"):
            _r = _qli_dg(st, ctx, w)
        else:
            _r = _qli_dg_cp(st, ctx, w, _cp)
    elif _cp is None or ctx.get("_cp_full"):
        _r = _qli_pk(st, ctx, w)        # 2026-10-26：varlen 是唯一口径
    else:
        _r = _qli_cp(st, ctx, w, _cp)
    _dump = os.environ.get("DS41M_QLI_DUMP", "").strip()
    if _dump:
        try:                             # 只给 A/B 定位用（逐层落 sp/tl，便于逐位 diff）
            os.makedirs(_dump, exist_ok=True)
            _sp = st.get("cmp_sparse_indices")
            _tl = st.get("cmp_topk_length")
            if not torch.is_tensor(_sp):
                # ★ 2026-10-31（主控）：dump 开启但 sp 不是张量 → 以前**静默不落盘**，
                #   排查时会误以为"没走到这条路径"。必须可见。
                print(f"      [qli-dump] idx={ctx.get('idx')} sp 非张量"
                      f"（{type(_sp).__name__}）→ 本层不落盘", flush=True)
            if torch.is_tensor(_sp):
                # ★ 2026-10-31：CP 下 `torch.distributed` 可能未初始化，
                #   旧代码此时 `_rk` 恒为 0 ⇒ 多 rank 写同一个文件名、后写覆盖，
                #   逐位对比会出现"一半对一半错"的假象。优先用 LOCAL_RANK。
                # ★★ 2026-10-31（最终版）：dump 文件名必须用 **CP rank**。
                #   踩过两个坑：
                #     v1 `torch.distributed.get_rank()`：CP 下未初始化 → 恒 0；
                #     v2 `os.getpid()`：单臂内唯一，但**跨臂不保证对齐**
                #        （CP2 实测 dg 的 r1 文件装的是 vendor r0 的数据）。
                #   `_CORE.CP_STATE.cp_rank` 是 CP 分组的 rank（CP1 时 0），
                #   同一运行内稳定、跨臂语义一致。
                _rk = 0
                try:
                    _cps = getattr(_CORE, "CP_STATE", None)
                    if _cps is not None:
                        _rk = int(getattr(_cps, "cp_rank", 0) or 0)
                except Exception:        # noqa: BLE001
                    _rk = 0
                _fn = os.path.join(_dump, f"r{_rk}_L{int(ctx.get('idx', -1))}_p"
                                          f"{int(getattr(_CORE, 'PASS_NO', 0))}.pt")
                _qli_dbg(f"DUMP idx={ctx.get('idx')} pass={getattr(_CORE, 'PASS_NO', 0)} "
                         f"n_log={len(_IDXK_LOG)}")
                _h = _CORE._hold(st)
                _ratio = max(1, int(ctx["ratio"]))
                _k = _h.get(f"_idx_k_r{_ratio}")
                _bt = _h.get(f"_idx_k_bt_r{_ratio}")
                import hashlib as _hl
                _ksha = None
                if torch.is_tensor(_k):
                    _ksha = _hl.sha1(_k.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]
                _lat = st.get("cmp_latent")
                torch.save({"sp": _sp.detach().cpu(),
                            "tl": _tl.detach().cpu() if torch.is_tensor(_tl) else None,
                            # DS41M_QLI_LATDUMP=1 时连带落 latent（比 latent 逐行差异用）
                            "lat": _lat.detach().cpu() if (os.environ.get("DS41M_QLI_LATDUMP")
                                                           and torch.is_tensor(_lat)) else None,
                            "off": int(ctx.get("pos_offset", 0) or 0),
                            "Tloc": _sp.shape[0],
                            "q_rows": st["idx_q"].shape[0],
                            "k_sha": _ksha,
                            "k_shape": tuple(_k.shape) if torch.is_tensor(_k) else None,
                            "bt": _bt.detach().cpu().tolist()[:8] if torch.is_tensor(_bt) else None,
                            "bt_len": int(_bt.numel()) if torch.is_tensor(_bt) else None,
                            "bt_sha": _hl.sha1(_bt.detach().cpu().numpy().tobytes()).hexdigest()[:16]
                                      if torch.is_tensor(_bt) else None,
                            "idxk_log": list(_IDXK_LOG)}, _fn)
        except Exception as _e:            # noqa: BLE001
            print(f"      [qli-dump] 失败 {type(_e).__name__}: {_e}", flush=True)
    return _r


def _qli_cp(st, ctx, w, cp):
    """CP 的 QLI：**本地 q 直接对 full K 查**（DeepInfer `attn.py:550-580` 同口径）。

    本地行 = head/tail 两块（+pad），每块连续 → 按块各跑一次标准 prefill：
    块 [gs, gs+nv) 是"长为 gs+nv 的前缀的尾部"，seq_k=(gs+nv)//ratio、
    residual=(gs+nv)%ratio（residual 就是官方 `compressed_seq_remainders`，
    让内核把段的绝对位置摆正）。K 缓存用**全长 latent**在 off=0 几何下写一次。
    """
    # ★★ 2026-10-28（命中 + CP）：命中段（pos_offset>0）默认走**全量 packed** 路径——
    #   与 CP1 的 `_qli_pk` 同一实现；分块实现的选择顺序与它不等价（实测层 2 末行
    # ★ 2026-10-30 抢修：以前写的是 `env in (1/on) or pos_offset>0` —— **显式 =0 也切不走**
    #   （命中段永远 >0）⇒ 这个 A/B 开关实际不可用，分块版无法被选中。现在显式 0/off 优先。
    ratio = max(1, int(ctx["ratio"]))
    topk = int(ctx["index_topk"])
    dev = st["idx_q"].device
    Tloc = st["idx_q"].shape[0]
    is_cand = bool(ctx.get("is_candidate"))
    hold = _CORE._hold(st)
    lat = st.get("cmp_latent")
    if lat is not None and not ctx.get("idx_k_ready") and int(ctx["ratio"]) > 1:
        # 全长 latent：写入行 row0=off//ratio 起（块几何下 row0=块起点 + 全量 latent = 错行）
        # ★ 2026-10-28（命中 + CP）：必须用**绝对起点**（stage B = L-R）与**全局 L**，
        #   否则 stage B 会把尾部 latent 写到行 0 起、并重建池几何 ⇒ 覆盖/丢掉 stage A 前缀。
        #   plain prefill（off=0、L_full=total_q）与本改动前逐位相同。
        # ★ 2026-10-30 修（A/B 实证）：以前这里手搓 ctx_full，没有把 T 覆盖成全局行数，
        #   （同形状 (130,128,1,64)、sha 不同）⇒ QLI 输出整体不同（1707/1710 行、
        #   top-512 集合差 ~58%）。现在统一用同一个 helper，K 写入几何逐位一致。
        _qli_dbg(f"cp-write idx={ctx.get('idx')} ratio={ratio} lat={lat.shape[0]}")
        # ★ 2026-10-30 修（A/B + 分支打点实证的根因）：
        #   旧代码把 rank0 的**两段本地 latent 拼成 480 行连续写**到 row0=3616 起。
        #   但 zigzag 两段在压缩空间**不连续**（CP2/L8192：rank0 = 全局 [0,240) ∪ [720,959)
        #   ⇒ 压缩行 3616..3735 与 3976..4095，中间 3736..3975 是 rank1 的数据）。
        #   连续写会把第二段落到 rank1 的槽位，且 3856..4095 完全没写 ⇒
        #   分块臂 top-512 读到空槽（实测出现 1873..2287 的垃圾索引、row3 起
        #   jaccard 掉到 0.038、k_sha 与全长臂不同）。
        #   官方 `update_cp_indexer_cache` 的做法是**逐段在本地算 K、按绝对压缩行号
        #   跨 rank 收行写池**，正是 `_idx_k_rows_cp`。这里改走同一条路径。
        _kg_ok = False
        _kg_env = os.environ.get("DS41M_CP_K_GATHER", "").strip().lower()
        if _kg_env not in ("0", "off", "false", "no"):
            try:
                _kg_ok = bool(_idx_k_rows_cp(st, ctx, cp, w))
            except Exception as _e:               # noqa: BLE001
                print(f"  [cp-k-gather] 失败 {type(_e).__name__}: {_e}（回退拼接写池）", flush=True)
                _kg_ok = False
        if not _kg_ok:
            # ★ 2026-11-02（用户裁决）：删「gather 全长 → 全长写池」回退。
            #   真实推理应只按本地 q 算；全长 gather 是白算。铁律 0：不静默降级。
            raise RuntimeError(
                "CP K 池写入失败：`_idx_k_rows_cp` 未成功"
                "（或 DS41M_CP_K_GATHER=0）。全长回退路径已删除（白算）；"
                "请修 CP 分块 K 池写入。")
    base = dict(ctx)
    base["idx_k_ready"] = f"_idx_k_r{ratio}" in hold
    # ★ 压缩侧用**原序列长**（compact 段 = L）；块起点要加 pos_offset 变绝对位置
    _seq_cp = int(getattr(cp, "seq_len", 0) or cp.plan.total_q)
    _pofs_cp = int(getattr(cp, "pos_offset", 0) or 0)
    base["L_full"] = _seq_cp
    base["cu_seqlens"] = None
    # ★ 2026-10-28（命中 + CP）：逻辑长用 L（plain 时 _seq_cp==tot，逐位不变）
    base["seq_lens"] = [int(_seq_cp)]
    base["row_lens"] = None
    base["n_seqs"] = 1
    base["_cp_chunk"] = 1          # 分块是正常 prefill 子段，禁 DSpark 小 T 特判
    base.pop("cp_positions", None)
    base.pop("cp_total_q", None)
    # ★ 2026-10-28（P0-2）：`sp`/`tl` 是**累加器**（下面按块切片写），改 `_meta` 缓存复用；
    #   key 必须带**索引源层后缀** —— 它们会以 `_cmp_sparse_r*` 进 hold，被下游 consumer 层
    #   跨层读，不带后缀会让两个源层共用同一块 → 后者覆盖前者。缓存命中后**必须显式复位**
    #   （原值全靠"每次新建"保证）。
    if _CORE._HOST_OPT:
        # key 带 ratio（hold 就是按 `_cmp_sparse_r{ratio}` 分桶的）+ 索引源层后缀。
        sp = _CORE._meta(("cmp_sp_cp", int(Tloc), int(topk), int(ratio),
                          _CORE._sfx(ctx, "idx"), str(dev)),
                         lambda: torch.full((Tloc, 1, topk), -1, dtype=torch.int32, device=dev))
        sp.fill_(-1)
        tl = _CORE._meta(("cmp_tl_cp", int(Tloc), int(ratio),
                          _CORE._sfx(ctx, "idx"), str(dev)),
                         lambda: torch.zeros((Tloc, 1), dtype=torch.int32, device=dev))
        tl.zero_()
    else:
        sp = torch.full((Tloc, 1, topk), -1, dtype=torch.int32, device=dev)
        tl = torch.zeros((Tloc, 1), dtype=torch.int32, device=dev)
    cand = [None, None]
    _ck = ("_li_cand_r1", "_li_candlen_r1")
    for ls, gs, nv in cp.plan.local_blocks():
            if nv <= 0:
                continue
            st2 = dict(st)
            st2["idx_q"] = st["idx_q"][ls:ls + nv].contiguous()
            st2["idx_descale_q"] = st["idx_descale_q"][ls:ls + nv].contiguous()
            st2["idx_w"] = st["idx_w"][ls:ls + nv].contiguous()
            _qli_chunk(st2, dict(base, pos_offset=int(gs) + _pofs_cp, T=int(nv)), w)
            sp[ls:ls + nv] = st2["cmp_sparse_indices"].reshape(nv, 1, -1).to(torch.int32)
            tl[ls:ls + nv] = st2["cmp_topk_length"].reshape(nv, 1).to(torch.int32)
            if is_cand:
                for i, key in enumerate(_ck):
                    v = hold.get(key)
                    if v is None:
                        continue
                    if cand[i] is None:
                        cand[i] = torch.zeros((Tloc,) + tuple(v.shape[1:]),
                                              dtype=v.dtype, device=v.device)
                    cand[i][ls:ls + nv] = v.reshape((nv,) + tuple(v.shape[1:]))
    st["idx_sparse"] = sp
    st["cmp_sparse_indices"] = sp
    st["cmp_topk_length"] = tl
    hold[f"_cmp_sparse_r{ratio}" + _CORE._sfx(ctx, "idx")] = sp
    hold[f"_cmp_topklen_r{ratio}" + _CORE._sfx(ctx, "idx")] = tl
    if is_cand:
        for key, v in zip(_ck, cand):
            if v is not None:
                hold[key] = v
    return sp


def _idx_k_rows_cp(st, ctx, cp, w):
    """官方 `update_cp_indexer_cache`：每段本地算 K+scale → 行 gather(68B/行) → 散写池。

    行数/槽位严格按官方 `generate_compressed_position_ids`：
      compressed_len = (start+seq)//ratio - start//ratio；ids = start//ratio + [0, len)
    本地 latent 末尾多出的 carry 行不参与（与官方 slot_mapping 的 pad 同口径）。
    返回 True 表示池已按官方口径写好（调用方应传 idx_k_ready=True 只读池）。
    """
    lats = st.get("_cp_seg_latents")
    blocks = st.get("_cp_seg_blocks")
    if not lats or not blocks or cp is None:
        return False
    ratio = max(1, int(ctx["ratio"]))
    if ratio <= 1:
        return False
    dev = st["idx_q"].device
    off0 = int(getattr(cp, "pos_offset", 0) or 0)
    rpb = ROWS_PER_BLOCK
    hold = _CORE._hold(st)
    n_full = max(int(ctx.get("L_full", 0) or 0) // ratio + 1, 1)
    nblk = max(1, -(-(n_full + 1) // rpb))
    kk, sk = f"_idx_k_r{ratio}", f"_idx_ks_r{ratio}"
    k_pool = hold.get(kk)
    s_pool = hold.get(sk)
    if k_pool is None or k_pool.shape[0] < 1 + nblk:
        if _CORE._KV_MGR is not None:
            k_pool, s_pool = _CORE._KV_MGR.idx_k(hold, ratio, nblk, dev)
        else:
            k_pool = torch.zeros(1 + nblk, rpb, 1, PACKED_D, dtype=torch.uint8, device=dev)
            s_pool = torch.full((1 + nblk, rpb, 1, 2, 2), 127, dtype=torch.uint8, device=dev)
            hold[kk], hold[sk] = k_pool, s_pool
    ps = int(blocks[0][2])

    def _clen(gs, nv):
        s_c = (off0 + int(gs)) // ratio
        e_c = (off0 + int(gs) + int(nv)) // ratio
        return max(0, e_c - s_c)

    lens = [_clen(gs, nv) for _ls, gs, nv in blocks]
    # ★★ 2026-10-31（64k@0.97 + CP2 + dg 的 topk NaN 实测定根因）：
    #   这里**不能用本 rank 的 max(lens)** —— `_cp_gather_2seg` 的载荷形状是
    #   [2, R, 68]，`torch.distributed.all_gather` 要求**所有 rank 的 R 相同**。
    #   实测 CP2/L65536/hit0.97：off0=63430，ratio=2，段行数 527/527/527/525
    #   ⇒ rank0 的 lens=[263,263]（R=263）、rank1 的 lens=[264,263]（R=264）。
    #   形状不一致时收来的行按错位写进池：实测 rank0 的 K 行 31978.. == rank1
    #   同块数据**整体前移 2 行**（逐字节相等 262/264 行），块 3 前移 1 行。
    #   错位的 68B 行会让 4 个 scale 字节取到相邻行的 K nibble（实测 K 行 31978
    #   的 sf=[34,191,251,61] ⇒ 2^124），`fp8_fp4_mqa_logits` 的 q·k 溢出成 inf，
    #   再 `relu(inf)*w` 出 NaN ⇒ `topk_select` trap "NaN detected" ⇒ NonZero 507035。
    #   修法：R 取**与 rank 无关的段上界**，与已验收的压缩行 gather
    #   `exec_ops._cp_cmp_gather_rows` 完全同口径（`ceil(ps/ratio)+1`）。
    R = max(1, -(-ps // max(1, ratio)) + 1)
    if max(lens) > R:                              # 防御：上界假设被打破就当场报
        raise RuntimeError(
            f"_idx_k_rows_cp: 段压缩行 {max(lens)} > R={R} "
            f"(ps={ps} ratio={ratio} lens={lens} off0={off0})")
    # ★ scratch 用**与池相同的旋转块表**槽位（2026-10-30：恒等槽位与池不等价）
    _tb_s = _qslitab(nblk + 1, dev).long()
    # ★★ 2026-10-31（CP4 空泡）：scratch/pad 改**常驻缓冲**（以前每次调用都新开一份
    #   zeros/full → 3 个 K 级大 ZerosLike + 分配器抖动；本函数每 forward 调 3 次）。
    #   安全性：prolog_k 只写 `ci` 行、随后只读同一批 `ci` 行；pad 只有各段
    #   [i*R, i*R+n) 被读（n 是全局几何，跨 rank 一致）⇒ 未写行永远不会被消费。
    _nb_rows = (1 + nblk) * rpb
    k_s = _CORE._cache_buf(st, f"_idx_k_scratch_r{ratio}", _nb_rows, PACKED_D, dev,
                           torch.uint8)[:_nb_rows].view(1 + nblk, rpb, 1, PACKED_D)
    _s_buf = _CORE._hold(st).get(f"_idx_ks_scratch_r{ratio}")
    if (_s_buf is None or _s_buf.shape[0] < _nb_rows
            or _s_buf.dtype != torch.uint8):
        _s_buf = torch.full((_nb_rows, 4), 127, dtype=torch.uint8, device=dev)
        _CORE._hold(st)[f"_idx_ks_scratch_r{ratio}"] = _s_buf
    s_s = _s_buf[:_nb_rows].view(1 + nblk, rpb, 1, 2, 2)
    pad = _CORE._cache_buf(st, f"_idx_kpad_r{ratio}", 2 * R, PACKED_D + 4, dev,
                           torch.uint8)[:2 * R]
    _nw = (w.get("_packed") or {}).get("attn.indexer.k_norm.weight.f32")
    if _nw is None:
        _nw = w["attn.indexer.k_norm.weight"].to(torch.float32).contiguous()
    for i, (ls, gs, nv) in enumerate(blocks):
        n = lens[i]
        if n <= 0 or i >= len(lats) or lats[i] is None:
            continue
        lat = lats[i][:n]                      # ★ 丢掉末尾 carry 行（官方 pad 口径）
        pos0 = (off0 + int(gs)) // ratio
        cos, sin = _CORE._comp_rope_tables(n, ratio, ctx, dev, row_off=pos0)
        # ★ 2026-10-31：`_p_s`/`ci` 是 (n, pos0, rpb, nblk, device) 的纯几何量（rotation
        #   块表对给定 nblk 是常量）→ 每层每块重发 Range/Add/FloorDiv/Index/Mul/FloorMod/Add
        #   纯属浪费；`_meta` 缓存后整段消失。
        _p_s, ci = _CORE._meta(
            ("idxk_ci", int(n), int(pos0), int(rpb), int(nblk), str(dev)),
            lambda: ((torch.arange(n, device=dev) + pos0),
                     _tb_s[(torch.arange(n, device=dev) + pos0) // rpb] * rpb
                     + (torch.arange(n, device=dev) + pos0) % rpb))
        _ifc.prolog_k((), dict(
            latent=lat.reshape(-1, lat.shape[-1]),
            wk=_nz("wk"),
            norm_weight=_nw,
            rope_sin=sin.reshape(-1, ROPE_DIM), rope_cos=cos.reshape(-1, ROPE_DIM),
            k_cache=k_s, k_scale_cache=s_s.view(s_s.shape[0], rpb, 1, 4),
            cache_index=ci, norm_eps=NORM_EPS, storage_mode=0, combined_block_size=-1))
        pad[i * R:i * R + n, :PACKED_D] = k_s.view(-1, PACKED_D).index_select(0, ci)
        pad[i * R:i * R + n, PACKED_D:] = s_s.reshape(-1, 4).index_select(0, ci)
    gather = getattr(_CORE, "_cp_gather_2seg", None)
    if gather is None:
        raise RuntimeError("_idx_k_rows_cp 需要 exec_ops._cp_gather_2seg")
    allr = gather(pad.reshape(2, R, -1), cp)     # [2c·R, 68]
    tb = _qslitab(nblk + 1, dev).long()
    total = 2 * int(cp.plan.cp_size)
    for b in range(total):
        gs_b = b * ps
        nv_b = max(0, min(ps, int(cp.plan.total_q) - gs_b))
        n_b = _clen(gs_b, nv_b)
        if n_b <= 0:
            continue
        # ★ 2026-10-31：每段 (p, slot) 也是纯几何（8 段/层 × Range/Add/FloorDiv/Index/
        #   Mul/FloorMod/Add）→ `_meta` 缓存，稳态零算子。
        p, slot = _CORE._meta(
            ("idxk_sc", int(n_b), int(off0), int(gs_b), int(ratio), int(rpb),
             int(nblk), str(dev)),
            lambda: (lambda _p: (_p, tb[_p // rpb] * rpb + _p % rpb))(
                (off0 + gs_b) // ratio + torch.arange(n_b, device=dev)))
        rows = allr[b * R:b * R + n_b]
        k_pool.view(-1, PACKED_D).index_copy_(0, slot, rows[:, :PACKED_D])
        s_pool.reshape(-1, 4).index_copy_(0, slot, rows[:, PACKED_D:])
    # ★ 2026-10-31：池内容变了 → dense-K 缓存（`_dg_paged_to_dense`）失效。
    _IDX_K_GEN[0] += 1
    hold[f"_idx_k_bt_r{ratio}"] = batch_layout.rot_block_table(nblk, 1, dev)
    hold[f"_idx_nblk_r{ratio}"] = nblk
    if os.environ.get("DS41M_CP_K_GATHER_DBG"):
        # ★ 2026-10-31：R 必须**跨 rank 相同**（all_gather 形状契约）；这里把
        #   rank/lens/off0 一起打出来，A/B 时一眼能看出两 rank 的 R 是否一致。
        import hashlib as _hlg
        _ksha = _hlg.sha1(k_pool.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        print(f"  [cp-k-gather] idx={ctx.get('idx')} rank={_dg_rank()} off0={off0} "
              f"ps={ps} lens={lens} R={R} segs={total} k_sha={_ksha} "
              f"payload={total * R * (PACKED_D + 4) // 1024} KB", flush=True)
    return True


def _idx_k_cp_probe(st, ctx, cp, w):
    """校验探针（DS41M_CP_K_PROBE=1）：逐段本地算 K/scale → 与现有池对应行逐位对比。

    ★ 不改任何行为，只验证「官方 K 行 gather」的核心步骤（每段 prolog_k 到 scratch、
      恒等槽位、按绝对行号寻址）与现行「全量 latent 写池」是否逐位一致。
    """
    lats = st.get("_cp_seg_latents")
    blocks = st.get("_cp_seg_blocks")
    if not lats or not blocks or cp is None:
        return
    ratio = max(1, int(ctx["ratio"]))
    if ratio <= 1:
        return
    dev = st["idx_q"].device
    off0 = int(getattr(cp, "pos_offset", 0) or 0)
    rpb = ROWS_PER_BLOCK
    hold = _CORE._hold(st)
    k_pool = hold.get(f"_idx_k_r{ratio}")
    s_pool = hold.get(f"_idx_ks_r{ratio}")
    if not torch.is_tensor(k_pool) or not torch.is_tensor(s_pool):
        print("  [cp-k-probe] 池不存在，跳过", flush=True)
        return
    R = max(int(-(-int(nv) // ratio)) + 1 for _, _, nv in blocks)
    nb_s = max(1, -(-R // rpb))
    k_s = torch.zeros(nb_s, rpb, 1, PACKED_D, dtype=torch.uint8, device=dev)
    s_s = torch.full((nb_s, rpb, 1, 2, 2), 127, dtype=torch.uint8, device=dev)
    bad = tot = 0
    _nw = (w.get("_packed") or {}).get("attn.indexer.k_norm.weight.f32")
    if _nw is None:
        _nw = w["attn.indexer.k_norm.weight"].to(torch.float32).contiguous()
    for i, (ls, gs, nv) in enumerate(blocks):
        if i >= len(lats) or lats[i] is None:
            continue
        lat = lats[i]
        n = lat.shape[0]
        pos0 = (off0 + int(gs)) // ratio
        print(f"  [cp-k-probe] seg i={i} gs={int(gs)} nv={int(nv)} lat_rows={n} pos0={pos0}", flush=True)
        cos, sin = _CORE._comp_rope_tables(n, ratio, ctx, dev, row_off=pos0)
        ci = torch.arange(n, dtype=torch.int64, device=dev)
        _ifc.prolog_k((), dict(
            latent=lat.reshape(-1, lat.shape[-1]),
            wk=_nz("wk"),
            norm_weight=_nw,
            rope_sin=sin.reshape(-1, ROPE_DIM), rope_cos=cos.reshape(-1, ROPE_DIM),
            k_cache=k_s, k_scale_cache=s_s.view(s_s.shape[0], rpb, 1, 4),
            cache_index=ci, norm_eps=NORM_EPS, storage_mode=0, combined_block_size=-1))
        ks = k_s.view(-1, PACKED_D)[:n]
        ss = s_s.reshape(-1, 4)[:n]
        # ★ 池用**旋转块表** _qslitab 的槽位（不是 (1+p//BS)*BS+p%BS —— 那是压缩 KV 池）
        p = torch.arange(n, device=dev) + pos0
        _nb_pool = int(hold.get(f"_idx_nblk_r{ratio}") or 0)
        if _nb_pool <= 0:
            _nb_pool = max(1, -(-(int(ctx.get("L_full", 0) or 0) // ratio + 1) // rpb))
        _tb = _qslitab(_nb_pool + 1, dev).long()
        slot = _tb[p // rpb] * rpb + p % rpb
        kp = k_pool.view(-1, PACKED_D).index_select(0, slot)
        sp = s_pool.reshape(-1, 4).index_select(0, slot)
        dk = int((ks != kp).any(-1).sum()); ds = int((ss != sp).any(-1).sum())
        bad += max(dk, ds); tot += n
        print(f"  [cp-k-probe] i={i} gs={int(gs)} n={n} K不同行={dk} scale不同行={ds}", flush=True)
    print(f"  [cp-k-probe] 汇总 不同行 {bad}/{tot}", flush=True)



def _qli_chunk(st, ctx, w):
    """`quant_lightning_indexer`：q/k/w → 每行 topk=512 个稀疏压缩 KV 位置。官方 indexer.py:331-393。"""
    d = torch.ops.cann_ops_transformer.ds41
    q = st.get("idx_q")
    if q is None:
        raise RuntimeError("qli 需要 st['idx_q']：请确认本层先跑了 idx_prolog")
    T = q.shape[0]
    ratio = max(1, int(ctx["ratio"]))                  # c2a 层 =2 / c1a 层 =1（都走这条路）
    topk = int(ctx["index_topk"])
    # 压缩后的 K 长度 = **压缩器实际输出行数**（满组 + 余数；官方 compressed_seq_lens(+remainder)）
    lat = st.get("cmp_latent")
    seq_k = lat.shape[0] if lat is not None else T // ratio
    if os.environ.get("DS41M_SEQK_OVERRIDE"):
        seq_k = int(os.environ["DS41M_SEQK_OVERRIDE"])
    # ★★ 2026-10-17（**病C 的根因**，据此修）：可见的压缩 K 长度必须是
    #   **段起点 + 本段行数 = (off+T)//ratio**，而**不是 L//ratio** ✗。
    #   · stage B（off = 段基址）：off+T == L ⇒ 与旧值**相同** ✓（前缀行在池里、绝对行号寻址 ✓）；
    #   · prefill / 无命中（off=0, T==L）：两值**相同** ✓ ⇒ 零回归 ✓；
    #   · **stage A（off=0, T=H<L）**：旧值 L//ratio 会多声明 (L-H)//ratio = 256 行
    #     **本段从没写过的零行**为"有效压缩 K" ✗ ⇒ 索引器把它们当候选 ⇒ 读零行 ⇒
    #     fp8 scale 字节=0 ⇒ 反量化 0/0 = **NaN** ✗ ⇒ 崩在下一层的 `moe_topk_gate_forward`
    #     （实测崩点恒定在 c2a_source 层 L8/L14 ✓✓；prefill 因 T==L 两值相等 ⇒ **从不崩** ✓✓，
    #      这解释了"prefill 从不崩、命中 stage A 崩"的全部现象 ✓）。
    #   `DS41M_SEQK_LFULL=1` 恢复旧（错）值 ⇒ **可按需复现该崩** ✓（A/B 证据）。
    _L = int(ctx.get("L_full", 0) or 0)
    _offK = int(ctx.get("pos_offset", 0))
    if os.environ.get("DS41M_SEQK_LFULL", "").strip().lower() in ("1", "on", "true", "yes"):
        _spanK = _L // ratio
    else:
        _spanK = (_offK + T) // ratio
    k_hit = _spanK if (_L > T and _spanK > T // ratio) else None
    if k_hit is not None:
        seq_k = k_hit
    # ★ seqused_k 用官方 `compressed_seq_lens[ratio]`（**不含** latent 的 +1 overhang 行）：
    #   曾经传 latent 行数（ratio2 → 8193），比官方多一行"有效 K"。在 token≈1023 这类
    #   "可见位置 ≈ topk"的边界行上，多出的那行会挤掉一个真元素
    #   （实测：官方把 345 挪到末位、mini 挪 403；集合相同、只差这一个位置）。
    if os.environ.get("DS41M_SEQK_EXACT", "1") == "1":
        seq_k = min(seq_k, k_hit if k_hit is not None else T // ratio)
    n_rows_k = lat.shape[0] if lat is not None else T // ratio   # 写缓存用**latent 行数**
    device = q.device

    # ★ 候选源层（assemble 的 is_candidate，官方 candidate_source_layer=20）：
    #   qli 用**带候选**的 metadata（candidate_topk_blocks/block_size），返回值里多出候选块
    is_cand = bool(ctx.get("is_candidate"))
    # K 缓存要在 per-row 分支之前拿到 nblk（块表宽度）；写缓存本身与 metadata 无关。
    # ★ 边界拆分：KV 路径已在 cut 前用全长 latent 写过（`idx_k_write`），compact 段
    #   只读池、**绝不重写**（否则 full latent + row0=pos_offset 会写错行）。
    _holdd = _CORE._hold(st)
    if (bool(ctx.get("idx_k_ready"))
            and f"_idx_k_r{ratio}" in _holdd and f"_idx_nblk_r{ratio}" in _holdd):
        _qli_dbg(f"ch-readonly idx={ctx.get('idx')} ratio={ratio} T={T}")
        k = _holdd[f"_idx_k_r{ratio}"]
        dk = _holdd[f"_idx_ks_r{ratio}"]
        nblk = int(_holdd[f"_idx_nblk_r{ratio}"])
    else:
        k, dk, nblk = _idx_k_cache(st, ctx, w, n_rows_k, device)
    for name, t, stride in (("k", k, (PACKED_D, PACKED_D, 1)),
                            ("descale_k", dk, (4, 4, 2, 1))):
        if tuple(t.stride()[1:]) != stride:
            raise RuntimeError(f"{name} 内轴 stride 必须为 {stride}，实际 {tuple(t.stride())}")
    # ★★ 2026-10-20 DSpark 因果泄漏的彻底修法（QLI 逐行 batch）：
    #   单 batch + cu=[0,T] 时，QLI 内核把 **T 个查询当成一个序列的尾部**，压缩行
    #   的 per-row 可见长度只能由 seqused_k/residual 推；实测未来压缩行仍会进入
    #   row0 的 top-k。改成 **每行一个 batch**：cu=[0,1,..,T]、逐行 seqused_k 和
    #   residual = (off+1+i)//ratio / %ratio、block_table 重复 T 行 → 每行的因果
    #   长度由内核直接按 batch 处理，未来行根本不参与选择。
    #   只对 T>1、off>0、非候选源生效（prefill off=0 与 T=1 保持原路径；候选源 QSLI 另算）。
    # ★ 只对**小 T**（DSpark verify T=6 这类）逐行；hit stage B 的 T≈R(数千) 保持
    #   原来的单 batch 路径（口径与既有 hit 基准/PARAMS replay_tail=128 一致）。
    # ★ CP 分块：即使 T≤32 也走标准单 batch（与 CP1 的整段 prefill 同口径；
    #   DSpark 的逐行路径只用于 verify 的 T≤32 小段）
    _per_row = (1 < int(T) <= 32 and _offK > 0
                and os.environ.get("DS41M_QLI_PER_ROW", "1") != "0"
                and not ctx.get("_cp_chunk"))
    if os.environ.get("DS41M_IDX_TRACE") and os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"[idx-trace] qli T={T} ratio={ratio} seq_k={seq_k} nblk={nblk} "
              f"is_cand={is_cand} per_row={_per_row} off={_offK}", flush=True)
    if _per_row:
        _i64 = torch.arange(int(T), dtype=torch.int64, device=device)

        def _cl_rows(p):
            # ★ 不能 clamp 到 index_topk(512)：seqused_k 是**可见压缩行数**，长上下文
            #   (压缩行 >512) 时截断会让 verify 只看前 512 行（单 batch 路径同样不截断）。
            return ((_i64 + int(p) + 1) // ratio).to(torch.int32)

        def _rs_rows(p):
            return ((_i64 + int(p) + 1) % ratio).to(torch.int32)

        if _CORE.dg_on():
            cu = _CORE.dg_const(("qli_cu_rows", int(T)),
                                list(range(int(T) + 1)), torch.int32, device)
            sk = _CORE.dg_meta(
                ("qli_cl_rows", int(T), int(ratio), int(topk), str(device)),
                lambda: _cl_rows(_offK), lambda b, p: b.copy_(_cl_rows(p)))
            rs = None if ratio == 1 else _CORE.dg_meta(
                ("qli_rs_rows", int(T), int(ratio), str(device)),
                lambda: _rs_rows(_offK), lambda b, p: b.copy_(_rs_rows(p)))
        else:
            _key = (int(T), int(_offK), int(ratio), int(topk), str(device))
            cu = _CORE._meta(
                ("cu_q_rows",) + _key,
                lambda: torch.tensor(list(range(int(T) + 1)), dtype=torch.int32,
                                     device=device))
            sk = _CORE._meta(("cl_q_rows",) + _key, lambda: _cl_rows(_offK))
            rs = None if ratio == 1 else _CORE._meta(
                ("rs_q_rows",) + _key, lambda: _rs_rows(_offK))
        md = torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer_metadata(
            cu_seqlens_q=cu, seqused_q=None, seqused_k=sk, cmp_residual_k=rs,
            batch_size=int(T), max_seqlen_q=-1, max_seqlen_k=-1,
            num_heads_q=N_HEADS, num_heads_k=1, head_dim=HEAD_DIM, topk=topk,
            mask_mode=3, cmp_ratio=ratio, layout_q="TND", layout_k="PA_BBND",
            # 候选源层（20）：逐行 batch 下仍要产候选块（否则 QSLI 读到空候选）
            candidate_topk_blocks=(int(ctx.get("candidate_topk_blocks", CAND_TOPK))
                                   if is_cand else -1),
            candidate_block_size=(int(ctx.get("candidate_block_size", CAND_BLOCK))
                                  if is_cand else -1))
        mdi = (md, cu, sk, rs)
        bt_rows = _qslitab(nblk + 1, device)[:nblk].unsqueeze(0).repeat(int(T), 1)
    else:
        mdi = _qli_metadata(T, ratio, topk, device,
                            cand_topk=int(ctx.get("candidate_topk_blocks", CAND_TOPK)) if is_cand else -1,
                            cand_size=int(ctx.get("candidate_block_size", CAND_BLOCK)) if is_cand else -1,
                            seq_k_len=seq_k,
                            # 官方 compressed_seq_remainders = actual_seq_k % ratio
                            # （actual_seq_k = 绝对 KV 长度 = off+T，不是 compressed_len）
                            residual_k=(_offK + T) % max(1, ratio))
        bt_rows = _qslitab(nblk + 1, device)[:nblk].unsqueeze(0)
    md = mdi[0]

    sparse, _val, _ci, _cl = _ifc.qli(
        (q, k, st["idx_w"].contiguous(), st["idx_descale_q"], dk, topk, 1),
        dict(
        cu_seqlens_q=mdi[1], seqused_q=None, seqused_k=mdi[2], cmp_residual_k=mdi[3],

        block_table=bt_rows,
        metadata=md, max_seqlen_q=-1, mask_mode=3, cmp_ratio=ratio,
        layout_q="TND", layout_k="PA_BBND", return_value=False,
        # ★ 候选源层（层 20）必须打开候选：官方 quant_lightning_indexer 用
        #   candidate_topk_blocks=self.candidate_topk_blocks 调用；op 内部
        #   `candidate_enabled = int(candidate_topk_blocks) > 0`（dsl:4484）→ 传 -1 时
        #   候选输出是**空张量**（实测 indices(0,)/len(0,) 被静默吞掉）
        candidate_topk_blocks=int(ctx.get("candidate_topk_blocks", CAND_TOPK)) if is_cand else -1,
        candidate_block_size=int(ctx.get("candidate_block_size", CAND_BLOCK)) if is_cand else -1))
    if (os.environ.get("DS41M_QLI_CHECK") and int(T) > 1
            and int(os.environ.get("RANK", "0")) == 0
            and not _CORE.dg_on()):
        # 诊断（因果泄漏）：row0 的原始 top-k 里是否出现"未来压缩行"（>= 本行 seqused_k）
        try:
            _r0 = sparse[0].reshape(-1)
            _sk = mdi[2].reshape(-1)
            _lim0 = int(_sk[0]) if _sk.numel() > 1 else int(_sk[0])
            _v = _r0[_r0 >= 0]
            _fut = int((_v >= _lim0).sum())
            print(f"  [qli-check] L{ctx.get('idx')} T={int(T)} off={_offK} "
                  f"row0 seqk={_lim0} max={int(_v.max()) if _v.numel() else -1} "
                  f"nvalid={int(_v.numel())} future={_fut}", flush=True)
        except Exception as _e:                        # noqa: BLE001
            print(f"  [qli-check] 失败 {type(_e).__name__}: {_e}", flush=True)
    st["idx_sparse"] = sparse                           # [T,1,512] int32（-1 = 无效）
    # ── 给 mqsmla 的消费端（约定见协作消息）：真实压缩 KV 索引 + 每行有效长度 ──
    st["cmp_sparse_indices"] = sparse                   # int32 [T,1,512]（topk 维 = index_topk）
    # ★ 跨层共享（官方 kv_cache.shared_idx 口径）：消费者层没有索引器，要读源层这份
    _CORE._hold(st)[f"_cmp_sparse_r{ratio}" + _CORE._sfx(ctx, "idx")] = sparse
    _CORE._hold(st)[f"_cmp_topklen_r{ratio}" + _CORE._sfx(ctx, "idx")] = (sparse >= 0).sum(-1).to(torch.int32)
    # 长度必须 [T,1]：对齐 exec_ops 的 ori 侧 length（_win_indices）与 metadata 的 [T1,N2] 校验
    st["cmp_topk_length"] = (sparse >= 0).sum(-1).to(torch.int32)                 # int32 [T,1]
    if is_cand:                                       # 候选块（官方 kv_cache.candidates）
        _CORE._hold(st)["_li_cand_r1"] = _ci
        _CORE._hold(st)["_li_candlen_r1"] = _cl
        if os.environ.get("DS41M_CAND_DBG"):
            # ★ 2026-10-31（主控修）：`cand_size` 只在本函数的 metadata kwargs 里出现过，
            #   **不是局部变量**；误用会在 L20 抛 NameError ⇒ qli 算子失败 ⇒ 索引为空 ⇒
            #   下一层 mqsmla 形状错。块宽必须现算。
            _cand_dbg_check(ctx, _ci, _cl, topk,
                            int(ctx.get("candidate_block_size", CAND_BLOCK)), ratio)
        if os.environ.get("DS41M_DEBUG"):
            print(f"      [qli] 候选块: indices{tuple(_ci.shape)} len{tuple(_cl.shape)}", flush=True)
    if os.environ.get("DS41M_DEBUG"):
        s = sparse.reshape(-1)
        print(f"      [qli] ratio={ratio} k_pool={nblk}×{k.shape[1]} seq_k={seq_k} "
              f"sparse={tuple(sparse.shape)} 有效={int((s >= 0).sum())} "
              f"每行长度={st['cmp_topk_length'].reshape(-1)[:4].tolist()}…", flush=True)


# ── QSLI 旋转块表 / 槽位（官方实测口径，读写共用）──
_QSLI_TB: dict = {}


def _qslitab(nblk2, device):
    """旋转块表 [n-1, 1, 2, …, n-2]（官方 QLI/QSLI 缓存共用这一形态）。"""
    key = ("q", nblk2, str(device))
    t = _QSLI_TB.get(key)
    if t is None:
        t = torch.cat([torch.tensor([nblk2 - 1], dtype=torch.int32, device=device),
                       torch.arange(1, nblk2 - 1, dtype=torch.int32, device=device)])
        if len(_QSLI_TB) >= 8:
            _QSLI_TB.clear()
        _QSLI_TB[key] = t
    return t


def _qsli_table(nblk2, device):
    """官方 QSLI 的组合缓存块表：**[1025, 1, 2, …, 1024]**（首块是池里最后一块）。

    实测来源：official_all8 的 `L20_idx_slots_m1` 前 8 个 = 16400..16407 = 1025*16 + 0..7，
    而 `block_tables_c1a_qsli_combined_kv` 前 8 个 = [1025,1,2,…,7]。
    对 mini 而言"1025"只是自家张量表里最后一块（索引 nblk2-1 必在界内），语义完全一致。
    """
    key = (nblk2, str(device))
    t = _QSLI_TB.get(key)
    if t is None:
        t = torch.cat([torch.tensor([nblk2 - 1], dtype=torch.int32, device=device),
                       torch.arange(1, nblk2 - 1, dtype=torch.int32, device=device)])
        if len(_QSLI_TB) >= 8:
            _QSLI_TB.clear()
        _QSLI_TB[key] = t
    return t


def _qsli_slots(n_rows, rows_blk, nblk2, device, row_off=0, ratio=1, ring=0):
    """压缩位置 p → 槽位 = 块表[p//rows_blk]*rows_blk + p%rows_blk（命中时 p 从 row_off 起）。

    官方 `_prepare_indexer_slot_mapping` 里的 `combined_block_size = (self.block_size*ratio)
    // candidate_block_size`（128*ratio/cbs），对 ratio=1 就是 16 = 组合缓存每块行数 rows_blk。

    `ring>0`：组合缓存是**定长镜像环**（可选，`DS41M_QSLI_CAP=16384`；默认 0=按 L 开）。
    注意：QSLI 的**计算量**本来就是固定的（T=140 行 × 最多 2048 个候选块 =
    ≤16k 个候选 K）；这里改的是 QSLI 寻址的 **K 镜像空间**——候选块号是全局的
    （0..L/8-1），按 L 开镜像时池/块表随 L 变；按环长开则取模复用固定块表。
    绝对压缩行先对环长取模再查块表 ⇒ 池形状、块表宽度、槽位值域不随 L/命中率变。
    """
    ring = int(ring or 0)
    if _CORE.dg_on():
        def _build(p):
            return _qsli_slots_build(int(n_rows), int(rows_blk), int(nblk2), device,
                                     int(p) // max(1, int(ratio)), ring=ring)
        return _CORE.dg_meta(
            ("qsli_slots_step", int(n_rows), int(rows_blk), int(nblk2), ring, str(device)),
            lambda: _build(row_off), lambda b, p: b.copy_(_build(p)))
    return _CORE._meta(("qsli_slots", int(n_rows), int(rows_blk), int(nblk2), int(row_off),
                        ring, str(device)),
                       lambda: _qsli_slots_build(int(n_rows), int(rows_blk), int(nblk2), device,
                                                 int(row_off), ring=ring))


def _qsli_slots_build(n_rows, rows_blk, nblk2, device, row_off, ring=0):
    idx = torch.arange(n_rows, dtype=torch.int64, device=device) + int(row_off)
    if int(ring or 0) > 0:
        idx = idx % int(ring)
    tb = _qsli_table(nblk2, device)
    return (tb[idx // rows_blk].long() * rows_blk + idx % rows_blk)


# ─────────────────────────── ③ qsli（候选层专用）───────────────────────────
def _qsli(st, ctx, w):
    """CP 分块包装：同 `_qli_cp`，本地 head/tail 两块各跑一次（候选块逐行切）。

    组合缓存（`_idx_qsli_*`）由源层（层 20）的 `idx_k_cache` 用全长 latent 写好，
    reader 的块表宽度按全局 seq_k 算 → 分块只改 query 行归属，不改 K 侧几何。
    """
    _cp = _CORE._cp_of(ctx)
    if _cp is None or ctx.get("_cp_full"):
        return _qsli_pk(st, ctx, w)     # 2026-10-26：varlen 是唯一口径
    return _qsli_cp(st, ctx, w, _cp)


def _qsli_cp(st, ctx, w, cp):
    """CP 的 QSLI：**本地 q 直接查 full 组合缓存**（DeepInfer 同口径；按 head/tail 块）。

    候选块 `_li_cand_r1` 是逐行张量（层 20 的 QLI 产物，本地行序）→ 按块切片即可，
    """
    ratio = max(1, int(ctx["ratio"]))
    topk = int(ctx["index_topk"])
    dev = st["idx_q"].device
    Tloc = st["idx_q"].shape[0]
    # ★ 2026-10-28（P0-5，已否决）：双块合并（一次厂商调用）在 8k@0.90 CP2 上设备挂死
    #   （rank0 卡死、rc=124；见 NOTES/2026-10-28-tail-bubble-host-opt-plan.md「否定结果」）。
    #   代码已删除，勿再尝试同一姿势（共享块表 repeat + batch=2）。
    hold = _CORE._hold(st)
    ci_full = hold.get("_li_cand_r1")
    cl_full = hold.get("_li_candlen_r1")
    if os.environ.get("DS41M_CAND_DBG"):
        # ★ 2026-10-31（主控）：候选池是不是真的存在？QSLI 没候选会退回"全量 K"，
        #   数值可能仍对但性能与官方不同口径 —— 必须可见。
        print(f"  [cand-dbg] qsli_cp idx={ctx.get('idx')} ratio={ratio} "
              f"cand={'None' if ci_full is None else tuple(ci_full.shape)} "
              f"len={'None' if cl_full is None else tuple(cl_full.shape)}", flush=True)
    base = dict(ctx)
    # ★ 压缩侧用**原序列长**（compact 段 = L）；块起点要加 pos_offset 变绝对位置
    _seq_cp = int(getattr(cp, "seq_len", 0) or cp.plan.total_q)
    _pofs_cp = int(getattr(cp, "pos_offset", 0) or 0)
    base["L_full"] = _seq_cp
    base["cu_seqlens"] = None
    # ★ 2026-10-28（命中 + CP）：逻辑长用 L（plain 时 _seq_cp==tot，逐位不变）
    base["seq_lens"] = [int(_seq_cp)]
    base["row_lens"] = None
    base["n_seqs"] = 1
    base["_cp_chunk"] = 1          # 分块是正常 prefill 子段，禁 DSpark 小 T 特判
    base.pop("cp_positions", None)
    base.pop("cp_total_q", None)
    # ★ 2026-10-28（P0-2）：同 `_qli_cp` 那处——缓存复用 + **带索引源层后缀** + 显式复位。
    if _CORE._HOST_OPT:
        # key 带 ratio（hold 就是按 `_cmp_sparse_r{ratio}` 分桶的）+ 索引源层后缀。
        sp = _CORE._meta(("cmp_sp_cp", int(Tloc), int(topk), int(ratio),
                          _CORE._sfx(ctx, "idx"), str(dev)),
                         lambda: torch.full((Tloc, 1, topk), -1, dtype=torch.int32, device=dev))
        sp.fill_(-1)
        tl = _CORE._meta(("cmp_tl_cp", int(Tloc), int(ratio),
                          _CORE._sfx(ctx, "idx"), str(dev)),
                         lambda: torch.zeros((Tloc, 1), dtype=torch.int32, device=dev))
        tl.zero_()
    else:
        sp = torch.full((Tloc, 1, topk), -1, dtype=torch.int32, device=dev)
        tl = torch.zeros((Tloc, 1), dtype=torch.int32, device=dev)
    # ★ 2026-10-28（P0-3）：idx_w 的 `contiguous+view+float()` 每层只做一次
    #   （原来每个 block 各做一次；CP2 两块 → 每层省 1 次 cast 拷贝）。
    _w_flat = (st["idx_w"].contiguous().view(-1, N_HEADS).float()
               if _CORE._HOST_OPT else None)
    try:
        for ls, gs, nv in cp.plan.local_blocks():
            if nv <= 0:
                continue
            st2 = st.copy() if _CORE._HOST_OPT else dict(st)
            st2["idx_q"] = st["idx_q"][ls:ls + nv].contiguous()
            st2["idx_descale_q"] = st["idx_descale_q"][ls:ls + nv].contiguous()
            st2["idx_w"] = (_w_flat[ls:ls + nv] if _w_flat is not None
                            else st["idx_w"][ls:ls + nv].contiguous())
            if ci_full is not None:
                hold["_li_cand_r1"] = ci_full[ls:ls + nv].contiguous()
                hold["_li_candlen_r1"] = cl_full[ls:ls + nv].contiguous()
            _qsli_chunk(st2, dict(base, pos_offset=int(gs) + _pofs_cp, T=int(nv)), w)
            sp[ls:ls + nv] = st2["cmp_sparse_indices"].reshape(nv, 1, -1).to(torch.int32)
            tl[ls:ls + nv] = st2["cmp_topk_length"].reshape(nv, 1).to(torch.int32)
    finally:
        if ci_full is not None:
            hold["_li_cand_r1"] = ci_full
            hold["_li_candlen_r1"] = cl_full
    st["idx_sparse"] = sp
    st["cmp_sparse_indices"] = sp
    st["cmp_topk_length"] = tl
    hold[f"_cmp_sparse_r{ratio}" + _CORE._sfx(ctx, "idx")] = sp
    hold[f"_cmp_topklen_r{ratio}" + _CORE._sfx(ctx, "idx")] = tl
    return sp


def _qsli_dg_chunk(st, ctx, w):
    """QSLI 的 DeepInfer 内核版：组合缓存 → dense K，再走官方稀疏打分 + topk。

    对齐关系（DeepInfer `v4_modules/index_attn.py`）：
        vendor `quant_sparse_lightning_indexer`（一个融合 aclnn）
        → `deep_gemm.get_sparse_mqa_logits_metadata`
        + `deep_gemm.fp8_fp4_sparse_mqa_logits`
        + `deep_select.interface.topk`（index_attn.py:273/317 + 365 的 topk）

    输入口径与 vendor 版**完全一致**（源层写的组合缓存 `_idx_qsli_r*` + 源层 qli 的候选块
    `_li_cand_r1/_li_candlen_r1`），只换最后一步的打分/topk 内核。

    K 的取法：组合缓存每行 = cbs 个连续压缩键（64 B/键）+ 它们的 scale（4 B/键），
    槽位由 `_qsli_slots` 给出（写侧 prolog_k 一个压缩位置一行）⇒ **每个压缩位置的键就是
    该行窗口的第 0 个**，所以 dense K = `sel[:, 0, :]`（无需 gather/重排）。
    """
    import deep_gemm_di as _dgd
    from deep_select import interface as _dsi
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dg_candidate as _dgc

    q = st.get("idx_q")
    if q is None:
        raise RuntimeError("qsli(dg) 需要 st['idx_q']（本层要先跑 idx_prolog）")
    T = q.shape[0]
    ratio = max(1, int(ctx["ratio"]))
    topk = int(ctx["index_topk"])
    cbs = int(ctx.get("candidate_block_size", CAND_BLOCK))
    dev = q.device
    hold = _CORE._hold(st)
    cand_i = hold.get("_li_cand_r1")
    cand_l = hold.get("_li_candlen_r1")
    if cand_i is None or cand_l is None:
        raise RuntimeError("qsli(dg) 需要候选块 —— 请先跑候选源层（层 %s）的 qli"
                           % ctx.get("candidate_source_layer", 20))
    kc = hold.get(f"_idx_qsli_r{ratio}")
    if kc is None:
        raise RuntimeError(f"qsli(dg) 需要组合索引器缓存（源层的 _idx_qsli_r{ratio}）")
    rows_blk, nblk2 = hold[f"_idx_qslis_r{ratio}"]
    qcap = int(hold.get(f"_idx_qsli_cap_r{ratio}", 0) or 0)

    cu = batch_layout.cu_of(ctx) or [0, int(T)]
    nseq = len(cu) - 1
    # ★ 2026-11-01（P1-2，缺陷 G3）：packed 多请求也走 dg 内核。
    #   口径与 `_qli_dg` 的 b>1 分支逐项一致：
    #     · dense K 行轴 = 各序列压缩行**拼接**（`_cu_k`），按各自块表映射；
    #     · `ks/ke` 用**绝对**行号（`_dg_build_ks_ke` 内部 base 累加）；
    #     · 块号保持**绝对**块号（源层写的就是绝对块号）；
    #     · topk 还原成绝对压缩行后，按请求减 `_cu_k[s]` 局部化 —— mqsmla 的
    #       `cmp_sparse_indices` 要的是**请求内局部**行号（同 `_qli_dg`:3005-3018）。
    raw_lens = [int(x) for x in (ctx.get("seq_lens")
                                 or [cu[b + 1] - cu[b] for b in range(nseq)])]
    L_full = int(ctx.get("L_full", 0) or 0)
    off = int(ctx.get("pos_offset", 0) or 0)
    span_end = off + int(T)
    if L_full > 0 and span_end < L_full:
        kv_lens = [span_end for _ in raw_lens]
    else:
        kv_lens = [(L_full if L_full > 0 else int(x)) for x in raw_lens]
    # 每序列可见压缩行数起点（nseq==1 时 _cu_k[-1] 与旧 `seq_k` 表达式等价）
    _cu_k = [0]
    for _sk in kv_lens:
        _cu_k.append(_cu_k[-1] + max(1, int(_sk) // ratio))
    _nk = int(_cu_k[-1])

    # ── 可见压缩行数（与 vendor 同口径：span_end//ratio 优先，不从 L_full 推）──
    lat = st.get("cmp_latent")
    seq_k = lat.shape[0] if lat is not None else int(T) // ratio
    _spanK = span_end // ratio
    if _spanK > seq_k:
        seq_k = _spanK
    if nseq > 1:
        seq_k = max(int(seq_k), _nk)
    if seq_k <= 0:
        raise RuntimeError(f"qsli(dg): seq_k={seq_k}（off={off} T={T} ratio={ratio}）")
    # b>1 的 K 轴总量（= 各序列压缩行拼接长度）；b=1 时与 `seq_k` 同值
    _nk_eff = int(_nk) if nseq > 1 else int(seq_k)

    # ── dense K：**优先用共享的 split index-K 池**（DeepInfer `use_shared_index_k` 同口径）──
    #   ★★ 2026-10-31（A/B + 探针实测后改）：组合缓存是 **vendor 融合算子专用的 repack**，
    #   它的行字节与 split 池**对不上**（探针在源层拿已校验的 split K 当真值，修正坐标后
    #   8 个窗口位置仍 0 命中）⇒ 不能拿它喂 deep_gemm。DeepInfer 的候选层本来就是
    #   `fp8_fp4_sparse_mqa_logits(q, kv, ...)`，kv 与 dense 路径**同一份**（共享 index-K）。
    #   所以这里与 `_qli_dg` 走完全相同的 K 获取逻辑；组合缓存只留给 vendor 算子。
    k_src = None
    # 块表是 nblk 的纯函数：池在、块表缺时安全重建（同 `_qli_dg_cp` 的兜底）
    if f"_idx_k_r{ratio}" in hold and f"_idx_k_bt_r{ratio}" not in hold:
        _nf = max(int(L_full) // ratio + 1, 1)
        hold[f"_idx_k_bt_r{ratio}"] = batch_layout.rot_block_table(
            max(1, -(-(_nf + 1) // ROWS_PER_BLOCK)), 1, dev)
    if (f"_idx_k_r{ratio}" in hold and f"_idx_ks_r{ratio}" in hold
            and f"_idx_k_bt_r{ratio}" in hold):
        k_src = (hold[f"_idx_k_r{ratio}"], hold[f"_idx_ks_r{ratio}"],
                 hold[f"_idx_k_bt_r{ratio}"])
    if k_src is None:
        # 源层之后池被释放时，只能从本层 latent 重建（CP 下同 `_idx_k_rows_cp`）
        lat = st.get("cmp_latent")
        if lat is None:
            raise RuntimeError(
                f"qsli(dg) 既没有共享 index-K 池（_idx_k_r{ratio}）也没有 cmp_latent ⇒ "
                f"无法构造 dense K；请显式 DS41M_QSLI_KERNEL=vendor")
        _k, _dk, _bt = idx_k_cache(st, ctx, w, lat.shape[0], dev)
        k_src = (_k, _dk, _bt)
    _n_full = (_nk_eff if nseq > 1
               else (int(L_full) // ratio or int(seq_k)))
    k_dense, k_sf = _dg_paged_to_dense(k_src[0], k_src[1], k_src[2],
                                       int(seq_k), ratio, dev, n_full=_n_full,
                                       cu_k=(_cu_k if nseq > 1 else None))
    k_dense = k_dense.view(torch.int8)
    k_sf = k_sf.reshape(k_dense.shape[0], 4).contiguous().view(torch.int16)

    # ── ks/ke（逐行因果，绝对压缩行号；与 `_qli_dg` 同一公式）──
    ks, ke = _dg_build_ks_ke(cu, raw_lens, kv_lens, ratio, dev, row_off=0)

    q_u8 = q.contiguous().view(torch.int8)
    q_sf = st["idx_descale_q"].reshape(int(T) * N_HEADS, 4).contiguous().view(torch.int16)
    wq = st["idx_w"].contiguous().to(torch.bfloat16)

    # ── 候选块（源层产物，已是**绝对块号**）+ 每行有效块数 ──
    # ★★ 2026-10-31（对齐 DeepInfer `_get_candidate_inputs`，index_attn.py:234-243）：
    #   候选块清洗（where/clamp）在**候选组内**是同一个纯函数结果（源层写、消费层只读），
    #   按 forward 缓存一次即可；以前每个消费层都重算一遍 → 每层多 2~3 个 kernel。
    _cap = int(cand_i.numel()) // max(1, int(T))
    # ★ P1-2：块号是**绝对**块号 ⇒ 上界用拼接后的 K 轴总量（b=1 时 _nk_eff==seq_k）
    _nb = max(1, -(-int(_nk_eff) // cbs))
    _ckey = ("qsli_cand", int(ratio), int(T), int(cbs), _cap, int(_nk_eff),
             int(cand_i.data_ptr()), int(cand_l.data_ptr()))

    def _cand_build():
        _b = cand_i.reshape(int(T), -1).to(torch.int32)
        _v = cand_l.reshape(int(T)).to(torch.int32).clamp(1, _cap)
        # 官方约束：块号必须落在可见 K 内（`i*B + ks%B < end`）
        _b = torch.where(_b >= 0, _b, torch.zeros_like(_b)).clamp(max=_nb - 1)
        return _b, _v

    blocks, vcnt = _CORE._meta_fwd(_ckey, _cand_build)

    # ★ 官方 `cross_layer_runtime.candidate_mqa_lens`（index_attn.py:238-243）：
    #   候选组内只算一次并挂 runtime；以前每个消费层各自 `compact_lens` 一遍。
    clens = _CORE._meta_fwd(
        ("qsli_cand_lens", int(ratio), int(T), int(cbs), _cap, int(_nk_eff),
         int(cand_i.data_ptr()), int(cand_l.data_ptr()), int(ke.data_ptr())),
        lambda: _dgc.compact_lens(blocks, ke, int(cbs), vcnt))

    if not _dgd._ensure_jit():
        raise RuntimeError("deep_gemm_di JIT 初始化失败（_C.init 返回非 0）")
    # ★★ 官方 `_get_sparse_prefill_metadata`（index_attn.py:246-271）：sparse schedule
    #   按 (phase, ratio, chunk.start, chunk.stop, block_size, 块数, q_dtype) 缓存在
    #   `cross_layer_runtime.candidate_mqa_schedule_metadata` 里**跨层复用**（同候选组的
    #   所有消费层共用一份）。以前每层都调一次 `get_sparse_mqa_logits_metadata`。
    #   键里额外带输入张量 data_ptr，避免缓冲复用后串味。
    meta = _CORE._meta_fwd(
        ("qsli_sched", int(ratio), 0, int(T), int(cbs), blocks.shape[1],
         int(_nk_eff), str(q_u8.dtype), int(cand_i.data_ptr()),
         int(ks.data_ptr()), int(ke.data_ptr())),
        lambda: _dgd.get_sparse_mqa_logits_metadata(
            ks, ke, int(_nk_eff), blocks, q_u8.dtype, int(cbs)))
    p_sum = _dgd.fp8_fp4_sparse_mqa_logits(
        (q_u8, q_sf), (k_dense, k_sf), wq, meta, blocks.shape[1], int(cbs))
    if os.environ.get("DS41M_QSLI_DG_SHA"):
        import hashlib as _hl
        print(f"  [qsli-dg] idx={ctx.get('idx')} T={T} seq_k={seq_k} rate={ratio} "
              f"cbs={cbs} blocks={tuple(blocks.shape)} p_sum={tuple(p_sum.shape)} "
              f"ke_last={int(ke[-1])} ksha={_hl.sha1(k_dense.cpu().numpy().tobytes()).hexdigest()[:12]}",
              flush=True)

    # ★★ 官方 `_topk_prefill`（index_attn.py:347-372）：q 按 `compute_chunk_size`
    #   （index_attn.py:117）分段，每段单独取 sparse schedule → logits → topk → 还原。
    #   `T <= chunk` 时走**单块**整段路径（与旧实现逐位、逐 kernel 完全相同）；
    #   只有超长（> compute_chunk_size）才分段。`DS41M_DG_QCHUNK` 可强制小 chunk 做
    #   等价自检（配合下方 `DS41M_DG_QCHUNK_CHK`）。
    _qchunk = int(os.environ.get("DS41M_DG_QCHUNK", "0") or 0) or min(
        8192, int(10 * (2 ** 30)) // max(1, int(_nk_eff) * 2) // 8 * 8)
    _qchunk = max(1, int(_qchunk))
    # ★ 2026-11-02 诊断（默认关）：确认"同一层 mqa_logits 调几次"的账
    if os.environ.get("DS41M_QLI_TRACE", "").strip().lower() in ("1", "on", "true", "yes") \
            and os.environ.get("RANK", "0") == "0" and int(ctx.get("idx", -1)) >= 0:
        _nseg = -(-int(T) // int(_qchunk))
        print(f"  [qli-trace] L{ctx.get('idx')} T={int(T)} maxk={int(_nk_eff)} "
              f"qchunk={int(_qchunk)} segs={_nseg} nseq={int(nseq)}", flush=True)
    out_idx = torch.full((int(T), topk), -1, dtype=torch.int32, device=dev)
    if int(T) <= _qchunk:
        _dsi.topk(p_sum, topk, sorted=False, indices_type=torch.int32, sorted_index=True,
                  output_idx=out_idx, end=clens, idx_oob_fill_value=-1, return_value=False)
        out_idx = _dgc.restore_indices(out_idx, blocks, int(cbs), 0)
    else:
        _qsf_n = max(1, q_sf.shape[0] // int(T))
        for _qs in range(0, int(T), _qchunk):
            _qe = min(_qs + _qchunk, int(T))
            _sl = slice(_qs, _qe)
            _mt = _CORE._meta_fwd(
                ("qsli_sched", int(ratio), _qs, _qe, int(cbs), blocks.shape[1],
                 int(_nk_eff), str(q_u8.dtype), int(cand_i.data_ptr()),
                 int(ks.data_ptr()), int(ke.data_ptr())),
                lambda _a=_qs, _b=_qe: _dgd.get_sparse_mqa_logits_metadata(
                    ks[_a:_b], ke[_a:_b], int(_nk_eff), blocks[_a:_b],
                    q_u8.dtype, int(cbs)))
            _ps = _dgd.fp8_fp4_sparse_mqa_logits(
                (q_u8[_sl], q_sf[_qs * _qsf_n:_qe * _qsf_n]), (k_dense, k_sf),
                wq[_sl], _mt, blocks.shape[1], int(cbs))
            _oi = torch.full((_qe - _qs, topk), -1, dtype=torch.int32, device=dev)
            _dsi.topk(_ps, topk, sorted=False, indices_type=torch.int32,
                      sorted_index=True, output_idx=_oi, end=clens[_sl],
                      idx_oob_fill_value=-1, return_value=False)
            out_idx[_qs:_qe] = _dgc.restore_indices(_oi, blocks[_sl], int(cbs), 0)
            del _ps, _oi
        # 等价自检：同一进程里再算一遍整段，与分段结果逐位比较（仅调试用）
        if os.environ.get("DS41M_DG_QCHUNK_CHK"):
            _ref = torch.full((int(T), topk), -1, dtype=torch.int32, device=dev)
            _dsi.topk(p_sum, topk, sorted=False, indices_type=torch.int32,
                      sorted_index=True, output_idx=_ref, end=clens,
                      idx_oob_fill_value=-1, return_value=False)
            _ref = _dgc.restore_indices(_ref, blocks, int(cbs), 0)
            _eq = bool(torch.equal(_ref, out_idx))
            print(f"  [qchunk-chk] idx={ctx.get('idx')} T={int(T)} chunk={_qchunk} "
                  f"分段结果==整段结果: {_eq}", flush=True)
            if not _eq:
                raise RuntimeError(f"[qchunk-chk] 分段与整段不一致 idx={ctx.get('idx')}")

    # ★ P1-2（b>1）：topk 还原出的是**绝对**压缩行号 ⇒ 按请求减本序列起点，得到
    #   mqsmla 要的**请求内局部**行号（-1 保持 -1）。与 `_qli_dg`:3005-3018 同一口径；
    #   b=1 时 `_cu_k[0]==0`，整段不动 ⇒ 零回归。
    if nseq > 1 and os.environ.get("DS41M_BIDX_SPACE", "local").strip().lower() != "abs":
        for _i in range(nseq):
            _a, _b = int(cu[_i]), int(cu[_i + 1])
            _bn = int(_cu_k[_i])
            if _bn:
                _s = out_idx[_a:_b]
                _m = _s >= 0
                _s.sub_(_bn)
                _s.masked_fill_(~_m, -1)

    sparse = out_idx.view(int(T), 1, topk)
    # ★ 自洽验证（DS41M_QS_SPCHK）：同一进程里再算一遍**全量 dense** top-k，与 sparse 比。
    #   两者应当一致（候选块覆盖全部 K 时）⇒ 证明 dg 稀疏路径本身正确，
    #   vendor 那 19/512 的差异来自它的 repack（组合缓存），不是我们的错。
    if os.environ.get("DS41M_QS_SPCHK"):
        try:
            _pf = _dgd.fp8_fp4_mqa_logits(
                (q_u8, q_sf), (k_dense, k_sf), wq, ks, ke,
                max_seqlen_k=int(_nk_eff))
            _di = torch.full((int(T), topk), -1, dtype=torch.int32, device=dev)
            _dsi.topk(_pf, topk, sorted=False, indices_type=torch.int32,
                      sorted_index=True, output_idx=_di,
                      output_idx_offset=ks, end=ke.contiguous(),
                      idx_oob_fill_value=-1, return_value=False)
            _s, _d = 0, 0
            for _i in range(int(T)):
                _x = set(int(v) for v in out_idx[_i] if v >= 0)
                _y = set(int(v) for v in _di[_i] if v >= 0)
                if _x == _y:
                    _s += 1
                else:
                    _d += 1
            _x0 = set(int(v) for v in out_idx[0] if v >= 0)
            _y0 = set(int(v) for v in _di[0] if v >= 0)
            print(f"  [qs-spchk] idx={ctx.get('idx')} T={int(T)} 全行相同={_s} 不同={_d} "
                  f"row0 交集={len(_x0 & _y0)}/{topk}", flush=True)
        except Exception as _e:                    # noqa: BLE001
            print(f"  [qs-spchk] 失败 {type(_e).__name__}: {_e}", flush=True)
    # 诊断（与 vendor 版同一开关/同一层）：A/B 逐位对拍用
    if (os.environ.get("DS41M_QS_DUMP")
            and int(ctx.get("idx", -1)) == int(os.environ.get("DS41M_QS_DUMP_LAYER", "24"))):
        _dd = os.environ["DS41M_QS_DUMP"]
        os.makedirs(_dd, exist_ok=True)
        torch.save({"kind": "seq", "kernel": "dg", "sparse": sparse.detach().cpu(),
                    "cand_i": cand_i.detach().cpu(), "cand_l": cand_l.detach().cpu(),
                    "cand_used": blocks.detach().cpu(), "clens": clens.detach().cpu(),
                    "ke": ke.detach().cpu(), "kc_shape": [int(x) for x in kc.shape]},
                   os.path.join(_dd, f"qsli_dg_{os.environ.get('RANK', '0')}_seq.pt"))
    st["idx_sparse"] = sparse
    st["cmp_sparse_indices"] = sparse
    st["cmp_topk_length"] = (sparse >= 0).sum(-1).to(torch.int32)
    hold[f"_cmp_sparse_r{ratio}" + _CORE._sfx(ctx, "idx")] = sparse
    hold[f"_cmp_topklen_r{ratio}" + _CORE._sfx(ctx, "idx")] = st["cmp_topk_length"]
    if os.environ.get("DS41M_DEBUG"):
        print(f"      [qsli-dg] ratio={ratio} 组合缓存{tuple(kc.shape)} "
              f"候选{tuple(cand_i.shape)} → sparse{tuple(sparse.shape)} "
              f"有效={int((sparse >= 0).sum())}", flush=True)
    return sparse


def _qsli_chunk(st, ctx, w):
    """`quant_sparse_lightning_indexer`：吃源层写的**组合缓存** + 源层 qli 产出的候选块。

    官方 indexer.py:395-446。与 qli 的差别：
      · k 用组合缓存（每行 = cbs 个键 + 它们的 scale；4D 且 shape[-2]==1 → 官方会 squeeze）
      · 多两个输入 candidate_block_indices / candidate_block_length（来自**源层** qli）
      · seqused_k = compressed_seq_lens[ratio]（不含余数）；cmp_residual_k = None
      · metadata = kernel_metadata[f"c{ratio}a_qsli_metadata"]，其 candidate_block_length =
          floor((pos + ratio*cbs)/(ratio*cbs)).clamp(1, candidate_topk)（modeling_deepseek.py:3094）
      · block table 用 `c{ratio}a_qsli_combined_kv`（组合缓存的块表，块 0 空）
    """
    # ★ 2026-10-31：内核分派。`DS41M_QSLI_KERNEL=dg` 走 DeepInfer 稀疏打分 +
    #   DeepSelect topk；默认 vendor（dg 版数值未对齐，见 WORKPLAN E5）。
    #   ★ 注：**与入图无关**——PERF_RULES §0（2026-10-27）已永久删除图路径，
    #     `dg_on()` 现在恒为 False（`dg_begin()` 无调用者）；DeepInfer 的 CUDA graph
    #     也只在 decode（`engine_type != PROMPT`），prefill 永远 eager。
    # ★★ 2026-10-31 定稿：**默认 dg**。判据＝自洽验证 `DS41M_QS_SPCHK`：
    #   同一进程里 dg 的「候选内稀疏 top-k」与「全量 dense top-k」逐行**完全一致**
    #   （L24 T=140：全行相同 140/140、row0 512/512）⇒ 稀疏路径本身正确。
    #   dg 的 K 与 dense 路径**同一份**（共享 split index-K 池）= DeepInfer
    #   `use_shared_index_k` 口径；vendor 融合算子吃的是**组合缓存 repack** 的 K，
    #   两者在近似并列处有 ~4% 差（493/512），那是 repack 的口径差，不是我们的错。
    #   回退：`DS41M_QSLI_KERNEL=vendor`（仅 A/B/复现）。
    _qk = os.environ.get("DS41M_QSLI_KERNEL", "dg").strip().lower()
    if _qk == "dg":
        return _qsli_dg_chunk(st, ctx, w)
    d = torch.ops.cann_ops_transformer.ds41
    q = st.get("idx_q")
    if q is None:
        raise RuntimeError("qsli 需要 st['idx_q']（本层要先跑 idx_prolog）")
    T = q.shape[0]
    ratio = max(1, int(ctx["ratio"]))
    topk = int(ctx["index_topk"])
    cbs = int(ctx.get("candidate_block_size", CAND_BLOCK))
    cand_topk = int(ctx.get("candidate_topk_blocks", CAND_TOPK))
    device = q.device
    hold = _CORE._hold(st)
    cand_i = hold.get("_li_cand_r1")
    cand_l = hold.get("_li_candlen_r1")
    if cand_i is None:
        raise RuntimeError("qsli 需要候选块 —— 请先跑候选源层（层 %s）的 qli"
                           % ctx.get("candidate_source_layer", 20))
    kc = hold.get(f"_idx_qsli_r{ratio}")
    if kc is None:
        raise RuntimeError(f"qsli 需要组合索引器缓存（源层的 _idx_qsli_r{ratio}）")
    rows_blk, nblk2 = hold[f"_idx_qslis_r{ratio}"]
    # ★ 定长环：候选位置也要对环长取模（reader 的 page = pos//rows_blk 必须落在
    #   固定块表宽度内）；-1/无效值保持原样（由 candidate_block_length 屏蔽）。
    _qcap = int(hold.get(f"_idx_qsli_cap_r{ratio}", 0) or 0)
    if _qcap > 0:
        cand_i = torch.where(cand_i >= 0, cand_i % _qcap, cand_i)
    lat = st.get("cmp_latent")
    seq_k = lat.shape[0] if lat is not None else T // ratio
    # ★★ 同 `_qli` 上面那条修（病C 根因）：可见长度 = **(off+T)//ratio**，
    #   不是 L//ratio ✗ —— stage A 时后者会多声明 256 行零行 ⇒ 读零行 ⇒ 0/0 = NaN ✗。
    #   `DS41M_SEQK_LFULL=1` 恢复旧（错）值供复现 ✓。
    _L = int(ctx.get("L_full", 0) or 0)
    _offK = int(ctx.get("pos_offset", 0))
    if os.environ.get("DS41M_SEQK_LFULL", "").strip().lower() in ("1", "on", "true", "yes"):
        _spanK = _L // ratio
    else:
        _spanK = (_offK + T) // ratio
    if _spanK > seq_k:
        seq_k = _spanK
    if os.environ.get("DS41M_SEQK_OVERRIDE"):
        seq_k = int(os.environ["DS41M_SEQK_OVERRIDE"])

    # candidate_block_length：官方用位置推（与 qli/qslis 的"位置驱动"口径一致）；
    # 命中时用的是**绝对位置**（query 在 L-R .. L-1 上），否则候选块会全部算成前缀里的块
    _off = int(ctx.get("pos_offset", 0))

    def _qsli_cl_build(p):
        return torch.div(torch.arange(T, dtype=torch.int64, device=device) + int(p)
                         + ratio * cbs, ratio * cbs, rounding_mode="floor") \
            .clamp_min(1).clamp_max(cand_topk).to(torch.int32).view(-1, 1)

    # ★ 逐行因果（同 `_qli` v2）：以前 seqused_k 是整段一个标量 (off+T)//ratio，
    #   row0 能看到未来的压缩行 ⇒ 候选被未来行挤占、归一化随草稿值变化（T=6 verify
    #   fake id 0/5 的 row0 仍差 ~0.5–1.0）。改成每行一个 batch：
    #   cu=[0..T]、seqused_k[i]=(off+i+1)//ratio、block_table 重复 T 行。
    _per_row = (1 < int(T) <= 32 and _offK > 0
                and os.environ.get("DS41M_QSLI_PER_ROW", "1") != "0"
                and not ctx.get("_cp_chunk"))

    def _qsli_sk_rows(p):
        return ((torch.arange(int(T), dtype=torch.int64, device=device) + int(p) + 1)
                // ratio).to(torch.int32)

    if _CORE.dg_on():
        cl = _CORE.dg_meta(
            ("qsli_cl_step", int(T), int(ratio), int(cbs), int(cand_topk), str(device)),
            lambda: _qsli_cl_build(_off), lambda b, p: b.copy_(_qsli_cl_build(p)))
        if _per_row:
            _cuq = _CORE.dg_const(("qsli_cu_rows", int(T)),
                                  list(range(int(T) + 1)), torch.int32, device)
            _seqk = _CORE.dg_meta(
                ("qsli_seqk_rows", int(T), int(ratio), str(device)),
                lambda: _qsli_sk_rows(_off), lambda b, p: b.copy_(_qsli_sk_rows(p)))
        else:
            _cuq = _CORE.dg_const(("qsli_cu", int(T)), [0, int(T)], torch.int32, device)
            _seqk = _CORE.dg_meta(
                ("qsli_seqk_step", int(T), int(ratio), str(device)),
                lambda: torch.tensor([seq_k], dtype=torch.int32, device=device),
                lambda b, p: b.fill_(max(lat.shape[0] if lat is not None else 0,
                                         (int(p) + int(T)) // max(1, int(ratio)))))
    else:
        cl = _CORE._meta(("qsli_cl", int(T), int(_off), int(ratio), int(cbs),
                          int(cand_topk), str(device)), lambda: _qsli_cl_build(_off))
        if _per_row:
            _cuq = _CORE._meta(
                ("cu_q_rows", int(T), str(device)),
                lambda: torch.tensor(list(range(int(T) + 1)), dtype=torch.int32, device=device))
            _seqk = _CORE._meta(
                ("qsli_seqk_rows", int(T), int(ratio), int(_off), str(device)),
                lambda: _qsli_sk_rows(_off))
        else:
            _cuq = _CORE._meta(("cu_q", int(T), str(device)),
                               lambda: torch.tensor([0, T], dtype=torch.int32, device=device))
            _seqk = _CORE._meta(("seq_k", int(seq_k), str(device)),
                                lambda: torch.tensor([seq_k], dtype=torch.int32, device=device))
    def _build_qsli_md():
        return d.quant_sparse_lightning_indexer_metadata(
            candidate_block_length=cl,
            cu_seqlens_q=_cuq,
            seqused_q=None,
            seqused_k=_seqk,
            cmp_residual_k=None,
            batch_size=(int(T) if _per_row else 1), max_seqlen_q=-1, max_seqlen_k=-1,
            num_heads_q=N_HEADS, num_heads_k=1, head_dim=HEAD_DIM, topk=topk,
            quant_mode=1, candidate_block_size=cbs, mask_mode=3, cmp_ratio=ratio,
            layout_q="TND", layout_k="PA_BBND")

    # ★ 2026-10-28（P0-3）：metadata 只由上面这些**已缓存**张量的几何决定
    #   （同一 forward 里 4 个索引源层的 T/off/ratio 完全相同）→ 进程内复用一次构建，
    #   省掉每层 2 个 AI_CPU metadata kernel 与 host 下发。图路径/小 T 不走缓存
    #   （图要静态地址；`_meta` 在几何变化时由 runner 清空）。
    if _QS_MD_CACHE and _CORE._HOST_OPT and int(T) > 32 and not _CORE.dg_on():
        md = _CORE._meta(("qsli_md", int(T), int(_off), int(seq_k), int(ratio), int(cbs),
                          int(cand_topk), int(topk), str(device)), _build_qsli_md)
    else:
        md = _build_qsli_md()

    q2 = q.view(-1, N_HEADS, PACKED_D)
    dq = st["idx_descale_q"].view(torch.uint8).view(
        q2.shape[0], N_HEADS, st["idx_descale_q"].shape[-2], st["idx_descale_q"].shape[-1])
    # 块表宽度（= 请求占用的块数；官方从调度器拿）。DS41M_QSLI_BT 可覆盖用于二分。
    # ★ 定长环（_idx_qsli_cap_r>0）时块表宽度必须固定 = 组合缓存块数：候选块索引
    #   是**最近一窗**的块号（candidate_topk_blocks×cbs 已覆盖），不再随 L/seq_k 长。
    _qcap = int(hold.get(f"_idx_qsli_cap_r{ratio}", 0) or 0)
    bt_w = int(os.environ.get("DS41M_QSLI_BT", str(nblk2)))
    if _qcap > 0:
        bt_w = int(nblk2)
    else:
        bt_w = max(bt_w, 1 + max(1, -(-seq_k // 128)))      # 至少覆盖 reader 会读到的页
        if _CORE.dg_on():
            # 图里块表是静态切片 → 宽度按 L_full 的最大 seq_k 一次开够（多出的块只是容量）
            _skmax = int(ctx.get("L_full", 0) or 0) // max(1, int(ctx["ratio"]))
            bt_w = max(bt_w, 1 + max(1, -(-_skmax // 128)))
    # ★ 与写侧同一张**旋转块表**（官方实测 [1025,1,2,…]）：reader 按
    #   (块表[b//16], b%16) 寻址，写侧按 块表[p//16]*16+p%16 落位 → 两侧自洽。
    block_table = _qsli_table(max(bt_w, nblk2), device)[:bt_w].unsqueeze(0)
    if _per_row:
        block_table = block_table.repeat(int(T), 1)
    # 诊断（DS41M_QS_DUMP=目录）：落 QSLI 入参快照，与 packed 实现对比（默认关）
    if (os.environ.get("DS41M_QS_DUMP")
            and int(ctx.get("idx", -1)) == int(os.environ.get("DS41M_QS_DUMP_LAYER", "24"))):
        _dd = os.environ["DS41M_QS_DUMP"]
        os.makedirs(_dd, exist_ok=True)
        torch.save({
            "kind": "seq", "T": int(T), "ratio": int(ratio), "off": int(_off),
            "cu": [0, int(T)], "lens": [int(seq_k)],
            "cl": cl.detach().cpu(), "cand_i": cand_i.detach().cpu(),
            "cand_l": cand_l.detach().cpu(), "bt": block_table.detach().cpu(),
            "kc_shape": [int(x) for x in kc.shape],
            "cand_min": int(cand_i.min()) if cand_i.numel() else -1,
            "cand_max": int(cand_i.max()) if cand_i.numel() else -1,
            "cand_len_max": int(cand_l.max()) if cand_l.numel() else -1,
        }, os.path.join(_dd, f"qsli_{os.environ.get('RANK', '0')}_seq.pt"))
    if os.environ.get("DS41M_IDX_TRACE") and os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"[idx-trace] qsli T={T} ratio={ratio} seq_k={seq_k} "
              f"cand={tuple(cand_i.shape)} candlen={tuple(cand_l.shape)} "
              f"cl={tuple(cl.shape)} bt={tuple(block_table.shape)} "
              f"per_row={_per_row} nblk2={nblk2} off={_off}", flush=True)
    out = _ifc.qsli((), dict(
        # k 必须 3D：(块数, 块行数, 544)；官方 indexer.py:410-412 也把 (..,1) 那维 squeeze 掉
        q=q2, k=kc.view(kc.shape[0], kc.shape[1], kc.shape[3]),
        w=st["idx_w"].contiguous().view(-1, N_HEADS).float(),
        descale_q=dq,
        candidate_block_indices=cand_i, candidate_block_length=cand_l,
        cu_seqlens_q=_cuq,
        seqused_q=None,
        seqused_k=_seqk,
        cmp_residual_k=None,
        block_table=block_table,
        output_idx_offset=None,
        metadata=md, topk=topk, candidate_block_size=cbs, quant_mode=1,
        max_seqlen_q=-1, mask_mode=3, cmp_ratio=ratio,
        layout_q="TND", layout_k="PA_BBND", return_value=False))
    sparse = out[0] if isinstance(out, (tuple, list)) else out
    # 诊断（DS41M_QS_DUMP=目录）：QSLI **输出**快照（环上限 A/B 时逐位对拍用）
    if (os.environ.get("DS41M_QS_DUMP")
            and int(ctx.get("idx", -1)) == int(os.environ.get("DS41M_QS_DUMP_LAYER", "24"))):
        _dd = os.environ["DS41M_QS_DUMP"]
        os.makedirs(_dd, exist_ok=True)
        torch.save({"kind": "seq", "sparse": sparse.detach().cpu(),
                    "cand_i": cand_i.detach().cpu(), "cand_l": cand_l.detach().cpu(),
                    "kc_shape": [int(x) for x in kc.shape]},
                   os.path.join(_dd, f"qsli_out_{os.environ.get('RANK', '0')}_seq.pt"))
    if (os.environ.get("DS41M_QLI_CHECK") and int(T) > 1 and _per_row
            and int(os.environ.get("RANK", "0")) == 0
            and not _CORE.dg_on()):
        try:
            _r0 = sparse[0].reshape(-1)
            _sk = _seqk.reshape(-1)
            _lim0 = int(_sk[0])
            _v = _r0[_r0 >= 0]
            _fut = int((_v >= _lim0).sum())
            print(f"  [qsli-check] L{ctx.get('idx')} T={int(T)} off={_offK} "
                  f"row0 seqk={_lim0} max={int(_v.max()) if _v.numel() else -1} "
                  f"nvalid={int(_v.numel())} future={_fut}", flush=True)
        except Exception as _e:                        # noqa: BLE001
            print(f"  [qsli-check] 失败 {type(_e).__name__}: {_e}", flush=True)
    st["idx_sparse"] = sparse
    st["cmp_sparse_indices"] = sparse
    st["cmp_topk_length"] = (sparse >= 0).sum(-1).to(torch.int32)
    hold[f"_cmp_sparse_r{ratio}" + _CORE._sfx(ctx, "idx")] = sparse
    hold[f"_cmp_topklen_r{ratio}" + _CORE._sfx(ctx, "idx")] = st["cmp_topk_length"]
    if os.environ.get("DS41M_DEBUG"):
        print(f"      [qsli] ratio={ratio} 组合缓存{tuple(kc.shape)} 候选{tuple(cand_i.shape)} "
              f"→ sparse{tuple(sparse.shape)} 有效={int((sparse >= 0).sum())}", flush=True)
    return sparse


# ─────────────────────────── 注册 ───────────────────────────
def register(OPS, SKIP, core, cfg=None):                # noqa: ARG001  （exec_ops 新签名带 cfg，旧的是 3 参）
    global _CORE
    _CORE = core
    _apply_spec(cfg)
    OPS["idx_prolog"] = _idx_prolog
    OPS["idx_k_write"] = _idx_k_write
    OPS["qli"] = _qli
    OPS["qsli"] = _qsli
    global _IDX
    _IDX = sys.modules[__name__]          # varlen 实现已合并进本文件（2026-10-26）
    for k in ("idx_prolog", "qli", "qsli"):
        SKIP.pop(k, None)


# ───────────── packed 实现（原 exec_indexer_packed.py，2026-10-26 合并）─────────────
_IDX = None   # register() 时指向本模块

def idx_k_cache(st, ctx, w, n_rows, device):
    """packed：**一个池 + 逐请求块表**（官方 `_prepare_indexer_slot_mapping` 口径）。

    逐 token 压缩位置 → (请求号, 请求内压缩行号)，slot = block_table[req, p//rpb]*rpb + p%rpb；
    pad 行 slot = -1（算子在负槽位跳过）。
    """
    hold = _CORE._hold(st)
    ratio = max(1, int(ctx["ratio"]))
    cu = batch_layout.cu_of(ctx)
    nseq = len(cu) - 1
    _qli_dbg(f"WRITE-packed idx={ctx.get('idx')} T={ctx.get('T')} n_rows={n_rows} "
             f"cu={tuple(cu)} pos={ctx.get('pos_offset')}")
    if os.environ.get("DS41M_QLI_DUMP"):
        _latp = st.get("cmp_latent")
        _lsha = None
        if torch.is_tensor(_latp):
            import hashlib as _h2
            _lsha = _h2.sha1(_latp.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]
        _IDXK_LOG.append(dict(
            who="packed:idx_k_cache", L=int(ctx.get("idx", -1) or -1),
            cu=tuple(int(x) for x in cu), T=int(ctx.get("T", -1) or -1),
            L_full=int(ctx.get("L_full", -1) or -1),
            pos=int(ctx.get("pos_offset", 0) or 0),
            n_rows=int(n_rows), lat_sha=_lsha))
    lens = [(cu[b + 1] - cu[b]) // ratio for b in range(nseq)]   # 本段写入行（R//ratio）
    # ★ packed-hit：**逻辑长度**（池覆盖 L//ratio）与写入行（R//ratio）分离；
    #   写入/读取的压缩行号都用绝对行号 off//ratio + 局部（plain: off=0 逐位不变）。
    _sl = ctx.get("seq_lens") or [cu[b + 1] - cu[b] for b in range(nseq)]
    lens_log = [max(1, int(x)) // ratio for x in _sl]
    # ★ 2026-10-21（逐序列几何）：各批不同 L/命中率 → 每请求各自的压缩起始行。
    _offs = [int(v) for v in (ctx.get("pos_offsets") or [])]
    if len(_offs) != nseq:
        _offs = [int(ctx.get("pos_offset", 0) or 0)] * nseq
    elif nseq == 1 and int(_offs[0]) != int(ctx.get("pos_offset", 0) or 0):
        # ★★ 2026-10-31（同 exec_ops._kv_compress_packed 的根因）：CP 逐块调用时
        #   `pos_offsets` 是从父 ctx 继承的**上一块**的值，会把本块的索引 K 写到错行。
        #   单序列时以标量 pos_offset 为准。
        _offs = [int(ctx.get("pos_offset", 0) or 0)]
    _off_cs = [int(o) // max(1, ratio) for o in _offs]
    off_c = max(_off_cs) if _off_cs else 0
    # 实验开关（默认沿用官方口径 = 压缩位置）：`DS41M_PK_IDXOFF=seq` 时**仅槽位基址**
    # 再除一次 ratio，与顺序实现 `_idx_k_cache` 的 `r0 = row0 // ratio`（二次除）对齐；
    # rope 位置仍用 `off_c`（官方 (row_off+i)*ratio 口径）。用于 A/B 定位。
    _seqslot = os.environ.get("DS41M_PK_IDXOFF", "").strip().lower() == "seq"
    _off_slots = [(c // max(1, ratio)) if _seqslot else c for c in _off_cs]
    n_real = int(sum(lens))
    n_pad = nseq if ratio > 1 else 0          # ratio=1 无 pad 行（官方同口径）
    if os.environ.get("DS41M_CMP_ARGDBG") and int(ctx.get("idx", -1)) == 20 and int(n_rows) == int(n_real + n_pad):
        print(f"  [idxk-write] n_rows={int(n_rows)} cu={ctx.get('cu_seqlens')} "
              f"sfx={_CORE._sfx(ctx,'idx')!r} off={ctx.get('pos_offset')} idx={ctx.get('idx')}", flush=True)
    if int(n_rows) != n_real + n_pad and os.environ.get("DS41M_CMP_ARGDBG"):
        import traceback as _tb
        _stk = " << ".join(f"{fr.filename.split('/')[-1]}:{fr.lineno}:{fr.name}"
                           for fr in _tb.extract_stack()[-7:-1])
        print(f"  [idxk-geom] n_rows={int(n_rows)} n_real={n_real} n_pad={n_pad} cu={cu} "
              f"seq_lens={ctx.get('seq_lens')} row_lens={ctx.get('row_lens')} "
              f"off={ctx.get('pos_offset')} L={ctx.get('L_full')} cpfull={ctx.get('_cp_full')} "
              f"idx={ctx.get('idx')} | {_stk}", flush=True)
    if int(n_rows) != n_real + n_pad:
        raise RuntimeError(f"packed latent {int(n_rows)} 行，期望 {n_real}+{n_pad}"
                           f"（各请求 s//ratio 打包 + pad 甩尾）")
    rpb = _IDX.ROWS_PER_BLOCK
    Pb = max(1, -(-(max(lens_log) + 2) // rpb))     # 池覆盖整条逻辑序列的压缩行
    kk, sk = f"_idx_k_r{ratio}", f"_idx_ks_r{ratio}"
    k, s = hold.get(kk), hold.get(sk)
    if k is None or k.shape[0] < 1 + nseq * Pb:
        k = torch.zeros(1 + nseq * Pb, rpb, 1, _IDX.PACKED_D, dtype=torch.uint8, device=device)
        s = torch.full((1 + nseq * Pb, rpb, 1, 2, 2), 127, dtype=torch.uint8, device=device)
        hold[kk], hold[sk] = k, s
    bt = batch_layout.rot_block_table(Pb, nseq, device)
    # ★ 2026-10-28：顺序版也要把块表存进 hold —— compact 段的 `_qli_pk` 复用 K 池时
    #   要求 `_idx_k_r / _idx_ks_r / _idx_k_bt_r` 三键齐全；以前只存前两个，导致
    #   截尾后误判"K 池没写过"→ 拿全长 latent + compact ctx 重写 → 形状校验失败。
    _CORE._hold(st)[f"_idx_k_bt_r{ratio}"] = bt
    # ★ 2026-10-21 packed hostbound：同 exec_ops_packed，`torch.tensor(lens, device=npu)`
    #   每次 qli 都是一次阻塞 H2D（等队列排空）→ 按值缓存。
    _lens_t = _CORE._meta(("idx_lens", tuple(lens), str(device)),
                          lambda: torch.tensor(lens, device=device))
    # ★ output_size 必须给：repeats 是设备张量 → 算子要回读算输出长度（一次流同步）。
    sid = torch.repeat_interleave(torch.arange(nseq, device=device), _lens_t,
                                  output_size=n_real)
    loc = torch.cat([torch.arange(x, device=device) for x in lens])
    _offc_t = _CORE._meta(("idx_offc", tuple(_off_cs), str(device)),
                          lambda: torch.tensor(_off_cs, dtype=torch.long, device=device))
    _offslot_t = _CORE._meta(("idx_offslot", tuple(_off_slots), str(device)),
                             lambda: torch.tensor(_off_slots, dtype=torch.long, device=device))
    loc_abs = loc + _offc_t[sid]   # rope/位置口径（官方）
    ci = torch.full((int(n_rows),), -1, dtype=torch.long, device=device)
    loc_slot = loc + _offslot_t[sid]   # 槽位口径（PK_IDXOFF=seq 时与顺序版对齐）
    ci[:n_real] = bt[sid, loc_slot // rpb].long() * rpb + loc_slot % rpb
    # rope：与压缩器输出同布局（真实压缩行打包在前 + pad 甩尾）
    # ★ 表行数 n 直接按几何算（= max(pos)+1），不再 `pos.max().item()` ——
    #   后者是设备 max + 同步，packed 下每次 qli 都要等一次队列排空。
    if ratio > 1:
        pos = torch.cat([torch.arange(l, dtype=torch.long, device=device) + _off_cs[i]
                         for i, l in enumerate(lens)]
                        + [torch.ones(nseq, dtype=torch.long, device=device)]) * ratio
        _n = max(max(lens[i] - 1 + _off_cs[i] for i in range(nseq)), 1) * ratio + 1
    else:
        pos, _ = batch_layout.local_positions(ctx, int(n_rows), device)
        # local_positions = 请求内行号 + pos_offset（逐序列偏移已内置）
        _n = max(1, max(int(lens[i]) + int(_offs[i]) for i in range(nseq)))
    cos, sin = _CORE._rope_at_pos(pos, ctx, device, n=_n)
    latent = st.get("cmp_latent")
    if latent is None:
        raise RuntimeError("索引 K 需要 st['cmp_latent']（本层要先跑 kv_compress）")
    d = torch.ops.cann_ops_transformer.ds41
    knw = (w.get("_packed") or {}).get("attn.indexer.k_norm.weight.f32")
    if knw is None:
        knw = w["attn.indexer.k_norm.weight"].to(torch.float32).contiguous()
    # ★★ 2026-10-31（主控修）：把静默回绕变成可定位异常。
    #   ⚠️ 旧版有两个错：①`-1` 是**合法 pad**（ratio>1 时每序列一行甩尾），不能判越界；
    #   ②消息里引用了本函数不存在的 `nblk`（packed 用的是 `Pb`）⇒ 真越界时抛 NameError，
    #   把 OOB 报告变成一条无关错误，还会让 qli 算子**整体失败**（runner 只记一笔 ⚠️ 继续跑）。
    # ★ 2026-11-01（交接 §6.2④ 清理）：这条守卫的 `int(_ci_ok.max())` 是**每次 qli 一次
    #   设备 max + 主机同步**（host-bound 热点）。真越界在设备侧本来就是 aicore trap，
    #   守卫只用于定位，默认关闭；排查时 `DS41M_IDX_OOB_CHECK=1` 打开。
    if os.environ.get("DS41M_IDX_OOB_CHECK", "").strip().lower() in ("1", "on", "true", "yes"):
        _ci_ok = ci[ci >= 0]
        _ci_max = int(_ci_ok.max()) if _ci_ok.numel() else -1
        if _ci_max >= k.shape[0] * rpb:
            raise RuntimeError(
                f"idx_k slot OOB idx={ctx.get('idx')} ci_max={_ci_max} "
                f"pool={tuple(k.shape)} blocks={k.shape[0] - 1} rpb={rpb} "
                f"Pb={Pb} nseq={nseq} n_rows={n_rows} n_real={n_real} n_pad={n_pad} "
                f"off={ctx.get('pos_offset')} L={ctx.get('L_full')}")
    _ifc.prolog_k((), dict(
        latent=latent.reshape(-1, latent.shape[-1]),
        wk=_IDX._nz("wk"),
        norm_weight=knw,
        rope_sin=sin.reshape(-1, _IDX.ROPE_DIM),
        rope_cos=cos.reshape(-1, _IDX.ROPE_DIM),
        k_cache=k,
        k_scale_cache=s.view(s.shape[0], rpb, 1, 4),
        cache_index=ci,
        norm_eps=_IDX.NORM_EPS,
        storage_mode=0,
        combined_block_size=-1,
    ))
    hold[f"_idx_bt_r{ratio}"] = bt
    if ctx.get("is_candidate"):
        cbs = int(ctx.get("candidate_block_size", _IDX.CAND_BLOCK))
        width = cbs * (_IDX.PACKED_D + 4)
        rows_blk = max(1, rpb // cbs)
        # ★ 2026-10-22 QSLI 定长环（与顺序版同口径）：QSLI 每次最多 gather
        #   candidate_topk_blocks×candidate_block_size = 16384 个候选 K（**候选个数**
        #   上限，不是序列跨度；候选块号仍是全局 0..L/8-1）。组合缓存是 QSLI 的
        #   K 镜像，按环长开后写入行取模、块表固定。**默认关**（跨 L 复用已放弃，
        #   按 L 开=精确全 L 镜像）；`DS41M_QSLI_CAP=16384` 可开环做固定几何/显存实验。
        _qcap = int(os.environ.get("DS41M_QSLI_CAP", "0") or 0)
        if _qcap > 0:
            _ql = min(int(max(lens_log)), int(_qcap))
            Qb = 1 + max(1, -(-_ql // rows_blk)) + 1
        else:
            Qb = max(1 + -(-(max(lens_log) + 1) // rows_blk), 1 + Pb)
        kk2, sk2 = f"_idx_qsli_r{ratio}", f"_idx_qslis_r{ratio}"
        kc = hold.get(kk2)
        if kc is None or kc.shape[0] < 1 + nseq * Qb:
            kc = torch.zeros(1 + nseq * Qb, rows_blk, 1, width,
                             dtype=torch.uint8, device=device)
            hold[kk2], hold[sk2] = kc, (rows_blk, Qb)
        bt2 = batch_layout.rot_block_table(Qb, nseq, device)
        ci2 = torch.full((int(n_rows),), -1, dtype=torch.long, device=device)
        _slot_pos = loc_abs % int(_qcap) if _qcap > 0 else loc_abs
        ci2[:n_real] = bt2[sid, _slot_pos // rows_blk].long() * rows_blk + _slot_pos % rows_blk
        hold[f"_idx_qsli_cap_r{ratio}"] = int(_qcap)
        _ifc.prolog_k((), dict(
            latent=latent.reshape(-1, latent.shape[-1]),
            wk=_IDX._nz("wk"),
            norm_weight=knw,
            rope_sin=sin.reshape(-1, _IDX.ROPE_DIM),
            rope_cos=cos.reshape(-1, _IDX.ROPE_DIM),
            k_cache=kc,
            k_scale_cache=None,
            cache_index=ci2,
            norm_eps=_IDX.NORM_EPS,
            storage_mode=1,
            combined_block_size=cbs,
        ))
        hold[f"_idx_qsli_bt_r{ratio}"] = bt2
        if os.environ.get("DS41M_DEBUG"):
            print(f"      [idx_k packed] 组合缓存 {tuple(kc.shape)}（块 {Qb}×行 {rows_blk}，"
                  f"宽 {width}）", flush=True)
    if os.environ.get("DS41M_DEBUG"):
        print(f"      [idx_k packed] ratio={ratio} lens={lens} Pb={Pb} "
              f"池{tuple(k.shape)}", flush=True)
    hold[f"_idx_k_bt_r{ratio}"] = bt
    # ★★ 2026-10-31（CP1-vs-CP2 逐位定位）：**packed 写者同样要 bump 池代数**。
    #   漏了这一行 ⇒ `_dg_paged_to_dense` 的 dense-K 缓存（键含 `_IDX_K_GEN`）在
    #   packed/CP1 路径**永不失效**：池/块表跨层复用同一 hold 键、data_ptr 也相同，
    #   于是层 8/14（ratio=2）读到的是**层 2 的 dense K**（实测 CP1 的 dg_in：L2/L8/L14
    #   的 k 字节完全相同，而三层 lat_sha 各不相同 ⇒ 池确实重写了、dense 没重算）。
    #   CP 路径（`_idx_k_rows_cp`）自带 bump 所以 CP2 是正确的 —— 这才是
    #   "L8/L14/L20 CP1≠CP2" 的真因，不是合成权重的 top-k 并列取舍。
    #   放在**两处 prolog_k 之后**：主池与候选组合缓存都写完，统一失效一次。
    _IDX_K_GEN[0] += 1
    return k, s, bt


def _idx_k_write_pk(st, ctx, w):
    """packed 版 K 缓存写入（层 20 边界：KV 路径全长写，compact qli 只读）。

    与顺序版 `exec_indexer.idx_k_write` 同语义；pool/块表按 packed 逐请求几何。
    """
    lat = st.get("cmp_latent")
    if lat is None:
        raise RuntimeError("idx_k_write 需要 st['cmp_latent']")
    idx_k_cache(st, ctx, w, lat.shape[0], lat.device)


def _qli_pk(st, ctx, w):
    """QLI（varlen 多请求，单次 batched 调用）：**单次 batched 调用**（官方 indexer.quant_lightning_indexer 口径）。"""
    d = torch.ops.cann_ops_transformer.ds41
    q = st["idx_q"]
    T = q.shape[0]
    ratio = max(1, int(ctx["ratio"]))
    topk = int(ctx["index_topk"])
    cu = batch_layout.cu_of(ctx)
    nseq = len(cu) - 1
    # ★ 2026-10-21 防御（同 qsli）：cu 末项 = 输入行数；不一致会让 kernel 按不存在的行调度。
    if int(cu[-1]) != int(T):
        raise RuntimeError(
            f"qli: cu[-1]={int(cu[-1])} 与输入行数 T={int(T)} 不一致（cu={list(cu)}）")
    lens = [(cu[b + 1] - cu[b]) // ratio for b in range(nseq)]
    # seqused_k = 官方 compressed_seq_lens = **整条序列**的 S//ratio；replay 裁尾只改 cu。
    raw_lens = [int(x) for x in (ctx.get("seq_lens") or
                                 [cu[b + 1] - cu[b] for b in range(nseq)])]
    slens = [x // ratio for x in raw_lens]
    # ★ 官方 compressed_seq_remainders = actual_seq_k % ratio（绝对 KV 长度），
    #   不是 compressed_len % ratio（2026-10-20 与 exec_indexer.py 同一崩因）。
    resids = [x % ratio for x in raw_lens]
    hold = _CORE._hold(st)
    is_cand = bool(ctx.get("is_candidate"))
    cand_topk = int(ctx.get("candidate_topk_blocks", _IDX.CAND_TOPK)) if is_cand else -1
    cand_size = int(ctx.get("candidate_block_size", _IDX.CAND_BLOCK)) if is_cand else -1
    i32 = torch.int32
    if os.environ.get("DS41M_IDX_TRACE") and os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"[idx-trace] qli_pk T={T} ratio={ratio} slens={slens} nseq={nseq} "
              f"is_cand={is_cand}", flush=True)

    def _mk_md():
        _cu = torch.tensor(cu, dtype=i32, device=q.device)
        _sk = torch.tensor(slens, dtype=i32, device=q.device)
        _rs = None if ratio == 1 else torch.tensor(
            resids, dtype=i32, device=q.device)
        _md = d.quant_lightning_indexer_metadata(
            cu_seqlens_q=_cu, seqused_q=None, seqused_k=_sk, cmp_residual_k=_rs,
            batch_size=nseq, max_seqlen_q=-1, max_seqlen_k=-1,
            num_heads_q=_IDX.N_HEADS, num_heads_k=1, head_dim=_IDX.HEAD_DIM, topk=topk,
            mask_mode=3, cmp_ratio=ratio, layout_q="TND", layout_k="PA_BBND",
            candidate_topk_blocks=cand_topk, candidate_block_size=cand_size)
        return _cu, _sk, _rs, _md

    cu_t, seq_k, resid, md = _CORE._meta(
        ("qli_pk_md", tuple(cu), tuple(slens), tuple(resids), int(ratio), int(topk),
         int(cand_topk), int(cand_size), str(q.device)), _mk_md)
    lat = st.get("cmp_latent")
    n_rows_k = lat.shape[0] if lat is not None else T // ratio
    # ★ 边界拆分：K 池已在 cut 前由 `idx_k_write` 全长写好（含块表），
    #   compact qli 只读、不再重写（否则 full latent + 512 行的 cu 会触发行数校验）。
    if (bool(ctx.get("idx_k_ready")) and f"_idx_k_r{ratio}" in hold
            and f"_idx_k_bt_r{ratio}" in hold):
        _qli_dbg(f"pk-readonly idx={ctx.get('idx')} ratio={ratio} T={T}")
        k = hold[f"_idx_k_r{ratio}"]
        dk = hold[f"_idx_ks_r{ratio}"]
        bt = hold[f"_idx_k_bt_r{ratio}"]
    else:
        if os.environ.get("DS41M_CMP_ARGDBG") and int(ctx.get("idx", -1)) == 20:
            _hk = [x for x in _CORE._hold(st).keys() if "_idx_k" in x][:4]
            print(f"  [idxk-miss] ready={bool(ctx.get('idx_k_ready'))} hold={_hk} "
                  f"sfx={_CORE._sfx(ctx,'idx')!r} nrows_k={n_rows_k} idx={ctx.get('idx')}", flush=True)
        k, dk, bt = idx_k_cache(st, ctx, w, n_rows_k, q.device)
    for name, t, stride in (("k", k, (_IDX.PACKED_D, _IDX.PACKED_D, 1)),
                            ("descale_k", dk, (4, 4, 2, 1))):
        if tuple(t.stride()[1:]) != stride:
            raise RuntimeError(f"{name} 内轴 stride 必须为 {stride}，实际 {tuple(t.stride())}")
    sparse, _val, _ci, _cl = _ifc.qli(
        (q, k, st["idx_w"].contiguous(), st["idx_descale_q"], dk, topk, 1),
        dict(cu_seqlens_q=cu_t, seqused_q=None, seqused_k=seq_k, cmp_residual_k=resid,
             block_table=bt, metadata=md, max_seqlen_q=-1, mask_mode=3, cmp_ratio=ratio,
             layout_q="TND", layout_k="PA_BBND", return_value=False,
             candidate_topk_blocks=cand_topk, candidate_block_size=cand_size))
    st["idx_sparse"] = sparse
    st["cmp_sparse_indices"] = sparse
    st["cmp_topk_length"] = (sparse >= 0).sum(-1).to(torch.int32)
    hold[f"_cmp_sparse_r{ratio}" + _CORE._sfx(ctx, "idx")] = sparse
    hold[f"_cmp_topklen_r{ratio}" + _CORE._sfx(ctx, "idx")] = st["cmp_topk_length"]
    if is_cand:
        hold["_li_cand_r1"] = _ci
        hold["_li_candlen_r1"] = _cl
        if os.environ.get("DS41M_CAND_DBG"):
            _cand_dbg_check(ctx, _ci, _cl, topk, cand_size, ratio)
        if os.environ.get("DS41M_DEBUG"):
            print(f"      [qli packed] 候选块: indices{tuple(_ci.shape)} "
                  f"len{tuple(_cl.shape)}", flush=True)
    if os.environ.get("DS41M_DEBUG"):
        print(f"      [qli packed] ratio={ratio} lens={lens} bt{tuple(bt.shape)} "
              f"sparse{tuple(sparse.shape)} 有效={int((sparse >= 0).sum())}", flush=True)



# ═══════════════════ DeepInfer 内核路径（阶段 1a）═══════════════════
# 开关：DS41M_IDX_KERNEL=dg(默认，2026-10-31 起，DeepInfer 口径) | vendor(旧融合算子)
#   ★ 默认切 dg 的依据：plain CP1 vs CP2 按全局行 **diff=0（含段边界 carry 组）**、
#     `tl` 全等；trace 里是 `deep_gemm::mqa_logits_impl` + `topk_select_ascend::topk_select_kernel`。
#     回退：`DS41M_IDX_KERNEL=vendor`（铁律 0：只用于 A/B，不许当默认）。
# 与 `_qli_pk` 的**唯一**差异是最后一步：融合 `quant_lightning_indexer`
#   → `deep_gemm_di.fp8_fp4_mqa_logits` + `deep_select.interface.topk`
# 前半段（q/T/ratio/cu/slens/resids/K 池/块表）逐字复用，保证 A/B 输入一致。


# ★ 2026-11-01（P4 归因）：`_DG_SLOT` 命中/未命中诊断开关（只加日志，不改路径）
_SLOT_STAT = os.environ.get("DS41M_DG_SLOT_STAT", "").strip().lower() in (
    "1", "on", "true", "yes")


def _dg_paged_to_dense(k_pool, dk_pool, bt, n_rows_k, ratio, device, n_full=None,
                       cu_k=None):
    """分页 K 池（68 B/行）→ dense (k_u8, k_sf)，供 deep_gemm 消费。

    池布局与 vendor 完全一致（`_idx_k_cache` / `_idx_k_rows_cp` 写的）：
      k_pool : [1+nblk, ROWS_PER_BLOCK, 1, PACKED_D] uint8   块 0 恒空
      dk_pool: [1+nblk, ROWS_PER_BLOCK, 1, 2, 2]     uint8
      bt     : [1, nblk] int32（旋转块表，单序列）
    行 p 的槽位 = bt[0][p // ROWS_PER_BLOCK] * ROWS_PER_BLOCK + p % ROWS_PER_BLOCK。

    ★ 必须用**同一张块表**导出（要求 2），否则比的不是内核是数据。
    """
    rpb = ROWS_PER_BLOCK
    # ★★ 2026-10-31：DeepInfer 的 K 是**全长 dense**（含 fake 前缀），
    #   `max_seqlen_k` = 全压缩行数，ks/ke 是**绝对**行号。
    #   我们只取尾部 n_rows_k 行会让 ke 超出、mask 失效（实测 dg 输出 480 而非因果的 1）。
    _n = int(n_full) if n_full else int(n_rows_k)
    # ★★ 2026-10-31（CP4 空泡）：(p, slot) 与两个**主机侧越界判据**都只跟 (n, bt, 池形, device)
    #   有关。旧代码每次调用：Range/FloorDiv/Mul/FloorMod/Add + `int(slot.max())`（设备
    #   reduce + **同步**）。CP4 每 forward 调 16 次（4 源层 ×2 块 + 4 QSLI 层 ×2 块）⇒
    #   每层一次全流水线同步。现在按 bt 对象身份缓存（强引用 bt 防地址复用）。
    _multiseq = cu_k is not None and len(cu_k) > 2
    _skey = ("dg_pg_slot", int(_n), int(bt.data_ptr()), int(rpb), str(device),
             tuple(int(x) for x in cu_k) if _multiseq else ())
    _shit = _DG_SLOT.get(_skey)
    if _SLOT_STAT:
        # ★ 2026-11-01（P4 归因）：判定"rank1 的 qli.kpool 比 rank0 慢 4.4 倍"是不是
        #   `_DG_SLOT` 缓存未命中导致（每层重建 p/slot + 一次 `slot.max()` 同步）。
        #   只加日志，不改执行路径。
        _rk = str(os.environ.get("RANK", "?"))
        _hit = _shit is not None and _shit[0] is bt
        # 注意：本函数没有 ctx（纯张量工具），所以只打 rank/n/bt 指针与缓存规模
        print(f"  [dg-slot] rank={_rk} {'HIT' if _hit else 'MISS'} n={int(_n)} "
              f"bt_ptr={int(bt.data_ptr())} n_cached={len(_DG_SLOT)} "
              f"multiseq={_multiseq}", flush=True)
    if _shit is not None and _shit[0] is bt:
        p, slot = _shit[1], _shit[2]
    else:
        p = torch.arange(_n, dtype=torch.int64, device=device)
        if _multiseq:
            # ★★ 2026-10-31（b>1 打包）：块表是**逐请求**的 [nseq, Pb]
            #   （`idx_k_cache` = 官方 `_prepare_indexer_slot_mapping` 口径），
            #   而 dense 行轴是各序列压缩行**拼接** ⇒ 必须分段用各自块表映射，
            #   否则第 2 条起读到上一条序列的 K（实测 507035 NaN trap）。
            parts = []
            for _i in range(len(cu_k) - 1):
                _lo, _hi = int(cu_k[_i]), int(cu_k[_i + 1])
                if _hi <= _lo:
                    continue
                _loc = torch.arange(_hi - _lo, dtype=torch.int64, device=device)
                _bti = bt[_i].long()
                if int((_hi - _lo - 1) // rpb) >= int(_bti.numel()):
                    raise RuntimeError(
                        f"_dg_paged_to_dense: 行号超块表 seq={_i} 行={_hi - _lo} "
                        f"bt_w={int(_bti.numel())} n_full={int(_n)} n_rows_k={int(n_rows_k)}")
                parts.append(_bti[_loc // rpb] * rpb + _loc % rpb)
            slot = torch.cat(parts)
            if int(slot.numel()) != int(_n):
                raise RuntimeError(
                    f"_dg_paged_to_dense: cu_k 行数 {int(slot.numel())} != n_full={int(_n)}"
                    f"（cu_k={[int(x) for x in cu_k]}）")
        else:
            _bt = bt[0].long()
            if int(p[-1] // rpb) >= int(_bt.numel()):
                raise RuntimeError(
                    f"_dg_paged_to_dense: 行号超块表 p_max={int(p[-1])} p//rpb={int(p[-1]//rpb)} "
                    f"bt_w={int(_bt.numel())} n_full={int(_n)} n_rows_k={int(n_rows_k)}")
            slot = _bt[p // rpb] * rpb + p % rpb
        if len(_DG_SLOT) >= 32:
            _DG_SLOT.clear()
        _DG_SLOT[_skey] = (bt, p, slot)
    # ★★ 2026-10-31：**槽位越界会在设备侧变成 aicore trap 507015（无栈无行号）**，
    #   这里先在主机侧判一次（CP4+hit 就踩这个：池按局部块开、而 n_full 按全长取行）。
    #   判据结果只跟 (slot, 池行数) 有关 → 同一次判定缓存，热路径零同步。
    _cap = k_pool.shape[0] * k_pool.shape[1]
    _CORE._meta(("dg_pg_cap", _skey, int(_cap)), lambda: _dg_slot_cap(
        slot, _cap, k_pool, n_full=_n, n_rows_k=n_rows_k, ratio=ratio))
    # ★★ 2026-10-31：dense K 本身也缓存 —— CP 下同一 forward 里 8 次调用读的是**同一个池**
    #   （层 20 写、层 24/28/32/36 只读；同层两个 CP 块也同池）。池写入用 `_IDX_K_GEN`
    #   单调计数做失效标记（`_idx_k_cache`/`_idx_k_rows_cp` 写完 +1）。
    _dkey = ("dg_pg_dense", int(k_pool.data_ptr()), int(dk_pool.data_ptr()),
             int(slot.data_ptr()), int(_n), int(_IDX_K_GEN[0]), str(device))
    _dhit = _DG_DENSE.get(_dkey)
    if _dhit is not None and _dhit[0] is k_pool and _dhit[1] is dk_pool:
        return _dhit[2], _dhit[3]
    k_flat = k_pool.view(-1, PACKED_D).index_select(0, slot).contiguous()
    sf_flat = dk_pool.reshape(-1, 4).index_select(0, slot).contiguous()
    # ★★ 2026-10-31（64k@0.97 + CP2 实测定根因）：命中仿真的**假前缀**行是全零 K + 全零 scale
    #   （实测 32253/33024 行）。零 scale 在 ue8m0 反量化 + fp4 乘加里会出 NaN ⇒
    #   `topk_select` 报 "NaN detected at batch index ..." ⇒ 随后 NonZero 507035。
    #   这里把「全零 scale」的行改成中性指数 127（值 2^0=1）：K 字节本身是 0，
    #   贡献仍为 0，但不再产生 NaN。真数据行（scale != 0）逐字节不动。
    #   ★ 2026-10-31：旧代码 `bool(_bad_sf.any())` 是**每调用一次设备→主机同步**；
    #   改成 `torch.where`（多 3 个 AIV 小算子，但**零同步**，不打断 host 抢跑）。
    #   `DS41M_DG_SF_SYNC=1` 回退旧的同步写法（A/B 用）。
    if os.environ.get("DS41M_DG_SF_SANITIZE", "1") != "0":
        # 0x00 = 未写的假前缀（贡献 0）；0xFF = **ue8m0 的 NaN 编码**（未初始化池内存，
        # 实测 rank0 行 402-406 出 NaN 就是它）⇒ 两者都改中性指数 127。真 scale 逐字节不动。
        _bad_sf = (sf_flat == 0) | (sf_flat == 0xFF)
        if os.environ.get("DS41M_DG_SF_SYNC", "").strip().lower() in (
                "1", "on", "true", "yes"):
            if bool(_bad_sf.any()):
                sf_flat = sf_flat.clone()
                sf_flat[_bad_sf] = 127
        else:
            sf_flat = torch.where(_bad_sf, _sf_neutral(sf_flat), sf_flat)
    if len(_DG_DENSE) >= 8:
        _DG_DENSE.clear()
    _DG_DENSE[_dkey] = (k_pool, dk_pool, k_flat, sf_flat)
    return k_flat, sf_flat


# ★ 2026-10-31（CP4 空泡）：`_dg_paged_to_dense` 的两级缓存。
#   · `_DG_SLOT`：几何（p/slot）按 bt 对象身份缓存，强引用 bt ⇒ 地址复用不可能串味
#   · `_DG_DENSE`：dense K/SF 按 (池身份, _IDX_K_GEN) 缓存 ⇒ 同一 forward 的
#     8 次读池只做 1 次 gather（池写入由 `_IDX_K_GEN` 单调递增失效）
#   · `_IDX_K_GEN`：池写入代数，`_idx_k_cache` / `_idx_k_rows_cp` 每次写完 +1
_DG_SLOT: dict = {}
_DG_DENSE: dict = {}
_IDX_K_GEN = [0]
_SF_NEU: dict = {}


def _sf_neutral(sf_flat):
    """ue8m0 中性指数 127 的 0 维张量（按 device 缓存，避免每次 H2D）。"""
    k = str(sf_flat.device)
    t = _SF_NEU.get(k)
    if t is None:
        t = torch.tensor(127, dtype=torch.uint8, device=sf_flat.device)
        _SF_NEU[k] = t
    return t


def _dg_slot_cap(slot, cap, k_pool, n_full=0, n_rows_k=0, ratio=0):
    """槽位越界判据（只在缓存建/池形变化时做一次设备 max + 同步）。"""
    _mx = int(slot.max())
    if _mx >= cap:
        raise RuntimeError(
            f"_dg_paged_to_dense 槽位越界：slot_max={_mx} 池行={cap} "
            f"(k_pool={tuple(k_pool.shape)} n_full={n_full} "
            f"n_rows_k={n_rows_k} ratio={ratio})")
    return True


def _dg_build_ks_ke(cu, raw_lens, kv_lens, ratio, device, row_off=0):
    """按 DeepInfer `prepare_cp_mirror_ref` 的公式逐行算 ks/ke。

    公式（`tilelang_ops/prepare_cp_mirror.py:118-125`）：
        cu_k_cn[i] = accumulate(max(1, sk // ratio))     # 每序列压缩 K 起点（批内累积）
        delta      = sk - sq + local_idx + 1
        ks[i, j]   = cu_k_cn[i]
        ke[i, j]   = cu_k_cn[i] + delta // ratio

    ★★ 2026-10-31 修正（首个 A/B 实证）：`sk` 是 **KV 长度（cu_seqlens_k）**，
    不是 query 长度 `sq`。命中场景下 `cu_seqlens_k` = 原始 KV 长（L_full = 8192），
    而 `cu_seqlens_q` = 本次新增 query 数（T_pass = 959）。
    证据：DeepInfer `input_metadata.py:185-193` 调 `prepare_cp_mirror(cu_seqlens_q,
    cu_seqlens_k, ...)`；`yoco.py:136` 注释明写「喂 compact cu_seqlens_q + **原始
    cu_seqlens_k**」。
    ⇒ delta = kv_len - q_len + j + 1，最后一行 ke = (kv_len)//ratio。
    用 959 当 sk 会让 ke 封顶在 479，而正确值是 4096（8192//2）。
    """
    nseq = len(cu) - 1
    i32 = torch.int32
    # ★★ ks/ke 必须是**相对 K 池**的行号（池第 0 行 = 绝对压缩行 row_off）
    #   内核判据：`kv_idx < ke[row] and ks[row] < kv_end`（mqa_logits.hpp:334-335），
    #   kv_idx 是**池内行号** ⇒ 必须减掉 row_off，否则 ke 超出池长、内核只用到池尾。
    ro = int(row_off)

    def _build():
        ks_list, ke_list = [], []
        # ★★ 2026-10-31（b>1 打包缺陷；DeepInfer `prepare_cp_mirror_ref:152`）：
        #   `ks` 是**批内压缩 K 前缀** `cu_k_cn[i] = Σ_{j<i} max(1, sk_j // ratio)`，
        #   不是常量 0。单序列（b=1）时 base 恒 0 ⇒ 零回归；多序列时第 2 条起的
        #   行会错落到池头部（读到上一条序列的压缩 K）。
        base = 0
        for b in range(nseq):
            sq = int(cu[b + 1] - cu[b])
            sk = int(kv_lens[b])          # ★ KV 长度（不是 query 长度）
            # ★ ks/ke 必须是**相对 K 池**的行号（池第 0 行 = 绝对行 off//ratio）
            for j in range(sq):
                delta = sk - sq + j + 1
                ks_list.append(base)
                ke_list.append(max(0, base + max(0, delta // ratio) - ro))
            base += max(1, sk // ratio)
        return (torch.tensor(ks_list, dtype=i32, device=device),
                torch.tensor(ke_list, dtype=i32, device=device))

    # ★★ 2026-10-31（CP4 空泡）：结果只跟 (cu, raw_lens, kv_lens, ratio, row_off, device)
    #   有关 —— 全是几何量。旧代码每次调用都重跑 Python 循环并做 **2 次阻塞 H2D**
    #   （`torch.tensor(list, device=npu)` 要等设备队列排空）。CP4 每 forward 调 16 次
    #   （源层 + QSLI 层 × 2 块）⇒ 32 次阻塞 H2D/forward，是 host 空泡的直接来源。
    return _CORE._meta(
        ("dg_ks_ke", tuple(int(x) for x in cu), tuple(int(x) for x in raw_lens),
         tuple(int(x) for x in kv_lens), int(ratio), ro, str(device)),
        _build)


def _dg_input_sha(q, w, k_dense, ks, ke):
    """A/B 输入 sha（要求 1）：q / w / K 行集 / ks / ke 各取一段指纹。"""
    import hashlib as _hl

    def _h(t):
        if t is None:
            return "none"
        a = t.detach().float().cpu().numpy().tobytes()
        return _hl.sha1(a).hexdigest()[:12]

    return dict(q=_h(q), w=_h(w), k=_h(k_dense), ks=_h(ks), ke=_h(ke))


def _dg_rank():
    """本进程 rank（torch.distributed 优先，其次 RANK 环境变量）。"""
    try:
        import torch.distributed as _dist
        if _dist.is_initialized():
            return int(_dist.get_rank())
    except Exception:                              # noqa: BLE001
        pass
    return int(os.environ.get("RANK", 0) or 0)


def _dg_psum_nandbg(p_sum, k_dense, k_sf, q_sf, ks, ke, wq, ctx):
    """p_sum 出口 NaN 判定（`DS41M_PSUM_NANDBG`）。

    ★★ 2026-10-31 晚：`fp8_fp4_mqa_logits` 的 logits 是 `torch::empty_strided`
    （attention.hpp:307）——**只有 [ks,ke) 段被内核写**，其余是未初始化的旧内存。
    旧探针把整行都算进 NaN，会把"尾部垃圾"误报成打分 NaN。这里按 ke 分成
    valid（内核写过、topk 真会读）与 tail（未写）两段分别计数。
    同时按**字节**统计 ue8m0：0xFF = NaN 编码，0xFE = 2^127（可致 bf16 inf），
    旧的 `isnan(k_sf.float())` 是对 int16 取 isnan（恒 0），没有诊断力。
    """
    try:
        _bad = torch.isnan(p_sum)
        _ke64 = ke.to(torch.int64)
        _cols = torch.arange(p_sum.shape[1], device=p_sum.device, dtype=torch.int64)
        _valid = _cols[None, :] < _ke64[:, None]
        _bad_v = _bad & _valid
        _bad_t = _bad & (~_valid)
        _nv = int(_bad_v.sum())
        _nt = int(_bad_t.sum())
        if _nv:
            _rows = _bad_v.any(dim=1).nonzero().flatten()[:5].tolist()
            _r0 = int(_rows[0])
            _c0 = _bad_v[_r0].nonzero().flatten()[:3].tolist()
            _nb_row = int(_bad_v[_r0].sum())
        else:
            _rows, _c0, _nb_row = [], [], 0
        # ── 输入字节统计（真正能解释 NaN 的量）──
        _ku8 = k_sf.view(torch.uint8)                  # [n_kv, 4] ue8m0
        _qu8 = q_sf.view(torch.uint8)                  # [T*32, 4] ue8m0
        _kem = int(ke.max())
        _kff = int((_ku8[:_kem] == 0xFF).sum())
        _kfe = int((_ku8[:_kem] == 0xFE).sum())
        _k00 = int((_ku8[:_kem] == 0x00).sum())
        _qff_rows = (_qu8 == 0xFF).any(dim=1).nonzero().flatten()
        _qff_tokv = (_qff_rows // _IDX.N_HEADS).unique()[:5].tolist() if _qff_rows.numel() else []
        _qfe = int((_qu8 == 0xFE).sum())
        _kzer = int((k_dense.view(torch.uint8) == 0).all(dim=1).sum())
        _khi = int((k_dense.view(torch.uint8) > 0x77).sum())
        print(f"  [psum-nan] idx={ctx.get('idx')} T={p_sum.shape[0]} "
              f"W={p_sum.shape[1]} NaN_valid={_nv} NaN_tail={_nt} "
              f"valid行={_rows} 行首列={_c0} 首行列数={_nb_row} "
              f"ks0={int(ks[0])} ke_max={_kem} k_rows={k_dense.shape[0]} "
              f"k_sf_mean16={float(k_sf.float().mean()):.1f} "
              f"k_sf_FF={_kff} k_sf_FE={_kfe} k_sf_00={_k00} "
              f"q_sf_FF={int(_qff_rows.numel())} q_FF_tok={_qff_tokv} q_sf_FE={_qfe} "
              f"wqNaN={int(torch.isnan(wq.float()).sum())} "
              f"k_zero_rows={_kzer} k_hi_bytes={_khi}", flush=True)
    except Exception as _e:                        # noqa: BLE001
        print(f"  [psum-nan] 失败 {type(_e).__name__}: {_e}", flush=True)


def _dg_dump_inputs(dd, ctx, q_u8, q_sf, k_dense, k_sf, wq, ks, ke, p_sum,
                    ratio, topk, n_full, n_rows_k, kv_lens, raw_lens, cu,
                    cand=None):
    """落 dg 的输入（`DS41M_DG_DUMP_IN`），供 host torch 参考复算。

    ★★ 2026-10-31 晚：**必须在 topk 之前调** —— topk_select 一旦 Nan trap，
      后面任何设备侧读（含这里的 .cpu()）都会 507035，dump 就永远拿不到。
    CP 下两个 rank / 每 rank 两块会写同名文件 ⇒ 名字带 rank + pos_offset + T。
    """
    try:
        os.makedirs(dd, exist_ok=True)
        _rk = _dg_rank()
        _off = int(ctx.get("pos_offset", 0) or 0)
        _p = os.path.join(
            dd, f"dg_in_L{ctx.get('idx')}_off{_off}_T{p_sum.shape[0]}_r{_rk}.pt")
        torch.save(dict(
            q=q_u8.detach().cpu(), q_sf=q_sf.detach().cpu(),
            k=k_dense.detach().cpu(), k_sf=k_sf.detach().cpu(),
            w=wq.detach().cpu(), ks=ks.detach().cpu(), ke=ke.detach().cpu(),
            ratio=ratio, topk=topk,
            n_heads=_IDX.N_HEADS, head_dim=_IDX.HEAD_DIM,
            # ★ 原始 ue8m0 字节（int16 视图不可读字节；复算按字节判 NaN）
            q_sf_u8=q_sf.detach().view(torch.uint8).cpu(),
            k_sf_u8=k_sf.detach().view(torch.uint8).cpu(),
            # ★ 几何（host 复算需要的全部 ctx）
            idx=ctx.get("idx"), rank=_rk, pos_offset=_off,
            T=p_sum.shape[0],
            L_full=int(ctx.get("L_full", 0) or 0), n_full=int(n_full),
            n_rows_k=int(n_rows_k), is_candidate=bool(ctx.get("is_candidate")),
            kv_lens=list(kv_lens), raw_lens=list(raw_lens), cu=list(cu),
            cand_blocks=(cand[0].detach().cpu() if cand and cand[0] is not None else None),
            cand_lens=(cand[1].detach().cpu() if cand and cand[1] is not None else None),
            vcnt=(cand[2].detach().cpu() if cand and cand[2] is not None else None),
            p_sum=p_sum.detach().cpu(),
        ), _p)
        print(f"  [dg-in] saved {_p}", flush=True)
    except Exception as _e:                        # noqa: BLE001
        print(f"  [dg-in] 失败 {type(_e).__name__}: {_e}", flush=True)


def _qli_dg(st, ctx, w):
    """QLI 的 DeepInfer 内核实现（`deep_gemm_di.fp8_fp4_mqa_logits` + `deep_select.topk`）。

    前半段与 `_qli_pk` 逐字相同；只替换最后的打分+topk。
    """
    # ★ 懒加载（要求：vendor 路径不许碰这两个包）
    import deep_gemm_di as _dgd
    from deep_select import interface as _dsi

    q = st["idx_q"]
    T = q.shape[0]
    ratio = max(1, int(ctx["ratio"]))
    topk = int(ctx["index_topk"])
    cu = batch_layout.cu_of(ctx)
    if not cu:
        # ★ CP 分块路径（`_qli_dg_cp`）：每块是"某前缀的尾部"单序列，
        #   ctx 里 cu_seqlens=None（与 vendor `_qli_cp` 同口径）⇒ 合成 [0, T]。
        cu = [0, int(T)]
    nseq = len(cu) - 1
    if int(cu[-1]) != int(T):
        raise RuntimeError(f"qli_dg: cu[-1]={int(cu[-1])} != T={int(T)}")
    raw_lens = [int(x) for x in (ctx.get("seq_lens") or
                                 [cu[b + 1] - cu[b] for b in range(nseq)])]
    hold = _CORE._hold(st)
    dev = q.device
    # ★ 2026-10-31：分段 host 计时（DS41M_HOSTTIME=1）——定位 qli 每次 3.8ms 花在哪。
    _tq = None
    if os.environ.get("DS41M_HOSTTIME", "").strip().lower() in ("1", "on", "true", "yes"):
        import time as _tqq
        _tq = _tqq.perf_counter
    _ts0 = _tq() if _tq else 0.0

    # ── K 池（与 `_qli_pk` 完全相同的获取逻辑）──
    lat = st.get("cmp_latent")
    n_rows_k = lat.shape[0] if lat is not None else T // ratio
    if (bool(ctx.get("idx_k_ready")) and f"_idx_k_r{ratio}" in hold
            and f"_idx_k_bt_r{ratio}" in hold):
        _qli_dbg(f"dg-readonly idx={ctx.get('idx')} ratio={ratio} T={T}")
        k = hold[f"_idx_k_r{ratio}"]
        dk = hold[f"_idx_ks_r{ratio}"]
        bt = hold[f"_idx_k_bt_r{ratio}"]
    else:
        k, dk, bt = idx_k_cache(st, ctx, w, n_rows_k, q.device)

    # ── paged → dense（单独计时，要求：paged→dense 开销写进报告）──
    _t0 = None
    if os.environ.get("DS41M_DG_TIMING"):
        torch.npu.synchronize()
        _t0 = time.time()
    # ── ks/ke 的可见 K 长度（必须在 dense 之前算：b>1 的 dense 宽度由它决定）──
    _Lfull = int(ctx.get("L_full", 0) or 0)
    _offK = int(ctx.get("pos_offset", 0) or 0)
    _span_end = _offK + int(T)
    if _Lfull > 0 and _span_end < _Lfull:
        # 分块 / 命中 stage A：用本块终点
        kv_lens = [_span_end for _ in raw_lens]
    else:
        kv_lens = [(int(_Lfull) if _Lfull > 0 else int(x)) for x in raw_lens]
    # ★★ 2026-10-31（b>1 打包）：多序列的 dense K 是**各序列压缩行拼接**
    #   （DeepInfer `prepare_cp_mirror_ref:152` 的 `cu_k_cn[-1]`），不是单条序列的
    #   `L_full//ratio`。nseq==1 时两条表达式逐位相同 ⇒ 零回归。
    _cu_k = [0]
    for _sk in kv_lens:
        _cu_k.append(_cu_k[-1] + max(1, int(_sk) // ratio))
    _n_full = (_cu_k[-1] if nseq > 1
               else (int(_Lfull) // ratio or int(n_rows_k)))
    k_dense, k_sf = _dg_paged_to_dense(k, dk, bt, n_rows_k, ratio, dev, n_full=_n_full,
                                       cu_k=(_cu_k if nseq > 1 else None))
    k_dense = k_dense.view(torch.int8)   # 同上：kPackedFP4 == int8
    # K 的 SF 同样要 int16 [n_kv, 2]（我们池里是 [n_kv, 4] uint8）
    k_sf = k_sf.reshape(k_dense.shape[0], 4).contiguous().view(torch.int16)
    if _t0 is not None:
        torch.npu.synchronize()
        print(f"  [dg-timing] idx={ctx.get('idx')} paged2dense="
              f"{(time.time() - _t0) * 1000:.3f} ms rows={n_rows_k}", flush=True)
    if _tq:
        _CORE._ht_add("qli.kpool", _tq() - _ts0)
        _ts0 = _tq()

    # ── ks / ke（DeepInfer 公式）──
    # ★★ 2026-10-31（CP2 实测根因）：可见 K 长度必须是**本块前缀的终点**
    #   `pos_offset + T`，不是 `L_full`。
    #   CP 下每个分块是「长为 pos_offset+T 的前缀的尾部」——用 L_full 会让
    #   因果 mask 完全失效（row0 索引到 3071，正确应接近 0），实测 CP2 的
    #   jaccard 掉到 0.30。vendor 同口径见 `_qli_chunk` 的
    #   `_spanK = (_offK + T) // ratio`（exec_indexer.py:1117）。
    #   CP1（pos_offset=0 且 T==L_full）时与旧值相同 ⇒ 零回归。
    #   （`kv_lens` 已在上面 dense 之前算好：b>1 的 dense 宽度也由它决定。）
    # ★ ks/ke 用**绝对**行号（与全长 dense K 对齐）
    ks, ke = _dg_build_ks_ke(cu, raw_lens, kv_lens, ratio, dev, row_off=0)
    if os.environ.get("DS41M_DG_SHA"):
        print(f"  [dg-kv] idx={ctx.get('idx')} L_full={_Lfull} "
              f"raw_lens={raw_lens[:2]} kv_lens={kv_lens[:2]} "
              f"ke_first={int(ke[0])} ke_last={int(ke[-1])} ke_max={int(ke.max())}",
              flush=True)

    if _tq:
        _CORE._ht_add("qli.kske", _tq() - _ts0)
        _ts0 = _tq()

    # ── 输入 sha（要求 1：A/B 归因）──
    if os.environ.get("DS41M_DG_SHA"):
        _s = _dg_input_sha(q, st["idx_w"], k_dense, ks, ke)
        print(f"  [dg-sha] idx={ctx.get('idx')} {_s}", flush=True)

    if os.environ.get("DS41M_DG_SHA"):
        print(f"  [dg-enter] idx={ctx.get('idx')} is_cand={bool(ctx.get('is_candidate'))} "
              f"T={T} n_full={_n_full} k_rows={k_dense.shape[0]} "
              f"ks0={int(ks[0])} ke_last={int(ke[-1])}", flush=True)

    # ── logits ──
    # 我们的 q: [T,32,64] uint8（FP4 打包），descale_q: [T,32,2,2] uint8
    # deep_gemm 要 (q_fp4, q_sf)：q_sf 用 uint8 视图展平成 [T,32,4]
    # ★ deep_gemm 的 kPackedFP4 == torch::kInt8（不是 uint8），必须 .view(int8)
    q_u8 = q.contiguous().view(torch.int8)
    # ★ deep_gemm 的 SF 契约（attention.hpp:289-292）：
    #   sf_q : int16, [T*num_heads, head_dim/64] = [T*32, 2]
    #   sf_kv: int16, [num_kv, head_dim/64]      = [n_kv, 2]
    #   我们的 descale 是 uint8 [T, 32, 2, 2]（4 个/head）⇒ 每 2 个 uint8 打成一个 int16
    q_sf = st["idx_descale_q"].reshape(T * _IDX.N_HEADS, 4).contiguous().view(torch.int16)
    wq = st["idx_w"].contiguous().to(torch.bfloat16)   # [T,32] fp32 → bf16（deep_gemm 要求）
    # k_sf: [n_rows, 4] uint8 → deep_gemm 要的粒度是每 32 元素一个 scale（128/32=4）✓
    # ★★ 2026-10-31（命中 + CP2 实测）：dense 路径**也需要** JIT 初始化。
    #   以前只在 candidate 分支调 `_ensure_jit`，命中场景下 L2/L8/L14 的
    #   `fp8_fp4_mqa_logits` 报 `lazy object must be initialized before use`
    #   （plain 场景不走 JIT，所以阶段 1a/2 没暴露）。
    if not _dgd._ensure_jit():
        raise RuntimeError("deep_gemm_di JIT 初始化失败（_C.init 返回非 0）")
    # ★★ 2026-10-31（对齐 DeepInfer，替换主控临时的 `ke.max()` 口径）：
    #   DeepInfer `index_attn.py:95`   self.max_kv_len = cfg.runtime_max_model_len // kv_compress_ratio
    #   DeepInfer `index_attn.py:305`  max_seqlen_k=self.max_kv_len
    #   —— `max_seqlen_k` 是**层/ratio 级常量上界**（由运行时最大长度决定），
    #      **不是本块 `ke.max()`**（后者是设备→主机同步，且随块变化）。
    #   对应到我们：`_n_full = L_full // ratio`（全局压缩行上界，L_full=运行时最大长度）。
    #   为了让"每 forward 只算一次"落到代码上，这里走 `_meta`（键 = (L_full, ratio)），
    #   所有层/所有 CP 块共享同一个 int，无设备同步、无逐块重算。
    #   `DS41M_DG_MAXK_FULL=1` 与该口径**等价**（保留只为兼容旧 A/B 脚本）。
    _maxk = _CORE._meta(("dg_maxk", int(ctx.get("L_full", 0) or 0), int(ratio)),
                        lambda: int(_n_full) if int(_n_full) > 0 else max(int(n_rows_k), 1))
    # ★★ 2026-10-31（对齐 DeepInfer `index_attn.py:117` + `:347-372` 的 q 分块）：
    #   DeepInfer `compute_chunk_size = min(8192, DEEPINFER_INDEX_ATTN_MEMORY_LIMIT
    #                                       // psum_row_bytes // 8 * 8)`，默认 LIMIT=10GiB；
    #   我们 psum_row_bytes = max_kv_len*2 ≤ 2*65536 ⇒ 上限恒为 8192。
    #   每个 chunk 调一次 `_compute_prefill_p_sum`（= 这里的 mqa logits）；
    #   逐行结果与整段调用逐位相同（logits/topk 都是逐行独立），所以**只是结构对齐**。
    #   `DS41M_DG_QCHUNK` 可覆盖（0 = 整段，复现旧口径）。
    _qchunk = int(os.environ.get("DS41M_DG_QCHUNK", "0") or 0) or min(
        8192, int(10 * (2 ** 30)) // max(1, int(_maxk) * 2) // 8 * 8)
    if nseq > 1:
        # ★★ 2026-10-31（b>1 打包）：logits 的 [0, ks) 列**不会被内核写**（`empty_strided`
        #   + 只写 [ks,ke) 区间，见 `mqa_logits.hpp:590-599`）⇒ 第 2 条序列的行会读到
        #   未初始化列，topk 直接 NaN/MTE trap（实测 507035/507015）。
        #   DeepInfer 的 dense 路径每 rank 单请求（DP 分摊 batch），不覆盖这条；
        #   我们按序列**逐段调用**：段内 ks=0（K 用该序列自己的行段），ke=段内因果长度，
        #   先把 p_sum 填 -inf，再把每段结果写进自己的列区间 ⇒ 段外列恒 -inf，
        #   candidate 块最大值与 topk（`end=ke` 绝对列）都只看到本序列的可见域。
        _nh = int(_IDX.N_HEADS)
        p_sum = torch.full((int(T), int(_maxk)), float("-inf"),
                           dtype=torch.bfloat16, device=dev)
        for _i in range(nseq):
            _r0, _r1 = int(cu[_i]), int(cu[_i + 1])
            _k0 = int(_cu_k[_i])
            _ni = max(1, int(_cu_k[_i + 1]) - _k0)
            _ke_i = (ke[_r0:_r1] - _k0).to(torch.int32)
            _ks_i = torch.zeros(_r1 - _r0, dtype=torch.int32, device=dev)
            p_sum[_r0:_r1, _k0:_k0 + _ni] = _dgd.fp8_fp4_mqa_logits(
                (q_u8[_r0:_r1], q_sf[_r0 * _nh:_r1 * _nh]),
                (k_dense[_k0:_k0 + _ni].contiguous(), k_sf[_k0:_k0 + _ni].contiguous()),
                wq[_r0:_r1], _ks_i, _ke_i,
                max_seqlen_k=int(_ni))
    elif _qchunk <= 0 or int(T) <= _qchunk:
        p_sum = _dgd.fp8_fp4_mqa_logits(
            (q_u8, q_sf), (k_dense, k_sf), wq, ks, ke,
            max_seqlen_k=int(_maxk))
    else:
        _nh = int(_IDX.N_HEADS)
        p_sum = torch.empty((int(T), int(_maxk)), dtype=torch.bfloat16, device=dev)
        for _qs in range(0, int(T), _qchunk):
            _qe = min(_qs + _qchunk, int(T))
            p_sum[_qs:_qe] = _dgd.fp8_fp4_mqa_logits(
                (q_u8[_qs:_qe], q_sf[_qs * _nh:_qe * _nh]),
                (k_dense, k_sf), wq[_qs:_qe], ks[_qs:_qe], ke[_qs:_qe],
                max_seqlen_k=int(_maxk))

    if _tq:
        _CORE._ht_add("qli.logits", _tq() - _ts0)
        _ts0 = _tq()

    # ── candidate 分支（L20 等候选源层）──
    # DeepInfer `index_attn.py:308`：先用块 max 选候选块，再只在候选块内算 logits。
    # 这里先按**全量** logits 选块（等价于官方 dense→block_max→sparse 的语义，
    # 只是少了 sparse 加速），再在候选块内重算 sparse logits。
    _is_cand = bool(ctx.get("is_candidate"))
    _cand_blocks = None
    if os.environ.get("DS41M_DG_SHA"):
        print(f"  [dg-is-cand] idx={ctx.get('idx')} is_candidate={_is_cand} "
              f"cand_src={ctx.get('candidate_source_layer')}", flush=True)
    if _is_cand:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import dg_candidate as _dgc
            _cbs = int(ctx.get("candidate_block_size", _IDX.CAND_BLOCK))
            _ctopk = int(ctx.get("candidate_topk_blocks", _IDX.CAND_TOPK))
            _bs, _valid = _dgc.candidate_block_max(p_sum, ke, _cbs)
            # ★ 官方 `index_attn.py:271`：candidate_base = ks // block_size（向下取整），
            #   把序列局部块号变**绝对块号**；后续 restore 直接 block_id*bs 不再加 base。
            # ★★ b>1：这里的 `p_sum` 是**绝对列**（多序列按段写入，段外 -inf），
            #   `candidate_block_max` 算出的块号本来就是绝对块号 ⇒ base 必须为 0，
            #   再加 ks//bs 会二次偏移（b=1 时 ks≡0，所以旧式在已验证口径上恒等）。
            _base = (torch.div(ks, _cbs, rounding_mode="floor").to(torch.int32)
                     if nseq == 1 else torch.zeros_like(ks))
            _cand_blocks, _vcnt = _dgc.select_candidate_blocks(
                _bs, _valid, _cbs, _ctopk, block_base=_base)
            _clens = _dgc.compact_lens(_cand_blocks, ke, _cbs, _vcnt)
            # ★ 官方约束（`get_sparse_mqa_logits_metadata` docstring）：
            #   "Each block ID i must satisfy ks <= i*B + ks%B < end"
            #   我们传的是**绝对块号**，需要满足 `i*B < nk`；越界会让 aicore trap。
            _nblk = k_dense.shape[0] // _cbs
            _cb = _cand_blocks.clamp(max=_nblk - 1)
            if os.environ.get("DS41M_DG_SHA"):
                _bad = int(((_cand_blocks >= 0) & (_cand_blocks >= _nblk)).sum())
                print(f"  [dg-cand-chk] idx={ctx.get('idx')} nblk={_nblk} "
                      f"blk_max={int(_cand_blocks.max())} 越界块数={_bad}", flush=True)
            # ★ 2026-10-31：源层**不再自用**这份 sparse logits（DeepInfer 源层 topk 走全量，
            #   `index_attn.py:339-360`）⇒ 默认**不调**这个内核（省一次打分）；
            #   `DS41M_DG_SRC_SPARSE=1` 恢复旧口径做 A/B/复现。
            if os.environ.get("DS41M_DG_SRC_SPARSE") == "1":
                if not _dgd._ensure_jit():
                    raise RuntimeError("JIT 不可用（_C.init 失败）")
                _meta = _dgd.get_sparse_mqa_logits_metadata(
                    ks, ke, k_dense.shape[0], _cb, q_u8.dtype, _cbs)
                p_sum = _dgd.fp8_fp4_sparse_mqa_logits(
                    (q_u8, q_sf), (k_dense, k_sf), wq, _meta,
                    _cand_blocks.shape[1], _cbs)
            if os.environ.get("DS41M_DG_SHA"):
                _bl = _cand_blocks[_cand_blocks >= 0]
                print(f"  [dg-cand] idx={ctx.get('idx')} cbs={_cbs} ctopk={_ctopk} "
                      f"blocks={tuple(_cand_blocks.shape)} lens_sum={int(_clens.sum())} "
                      f"p_sum={tuple(p_sum.shape)} "
                      f"blk_min={int(_bl.min()) if _bl.numel() else -1} "
                      f"blk_max={int(_bl.max()) if _bl.numel() else -1} "
                      f"nk={k_dense.shape[0]} ks0={int(ks[0])} ke_last={int(ke[-1])} "
                      f"ksmax={int(ks.max())} nblk_cap={k_dense.shape[0]//_cbs}",
                      flush=True)
        except Exception as _e:                    # noqa: BLE001
            print(f"  [dg-cand] 失败 {type(_e).__name__}: {_e}（回退全量 dense）", flush=True)
            _cand_blocks = None
    if _tq:
        _CORE._ht_add("qli.cand", _tq() - _ts0)
        _ts0 = _tq()

    # ── topk（DeepSelect，就地写 output_idx）──
    # ★ 语义归一化（要求 3）：DeepInfer 的 sp 是 `output_idx_offset=ks` 之后的**批内全局位置**；
    #   vendor 的 sp 是压缩 K 位置。这里用 offset=ks[b] 让 dg 的输出直接对齐 vendor 口径。
    #   公式：dg_sp[b, j] = ks[b] + (批内位置)
    # ★ 诊断：logits 统计（判断是否全 0/全等）
    if os.environ.get("DS41M_DG_SHA"):
        try:
            _ps = p_sum.float()
            print(f"  [dg-logits] idx={ctx.get('idx')} shape={tuple(_ps.shape)} "
                  f"min={float(_ps.min()):.4g} max={float(_ps.max()):.4g} "
                  f"mean={float(_ps.mean()):.4g} "
                  f"row0[:6]={_ps[0][:6].tolist()} "
                  f"row0[3616:3622]={_ps[0][3616:3622].tolist()} "
                  f"nonzero={int((_ps!=0).sum())}", flush=True)
        except Exception as _e:                    # noqa: BLE001
            print(f"  [dg-logits] 失败 {type(_e).__name__}: {_e}", flush=True)

    # ★★ 2026-10-31 晚：诊断/落盘必须在 topk **之前**（topk trap 后设备读会 507035）。
    if os.environ.get("DS41M_PSUM_NANDBG"):
        _dg_psum_nandbg(p_sum, k_dense, k_sf, q_sf, ks, ke, wq, ctx)
    _dd = os.environ.get("DS41M_DG_DUMP_IN", "").strip()
    if _dd:
        _dg_dump_inputs(_dd, ctx, q_u8, q_sf, k_dense, k_sf, wq, ks, ke, p_sum,
                        ratio, topk, _n_full, n_rows_k, kv_lens, raw_lens, cu,
                        cand=((_cand_blocks, _clens, _vcnt)
                              if _cand_blocks is not None else None))
    out_idx = torch.full((T, topk), -1, dtype=torch.int32, device=dev)
    # ★★ 2026-10-31（对齐 DeepInfer `index_attn.py:365`）：
    #   `end=topk_lens` 才是每行的**可见 K 终点**（mask 真正生效的地方）；
    #   不传 end 会让 topk 在整条 logits 宽度内选，ke 的因果 mask 形同虚设。
    #   DeepInfer 的 `klen = ke - ks`，这里 end 用 ke（相对 logits 起点 0）。
    # ★★ 2026-10-31（对齐 DeepInfer `index_attn.py:339-365`）：**源层自用 topk 走全量 logits**。
    #   DeepInfer 里 `candidate_blocks` 是 `_get_candidate_inputs` 从 cross-layer runtime 取的
    #   ——源层自己算的时候还是 None ⇒ `_compute_prefill_p_sum` 走 dense、
    #   `_topk_candidate_block` 只为**下游**备候选块、topk 用 `output_idx_offset=ks`
    #   且**不做 restore**。我们以前在源层把 p_sum 换成"候选块内 sparse logits"再 topk+restore：
    #   当候选块数 == 全量块数（L ≤ candidate_topk_blocks×cbs）时结果逐位相同，
    #   但**多算一个 sparse 内核**；L 更大时是错的（top-k 被限制在子集里）。
    if _cand_blocks is not None and os.environ.get("DS41M_DG_SRC_SPARSE") == "1":
        # 旧口径（仅复现用）：candidate 口径，end = 紧凑长度，offset 不传，最后还原全局索引
        _dsi.topk(
            p_sum, topk,
            sorted=False, indices_type=torch.int32, sorted_index=True,
            output_idx=out_idx,
            end=_clens,
            idx_oob_fill_value=-1, return_value=False)
        out_idx = _dgc.restore_indices(out_idx, _cand_blocks, _cbs,
                                       int(ks[0]) if ks.numel() else 0)
    else:
        _topk_lens = ke.contiguous()
        # ★★ b>1：`p_sum` 的列是**绝对**压缩行号（段外 -inf）⇒ topk 返回的列号
        #   已经是全局行号，**不能再加 ks**（加了会二次偏移到 2×ks，下游 gather
        #   MTE 越界，实测 507015 延到下一层才报）。b=1 时 ks≡0，零回归。
        _off_t = ks if nseq == 1 else torch.zeros_like(ks)
        _dsi.topk(
            p_sum, topk,
            sorted=False, indices_type=torch.int32, sorted_index=True,
            output_idx=out_idx,
            output_idx_offset=_off_t,
            end=_topk_lens,
            idx_oob_fill_value=-1, return_value=False)
        if nseq > 1 and os.environ.get("DS41M_BIDX_SPACE", "local").strip().lower() != "abs":
            # ★★ 2026-10-31（b>1）：mqsmla 的 `cmp_sparse_indices` 是**请求内局部**
            #   压缩行号（`cmp_block_table[b] = b*Pc+1..`，见 exec_ops `cmp_bt_pk`；
            #   `seqused_cmp_kv` = 该请求的可见压缩长度）⇒ topk 的绝对列号要减掉
            #   本序列的起始压缩行 `cu_k[i]`。-1（无效位）保持 -1。
            #   逐序列原地切片做（nseq 很小；避免 repeat_interleave/where 的整表算子）。
            for _i in range(nseq):
                _a, _b = int(cu[_i]), int(cu[_i + 1])
                _bn = int(_cu_k[_i])
                if _bn:
                    _s = out_idx[_a:_b]
                    _m = _s >= 0
                    _s.sub_(_bn)
                    _s.masked_fill_(~_m, -1)

    if _tq:
        _CORE._ht_add("qli.topk", _tq() - _ts0)
        _ts0 = _tq()

    sparse = out_idx.reshape(T, 1, topk).to(torch.int32)
    st["idx_sparse"] = sparse
    st["cmp_sparse_indices"] = sparse
    st["cmp_topk_length"] = (sparse >= 0).sum(-1).to(torch.int32)
    hold[f"_cmp_sparse_r{ratio}" + _CORE._sfx(ctx, "idx")] = sparse
    hold[f"_cmp_topklen_r{ratio}" + _CORE._sfx(ctx, "idx")] = st["cmp_topk_length"]
    # ★★ 2026-10-31：候选源层必须把候选块交给下游 qsli（L24/28/32）。
    #   vendor 在 `_qli_pk` 里写 `_li_cand_r1` / `_li_candlen_r1`
    #   （见 exec_indexer.py:1242-1243），dg 路径漏了 → 下游报
    #   「qsli 需要候选块 —— 请先跑候选源层的 qli」。
    #   口径：indices = **绝对块号**（block_base 已加），length = 有效块数。
    # ★ 2026-10-31 探针（DS41M_QS_KCHK）：源层同时有**已校验的 split index-K 池**
    #   （`k_dense`，来自 `_idx_k_r{ratio}`）与刚写好的**组合缓存** ⇒ 逐窗口位置比对，
    #   确定「组合缓存第 p 行的第几个键 == 压缩位置 p 的真 K」。QSLI 的 dg 移植就靠这个。
    if os.environ.get("DS41M_QS_KCHK") and _is_cand:
        try:
            _kc = hold.get(f"_idx_qsli_r{ratio}")
            _qi = hold.get(f"_idx_qslis_r{ratio}")
            if _kc is not None and _qi is not None:
                _rb, _n2 = _qi
                _cap = int(hold.get(f"_idx_qsli_cap_r{ratio}", 0) or 0)
                _nb = min(64, k_dense.shape[0])
                # ★ 坐标修正：`k_dense` 是**本层 K 池**（L20 边界截断后重基过）⇒
                #   第 0 行 = 全局压缩行 `pos_offset//ratio`。不加这个偏移就是拿两段
                #   不相干的行比对（第一版因此得出「全 0 命中」的假结论）。
                _ro = int(ctx.get("pos_offset", 0) or 0) // max(1, int(ratio))
                _sl = _qsli_slots(_nb, int(_rb), int(_n2), dev, row_off=_ro, ratio=1, ring=_cap)
                _fl = _kc.reshape(-1, _kc.shape[-1])
                _win = _fl.index_select(0, _sl).reshape(_nb, _cbs, PACKED_D + 4)
                _truth = k_dense[:_nb].view(torch.uint8).reshape(_nb, PACKED_D)
                _hits = []
                for _k in range(int(_cbs)):
                    _eq = int((_win[:, _k, :PACKED_D] == _truth).all(dim=1).sum())
                    _hits.append(_eq)
                print(f"  [qs-kchk] idx={ctx.get('idx')} n={_nb} 窗口位置命中数={_hits} "
                      f"(k 号 == 位置 p 的最多者即正确偏移)", flush=True)
        except Exception as _e:                    # noqa: BLE001
            print(f"  [qs-kchk] 失败 {type(_e).__name__}: {_e}", flush=True)

    if _is_cand and _cand_blocks is not None:
        # 形状契约（exec_ops.py:3241）：indices 必须 (T,1,cand_topk)，length (T,1)。
        hold["_li_cand_r1"] = _cand_blocks.contiguous().view(T, 1, -1)
        hold["_li_candlen_r1"] = _vcnt.to(torch.int32).view(-1, 1)
    if _tq:
        _CORE._ht_add("qli.wb", _tq() - _ts0)
    return sparse


def _qli_dg_cp(st, ctx, w, cp):
    """CP 下的 dg 路径：**逐块**调 `_qli_dg`（与 vendor `_qli_cp` 同构）。

    背景：CP 下 `st['idx_q']` 只有本地行（zigzag 的 head/tail 两块），而
    `ctx["cu_seqlens"]` 是**全局**序列的 → 直接调 `_qli_dg` 会 `cu[-1] != T`。

    与 vendor 的分块口径一致：块 [gs, gs+nv) 是"长为 gs+nv 的前缀的尾部"，
    `pos_offset=gs`、`seq_lens=[L]`、`n_seqs=1`、`cu_seqlens=None`。

    K 池：必须已由 `_idx_k_rows_cp` 按全量 latent 写好（`idx_k_ready`）；
    分块只改 query 行归属，不改 K 侧几何。
    """
    ratio = max(1, int(ctx["ratio"]))
    topk = int(ctx["index_topk"])
    dev = st["idx_q"].device
    Tloc = st["idx_q"].shape[0]
    is_cand = bool(ctx.get("is_candidate"))
    hold = _CORE._hold(st)

    # ★ K 池：必须先用**全量 latent** 按 off=0 几何写好（与 vendor `_qli_cp` 同口径），
    #   分块只改 query 行归属、不改 K 侧几何。漏了这步会让块内 `_qli_dg`
    #   走 `idx_k_cache`，而块 ctx 的 cu_seqlens=None ⇒ `len(cu)` 崩。
    lat = st.get("cmp_latent")
    if lat is not None and not ctx.get("idx_k_ready") and int(ctx["ratio"]) > 1:
        _kg_ok = False
        _kg_env = os.environ.get("DS41M_CP_K_GATHER", "").strip().lower()
        if _kg_env not in ("0", "off", "false", "no"):
            try:
                _kg_ok = bool(_idx_k_rows_cp(st, ctx, cp, w))
            except Exception as _e:               # noqa: BLE001
                print(f"  [cp-k-gather] 失败 {type(_e).__name__}: {_e}（回退拼接写池）",
                      flush=True)
                _kg_ok = False
        if not _kg_ok:
            # ★ 2026-11-02（用户裁决）：删「gather 全长 → 全长写池」回退。
            #   真实推理应只按本地 q 算；全长 gather 是白算。铁律 0：不静默降级。
            raise RuntimeError(
                "CP K 池写入失败：`_idx_k_rows_cp` 未成功"
                "（或 DS41M_CP_K_GATHER=0）。全长回退路径已删除（白算）。")

    if os.environ.get("DS41M_OP_TB"):
        _hkeys = [k for k in hold if "_idx_k" in str(k)]
        _haslat = lat is not None
        print("  [dg-cp-kw] idx=%s ratio=%s lat=%s hold_keys=%s"
              % (ctx.get("idx"), ratio, _haslat, _hkeys), flush=True)
    # ★ 块表兜底：`_qli_dg` 的只读判据是三键齐全（k / ks / bt）。
    #   CP 下池可能由别的路径写好（只有 k/ks），缺 bt 会被判成"没写过"→
    #   回退 `idx_k_cache`，而块 ctx 的 cu_seqlens=None ⇒ 崩。
    #   块表是 nblk 的纯函数，安全重建。
    _bt_key = f"_idx_k_bt_r{ratio}"
    _k_key = f"_idx_k_r{ratio}"
    if _k_key in hold and _bt_key not in hold:
        _n_full = max(int(ctx.get("L_full", 0) or 0) // ratio + 1, 1)
        _nblk = max(1, -(-(_n_full + 1) // ROWS_PER_BLOCK))
        hold[_bt_key] = batch_layout.rot_block_table(_nblk, 1, dev)
        hold[f"_idx_nblk_r{ratio}"] = _nblk

    if os.environ.get("DS41M_OP_TB"):
        if torch.is_tensor(lat):
            import hashlib as _hl3
            print("  [dg-lat] idx=%s lat_rows=%s lat_sha=%s"
                  % (ctx.get("idx"), lat.shape[0],
                     _hl3.sha1(lat.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]),
                  flush=True)
        else:
            print("  [dg-lat] idx=%s lat=None" % ctx.get("idx"), flush=True)
    if os.environ.get("DS41M_OP_TB"):
        _kk2 = hold.get(_k_key)
        if torch.is_tensor(_kk2):
            import hashlib as _hl2
            _s2 = _hl2.sha1(_kk2.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]
        else:
            _s2 = None
        print("  [dg-k-sha] idx=%s after_write k_sha=%s shape=%s"
              % (ctx.get("idx"), _s2,
                 tuple(_kk2.shape) if torch.is_tensor(_kk2) else None), flush=True)

    base = dict(ctx)
    base["idx_k_ready"] = _k_key in hold and _bt_key in hold
    _seq_cp = int(getattr(cp, "seq_len", 0) or cp.plan.total_q)
    _pofs_cp = int(getattr(cp, "pos_offset", 0) or 0)
    base["L_full"] = _seq_cp
    base["cu_seqlens"] = None
    base["seq_lens"] = [int(_seq_cp)]
    base["row_lens"] = None
    base["n_seqs"] = 1
    base["_cp_chunk"] = 1
    base.pop("cp_positions", None)
    base.pop("cp_total_q", None)

    sp = torch.full((Tloc, 1, topk), -1, dtype=torch.int32, device=dev)
    tl = torch.zeros((Tloc, 1), dtype=torch.int32, device=dev)
    cand = [None, None]
    _ck = ("_li_cand_r1", "_li_candlen_r1")
    for ls, gs, nv in cp.plan.local_blocks():
        if nv <= 0:
            continue
        st2 = dict(st)
        st2["idx_q"] = st["idx_q"][ls:ls + nv].contiguous()
        st2["idx_descale_q"] = st["idx_descale_q"][ls:ls + nv].contiguous()
        st2["idx_w"] = st["idx_w"][ls:ls + nv].contiguous()
        # 块内候选块 / 长度由 _qli_dg 写进 hold，先清掉防止上一块的残留
        for key in _ck:
            hold.pop(key, None)
        _qli_dg(st2, dict(base, pos_offset=int(gs) + _pofs_cp, T=int(nv)), w)
        sp[ls:ls + nv] = st2["cmp_sparse_indices"].reshape(nv, 1, -1).to(torch.int32)
        tl[ls:ls + nv] = st2["cmp_topk_length"].reshape(nv, 1).to(torch.int32)
        if is_cand:
            for i, key in enumerate(_ck):
                v = hold.get(key)
                if v is None:
                    continue
                if cand[i] is None:
                    cand[i] = torch.zeros((Tloc,) + tuple(v.shape[1:]),
                                          dtype=v.dtype, device=v.device)
                cand[i][ls:ls + nv] = v.reshape((nv,) + tuple(v.shape[1:]))
    st["idx_sparse"] = sp
    st["cmp_sparse_indices"] = sp
    st["cmp_topk_length"] = tl
    hold[f"_cmp_sparse_r{ratio}" + _CORE._sfx(ctx, "idx")] = sp
    hold[f"_cmp_topklen_r{ratio}" + _CORE._sfx(ctx, "idx")] = tl
    # ★★ 2026-10-31（CP4 + hit + dg 挂死）：出口**有界断言** —— 越界索引会在下一层
    #   mqsmla 变成设备侧 aicore timeout（无栈无行号）。这里先在主机侧判一次。
    if os.environ.get("DS41M_CP_IDX_CHECK", "1") != "0":
        # ★ 2026-10-31（CP4 空泡）：这条是**主机侧防御断言**，但旧写法每次调用发
        #   GreaterEqual/NonZero/Index/ReduceMax + `int(max())` 一次**设备→主机同步**
        #   → 8 个带索引器的层 × 每 forward 一次全流水线同步（trace 里紧贴 Mqsmla 前
        #   那 5 个算子）。判据只跟 (sp 构造几何, 层) 有关 ⇒ 每几何每层判一次即可
        #   （META 随几何清理，重新开启）。`DS41M_CP_IDX_CHECK=0` 仍可整体关闭。
        _ckey = ("cp_idx_chk", int(Tloc), int(ratio), int(ctx.get("L_full", 0) or 0),
                 int(getattr(cp, "pos_offset", 0) or 0), int(ctx.get("idx", -1)),
                 str(dev))
        if _CORE.META.get(_ckey) is None:
            _cap = max(int(ctx.get("L_full", 0) or 0) // max(1, int(ratio)) + 1,
                       int(getattr(cp, "pos_offset", 0) or 0) // max(1, int(ratio))
                       + int(Tloc)) + 2
            _v = sp[sp >= 0]
            if _v.numel() and int(_v.max()) >= _cap:
                raise RuntimeError(
                    f"qli_dg_cp sparse 越界：max={int(_v.max())} cap={_cap} "
                    f"(idx={ctx.get('idx')} L_full={ctx.get('L_full')} ratio={ratio} "
                    f"Tloc={Tloc} off={getattr(cp, 'pos_offset', None)})")
            _CORE.META[_ckey] = True
    if is_cand:
        for key, v in zip(_ck, cand):
            if v is not None:
                hold[key] = v
    return sp



def _qsli_pk(st, ctx, w):
    """QSLI（varlen 多请求，单次 batched 调用）：**单次 batched 调用**（官方 indexer.quant_sparse_lightning_indexer 口径）。"""
    # ★ 2026-10-31：内核分派（同 `_qsli_chunk`；这是 b=1 预填的**实际**路径）。
    # ★★ 2026-10-31 定稿：**默认 dg**。判据＝自洽验证 `DS41M_QS_SPCHK`：
    #   同一进程里 dg 的「候选内稀疏 top-k」与「全量 dense top-k」逐行**完全一致**
    #   （L24 T=140：全行相同 140/140、row0 512/512）⇒ 稀疏路径本身正确。
    #   dg 的 K 与 dense 路径**同一份**（共享 split index-K 池）= DeepInfer
    #   `use_shared_index_k` 口径；vendor 融合算子吃的是**组合缓存 repack** 的 K，
    #   两者在近似并列处有 ~4% 差（493/512），那是 repack 的口径差，不是我们的错。
    #   回退：`DS41M_QSLI_KERNEL=vendor`（仅 A/B/复现）。
    _qk = os.environ.get("DS41M_QSLI_KERNEL", "dg").strip().lower()
    if _qk == "dg":
        return _qsli_dg_chunk(st, ctx, w)
    d = torch.ops.cann_ops_transformer.ds41
    q = st["idx_q"]
    T = q.shape[0]
    ratio = max(1, int(ctx["ratio"]))
    topk = int(ctx["index_topk"])
    cbs = int(ctx.get("candidate_block_size", _IDX.CAND_BLOCK))
    cand_topk = int(ctx.get("candidate_topk_blocks", _IDX.CAND_TOPK))
    cu = batch_layout.cu_of(ctx)
    nseq = len(cu) - 1
    # ★ 2026-10-21 防御：cu 是**输入行边界**，末项必须等于 q 的行数。回放裁尾时
    #   replay 若没同步改写 cu（b=1 曾漏），这里会以 2048 行去调度 128 行的 q →
    #   metadata AICPU 越界（507018）。宁可在这里报清楚的错，也不要设备侧挂死。
    if int(cu[-1]) != int(T):
        raise RuntimeError(
            f"qsli: cu[-1]={int(cu[-1])} 与输入行数 T={int(T)} 不一致"
            f"（回放裁尾未同步 cu？cu={list(cu)} n={nseq}）")
    lens = [int(x) // ratio for x in (ctx.get("seq_lens") or
                                      [cu[b + 1] - cu[b] for b in range(nseq)])]
    hold = _CORE._hold(st)
    cand_i = hold.get("_li_cand_r1")
    cand_l = hold.get("_li_candlen_r1")
    if cand_i is None:
        raise RuntimeError("qsli 需要候选块 —— 请先跑候选源层的 qli")
    kc = hold.get(f"_idx_qsli_r{ratio}")
    if kc is None:
        raise RuntimeError(f"qsli 需要组合索引器缓存（源层的 _idx_qsli_r{ratio}）")
    bt2 = hold.get(f"_idx_qsli_bt_r{ratio}")
    if bt2 is None:
        raise RuntimeError(f"qsli 需要组合缓存块表（源层的 _idx_qsli_bt_r{ratio}）")
    # ★ 定长环：候选位置对环长取模（与写侧 `_slot_pos` 同一张固定块表）；
    #   -1/无效值保持原样（由 candidate_block_length 屏蔽）。
    _qcap = int(hold.get(f"_idx_qsli_cap_r{ratio}", 0) or 0)
    if _qcap > 0:
        cand_i = torch.where(cand_i >= 0, cand_i % _qcap, cand_i)
    i32 = torch.int32
    loc, _ = batch_layout.local_positions(ctx, T, q.device)
    # candidate_block_length：官方 (pos + ratio*cbs)//(ratio*cbs)，clamp[1, cand_topk]
    # ★ T-042：cl 只跟 `loc`（逐请求行号）+ 几何有关，而 `loc` 来自 `batch._CACHE`
    #   （key=cu/T/off/offs/device）——同一几何下是**同一个只读对象**。按对象身份缓存，
    #   消掉每层 5 个设备算子（add/div/clamp_min/clamp_max/to）。逐位前提：本文件对 loc
    #   只读；对象一换（新几何/新 forward 布局）身份比对失败 → 重算。
    def _cl_build():
        return ((loc + ratio * cbs) // (ratio * cbs)).clamp_min(1).clamp_max(cand_topk) \
            .to(i32).view(-1, 1)

    _clk = (int(loc.data_ptr()), loc.shape[0], int(ratio * cbs), int(cand_topk),
            str(q.device))
    _cl_ent = _QS_CL_CACHE.get(_clk)
    if _cl_ent is not None and _cl_ent[0] is loc:
        cl = _cl_ent[1]
    else:
        cl = _cl_build()
        _QS_CL_CACHE.clear()
        _QS_CL_CACHE[_clk] = (loc, cl)
    if os.environ.get("DS41M_IDX_TRACE") and os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"[idx-trace] qsli_pk T={T} ratio={ratio} lens={lens} "
              f"cand={tuple(cand_i.shape)} candlen={tuple(cand_l.shape)} "
              f"cl={tuple(cl.shape)} bt={tuple(bt2.shape)}", flush=True)

    # 诊断（DS41M_QS_DUMP=目录）：落 QSLI 入参快照，与顺序实现对比（默认关）
    if (os.environ.get("DS41M_QS_DUMP")
            and int(ctx.get("idx", -1)) == int(os.environ.get("DS41M_QS_DUMP_LAYER", "24"))):
        _dd = os.environ["DS41M_QS_DUMP"]
        os.makedirs(_dd, exist_ok=True)
        torch.save({
            "kind": "packed", "T": int(T), "ratio": int(ratio),
            "off": int(ctx.get("pos_offset", 0) or 0),
            "cu": [int(x) for x in cu], "lens": [int(x) for x in lens],
            "cl": cl.detach().cpu(), "cand_i": cand_i.detach().cpu(),
            "cand_l": cand_l.detach().cpu(), "bt": bt2.detach().cpu(),
            "kc_shape": [int(x) for x in kc.shape],
            "cand_min": int(cand_i.min()) if cand_i.numel() else -1,
            "cand_max": int(cand_i.max()) if cand_i.numel() else -1,
            "cand_len_max": int(cand_l.max()) if cand_l.numel() else -1,
        }, os.path.join(_dd, f"qsli_{os.environ.get('RANK', '0')}_packed.pt"))

    def _mk_md2():
        _cu = torch.tensor(cu, dtype=i32, device=q.device)
        _sk = torch.tensor(lens, dtype=i32, device=q.device)
        _md = d.quant_sparse_lightning_indexer_metadata(
            candidate_block_length=cl, cu_seqlens_q=_cu, seqused_q=None,
            seqused_k=_sk, cmp_residual_k=None,
            batch_size=nseq, max_seqlen_q=-1, max_seqlen_k=-1,
            num_heads_q=_IDX.N_HEADS, num_heads_k=1, head_dim=_IDX.HEAD_DIM, topk=topk,
            quant_mode=1, candidate_block_size=cbs, mask_mode=3, cmp_ratio=ratio,
            layout_q="TND", layout_k="PA_BBND")
        return _cu, _sk, _md

    cu_t, seq_k, md = _CORE._meta(
        ("qsli_pk_md", tuple(cu), tuple(lens), int(ratio), int(topk), int(cbs),
         int(cand_topk), int(T), int(ctx.get("pos_offset", 0) or 0), str(q.device)),
        _mk_md2)
    q2 = q.view(-1, _IDX.N_HEADS, _IDX.PACKED_D)
    dq = st["idx_descale_q"].view(torch.uint8).view(
        q2.shape[0], _IDX.N_HEADS, st["idx_descale_q"].shape[-2], st["idx_descale_q"].shape[-1])
    out = _ifc.qsli((), dict(
        q=q2, k=kc.view(kc.shape[0], kc.shape[1], kc.shape[3]),
        w=st["idx_w"].contiguous().view(-1, _IDX.N_HEADS).float(),
        descale_q=dq,
        candidate_block_indices=cand_i, candidate_block_length=cand_l,
        cu_seqlens_q=cu_t, seqused_q=None, seqused_k=seq_k, cmp_residual_k=None,
        block_table=bt2, output_idx_offset=None,
        metadata=md, topk=topk, candidate_block_size=cbs, quant_mode=1,
        max_seqlen_q=-1, mask_mode=3, cmp_ratio=ratio,
        layout_q="TND", layout_k="PA_BBND", return_value=False))
    sparse = out[0] if isinstance(out, (tuple, list)) else out
    # 诊断（DS41M_QS_DUMP=目录）：QSLI **输出**快照（环上限 A/B 时逐位对拍用）
    if (os.environ.get("DS41M_QS_DUMP")
            and int(ctx.get("idx", -1)) == int(os.environ.get("DS41M_QS_DUMP_LAYER", "24"))):
        _dd = os.environ["DS41M_QS_DUMP"]
        os.makedirs(_dd, exist_ok=True)
        torch.save({"kind": "packed", "sparse": sparse.detach().cpu(),
                    "cand_i": cand_i.detach().cpu(), "cand_l": cand_l.detach().cpu(),
                    "kc_shape": [int(x) for x in kc.shape]},
                   os.path.join(_dd, f"qsli_out_{os.environ.get('RANK', '0')}_packed.pt"))
    st["idx_sparse"] = sparse
    st["cmp_sparse_indices"] = sparse
    st["cmp_topk_length"] = (sparse >= 0).sum(-1).to(torch.int32)
    hold[f"_cmp_sparse_r{ratio}" + _CORE._sfx(ctx, "idx")] = sparse
    hold[f"_cmp_topklen_r{ratio}" + _CORE._sfx(ctx, "idx")] = st["cmp_topk_length"]
    if os.environ.get("DS41M_DEBUG"):
        print(f"      [qsli packed] ratio={ratio} lens={lens} bt{tuple(bt2.shape)} "
              f"→ sparse{tuple(sparse.shape)} 有效={int((sparse >= 0).sum())}", flush=True)
    return sparse

# ── 点位边界（core/memlife.py）：几何相关的元数据/rope 表缓存 ──────────────
# `_NZ_BY_NAME`（权重 NZ 布局）与 `_QSLI_TB` 的模板表**不登记**：与点位几何无关。
POINT_CACHES = (
    _MD_CACHE,        # (md, cu_q, seq_k, residual)：按 T/off/ratio
    _QS_CL_CACHE,     # QSLI 候选长度（按 T/off/ratio）
    _CS_CACHE,        # cos/sin 表（按 pos/off）
    _QSLI_TB,         # 旋转块表（按 nblk2，8 槽自淘汰；小但仍是几何）
)

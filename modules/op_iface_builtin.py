"""历史 Python 兜底表（**已弃用**；唯一事实源是 `modules/specs/model_modules.yaml`）。

`modules/op_iface.py` 默认从 YAML 加载；只有配置文件缺失/解析失败且
`DS41M_SPEC_STRICT=0` 时才回退到这里。保留它只为迁移/可移植包兜底，
新增或修改模块请改配置，不要改本文件。

本文件不导入 torch/exec_*；纯声明。
"""
from __future__ import annotations

from dataclasses import replace

from modules.spec import OpInterface
from modules.keys import K, W

# 插件分类（积木库分组 / 新模块的默认 category）
_CATEGORY = {
    "rmsnorm": "common", "hc_pre": "common", "hc_post": "common",
    "mul": "placeholder", "add": "placeholder", "cast": "placeholder",
    "tmove": "placeholder", "dyn_mx_quant": "implicit",
    "wq_a": "attention", "q_norm": "attention", "wq_b": "attention",
    "wkv": "attention", "kv_norm": "attention", "rope": "attention",
    "kv_epilog": "attention", "mqsmla": "attention",
    "wo_a": "attention", "wo_b": "attention",
    "kv_compress": "compressor",
    "idx_prolog": "indexer", "idx_k_write": "indexer",
    "qli": "indexer", "qsli": "indexer",
    "gate": "moe", "topk_gate": "moe", "mega_moe": "moe",
    "gating": "moe", "gating_sort": "moe", "reroute": "moe", "dispatch": "moe",
    "gmm": "moe", "swiglu_quant": "moe", "combine": "moe", "finalize": "moe",
    "gate_topk": "moe",
    "engram_fetch": "engram", "engram_wait": "engram", "engram_proj": "engram",
    "boundary_cut": "boundary", "gather_tail": "boundary", "scatter_tail": "boundary",
}


# ── 通用/注意力 ────────────────────────────────────────────────────────
OP_IFACE: dict[str, OpInterface] = {}


def _add(iface: OpInterface) -> None:
    if iface.op in OP_IFACE:
        raise KeyError(f"OP_IFACE 重复登记 {iface.op!r}")
    cat = _CATEGORY.get(iface.op)
    if cat and cat != iface.category:
        iface = replace(iface, category=cat)
    OP_IFACE[iface.op] = iface


def register_op(iface: OpInterface) -> OpInterface:
    """插件注册入口（`modules.plugin.module` 装饰器调用）。"""
    _add(iface)
    return OP_IFACE[iface.op]


_add(OpInterface(
    op="rmsnorm", impl="exec_ops:_rmsnorm", source=("exec_ops.py", 219, 224),
    requires=(K("hidden.x", "Tensor", "[T,H]", shadow="stream"),
              K("hc.stage", "str", storage="container", tier="container"),
              K("model.hidden"), K("model.rms_norm_eps")),
    provides=(K("hidden.x", "Tensor", "[T,H]", mutation="in_place", shadow="stream"),),
    weights=(W("attn_norm.weight", cond="stage=attn"),
             W("ffn_norm.weight", cond="stage=ffn")),
    doc="attn_norm / ffn_norm（按 hc.stage 选权重），原地归一化 x"))

_add(OpInterface(
    op="hc_pre", impl="exec_ops:_hc_pre", source=("exec_ops.py", 289, 363),
    requires=(K("hidden.x", "Tensor", "[T,hc_mult,H]", shadow="stream"),
              K("hc.stage", "str", storage="container", tier="container"),
              K("hc.pre_t", "Tensor", "[T,hc_mult]", optional=True),
              K("hc.ffn_pre", "Tensor", "[T,hc_mult]", storage="cache", optional=True),
              K("cache.root", "dict", optional=True, shadow="allow_multi"),
              K("model.hidden"), K("model.hc_mult"),
              K("model.hc_sinkhorn_iters"), K("model.hc_eps"),
              K("hc.pad_ref", optional=True), K("hc.pad_cap", optional=True),
              K("layer.index")),
    provides=(K("hc.pre_t", "Tensor", "[T,hc_mult]", shadow="allow_multi"),
              K("hc.post_t", "tuple[Tensor,Tensor]", shadow="allow_multi"),
              K("hidden.x", "Tensor", "[T,hc_mult,H]", mutation="in_place", shadow="stream"),
              K("hc.ffn_pre", "Tensor", "[T,hc_mult]", storage="cache",
                shadow="allow_multi", condition="stage=ffn")),
    weights=(W("hc_attn_fn", cond="stage=attn"), W("hc_attn_scale", cond="stage=attn"),
             W("hc_attn_base", cond="stage=attn"),
             W("hc_ffn_fn", cond="stage=ffn"), W("hc_ffn_scale", cond="stage=ffn"),
             W("hc_ffn_base", cond="stage=ffn")),
    doc="hyper-connection 预处理（npu_hc_pre_v2）：出 post/comb/pre；ffn 段的 pre 跨层传递"))

_add(OpInterface(
    op="hc_post", impl="exec_ops:_hc_post", source=("exec_ops.py", 366, 413),
    requires=(K("hidden.x", "Tensor", "[T,hc_mult,H]", shadow="stream"),
              K("hidden.residual", "Tensor", "[T,hc_mult,H]", shadow="stream"),
              K("hc.post_t", "tuple[Tensor,Tensor]"),
              K("hc.stage", "str", storage="container", tier="container"),
              K("process.backends", "dict", storage="st"), K("model.hidden"),
              K("model.hc_mult")),
    provides=(K("hidden.x", "Tensor", "[T,H]", mutation="replace", shadow="stream"),
              K("hidden.residual", "Tensor", "[T,H]", mutation="replace",
                shadow="stream", condition="stage=attn")),
    doc="hyper-connection 后处理（mhc_post）；attn 段后同时刷新 residual"))

_add(OpInterface(
    op="wq_a", impl="exec_ops:_wq_a", source=("exec_ops.py", 443, 445),
    requires=(K("hidden.x", "Tensor", "[T,H]"), K("model.hidden"),
              K("model.q_lora_rank")),
    provides=(K("attn.qr", "Tensor", "[T,q_lora]"),),
    weights=(W("attn.wq_a.weight", via="_packed", packed="attn.wq_a.weight"),
             W("attn.wq_a.scale", via="_packed", packed="attn.wq_a.weight")),
    doc="MLA q 低秩投影（mxfp8 QMM）"))

_add(OpInterface(
    op="q_norm", impl="exec_ops:_q_norm", source=("exec_ops.py", 448, 451),
    requires=(K("attn.qr", "Tensor", "[T,q_lora]"), K("model.q_lora_rank")),
    provides=(K("attn.qr", "Tensor", "[T,q_lora]", mutation="in_place"),),
    weights=(W("attn.q_norm.weight"),),
    doc="q_lora 上的 RmsNorm"))

_add(OpInterface(
    op="wq_b", impl="exec_ops:_wq_b", source=("exec_ops.py", 454, 457),
    requires=(K("attn.qr", "Tensor", "[T,q_lora]"), K("model.q_lora_rank"),
              K("model.head_dim"), K("model.n_attention_heads")),
    provides=(K("attn.q", "Tensor", "[T,n_heads,head_dim]"),),
    weights=(W("attn.wq_b.weight", via="_packed", packed="attn.wq_b.weight"),
             W("attn.wq_b.scale", via="_packed", packed="attn.wq_b.weight")),
    doc="q 上投影（→ 64 头 × 512）"))

_add(OpInterface(
    op="wkv", impl="exec_ops:_wkv", source=("exec_ops.py", 460, 462),
    requires=(K("hidden.x", "Tensor", "[T,H]"), K("model.hidden"),
              K("model.head_dim")),
    provides=(K("attn.kv", "Tensor", "[T,512]"),),
    weights=(W("attn.wkv.weight", via="_packed", packed="attn.wkv.weight"),
             W("attn.wkv.scale", via="_packed", packed="attn.wkv.weight")),
    doc="压缩 KV 投影（512 维）"))

_add(OpInterface(
    op="kv_norm", impl="exec_ops:_kv_norm", source=("exec_ops.py", 465, 467),
    requires=(K("attn.kv", "Tensor", "[T,512]"),),
    provides=(K("attn.kv", "Tensor", "[T,512]", mutation="in_place"),),
    weights=(W("attn.kv_norm.weight"),),
    doc="压缩 KV 上的 RmsNorm"))

_add(OpInterface(
    op="rope", impl="exec_ops:_rope", source=("exec_ops.py", 504, 523),
    requires=(K("hidden.x", "Tensor", "[T,H]"),
              K("attn.q", "Tensor", "[T,n_heads,head_dim]", optional=True),
              K("attn.kv", "Tensor", "[T,512]", optional=True),
              K("model.rope_dim"), K("model.rope_theta"),
              K("model.compress_rope_theta", "float", optional=True),
              K("model.rope_scaling", "dict", optional=True),
              K("position.offset", optional=True),
              K("position.offsets", "list[int]", optional=True),
              K("batch.cu", "list[int]", optional=True),
              K("batch.n_seqs", optional=True),
              K("layer.ratio", optional=True),
              K("cp.positions", "Tensor", optional=True),
              K("cp.abs_positions", "Tensor", optional=True),
              K("cp.total_q", optional=True),
              K("geometry.full_len", optional=True)),
    provides=(K("attn.q", "Tensor", mutation="in_place", shadow="stream"),
              K("attn.kv", "Tensor", mutation="in_place", shadow="stream"),
              K("attn.rope_done", "set", storage="st", shadow="hidden")),
    dispatch="done-set：q/kv 只 rope 一次（split 视图调用两次）",
    doc="inplace_partial_rotary_mul（q/kv 尾部 rope 维）"))

_add(OpInterface(
    op="kv_epilog", impl="exec_ops:_kv_epilog", source=("exec_ops.py", 925, 1017),
    requires=(K("attn.kv", "Tensor", "[T,512]"),
              K("position.offset"), K("geometry.full_len"),
              K("model.sliding_window"), K("model.head_dim"), K("model.win_keep"),
              K("position.win_abs", "bool"), K("batch.n_seqs"),
              K("batch.cu", "list[int]", optional=True),
              K("cache.root", "dict", optional=True, shadow="allow_multi"),
              K("cache.suffix.win", "str", optional=True),
              K("cache.win.pool", "Tensor", storage="cache", optional=True,
                shadow="allow_multi")),
    provides=(K("cache.win.pool", "Tensor", "[win_slots,544]", storage="cache",
                shadow="allow_multi"),
              K("cache.win.nblk", storage="cache", shadow="allow_multi")),
    doc="KV cache epilog：把 kv 写进滑窗池 A/B（KvCompressEpilogV2）"))

_add(OpInterface(
    op="dyn_mx_quant", impl=None, source=None,
    requires=(K("hidden.x", "Tensor", "[T,H]"),),
    flags=frozenset({"implicit", "trace_only", "no_impl"}),
    doc="★隐式算子：由 _qmm / wo_a 内部产生，执行侧不单独下发（只做 trace 计数）"))

_add(OpInterface(
    op="mqsmla", impl="exec_ops:_mqsmla", source=("exec_ops.py", 1503, 1905),
    requires=(K("attn.q", "Tensor", "[T,n_heads,head_dim]"),
              K("cmp.latent", "Tensor", "[T//ratio,512]", optional=True),
              K("idx.sparse", "Tensor", optional=True),
              K("cp.positions", "Tensor", optional=True),
              K("cp.total_q", optional=True),
              K("batch.cu", "list[int]", optional=True),
              K("batch.seq_lens", "list[int]", optional=True),
              K("batch.n_seqs"), K("layer.index"), K("layer.ratio"),
              K("model.index_topk"), K("position.offset"),
              K("position.offsets", "list[int]", optional=True),
              K("position.win_abs", "bool"),
              K("cache.root", "dict", optional=True, shadow="allow_multi"),
              K("cache.suffix.win", "str", optional=True),
              K("cache.suffix.cmp", "str", optional=True),
              K("cache.suffix.idx", "str", optional=True),
              K("cache.win.pool", "Tensor", storage="cache", shadow="allow_multi"),
              K("cache.cmp.pool", "Tensor", storage="cache", optional=True,
                shadow="allow_multi"),
              K("cache.cmp.len", "int|Tensor[B]", storage="cache", optional=True,
                shadow="allow_multi"),
              K("cache.idx.k", "Tensor", storage="cache", optional=True,
                shadow="allow_multi")),
    provides=(K("attn.out", "Tensor", "[T,n_heads*head_dim]"),),
    weights=(W("attn.attn_sink", "F32"),),
    doc="ds41 稀疏量化 MLA（窗口池 + 压缩索引）"))

_add(OpInterface(
    op="wo_a", impl="exec_ops:_wo_a", source=("exec_ops.py", 727, 765),
    requires=(K("attn.out", "Tensor", "[T,o_groups,head_dim*64]"),
              K("attn.q", "Tensor", optional=True),
              K("hidden.x", "Tensor", "[T,H]"),
              K("batch.cu", "list[int]", optional=True), K("batch.n_seqs"),
              K("layer.ratio"), K("position.offset"),
              K("position.offsets", "list[int]", optional=True),
              K("cp.positions", "Tensor", optional=True),
              K("model.rope_dim"), K("model.rope_theta", "float"),
              K("model.compress_rope_theta", "float", optional=True),
              K("model.rope_scaling", "dict", optional=True),
              K("process.allow_placeholder", "bool", storage="st"),
              K("process.placeholder_used", "set", storage="st", optional=True)),
    provides=(K("attn.o_lora", "Tensor", "[T,o_groups*o_lora]"),
              K("attn.out", "Tensor", mutation="in_place")),
    weights=(W("attn.wo_a.weight", via="_packed", packed="attn.wo_a.weight"),
             W("attn.wo_a.scale", via="_packed", packed="attn.wo_a.weight")),
    doc="o_lora 投影（TransposeQuantBatchMatMul；内含量化）"))

_add(OpInterface(
    op="wo_b", impl="exec_ops:_wo_b", source=("exec_ops.py", 768, 771),
    requires=(K("attn.o_lora", "Tensor", "[T,o_groups*o_lora]"),
              K("model.hidden"), K("model.o_groups"), K("model.o_lora_rank")),
    provides=(K("hidden.x", "Tensor", "[T,H]", mutation="replace", shadow="stream"),),
    weights=(W("attn.wo_b.weight", via="_packed", packed="attn.wo_b.weight"),
             W("attn.wo_b.scale", via="_packed", packed="attn.wo_b.weight")),
    doc="输出投影：o_groups×o_lora → hidden"))

# ── 压缩器 ────────────────────────────────────────────────────────────
_add(OpInterface(
    op="kv_compress", impl="exec_ops:_kv_compress", source=("exec_ops.py", 1085, 1404),
    requires=(K("hidden.x", "Tensor", "[T,H]"), K("model.hidden"),
              K("geometry.full_len"), K("position.offset"),
              K("position.offsets", "list[int]", optional=True),
              K("position.hit_replay_tail", optional=True),
              K("layer.index"), K("layer.ratio"), K("model.head_dim"),
              K("model.rope_dim"), K("model.compress_rope_theta"),
              K("model.rope_theta", "float", optional=True),
              K("model.rope_scaling", "dict", optional=True),
              K("cp.positions", "Tensor", optional=True),
              K("cache.root", "dict", optional=True, shadow="allow_multi"),
              K("cache.suffix.cmp", "str", optional=True),
              K("compressor.state_block", optional=True),
              K("compressor.t_override", optional=True),
              K("compressor.pre", "Tensor", optional=True),
              K("batch.n_seqs"), K("batch.cu", "list[int]", optional=True),
              K("cache.cmp.state", "Tensor", storage="cache", optional=True,
                shadow="allow_multi")),
    provides=(K("cmp.latent", "Tensor", "[T//ratio+1,512]"),
              K("cmp.kv", "Tensor", "[T//ratio+1,512]"),
              K("cache.cmp.pool", "Tensor", storage="cache", shadow="allow_multi"),
              K("cache.cmp.nblk", storage="cache", shadow="allow_multi"),
              K("cache.cmp.len", "int|Tensor[B]", storage="cache", shadow="allow_multi"),
              K("cache.cmp.state", "Tensor", storage="cache", shadow="allow_multi")),
    weights=(W("attn.compressor.wkv.weight"),
             W("attn.compressor.wgate.weight", cond="ratio>1"),
             W("attn.compressor.norm.weight"),
             W("attn.compressor.norm.weight.f32", via="synthetic",
               packed="attn.compressor.norm.weight.f32")),
    flags=frozenset({"packed_variant"}),
    dispatch="packed_if:batch.is_packed && !cp.active",
    doc="压缩 KV 写入（DeepInfer 口径；CP 时先 gather hidden 再跑全长）"))

# ── 索引器 ────────────────────────────────────────────────────────────
_add(OpInterface(
    op="idx_prolog", impl="exec_indexer:_idx_prolog", source=("exec_indexer.py", 154, 226),
    requires=(K("attn.qr", "Tensor", "[T,q_lora]"),
              K("hidden.x", "Tensor", "[T,H]"),
              K("batch.cu", "list[int]", optional=True),
              K("batch.n_seqs", optional=True),
              K("layer.ratio"), K("position.offset"),
              K("position.offsets", "list[int]", optional=True),
              K("model.head_dim"), K("model.index_head_dim"),
              K("model.index_n_heads"),
              K("model.rope_theta", "float", optional=True),
              K("model.rope_dim", optional=True),
              K("model.rope_scaling", "dict", optional=True),
              K("model.compress_rope_theta", "float", optional=True),
              K("cp.positions", "Tensor", optional=True),
              K("cp.abs_positions", "Tensor", optional=True),
              K("cp.total_q", optional=True)),
    provides=(K("idx.q", "Tensor", "[T,n_heads,64]"),
              K("idx.descale_q", "Tensor", "[T,n_heads,2,2]"),
              K("idx.w", "Tensor", "[T,n_heads]")),
    weights=(W("attn.indexer.wq_b.weight", via="_packed",
               packed="attn.indexer.wq_b.weight"),
             W("attn.indexer.wq_b.scale", via="_packed",
               packed="attn.indexer.wq_b.weight"),
             W("attn.indexer.weights_proj.weight"),
             W("attn.indexer.wq_b.scale.pairs", via="synthetic",
               packed="attn.indexer.wq_b.scale.pairs")),
    doc="索引器 query 侧（indexer_prologue_qw；CP 下按参考 M 补零）"))

_IDX_K_WEIGHTS = (
    W("attn.indexer.wk.weight"),
    W("attn.indexer.k_norm.weight"),
    W("attn.indexer.k_norm.weight.f32", via="synthetic",
      packed="attn.indexer.k_norm.weight.f32"),
)
_IDX_K_PROVIDES = (
    K("cache.idx.k", "Tensor", "[1+nblk,128,1,64]", storage="cache",
      shadow="allow_multi"),
    K("cache.idx.ks", "Tensor", "[1+nblk,128,1,2,2]", storage="cache",
      shadow="allow_multi"),
    K("cache.idx.nblk", storage="cache", shadow="allow_multi"),
    K("cache.idx.qsli", "Tensor", storage="cache", optional=True,
      shadow="allow_multi", condition="is_candidate"),
)

_add(OpInterface(
    op="idx_k_write", impl="exec_indexer:_idx_k_write",
    source=("exec_indexer.py", 513, 526),
    requires=(K("cmp.latent", "Tensor", "[N,512]"), K("layer.ratio"),
              K("position.offset"),
              K("position.offsets", "list[int]", optional=True),
              K("batch.cu", "list[int]", optional=True),
              K("batch.seq_lens", "list[int]", optional=True),
              K("batch.n_seqs", optional=True),
              K("cache.root", "dict", optional=True, shadow="allow_multi"),
              K("cp.positions", "Tensor", optional=True),
              K("model.rope_theta", "float", optional=True),
              K("model.rope_dim", optional=True),
              K("model.rope_scaling", "dict", optional=True),
              K("model.compress_rope_theta", "float", optional=True),
              K("model.candidate_block_size", optional=True),
              K("layer.roles.candidate_source", "bool")),
    provides=_IDX_K_PROVIDES,
    weights=_IDX_K_WEIGHTS,
    flags=frozenset({"boundary_only"}),
    doc="索引 K 缓存写入（边界 split 的 KV 路径；官方 update_indexer_cache）"))

_add(OpInterface(
    op="qli", impl="exec_indexer:_qli", source=("exec_indexer.py", 529, 539),
    requires=(K("idx.q", "Tensor", "[T,n_heads,64]"),
              K("idx.descale_q", "Tensor", "[T,n_heads,2,2]"),
              K("idx.w", "Tensor", "[T,n_heads]"),
              K("cmp.latent", "Tensor", optional=True),
              K("geometry.full_len"), K("position.offset"), K("layer.ratio"),
              K("position.offsets", "list[int]", optional=True),
              K("batch.cu", "list[int]", optional=True),
              K("batch.seq_lens", "list[int]", optional=True),
              K("batch.n_seqs", optional=True),
              K("cache.root", "dict", optional=True, shadow="allow_multi"),
              K("cache.suffix.idx", "str", optional=True),
              K("cp.positions", "Tensor", optional=True),
              K("model.rope_theta", "float", optional=True),
              K("model.rope_dim", optional=True),
              K("model.rope_scaling", "dict", optional=True),
              K("model.compress_rope_theta", "float", optional=True),
              K("model.candidate_block_size", optional=True),
              K("model.candidate_topk_blocks", optional=True),
              K("model.index_topk"), K("layer.roles.candidate_source", "bool"),
              K("indexer.k_ready", "bool", optional=True),
              K("cache.idx.k", "Tensor", storage="cache", optional=True,
                shadow="allow_multi")),
    provides=(K("idx.sparse", "Tensor", "[T,1,index_topk]"),
              K("idx.topk_length", "Tensor", "[T]", storage="cache",
                shadow="allow_multi"),
              K("cache.idx.k", "Tensor", storage="cache", shadow="allow_multi",
                condition="!idx_k_ready"),
              K("cache.idx.ks", "Tensor", storage="cache", shadow="allow_multi",
                condition="!idx_k_ready"),
              K("cache.idx.nblk", storage="cache", shadow="allow_multi",
                condition="!idx_k_ready"),
              K("cache.idx.qsli", "Tensor", storage="cache", shadow="allow_multi",
                condition="is_candidate")),
    weights=_IDX_K_WEIGHTS,
    flags=frozenset({"packed_variant"}),
    dispatch="packed_if:batch.is_packed && !cp.active",
    intrinsics=("idx_k_cache",),
    doc="QuantLightningIndexer（cp 下按块跑；内含 K 缓存写入）"))

_add(OpInterface(
    op="qsli", impl="exec_indexer:_qsli", source=("exec_indexer.py", 898, 907),
    requires=(K("idx.q", "Tensor"), K("idx.descale_q", "Tensor"),
              K("idx.w", "Tensor"), K("geometry.full_len"),
              K("position.offset"), K("position.offsets", "list[int]", optional=True),
              K("layer.ratio"),
              K("batch.cu", "list[int]", optional=True),
              K("batch.seq_lens", "list[int]", optional=True),
              K("batch.n_seqs", optional=True),
              K("cache.root", "dict", optional=True, shadow="allow_multi"),
              K("cache.suffix.idx", "str", optional=True),
              K("cp.positions", "Tensor", optional=True),
              K("model.candidate_block_size", optional=True),
              K("model.candidate_topk_blocks", optional=True),
              K("model.index_topk"), K("layer.roles.candidate_source", "bool"),
              K("cache.idx.qsli", "Tensor", storage="cache", optional=True,
                shadow="allow_multi")),
    provides=(K("idx.sparse", "Tensor"),
              K("idx.topk_length", "Tensor", storage="cache", shadow="allow_multi")),
    flags=frozenset({"packed_variant"}),
    dispatch="packed_if:batch.is_packed && !cp.active",
    doc="QuantSparseLightningIndexer（吃层 20 的候选块组合缓存）"))

_add(OpInterface(
    op="boundary_cut", impl="runner:_boundary_cut_dispatch",
    source=("runner.py", 1846, 1990),
    requires=(K("hidden.x", "Tensor", "[T,H]", shadow="stream"),
              K("hidden.residual", "Tensor", "[T,H]", optional=True),
              K("hc.pre_t", "Tensor", optional=True),
              K("hc.post_t", "tuple[Tensor,Tensor]", optional=True),
              K("attn.kv", "Tensor", optional=True),
              K("geometry.full_len"), K("batch.row_lens", "list[int]", optional=True),
              K("batch.seq_lens", "list[int]", optional=True),
              K("batch.n_seqs", optional=True),
              K("cache.root", "dict", optional=True, shadow="allow_multi"),
              K("cache.suffix.win", "str", optional=True),
              K("external.cp_state", "CpState|None", storage="external",
                tier="external", optional=True)),
    provides=(K("geometry.rows", mutation="replace"),
              K("position.offset", mutation="replace",
                roles=("compact_start",)),
              K("batch.cu", "list[int]", mutation="replace"),
              K("indexer.k_ready", "bool"),
              K("hidden.residual", mutation="in_place"),
              K("hc.pre_t", mutation="in_place"),
              K("hc.post_t", mutation="in_place"),
              K("attn.kv", mutation="in_place", condition="cp.active")),
    flags=frozenset({"boundary_only", "runner_hook"}),
    doc="边界层截尾（host 钩子，非 OPS 函数）：hidden/residual/hc/q/idx_*/kv → keep 行"))

# ── FFN / MoE ────────────────────────────────────────────────────────
_add(OpInterface(
    op="gate", impl="exec_ops:_gate", source=("exec_ops.py", 774, 783),
    requires=(K("hidden.x", "Tensor", "[T,H]"), K("model.hidden"),
              K("model.n_routed_experts")),
    provides=(K("moe.logits", "Tensor", "[T,n_experts]"),),
    weights=(W("ffn.gate.weight"), W("ffn.gate.weight.T", via="synthetic",
                                     packed="ffn.gate.weight.T")),
    doc="路由打分（FusedMatMul：T×5120 @ 384×5120）"))

_add(OpInterface(
    op="topk_gate", impl="exec_ops:_topk_gate", source=("exec_ops.py", 786, 821),
    requires=(K("moe.logits", "Tensor", "[T,n_experts]"),
              K("input.ids", "Tensor", optional=True),
              K("model.routed_scaling_factor"), K("model.moe_topk")),
    provides=(K("moe.topk_idx", "Tensor", "[T,topk]"),
              K("moe.topk_weight", "Tensor", "[T,topk]")),
    weights=(W("ffn.gate.bias"), W("ffn.gate.bias.f32", via="synthetic",
                                   packed="ffn.gate.bias.f32")),
    doc="top-k 路由（bias 转 fp32）"))

_add(OpInterface(
    op="mega_moe", impl="exec_moe:_mega_moe", source=("exec_moe.py", 919, 1088),
    requires=(K("hidden.x", "Tensor", "[T,H]"),
              K("moe.topk_idx", "Tensor", "[T,topk]"),
              K("moe.topk_weight", "Tensor", "[T,topk]"),
              K("geometry.full_len"), K("layer.index"),
              K("model.hidden"), K("model.moe_topk"),
              K("moe.e_local"), K("geometry.chunk"), K("moe.nspan", optional=True),
              K("model.moe_intermediate"), K("model.swiglu_limit"),
              K("process.src", storage="ctx", optional=True)),
    provides=(K("moe.out", "Tensor", "[T,H]"),
              K("hidden.x", "Tensor", "[T,H]", mutation="replace", shadow="stream")),
    weights=(W("ffn.experts.{E}.w1.weight"), W("ffn.experts.{E}.w1.scale"),
             W("ffn.experts.{E}.w2.weight"), W("ffn.experts.{E}.w2.scale"),
             W("ffn.experts.{E}.w3.weight"), W("ffn.experts.{E}.w3.scale"),
             W("ffn.shared_experts.w1.weight"), W("ffn.shared_experts.w1.scale"),
             W("ffn.shared_experts.w2.weight"), W("ffn.shared_experts.w2.scale"),
             W("ffn.shared_experts.w3.weight"), W("ffn.shared_experts.w3.scale")),
    flags=frozenset({"fused"}),
    intrinsics=("dispatch", "gmm", "swiglu_quant", "combine", "finalize"),
    doc="融合 MoE（MegaMoE：路由/分发/分组MM/SwiGLU/合并；按 chunk 分 span）"))

# ── engram ───────────────────────────────────────────────────────────
_add(OpInterface(
    op="engram_fetch", impl="exec_engram:_engram_fetch",
    source=("exec_engram.py", 598, 649),
    requires=(K("hidden.x", "Tensor", "[T,H]"),
              K("input.ids", "Tensor", "[B,L]", optional=True),
              K("layer.index"),
              K("model.engram_rows"), K("model.engram_tp_size"),
              K("process.engram_tp_rank")),
    provides=(K("engram.ids", "Tensor", "[T,n_heads]", storage="st"),
              K("engram.future", "object", storage="st", shadow="hidden")),
    weights=(W("engram.embed.weight"), W("engram.embed.scale")),
    doc="ngram hash + 取表（可走侧流/厂商 ElasticBuffer）"))

_add(OpInterface(
    op="engram_wait", impl="exec_engram:_engram_wait",
    source=("exec_engram.py", 652, 730),
    requires=(K("engram.ids", "Tensor", storage="st", optional=True),
              K("engram.future", "object", storage="st", optional=True),
              K("hidden.x", "Tensor", "[T,H]"), K("layer.index")),
    provides=(K("engram.kv", "Tensor", "[T,6144]"),),
    doc="join 后台取表 → embeddings（H2D）"))

_add(OpInterface(
    op="engram_proj", impl="exec_engram:_engram_proj",
    source=("exec_engram.py", 771, 783),
    requires=(K("engram.kv", "Tensor", "[T,6144]"),
              K("hidden.x", "Tensor", "[T,H]"), K("model.hc_mult"),
              K("model.hidden"), K("model.rms_norm_eps", "float"),
              K("layer.index")),
    provides=(K("engram.proj_out", "Tensor", "[T,hc_mult+1,H]"),
              K("engram.out", "Tensor", "[T,H]"),
              K("hidden.x", "Tensor", "[T,H]", mutation="replace", shadow="stream"),
              K("hidden.residual", "Tensor", "[T,H]", mutation="replace",
                shadow="stream")),
    weights=(W("engram.wkv.weight", via="_packed", packed="engram.wkv.weight"),
             W("engram.wkv.scale", via="_packed", packed="engram.wkv.weight"),
             W("engram.q_weight"), W("engram.k_weight")),
    doc="engram 投影 + gate（覆盖残差流；官方 DecoderLayer 第一步）"))

# ── 未融合 MoE 链（recipes moe.unfused；当前配置不执行）────────────────
for _op in ("gating", "gating_sort", "reroute", "dispatch", "gmm",
            "swiglu_quant", "combine", "finalize"):
    _add(OpInterface(
        op=_op, impl=None, source=None,
        requires=(K("hidden.x", "Tensor", "[T,H]"),),
        flags=frozenset({"no_impl"}),
        doc="未融合 MoE 链（megamoe=none 才走；当前配置被 mega_moe 取代）"))

# ── 占位/未用算子 ────────────────────────────────────────────────────
for _op in ("mul", "add", "cast", "tmove"):
    _add(OpInterface(
        op=_op, impl=f"exec_ops:_{_op}", source=("exec_ops.py", 1910, 1926),
        requires=(K("hidden.x", "Tensor", "[T,H]"),),
        provides=(K("hidden.x", "Tensor", mutation="in_place"),),
        flags=frozenset({"placeholder"}),
        doc="纯占位（数值恒等；默认不下发，DS41M_PAD_OPS=1 才进 trace）"))

for _op in ("gate_topk", "gather_tail", "scatter_tail"):
    _add(OpInterface(
        op=_op, impl=None, source=None,
        requires=(K("hidden.x", "Tensor", "[T,H]"),),
        flags=frozenset({"no_impl", "declared_only"}),
        doc="opgraph 声明但当前配方未使用"))

# ── 一致性自检（导入时）──────────────────────────────────────────────
assert len(OP_IFACE) == 41, f"OP_IFACE 条目数异常：{len(OP_IFACE)}"


def get(op: str) -> OpInterface:
    return OP_IFACE[op]


def all_ops() -> tuple[str, ...]:
    return tuple(OP_IFACE)

"""历史 Python 兜底表（**已弃用**；唯一事实源是 `modules/specs/key_names.yaml`）。

`modules/names.py` 默认从 YAML 加载；只有配置文件缺失/解析失败且
`DS41M_SPEC_STRICT=0` 时才回退到这里。改键名/别名请改配置，不要改本文件。

命名空间（第一个点之前）：
  model.*     模型常量（hidden/head_dim/ratio 的静态部分）
  geometry.*  本次 forward 的几何（行数/全长/分块）
  position.*  位置语义（rope/窗口/命中起点/decode 位置）
  batch.*     packed 批次（cu/seq_lens/row_lens/n_seqs）
  layer.*     层身份与角色
  hidden.*    残差流（x/residual）
  attn.*      注意力激活与中间量
  cmp.*       压缩器激活
  idx.*       索引器激活
  moe.*       MoE 激活与路由
  engram.*    engram 激活
  hc.*        hyper-connection 瞬态
  cp.*        上下文并行（本地/全局位置）
  cache.*     跨层共享池与行账（不走词法作用域）
  process.*   进程级（device/src/backends）
  external.*  进程级全局（CP_STATE 等）
  input.*     输入（input_ids）
"""
from __future__ import annotations

# ── legacy 扁平键 → 规范键 ──────────────────────────────────────────────
# 只登记"运行时代码里真实出现"的键；新增/改名时必须同步 OP_IFACE 并跑
# tools/check_module_spec.py。
LEGACY: dict[str, str] = {
    # ctx：模型/几何
    "H": "model.hidden",
    "I": "model.moe_intermediate",
    "T": "geometry.rows",
    "chunk": "geometry.chunk",
    "L_full": "geometry.full_len",
    "ratio": "layer.ratio",
    "rtopk": "model.moe_topk",
    "e_local": "moe.e_local",
    "S": "model.sliding_window",
    "idx": "layer.index",
    "head_dim": "model.head_dim",
    "index_topk": "model.index_topk",
    "rope_dim": "model.rope_dim",
    # ctx：层角色
    "is_kv_src": "layer.roles.kv_source",
    "is_idx_src": "layer.roles.index_source",
    "is_candidate": "layer.roles.candidate_source",
    "engram": "layer.roles.engram",
    # ctx：位置/批次
    "pos_offset": "position.offset",
    "pos_offsets": "position.offsets",
    "hit_replay_tail": "position.hit_replay_tail",
    "win_abs": "position.win_abs",
    "win_keep": "model.win_keep",
    "input_ids": "input.ids",
    "cu_seqlens": "batch.cu",
    "seq_lens": "batch.seq_lens",
    "row_lens": "batch.row_lens",
    "n_seqs": "batch.n_seqs",
    "is_decode_step": "position.is_decode",
    # ctx：CP
    "cp_positions": "cp.positions",
    "cp_abs_positions": "cp.abs_positions",
    "cp_total_q": "cp.total_q",
    "_cp_full": "cp.full",
    "_cp_chunk": "cp.chunk",
    # ctx：压缩器
    "state_block": "compressor.state_block",
    "_cmp_T": "compressor.t_override",
    "_cmp_pre": "compressor.pre",
    "cmp_len": "cache.cmp.len",
    # ctx：HC
    "hc_mult": "model.hc_mult",
    "hc_sinkhorn_iters": "model.hc_sinkhorn_iters",
    "hc_eps": "model.hc_eps",
    "hc_pad_ref": "hc.pad_ref",
    "hc_pad_cap": "hc.pad_cap",
    # ctx：其它常量
    "routed_scaling_factor": "model.routed_scaling_factor",
    "topk": "model.moe_topk",
    "swiglu_limit": "model.swiglu_limit",
    "eps": "model.rms_norm_eps",
    "rope_theta": "model.rope_theta",
    "compress_rope_theta": "model.compress_rope_theta",
    "rope_scaling": "model.rope_scaling",
    "device": "process.device",
    "_src": "process.src",
    "_backends": "process.backends",
    # st：残差流/注意力
    "x": "hidden.x",
    "residual": "hidden.residual",
    "qr": "attn.qr",
    "q": "attn.q",
    "kv": "attn.kv",
    "attn_out": "attn.out",
    "o_lora": "attn.o_lora",
    "_rope_done": "attn.rope_done",
    # st：压缩/索引
    "cmp_latent": "cmp.latent",
    "cmp_kv": "cmp.kv",
    "cmp_sparse_indices": "idx.sparse",
    "cmp_topk_length": "idx.topk_length",
    "idx_sparse": "idx.sparse",
    "idx_q": "idx.q",
    "idx_descale_q": "idx.descale_q",
    "idx_w": "idx.w",
    # st：按层缓存后缀（_sfx 读写；OVERLOADS 里已有规范名）
    "suffix_win": "cache.suffix.win",
    "suffix_cmp": "cache.suffix.cmp",
    "suffix_idx": "cache.suffix.idx",
    # ctx：索引器 / MoE / engram 的旁路常量
    "idx_k_ready": "indexer.k_ready",
    "moe_nspan": "moe.nspan",
    "candidate_block_size": "model.candidate_block_size",
    "candidate_topk_blocks": "model.candidate_topk_blocks",
    "engram_rows": "model.engram_rows",
    "engram_tp_size": "model.engram_tp_size",
    "engram_tp_rank": "process.engram_tp_rank",
    # st：MoE
    "logits": "moe.logits",
    "topk_idx": "moe.topk_idx",
    "topk_weight": "moe.topk_weight",
    "moe_out": "moe.out",
    # st：engram
    "engram_kv": "engram.kv",
    "engram_proj_out": "engram.proj_out",
    "engram_out": "engram.out",
    # st：HC
    "hc_pre_t": "hc.pre_t",
    "hc_post_t": "hc.post_t",
    "_norm_stage": "hc.stage",
    "_ffn_pre": "hc.ffn_pre",
    "_engram_fut": "engram.future",
    "_w": "process.weights",
    "_shared": "cache.root",
    "_allow_placeholder": "process.allow_placeholder",
    "_placeholder_used": "process.placeholder_used",
}

# ── 多义键（同名不同义，导出器打 badge）──────────────────────────────
OVERLOADS: dict[str, tuple[str, ...]] = {
    "position.offset": ("hit/replay segment start", "decode absolute position",
                        "compact segment start"),
    "geometry.rows": ("layer['T'] hint (stale after replay)", "effective st['x'] rows"),
    "indexer.k_ready": ("boundary compact no-rewrite flag",
                        "CP inferred from cache presence"),
    "cache.cmp.len": ("int (single sequence)", "Tensor[B] (packed)"),
    "batch.n_seqs": ("explicit n_seqs", "derived from batch.cu"),
    "batch.seq_lens": ("logical length", "physical rows without hit", "CP local lens"),
    "cache.suffix.win": ("per-layer window pool suffix",),
    "cache.suffix.cmp": ("compressor source-owner suffix",),
    "cache.suffix.idx": ("indexer source-owner suffix",),
    "position.offsets": ("per-request hit start (packed hit)", "decode positions"),
    "hc.pad_ref": ("M padding reference for hc_pre (CP/boundary)",),
    "compressor.state_block": ("state ring block size", "candidate block size"),
}

# ── 语义命名空间 → tier 的默认值（KeySpec 未显式给 tier 时用）──────────
_TIER_BY_PREFIX = {
    "model": "const",
    "process": "const",
    "geometry": "pass",
    "position": "pass",
    "batch": "pass",
    "cp": "pass",
    "input": "pass",
    "layer": "layer",
    "hidden": "module",
    "attn": "module",
    "cmp": "module",
    "idx": "module",
    "moe": "module",
    "engram": "module",
    "hc": "module",
    "cache": "cache",
    "external": "external",
}


# ── 规范名 → 全部 legacy 别名（反向索引，运行期 O(1)）───────────────
ALIASES_BY_CANON: dict[str, tuple[str, ...]] = {}
for _raw, _canon in LEGACY.items():
    ALIASES_BY_CANON.setdefault(_canon, []).append(_raw)
ALIASES_BY_CANON = {k: tuple(v) for k, v in ALIASES_BY_CANON.items()}


def aliases_for(canonical_name: str) -> tuple[str, ...]:
    """规范键对应的所有 legacy 原始键（含自身？不含；自身由调用方先查）。"""
    return ALIASES_BY_CANON.get(canonical_name, ())


def canonical(key: str) -> str:
    """legacy/规范键 → 规范键。"""
    return LEGACY.get(key, key)


def namespace(key: str) -> str:
    return canonical(key).split(".", 1)[0]


def default_tier(key: str) -> str:
    return _TIER_BY_PREFIX.get(namespace(key), "module")


def is_overloaded(key: str) -> bool:
    return canonical(key) in OVERLOADS


def overload_roles(key: str) -> tuple[str, ...]:
    return OVERLOADS.get(canonical(key), ())

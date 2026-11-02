"""op 之外的上下文提供者（runner / ctx_for / st 初始化 / 缓存 / 全局）。

导出器用它做静态解析：每个 require 先查"最近祖先 + 前序兄弟"，再查这里的
process/pass/layer/st/cache/external 提供者。`source` 是给人看的代码位置。
"""
from __future__ import annotations

from dataclasses import dataclass

from modules import names


@dataclass(frozen=True)
class Provider:
    key: str
    tier: str
    source: str
    condition: str | None = None
    doc: str = ""

    def to_json(self) -> dict:
        d = dict(key=self.key, tier=self.tier, source=self.source)
        if self.condition:
            d["condition"] = self.condition
        if self.doc:
            d["doc"] = self.doc
        return d


# 模型常量（值来自 ModelSpec；这里只声明"谁提供"）
_MODEL_KEYS = {
    "model.hidden": "hidden",
    "model.moe_intermediate": "moe_intermediate",
    "model.head_dim": "head_dim",
    "model.q_lora_rank": "q_lora_rank",
    "model.o_lora_rank": "o_lora_rank",
    "model.o_groups": "o_groups",
    "model.n_attention_heads": "n_attention_heads",
    "model.n_routed_experts": "n_routed_experts",
    "model.moe_topk": "topk",
    "model.routed_scaling_factor": "routed_scaling_factor",
    "model.rms_norm_eps": "rms_norm_eps",
    "model.hc_mult": "hc_mult",
    "model.hc_sinkhorn_iters": "hc_sinkhorn_iters",
    "model.hc_eps": "hc_eps",
    "model.sliding_window": "sliding_window",
    "model.win_keep": "win_keep",
    "model.index_topk": "index_topk",
    "model.index_head_dim": "index_head_dim",
    "model.index_n_heads": "index_n_heads",
    "model.compress_rope_theta": "compress_rope_theta",
    "model.rope_dim": "qk_rope_head_dim",
    "model.rope_theta": "rope_theta",
    "model.rope_scaling": "rope_scaling",
    "model.swiglu_limit": "swiglu_limit",
    "model.candidate_block_size": "candidate_block_size",
    "model.candidate_topk_blocks": "candidate_topk_blocks",
    # engram 表几何不是 ModelSpec 字段，由 runner 的 weight_src.engram_placement 计算
    "model.engram_rows": "engram rows/rank（runner._ENG，weight_src.engram_placement）",
    "model.engram_tp_size": "engram_tp_size（runner._ENG）",
    "process.engram_tp_rank": "_RANK（runner）",
}

_PASS_KEYS = {
    "geometry.rows": "runner.py:1559-1638（层 T 提示；replay 后以 st.x 行数为准）",
    "geometry.chunk": "run.ctx_for:59-75（moe_chunk_max_len）",
    "geometry.full_len": "runner.py:1559-1638（L_full）",
    "geometry.nchunks": "geometry.rows / geometry.chunk",
    "position.offset": "runner.py:1559-1638（pos_offset）",
    "position.offsets": "runner.py:1559-1638（packed hit 的逐请求起点）",
    "position.win_abs": "run.ctx_for:59-75（PERLAYER_CACHE）",
    "position.hit_replay_tail": "runner.py:1559-1638",
    "position.is_decode": "decode 路径（decode_layout）",
    "batch.cu": "runner.py:1559-1638（cu_seqlens）",
    "batch.seq_lens": "runner.py:1559-1638",
    "batch.row_lens": "runner.py:1559-1638",
    "batch.n_seqs": "runner.py:1559-1638",
    "cp.positions": "runner.py:1562-1580（CP_STATE）",
    "cp.abs_positions": "runner.py:1562-1580（CP_STATE）",
    "cp.total_q": "runner.py:1562-1580（CP_STATE）",
    "cp.chunk": "exec_ops._cp_of（CP 分块标志）",
    "hc.pad_ref": "runner.py:1572-1580（CP/boundary M 参考）",
    "hc.pad_cap": "runner.py:1572-1580",
    "input.ids": "runner forward 入参",
    "compressor.state_block": "runner.py:1559-1638（可缺省）",
    "compressor.t_override": "exec_ops._kv_compress（CP full 视角）",
    "compressor.pre": "exec_ops._kv_compress（DeepInfer 内核路径）",
    "moe.e_local": "run.ctx_for:59-75（EP 切分）",
    "moe.nspan": "runner.py:1587-1595（per-DP dummy span）",
    "indexer.k_ready": "runner.py:1846-1990（boundary compact 只读）",
    "process.device": "run.ctx_for:59-75",
    "process.src": "runner.py:1561（decode MoE 的 NZ 缓存来源）",
}

_LAYER_KEYS = ("layer.index", "layer.ratio", "layer.kind",
               "layer.roles.kv_source", "layer.roles.index_source",
               "layer.roles.candidate_source", "layer.roles.engram")

_ST_INIT = ("hidden.x", "hidden.residual", "process.weights", "cache.root",
            "process.backends", "process.allow_placeholder",
            "process.placeholder_used")


def providers_for(layer: dict, cfg: dict) -> list[Provider]:
    """→ 本层可见的 process/pass/layer/st/external 提供者。"""
    out: list[Provider] = []
    for k in _MODEL_KEYS:
        out.append(Provider(k, "const", "model_spec.build(cfg) / run.ctx_for:59-75"))
    for k, src in _PASS_KEYS.items():
        out.append(Provider(k, "pass", src))
    roles = dict(
        kv_source=bool(layer.get("is_kv_src")),
        index_source=bool(layer.get("is_idx_src")),
        candidate_source=bool(layer.get("is_candidate")),
        engram=bool(layer.get("engram")),
    )
    for k in _LAYER_KEYS:
        if k.startswith("layer.roles."):
            role = k.rsplit(".", 1)[-1]
            out.append(Provider(k, "layer", "assemble.build_stack:145-156",
                                condition=("true" if roles.get(role) else "false")))
        elif k == "layer.ratio":
            out.append(Provider(k, "layer", "assemble.build_stack:145-156",
                                doc=f"ratio={layer.get('ratio')}"))
        else:
            out.append(Provider(k, "layer", "assemble.build_stack:145-156"))
    for k in _ST_INIT:
        out.append(Provider(k, "st", "runner.py:1701-1703（st 初始化）"))
    out.append(Provider("external.cp_state", "external",
                        "exec_ops.CP_STATE（runner 每 forward 设置）", condition="cp.active"))
    # cache.* 由 providers.resolve 用"最近前序写者"解析，这里只声明命名空间
    out.append(Provider("cache.*", "cache",
                        "kv_manager.py:217-260 / exec_*._hold(st)（按写者解析）"))
    return out


def canonical(key: str) -> str:
    return names.canonical(key)

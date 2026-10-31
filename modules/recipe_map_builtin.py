"""历史 Python 兜底表（**已弃用**；唯一事实源是 `modules/specs/recipe_map.yaml`）。

`modules/recipe_map.py` 默认从 YAML 加载；只有配置文件缺失/解析失败且
`DS41M_SPEC_STRICT=0` 时才回退到这里。新增容器/路径请改配置，
不要改本文件。
"""
from __future__ import annotations

from modules.spec import ContainerInfo

# ── 容器展示信息（按树深度优先；未列出的容器自动用末段名）─────────────
CONTAINERS: dict[str, ContainerInfo] = {}


def _c(path, title, doc="", order=0):
    CONTAINERS[path] = ContainerInfo(path, title, doc, order)


_c("attn", "Attention", "注意力块：HC → norm → q/kv → rope → cache → 压缩/索引 → 核心 → 输出投影")
_c("attn.hc", "Hyper-connection (attn)", "attn 段的 hc_pre / hc_post")
_c("attn.q_proj", "Q projection", "wq_a → q_norm → wq_b（MLA 低秩）")
_c("attn.kv_proj", "KV projection", "wkv → kv_norm（+ split 视图的 kv rope）")
_c("attn.compressor", "Compressor", "压缩 KV 写入（kv_src 层）")
_c("attn.indexer", "Indexer", "索引器（kv_src / index_src 层）")
_c("attn.indexer.q_proj", "Indexer query", "idx_prolog")
_c("attn.kv_indexer", "Indexer K cache (KV path)",
   "索引 K 缓存写入（split 的 KV 路径；在 boundary_cut 之前）")
_c("attn.indexer.search", "Sparse search", "QLI / QSLI（稀疏索引）")
_c("attn.core", "Attention core", "mqsmla（窗口 + 压缩 MLA）")
_c("attn.o_proj", "Output projection", "wo_a → wo_b")
_c("ffn", "FFN / MoE", "FFN 块：HC → norm → 路由 → 专家 → HC")
_c("ffn.hc", "Hyper-connection (ffn)", "ffn 段的 hc_pre / hc_post")
_c("ffn.moe", "MoE", "融合 mega_moe 或未融合链")
_c("ffn.moe.routing", "Routing", "gate/topk_gate（或 gating 链）")
_c("ffn.moe.experts", "Experts", "专家分组 MM / SwiGLU")
_c("engram", "Engram", "层首 engram：hash → 取表 → 投影/gate")
_c("layer.epilogue", "Elementwise epilogue", "占位小算子（默认不下发）")

# ── (segment, op, occurrence) → 层内路径 ──────────────────────────────
# segment 是展开后的**最内层**命名段：common / attn_core / attn_tail /
# attention.<kind> / moe.<key> / engram。
PATHS: dict[tuple[str, str, int], str] = {}


def _p(segment, op, path, occ=0):
    PATHS[(segment, op, occ)] = path


# common：两次 hc_pre / rmsnorm / hc_post 由 attn / ffn 容器区分
_p("common", "hc_pre", "attn.hc.pre", 0)
_p("common", "hc_pre", "ffn.hc.pre", 1)
_p("common", "rmsnorm", "attn.norm", 0)
_p("common", "rmsnorm", "ffn.norm", 1)
_p("common", "wo_b", "attn.o_proj.wo_b")
_p("common", "hc_post", "attn.hc.post", 0)
_p("common", "hc_post", "ffn.hc.post", 1)
for _op in ("mul", "add", "cast", "tmove"):
    _p("common", _op, f"layer.epilogue.{_op}")

# attn_core / attn_tail
_p("attn_core", "wq_a", "attn.q_proj.wq_a")
_p("attn_core", "q_norm", "attn.q_proj.q_norm")
_p("attn_core", "wq_b", "attn.q_proj.wq_b")
_p("attn_core", "wkv", "attn.kv_proj.wkv")
_p("attn_core", "kv_norm", "attn.kv_proj.kv_norm")
_p("attn_core", "rope", "attn.rope")
_p("attn_core", "kv_epilog", "attn.kv_epilog")
_p("attn_core", "dyn_mx_quant", "attn.dyn_mx_quant")
_p("attn_tail", "mqsmla", "attn.core.mqsmla")
_p("attn_tail", "wo_a", "attn.o_proj.wo_a")

# attention.<kind> 里的直接算子
for _seg in ("attention.win", "attention.c2a_source", "attention.c2a_consumer",
             "attention.c1a_source", "attention.c1a_indexsource",
             "attention.c1a_consumer"):
    _p(_seg, "kv_compress", "attn.compressor.kv_compress")
    _p(_seg, "idx_prolog", "attn.indexer.q_proj.idx_prolog")
    _p(_seg, "qli", "attn.indexer.search.qli")
    _p(_seg, "qsli", "attn.indexer.search.qsli")

# split 视图：两个 rope
_SEG = "attention.c1a_source_split"
_p(_SEG, "wkv", "attn.kv_proj.wkv")
_p(_SEG, "kv_norm", "attn.kv_proj.kv_norm")
_p(_SEG, "rope", "attn.kv_proj.rope", 0)
_p(_SEG, "kv_epilog", "attn.kv_epilog")
_p(_SEG, "kv_compress", "attn.compressor.kv_compress")
_p(_SEG, "idx_k_write", "attn.kv_indexer.idx_k_write")
_p(_SEG, "boundary_cut", "attn.boundary_cut")
_p(_SEG, "wq_a", "attn.q_proj.wq_a")
_p(_SEG, "q_norm", "attn.q_proj.q_norm")
_p(_SEG, "wq_b", "attn.q_proj.wq_b")
_p(_SEG, "rope", "attn.q_proj.rope", 1)
_p(_SEG, "idx_prolog", "attn.indexer.q_proj.idx_prolog")
_p(_SEG, "qli", "attn.indexer.search.qli")

# moe
_p("moe.fused", "gate", "ffn.moe.routing.gate")
_p("moe.fused", "topk_gate", "ffn.moe.routing.topk_gate")
_p("moe.fused", "mega_moe", "ffn.moe.experts.mega_moe")
_p("moe.unfused", "gating", "ffn.moe.routing.gating")
_p("moe.unfused", "gating_sort", "ffn.moe.routing.gating_sort")
_p("moe.unfused", "reroute", "ffn.moe.routing.reroute")
_p("moe.unfused", "dispatch", "ffn.moe.dispatch")
_p("moe.unfused", "gmm", "ffn.moe.experts.gmm")
_p("moe.unfused", "swiglu_quant", "ffn.moe.experts.swiglu_quant")
_p("moe.unfused", "combine", "ffn.moe.combine")
_p("moe.unfused", "finalize", "ffn.moe.finalize")

# engram
_p("engram", "engram_fetch", "engram.fetch")
_p("engram", "engram_wait", "engram.wait")
_p("engram", "engram_proj", "engram.proj")


def path_of(segment: str, op: str, occurrence: int) -> str | None:
    return PATHS.get((segment, op, occurrence))

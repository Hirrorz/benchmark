"""mini 模型结构参数的**单一事实源**（2026-10-19）。

背景：结构参数以前散落在三处 —— exec_* 的模块级常量、weights_manifest/engram_layout
的字面层号列表、assemble/runner 的 magic number。config 的 `model:` 段其实已经承载了
大部分，但模块不读它 → 换结构必须改代码，配置写错也没有校验。

本模块：
  · `ModelSpec`：所有结构/维度/尺寸参数，默认值 **逐字等于** 改造前的硬编码值
    → 现有 config 的行为逐位不变。
  · `build(cfg)`：cfg["model"]/["run"]/["parallel_config"] 覆盖默认值。
  · `validate(spec, cfg)`：聚合式校验（长度/范围/整除/对齐），返回错误列表。
  · `set_spec(cfg)` / `get()`：进程级单例；未设置时 `get()` 返回默认 spec
    （tools/离线脚本可直接用）。

约定：
  · 只有**格式不变量**（NZ=29、MXFP8 块=32 这类硬件/算子契约）可以继续写在代码里；
  · 其它一切尺寸都走这里；新增配置键全部可选，缺省 = 今天的值。
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field


# 默认 compress_ratios（= configs/mini_ds41_flash_hit97_64k.yaml 的 model.compress_ratios）
_DEFAULT_RATIOS = [0, 0] + [2] * 18 + [1] * 20


@dataclass
class ModelSpec:
    # ── 结构 ──
    hidden: int = 5120
    moe_intermediate: int = 2304
    n_layers: int = 40
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    topk: int = 6
    sliding_window: int = 128
    compress_ratios: list = field(default_factory=lambda: list(_DEFAULT_RATIOS))
    kv_source_layers: list = field(default_factory=lambda: [2, 8, 14, 20])
    index_source_layers: list = field(default_factory=lambda: [2, 8, 14, 20, 24, 28, 32, 36])
    candidate_source_layer: int = 20
    engram_layers: list = field(default_factory=lambda: [1, 14])

    # ── 维度 ──
    head_dim: int = 512                # MLA 主 head dim（≠ indexer 的 128）
    n_attention_heads: int = 64
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    index_n_heads: int = 32            # index_n_heads
    index_head_dim: int = 128          # index_head_dim
    index_topk: int = 512
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_scaling: dict = field(default_factory=dict)
    vocab: int = 129280

    # ── 数值 ──
    rms_norm_eps: float = 1e-20
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0

    # ── 尺寸 / 格式（格式不变量允许有代码默认，但仍集中在这里便于查）──
    block_size: int = 128              # attention page / 压缩块（算子契约：必须 128）
    quant_block: int = 32              # mxfp8 量化块
    scale_pair: int = 64               # indexer E8M0 scale 末维配对宽度
    nz_format: int = 29                # FRACTAL_NZ（acl format）
    kv_row_width: int = 544            # 窗口池行宽 = head_dim(512) + scale(32)
    cmp_row_width: int = 320           # 压缩池行宽 = mxfp4(256) + scale(64)
    win_pool_slots: int = 384          # = 3 × sliding_window
    candidate_block_size: int = 8      # 候选块 (2048 × 8)
    candidate_topk_blocks: int = 2048
    engram_prefetch_T: int = 2048

    # ── engram 布局 ──
    engram_max_ngram: int = 4
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_vocab: int = 16_000_000
    engram_num_embeddings: tuple = (384006168, 384016682)
    pad_token_id: int = 2
    compressed_vocab: int = 99092

    # ── 并行 ──
    moe_ep_size: int = 8
    world_size: int = 8
    cp_size: int = 1

    # 便捷派生量（只读）
    @property
    def n_index_layers(self):
        return len(self.index_source_layers)

    @property
    def o_head_dim_per_group(self):
        return self.n_attention_heads * self.head_dim // self.o_groups

    def derived(self) -> dict:
        """运行期会反复用到的派生量（集中算一次，避免各处重复拼公式）。"""
        return dict(
            q_lora=self.q_lora_rank,
            o_lora=self.o_lora_rank,
            o_groups=self.o_groups,
            o_head_dim_per_group=self.n_attention_heads * self.head_dim // self.o_groups,
            index_packed_d=self.index_head_dim // 2,
            index_q_dim=self.index_n_heads * self.index_head_dim,
            attn_q_dim=self.n_attention_heads * self.head_dim,
            i_local=self.moe_intermediate,
            win_pool_slots=self.win_pool_slots,
            kv_row_width=self.kv_row_width,
            cmp_row_width=self.cmp_row_width,
        )


def _as_int_list(v):
    return [int(x) for x in (v or [])]


def _as_int_tuple(v, n=None):
    if v is None:
        return None
    t = tuple(int(x) for x in v)
    return t


def build(cfg) -> ModelSpec:
    """cfg → ModelSpec。所有 key 都可能缺失（= 用默认值）。"""
    m = (cfg or {}).get("model") or {}
    run = (cfg or {}).get("run") or {}
    pc = (cfg or {}).get("parallel_config") or {}
    s = ModelSpec()

    # 结构
    for k in ("hidden", "moe_intermediate", "n_layers", "n_routed_experts",
              "n_shared_experts", "topk", "sliding_window", "candidate_source_layer"):
        if m.get(k) is not None:
            setattr(s, k, int(m[k]))
    for k in ("compress_ratios", "kv_source_layers", "index_source_layers", "engram_layers"):
        if m.get(k) is not None:
            setattr(s, k, _as_int_list(m[k]))

    # 维度
    for k in ("head_dim", "n_attention_heads", "qk_rope_head_dim", "q_lora_rank",
              "o_lora_rank", "o_groups", "index_n_heads", "index_head_dim",
              "index_topk", "vocab"):
        if m.get(k) is not None:
            setattr(s, k, int(m[k]))
    for k in ("rope_theta", "compress_rope_theta", "rms_norm_eps", "hc_eps",
              "routed_scaling_factor", "swiglu_limit"):
        if m.get(k) is not None:
            setattr(s, k, float(m[k]))
    if m.get("rope_scaling") is not None:
        s.rope_scaling = dict(m["rope_scaling"])

    # 数值（hc 两个整数）
    for k in ("hc_mult", "hc_sinkhorn_iters"):
        if m.get(k) is not None:
            setattr(s, k, int(m[k]))

    # 尺寸（可选新键）
    for k in ("block_size", "quant_block", "scale_pair", "nz_format",
              "kv_row_width", "cmp_row_width", "win_pool_slots",
              "candidate_block_size", "candidate_topk_blocks",
              "engram_prefetch_T"):
        if m.get(k) is not None:
            setattr(s, k, int(m[k]))
    # 也允许从 scheduler_config.block_size 取（与官方 config 对齐）
    sc = (cfg or {}).get("scheduler_config") or {}
    if m.get("block_size") is None and sc.get("block_size") is not None:
        s.block_size = int(sc["block_size"])

    # engram 布局（可选）
    for k in ("engram_max_ngram", "engram_n_heads", "engram_head_dim",
              "engram_vocab", "pad_token_id", "compressed_vocab"):
        if m.get(k) is not None:
            setattr(s, k, int(m[k]))
    t = _as_int_tuple(m.get("engram_num_embeddings"))
    if t is not None:
        s.engram_num_embeddings = t

    # 并行
    for k in ("moe_ep_size", "world_size", "cp_size"):
        if pc.get(k) is not None:
            setattr(s, k, int(pc[k]))

    # 派生默认：行宽/池槽位若没显式给，按模型维度算（值必须与旧硬编码一致）
    if m.get("kv_row_width") is None:
        s.kv_row_width = s.head_dim + s.head_dim // 16        # 512 + 32 = 544
    if m.get("cmp_row_width") is None:
        s.cmp_row_width = s.index_head_dim * 2 + s.index_head_dim // 2   # 256 + 64 = 320
    if m.get("win_pool_slots") is None:
        s.win_pool_slots = 3 * s.sliding_window               # 3 × 128 = 384
    return s


def validate(spec: ModelSpec, cfg=None) -> list[str]:
    """聚合式校验；返回错误字符串列表（空 = 通过）。

    只检查**结构一致性**，不碰数值/权重内容：
      · 列表长度/取值范围/子集关系；
      · 整除/对齐（算子契约）；
      · 当前实现只支持的部分（n_shared_experts==1、block_size==128）。
    """
    e = []
    n = spec.n_layers

    # ── 层表 ──
    if len(spec.compress_ratios) != n:
        e.append(f"compress_ratios 长度 {len(spec.compress_ratios)} ≠ n_layers {n}")
    bad = [r for r in spec.compress_ratios if r not in (0, 1, 2)]
    if bad:
        e.append(f"compress_ratios 只能取 0/1/2，发现 {sorted(set(bad))}")
    for name in ("kv_source_layers", "index_source_layers", "engram_layers"):
        lst = getattr(spec, name)
        # engram_layers 允许为空（no-engram 诊断口径）；其余两张表必须有内容
        if not lst and name != "engram_layers":
            e.append(f"{name} 不能为空")
        oob = [x for x in lst if not (0 <= x < n)]
        if oob:
            e.append(f"{name} 越界（n_layers={n}）：{oob}")
        if len(set(lst)) != len(lst):
            e.append(f"{name} 有重复层号：{lst}")
    if not spec.kv_source_layers:
        e.append("kv_source_layers 不能为空（decoder replay 的 cut 依赖它）")
    if not spec.index_source_layers:
        e.append("index_source_layers 不能为空")
    if not (0 <= spec.candidate_source_layer < n):
        e.append(f"candidate_source_layer={spec.candidate_source_layer} 越界（n_layers={n}）")
    elif spec.compress_ratios and spec.candidate_source_layer < len(spec.compress_ratios) \
            and spec.compress_ratios[spec.candidate_source_layer] == 0:
        e.append(f"candidate_source_layer={spec.candidate_source_layer} 是 win 层（ratio=0），"
                 f"不会产生候选块")
    for name in ("kv_source_layers", "index_source_layers"):
        lst = getattr(spec, name)
        if lst and n and max(lst) >= n:
            e.append(f"{name} 最大值 {max(lst)} ≥ n_layers {n}")
    if set(spec.kv_source_layers) - set(spec.index_source_layers):
        e.append("kv_source_layers 必须 ⊆ index_source_layers（压缩源层要跑索引器）")
    if spec.kv_source_layers and spec.candidate_source_layer not in spec.kv_source_layers:
        e.append("candidate_source_layer 必须在 kv_source_layers 里（要真跑 qli 产候选块）")

    # ── 并行/专家 ──
    if spec.topk > spec.n_routed_experts:
        e.append(f"topk={spec.topk} > n_routed_experts={spec.n_routed_experts}")
    if spec.moe_ep_size > 0 and spec.n_routed_experts % spec.moe_ep_size:
        e.append(f"n_routed_experts={spec.n_routed_experts} 不能被 moe_ep_size="
                 f"{spec.moe_ep_size} 整除")
    if spec.n_shared_experts != 1:
        e.append(f"n_shared_experts={spec.n_shared_experts}：当前实现只支持 1")
    if spec.cp_size != 1:
        # decoder replay 只支持 cp1；不阻断，交给 assemble 的原判据
        pass

    # ── 算子契约/对齐 ──
    if spec.block_size != 128:
        e.append(f"block_size={spec.block_size}：attention/压缩算子契约要求 128")
    if spec.sliding_window <= 0 or spec.sliding_window % spec.block_size:
        e.append(f"sliding_window={spec.sliding_window} 必须 >0 且 {spec.block_size} 对齐")
    if spec.block_size % max(1, spec.candidate_block_size):
        e.append(f"block_size={spec.block_size} 必须能被 candidate_block_size="
                 f"{spec.candidate_block_size} 整除")
    if spec.quant_block <= 0 or spec.head_dim % spec.quant_block:
        e.append(f"quant_block={spec.quant_block} 必须整除 head_dim={spec.head_dim}")
    for name in ("hidden", "moe_intermediate"):
        v = getattr(spec, name)
        if spec.quant_block > 0 and v % spec.quant_block:
            e.append(f"{name}={v} 必须能被 quant_block={spec.quant_block} 整除")
    if spec.quant_block > 0 and spec.q_lora_rank % spec.quant_block:
        e.append(f"q_lora_rank={spec.q_lora_rank} 必须能被 quant_block={spec.quant_block} 整除")
    for name in ("hidden", "moe_intermediate", "q_lora_rank", "head_dim",
                 "index_head_dim", "o_lora_rank", "o_groups"):
        v = getattr(spec, name)
        if not isinstance(v, int) or v <= 0:
            e.append(f"{name} 必须是正整数，实际 {v!r}")
    if spec.o_groups and spec.n_attention_heads * spec.head_dim % spec.o_groups:
        e.append(f"n_attention_heads*head_dim={spec.n_attention_heads * spec.head_dim} "
                 f"不能被 o_groups={spec.o_groups} 整除")
    if spec.index_head_dim % 2:
        e.append(f"index_head_dim={spec.index_head_dim} 必须为偶数（MXFP4 两值一字节）")

    # ── engram ──
    if spec.engram_layers and spec.engram_n_heads <= 0:
        e.append("engram_n_heads 必须 > 0")
    if spec.engram_num_embeddings and len(spec.engram_num_embeddings) != len(spec.engram_layers):
        e.append(f"engram_num_embeddings 有 {len(spec.engram_num_embeddings)} 项，"
                 f"engram_layers 有 {len(spec.engram_layers)} 层（必须一一对应）")
    return e


# ── 进程级单例 ──
_SPEC: ModelSpec | None = None


def set_spec(cfg, strict: bool = True) -> ModelSpec:
    global _SPEC
    s = build(cfg)
    errs = validate(s, cfg)
    if errs and strict:
        raise SystemExit("✗ 配置校验失败（model_spec.validate）：\n  - " + "\n  - ".join(errs))
    if errs and not strict:
        print("  [model_spec] ⚠️ 配置校验问题（不阻断）：", flush=True)
        for x in errs:
            print(f"    - {x}", flush=True)
    _SPEC = s
    return s


def get() -> ModelSpec:
    return _SPEC if _SPEC is not None else build({})


def reset():
    global _SPEC
    _SPEC = None


def describe(s: ModelSpec | None = None) -> str:
    s = s or get()
    return (f"spec: layers={s.n_layers} hidden={s.hidden} I={s.moe_intermediate} "
            f"experts={s.n_routed_experts}(ep{s.moe_ep_size}) topk={s.topk} "
            f"head={s.n_attention_heads}×{s.head_dim} q_lora={s.q_lora_rank} "
            f"index={s.index_n_heads}×{s.index_head_dim} win={s.sliding_window} "
            f"kv_src={s.kv_source_layers} idx_src={s.index_source_layers} "
            f"engram={s.engram_layers}")

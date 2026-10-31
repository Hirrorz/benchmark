#!/usr/bin/env python3
"""dg 路径的 candidate 分支（L20 等 candidate 源层）。

参考 DeepInfer：
  - `index_attn.py:215` `candidate_block_max(p_sum, k_len, block_size)` → 块内 max
  - `index_attn.py:234` `_get_candidate_inputs` → candidate_blocks / block_size / topk_lens
  - `index_attn.py:255` `_get_sparse_prefill_metadata` → get_sparse_mqa_logits_metadata
  - `index_attn.py:308` `fp8_fp4_sparse_mqa_logits`
  - `index_attn.py:365` `topk(..., output_idx_offset=None, restore_candidate_indices_inplace(...))`

实现策略：块内 max / 紧凑长度 / 索引还原三步都是**纯索引运算**（非热点），
用 torch 实现，避免依赖 tilelang 的 `candidate_block_max_tl` 等 kernel。
只有 logits 用 deep_gemm 的 sparse 变体。
"""
import torch


def candidate_block_max(p_sum, k_lens, block_size):
    """块内取 max。

    p_sum   : [T, N] bf16/fp32（logits，N = 压缩 K 行数）
    k_lens  : [T] int32（每行可见长度）
    block_size: 8 或 16

    返回 (block_scores [T, ceil(N/block)], valid_blocks [num_valid])
    """
    T, N = p_sum.shape
    nblk = (N + block_size - 1) // block_size
    pad = nblk * block_size - N
    x = p_sum.float()
    if pad:
        x = torch.cat([x, torch.full((T, pad), float("-inf"), dtype=x.dtype,
                                     device=x.device)], dim=1)
    # 把超出 k_lens 的位置也置 -inf
    idx = torch.arange(nblk * block_size, device=x.device).unsqueeze(0)
    mask = idx < k_lens.unsqueeze(1)
    x = torch.where(mask, x, torch.full_like(x, float("-inf")))
    scores = x.reshape(T, nblk, block_size).max(dim=-1).values
    # ★ 官方 `candidate_block_max_ref`（candidate_blocks.py:70-75）：
    #   把每行**最后一个有效块**强制置 +inf，保证边界块一定被选中。
    valid_blocks = ((k_lens + block_size - 1) // block_size).clamp(max=nblk)
    last_blocks = (valid_blocks - 1).clamp(min=0).unsqueeze(1).long()
    last_scores = scores.gather(1, last_blocks)
    last_scores = torch.where(valid_blocks.unsqueeze(1) > 0,
                              torch.full_like(last_scores, float("inf")),
                              last_scores)
    scores = scores.scatter(1, last_blocks, last_scores)
    return scores, valid_blocks.to(torch.int32)


# 注意：**不要**在 NPU 上用 `nonzero()`（`aclnnNonzeroV2` 会崩 507015）。
# 旧版这里有一句 `(scores > -inf).any(0).nonzero()`，已删除。


def select_candidate_blocks(block_scores, valid_blocks, block_size, topk_blocks,
                            block_base=None):
    """每行选分数最高的 topk_blocks 个候选块。

    ★ 对齐官方 `_topk_candidate_block`（index_attn.py:206-215）：
        topk(block_scores, topk_blocks, end=valid_blocks, sorted_index=True,
             output_idx=..., idx_oob_fill_value=-1)
      ⇒ 无效位填 **-1**（不是 0），`restore` 里靠 `block_ids >= 0` 判无效。

    `block_base`：[T] int32，`candidate_base = ks // block_size`（向下取整）。
      官方在 `index_attn.py:271` 把序列局部块号加上 base 变**绝对块号**，
      所以这里直接输出绝对块号，`restore` 就不用再加。

    ★★ 2026-10-31（507015 根因，两条都是硬约束）：

      1) 块号必须**按块号升序**。`sparse_mqa_logits_metadata.hpp:9` 明写
         "The context determines the used prefix; its block indices must be
          sorted, unique and in range."，内核的 `sorted_union_advance` 依赖
         升序输入；`torch.topk(sorted=True)` 给的是**按分数降序**，直接传会
         让 `physical_kv_token_starts` 算成垃圾地址 → gather 越界 → aicore trap。
         官方用 DeepSelect 的 `sorted_index=True`（= 按 index 升序）。

      2) padding 位**必须重复最后一个有效块，不能填 -1**。metadata 内核按
         `num_blocks = (ke - ks) / B` 截断到 capacity 来读块表
         （`sparse_mqa_logits_metadata.hpp:127-129`），可见窗口内的块数
         通常 ≥ capacity ⇒ **整个 capacity 都会被读到**，-1 会变成非法 GM 地址。
         官方测试就是这么构造的：`blocks + [blocks[-1]] * pad`
         （`tests/test_attention.py:661`）。
         重复块由 `sorted_union` 去重，语义安全。
    """
    T, nblk = block_scores.shape
    kk = min(topk_blocks, nblk)
    vals, idxs = torch.topk(block_scores, kk, dim=1, largest=True, sorted=True)
    valid_cnt = (vals > float("-inf")).sum(dim=1).to(torch.int32)
    # 无效位（分数 -inf）先标 -1，稍后统一替换成「最后一个有效块」。
    idxs = torch.where(vals > float("-inf"), idxs, torch.full_like(idxs, -1))
    if kk < topk_blocks:
        pad = torch.full((T, topk_blocks - kk), -1, dtype=idxs.dtype, device=idxs.device)
        idxs = torch.cat([idxs, pad], dim=1)
    idxs = idxs.to(torch.int32)
    if block_base is not None:
        bb = block_base.to(torch.int32).unsqueeze(1)
        idxs = torch.where(idxs >= 0, idxs + bb, idxs)
    # 按块号升序（-1 排到末尾），满足内核的 sorted 契约
    key = torch.where(idxs >= 0, idxs, torch.full_like(idxs, 2**30))
    idxs = torch.gather(idxs, 1, torch.argsort(key, dim=1, stable=True))
    # ★ padding 位填「最后一个有效块」而非 -1（见 docstring 第 2 条）。
    #   每行至少有一个有效块：candidate_block_max 把最后一个有效块置 +inf。
    _last = idxs.gather(1, (valid_cnt - 1).clamp(min=0).long().unsqueeze(1))
    idxs = torch.where(idxs >= 0, idxs, _last.expand_as(idxs))
    return idxs, valid_cnt


def compact_lens(candidate_blocks, k_lens, block_size, valid_cnt):
    """每行候选块内实际有效的 K 个数（紧凑长度）—— 向量化。"""
    T, cap = candidate_blocks.shape
    bs = block_size
    g = candidate_blocks.long() * bs                       # [T, cap]
    hi = g + bs
    kl = k_lens.long().unsqueeze(1)                        # [T, 1]
    cnt = (torch.clamp(hi, max=kl) - g).clamp(min=0)       # [T, cap]
    # 只统计前 valid_cnt 个块
    ar = torch.arange(cap, device=candidate_blocks.device).unsqueeze(0)
    cnt = torch.where(ar < valid_cnt.unsqueeze(1), cnt, torch.zeros_like(cnt))
    return cnt.sum(dim=1).to(torch.int32)


def restore_indices(compact_idx, candidate_blocks, block_size, ks_row):
    """把「候选块内的紧凑位置」还原成「全局压缩行号」—— 向量化。

    参考 `restore_candidate_indices_inplace_tl`：
        compact p → 块号 b = p // bs，块内偏移 o = p % bs
        global = (cand[t,b] + ks//bs) * bs + ks%bs + o
    """
    bs = block_size
    cap = candidate_blocks.shape[1]
    capacity = cap * bs
    valid = (compact_idx >= 0) & (compact_idx < capacity)
    p = torch.clamp(compact_idx.long(), 0, capacity - 1)
    slots = p // bs
    offs = p % bs
    # ★ 官方 `restore_candidate_indices_inplace_ref`（candidate_blocks.py:170-180）：
    #   candidate_blocks 里已经是**绝对块号**（调用方加过 candidate_base），
    #   还原 = block_id * bs + offset，**不再加 base/resid**。
    block_ids = torch.gather(candidate_blocks.long(), 1,
                             slots.reshape(compact_idx.shape[0], -1)
                             ).reshape(compact_idx.shape)
    out = (block_ids * bs + offs).to(torch.int32)
    out = torch.where(valid & (block_ids >= 0), out, torch.full_like(out, -1))
    # ★ 还原后必须**重新排序**：DeepSelect 的 sorted_index 是「紧凑位置有序」，
    #   映射回全局行号后不再有序（实测 row0 有 84 个逆序对）。
    #   vendor 的融合算子输出是升序 ⇒ 这里排序对齐（-1 保持在末尾）。
    key = torch.where(out >= 0, out, torch.full_like(out, 2**30))
    out = torch.gather(out, 1, torch.argsort(key, dim=1, stable=True))
    return out

def sparse_logits_via_dense(p_sum, candidate_blocks, block_size, ks_row):
    """方案 B：不用 `fp8_fp4_sparse_mqa_logits`（需 deep-jit），从**全量 dense logits**
    抽取候选块内的 logits，按「候选块内紧凑顺序」重排（向量化）。

    全局行 = (cand[t,b] + ks//bs) * bs + ks%bs + o
    """
    T, nk = p_sum.shape
    cap = candidate_blocks.shape[1]
    bs = block_size
    width = cap * bs
    # ★ candidate_blocks 已是**绝对块号**（select 时加过 block_base），直接乘 bs
    g = candidate_blocks.long().clamp(min=0) * bs                   # [T, cap]
    o = torch.arange(bs, device=p_sum.device).unsqueeze(0)         # [1, bs]
    rows = g.unsqueeze(-1) + o                                     # [T, cap, bs]
    valid = (rows < nk) & (candidate_blocks.unsqueeze(-1) >= 0)
    rows = torch.where(rows < nk, rows, torch.zeros_like(rows))
    out = torch.gather(p_sum, 1, rows.reshape(T, width))
    return torch.where(valid.reshape(T, width), out,
                       torch.full_like(out, float("-inf")))

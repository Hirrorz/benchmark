#!/usr/bin/env python3
"""索引器三件套的**独立探针**（不依赖 runner / exec_ops / exec_indexer）。

自己造张量，把三个 ds41 算子各跑一次并打印形状/dtype/统计：
  ① indexer_prologue_qw   —— idx_prolog：qr 量化 + wq_b + 部分 rope + MXFP4 量化 + weights_proj
  ② quant_lightning_indexer_metadata —— metadata（QLI 的前置，必须同 device/int32/[1024]）
  ③ quant_lightning_indexer          —— qli：稀疏索引（topk=512）

用法：
  python3.11 -u tools/probe_indexer.py [--T 1024] [--ratio 2] [--dev npu:1] [--topk 512]

口径来自官方调用点 models/modules/indexer.py:122-139 / 243-302 / 331-446
（真权重形状见 weights_manifest.py；rope theta 用 compress_rope_theta=160000）。
"""
import argparse
import sys

import torch
import torch_npu

import cann_ops_transformer            # 注册 torch.ops.cann_ops_transformer.ds41.*（**不要**再 import experimental）

D = torch.ops.cann_ops_transformer.ds41

N_HEADS = 32          # index_n_heads
HEAD_DIM = 128        # index_head_dim（逻辑维；存储 = 128/2 = 64 字节 MXFP4）
ROPE_DIM = 64         # qk_rope_head_dim（就是上面存下来的 Dr）
Q_LORA = 1280         # q_lora_rank
HIDDEN = 5120         # hidden
SOFTMAX_SCALE = HEAD_DIM ** -0.5 * N_HEADS ** -0.5     # = 0.015625
ROPE_THETA = 160000.0                                  # compress_rope_theta


def stat(name, t):
    print(f"  {name:<12} shape={tuple(t.shape)} dtype={t.dtype} dev={t.device}", flush=True)
    return t


def make_rope(T, dev):
    """[T,64] fp32（interleave 口径）：官方 RotaryEmbedding(theta=160000) 拍平。"""
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32, device=dev) / ROPE_DIM))
    pos = torch.arange(T, dtype=torch.float32, device=dev)
    freqs = torch.outer(pos, inv)                      # [T,32]
    emb = torch.cat((freqs, freqs), dim=-1)            # [T,64]（interleave）
    return emb.sin().contiguous(), emb.cos().contiguous()


def to_nz(dev_name, w, what):
    """ND → FRACTAL_NZ(29)（官方 to_nz；prologue_qw 对 wqb/ww 的硬要求）。"""
    out = torch_npu.npu_format_cast(w.contiguous(), 29)
    print(f"  {what}: {tuple(w.shape)} {w.dtype} → NZ fmt={torch_npu.get_npu_format(out)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--ratio", type=int, default=2, help="compress_ratio（2=c2a，1=c1a）")
    ap.add_argument("--dev", default="npu:1")
    ap.add_argument("--topk", type=int, default=512)
    a = ap.parse_args()

    T, ratio, dev, topk = a.T, a.ratio, a.dev, a.topk
    torch.npu.set_device(dev)
    print(f"== probe_indexer：T={T} ratio={ratio} topk={topk} dev={dev} "
          f"dawn={torch_npu.npu.get_device_name(dev)}", flush=True)

    # ── ① idx_prolog = indexer_prologue_qw ────────────────────────────────
    print("\n① indexer_prologue_qw（idx_prolog）", flush=True)
    x = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device=dev)
    # qr = npu_dynamic_mx_quant(q_norm(wq_a(x)))[0].view(uint8)：这里直接造同样口径的 qr
    qr_bf = torch.randn(T, Q_LORA, dtype=torch.bfloat16, device=dev)
    qr8, qr_scale = torch_npu.npu_dynamic_mx_quant(qr_bf, dst_type=torch.float8_e4m3fn)
    qr = qr8.view(torch.uint8).contiguous()
    n_pair = Q_LORA // 64                                   # 20
    descale_qr = qr_scale.view(torch.uint8).view(T, n_pair, 2).contiguous()
    # wq_b：真权重是 [4096,1280] F8_E4M3 + scale [4096,40] E8M0（此处合成同形状同 dtype）
    wqb = torch.randn(N_HEADS * HEAD_DIM, Q_LORA, dtype=torch.float32, device=dev).to(torch.float8_e4m3fn)
    descale_wqb = torch.full((N_HEADS * HEAD_DIM, n_pair * 2), 127, dtype=torch.uint8, device=dev) \
        .view(N_HEADS * HEAD_DIM, n_pair, 2).contiguous()
    ww = torch.randn(N_HEADS, HIDDEN, dtype=torch.bfloat16, device=dev)
    sin, cos = make_rope(T, dev)
    stat("x", x); stat("qr", qr); stat("wqb", wqb); stat("ww", ww)
    stat("descale_qr", descale_qr); stat("descale_wqb", descale_wqb); stat("rope_sin", sin)
    wqb_nz, ww_nz = to_nz(dev, wqb, "wqb"), to_nz(dev, ww, "ww")
    print("  调用 indexer_prologue_qw …", flush=True)
    q, dq, wt = D.indexer_prologue_qw(x, qr, wqb_nz, ww_nz, descale_qr, descale_wqb,
                                      sin, cos, softmax_scale=SOFTMAX_SCALE)
    torch.npu.synchronize()
    stat("q", q); stat("descale_q", dq); stat("w", wt)
    print(f"    q 统计:   absmax={q.view(torch.uint8).abs().max().item()} 非零={int((q != 0).sum())}"
          f"/{q.numel()}", flush=True)
    print(f"    w 统计:   absmax={wt.abs().max().item():.6f} mean={wt.mean().item():.6f} "
          f"（应 ≈ softmax_scale×weights_proj，量级 ~1e-2）", flush=True)

    # ── ② metadata ────────────────────────────────────────────────────────
    print("\n② quant_lightning_indexer_metadata（metadata 前置）", flush=True)
    seq_used_k = T // ratio                                 # 压缩后 K 长度
    cu_q = torch.tensor([0, T], dtype=torch.int32, device=dev)
    seq_k = torch.tensor([seq_used_k], dtype=torch.int32, device=dev)
    residual = None if ratio == 1 else torch.tensor([T % ratio], dtype=torch.int32, device=dev)
    stat("cu_seqlens_q", cu_q); stat("seqused_k", seq_k)
    md = D.quant_lightning_indexer_metadata(
        cu_seqlens_q=cu_q, seqused_q=None, seqused_k=seq_k, cmp_residual_k=residual,
        batch_size=1, max_seqlen_q=-1, max_seqlen_k=-1,
        num_heads_q=N_HEADS, num_heads_k=1, head_dim=HEAD_DIM, topk=topk,
        mask_mode=3, cmp_ratio=ratio, layout_q="TND", layout_k="PA_BBND",
        candidate_topk_blocks=-1, candidate_block_size=-1)
    torch.npu.synchronize()
    stat("metadata", md)
    print(f"    metadata 非零={int((md != 0).sum())}/{md.numel()} 前 16 项={md[:16].tolist()}", flush=True)

    # ── ③ qli = quant_lightning_indexer ───────────────────────────────────
    print("\n③ quant_lightning_indexer（qli）", flush=True)
    rows_per_block = 128 * ratio                            # c2a: 256 / c1a: 128
    nblk = max(1, -(-seq_used_k // rows_per_block))
    # K：第一版用 zeros（只为跑通形状）；真实路径应由 indexer_prologue_k 写
    k = torch.zeros(nblk, rows_per_block, 1, 64, dtype=torch.uint8, device=dev)
    descale_k = torch.full((nblk, rows_per_block, 1, 2, 2), 127, dtype=torch.uint8, device=dev)
    print(f"  K 池：nblk={nblk} rows_per_block={rows_per_block} "
          f"stride={k.stride()}（要求内轴 (64,64,1)）", flush=True)
    stat("k", k); stat("descale_k", descale_k)
    out = D.quant_lightning_indexer(
        q, k, wt.contiguous(), dq, descale_k, topk, 1,
        cu_seqlens_q=cu_q, seqused_q=None, seqused_k=seq_k, cmp_residual_k=residual,
        block_table=None, metadata=md, max_seqlen_q=-1, mask_mode=3, cmp_ratio=ratio,
        layout_q="TND", layout_k="PA_BBND", return_value=False,
        candidate_topk_blocks=-1, candidate_block_size=-1)
    torch.npu.synchronize()
    sparse, sparse_val, cand, cand_len = out
    stat("sparse_indices", sparse); stat("sparse_values", sparse_val)
    stat("candidate_indices", cand); stat("candidate_lengths", cand_len)
    sidx = sparse.reshape(-1)
    neg = int((sidx < 0).sum())
    valid = sidx[sidx >= 0]
    print(f"    sparse_indices：-1 共 {neg} 个；有效 {valid.numel()} 个；"
          f"有效值范围 [{int(valid.min()) if valid.numel() else '-'}, "
          f"{int(valid.max()) if valid.numel() else '-'}]；唯一值 {int(valid.unique().numel())} 个", flush=True)
    print(f"    第 0 行前 16 个：{sparse.reshape(T, -1)[0][:16].tolist()}", flush=True)
    print(f"    末行前 16 个：  {sparse.reshape(T, -1)[-1][:16].tolist()}", flush=True)

    print("\n✓ 三件套全部跑通：idx_prolog / metadata / qli", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

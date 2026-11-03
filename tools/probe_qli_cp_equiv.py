#!/usr/bin/env python3
"""QLI 的 CP 分块等价性探针（独立、秒级，不跑模型）。

问题：命中 + CP 时索引器该不该按 CP 切？分块版（`_qli_cp`：每块当"某前缀的尾部"，
`seq_k=(gs+nv)//ratio`、`residual=(gs+nv)%ratio`、块表按块重建）与全长版
（全长版：`seq_k=L//ratio`、`residual=L%ratio`、块表按全局）**参数不同**，
2026-10-28 实测输出不等价。本探针把四个可变参数拆开做对照，定位到底是哪一个导致差异：

    A  packed   : seq_k=L//ratio,   residual=L%ratio,   table=ring(N_full)
    B  chunk    : seq_k=end//ratio, residual=end%ratio, table=ring(n_chunk)
    C  只换表   : seq_k/residual 同 A, table=ring(n_chunk)
    D  只换 seq : seq_k/residual 同 B, table=ring(N_full)

同一批 q 行在四种参数下跑 QLI，逐行比较 top-k 的**集合**与**顺序**。

用法：
    python3.11 tools/probe_qli_cp_equiv.py [--L 32768] [--T 512] [--ratio 2]
                                           [--topk 64] [--dev npu:0]
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch_npu

import cann_ops_transformer                                  # noqa: F401  注册 ds41 ops

D = torch.ops.cann_ops_transformer.ds41
N_HEADS = 32
HEAD_DIM = 128
ROWS_PER_BLOCK = 128          # c1a；c2a 时 ×ratio
SOFTMAX_SCALE = HEAD_DIM ** -0.5 * N_HEADS ** -0.5


def ring(nblk: int, dev) -> torch.Tensor:
    """与 exec_indexer._qslitab 同构：[n-1, 1, 2, …, n-2]（长度 n-1）。"""
    return torch.cat([torch.tensor([nblk - 1], dtype=torch.int32, device=dev),
                      torch.arange(1, nblk - 1, dtype=torch.int32, device=dev)])


def run_case(q, k, dk, w, topk, ratio, seq_k, residual, bt, dev, dq=None):
    cu = torch.tensor([0, int(q.shape[0])], dtype=torch.int32, device=dev)
    sk = torch.tensor([int(seq_k)], dtype=torch.int32, device=dev)
    rs = None if ratio == 1 else torch.tensor([int(residual) % ratio],
                                              dtype=torch.int32, device=dev)
    md = D.quant_lightning_indexer_metadata(
        cu_seqlens_q=cu, seqused_q=None, seqused_k=sk, cmp_residual_k=rs,
        batch_size=1, max_seqlen_q=-1, max_seqlen_k=-1,
        num_heads_q=N_HEADS, num_heads_k=1, head_dim=HEAD_DIM, topk=topk,
        mask_mode=3, cmp_ratio=ratio, layout_q="TND", layout_k="PA_BBND",
        candidate_topk_blocks=-1, candidate_block_size=-1)
    out = D.quant_lightning_indexer(
        q, k, w, dq, dk, topk, 1,
        cu_seqlens_q=cu, seqused_q=None, seqused_k=sk, cmp_residual_k=rs,
        block_table=bt, metadata=md, max_seqlen_q=-1, mask_mode=3, cmp_ratio=ratio,
        layout_q="TND", layout_k="PA_BBND", return_value=False,
        candidate_topk_blocks=-1, candidate_block_size=-1)
    torch.npu.synchronize()
    return out[0].reshape(int(q.shape[0]), topk)


def diff(a, b, topk):
    """逐行比较 top-k：集合差 / 顺序差。"""
    eq = (a == b).all().item()
    rows_set = rows_ord = 0
    first = None
    for r in range(a.shape[0]):
        sa = [x for x in a[r].tolist() if x >= 0]
        sb = [x for x in b[r].tolist() if x >= 0]
        if set(sa) != set(sb):
            rows_set += 1
            first = first if first is not None else ("集合", r, sorted(set(sa) ^ set(sb))[:6])
        elif sa != sb:
            rows_ord += 1
            first = first if first is not None else ("顺序", r, None)
    return eq, rows_set, rows_ord, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=32768, help="整条序列长")
    ap.add_argument("--T", type=int, default=512, help="本段 q 行数（=命中段的 non-cached 行）")
    ap.add_argument("--ratio", type=int, default=2)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--dev", default="npu:0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = a.dev
    torch.npu.set_device(dev)
    L, T, ratio, topk = a.L, a.T, a.ratio, a.topk
    rpb = ROWS_PER_BLOCK * ratio                     # 每块的压缩行数
    N = max(1, -(-(L // ratio) // rpb))              # 全局块数
    off = L - T                                      # 命中段绝对起点
    print(f"== L={L} T={T} ratio={ratio} topk={topk} off={off} "
          f"rows_per_block={rpb} N_full={N}", flush=True)

    q = torch.randn(T, N_HEADS, HEAD_DIM // 2, dtype=torch.bfloat16,
                    device=dev).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    dq = torch.full((T, HEAD_DIM // 2 // 32, 2), 127, dtype=torch.uint8,
                    device=dev).contiguous()
    w = torch.rand(N_HEADS, dtype=torch.float32, device=dev).to(torch.bfloat16) / N_HEADS
    k = torch.randint(0, 255, (N, rpb, 1, HEAD_DIM // 2), dtype=torch.uint8,
                      device=dev).contiguous()
    dk = torch.full((N, rpb, 1, 2, 2), 127, dtype=torch.uint8, device=dev).contiguous()

    full_sk, full_rs = L // ratio, L % ratio
    end = off + T
    ch_sk, ch_rs = end // ratio, end % ratio
    n_ch = max(1, -(-(ch_sk) // rpb))
    print(f"   packed: seq_k={full_sk} residual={full_rs} 表长={N - 1}")
    print(f"   chunk : seq_k={ch_sk} residual={ch_rs} 表长={n_ch - 1}", flush=True)

    bt_full = ring(N, dev).unsqueeze(0)              # [1, N-1]
    bt_ch = ring(n_ch, dev).unsqueeze(0)             # [1, n_ch-1]
    # 块表必须覆盖到 seq_k 用到的块：分块时把 ring 补到 n_ch 长度（同 _qli_cp 的 [:nblk] 切法）
    if bt_ch.shape[1] < n_ch:
        bt_ch = torch.cat([bt_ch, torch.arange(bt_ch.shape[1], n_ch,
                                               dtype=torch.int32, device=dev).unsqueeze(0)], 1)

    A = run_case(q, k, dk, w, topk, ratio, full_sk, full_rs, bt_full, dev, dq)
    B = run_case(q, k, dk, w, topk, ratio, ch_sk, ch_rs, bt_ch, dev, dq)
    C = run_case(q, k, dk, w, topk, ratio, full_sk, full_rs, bt_ch, dev, dq)
    Dd = run_case(q, k, dk, w, topk, ratio, ch_sk, ch_rs, bt_full, dev, dq)

    print("\n== 与 A（packed）对比 ==", flush=True)
    for name, X, why in (("B 分块(全部换)", B, "seq_k+residual+表"),
                         ("C 只换表", C, "表(ring 长度不同)"),
                         ("D 只换 seq_k/residual", Dd, "seq_k + residual")):
        eq, rs, ro, first = diff(A, X, topk)
        print(f"  {name:<22}({why}): 逐位相同={eq}  集合不同行={rs}  仅顺序不同行={ro}"
              f"  {'' if first is None else f'首个: {first}'}")

    print("\n（参考）A 的行 0 前 8 个: ", A[0][:8].tolist())
    print("（参考）A 的末行 前 8 个: ", A[-1][:8].tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())

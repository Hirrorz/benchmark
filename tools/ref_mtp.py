#!/usr/bin/env python3.11
"""DSpark MTP 草稿的**稠密 PyTorch 参考**（同输入对拍 mini 的首个 propose）。

输入：mini `DS41M_DSPARK_DUMP` 落盘的 .pt（aux 3×[S,H]、start_pos、seed、draft、logits0）。
做法：从 checkpoint 直接读 mtp.0/1/2 权重，fp8/fp4 反量化成 bf16/fp32，按 checkpoint
参考 `inference/model.py` 的公式（hc_split_sinkhorn / Block.forward / DSparkAttention /
Gate sqrtsoftplus / Expert SiLU / ParallelHead / Markov）逐层跑 3 个 MTP 块；
draft 块位置 = last_context_pos+1（引擎口径，= start_pos+S）。

用法：
  python3.11 tools/ref_mtp.py /mnt/xds/sfs/l00972766/tmp/dspark_aux_dump.pt
"""
from __future__ import annotations

import json
import math
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = "/mnt/xds/sfs/DeepSeek-V4.1-Flash-cann-mxfp8-mxfp4"
VOCAB, DIM, H = 129280, 5120, 5120
N_HEADS, HEAD_DIM, ROPE_DIM = 64, 512, 64
Q_LORA, O_LORA, O_GROUPS = 1280, 1024, 8
INTER = 2304
HC, HC_ITERS, HC_EPS = 4, 20, 1e-6
EPS = 1e-20
SWIGLU_LIMIT = 10.0
ROUTE_SCALE = 1.5
GATE_TEMP = 1.0
TOPK = 3
WIN = 128
BLOCK = 5
N_STAGES = 3
NOISE = 128799

_idx = json.load(open(f"{ROOT}/model.safetensors.index.json"))["weight_map"]
_handles: dict = {}


def raw(name: str) -> torch.Tensor:
    fn = _idx[name]
    h = _handles.get(fn)
    if h is None:
        h = _handles[fn] = safe_open(f"{ROOT}/{fn}", framework="pt", device="cpu")
    return h.get_tensor(name)


def _e8m0(t: torch.Tensor) -> torch.Tensor:
    u = t.view(torch.uint8) if t.dtype != torch.uint8 else t
    return ((u.to(torch.int32)) << 23).view(torch.float32)


def deq_fp8(name: str, block: int = 32) -> torch.Tensor:
    w = raw(name).to(torch.float32)
    s = _e8m0(raw(name.replace(".weight", ".scale")))
    return w * s.repeat_interleave(block, dim=-1)


_FP4 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def deq_fp4(name: str, block: int = 32) -> torch.Tensor:
    b = raw(name).to(torch.int32)                      # [out, in/2]
    lo = (b & 0xF)
    hi = ((b >> 4) & 0xF)
    out = torch.empty(b.shape[0], b.shape[1] * 2, dtype=torch.float32)
    import os as _os
    if _os.environ.get("REF_FP4_HIGH_FIRST", "0") == "1":
        out[:, 0::2] = _FP4[hi]                        # 高半字节 = 前一个元素
        out[:, 1::2] = _FP4[lo]
    else:
        out[:, 0::2] = _FP4[lo]                        # 低半字节 = 前一个元素
        out[:, 1::2] = _FP4[hi]
    s = _e8m0(raw(name.replace(".weight", ".scale")))
    return out * s.repeat_interleave(block, dim=-1)


def rms(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (w.float() * y).to(x.dtype)


# ── hc（照 checkpoint kernel.py:407-470 的公式）─────────────────────────────
def hc_split(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor):
    t = mixes.shape[0]
    pre = torch.sigmoid(mixes[:, :HC] * scale[0] + base[:HC]) + HC_EPS
    post = 2.0 * torch.sigmoid(mixes[:, HC:2 * HC] * scale[1] + base[HC:2 * HC])
    comb = (mixes[:, 2 * HC:].reshape(t, HC, HC) * scale[2]
            + base[2 * HC:].reshape(HC, HC))
    comb = comb.softmax(dim=-1) + HC_EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_EPS)
    for _ in range(HC_ITERS - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_EPS)
    return pre, post, comb


def hc_mixes(x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor):
    xf = x.reshape(x.shape[0], -1).float()      # [T, hc*d]（我们的 x 是 3-D）
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    mixes = F.linear(xf, fn.float()) * rsqrt
    return hc_split(mixes, scale, base)


def hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    # 我们的布局是 [T, hc, d]（参考实现是 [b,s,hc,d]），hc 维是 dim=1
    return torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=1).to(x.dtype)


def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
    y = (post.unsqueeze(-1) * x.unsqueeze(-2)
         + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=1))
    return y.type_as(x)


# ── rope（交错、partial 64 维；YaRN 在 <65536 位置为恒等）─────────────────────
def _freqs(n: int, dim: int = ROPE_DIM, base: float = 10000.0) -> torch.Tensor:
    f = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    return torch.outer(torch.arange(n, dtype=torch.float32), f)


_FREQ = _freqs(262144)


def rope(x: torch.Tensor, pos: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """x [...,64]（partial rope 的尾部），pos [L] 绝对位置；inverse=逆旋转（共轭）。"""
    # x 的第 0 维是 token（[L,64] 或 [L,H,64]）
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))     # [L,...,32]
    _ang = -_FREQ[pos] if inverse else _FREQ[pos]
    fc = torch.polar(torch.ones(pos.numel(), ROPE_DIM // 2), _ang)    # [L,32]
    fc = fc.view(pos.numel(), *([1] * (xc.dim() - 2)), -1)
    out = torch.view_as_real(xc * fc).flatten(-2)
    return out.to(x.dtype)


def attention(stage: int, x: torch.Tensor, main_x: torch.Tensor, draft_pos: torch.Tensor,
              main_pos: torch.Tensor, ret_internals: bool = False):
    # 官方 draft 只保留最后 win 个上下文（窗环）；S>WIN 时必须同样裁剪，
    # 否则参考实现比 mini 多看旧行（长上下文对拍会假性分叉）。
    if main_x.shape[0] > WIN:
        main_x = main_x[-WIN:]
        main_pos = main_pos[-WIN:]
    p = f"mtp.{stage}.attn."
    wq_a = deq_fp8(p + "wq_a.weight")
    wq_b = deq_fp8(p + "wq_b.weight")
    wkv = deq_fp8(p + "wkv.weight")
    wo_a = deq_fp8(p + "wo_a.weight")
    wo_b = deq_fp8(p + "wo_b.weight")
    q_norm = raw(p + "q_norm.weight")
    kv_norm = raw(p + "kv_norm.weight")
    sink = raw(p + "attn_sink").float()

    qr = rms(F.linear(x.float(), wq_a), q_norm)
    q = F.linear(qr, wq_b).reshape(x.shape[0], N_HEADS, HEAD_DIM).float()
    q[..., -ROPE_DIM:] = rope(q[..., -ROPE_DIM:], draft_pos)

    kv = rms(F.linear(x.float(), wkv), kv_norm)
    kv[..., -ROPE_DIM:] = rope(kv[..., -ROPE_DIM:], draft_pos)
    mkv = rms(F.linear(main_x.float(), wkv), kv_norm)
    mkv[..., -ROPE_DIM:] = rope(mkv[..., -ROPE_DIM:], main_pos)

    keys = torch.cat([mkv, kv], dim=0).float()          # [S+B, 512]
    scores = torch.einsum("qhd,kd->qhk", q, keys) * (HEAD_DIM ** -0.5)
    # ★ attn_sink 是**只进 softmax 分母的虚拟 key**（官方 kernel.py:381-384：
    #   `sum_exp += exp(attn_sink - scores_max)`，不参与 value 加权）。之前把它加到
    #   每个 key 的 logit 上 → softmax 平移不变 → 等于没用 sink；各 head 的 sink 不同，
    #   输出逐 head 缩放，经 wo_a 混合后 attn_x cos 掉到 0.3 的主要嫌疑。
    m = scores.max(dim=-1, keepdim=True).values
    p = torch.exp(scores - m)
    denom = p.sum(dim=-1, keepdim=True) + torch.exp(sink.view(1, N_HEADS, 1) - m)
    attn = p / denom
    o = torch.einsum("qhk,kd->qhd", attn, keys)         # [B,64,512]
    o_raw = o.clone()
    o[..., -ROPE_DIM:] = rope(o[..., -ROPE_DIM:], draft_pos, inverse=True)
    o_inv = o.clone()

    o = o.reshape(x.shape[0], O_GROUPS, -1)             # [B,8,4096]
    wa = wo_a.view(O_GROUPS, O_LORA, -1).float()
    g = torch.einsum("bgd,grd->bgr", o, wa)             # [B,8,1024]
    out = F.linear(g.flatten(1), wo_b)
    out = out.to(x.dtype)
    if ret_internals:
        return out, dict(q=q, kv=kv, mkv=mkv, keys=keys, o_raw=o_raw, o_inv=o_inv)
    return out


def moe(stage: int, x: torch.Tensor, caches: dict) -> torch.Tensor:
    p = f"mtp.{stage}.ffn."
    gw = raw(p + "gate.weight").float()
    gb = raw(p + "gate.bias").float()
    scores = (x.float() @ gw.t()) / GATE_TEMP
    scores = F.softplus(scores).sqrt()
    idx = (scores + gb).topk(TOPK, dim=-1).indices    # [B,3]
    w = scores.gather(1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20)
    w = w * ROUTE_SCALE
    caches[f"gate{stage}"] = (idx.cpu(), w.cpu())

    shared = caches.get("shared")
    if shared is None:
        shared = caches["shared"] = (
            deq_fp8(p + "shared_experts.w1.weight"),
            deq_fp8(p + "shared_experts.w3.weight"),
            deq_fp8(p + "shared_experts.w2.weight"),
        )
    y = torch.zeros_like(x.float())
    used = sorted(set(int(i) for i in idx.flatten().tolist()))
    for e in used:
        key = ("expert", stage, e)
        ew = caches.get(key)
        if ew is None:
            ew = caches[key] = (
                deq_fp4(p + f"experts.{e}.w1.weight"),
                deq_fp4(p + f"experts.{e}.w3.weight"),
                deq_fp4(p + f"experts.{e}.w2.weight"),
            )
        rows = (idx == e).nonzero(as_tuple=False)
        for b, k in rows.tolist():
            h = x[b:b + 1].float()
            gate = F.linear(h, ew[0])
            up = F.linear(h, ew[1])
            up = torch.clamp(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
            gate = torch.clamp(gate, max=SWIGLU_LIMIT)
            z = F.linear(F.silu(gate) * up, ew[2])
            y[b] += w[b, k] * z[0]
    # shared expert（每个 token 都过）
    h = x.float()
    gate = F.linear(h, shared[0])
    up = F.linear(h, shared[1])
    up = torch.clamp(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    gate = torch.clamp(gate, max=SWIGLU_LIMIT)
    y = y + F.linear(F.silu(gate) * up, shared[2])
    return y.to(x.dtype)


def cmp_tensor(tag: str, ref: torch.Tensor, mini: torch.Tensor):
    if mini is None:
        return
    r = ref.float().reshape(-1)
    m = mini.float().reshape(-1)
    if r.numel() != m.numel():
        print(f"[cmp:{tag}] shape 不同 ref={tuple(ref.shape)} mini={tuple(mini.shape)}")
        return
    d = (r - m).abs()
    cos = torch.nn.functional.cosine_similarity(r, m, dim=0).item()
    print(f"[cmp:{tag}] max|Δ|={d.max():.5f} mean|Δ|={d.mean():.5f} cos={cos:.6f}")


def run(dump_path: str):
    d = torch.load(dump_path, map_location="cpu", weights_only=False)
    aux = [a.to(torch.bfloat16) for a in d["aux"]]
    main_hidden = torch.cat(aux, dim=-1)               # [S, 3H]
    seed = int(d["seed"])
    start_pos = int(d["start_pos"])
    S = main_hidden.shape[0]
    last_context_pos = start_pos + S - 1
    draft_pos = torch.arange(last_context_pos + 1, last_context_pos + 1 + BLOCK)
    main_pos = torch.arange(start_pos, start_pos + S)
    print(f"[ref] S={S} start={start_pos} seed={seed} draft_pos={draft_pos.tolist()}")

    # main_x（stage0 的 main_proj/main_norm；3 层共用）
    w0 = f"mtp.0."
    main_x = rms(F.linear(main_hidden.float(), deq_fp8(w0 + "main_proj.weight")),
                 raw(w0 + "main_norm.weight")).to(torch.bfloat16)
    print(f"[ref] main_x {tuple(main_x.shape)} absmax={main_x.abs().max():.4f}")
    cmp_tensor("main_x", main_x, d.get("main_x"))

    # 块输入：[seed, noise×4]
    ids = [seed] + [NOISE] * (BLOCK - 1)
    emb_w = raw("embed.weight").to(torch.bfloat16)
    x = emb_w[torch.tensor(ids)].float().unsqueeze(1).repeat(1, HC, 1).to(torch.bfloat16)
    pre_mix = torch.zeros(BLOCK, HC, dtype=torch.float32)
    pre_mix[:, 0] = 1.0

    caches: dict = {}
    for s in range(N_STAGES):
        p = f"mtp.{s}."
        # —— attention 段 ——
        residual = x
        ap, apo, ac = hc_mixes(x, raw(p + "hc_attn_fn"), raw(p + "hc_attn_scale"),
                               raw(p + "hc_attn_base"))
        h = hc_pre(x, pre_mix)
        h = rms(h, raw(p + "attn_norm.weight"))
        cmp_tensor(f"attn_in{s}", h, (d.get("attn_in") or [None] * N_STAGES)[s])
        out, it = attention(s, h, main_x, draft_pos, main_pos, ret_internals=True)
        cmp_tensor(f"q{s}", it["q"], (d.get("q") or [None] * N_STAGES)[s])
        cmp_tensor(f"draft_kv{s}", it["kv"], (d.get("draft_kv") or [None] * N_STAGES)[s])
        cmp_tensor(f"o_raw{s}", it["o_raw"], (d.get("o_raw") or [None] * N_STAGES)[s])
        cmp_tensor(f"o_inv{s}", it["o_inv"], (d.get("o_inv") or [None] * N_STAGES)[s])
        x = hc_post(out, residual, apo, ac)
        cmp_tensor(f"attn_out{s}", out, (d.get("attn_out") or [None] * N_STAGES)[s])
        cmp_tensor(f"attn_x{s}", x, (d.get("attn_x") or [None] * N_STAGES)[s])
        # —— ffn 段 ——
        residual = x
        fp, fpo, fc = hc_mixes(x, raw(p + "hc_ffn_fn"), raw(p + "hc_ffn_scale"),
                               raw(p + "hc_ffn_base"))
        h = hc_pre(x, ap)
        h = rms(h, raw(p + "ffn_norm.weight"))
        h = moe(s, h, caches)
        x = hc_post(h, residual, fpo, fc)
        pre_mix = fp
        print(f"[ref] stage{s} x absmax={x.float().abs().max():.4f}")
        cmp_tensor(f"stage{s}", x, (d.get("stage_x") or [None] * N_STAGES)[s])

    # —— 对比 gate 的 top-3 选择 ——
    for s in range(N_STAGES):
        r = caches.get(f"gate{s}")
        m = (d.get("gate") or [None] * N_STAGES)[s]
        if r is None or m is None:
            continue
        mi = m[0].reshape(-1, TOPK)
        mw = m[1].reshape(-1, TOPK).float()
        print(f"[cmp:gate{s}] ref idx={r[0][0].tolist()} w={[round(float(v),4) for v in r[1][0]]}")
        print(f"[cmp:gate{s}] mini idx={mi[0].tolist()} w={[round(float(v),4) for v in mw[0]]}")

    # —— head ——
    h = hc_pre(x, pre_mix)
    h = rms(h, raw("mtp.2.norm.weight"))
    head = raw("head.weight").float()
    cmp_tensor("hh", h, d.get("hh"))
    logits = torch.matmul(h.float(), head.t())         # [B,V] fp32
    m_emb = raw("mtp.2.markov_head.embed.weight").to(torch.bfloat16)
    m_head = raw("mtp.2.markov_head.head.weight").float()
    prev = seed
    out = []
    lg0 = None
    for i in range(BLOCK):
        bias = torch.matmul(m_head, m_emb[prev].float())
        lg = logits[i] + bias
        if i == 0:
            lg0 = lg
        nxt = int(lg.argmax())
        out.append(nxt)
        prev = nxt
    print(f"[ref] draft={out}")
    ref_draft = [int(v) for v in d.get("draft") or []]
    print(f"[mini] draft={ref_draft}")
    print(f"[cmp] draft_equal={out == ref_draft}")

    mini_lg0 = d.get("logits0")
    if mini_lg0 is not None:
        mini_lg0 = mini_lg0.float()
        diff = (lg0 - mini_lg0).abs()
        cos = torch.nn.functional.cosine_similarity(lg0, mini_lg0, dim=0).item()
        r1 = torch.topk(lg0, 5).indices.tolist()
        r2 = torch.topk(mini_lg0, 5).indices.tolist()
        print(f"[cmp] row0 logits: max|Δ|={diff.max():.4f} mean|Δ|={diff.mean():.4f} "
              f"cos={cos:.6f}")
        print(f"[cmp] ref top5={r1}")
        print(f"[cmp] mini top5={r2}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    run(sys.argv[1])

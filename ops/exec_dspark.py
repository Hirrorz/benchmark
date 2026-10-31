"""exec_dspark.py —— DSpark 草稿模型（checkpoint `mtp.0/1/2`）。

实现照着 checkpoint 自带的最简参考：
`/mnt/xds/sfs/DeepSeek-V4.1-Flash-cann-mxfp8-mxfp4/inference/model.py:1020-1156`

一次 propose：
  1. main_x = main_norm(main_proj(cat(aux_hidden[37,38,39], -1)))   # 目标模型 3 层 aux
  2. 每层 MTP：main_kv = kv_norm(wkv(main_x)) → rope → 写 **128 行窗环**（槽 pos%128）
  3. 草稿块 ids = [seed, noise×(B-1)]；位置 = last_pos+1 .. +B
  4. 3 层 MTP 串行：hc_pre → attn_norm → q/draft_kv → **非因果 sparse FA**
     （窗环 128 + 草稿块 B，K=128+B）→ inverse rope → wo_a/wo_b →
     hc_post → ffn_norm → 128 专家 top-3 MoE → hc_post
  5. 所有 B 行 hc_pre_mix → mtp.2.norm → lm_head → [B,V]
  6. 逐位 argmax：logits[i] += markov_head(prev_token)

草稿层的权重键名与主模型一层完全相同（strip `mtp.N.`），因此直接复用
exec_ops 的 win 层算子和 exec_moe 的 TP MoE。
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager

import torch
import torch_npu

import exec_moe
import exec_ops
import weights_manifest as WM


def snapshot_cmp(shared):
    """strict-all verify 前的回滚点：压缩器 ring state（每层 fp32 ~16KB）。"""
    if not isinstance(shared, dict):
        return {}
    return {k: v.clone() for k, v in shared.items()
            if isinstance(k, str) and k.startswith("_cmp_state_")}


def restore_cmp(shared, snap):
    for k, v in (snap or {}).items():
        cur = shared.get(k)
        if cur is not None:
            cur.copy_(v)


@contextmanager
def _no_dg():
    """草稿阶段临时挂起主模型 DG 缓冲（T=1 图的 dg_* 不适用于草稿位置）。"""
    prev = exec_ops._DG.get("call", True)
    exec_ops.dg_call_enter(False)
    try:
        yield
    finally:
        exec_ops.dg_call_enter(prev)


class DSparkDraft:
    def __init__(self, cfg, source, dev, l_pool=1024, n_block=3,
                 rank=0, world=1, verbose=False):
        self.cfg = cfg
        self.source = source
        self.dev = dev
        self.l_pool = int(l_pool)
        self.B = int(n_block)
        self.rank = int(rank)
        self.world = int(world)
        self.verbose = bool(verbose)
        m = cfg["model"]
        self.H = int(m["hidden"])
        self.I = int(m.get("moe_intermediate", 2304))
        self.M = int(m.get("hc_mult", 4))
        self.eps = float(m.get("rms_norm_eps", 1e-20))
        self.win = int(m.get("sliding_window", 128))
        self.head_dim = int(m.get("head_dim", 512))
        self.n_heads = int(m.get("n_attention_heads", 64))
        self.rope_dim = int(m.get("qk_rope_head_dim", 64))
        self.noise = int(m.get("dspark_noise_token_id", 128799))
        self.n_stages = int(m.get("num_nextn_predict_layers", 3))
        self.n_experts = int(m.get("dspark_n_routed_experts", 128))
        self.topk = int(m.get("dspark_num_experts_per_tok",
                            m.get("dspark_n_activated_experts", 3)))
        # ★ 草稿 MoE 布局：TP（decode 低时延，全专家 × I/tp）或 EP（每 rank 连续 16 专家）。
        #   2026-10-21 用户口径：ep8 也要能跑 DSpark（不用 P→D 切权重）。
        pc = (cfg.get("parallel_config") or {})
        self._tp = int(pc.get("moe_tp_size", 1) or 1) > 1
        self._ep = max(1, int(pc.get("moe_ep_size", 1) or 1))
        self._ep_rank = int(pc.get("_ep_rank", self.rank) or 0)
        self._e_local = (self.n_experts if self._tp
                         else max(1, self.n_experts // self._ep))
        self._e_lo = 0 if self._tp else self._ep_rank * self._e_local
        self._e_hi = self._e_lo + self._e_local
        self.BS = int(exec_ops.BS)
        self.KV_ROW = int(exec_ops.KV_ROW_W)
        self._w = {}
        self._ctx = {}
        self._pool = {}
        self._rope = {}
        self._loaded = False
        self._d = None
        self._dbg = None
        # 草稿块图（3 层整段 ACL graph）：固定几何见 _capture；失败永久退回 eager。
        self._g = None
        self._g_failed = False
        self._g_in = None
        self._g_out = None
        self._g_const = {}
        self._ctx_seeded = False

    # ── 权重 / ctx ─────────────────────────────────────────────────────────
    def _nblk(self):
        return 3          # 块 0 空、块 1 窗环、块 2 草稿块

    def _load_stage(self, stage):
        import runner as _runner
        man = WM.mtp_layer_manifest(stage, self.n_stages,
                                    n_experts=self.n_experts,
                                    e_range=(self._e_lo, self._e_hi))
        w = {}
        for suf, dt, shp in man:
            w[suf] = self.source.get(f"mtp.{stage}.{suf}", dt, shp)
        w = _runner.pack_layer_weights(w, self.cfg, layer_idx=40 + int(stage))
        self._w[stage] = w

    def _ctx_for(self, stage):
        import run as _run
        fake = {"idx": 40 + int(stage), "T": self.B, "ratio": 0,
                "is_kv_src": 0, "is_idx_src": 0, "is_candidate": 0, "engram": 0}
        c = dict(_run.ctx_for(self.cfg, fake))
        c.update(topk=self.topk, rtopk=self.topk, e_local=self._e_local,
                 S=self.win, win_abs=True, L_full=self.l_pool,
                 hc_mult=self.M,
                 hc_sinkhorn_iters=int(self.cfg["model"].get("hc_sinkhorn_iters", 20)),
                 hc_eps=float(self.cfg["model"].get("hc_eps", 1e-6)),
                 rope_dim=self.rope_dim,
                 routed_scaling_factor=self.cfg["model"].get("routed_scaling_factor", 1.5),
                 swiglu_limit=self.cfg["model"].get("swiglu_limit", 10.0),
                 eps=self.eps, is_decode_step=True,
                 n_seqs=1, cu_seqlens=None, seq_lens=None,
                 # ★ 非 None：否则 topk_gate 会走"合成权重强均路由"
                 #   topk_idx=(t*K+k)%E，草稿 MoE 完全不看训练出的 gate → accept≈0。
                 input_ids=[0], pos_offset=0, engram=0, T=self.B)
        self._ctx[stage] = c

    def ensure_loaded(self):
        if self._loaded:
            return
        if self.world > 1 and not self._tp and self._ep <= 1:
            raise RuntimeError(
                "DSpark 草稿 MoE：world>1 需要 TP（moe_tp_size>1）或 EP（moe_ep_size>1）布局")
        with _no_dg():
            for s in range(self.n_stages):
                self._load_stage(s)
                self._ctx_for(s)
                self._pool[s] = torch.zeros(
                    self._nblk() * self.BS, self.KV_ROW,
                    dtype=torch.float8_e4m3fn, device=self.dev)
                n = self.l_pool + self.win + self.B + 8
                self._rope[s] = exec_ops._rope_cos_sin(n, self._ctx[s], self.dev, 0)
            if self._d is None:
                self._d = exec_ops._d41()
        self._loaded = True
        if self.verbose and self.rank == 0:
            print(f"  [dspark] 草稿模型就绪：{self.n_stages} 层 × "
                  f"{self.n_experts} 专家 top-{self.topk}，块长 B={self.B}，"
                  f"noise={self.noise}", flush=True)

    # ── rope / 投影 ────────────────────────────────────────────────────────
    def _rope_apply(self, x, pos, stage, inverse=False):
        cos, sin = self._rope[stage]
        T = int(x.shape[0])
        p = max(0, min(int(pos), cos.shape[0] - T))
        c, s = cos[p:p + T], sin[p:p + T]
        if inverse:
            s = -s
        v = x.view(T, -1, 1, x.shape[-1])
        torch.ops.cann_ops_transformer.inplace_partial_rotary_mul(
            v, c, s, rotary_mode="interleave",
            partial_slice=[x.shape[-1] - self.rope_dim, x.shape[-1]])
        return x

    def _project_kv(self, stage, x, pos):
        ctx, w = self._ctx[stage], self._w[stage]
        ctx["T"] = int(x.shape[0])
        ctx["pos_offset"] = int(pos)
        st = {"x": x}
        exec_ops.OPS["wkv"](st, ctx, w)
        exec_ops.OPS["kv_norm"](st, ctx, w)
        kv = st["kv"]
        self._rope_apply(kv, pos, stage)
        return kv

    def _main_x(self, aux_list):
        w0 = self._w[0]
        a = torch.cat([t.to(self.dev) for t in aux_list], dim=-1)
        h = exec_ops._qmm(a, "main_proj.weight", w0, self._ctx[0])
        return torch_npu.npu_rms_norm(h, w0["main_norm.weight"], self.eps)[0]

    def _write_rows(self, stage, kv, slots):
        pool = self._pool[stage]
        torch.ops.custom.kv_compress_epilog_v2(
            pool.view(-1, self.KV_ROW), kv,
            slots.to(torch.int32),
            quant_group_size=32, quant_mode="mxfp8_bf16", round_scale=True)

    def _write_main(self, stage, kv, positions):
        pos = positions.to(torch.int64)
        slots = (pos % self.win).to(torch.int32) + self.BS
        self._write_rows(stage, kv, slots)

    def _write_draft(self, stage, kv, slots=None):
        if slots is None:
            j = torch.arange(int(kv.shape[0]), device=self.dev, dtype=torch.int32)
            slots = j + 2 * self.BS
        self._write_rows(stage, kv, slots)

    # ── 草稿注意力 ─────────────────────────────────────────────────────────
    def _qkv(self, stage, x, pos):
        ctx, w = self._ctx[stage], self._w[stage]
        T = int(x.shape[0])
        ctx["T"] = T
        ctx["pos_offset"] = int(pos)
        st = {"x": x}
        exec_ops.OPS["wq_a"](st, ctx, w)
        exec_ops.OPS["q_norm"](st, ctx, w)
        exec_ops.OPS["wq_b"](st, ctx, w)
        q = st["q"].reshape(T, self.n_heads, self.head_dim).contiguous()
        self._rope_apply(q, pos, stage)
        kv = self._project_kv(stage, x, pos)
        if self._dbg is not None and stage < self.n_stages:
            self._dbg.setdefault("q", []).append(q.detach().float().cpu())
            self._dbg.setdefault("draft_kv", []).append(kv.detach().float().cpu())
        return q, kv

    def _attn(self, stage, q, last_pos):
        T = int(q.shape[0])
        K = self.win + T
        wcount = min(self.win, int(last_pos) + 1)
        idx = torch.full((T, 1, K), -1, dtype=torch.int32, device=self.dev)
        row = torch.cat([
            torch.arange(wcount, dtype=torch.int32, device=self.dev),
            torch.arange(T, dtype=torch.int32, device=self.dev) + self.win,
        ])
        idx[:, 0, :row.numel()] = row
        length = torch.full((T, 1), int(row.numel()), dtype=torch.int32, device=self.dev)
        cu = torch.tensor([0, T], dtype=torch.int32, device=self.dev)
        d = self._d
        md = d.mixed_quant_sparse_flash_mla_metadata(
            length, torch.zeros_like(length),
            cu_seqlens_q=cu, num_heads_q=self.n_heads, num_heads_kv=1,
            head_dim=self.head_dim, quant_mode=1,
            layout_q="TND", layout_kv="PA_BBND",
            has_ori_kv=True, has_cmp_kv=False)
        pool = self._pool[stage].view(self._nblk(), self.BS, 1, self.KV_ROW).view(torch.uint8)
        bt = torch.tensor([[1, 2]], dtype=torch.int32, device=self.dev)
        out, _ = d.mixed_quant_sparse_flash_mla(
            q, quant_mode=1, ori_kv=pool, ori_sparse_indices=idx,
            ori_block_table=bt, cu_seqlens_q=cu, ori_topk_length=length,
            # 官方 prepare_fa_kwargs 传 None；长度由 ori_topk_length 约束。
            seqused_ori_kv=None,
            sinks=self._w[stage]["attn.attn_sink"], metadata=md,
            softmax_scale=self.head_dim ** -0.5,
            layout_q="TND", layout_kv="PA_BBND", return_softmax_lse=False)
        return out.reshape(T, self.n_heads, self.head_dim).contiguous()

    # ── 草稿块图（3 层整段 ACL graph）──────────────────────────────────────
    #   固定条件：last_pos+1 >= win（窗口稀疏索引/长度恒定）、B 固定、池地址固定。
    #   每轮只换 3 个输入缓冲：块输入 x_in（seed+noise 的 embedding）与 rope 的
    #   cos/sin/-sin（draft 位置）。窗环内容由图外的 prefill_context/update_context
    #   维护 —— propose 原先的上下文重写是**冗余**的（同样的行/位置再写一遍），
    #   图路径不重写；32 token 对 T=1 基线逐位一致已验证。
    def _rope_apply_buf(self, x, cos, sin):
        T = int(x.shape[0])
        v = x.view(T, -1, 1, x.shape[-1])
        torch.ops.cann_ops_transformer.inplace_partial_rotary_mul(
            v, cos, sin, rotary_mode="interleave",
            partial_slice=[x.shape[-1] - self.rope_dim, x.shape[-1]])
        return x

    def _qkv_g(self, stage, x, cos, sin):
        ctx, w = self._ctx[stage], self._w[stage]
        T = int(x.shape[0])
        ctx["T"] = T
        ctx["pos_offset"] = 0                 # 位置只经 rope 缓冲进入，算子不吃 pos
        st = {"x": x}
        exec_ops.OPS["wq_a"](st, ctx, w)
        exec_ops.OPS["q_norm"](st, ctx, w)
        exec_ops.OPS["wq_b"](st, ctx, w)
        q = st["q"].reshape(T, self.n_heads, self.head_dim).contiguous()
        self._rope_apply_buf(q, cos, sin)
        st2 = {"x": x}
        exec_ops.OPS["wkv"](st2, ctx, w)
        exec_ops.OPS["kv_norm"](st2, ctx, w)
        kv = st2["kv"]
        self._rope_apply_buf(kv, cos, sin)
        return q, kv

    def _g_consts(self, stage):
        """窗口满 128 时的常量注意力几何（每 stage 一套，捕获前建好）。"""
        got = self._g_const.get(stage)
        if got is not None:
            return got
        B = self.B
        K = self.win + B
        dev = self.dev
        idx = torch.full((B, 1, K), -1, dtype=torch.int32, device=dev)
        row = torch.cat([torch.arange(self.win, dtype=torch.int32, device=dev),
                         torch.arange(B, dtype=torch.int32, device=dev) + self.win])
        idx[:, 0, :row.numel()] = row
        length = torch.full((B, 1), int(row.numel()), dtype=torch.int32, device=dev)
        cu = torch.tensor([0, B], dtype=torch.int32, device=dev)
        md = self._d.mixed_quant_sparse_flash_mla_metadata(
            length, torch.zeros_like(length), cu_seqlens_q=cu,
            num_heads_q=self.n_heads, num_heads_kv=1, head_dim=self.head_dim,
            quant_mode=1, layout_q="TND", layout_kv="PA_BBND",
            has_ori_kv=True, has_cmp_kv=False)
        got = dict(idx=idx, length=length, cu=cu, md=md,
                   bt=torch.tensor([[1, 2]], dtype=torch.int32, device=dev),
                   slots=(torch.arange(B, dtype=torch.int32, device=dev) + 2 * self.BS))
        self._g_const[stage] = got
        return got

    def _attn_g(self, stage, q, consts):
        T = int(q.shape[0])
        pool = self._pool[stage].view(self._nblk(), self.BS, 1, self.KV_ROW).view(torch.uint8)
        out, _ = self._d.mixed_quant_sparse_flash_mla(
            q, quant_mode=1, ori_kv=pool,
            ori_sparse_indices=consts["idx"],
            ori_block_table=consts["bt"], cu_seqlens_q=consts["cu"],
            ori_topk_length=consts["length"], seqused_ori_kv=None,
            sinks=self._w[stage]["attn.attn_sink"], metadata=consts["md"],
            softmax_scale=self.head_dim ** -0.5,
            layout_q="TND", layout_kv="PA_BBND", return_softmax_lse=False)
        return out.reshape(T, self.n_heads, self.head_dim).contiguous()

    def _g_layer(self, stage, h, shared, cos, sin, sin_i, consts):
        ctx, w = self._ctx[stage], self._w[stage]
        T = int(h.shape[0])
        ctx["T"] = T
        st = {"x": h, "residual": h.clone(), "_w": w, "_shared": shared,
              "_backends": (self.cfg.get("backends") or {}),
              "_allow_placeholder": False, "_placeholder_used": set(),
              "_norm_stage": "attn"}
        exec_ops.OPS["hc_pre"](st, ctx, w)
        exec_ops.OPS["rmsnorm"](st, ctx, w)
        q, kv = self._qkv_g(stage, st["x"], cos, sin)
        self._write_draft(stage, kv, consts["slots"])
        o = self._attn_g(stage, q, consts)
        self._rope_apply_buf(o, cos, sin_i)
        st["attn_out"] = o
        prev_ni = os.environ.get("DS41M_NO_INVROPE")
        os.environ["DS41M_NO_INVROPE"] = "1"      # 逆 rope 已用缓冲做完
        try:
            exec_ops.OPS["wo_a"](st, ctx, w)
        finally:
            if prev_ni is None:
                os.environ.pop("DS41M_NO_INVROPE", None)
            else:
                os.environ["DS41M_NO_INVROPE"] = prev_ni
        exec_ops.OPS["wo_b"](st, ctx, w)
        exec_ops.OPS["hc_post"](st, ctx, w)
        st["_norm_stage"] = "ffn"
        exec_ops.OPS["hc_pre"](st, ctx, w)
        exec_ops.OPS["rmsnorm"](st, ctx, w)
        exec_ops.OPS["gate"](st, ctx, w)
        exec_ops.OPS["topk_gate"](st, ctx, w)
        exec_ops.OPS["mega_moe"](st, ctx, w)
        exec_ops.OPS["hc_post"](st, ctx, w)
        return st["x"]

    def _capture(self):
        B, M, H = self.B, self.M, self.H
        dev = self.dev
        cos0, sin0 = self._rope[0]
        for s in range(1, self.n_stages):
            if not torch.equal(self._rope[s][0], cos0):
                raise RuntimeError("各 stage rope 表不一致（图缓冲只能一套）")
        cshape = tuple(cos0.shape[1:])
        self._g_in = dict(
            x=torch.zeros(B, M, H, dtype=torch.bfloat16, device=dev),
            cos=torch.zeros((B,) + cshape, dtype=cos0.dtype, device=dev),
            sin=torch.zeros((B,) + cshape, dtype=cos0.dtype, device=dev),
            sin_neg=torch.zeros((B,) + cshape, dtype=cos0.dtype, device=dev),
        )
        self._g_out = dict(
            x=torch.zeros(B, M, H, dtype=torch.bfloat16, device=dev),
            pre=torch.zeros(B, M, dtype=torch.float32, device=dev),
        )
        consts = [self._g_consts(s) for s in range(self.n_stages)]
        gin = self._g_in
        shared = {}
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g, pool=torch.npu.graph_pool_handle()):
            x = gin["x"]
            for s in range(self.n_stages):
                x = self._g_layer(s, x, shared, gin["cos"], gin["sin"],
                                  gin["sin_neg"], consts[s])
            self._g_out["x"].copy_(x)
            pre = shared.get("_ffn_pre")
            if pre is not None:
                self._g_out["pre"].copy_(pre)
        self._g = g
        if self.rank == 0:
            print(f"  [dspark] 草稿块入图：{self.n_stages} 层 × B={self.B}"
                  f"（窗 {self.win}+{self.B}，输入 x/rope 缓冲）", flush=True)

    def _graph_propose(self, seed_id, seed_pos, mark=None):
        B = self.B
        cos0, sin0 = self._rope[0]
        p = int(seed_pos)
        if p < 0 or p + B > int(cos0.shape[0]):
            raise RuntimeError(f"draft pos {p} 超出 rope 表 {int(cos0.shape[0])}")
        gin, gout = self._g_in, self._g_out
        gin["cos"].copy_(cos0[p:p + B])
        gin["sin"].copy_(sin0[p:p + B])
        torch.neg(sin0[p:p + B], out=gin["sin_neg"])
        ids = [int(seed_id)] + [self.noise] * (B - 1)
        import runner as _runner
        emb = _runner.embed_ids(self.cfg, ids, self.source, self.dev)
        gin["x"].copy_(emb.unsqueeze(1).repeat(1, self.M, 1))
        # 诊断（DS41M_DSPARK_DRAFT_SYNC=1）：replay 前跨 rank 对齐，把"等最慢 rank"的
        # 时间移出 replay 计时（判断图里的集合通信是否在吃 rank 漂移）。
        if os.environ.get("DS41M_DSPARK_DRAFT_SYNC", "").strip().lower() in (
                "1", "on", "true", "yes"):
            try:
                import torch.distributed as _d
                if _d.is_initialized():
                    _d.barrier()
            except Exception:                      # noqa: BLE001
                pass
            torch.npu.synchronize()
        if mark is not None:
            mark("sync")
        self._g.replay()
        if mark is not None:
            mark("replay_only")
        return gout["x"], gout["pre"], ids

    # ── 单阶段 ─────────────────────────────────────────────────────────────
    def _layer(self, stage, h, draft_pos, last_pos, shared):
        ctx, w = self._ctx[stage], self._w[stage]
        T = int(h.shape[0])
        ctx["T"] = T
        st = {"x": h, "residual": h.clone(), "_w": w, "_shared": shared,
              "_backends": (self.cfg.get("backends") or {}),
              "_allow_placeholder": False, "_placeholder_used": set(),
              "_norm_stage": "attn"}
        if self._dbg is not None:
            self._dbg.setdefault("x_in", []).append(h.detach().float().cpu())
            _pm0 = shared.get("_ffn_pre")
            if _pm0 is not None:
                self._dbg.setdefault("pre_in", []).append(_pm0.detach().float().cpu())
        exec_ops.OPS["hc_pre"](st, ctx, w)
        exec_ops.OPS["rmsnorm"](st, ctx, w)
        if self._dbg is not None:
            self._dbg.setdefault("attn_in", []).append(st["x"].detach().float().cpu())
        q, kv = self._qkv(stage, st["x"], draft_pos)
        self._write_draft(stage, kv)
        o = self._attn(stage, q, last_pos)
        if self._dbg is not None:
            self._dbg.setdefault("o_raw", []).append(o.detach().float().cpu())
        self._rope_apply(o, draft_pos, stage, inverse=True)
        if self._dbg is not None:
            self._dbg.setdefault("o_inv", []).append(o.detach().float().cpu())
        st["attn_out"] = o
        prev_ni = os.environ.get("DS41M_NO_INVROPE")
        os.environ["DS41M_NO_INVROPE"] = "1"      # 逆 rope 已在上一步用缓存表做完
        try:
            exec_ops.OPS["wo_a"](st, ctx, w)
        finally:
            if prev_ni is None:
                os.environ.pop("DS41M_NO_INVROPE", None)
            else:
                os.environ["DS41M_NO_INVROPE"] = prev_ni
        exec_ops.OPS["wo_b"](st, ctx, w)
        if self._dbg is not None:
            self._dbg.setdefault("attn_out", []).append(st["x"].detach().float().cpu())
        exec_ops.OPS["hc_post"](st, ctx, w)
        if self._dbg is not None:
            self._dbg.setdefault("attn_x", []).append(st["x"].detach().float().cpu())
        st["_norm_stage"] = "ffn"
        exec_ops.OPS["hc_pre"](st, ctx, w)
        exec_ops.OPS["rmsnorm"](st, ctx, w)
        exec_ops.OPS["gate"](st, ctx, w)
        exec_ops.OPS["topk_gate"](st, ctx, w)
        if self._dbg is not None:
            self._dbg.setdefault("gate", []).append(
                (st["topk_idx"].detach().cpu(), st["topk_weight"].detach().cpu()))
        exec_ops.OPS["mega_moe"](st, ctx, w)
        exec_ops.OPS["hc_post"](st, ctx, w)
        return st["x"]

    # ── 对外：prefill 播种 / propose / 接受后更新 ─────────────────────────
    @torch.no_grad()
    @contextmanager
    def _moe_layout(self):
        """草稿块（T=B）的 MoE 路径：TP=强制 _moe_tp；EP=强制融合 MegaMoE（不跟随 decode 特化）。"""
        keys = ("DS41M_MOE_TP", "DS41M_TP_DECODE_ONLY", "DS41M_MOEMODE")
        prev = {k: os.environ.get(k) for k in keys}
        try:
            if self._tp:
                os.environ["DS41M_MOE_TP"] = "1"
            else:
                # EP：保持 decode 口径（`_moe_decode` dispatch/combine）。**不要**强制
                # 融合 MegaMoE：草稿 topk=3，厂商融合算子在该 K 下实测 aicore
                # “scalar access internal buffer OOB”（2026-10-22，T=5/E=128）；
                # 官方 decode MoE 对 K=3 是原生支持的路径。
                os.environ.pop("DS41M_MOE_TP", None)
                os.environ.pop("DS41M_TP_DECODE_ONLY", None)
            yield
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def reset(self):
        self.ensure_loaded()
        with _no_dg():
            for s in self._pool:
                self._pool[s].zero_()

    @torch.no_grad()
    def prefill_context(self, aux_list, start_pos=0):
        """把 prompt 的 aux 投影进窗环（只保留最后 win 行有效）。

        ★★ 2026-10-20（长上下文接受率低的根因）：窗环只有 `win=128` 槽，槽位 = pos % win。
        以前把**整段** S 行一次 scatter 写进去：S>128 时同一槽位被写 ~S/128 次，
        最终哪一行胜出取决于算子的 scatter 顺序（不保证是最后一行）→ 草稿上下文是
        **随机混合的旧行**，长上下文反而更差（实测 3.5k prompt accept 0.83/5）。
        官方（model.py:1038-1046）prefill 只写最后 win 行（cutoff 旋转）。
        修法：只投影/写最后 win 行，槽位按它们的绝对位置取模。
        """
        self.ensure_loaded()
        if aux_list is None or aux_list[0] is None:
            return
        with _no_dg():
            main_x = self._main_x(aux_list)
            T = int(main_x.shape[0])
            if T > self.win:
                main_x = main_x[-self.win:].contiguous()
                p0 = int(start_pos) + T - self.win
            else:
                p0 = int(start_pos)
            T = int(main_x.shape[0])
            pos = torch.arange(p0, p0 + T, device=self.dev)
            for s in range(self.n_stages):
                kv = self._project_kv(s, main_x, p0)
                self._write_main(s, kv, pos)
            self._ctx_seeded = True

    @torch.no_grad()
    def update_context(self, aux_list, start_pos):
        """验证后只把**接受**的 aux 行写进窗环。"""
        if aux_list is None or aux_list[0] is None:
            return
        with _no_dg():
            main_x = self._main_x(aux_list)
            T = int(main_x.shape[0])
            pos = torch.arange(int(start_pos), int(start_pos) + T, device=self.dev)
            for s in range(self.n_stages):
                kv = self._project_kv(s, main_x, int(start_pos))
                self._write_main(s, kv, pos)
            self._ctx_seeded = True

    def _head_markov(self, x, pre, ids):
        """最后 hc collapse → mtp.2.norm → lm_head → Markov 链式 argmax（图外）。

        Markov 链的下一步依赖上一步 argmax ⇒ 保持 eager（每轮 5 个小 matmul；
        实测 head 0.44ms + markov ~1.0ms）。
        """
        if pre is not None and pre.dim() == 2 and pre.shape[1] == x.shape[1]:
            hh = (pre.float().unsqueeze(-1) * x.float()).sum(1).to(x.dtype)
        else:
            hh = x.mean(dim=1).to(x.dtype)
        w_last = self._w[self.n_stages - 1]
        hh = torch_npu.npu_rms_norm(hh, w_last["norm.weight"], self.eps)[0]
        import model_spec as _ms
        hw = self.source.get("head.weight", "BF16", (_ms.get().vocab, self.H))
        logits = torch.matmul(hh, hw.t()).float()
        _lg0 = None
        if "markov_head.embed.weight" in w_last:
            e_w = w_last["markov_head.embed.weight"]
            h_w = w_last["markov_head.head.weight"]
            prev = torch.tensor([ids[0]], dtype=torch.long, device=self.dev)
            out = []
            for i in range(self.B):
                embm = e_w.index_select(0, prev)
                bias = torch.matmul(h_w.float(), embm.reshape(-1).float())
                lg = logits[i] + bias
                nxt = lg.argmax()
                if i == 0:
                    _lg0 = lg.detach()
                if (i == 0 and self.rank == 0 and os.environ.get(
                        "DS41M_DSPARK_DRAFT_LOGITS", "").strip().lower()
                        in ("1", "on", "true", "yes")):
                    _v3, _i3 = torch.topk(lg.float(), 3)
                    _pairs = [(int(a), round(float(b), 5))
                              for a, b in zip(_i3.cpu().tolist(), _v3.cpu().tolist())]
                    print(f"  [draft-logits] row0 top3={_pairs} seed={ids[0]}",
                          flush=True)
                out.append(nxt)
                prev = nxt.view(1)
            draft = [int(v) for v in torch.stack(out).cpu().tolist()]
        else:
            draft = [int(v) for v in logits.argmax(dim=-1).cpu().tolist()]
            _lg0 = logits[0].detach()
        return draft, logits, _lg0, hh

    @torch.no_grad()
    def propose(self, aux_list, start_pos, seed_id):
        """aux_list: 接受上下文的 3 个 [S,H]（层 37/38/39 输入）；
        start_pos: 这些 aux 的第一行位置；seed_id: 块首 token。
        返回 (draft_ids[B], logits[B,V] or None)。
        """
        self.ensure_loaded()
        # 分段计时（`DS41M_DSPARK_PHASE=1`，仅诊断；每段前同步，否则量到的是入队时间）
        _phase = os.environ.get("DS41M_DSPARK_PHASE", "").strip().lower() in (
            "1", "on", "true", "yes")
        _ph = {}
        _t_p = time.perf_counter() if _phase else 0.0

        def _mark(name):
            nonlocal _t_p
            if not _phase:
                return
            torch.npu.synchronize()
            _now = time.perf_counter()
            _ph[name] = _ph.get(name, 0.0) + (_now - _t_p) * 1000
            _t_p = _now

        with _no_dg(), self._moe_layout():
            # ── 草稿块图（固定几何）──：上下文已由 prefill_context/update_context 写好，
            #   图里只跑 3 层 + rope 缓冲；head/Markov 保持在图外（链式 argmax + host 依赖）。
            _S0 = int(aux_list[0].shape[0]) if (aux_list and aux_list[0] is not None) else 0
            # ★ 默认**关**（2026-10-20 实测：图回放 3 层 18.6ms vs eager 9.4ms ——
            #   数值逐位一致（accept 分布/token 完全相同），但 3 个 128 专家 MoE 的集合
            #   通信压在单个 replay 里，固定开销比 eager 的流水下发还贵；rank 对齐等待
            #   只有 0.7ms，不是漂移问题）。显式 DS41M_DSPARK_DRAFT_GRAPH=1 可开（实验）。
            _want_g = (
                os.environ.get("DS41M_DSPARK_DRAFT_GRAPH", "0").strip().lower()
                in ("1", "on", "true", "yes")
                and not os.environ.get("DS41M_DSPARK_DUMP")
                and self._ctx_seeded
                and not self._g_failed
                and _S0 > 0
            )
            if _want_g:
                _seed_pos = int(start_pos) + _S0   # 上下文末位 +1 = seed 的位置
                if _seed_pos >= self.win:          # 窗口满 128 → 几何固定
                    if self._g is None:
                        try:
                            self._capture()
                            _mark("capture")
                        except Exception as _ce:   # noqa: BLE001
                            self._g_failed = True
                            print(f"  ⚠️ [dspark] 草稿图捕获失败 → eager："
                                  f"{type(_ce).__name__}: {_ce}", flush=True)
                    if self._g is not None:
                        try:
                            x, pre, ids = self._graph_propose(seed_id, _seed_pos, mark=_mark)
                            draft, logits, _lg0, _hh = self._head_markov(x, pre, ids)
                            _mark("headmk")
                            if _phase and self.rank == 0:
                                print("  [dspark-phase] " + " | ".join(
                                    f"{k}={v:.2f}ms" for k, v in _ph.items())
                                    + f" | total={sum(_ph.values()):.2f}ms", flush=True)
                            return draft, logits
                        except Exception as _re:   # noqa: BLE001
                            self._g_failed = True
                            print(f"  ⚠️ [dspark] 草稿图 replay 失败 → eager："
                                  f"{type(_re).__name__}: {_re}", flush=True)
            main_x = self._main_x(aux_list)
            S = int(main_x.shape[0])
            last_pos = int(start_pos) + S - 1
            draft_pos = last_pos + 1
            # ★ 同 prefill_context：窗环只有 win 槽，整段写会让同槽重复覆盖（胜负由
            #   scatter 顺序决定）。只写最后 min(S,win) 行，位置用它们的绝对坐标。
            W = min(S, self.win)
            mx_tail = main_x[-W:].contiguous()
            p0 = int(start_pos) + S - W
            for s in range(self.n_stages):
                kv = self._project_kv(s, mx_tail, p0)
                pos = torch.arange(p0, p0 + W, device=self.dev)
                self._write_main(s, kv, pos)
            _mark("ctx")

            ids = [int(seed_id)] + [self.noise] * (self.B - 1)
            import runner as _runner
            emb = _runner.embed_ids(self.cfg, ids, self.source, self.dev)
            h = emb.unsqueeze(1).repeat(1, self.M, 1)
            _mark("emb")
            shared = {}
            x = h
            _dump_env = os.environ.get("DS41M_DSPARK_DUMP", "").strip()
            _stage_x = [] if _dump_env else None
            self._dbg = {} if _dump_env else None
            for s in range(self.n_stages):
                x = self._layer(s, x, draft_pos, last_pos, shared)
                _mark(f"s{s}")
                if _stage_x is not None:
                    _stage_x.append(x.detach().float().cpu())
            pre = shared.get("_ffn_pre")
            draft, logits, _lg0, hh = self._head_markov(x, pre, ids)
            _mark("markov")
            if _phase and self.rank == 0:
                print("  [dspark-phase] " + " | ".join(
                    f"{k}={v:.2f}ms" for k, v in _ph.items())
                    + f" | total={sum(_ph.values()):.2f}ms", flush=True)
            # 诊断/参考对拍：把首个 propose 的输入（aux/seed/start）与输出（draft/row0
            # Markov logits）落盘，供 tools/ref_mtp.py 用同一输入跑 checkpoint 参考实现。
            _dump = os.environ.get("DS41M_DSPARK_DUMP", "").strip()
            if _dump and self.rank == 0:
                try:
                    import torch as _t
                    _t.save({"aux": [a.detach().float().cpu() for a in aux_list],
                             "start_pos": int(start_pos), "seed": int(seed_id),
                             "draft": list(draft),
                             "logits0": (_lg0.float().cpu() if _lg0 is not None else None),
                             "main_x": main_x.detach().float().cpu(),
                             "stage_x": _stage_x,
                             "attn_x": (self._dbg or {}).get("attn_x"),
                             "gate": (self._dbg or {}).get("gate"),
                             # 子块（注意力定位用）：每 stage 一份
                             "x_in": (self._dbg or {}).get("x_in"),
                             "pre_in": (self._dbg or {}).get("pre_in"),
                             "attn_in": (self._dbg or {}).get("attn_in"),
                             "q": (self._dbg or {}).get("q"),
                             "draft_kv": (self._dbg or {}).get("draft_kv"),
                             "o_raw": (self._dbg or {}).get("o_raw"),
                             "o_inv": (self._dbg or {}).get("o_inv"),
                             "attn_out": (self._dbg or {}).get("attn_out"),
                             "pre": (pre.detach().float().cpu()
                                     if pre is not None and hasattr(pre, "detach") else None),
                             "hh": hh.detach().float().cpu()},
                            _dump)
                    print(f"  [dspark-dump] 已写 {_dump}（aux {[tuple(a.shape) for a in aux_list]}，"
                          f"start={int(start_pos)} seed={int(seed_id)}）", flush=True)
                except Exception as _e:                    # noqa: BLE001
                    print(f"  [dspark-dump] 失败 {type(_e).__name__}: {_e}", flush=True)
            return draft, logits

#!/usr/bin/env python3
"""逐算子对拍：mini 的 DS41M_DUMP_TENSORS 快照 ↔ 官方(light) 的 DS41L_PARITY_OPS dump。

两边语义（都已核对代码）：
  · 官方 `_save`：float→fp32 后落盘，dim0 切前 512 行（池类 off=0），存 {"t","shape","dtype"}
  · mini `_dump_tensors`：**算子执行之后**调用，落 st/hold 里的张量（float→fp32），dim0 切前 512 行

映射表（官方 tag → mini 快照）：见 MAP 注释。纯 CPU，不碰卡。

用法：
  python3.11 tools/cmp_ops_vs_official.py \
      --mini /mnt/xds/sfs/l00972766/real_collect_1017/dump01/ops \
      --light /mnt/xds/sfs/l00972766/parity/official_ops2/ops \
      --layers 0,1 [--json out.json]

  # L2（压缩/索引器）用另一次官方跑：
  python3.11 tools/cmp_ops_vs_official.py --mini <...>/ops \
      --light /mnt/xds/sfs/l00972766/parity/official_all15/ops --layers 2
"""
import argparse
import glob
import json
import os
import re
import sys

import torch


# ── 官方 tag ↔ mini 快照取法 ────────────────────────────────────────────────
# ("官方后缀", "mini 算子 key", "stage 过滤(attn/ffn/None)", "快照字段", "说明")
# mini 快照 = 该算子**执行后**的状态；stage 用 dump 里的 _stage 区分两次 hc_pre/rmsnorm/hc_post。
BOUNDARY = [
    ("attn_norm_in",        "hc_pre",   "attn", "x",         "hc_pre(attn) 输出"),
    ("attn_in",             "rmsnorm",  "attn", "x",         "attn rmsnorm 输出"),
    ("q",                   "mqsmla",   None,   "q",         "q（rope 后，sparse_attn 输入）"),
    ("kv_pool",             "mqsmla",   None,   "_full_pool", "窗口 KV 池原始字节"),
    ("win_sparse",          "mqsmla",   None,   "_win_idx",  "窗口稀疏索引"),
    ("win_len",             "mqsmla",   None,   "_win_len",  "窗口有效长度"),
    ("block_table",         "mqsmla",   None,   "_bt",       "窗口池块表"),
    ("cu_seqlens_q",        "mqsmla",   None,   "_cu",       "query cu_seqlens"),
    ("attn_out",            "mqsmla",   None,   "attn_out",  "稀疏注意力原始输出（逆 rope 前）"),
    ("attn_post",           "wo_b",     None,   "x",         "wo_b 之后（attn 段输出）"),
    ("ffn_norm_in",         "hc_pre",   "ffn",  "x",         "hc_pre(ffn) 输出"),
    ("ffn_in",              "rmsnorm",  "ffn",  "x",         "ffn rmsnorm 输出"),
    ("moe_in",              "gate",     None,   "x",         "MoE 输入"),
    ("topk_ids",            "topk_gate", None,  "topk_idx",  "路由 id"),
    ("topk_weight",         "topk_gate", None,  "topk_weight", "路由权重"),
    ("moe_out",             "mega_moe", None,   "moe_out",   "MegaMoE 输出"),
    ("layer_out",           "hc_post",  "ffn",  "x",         "层输出（尾随 mul/add/cast/tmove 为恒等）"),
    # engram（层 1、14）
    ("engram_emb",          None,       None,   "engram_kv",  "engram 取表嵌入"),
    ("engram_out",          None,       None,   "engram_out", "engram 门控输出"),
    # 压缩 / 索引器（ratio>0 的源层，如 2/8/14/20）
    ("comp_in",             "kv_compress", None, "x",        "压缩器输入"),
    ("comp_latent",         "kv_compress", None, "cmp_latent", "压缩器输出 latent"),
    ("comp_kv",             "kv_compress", None, "cmp_kv",   "写入 cmp 池的 KV"),
    ("idx_q_in",            None,       None,   "idx_q",     "索引器 q（uint8）"),
    ("idx_qscale_in",       None,       None,   "idx_descale_q", "索引器 q scale"),
    ("idx_w_in",            None,       None,   "idx_w",     "索引器权重"),
    ("cmp_sparse",          "mqsmla",   None,   "_cmp_sparse_in", "喂给注意力的压缩稀疏索引"),
    ("qsli_q_in",           None,       None,   "idx_q",     "QSLI q（同 idx_q，共存时取一处）"),
    # 官方冗余项（自证用）
    ("attn_out_pre_invrope", "mqsmla",  None,   "attn_out",  "逆 rope 前（应与 attn_out 相同）"),
]

# 官方张量里 dim0 是“块号”的（行偏移不适用；比对时都压成 [dim0,-1]）
POOLS = {"kv_pool"}


def load_light(d, layer):
    """返回 {tag: tensor}。"""
    out = {}
    for f in glob.glob(os.path.join(d, f"L{layer}_*.pt")):
        tag = os.path.basename(f)[len(f"L{layer}_"):-3]
        try:
            o = torch.load(f, map_location="cpu", weights_only=False)
        except TypeError:
            o = torch.load(f, map_location="cpu")
        t = o.get("t") if isinstance(o, dict) else o
        if isinstance(t, torch.Tensor):
            out[tag] = t
    return out


def load_mini(d, layer):
    """返回按执行顺序排列的快照列表 [{_key,_stage,张量...}]。"""
    snaps = []
    pat = re.compile(rf"^L{layer}_(\d+)_(.+)\.pt$")
    for f in sorted(glob.glob(os.path.join(d, f"L{layer}_*.pt"))):
        m = pat.match(os.path.basename(f))
        if not m:
            continue
        try:
            o = torch.load(f, map_location="cpu", weights_only=False)
        except TypeError:
            o = torch.load(f, map_location="cpu")
        if isinstance(o, dict):
            o["_n"] = int(m.group(1))
            snaps.append(o)
    snaps.sort(key=lambda s: s["_n"])
    return snaps


def pick(snaps, key=None, stage=None, field=None, seen=None):
    """取满足 (key, stage) 的快照里第一个含 field 的；key=None → 全层搜 field。"""
    for s in snaps:
        if key is not None and s.get("_key") != key:
            continue
        if stage is not None and (s.get("_stage") or "") != stage:
            continue
        if field in s and isinstance(s[field], torch.Tensor):
            if seen is not None:
                seen.add(id(s))
            return s[field], s
    return None, None


def _align(ref, got, pool=False):
    """把 [T,M,H] 与 [T,H] 对齐；池压平；返回 (r,g,note)。"""
    note = ""
    if ref.numel() == 0 or got.numel() == 0:      # L2 的官方 kv_pool 有 0 元素（该点没缓存）
        return None, None, f"空张量 ref{tuple(ref.shape)} got{tuple(got.shape)}"
    if pool:                                  # 池：**按块**压成 [块号, 128×544 字节]
        BS, ROW = 128, 544                    # 官方 [nblk,128,1,544]；mini [(1+nblk)*128, 544]
        ref = ref.reshape(-1).view(-1, BS * ROW)
        got = got.reshape(-1)
        if got.numel() % (BS * ROW) == 0:
            got = got.view(-1, BS * ROW)
    if ref.dim() != got.dim():
        if ref.dim() == 3 and got.dim() == 2:
            ref = ref[:, 0, :]
            note = "ref取M0"
        elif ref.dim() == 2 and got.dim() == 3:
            # 检查各 M 份是否一致
            same = bool(torch.equal(got, got[:, :1, :].expand_as(got)))
            got = got[:, 0, :]
            note = "got取M0" + ("" if same else "(M份不一致!)")
    if ref.dim() == got.dim() and ref.dim() >= 2 and ref.shape[0] != got.shape[0]:
        n = min(ref.shape[0], got.shape[0])
        ref, got = ref[:n], got[:n]
        note += f" 行{ref.shape[0]}"
    if ref.shape != got.shape:
        return None, None, f"形状不匹配 ref{tuple(ref.shape)} vs got{tuple(got.shape)}"
    return ref, got, note


def _stat(ref, got):
    if ref.dtype != got.dtype and not (ref.is_floating_point() and got.is_floating_point()):
        pass
    if ref.is_floating_point() or got.is_floating_point():
        r = ref.to(torch.float64)
        g = got.to(torch.float64)
        d = (r - g).abs()
        nd = int((ref.to(torch.float32) != got.to(torch.float32)).sum())
        tot = ref.numel()
        maxa = float(d.max()) if tot else 0.0
        nr = float(r.norm())
        rel = float(d.norm()) / nr if nr > 0 else (0.0 if maxa == 0 else float("inf"))
        # bf16 分辨率下“显著不同”的比例（|Δ| > 1e-2 相对量级只作参考）
        return nd, tot, maxa, rel, ""
    # 整型/字节：逐元素精确
    nd = int((ref != got).sum())
    tot = ref.numel()
    return nd, tot, (0.0 if nd == 0 else 1.0), (0.0 if nd == 0 else 1.0), ""


def fmt(v):
    if v == 0:
        return "0"
    if v != v:
        return "nan"
    return f"{v:.3e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini", required=True)
    ap.add_argument("--light", required=True)
    ap.add_argument("--layers", default="0,1")
    ap.add_argument("--json", default="")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="rel_fro 超过它才算分歧（默认 0 = 任何不一致都算）")
    a = ap.parse_args()

    layers = [int(s) for s in a.layers.split(",") if s.strip()]
    report = {}
    print(f"mini = {a.mini}\nlight = {a.light}\nlayers = {layers}\n")
    print(f"{'层':>2} {'官方tag':<22} {'mini快照':<26} {'形状':<18} "
          f"{'不同/总数':>14} {'max|Δ|':>10} {'rel_fro':>10}")
    print("-" * 122)
    for L in layers:
        lt = load_light(a.light, L)
        ms = load_mini(a.mini, L)
        if not lt:
            print(f"{L:>2} ⚠️ 官方目录没有 L{L}_*.pt")
        if not ms:
            print(f"{L:>2} ⚠️ mini 目录没有 L{L}_*.pt")
            continue
        rows = []
        for tag, key, stage, field, desc in BOUNDARY:
            ref = lt.get(tag)
            if ref is None:
                continue
            got, snap = pick(ms, key, stage, field)
            if got is None:
                rows.append((tag, "(缺)", "", "", "", "", "官方有/mini 缺"))
                continue
            is_pool = tag in POOLS
            # 官方 `_save` 把 float8 池 cast 成 float32（is_floating_point ✓）→ 同款变换后再比
            if is_pool and ref.dtype == torch.float32 and got.dtype == torch.uint8:
                got = got.view(torch.float8_e4m3fn).to(torch.float32)
            r, g, note = _align(ref, got, pool=is_pool)
            if r is not None and is_pool and not torch.equal(r, g):
                # 块号偏移自适应：官方块表可能 0 基、mini 1 基（块 0 空）→ 试 -2..2
                best = None
                for sh in (-2, -1, 1, 2):
                    if sh > 0:
                        a_, b_ = r[sh:], g[:r.shape[0] - sh]
                    else:
                        a_, b_ = r[:sh], g[-sh:]
                    n = min(a_.shape[0], b_.shape[0])
                    d = int((a_[:n] != b_[:n]).sum())
                    if best is None or d < best[0]:
                        best = (d, sh, n)
                if best is not None and best[0] == 0:
                    note += f" 块号偏移{best[1]:+d}后逐字节一致"
                    r, g = g[:best[2]], g[:best[2]]      # 自比 → 全 0
            if r is None:
                rows.append((tag, f"{snap['_key']}#{snap['_n']}", "-", "-", "-", "-", note))
                continue
            nd, tot, maxa, rel, extra = _stat(r, g)
            if is_pool and nd > 0:
                rowbad = (r.reshape(r.shape[0], -1) != g.reshape(g.shape[0], -1)).any(dim=1)
                first = int(rowbad.to(torch.int32).argmax()) if bool(rowbad.any()) else -1
                note += f" 首错块={first}"
            snapname = f"{snap['_key']}#{snap['_n']}" + (f"[{snap['_stage']}]" if snap.get("_stage") else "")
            rows.append((tag, snapname, str(tuple(ref.shape)), f"{nd}/{tot}",
                         fmt(maxa), fmt(rel), note))
        # 打印本层
        first_bad = None
        for tag, snapname, shape, ndstr, maxs, rels, note in rows:
            bad = ""
            try:
                nd = int(ndstr.split("/")[0])
                rel = float(rels)
            except Exception:                     # noqa: BLE001
                bad = "⚠️"
                nd, rel = 1, 1.0
            if nd > 0 and (rel > a.tol):
                bad = "❌"
                if first_bad is None:
                    first_bad = tag
            elif nd == 0:
                bad = "✅"
            print(f"{L:>2} {tag:<22} {snapname:<26} {shape:<18} {ndstr:>14} {maxs:>10} {rels:>10} {bad} {note}")
            report.setdefault(str(L), {})[tag] = dict(
                snap=snapname, nd=ndstr, max_abs=maxs, rel_fro=rels, note=note)
        print(f"{'':>2} {'':<22} ⇒ 第一处分歧: " +
              (f"**{first_bad}**" if first_bad else "无（本层全部逐位相同）"))
        # 跨层连续性：本层 layer_out ↔ 官方下一层 layer_in（若官方有）
        out, sn = pick(ms, "hc_post", "ffn", "x")
        nxt = load_light(a.light, L + 1).get("layer_in")
        if out is not None and nxt is not None:
            r, g, note = _align(nxt, out)
            if r is not None:
                nd, tot, maxa, rel, _ = _stat(r, g)
                tag = "layer_out ↔ L%d_layer_in" % (L + 1)
                print(f"{L:>2} {tag:<22} {'hc_post#%d' % sn['_n']:<26} {str(tuple(nxt.shape)):<18} "
                      f"{nd}/{tot:>13} {fmt(maxa):>10} {fmt(rel):>10}"
                      f" {'✅' if nd == 0 else '❌'}")
                report.setdefault(str(L), {})[tag] = dict(nd=nd, tot=tot, rel_fro=rel)
        print("-" * 122)
        # 也把 mini 快照清单打出来，便于核对“官方缺哪些点”
        print(f"   mini 快照序列: " + ", ".join(f"{s['_key']}#{s['_n']}"
                                              + (f"[{s['_stage']}]" if s.get("_stage") else "")
                                              for s in ms))
        print()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print(f"已写 {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

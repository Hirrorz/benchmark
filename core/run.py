"""mini 驱动：--dry-run 先出算子序列（M1），M2 起接真跑（执行 + msprof + 进度）。

M1 目标：不碰卡，把"40 层每层发哪些算子、什么形状、各多少次"打印出来，
并用 tools/align_ops.py 与真实 trace 对齐。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ds41_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    _q = _os.path.join(_ds41_root, _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)


import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import assemble                      # noqa: E402
import opgraph as ops_mod                # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 树根（2026-10-26 分包）


_RECIPES_CACHE: dict = {}


def load_recipes(path=None):
    """读 `recipes.yaml`（按 **路径+mtime+size** 缓存，文件不变就复用同一份）。

    ★ T-042（host 瘦身）：`_run_forward_impl` 每次 forward 都会调用本函数 ——
      实测 **4.40 ms/次**（yaml 走的是纯 Python 解析器，13k+ 行的 reader 循环）。
      调用方（compose/expand_layer/tree）只读不写，所以复用同一份对象是安全的；
      键里带 mtime/size ⇒ 改了 recipes.yaml 会自动重读（不会吃到旧计划）。
    """
    p = path or os.path.join(HERE, "recipes.yaml")
    try:
        _st = os.stat(p)
        _key = (p, int(_st.st_mtime_ns), int(_st.st_size))
    except OSError:                          # 文件没了 → 交给 open 抛原来的错
        _key = None
    if _key is not None:
        _hit = _RECIPES_CACHE.get(_key)
        if _hit is not None:
            return _hit
    import yaml
    with open(p) as f:
        data = yaml.safe_load(f)
    if _key is not None:
        _RECIPES_CACHE.clear()               # 只留最新一份（文件通常只有一份）
        _RECIPES_CACHE[_key] = data
    return data


def ctx_for(cfg, layer):
    """喂给 ops.py 里 count/shapes 的上下文（层角色 flags 供"数量"判断用）。"""
    m = cfg["model"]
    run = cfg["run"]
    pc = cfg["parallel_config"]
    # ★★ 2026-10-17 —— 命中 stage B 读前缀"串到别的层数据"的根因修（`DS41M_PERLAYER_CACHE=1`，默认关）
    #   现状：所有池/缓存**只按 ratio 分**（`_cmp_pool_r2`、`_full_pool` …）⇒ 一个 pass 内多个
    #   写者共用一块，靠"读永远紧跟自己的写"侥幸成立（**故 prefill 与官方逐位一致 ✓**）；
    #   但**命中 stage B 读前缀**时池里是 stage A 的**最后一个写者**：
    #     · 压缩池 ratio=2：写者 = 源层 2/8/14 ⇒ 消费者 3..13 读到的是**源 14** 的前缀 KV ✗
    #     · 窗口池：`kv_epilog` **40 层全都写** ⇒ stage B 的 L0 读到的是**层 39** 的前缀窗口 ✗
    #   实测签名：命中末 token `' that' 19.250` vs prefill `33.750`（差 14.5，且**逐跑逐位可复现** ✗）。
    #   修：给每个缓存加"**拥有者**"后缀 —— 压缩/索引缓存按**产生它的源层**、窗口池按**本层**；
    #   命中时窗口同时切**绝对位置**（每层池已含本层前缀 ⇒ 可读 ✓）。off==0（prefill）时
    #   绝对==局部 ⇒ **对 prefill 逐位无损** ✓✓（后缀为空时与旧行为逐位相同 ✓）。
    _pl = os.environ.get("DS41M_PERLAYER_CACHE", "").strip().lower() in ("1", "on", "true", "yes")
    if _pl:
        _i = int(layer["idx"])
        _ks = [s for s in (m.get("kv_source_layers") or []) if s <= _i]
        _is = [s for s in (m.get("index_source_layers") or []) if s <= _i]
        suf_win = f"_L{_i}"
        suf_cmp = f"_o{max(_ks) if _ks else _i}"
        suf_idx = f"_o{max(_is) if _is else _i}"
        # 裁剪尾层（decoder_replay）**不能**用绝对窗口：stage A 只写前缀最后 w 行、stage B 只写
        # 尾段最后 w 行 ⇒ 池中间有缺口（位置 H-w..L-w 未写 ✗）⇒ 绝对索引会读到缺口。
        # 判据：本层 T == 本轮 T_main（assemble：`t = w if cut and i>=cut else T_main`）。
        _h = float((cfg.get("hit") or {}).get("ratio", 0.0) or 0.0)
        _T_main = assemble.t_main(cfg)
        win_abs = (int(layer.get("T", _T_main)) == _T_main)
    else:
        suf_win = suf_cmp = suf_idx = ""
        win_abs = False
    return dict(
        T=layer["T"], chunk=int(run["moe_chunk_max_len"]),
        H=m["hidden"], I=m["moe_intermediate"], ratio=layer["ratio"],
        rtopk=m["topk"], e_local=m["n_routed_experts"] // max(1, pc["moe_ep_size"]),
        S=m["sliding_window"], idx=layer["idx"], head_dim=m.get("head_dim", 512),
        # ★ DeepInfer `swa_block_size = window + 6`：固定常量（不随草稿长度变化），
        #   让 ring/图形状稳定。读取侧窗口仍是 S（128），多存的 6 行只在回退时被读到。
        win_keep=assemble.swa_block_tokens(cfg),
        index_topk=m.get("index_topk", 512),
        rope_theta=m.get("rope_theta", 10000),
        compress_rope_theta=m.get("compress_rope_theta", 160000),
        rope_scaling=m.get("rope_scaling") or {},
        device=(cfg["run"].get("device") or "npu:0"),
        is_kv_src=int(layer["is_kv_src"]), is_idx_src=int(layer["is_idx_src"]),
        is_candidate=int(layer["is_candidate"]), engram=int(layer["engram"]),
        suffix_win=suf_win, suffix_cmp=suf_cmp, suffix_idx=suf_idx, win_abs=win_abs,
    )


def _skip_moe():
    """性能口径：跳过 MoE（gate/topk/mega_moe 不跑，x 直通）。DFS 只测其余算子。"""
    return os.environ.get("DS41M_SKIP_MOE", "").strip().lower() in ("1", "on", "true", "yes")


def _skip_engram():
    """性能口径：跳过 engram（fetch/wait/proj 不跑，表也不加载）。"""
    return os.environ.get("DS41M_SKIP_ENGRAM", "").strip().lower() in ("1", "on", "true", "yes")


def expand_items(recipes, items, attn_kind, moe_key):
    """展开 `_xxx` 段落引用（attn_core / attention / moe / engram …）。"""
    seq = []
    for it in items:
        if not it.startswith("_"):
            seq.append(it)
            continue
        key = it[1:]
        if key == "attention":
            sub = recipes["attention"][attn_kind]
        elif key == "moe":
            sub = [] if _skip_moe() else recipes["moe"][moe_key]
        else:
            sub = recipes[key]
        seq += expand_items(recipes, sub, attn_kind, moe_key)
    return seq


# ★ 2026-10-18 性能对齐：recipes 里的 mul/add/cast/tmove 是**纯占位**
#   （x*1.0 / x+0.0 / x→fp32→bf16 / .contiguous()，数值恒等），只为对齐官方 trace 的
#   算子数。默认**不下发**（省 ~160 次 kernel + host 调度，实测空泡主体）；
#   需要 trace-shape 对拍时设 DS41M_PAD_OPS=1 恢复。
_PAD_OPS = {"mul", "add", "cast", "tmove"}


def _pad_ops_enabled():
    return os.environ.get("DS41M_PAD_OPS", "").strip().lower() in ("1", "on", "true", "yes")


def expand_layer(cfg, recipes, layer):
    backend = (cfg.get("backends") or {}).get("megamoe", "deepgemm")
    moe_key = "fused" if backend in ("ascendc", "deepgemm") else "unfused"
    seq = []
    if layer["engram"] and not _skip_engram():
        seq += list(recipes["engram"])
    seq += expand_items(recipes, recipes["common"], layer["attn"], moe_key)
    if not _pad_ops_enabled():
        seq = [n for n in seq if n not in _PAD_OPS]
    return seq


def apply_ops(counts, recipes, seq, ctx):
    for name in seq:
        spec = ops_mod.OPS.get(name)
        if spec is None:
            raise SystemExit(f"recipes 里有未知算子: {name!r}（请先登记到 opgraph.py）")
        n = spec["count"](ctx)
        if n:                       # 0 次的不进直方图（层角色没命中）
            counts[spec["kernel"]] += n


def dry_run(cfg, recipes, dump_json=""):
    import collections
    stack, meta = assemble.build_stack(cfg)
    hist = collections.Counter()
    layers_out = []
    for layer in stack:
        ctx = ctx_for(cfg, layer)
        seq = expand_layer(cfg, recipes, layer)
        counts = collections.Counter()
        apply_ops(counts, recipes, seq, ctx)
        hist.update(counts)
        layers_out.append(dict(idx=layer["idx"], attn=layer["attn"], T=layer["T"],
                               ops=dict(counts)))
    # 会话级残差（extras）
    extras = collections.Counter()
    for item in recipes.get("extras") or []:
        name, n = item
        spec = ops_mod.OPS.get(name)
        if spec is None:
            raise SystemExit(f"extras 里有未知算子: {name!r}")
        extras[spec["kernel"]] += int(n)
    hist.update(extras)
    return meta, layers_out, hist, extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true", help="只打印算子序列（不碰卡）")
    ap.add_argument("--dump-json", default="", help="把结果写成 json（align_ops 用）")
    ap.add_argument("--layers", default="", help="只打印这些层，如 0,20,21")
    ap.add_argument("--progress", default="auto")
    a = ap.parse_args()

    cfg = assemble.load_yaml(a.config)
    # varlen 是唯一 batch 口径：assemble.build_stack 里的 ensure_packed 会补默认描述
    recipes = load_recipes()
    meta, layers_out, hist, extras = dry_run(cfg, recipes, a.dump_json)

    print(f"== dry-run：L={meta['L']} → T_main={meta['T_main']}"
          + (f"（decoder128 cut={meta['cut']}，其后 T={meta['w']}）" if meta["cut"] else "")
          + f" | backend.megamoe={cfg['backends']['megamoe']} | chunks=T/{cfg['run']['moe_chunk_max_len']}")
    show = {int(x) for x in a.layers.split(",") if x.strip()} if a.layers else None
    print(f"\n{'层':>3} {'T':>7} {'attn':<16} 算子（次数）")
    for lo in layers_out:
        if show and lo["idx"] not in show:
            continue
        ops_str = ", ".join(f"{k}×{v}" for k, v in sorted(lo["ops"].items(), key=lambda x: -x[1]))
        print(f"{lo['idx']:>3} {lo['T']:>7} {lo['attn']:<16} {ops_str[:150]}")

    print("\n== 全会话算子直方图（应与 align_ops 的参考 trace 对齐）==")
    for k, v in hist.most_common(30):
        tag = "   ← extras" if k in extras else ""
        print(f"  {v:>7}  {k}{tag}")

    if a.dump_json:
        with open(a.dump_json, "w") as f:
            json.dump(dict(meta=meta, layers=layers_out, hist=dict(hist),
                           extras=dict(extras)), f, indent=1)
        print(f"\n已写 {a.dump_json}")


if __name__ == "__main__":
    main()

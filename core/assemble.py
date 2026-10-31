"""按 config 把 40 层堆起来（§7：结构变化只改 config），并解析每层的类型/角色。

层角色由 config 的四张表决定（与官方 config.json 同源）：
  compress_ratios / kv_source_layers / index_source_layers / candidate_source_layer / engram_layers
再叠加 features.decoder_replay 与 hit 口径，得到**每层实际处理的 token 数**。
"""
from __future__ import annotations

import metadata
import argparse
import math
import os


# ── DeepInfer 固定常量：KEEP / SWA block 都不随草稿长度动态变化 ──────────────
# 参考 deepinfer/environ.py:188-196：KEEP_TOKENS=140 是 V4.1 的**统一固定值**；
# 官方多留的 6 行余量是为了让所有 L/h/草稿配置共用同一个形状（利于入图），
# 不是“每跑一次按理论下界动态算”。ring 的 swa_block_size=window+6 同样是固定常量。
_DEEPINFER_SWA_EXTRA_TOKENS = 6      # window 之外的固定回退余量（swa_block_size 用）
_DEEPINFER_V41_KEEP_TOKENS = 140     # V4.1 固定 KEEP（replay / 命中保留后缀）


def swa_block_tokens(cfg) -> int:
    """SWA ring 的固定长度 = window + 6（DeepInfer `swa_block_size`）。"""
    return int(cfg["model"].get("sliding_window", 128)) + _DEEPINFER_SWA_EXTRA_TOKENS


def replay_keep_tokens(cfg) -> int:
    """decoder replay / 命中 replay 保留的末尾 token 数（DeepInfer 固定 KEEP）。

    固定 140（V4.1），不随 DSpark/回退长度变化 → 图捕获的 W 恒定。
    `DS41M_REPLAY_KEEP` 仅用于显式换形状（A/B），不是按回退动态算。
    """
    keep = max(_DEEPINFER_V41_KEEP_TOKENS, swa_block_tokens(cfg))
    raw = os.environ.get("DS41M_REPLAY_KEEP", "").strip()
    if raw:
        keep = max(keep, int(raw))
    return keep


def replay_tail_of(cfg) -> int:
    """`hit.replay_tail` 的有效值：显式配置与 keep_tokens 取大（旧口径已删除）。"""
    hit = cfg.get("hit") or {}
    v = int(hit.get("replay_tail", 0) or 0)
    return max(v, replay_keep_tokens(cfg))


def hit_m(cfg, L: int) -> int:
    """命中口径的 non-cached token 数（DeepInfer 口径）。

    `hit.non_cached_tokens` 优先，**直接使用、不做 128 对齐**（DeepInfer benchmark 的
    `--non_cached_tokens` 同义）；否则由 `hit.ratio` 换算 `round((1-h)*L)`（仅作
    CLI/旧配置兼容，仍不做块对齐）。旧 `round(.../128)*128` 口径已删除。
    """
    hit = cfg.get("hit") or {}
    nc = int(hit.get("non_cached_tokens", 0) or 0)
    if nc > 0:
        return min(nc, int(L))
    h = float(hit.get("ratio", 0.0) or 0.0)
    return int(round((1.0 - h) * int(L)))


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def attention_kind(cfg, layer_idx):
    """返回 recipes.yaml 里 attention 段的键名。"""
    m = cfg["model"]
    ratio = m["compress_ratios"][layer_idx]
    if ratio == 0:
        return "win"
    if layer_idx in m["kv_source_layers"]:
        return "c1a_source" if ratio == 1 else "c2a_source"
    if layer_idx in m["index_source_layers"]:
        return "c1a_indexsource"
    return "c1a_consumer" if ratio == 1 else "c2a_consumer"


def t_main(cfg) -> int:
    """本次前向实际算多少 token（命中：non_cached + keep；否则全长 L）。

    DeepInfer 口径：non_cached 由 `hit.non_cached_tokens` 直接给（无 128 对齐），
    或由 `hit.ratio` 换算（兼容 CLI，仍无块对齐）。
    """
    run = cfg["run"]
    hit = cfg.get("hit", {}) or {}
    L = int(run["seq_len"])
    if float(hit.get("ratio", 0.0) or 0.0) > 0 or int(hit.get("non_cached_tokens", 0) or 0) > 0:
        return hit_m(cfg, L) + replay_tail_of(cfg)
    return L


def ensure_packed(cfg):
    """varlen 是唯一 batch 口径（2026-10-26）：缺 `_packed` 时补默认单序列描述。

    供 dry-run / 工具（`check_module_spec`、`align_ops`）使用；真跑入口
    （points_runner/runner）自己会写真实的 n/lens/row_lens/cu。
    """
    if not isinstance(cfg.get("_packed"), dict):
        L = int((cfg.get("run") or {}).get("seq_len") or 0)
        T = int(t_main(cfg))
        cfg["_packed"] = {"n": 1, "lens": [L], "row_lens": [T], "cu": [0, T]}
    return cfg


def replay_cut_w(cfg):
    """decoder replay 的默认几何：
    (cut = 最后一个 kv_source + 1, w = KEEP_TOKENS)。

    DeepInfer 的 KEEP_TOKENS 规则（V4.1 默认 140，>= window + backtracking）：
    保留后缀必须覆盖下一 token 的完整 SWA 窗口，并在 DSpark/MTP 回退后仍然够用。
    """
    m = cfg["model"]
    cut = max(int(x) for x in (m.get("kv_source_layers") or [0])) + 1
    return cut, replay_keep_tokens(cfg)


def build_stack(cfg):
    """→ [ {idx, ratio, attn, is_kv_src, is_idx_src, is_candidate, engram, T} ]"""
    meta = metadata.ensure(cfg)   # 一次 forward 的唯一元数据（batch/geometry/policy）
    m = cfg["model"]
    run = cfg["run"]
    feat = cfg.get("features", {})
    hit = cfg.get("hit", {}) or {}
    L = int(run["seq_len"])

    # ── 本次前向实际要算多少 token（§10 命中口径；公式统一在 t_main）──
    h = float(hit.get("ratio", 0.0) or 0.0)
    T_main = t_main(cfg)                    # **每序列**本段 token 数（命中：M+replay_tail）
    _ov = (cfg.get("_meta_override") or {})   # ★ CP 命中段用全局口径覆盖（见 runner 的 CP 分支）
    if _ov:                                  # ★ CP 命中：L=全局序列长，T_main=全局 R
        L = int(_ov.get("L", L))
        T_main = int(_ov.get("T_main", T_main))
    # ★ 2026-10-28（命中 + CP）：`_packed` 里的 lens/row_lens 是**本段局部**几何，
    #   局部 lens-row_lens 恒为 0 ⇒ 命中段需要在正确的绝对起点上重算压缩/索引缓存。
    #   这里允许 CP 分支直接给出 pos_offs（全局起点），未给时沿用旧公式（逐位不变）。
    _ov_poffs = _ov.get("pos_offs") if _ov else None
    # ★ 规则（PERF_RULES §0）：packed batch = tokenwise 打包 + attention 恢复。
    #   `cfg["_packed"]["cu"]` 是输入行边界（每序列 R_i 行），`lens` 是每序列**逻辑**长度 L_i。
    #   ★ 2026-10-21（逐序列几何）：各批可以有**不同 L / 不同命中率** → R_i 逐序列不同，
    #     本 pass 行数 = ΣR_i（不再是 T_main×N）；T_main 仍是"单序列"口径（兼容老路径）。
    # ★ 2026-10-26：varlen 是**唯一** batch 口径 ⇒ `_packed` 必填（所有入口都会设）。
    _lay = meta.layout
    _pn = int(_lay.n)
    _rows = [int(x) for x in _lay.row_lens]
    _lens = [int(x) for x in _lay.seq_lens]
    assert len(_rows) == _pn and len(_lens) == _pn, (
        f"cfg['_packed'] 形状不一致：n={_pn} row_lens={_rows} lens={_lens}")
    T_pass = sum(_rows)

    # ── decoder128（§8）：cut 之后的层只算最后 w 个 token ──
    cut, w = None, None
    if feat.get("decoder_replay"):
        cut = max(m["kv_source_layers"]) + 1
        # ★ DeepInfer KEEP_TOKENS 规则：V4.1 默认 140（>= 窗口 + 最大回退）。
        w = replay_keep_tokens(cfg)
        # ★ 2026-10-23（CP 对齐 DeepInfer）：CP 下的截尾由 runner 的 `_cp_replay_trim`
        #   按 compact-mirror 跨 rank 重分布（contrib/boundary 索引），不再禁止 cp>1。
        #   （层 T 只是 ctx 提示；真实行数以 `_packed.cu` 为准。）
        if max(_rows) <= w:
            cut = None                      # 每序列都 <=w 时等价不裁

    stack = []
    for i in range(m["n_layers"]):
        t = (w * _pn) if (cut is not None and i >= cut) else T_pass
        stack.append(dict(
            idx=i,
            ratio=m["compress_ratios"][i],
            attn=attention_kind(cfg, i),
            is_kv_src=i in m["kv_source_layers"],
            is_idx_src=i in m["index_source_layers"],
            is_candidate=(i == m["candidate_source_layer"]),
            engram=(i in m["engram_layers"]),
            T=t,
        ))
    _poffs = ([max(0, _lens[i] - _rows[i]) for i in range(_pn)] if _ov_poffs is None
              else [int(x) for x in _ov_poffs])
    meta = dict(T_main=T_main, T_mains=_rows, T_pass=T_pass, n=_pn, cut=cut, w=w, L=L, hit=h,
                lens=_lens, pos_offs=_poffs)
    return stack, meta


def print_stack(cfg, stack, meta):
    m = cfg["model"]
    print(f"== DS4.1-Flash 最小 benchmark 层栈（{m['n_layers']} 层）==")
    print(f"L={meta['L']} → 本次前向 T_main={meta['T_main']}"
          + (f"；decoder128: cut={meta['cut']}，其后 T={meta['w']}" if meta["cut"] else "")
          + f"；Hit={meta['hit']:.2%}" if meta["hit"] else "")
    print(f"{'层':>3} {'ratio':>5} {'attn 类型':<16}{'kv_src':>7}{'idx_src':>8}{'候选':>5}{'engram':>7}{'T':>8}")
    for s in stack:
        print(f"{s['idx']:>3} {s['ratio']:>5} {s['attn']:<16}"
              f"{'✓' if s['is_kv_src'] else '':>7}{'✓' if s['is_idx_src'] else '':>8}"
              f"{'✓' if s['is_candidate'] else '':>5}{'✓' if s['engram'] else '':>7}{s['T']:>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--print-stack", action="store_true")
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    stack, meta = build_stack(cfg)
    if a.print_stack or True:
        print_stack(cfg, stack, meta)


if __name__ == "__main__":
    main()

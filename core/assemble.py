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
    """SWA ring 的固定长度 = window + 6（DeepInfer `swa_block_size`）。

    ★ 2026-11-01（B4 修复）：`cfg` 可能只是 `{"hit": {...}}` 的合成最小 dict
    （`points_runner._m_of` 就这样传），所以读 `model` 必须容忍缺失。
    """
    return int((cfg.get("model") or {}).get("sliding_window", 128)) + _DEEPINFER_SWA_EXTRA_TOKENS


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
    """`hit.replay_tail` 的有效值：显式配置与 keep_tokens 取大（旧口径已删除）。C5：走 metadata 单源。"""
    v = int(metadata.hit_of(cfg).replay_tail or 0)
    return max(v, replay_keep_tokens(cfg))


def hit_alignment(cfg) -> tuple:
    """命中对齐口径 —— `('token', 1)` 或 `('block', BS)`。

    ★ 2026-11-01（B2 吸收，GPT §12.2 `hit_alignment=token_exact|block_aligned`）：
      · `token`（**默认**，逐位不变）：非缓存段按 token 精确切，`M = round((1-h)L)`；
      · `block[:BS]`：把非缓存段**向上取整到 BS 的倍数**，让重算段与 KV 缓存块边界对齐，
        少算"块内的部分行"。默认 BS=128（与本仓 SWA/压缩块的粒度一致）。
    只影响 `hit_m`，几何收口（`check_hit_geometry`）仍会拒绝 `M==0` / `R>L`。
    """
    hit = metadata.hit_of(cfg)
    # ★ 优先级：cfg 显式 alignment > DS41M_HIT_ALIGN > 默认 token
    #   （HitPolicy.alignment 为空串表示"未显式指定"，env 才能生效）
    v = str(hit.alignment or os.environ.get("DS41M_HIT_ALIGN", "")
            or "token").strip().lower()
    if v in ("", "token", "token_exact", "exact", "none"):
        return "token", 1
    if v.startswith("block"):
        bs = 128
        if ":" in v:
            try:
                bs = int(v.split(":", 1)[1])
            except ValueError:
                raise ValueError(
                    f"hit.alignment 的块大小不合法：{v!r}（应写成 block:128）") from None
        if bs < 1:
            raise ValueError(f"hit.alignment 的块大小必须 ≥1：{v!r}")
        return "block", int(bs)
    # ★ 2026-11-01（B4 吸收，GPT §7/§14 `partition=mirror_aligned`）：
    #   CP 的 mirror 分块是「每 rank 两段，各 ps = ceil(R/(2·cp)) 行」。要让这两段
    #   **正好落在 KV 块边界**上（不产生半块重算/半块 pad），等价于要求
    #   `R % (2·cp·BS) == 0`。所以 B4 落在**几何层**：把 R 向上取整到 `2·cp·BS`。
    #   ★ 为什么不去改 `cp_layout` 的分块公式：ps 放大后块会越过 seq 末尾
    #   （R < 2·cp·ps 时 head/tail 段重叠或丢尾，`inverse_reverse` 只能指一个全局行）。
    #   这里只调"算多少行"，分块公式保持 exact mirror —— 不引入第三套几何。
    if v.startswith("cpblock"):
        bs = 128
        if ":" in v:
            try:
                bs = int(v.split(":", 1)[1])
            except ValueError:
                raise ValueError(
                    f"hit.alignment 的块大小不合法：{v!r}（应写成 cpblock:128）") from None
        if bs < 1:
            raise ValueError(f"hit.alignment 的块大小必须 ≥1：{v!r}")
        return "cpblock", int(bs)
    raise ValueError(
        f"未知的 hit.alignment={v!r}：只支持 token（默认）/ block[:BS] / cpblock[:BS]")


def hit_m(cfg, L: int) -> int:
    """命中口径的 non-cached token 数（DeepInfer 口径）。

    `hit.non_cached_tokens` 优先，**直接使用、不做 128 对齐**（DeepInfer benchmark 的
    `--non_cached_tokens` 同义）；否则由 `hit.ratio` 换算 `round((1-h)*L)`（仅作
    CLI/旧配置兼容，仍不做块对齐）。旧 `round(.../128)*128` 口径已删除。

    ★ 2026-11-01（B2）：换算完再按 `hit_alignment()` 处理 —— 默认 `token` 时**一位不变**；
    只有显式 `block[:BS]` 才向上取整到块边界（可能把 `R` 顶过 `L`，由几何收口显式拒绝）。
    """
    hit = metadata.hit_of(cfg)
    nc = int(hit.non_cached_tokens or 0)
    if nc > 0:
        m = min(nc, int(L))
    else:
        h = float(hit.ratio or 0.0)
        m = int(round((1.0 - h) * int(L)))
    _kind, _bs = hit_alignment(cfg)
    if _kind == "block" and m > 0:
        m = min(int(L), ((int(m) + _bs - 1) // _bs) * _bs)
    elif _kind == "cpblock" and m > 0:
        # ★ B4：R = M + keep 必须是 `2·cp·BS` 的整数倍（cp_layout 的 mirror 分块才会
        #   每段正好 BS 对齐）。cp 从 env 读（与 cp_state/cp_layout 同一口径）。
        cp = int(os.environ.get("DS41M_CP_SIZE", "1") or 1)
        unit = 2 * max(1, cp) * int(_bs)
        keep = int(replay_tail_of(cfg))
        R2 = ((int(m) + keep + unit - 1) // unit) * unit
        m = int(R2) - keep
    return int(m)


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


def hit_is_on(cfg) -> bool:
    """命中口径是否生效（`hit.ratio>0` 或显式 `non_cached_tokens>0`）。C5：走 metadata 单源。"""
    return bool(metadata.hit_of(cfg).is_on())


def check_hit_geometry(cfg, L: int, where: str = "hit") -> int:
    """命中几何的唯一收口：`M>0` 且 `R<=L`，否则**显式拒绝**（铁律 0：不静默 clamp、不改几何）。

    → 本段应算的行数 `R = hit_m + replay_tail`；命中关闭时返回全长 `L`。

    为什么要在算之前拦：`h < replay_tail/L`（短序列 + 低命中）会让 `R>L`，
    后续按 R 行的池/索引越界；`h=1`（或 `non_cached_tokens>=L`）会得到 `M=0` 的
    "纯 replay 段"，与 prefill 语义混用。历史脚本 `tools/finish_cp2cp4.sh:164`
    直接在生成点位时 `continue` 把这类点**静默丢掉**（点位矩阵缺测），
    这正是本函数要消灭的静默行为（见 WORKPLAN 缺陷 G1）。
    """
    L = int(L)
    if not hit_is_on(cfg):
        return L
    M = int(hit_m(cfg, L))
    keep = int(replay_tail_of(cfg))
    R = M + keep
    if M <= 0:
        raise ValueError(
            f"{where}：命中几何非法（非缓存段为空）—— L={L}、M={M}、replay_tail={keep}、"
            f"R={R}（hit={cfg.get('hit')!r}）。全命中没有可测的 prefill 语义，显式拒绝。")
    if R > L:
        raise ValueError(
            f"{where}：命中几何非法（要算的行数超过序列长）—— L={L}、M={M}、"
            f"replay_tail={keep}、R={R}（hit={cfg.get('hit')!r}）。"
            f"短序列/低命中（h < replay_tail/L）无几何解：提高 h 或缩短 replay_tail。")
    return R


def backends_of(cp: int) -> str:
    """本次实际生效的后端（默认值与各模块读取 env 的地方同口径）。`auto` = 运行期选。

    只用于**记账**（结果表 / 日志），不改变任何执行路径。
    """
    def _v(name, dflt):
        return (os.environ.get(name, "").strip().lower() or dflt)

    on = ("1", "on", "true", "yes")
    # ★ 2026-11-01（B2）：把命中对齐口径也记进后端串 —— 换 alignment 后结果表能分辨
    try:
        _ak, _ab = hit_alignment({})
        if _ak == "block":
            _al = f"block{_ab}"
        elif _ak == "cpblock":
            _al = f"cpblock{_ab}"
        else:
            _al = "token"
    except Exception:                                  # noqa: BLE001
        _al = "?"
    return ("idx={},qsli={},moe={},cp_cmp={},cp_hit={},cp={},k_gather={},qli={},align={}"
            .format(_v("DS41M_IDX_KERNEL", "dg"),
                    _v("DS41M_QSLI_KERNEL", "dg"),
                    _v("DS41M_MOEMODE", "auto"),
                    _v("DS41M_CP_CMP_FLOW", "seg"),
                    _v("DS41M_CP_HIT", "0"),
                    int(cp),
                    "on" if _v("DS41M_CP_K_GATHER", "1") not in ("0", "off", "false", "no")
                    else "off",
                    "full" if _v("DS41M_CP_QLI_FULL", "0") in on else "chunk",
                    _al))


def plan_hash(L: int, M: int, R: int, replay: bool, cp: int, world: int,
              backends: str) -> str:
    """点位几何 + 并行 + 后端 → 12 位指纹；任一输入变 ⇒ 变。"""
    import hashlib
    raw = f"L={int(L)};M={int(M)};R={int(R)};rep={int(bool(replay))};" \
          f"cp={int(cp)};world={int(world)};{backends}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def t_main(cfg, strict: bool = False) -> int:
    """本次前向实际算多少 token（命中：non_cached + keep；否则全长 L）。

    DeepInfer 口径：non_cached 由 `hit.non_cached_tokens` 直接给（无 128 对齐），
    或由 `hit.ratio` 换算（兼容 CLI，仍无块对齐）。

    `strict=True` 时非法命中几何直接抛错（见 `check_hit_geometry`）——
    **只在真正构造命中几何的入口用**（点位预检 / packed 命中 / CP 命中 / 真前缀），
    因为 `run.seq_len` 在单步 decode/verify 里是"本次行数"而不是序列长，
    拿它校验会误报（那些路径 T 由调用方直接给，命中口径不参与几何）。
    """
    run = cfg["run"]
    L = int(run["seq_len"])
    if strict:
        return check_hit_geometry(cfg, L, where="assemble.t_main")
    return (hit_m(cfg, L) + replay_tail_of(cfg)) if hit_is_on(cfg) else L


def ensure_packed(cfg):
    """varlen 是唯一 batch 口径（2026-10-26）：缺 `_packed` 时补默认单序列描述。

    供 dry-run / 工具（`check_module_spec`、`align_ops`）使用；真跑入口
    （points_runner/runner）自己会写真实的 n/lens/row_lens/cu。

    ★ 2026-11-01（C5）：**不再直写 `cfg["_packed"]`** —— 走 `metadata` 这个唯一入口
    （`attach` + `set_layout`），由它生成镜像。`tools/test_meta_single_source.py`
    会把"直写 `_packed`"当成违规拦住（这正是本次扫描抓出来的第二写入点）。
    """
    if not isinstance(cfg.get("_packed"), dict):
        import metadata as _md
        L = int((cfg.get("run") or {}).get("seq_len") or 0)
        T = int(t_main(cfg))
        _md.attach(cfg)
        _md.set_layout(cfg, n=1, cu=[0, T], seq_lens=[L], row_lens=[T])
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
    _ov = (cfg.get("_meta_override") or {})   # ★ CP 命中段用全局口径覆盖（见 runner 的 CP 分支）
    T_main = t_main(cfg)                    # **每序列**本段 token 数（命中：M+replay_tail）
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

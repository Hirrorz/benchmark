#!/usr/bin/env python3
"""命中几何的纯 CPU 回归（不碰 NPU、不加载权重）。

覆盖（铁律 0：非法几何必须**显式拒绝**，不许静默 clamp / 静默丢点）：
  · `R <= L`：h < replay_tail/L 的点必须 raise，且报错里带 L / M / R；
  · `M > 0`：h=1（或 non_cached_tokens>=L）必须 raise；
  · 合法点的几何与 rank 侧报表（L/M/R/H/S/actual_hit）自洽；
  · `plan_hash` 对几何 / 并行 / 后端敏感，重复调用稳定。

用法：python3 tools/test_hit_geometry.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_ds41_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    _q = _os.path.join(_ds41_root, _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)

import os                                                    # noqa: E402

import assemble                                              # noqa: E402

FAIL = []

LENS = (1, 127, 128, 1024, 8192)
HITS = (0, 0.01, 0.1, 0.5, 0.97, 0.99)


def chk(name, cond, detail=""):
    print(("PASS" if cond else "FAIL") + " · " + name + ("" if cond else f"  <<< {detail}"))
    if not cond:
        FAIL.append(name)


def _cfg(L, h=0.0, nc=0, keep=None, model=None):
    cfg = {"run": {"seq_len": int(L)},
           "hit": {"ratio": float(h), "non_cached_tokens": int(nc),
                   "source": "fake"},
           "model": model or {"sliding_window": 128}}
    if keep is not None:
        cfg["hit"]["replay_tail"] = int(keep)
    return cfg


def check_illegal_rejected():
    """所有 h/replay_tail 组合里 R>L 或 M==0 的，必须抛 ValueError 且信息可定位。

    `h=0` 且无 `non_cached_tokens` = **命中关闭**，几何即全长 L（不存在 R>L 约束），
    单独在 `check_metrics` 里验证 `t_main(L)==L`。
    """
    n_bad = n_ok = 0
    for L in LENS:
        for h in HITS:
            if h == 0:
                continue
            cfg = _cfg(L, h)
            keep = assemble.replay_tail_of(cfg)
            M = assemble.hit_m(cfg, L)
            R = M + keep
            legal = (M > 0 and R <= L)
            try:
                got = assemble.check_hit_geometry(cfg, L, where="test")
            except ValueError as e:
                msg = str(e)
                chk(f"拒绝 L={L} h={h}（M={M} R={R}）：报错含 L/M/R",
                    (not legal) and f"L={L}" in msg and f"M={M}" in msg
                    and (f"R={R}" in msg or "R=" in msg),
                    f"legal={legal} msg={msg[:80]}")
                n_bad += 1
                continue
            chk(f"放行 L={L} h={h}（M={M} R={R}）：几何合法且返回值 = M+keep",
                legal and got == R, f"legal={legal} got={got} want={R}")
            n_ok += 1
    print(f"  → 非法 {n_bad} 个 / 合法 {n_ok} 个")


def check_short_seq_low_hit():
    """文档里的实测反例：L=1024 h=0.1 → M=922, R=1062 > L（旧代码无校验，直接喂 R 行）。"""
    cfg = _cfg(1024, 0.1)
    M = assemble.hit_m(cfg, 1024)
    R = M + assemble.replay_tail_of(cfg)
    chk("L=1024 h=0.1 复现 R>L（M=922 / R=1062）", (M, R) == (922, 1062), f"M={M} R={R}")
    try:
        assemble.check_hit_geometry(cfg, 1024, where="反例")
    except ValueError as e:
        chk("该点必须被拒绝", "1062" in str(e) and "1024" in str(e), str(e)[:100])
    else:
        chk("该点必须被拒绝", False, "没有抛错")

    # 边界：命中关闭（h=0 且无 non_cached）= 全长，几何就是 L 本身。
    for L in LENS:
        chk(f"命中关闭 L={L}：几何 = 全长（不套 keep）",
            assemble.t_main(_cfg(L, 0.0)) == L
            and assemble.hit_is_on(_cfg(L, 0.0)) is False, L)


def check_full_hit_rejected():
    """h=1 会得到 M=0：语义是"纯 replay 段"，必须拒绝。

    注：`non_cached_tokens>=L` 被 `hit_m` 截到 L（`min(nc, L)`）⇒ M=L，
    落到"R>L"那条分支（不是 M=0）；两条都必须拒绝。
    """
    for L, h, nc in ((1024, 1.0, 0), (8192, 0.0, 8192), (8192, 0.99, 8192)):
        cfg = _cfg(L, h, nc=nc)
        M = assemble.hit_m(cfg, L)
        keep = assemble.replay_tail_of(cfg)
        R = M + keep
        legal = (M > 0 and R <= L)
        try:
            assemble.check_hit_geometry(cfg, L, where="全命中")
        except ValueError as e:
            msg = str(e)
            chk(f"非法点必须拒绝（L={L} h={h} nc={nc}；M={M} R={R}）",
                not legal and (f"M={M}" in msg and f"L={L}" in msg), f"msg={msg[:100]}")
        else:
            chk(f"非法点必须拒绝（L={L} h={h} nc={nc}；M={M} R={R}）",
                legal, f"M={M} R={R} 未抛错（legal={legal}）")


def check_metrics():
    """rank 侧报表口径：L/M/R/H/S/actual_hit 与命中几何自洽。"""
    for L, h in ((8192, 0.5), (65536, 0.97), (131072, 0.9)):
        cfg = _cfg(L, h)
        keep = assemble.replay_tail_of(cfg)
        M = assemble.hit_m(cfg, L)
        R = M + keep
        H = L - M
        S = max(0, L - R)
        actual = 1.0 - M / L
        chk(f"L={L} h={h}: H+M=L、S+R=L、actual_hit 与 M 自洽",
            H + M == L and S + R == L and abs(actual - (1 - M / L)) < 1e-12,
            f"H={H} M={M} R={R} S={S}")
        chk(f"L={L} h={h}: target_hit 与 actual_hit 只差取整",
            abs(h - actual) <= 1.0 / L + 1e-12,
            f"actual={actual:.8f} target={h:.8f}")


def check_matrix_gate():
    """历史点位矩阵（`finish_cp2cp4.sh` 的 LENS×HITS）必须一点不丢：
    新校验放行的点数与旧脚本 `continue` 之后剩下的点数**完全相同**，且 R 逐点相同。
    这是"收紧校验不改历史点位"的硬证据。"""
    lens = "1k 2k 4k 8k 12k 16k 20k 24k 32k 40k 48k 56k 64k 80k 96k 112k 128k 256k 512k 1m".split()
    hits = "0 0.1 0.6 0.7 0.8 0.9 0.95 0.96 0.97 0.98 0.99".split()
    mul = {"": 1, "k": 1024, "m": 1048576}

    def plen(s):
        import re
        m = re.fullmatch(r"(\d+(?:\.\d+)?)([km]?)", s.lower())
        return int(float(m.group(1)) * mul[m.group(2)])

    old, new, old_skip, new_skip = [], [], [], []
    for Ls in lens:
        L = plen(Ls)
        for h in hits:
            hf = float(h)
            # 旧脚本口径（finish_cp2cp4.sh:154-166）：直接 continue 掉 R>L 的点
            m_old = int(round((1 - hf) * L))
            tpass = L if hf <= 0 else m_old + 140
            old_legal = not (hf > 0 and tpass > L)
            cfg = _cfg(L, hf)
            try:
                R = assemble.check_hit_geometry(cfg, L, where="矩阵")
                new_legal = True
            except ValueError:
                R, new_legal = None, False
            (old if old_legal else old_skip).append((Ls, h, tpass))
            (new if new_legal else new_skip).append((Ls, h, R))
    chk(f"拒绝集合与旧脚本 continue 集合**完全一致**（{len(old_skip)} 点）",
        [x[:2] for x in old_skip] == [x[:2] for x in new_skip],
        f"旧丢 {[(a, b) for a, b in ((x[0], x[1]) for x in old_skip)]} / "
        f"新丢 {[(a, b) for a, b in ((x[0], x[1]) for x in new_skip)]}")
    chk(f"放行 {len(new)} 点与旧脚本 T_pass 逐点相同（旧共 {len(lens) * len(hits)} 点）",
        old == new, f"首个差异 {[(a, b) for a, b in zip(old, new) if a != b][:3]}")


def check_plan_hash():
    """指纹：几何/并行/后端任一变化都必须变；同样输入重复调用必须相同。"""
    bk = assemble.backends_of(4)
    h0 = assemble.plan_hash(65536, 1966, 2106, True, 4, 8, bk)
    chk("同输入重复调用稳定", h0 == assemble.plan_hash(65536, 1966, 2106, True, 4, 8, bk)
        and len(h0) == 12)
    chk("改 L → 变", h0 != assemble.plan_hash(65537, 1966, 2106, True, 4, 8, bk))
    chk("改 R → 变", h0 != assemble.plan_hash(65536, 1966, 2107, True, 4, 8, bk))
    chk("改 replay → 变", h0 != assemble.plan_hash(65536, 1966, 2106, False, 4, 8, bk))
    chk("改 cp → 变", h0 != assemble.plan_hash(65536, 1966, 2106, True, 2, 8, bk))
    chk("改 world → 变", h0 != assemble.plan_hash(65536, 1966, 2106, True, 4, 16, bk))

    saved = dict(os.environ)
    try:
        # 2026-11-02：全长 QLI/QSLI 已删，该 hash 敏感性用例移除
        os.environ["DS41M_IDX_KERNEL"] = "vendor"
        chk("改 DS41M_IDX_KERNEL → backends 含 vendor",
            "idx=vendor" in assemble.backends_of(4), assemble.backends_of(4))
    finally:
        os.environ.clear()
        os.environ.update(saved)
    chk("恢复 env 后 backends 与初始一致", assemble.backends_of(4) == bk, assemble.backends_of(4))


def check_backends_text():
    """backends 串必须把每个开关都写进去（结果表靠它判"哪个后端"。"""
    bk = assemble.backends_of(2)
    chk("backends 含 idx/qsli/moe/cp_cmp/cp_hit/cp/k_gather/qli 八项",
        all(k in bk for k in ("idx=", "qsli=", "moe=", "cp_cmp=", "cp_hit=",
                              "cp=", "k_gather=", "qli=")), bk)
    chk("backends 默认值 = dg/dg/auto/seg/0/chunk（与各模块 env 默认同口径）",
        "idx=dg" in bk and "qsli=dg" in bk and "moe=auto" in bk
        and "cp_cmp=seg" in bk and "cp=2" in bk and "qli=chunk" in bk, bk)


def check_block_alignment():
    """★ B2（GPT §12.2 hit_alignment）：`token` 默认逐位不变；`block[:BS]` 向上取整到块边界。

    只验几何算术 + 收口行为（不碰 NPU）：默认口径必须与历史一致，块对齐必须真的取整，
    且"取整后 R>L"必须被 `check_hit_geometry` **显式拒绝**（不许 clamp）。
    """
    # ① 默认（token）：历史值不能变。★ 显式写 cfg（优先级高于 env），
    #   这样本套件在 DS41M_HIT_ALIGN=block:128 的进程里跑也仍然验的是 token 口径。
    c_tok = _cfg(8192, 0.97)
    c_tok["hit"]["alignment"] = "token"
    chk("默认 token：M=round(0.03*8192)=246", assemble.hit_m(c_tok, 8192) == 246,
        str(assemble.hit_m(c_tok, 8192)))
    chk("默认 token：alignment=('token',1)", assemble.hit_alignment(c_tok) == ("token", 1),
        str(assemble.hit_alignment(c_tok)))

    # ② block:128 → 向上取整
    c = _cfg(1024, 0.53)
    c["hit"]["alignment"] = "block:128"
    chk("block:128：L=1024 h=0.53 → M 481→512", assemble.hit_m(c, 1024) == 512,
        str(assemble.hit_m(c, 1024)))
    c2 = _cfg(8192, 0.97)
    c2["hit"]["alignment"] = "block"
    chk("block（默认 BS=128）：L=8192 h=0.97 → 246→256",
        assemble.hit_m(c2, 8192) == 256 and assemble.hit_alignment(c2) == ("block", 128),
        f"{assemble.hit_m(c2, 8192)} {assemble.hit_alignment(c2)}")

    # ③ 块对齐把 R 顶过 L ⇒ 必须显式拒绝（M=1024、keep=140、R=1164 > 1024）
    c3 = _cfg(1024, 0.05)
    c3["hit"]["alignment"] = "block:128"
    M = assemble.hit_m(c3, 1024)
    try:
        assemble.check_hit_geometry(c3, 1024, where="block-test")
        chk("block 对齐到 M=L 时被拒绝", False, f"M={M} 竟然通过")
    except ValueError as e:
        chk("block 对齐把 R 顶过 L 时显式拒绝（含 L/M/R）",
            f"L=1024" in str(e) and f"M={M}" in str(e), str(e)[:90])

    # ④ 非法 alignment 必须报错（不静默回退）
    c4 = _cfg(1024, 0.5)
    c4["hit"]["alignment"] = "blk"
    try:
        assemble.hit_alignment(c4)
        chk("非法 alignment 报错", False)
    except ValueError as e:
        chk("非法 alignment 报错（不静默回退）", "alignment" in str(e), str(e)[:80])

    # ⑤ 对齐后 M 不同 ⇒ plan_hash 必须不同（否则结果表分不清口径）
    cA = _cfg(1024, 0.53)
    cB = _cfg(1024, 0.53)
    cB["hit"]["alignment"] = "block:128"
    hA = assemble.plan_hash(1024, assemble.hit_m(cA, 1024), 1, True, 1, 1, "")
    hB = assemble.plan_hash(1024, assemble.hit_m(cB, 1024), 1, True, 1, 1, "")
    chk("alignment 改变 ⇒ plan_hash 改变", hA != hB, f"{hA} vs {hB}")


def check_cpblock_alignment():
    """★ B4（GPT §7/§14 `partition=mirror_aligned`）：`cpblock[:BS]` 把 R 对齐到 `2·cp·BS`。

    这样 `cp_layout` 的 mirror 分块 `ps = R/(2·cp)` 就是 BS 的整数倍（每 rank 两段都落在
    块边界）。① cp=2 时 R 必须变成 512 的倍数；② cp=1 时退化成 `2·BS`；
    ③ `2·cp·BS > L` 时不可能对齐 ⇒ 必须被几何收口**显式拒绝**（不 clamp）。
    """
    import os as _os
    old = _os.environ.get("DS41M_CP_SIZE")

    def _with_cp(cp, fn):
        _os.environ["DS41M_CP_SIZE"] = str(cp)
        try:
            return fn()
        finally:
            if old is None:
                _os.environ.pop("DS41M_CP_SIZE", None)
            else:
                _os.environ["DS41M_CP_SIZE"] = old

    c = _cfg(8192, 0.5)
    c["hit"]["alignment"] = "cpblock:128"
    # cp=2: unit = 2*2*128 = 512；M=4096、keep=140 → R=4236 → R'=4608 → M'=4468
    M2 = _with_cp(2, lambda: assemble.hit_m(c, 8192))
    R2 = M2 + assemble.replay_tail_of(c)
    chk("cpblock cp=2：R 对齐到 512 的倍数（4468+140=4608）",
        M2 == 4468 and R2 % 512 == 0, f"M={M2} R={R2}")
    chk("cpblock cp=2：每 rank 两段各 1152 行 = 9×128（块对齐）",
        (R2 // (2 * 2)) % 128 == 0, f"ps={R2 // 4}")

    # cp=1: unit = 256；R=4236 → 4352 → M'=4212
    M1 = _with_cp(1, lambda: assemble.hit_m(c, 8192))
    R1 = M1 + assemble.replay_tail_of(c)
    chk("cpblock cp=1：R 对齐到 256 的倍数（4212+140=4352）",
        M1 == 4212 and R1 % 256 == 0, f"M={M1} R={R1}")

    # ③ cp=8、L=1024 ⇒ unit=2048 > L ⇒ 无论怎么取整都超长 ⇒ 必须拒绝
    c2 = _cfg(1024, 0.5)
    c2["hit"]["alignment"] = "cpblock:128"
    try:
        _with_cp(8, lambda: assemble.check_hit_geometry(c2, 1024, where="cpblock-test"))
        chk("cpblock 对齐不可能时被拒绝", False, "竟然通过")
    except ValueError as e:
        chk("cpblock 对齐不可能时显式拒绝（含 L/M/R）",
            "L=1024" in str(e) and "M=" in str(e), str(e)[:90])

    # ④ plan_hash 必须能区分 token / block / cpblock（用 h=0.53 让三者 M 各不相同：
    #    token 3850 / block 3968 / cpblock 3956）
    cts = []
    for _al in ("token", "block:128", "cpblock:128"):
        _c = _cfg(8192, 0.53)
        _c["hit"]["alignment"] = _al
        cts.append(_c)
    hs = {assemble.plan_hash(8192, _with_cp(2, lambda cc_=cc: assemble.hit_m(cc_, 8192)),
                             1, True, 2, 2, "")
          for cc in cts}
    chk("token/block/cpblock 三种口径的 plan_hash 互不相同", len(hs) == 3, str(hs))

    # ⑤ backends 串必须写出 cpblock（否则结果表分不清对齐口径）
    _old = _os.environ.get("DS41M_HIT_ALIGN")
    _os.environ["DS41M_HIT_ALIGN"] = "cpblock:128"
    try:
        bk = assemble.backends_of(2)
    finally:
        if _old is None:
            _os.environ.pop("DS41M_HIT_ALIGN", None)
        else:
            _os.environ["DS41M_HIT_ALIGN"] = _old
    chk("backends 串里出现 align=cpblock128", "align=cpblock128" in bk, bk)


def check_full_prefill():
    """★ 真权重入口的口径（P5 事故）——`M==L` 必须**拒绝**，正确写法是 `nc=0`。

    `tools/run_real.sh` 原来在 prefill 口径下写 `non_cached_tokens = round((1-0)*L) = L`，
    等于"命中开启 + 非缓存段 = 全长" ⇒ `R = L + replay_tail > L`，被几何收口拒绝
    （作业 P5 首跑 rc=1 的真因）。正确口径：prefill 用 **ratio=0 + nc=0**（命中关闭）。
    """
    # ① 错误口径（nc=L）必须被拒绝 —— 保持既有严格契约
    cfg = _cfg(16384, 0.0, nc=16384)
    try:
        assemble.check_hit_geometry(cfg, 16384, where="full-prefill")
        chk("nc==L 被拒绝（M=L, R=L+keep>L）", False, "竟然通过")
    except ValueError as e:
        chk("nc==L 被拒绝且信息含 L/M/R", "L=16384" in str(e) and "M=16384" in str(e), str(e)[:90])

    # ② 正确口径（ratio=0 + nc=0）：命中关闭 ⇒ 返回全长
    cfg2 = _cfg(16384, 0.0, nc=0)
    R = assemble.check_hit_geometry(cfg2, 16384, where="plain-prefill")
    chk("ratio=0 + nc=0（命中关闭）→ 返回全长 L", R == 16384, f"R={R}")
    chk("该口径 hit_is_on=False（几何走全长，hit_m 不参与判定）",
        not assemble.hit_is_on(cfg2), "hit_is_on=True")


def main():
    print("== 命中几何（M>0 且 R<=L）==")
    check_illegal_rejected()
    print("\n== 反例 / 边界 ==")
    check_short_seq_low_hit()
    print("\n== 全命中（M=0）==")
    check_full_hit_rejected()
    print("\n== 报表几何 ==")
    check_metrics()
    print("\n== 历史点位矩阵不回归 ==")
    check_matrix_gate()
    print("\n== 结果指纹 ==")
    check_plan_hash()
    check_backends_text()
    print("\n== B2 命中对齐（token / block）==")
    check_block_alignment()
    print("\n== B4 CP 块对齐（cpblock）==")
    check_cpblock_alignment()
    print("\n== 全量 prefill（M==L，真权重口径）==")
    check_full_prefill()
    print()
    if FAIL:
        print(f"✗ hit_geometry: {len(FAIL)} 项失败")
        for n in FAIL:
            print(f"   - {n}")
        return 1
    print("✓ hit_geometry: PASS")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
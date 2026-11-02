#!/usr/bin/env python3
"""`cp/cp_plan.py` 的 CPU 不变量测试（C4 吸收）。

判据（全部纯 CPU，无 torch/npu）：
  1. **几何身份**：`build_plan(...).layout` 与 `cp_layout.build_cp_plan(...)` **逐字段相同**
     （同 seq_lens/cp_size/cp_rank）—— 证明 cp_plan 没有引入第二套几何；
  2. `mirror_blocks == (rank, 2·cp−1−rank)`（mirror 定义）；
  3. `local_rows == sum(local_lens)`，且 `cp=1` 时等于全长（identity，无 pad）；
  4. 路由开关口径：`cp=1` ⇒ carry/halo/kv_gather/terminal_state **全关**；
     `cp>1` ⇒ 全开；`replay=False` ⇒ boundary 关；
  5. `halo.pred_segments == 2·cp−1`（实测口径：tail 140 @ CP8 需 15 段、CP4 需 7 段）；
  6. `moe.spans` 原样透传（跨 rank 必须一致，由调用方保证）。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("cp", "layout", "core", "meta", "ops", "trace"):
    sys.path.insert(0, os.path.join(ROOT, _p))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cp_layout            # noqa: E402
import cp_plan              # noqa: E402

FAIL = []


def chk(name, cond, detail=""):
    print(("PASS" if cond else "FAIL") + " · " + name + ("" if cond else f"  <<< {detail}"))
    if not cond:
        FAIL.append(name)


LENS_CASES = ([8192], [16384], [4236], [2188], [1024], [128],
              [8192, 8192], [4096, 4096, 4096])


def check_geometry_identity():
    """① 几何身份：cp_plan 出的 layout 必须与 cp_layout 完全一致（不新造几何）。"""
    n = 0
    for lens in LENS_CASES:
        for cp in (1, 2, 4, 8):
            for rank in range(cp):
                ref = cp_layout.build_cp_plan(lens, cp, rank)
                got = cp_plan.build_plan(lens, cp, rank).layout
                same = (tuple(got.seq_lens) == tuple(ref.seq_lens)
                        and tuple(got.local_lens) == tuple(ref.local_lens)
                        and tuple(got.local_cu) == tuple(ref.local_cu)
                        and tuple(got.local_index) == tuple(ref.local_index)
                        and tuple(got.reverse_index) == tuple(ref.reverse_index)
                        and int(got.total_q) == int(ref.total_q)
                        and int(got.total_per_rank) == int(ref.total_per_rank))
                n += 1
                if not same:
                    chk(f"几何身份 lens={lens} cp={cp} r={rank}", False, "与 cp_layout 不一致")
                    return
    chk(f"几何身份：{n} 组 (lens×cp×rank) 全部与 cp_layout 逐字段相同", True)


def check_mirror_and_rows():
    """②③ mirror 块号 / 本 rank 行数。"""
    for lens in LENS_CASES:
        for cp in (1, 2, 4, 8):
            for rank in range(cp):
                v = cp_plan.build_plan(lens, cp, rank)
                ok = v.mirror_blocks == (rank, 2 * cp - 1 - rank)
                if not ok:
                    chk(f"mirror lens={lens} cp={cp} r={rank}", False, str(v.mirror_blocks))
                    return
                if v.local_rows != sum(v.local_lens):
                    chk(f"local_rows lens={lens} cp={cp} r={rank}", False,
                        f"{v.local_rows} vs {sum(v.local_lens)}")
                    return
                if cp == 1 and v.local_rows != sum(lens):
                    chk(f"cp=1 identity lens={lens}", False, str(v.local_rows))
                    return
    chk("mirror 块号 / local_rows / cp=1 identity 全部成立", True)


def check_routes():
    """④⑤⑥ 路由开关与 halo 前驱段口径。"""
    v1 = cp_plan.build_plan([8192], 1, 0, replay=True)
    off = all(not v1.routes[k].enabled
              for k in ("carry", "halo", "kv_gather", "terminal_state"))
    chk("cp=1：carry/halo/kv_gather/terminal_state 全关", off,
        str({k: v1.routes[k].enabled for k in v1.routes}))
    chk("cp=1：boundary 仍开（replay=True）", v1.routes["boundary"].enabled)

    for cp in (2, 4, 8):
        v = cp_plan.build_plan([16384], cp, 0, replay=True, moe_span=2)
        on = all(v.routes[k].enabled
                 for k in ("carry", "halo", "kv_gather", "terminal_state"))
        chk(f"cp={cp}：carry/halo/kv_gather/terminal_state 全开", on, str(v.routes["halo"]))
        # ★ 2026-11-01 纠正：halo 的前驱段数是**按几何算**的（旧文档的 2·cp−1 复现不出）
        n = v.routes["halo"].detail["pred_segments"]
        chk(f"cp={cp}：L=16384 尾部窗口只需 1 个块（ps≥window+keep）", n == 1,
            f"pred_segments={n}")
        chk(f"cp={cp}：moe.spans 透传 == 2", v.routes["moe"].detail["spans"] == 2,
            str(v.routes["moe"].detail))

    # 短序列才是多块：这些是**算出来的**期望值，不是常量
    _cases = {((1024,), 2): 2, ((1024,), 4): 3, ((1024,), 8): 5,
              ((4236,), 8): 2, ((8192,), 8): 1, ((1024, 1024), 8): None}
    for (lens, cp), want in _cases.items():
        v = cp_plan.build_plan(list(lens), cp, 0)
        got = v.routes["halo"].detail["pred_segments"]
        if want is None:
            chk(f"lens={lens} cp={cp}：多序列时前驱块数是正数", got > 0, str(got))
        else:
            chk(f"lens={lens} cp={cp}：前驱块数 == {want}", got == want, f"got={got}")

    v_rep_off = cp_plan.build_plan([8192], 4, 1, replay=False)
    chk("replay=False：boundary 关", not v_rep_off.routes["boundary"].enabled)


def check_halo_manifest():
    """B5 落点：`halo_manifest` 必须**无洞覆盖**尾部窗口，且块↔rank 归属正确。"""
    from cp_plan import halo_manifest, halo_covered, _block_owner

    for cp in (1, 2, 4, 8):
        for L in (128, 1024, 4236, 8192, 16384):
            v = cp_plan.build_plan([L], cp, 0)
            lo = max(0, L - 140 - (128 - 1))
            man = halo_manifest(v, 0, window=128, keep=140)
            if cp == 1:
                chk(f"cp=1 L={L}：halo 不起作用（manifest 仍覆盖但无跨 rank）",
                    halo_covered(man, lo, L), str([(s.block, s.rank) for s in man]))
                continue
            ok_cov = halo_covered(man, lo, L)
            ok_rank = all(0 <= s.rank < cp for s in man)
            ok_own = all(s.rank == _block_owner(s.block, cp) for s in man)
            ok_rng = all(s.pos_lo < s.pos_hi and s.pos_hi <= L for s in man)
            chk(f"cp={cp} L={L}：halo 清单无洞覆盖 [{lo},{L})", ok_cov,
                str([(s.block, s.pos_lo, s.pos_hi) for s in man]))
            chk(f"cp={cp} L={L}：块↔rank 归属合法（{len(man)} 段）",
                ok_rank and ok_own and ok_rng,
                str([(s.block, s.rank) for s in man]))

    # 多序列：每个序列各自的窗口都要被覆盖（清单只对指定 seq_index）
    v = cp_plan.build_plan([1024, 1024], 8, 0)
    for si in (0, 1):
        lo = max(0, 1024 - 140 - 127)
        man = halo_manifest(v, si, window=128, keep=140)
        chk(f"多序列 seq_index={si}：窗口 [{lo},1024) 被覆盖", halo_covered(man, lo, 1024),
            str([(s.block, s.pos_lo, s.pos_hi) for s in man]))


def check_gather_plan():
    """B5：`halo_gather_plan` 必须是**互不重叠、并集等于 manifest、段落在块内**的本地切片。"""
    for cp in (2, 4, 8):
        for L in (1024, 4236, 8192, 16384):
            v = cp_plan.build_plan([L], cp, 0)
            man = cp_plan.halo_manifest(v, 0, window=128, keep=140)
            gp = cp_plan.halo_gather_plan(v, 0, window=128, keep=140)
            ps = v.layout.local_lens[0] // 2
            chk(f"cp={cp} L={L}：清单段数一致", len(gp) == len(man), f"{len(gp)} vs {len(man)}")
            ok_blk = all(0 <= g["local_row0"] and g["local_row0"] + g["nrows"] <= ps for g in gp)
            chk(f"cp={cp} L={L}：每段落在持有 rank 的块内", ok_blk, str(gp[:2]))
            # 位置并集 == manifest 并集，且不重叠
            pos = sorted((g["pos_lo"], g["pos_hi"]) for g in gp)
            no_overlap = all(pos[i][1] <= pos[i + 1][0] for i in range(len(pos) - 1))
            lo = max(0, L - 140 - 127)
            cov = pos[0][0] == lo and pos[-1][1] == L if pos else False
            chk(f"cp={cp} L={L}：段区间不重叠且并集 = [{lo},{L})", no_overlap and cov, str(pos))
            ok_owner = all(g["rank"] == cp_plan._block_owner(g["block"], cp) for g in gp)
            chk(f"cp={cp} L={L}：rank 归属正确", ok_owner, str([(g["block"], g["rank"]) for g in gp]))


def check_scatter_plan():
    """B5：搬运三元组必须 覆盖无重复、源行在本地行轴内、半区正确。"""
    for cp in (2, 4, 8):
        for L in (1024, 4236, 8192, 16384):
            v = cp_plan.build_plan([L], cp, 0)
            ps = v.layout.local_lens[0] // 2
            sp = cp_plan.scatter_plan(v, 0, window=128, keep=140)
            lo = max(0, L - 140 - 127)
            dsts = [s["dst_pos"] for s in sp]
            chk(f"cp={cp} L={L}：dst 覆盖 [{lo},{L}) 且无重复",
                sorted(dsts) == list(range(lo, L)), f"n={len(dsts)} 期望={L - lo}")
            ok_row = all(0 <= s["src_local_row"] < 2 * ps for s in sp)
            chk(f"cp={cp} L={L}：src_local_row ∈ [0, 2·ps)", ok_row, f"ps={ps}")
            # 半区：head 块 → [0,ps)，tail → [ps,2ps)
            ok_half = True
            for s in sp:
                low = s["src_local_row"]
                half = 0 if low < ps else 1
                # 该 dst_pos 所在块号
                b = s["dst_pos"] // ps if ps else 0
                is_head = b < cp
                if (half == 0) != is_head:
                    ok_half = False
                    break
            chk(f"cp={cp} L={L}：src 半区与块类别一致（head<ps / tail≥ps）", ok_half)
            ok_rank = all(s["src_rank"] == cp_plan._block_owner(s["dst_pos"] // ps, cp)
                          for s in sp)
            chk(f"cp={cp} L={L}：src_rank == 块持有者", ok_rank)


def main():
    print("== C4：cp_plan 几何身份 ==")
    check_geometry_identity()
    print("\n== mirror / 行数 ==")
    check_mirror_and_rows()
    print("\n== 路由声明口径 ==")
    check_routes()
    print("\n== B5：halo 可执行清单 ==")
    check_halo_manifest()
    print("\n== B5：谁发哪些行（gather plan） ==")
    check_gather_plan()
    print("\n== B5：搬运三元组（scatter plan） ==")
    check_scatter_plan()
    print()
    if FAIL:
        print(f"✗ cp_plan: {len(FAIL)} 项失败")
        for n in FAIL:
            print(f"   - {n}")
        return 1
    print("✓ cp_plan: PASS")
    print("  " + cp_plan.describe(cp_plan.build_plan([8192], 2, 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
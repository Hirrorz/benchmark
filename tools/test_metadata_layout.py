#!/usr/bin/env python3
"""`meta/metadata.py` 的 C5 不变量测试：**有作用域的布局覆盖**必须严格还原。

为什么要有它：compact/replay 会改 `InputMetadata.layout`（cu/row_lens/seq_lens/n）。
改完不还原 ⇒ 漏到下一个 pass（实测 b=2 第二轮 L0 就按 512/请求 摆行 → moe gate 吃 nan）。
本测试把四条语义钉死（纯 CPU，无 torch/npu）：

  1. `snapshot_layout` 存档后 `restore_layout` 能**逐字段**还原（含 `n`）；
  2. 布局没变时 `restore_layout` 返回 False（不产生无谓写入）；
  3. `stage_view(**kw)` 作用域内生效、退出**必定**还原；异常退出也还原；
  4. 嵌套 `stage_view` 逐层还原（内层退出回到外层视图，外层退出回到原始）。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("meta", "core", "layout", "ops", "cp", "trace"):
    sys.path.insert(0, os.path.join(ROOT, _p))

import metadata as md     # noqa: E402

FAIL = []


def chk(name, cond, detail=""):
    print(("PASS" if cond else "FAIL") + " · " + name + ("" if cond else f"  <<< {detail}"))
    if not cond:
        FAIL.append(name)


def _cfg(cu=(0, 10), row_lens=(10,), seq_lens=(10,), n=1):
    cfg = {"_packed": {"n": n, "cu": list(cu), "row_lens": list(row_lens),
                       "lens": list(seq_lens)}}
    md.ensure(cfg)
    return cfg


def _lay(cfg):
    l = md.layout_of(cfg)
    return (list(l.cu), list(l.row_lens), list(l.seq_lens), int(l.n))


def check_snapshot_restore():
    cfg = _cfg()
    base = _lay(cfg)
    snap = md.snapshot_layout(cfg)
    chk("snapshot 存档四个字段", set(snap) == set(md._SNAP_FIELDS), str(sorted(snap)))
    md.set_layout(cfg, cu=[0, 5, 10], row_lens=[5, 5], seq_lens=[10, 10], n=2)
    chk("set_layout 后布局确实变了", _lay(cfg) != base, str(_lay(cfg)))
    changed = md.restore_layout(cfg, snap)
    chk("restore 返回 True（真的改过）", changed is True, str(changed))
    chk("restore 后逐字段回到原值", _lay(cfg) == base, f"{_lay(cfg)} vs {base}")

    # ② 没变时不应写入
    chk("布局未变时 restore 返回 False", md.restore_layout(cfg, snap) is False)

    # 无 metadata 时 snapshot 返回 None
    chk("无 _meta_obj ⇒ snapshot=None", md.snapshot_layout({}) is None)


def check_stage_view():
    cfg = _cfg()
    base = _lay(cfg)
    with md.stage_view(cfg, cu=[0, 5], row_lens=[5], seq_lens=[10], n=1):
        inside = _lay(cfg)
        chk("stage_view 作用域内生效", inside[:2] == ([0, 5], [5]), str(inside))
    chk("stage_view 退出后还原", _lay(cfg) == base, f"{_lay(cfg)} vs {base}")

    # 异常退出也要还原
    try:
        with md.stage_view(cfg, cu=[0, 3], row_lens=[3]):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    chk("stage_view 异常退出也还原", _lay(cfg) == base, str(_lay(cfg)))

    # 嵌套：内层退出回外层视图，外层退出回原始
    with md.stage_view(cfg, cu=[0, 7], row_lens=[7], seq_lens=[10], n=1):
        outer = _lay(cfg)
        with md.stage_view(cfg, cu=[0, 2], row_lens=[2], seq_lens=[10], n=1):
            inner = _lay(cfg)
        chk("嵌套：内层生效", inner[:2] == ([0, 2], [2]), str(inner))
        chk("嵌套：内层退出回到外层", _lay(cfg) == outer, f"{_lay(cfg)} vs {outer}")
    chk("嵌套：外层退出回到原始", _lay(cfg) == base, f"{_lay(cfg)} vs {base}")


def check_push_pop():
    """C5 第二步：栈式 push/pop —— `pop_all_layouts` 必须能把本 pass 压的全部还原。"""
    cfg = _cfg()
    base = _lay(cfg)
    d0 = md.layout_depth(cfg)
    md.push_layout(cfg, cu=[0, 5], row_lens=[5])
    chk("push 后深度 +1", md.layout_depth(cfg) == d0 + 1, str(md.layout_depth(cfg)))
    chk("push 生效", _lay(cfg)[:2] == ([0, 5], [5]), str(_lay(cfg)))
    md.push_layout(cfg, cu=[0, 2], row_lens=[2])
    chk("二层 push", _lay(cfg)[:2] == ([0, 2], [2]) and md.layout_depth(cfg) == d0 + 2,
        f"{_lay(cfg)} depth={md.layout_depth(cfg)}")
    chk("pop 一层回到上一层视图", md.pop_layout(cfg) and _lay(cfg)[:2] == ([0, 5], [5]),
        str(_lay(cfg)))
    chk("pop_all 回到原值", md.pop_all_layouts(cfg) == 1 and _lay(cfg) == base,
        f"{_lay(cfg)} vs {base}")
    chk("空栈 pop 返回 False", md.pop_layout(cfg) is False)
    chk("空栈 pop_all 返回 0", md.pop_all_layouts(cfg) == 0)


def check_hit_policy():
    """C5 收尾：命中策略进 metadata（单源）+ `cfg["hit"]` 镜像同步。"""
    cfg = {"run": {"seq_len": 8192}, "hit": {"ratio": 0.5, "replay_tail": 140,
                                             "alignment": "block:128", "source": "fake"}}
    md.attach(cfg)
    hp = md.hit_of(cfg)
    chk("attach 后 hit_of 读到 ratio/replay_tail/alignment",
        (hp.ratio, hp.replay_tail, hp.alignment) == (0.5, 140, "block:128"),
        f"{hp.ratio},{hp.replay_tail},{hp.alignment}")
    chk("is_on：ratio>0 ⇒ True", hp.is_on() is True)

    # set_hit 必须同步 cfg["hit"] 镜像
    md.set_hit(cfg, ratio=0.97)
    chk("set_hit 改 metadata", abs(md.hit_of(cfg).ratio - 0.97) < 1e-12, str(md.hit_of(cfg).ratio))
    chk("set_hit 同步 cfg[\"hit\"] 镜像", abs(float(cfg["hit"]["ratio"]) - 0.97) < 1e-12,
        str(cfg["hit"]))

    # 非法字段必须报错（不静默）
    try:
        md.set_hit(cfg, nonsense=1)
        chk("set_hit 未知字段 ⇒ ValueError", False, "竟然通过")
    except ValueError as e:
        chk("set_hit 未知字段 ⇒ ValueError", "未知的 hit 字段" in str(e), str(e)[:60])

    # 最小 dict（无 _meta_obj）也要能读，且**不写回**
    mini = {"hit": {"ratio": 0.0, "non_cached_tokens": 0}}
    hp2 = md.hit_of(mini)
    chk("最小 dict：hit_of 可读且不改写 cfg",
        hp2.ratio == 0.0 and "_meta_obj" not in mini, str(sorted(mini)))

    # ★ C5 收尾：绕过 set_hit 直接改 cfg["hit"] 必须**显式报错**（不再静默刷新）
    cfg3 = {"run": {"seq_len": 1024}, "hit": {"ratio": 0.5, "replay_tail": 140}}
    md.attach(cfg3)
    cfg3["hit"]["ratio"] = 0.97            # 直写（模拟旧代码）
    try:
        md.hit_of(cfg3)
        chk("raw 写 cfg[\"hit\"] ⇒ 漂移报错", False, "竟然静默同步了")
    except RuntimeError as e:
        chk("raw 写 cfg[\"hit\"] ⇒ 漂移报错（含 hit-drift）", "hit-drift" in str(e), str(e)[:80])
    # 显式放行口仍然可用
    import os as _os
    _old = _os.environ.get("DS41M_HIT_DRIFT_OK")
    _os.environ["DS41M_HIT_DRIFT_OK"] = "1"
    try:
        import importlib as _il
        _il.reload(md)                     # 让 _HIT_DRIFT_OK 生效
        hp3 = md.hit_of({"run": {"seq_len": 1024},
                         "hit": {"ratio": 0.97, "replay_tail": 140},
                         "_meta_obj": md.InputMetadata(
                             hit=md.HitPolicy(ratio=0.5, replay_tail=140))})
        chk("DS41M_HIT_DRIFT_OK=1 ⇒ 放行并按 cfg 覆盖", abs(hp3.ratio - 0.97) < 1e-12,
            str(hp3.ratio))
    finally:
        if _old is None:
            _os.environ.pop("DS41M_HIT_DRIFT_OK", None)
        else:
            _os.environ["DS41M_HIT_DRIFT_OK"] = _old
        import importlib as _il2
        _il2.reload(md)


def main():
    print("== C5：snapshot/restore ==")
    check_snapshot_restore()
    print("\n== C5：stage_view 作用域 ==")
    check_stage_view()
    print("\n== C5：栈式 push/pop ==")
    check_push_pop()
    print("\n== C5 收尾：命中策略单源 ==")
    check_hit_policy()
    print()
    if FAIL:
        print(f"✗ metadata_layout: {len(FAIL)} 项失败")
        for n in FAIL:
            print(f"   - {n}")
        return 1
    print("✓ metadata_layout: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
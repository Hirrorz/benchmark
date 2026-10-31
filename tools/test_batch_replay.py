#!/usr/bin/env python3



"""batch.py / replay.py 的 CPU 单元测试（不碰 NPU）。

覆盖：
  · packed 局部位置 / 请求号 / 尾部切片 / 旋转块表 / 窗口索引；
  · replay 单序列与 packed 的裁尾（含 pos_off 与行账同步）；
  · ReplayPolicy 的 env / JSON 解析（含失败回退）。
用法：python3 tools/test_batch_replay.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_ds41_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    _q = _os.path.join(_ds41_root, _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)

import json
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import assemble                                           # noqa: E402
import batch                                              # noqa: E402
import replay                                             # noqa: E402

FAIL = []


def chk(name, cond, detail=""):
    print(("PASS" if cond else "FAIL") + " · " + name + ("" if cond else f"  <<< {detail}"))
    if not cond:
        FAIL.append(name)


def test_batch():
    ctx = {"n_seqs": 2, "cu_seqlens": [0, 4, 7], "seq_lens": [4, 3], "pos_offset": 0}
    loc, sid = batch.local_positions(ctx, 7, "cpu")
    chk("local_positions loc", loc.tolist() == [0, 1, 2, 3, 0, 1, 2], loc.tolist())
    chk("local_positions sid", sid.tolist() == [0, 0, 0, 0, 1, 1, 1], sid.tolist())
    ctx2 = dict(ctx, pos_offset=10)
    loc2, _ = batch.local_positions(ctx2, 7, "cpu")
    chk("local_positions +pos_offset", loc2.tolist() == [10, 11, 12, 13, 10, 11, 12], loc2.tolist())

    idx, new_cu = batch.tail_slice(ctx, 2, "cpu")
    chk("tail_slice index", idx.tolist() == [2, 3, 5, 6], idx.tolist())
    chk("tail_slice cu", new_cu == [0, 2, 4], new_cu)

    bt = batch.rot_block_table(3, 2, "cpu")
    chk("rot_block_table", bt.tolist() == [[3, 1, 2], [6, 4, 5]], bt.tolist())

    w_idx, w_len = batch.win_indices(torch.tensor([0, 1, 129]), 128)
    chk("win_indices shape", tuple(w_idx.shape) == (3, 1, 128) and tuple(w_len.shape) == (3, 1),
        (w_idx.shape, w_len.shape))
    chk("win_indices row0", w_idx[0, 0, 0].item() == 0 and (w_idx[0, 0, 1:] == -1).all().item())
    chk("win_indices len", w_len.flatten().tolist() == [1, 2, 128], w_len.flatten().tolist())

    chk("single → None", batch.local_positions({}, 5, "cpu") == (None, None))


def test_replay_single():
    x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    row = {"_cmp_sparse": torch.arange(10 * 3).reshape(10, 3),
           "_ffn_pre": torch.arange(10 * 2).reshape(10, 2),
           "_pool": torch.zeros(99)}          # 非行账、dim0 != T → 不动
    policy = replay.ReplayPolicy(on=True, w=3, cut=0)
    r = replay.apply(policy, x=x, input_ids=list(range(10)), shared=row,
                     layer_idx=0, pos_off=0, packed_ctx=None)
    chk("single applied", r.applied and r.x.shape[0] == 3, r.x.shape)
    chk("single pos_off", r.pos_off == 7, r.pos_off)
    chk("single x tail", r.x.tolist() == x[-3:].tolist())
    chk("single ids tail", r.input_ids == [7, 8, 9], r.input_ids)
    chk("single row-account", row["_cmp_sparse"].shape[0] == 3 and row["_ffn_pre"].shape[0] == 3,
        (row["_cmp_sparse"].shape, row["_ffn_pre"].shape))
    chk("single pool untouched", row["_pool"].shape[0] == 99)


def test_replay_packed():
    x = torch.arange(10 * 2, dtype=torch.float32).reshape(10, 2)
    row = {"_cmp_sparse": torch.arange(10).reshape(10, 1)}
    pctx = {"n": 2, "lens": [5, 5], "cu": [0, 5, 10]}
    policy = replay.ReplayPolicy(on=True, w=2, cut=0)
    r = replay.apply(policy, x=x, input_ids=list(range(10)), shared=row,
                     layer_idx=0, pos_off=0, packed_ctx=pctx)
    chk("packed applied", r.applied and r.x.shape[0] == 4, r.x.shape)
    chk("packed index", r.x.tolist() == x[[3, 4, 8, 9]].tolist())
    chk("packed cu", pctx["cu"] == [0, 2, 4], pctx["cu"])
    chk("packed pos_off", r.pos_off == 3, r.pos_off)
    chk("packed ids", r.input_ids == [3, 4, 8, 9], r.input_ids)
    chk("packed row-account", row["_cmp_sparse"].shape[0] == 4, row["_cmp_sparse"].shape)


def test_policy():
    old = os.environ.get("DS41M_DECODER_REPLAY")
    try:
        os.environ["DS41M_DECODER_REPLAY"] = "0"
        p = replay.ReplayPolicy.from_env({}, 21, 128)
        chk("policy off", (not p.on) and p.cut == 21 and p.w == 128, p)
        os.environ["DS41M_DECODER_REPLAY"] = "1"
        p = replay.ReplayPolicy.from_env({}, 21, 128)
        chk("policy on", p.on and p.w == 128 and p.cut == 21, p)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"on": True, "w": 1024, "cut": 7, "force": True}, fh)
            path = fh.name
        os.environ["DS41M_DECODER_REPLAY"] = path
        p = replay.ReplayPolicy.from_env({}, 21, 128)
        chk("policy json", p.on and p.w == 1024 and p.cut == 7 and p.force, p)
        os.unlink(path)
    finally:
        if old is None:
            os.environ.pop("DS41M_DECODER_REPLAY", None)
        else:
            os.environ["DS41M_DECODER_REPLAY"] = old


def test_keep_tokens():
    """DeepInfer 固定口径：KEEP=140、swa_block=window+6，均不随草稿长度变化。"""
    cfg = {"model": {"sliding_window": 128}, "hit": {"replay_tail": 0}}
    old = {k: os.environ.get(k) for k in (
        "DS41M_DSPARK", "DS41M_REPLAY_KEEP")}
    try:
        for k in old:
            os.environ.pop(k, None)
        chk("swa block fixed = 134", assemble.swa_block_tokens(cfg) == 134,
            assemble.swa_block_tokens(cfg))
        chk("keep fixed = V4.1 140", assemble.replay_keep_tokens(cfg) == 140,
            assemble.replay_keep_tokens(cfg))
        chk("tail default = keep", assemble.replay_tail_of(cfg) == 140,
            assemble.replay_tail_of(cfg))
        os.environ["DS41M_DSPARK"] = "8"
        chk("dspark does not change keep", assemble.replay_keep_tokens(cfg) == 140,
            assemble.replay_keep_tokens(cfg))
        chk("dspark does not change swa block", assemble.swa_block_tokens(cfg) == 134,
            assemble.swa_block_tokens(cfg))
        os.environ["DS41M_REPLAY_KEEP"] = "200"
        chk("keep explicit override", assemble.replay_keep_tokens(cfg) == 200,
            assemble.replay_keep_tokens(cfg))
        # DeepInfer 命中口径：non_cached 直接使用；h 仅换算，不做 128 对齐
        chk("hit_m non_cached direct",
            assemble.hit_m({"hit": {"non_cached_tokens": 1000}}, 16384) == 1000,
            assemble.hit_m({"hit": {"non_cached_tokens": 1000}}, 16384))
        chk("hit_m from ratio no align",
            assemble.hit_m({"hit": {"ratio": 0.93}}, 1000) == 70,
            assemble.hit_m({"hit": {"ratio": 0.93}}, 1000))
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    test_batch()
    test_replay_single()
    test_replay_packed()
    test_policy()
    test_keep_tokens()
    print(f"\n{'PASS' if not FAIL else 'FAIL'} · {len(FAIL)} 项失败")
    sys.exit(0 if not FAIL else 1)

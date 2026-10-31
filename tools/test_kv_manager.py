#!/usr/bin/env python3
import os as _os, sys as _sys
_ds41_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    _q = _os.path.join(_ds41_root, _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)



"""kv_manager 回归：几何/槽位/块表必须与 exec_ops/exec_indexer 的 inline 实现逐位一致。

跑法（节点上）：`source env_local.sh && python3.11 tools/test_kv_manager.py`
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kv_manager as kv  # noqa: E402


def _eq(a, b, what):
    assert torch.equal(a, b), f"{what} 不一致：{(a != b).sum().item()} 个元素不同"


def _cmp_inline(geo: kv.KVGeometry):
    from exec_ops import _win_indices_from_pos, _win_indices_build
    from exec_indexer import _qslitab, _qsli_table

    dev = torch.device("cpu")
    for T in (1, 7, 128, 129, 1000, 8192):
        pos = torch.arange(T, dtype=torch.long, device=dev)
        i1, l1 = geo.win_indices(pos)
        i2, l2 = _win_indices_from_pos(pos, 128)
        _eq(i1, i2, f"win_indices idx T={T}")
        _eq(l1, l2, f"win_indices len T={T}")
        # 全局位置不连续（CP head-tail）
        pos2 = torch.cat([pos[: T // 2], pos[T // 2:] + 5000])
        _eq(geo.win_indices(pos2)[0], _win_indices_from_pos(pos2, 128)[0], f"win_indices cp idx T={T}")
        _eq(geo.win_indices(pos2)[1], _win_indices_from_pos(pos2, 128)[1], f"win_indices cp len T={T}")
        # off=0 的单序列路径
        _eq(geo.win_indices(pos)[0], _win_indices_build(T, dev, 128, 0, False)[0], f"win build idx T={T}")
        _eq(geo.win_indices(pos)[1], _win_indices_build(T, dev, 128, 0, False)[1], f"win build len T={T}")

    for n in (2, 3, 5, 65, 129, 130):
        _eq(geo.idx_block_table(n, dev), _qslitab(n, dev), f"idx_block_table n={n}")
        _eq(geo.idx_block_table(n, dev), _qsli_table(n, dev), f"qsli_table n={n}")


def _cmp_slots(geo: kv.KVGeometry):
    dev = torch.device("cpu")
    BS = geo.block_size
    # full 池槽位（exec_ops._kv_epilog）
    for T in (1, 128, 129, 1001):
        pos = torch.arange(T, dtype=torch.int32, device=dev)
        inline = (1 + pos // BS) * BS + pos % BS
        _eq(geo.full_slots(pos), inline, f"full_slots T={T}")
    # 窗口池槽位（exec_ops._kv_epilog A）
    for wa, off in ((128, 0), (140, 0), (140, 15744), (384, 1000)):
        inline = (torch.arange(wa, dtype=torch.int32, device=dev) + off) % kv.WIN_SLOTS
        _eq(geo.win_slots_for(wa, off, dev), inline, f"win_slots wa={wa} off={off}")
    # 索引 K 槽位（exec_indexer._idx_k_cache _ci_build）
    for n_rows, row0, nblk in ((8193, 0, 65), (1001, 0, 9), (513, 0, 5), (128, 0, 2)):
        q = torch.arange(n_rows, dtype=torch.int64, device=dev) + row0
        tb = _qslitab_ref(nblk + 1, dev)
        inline = (tb[q // geo.rows_per_block].long() * geo.rows_per_block
                  + q % geo.rows_per_block)
        _eq(geo.idx_slots(n_rows, row0, nblk, dev), inline.to(torch.int32),
            f"idx_slots n={n_rows} row0={row0} nblk={nblk}")
    # 池块数
    assert geo.cmp_pool_nblk(16384, 2, 8193) == 65, geo.cmp_pool_nblk(16384, 2, 8193)
    assert geo.cmp_pool_nblk(1001, 2, 501) == 4, geo.cmp_pool_nblk(1001, 2, 501)
    assert geo.idx_pool_nblk(8193, 0, 16384, 2) == 65
    assert geo.idx_pool_nblk(16384, 0, 16384, 1) == 129
    assert geo.full_pool_rows(8) == 9 * 128


def _qslitab_ref(n, device):
    return torch.cat([torch.tensor([n - 1], dtype=torch.int32, device=device),
                      torch.arange(1, n - 1, dtype=torch.int32, device=device)])


def _layout(cfg_path):
    import yaml
    cfg = yaml.safe_load(open(cfg_path))
    lay = kv.KVLayout.from_cfg(cfg)
    roles = {p.layer: (p.has_cmp, p.has_idx, p.has_candidate, p.state_ring) for p in lay.layers}
    for i in range(40):
        r = roles[i]
        if i in (2, 8, 14):
            assert r[:2] == (True, True), (i, r)
        elif i == 20:
            assert r[:3] == (True, True, True), (i, r)
        elif i in (24, 28, 32, 36):
            assert r[:2] == (False, True), (i, r)
        else:
            assert r[:2] == (False, False), (i, r)
        assert (r[3] > 0) == (i in (2, 8, 14)), (i, r)
    print("layout: PASS")
    print(lay.print_table())
    return lay


def _block_manager():
    bm = kv.BlockManager(num_blocks=8, block_size=128, first_block=1)
    h = bm.allocate(1000)          # 8 块
    assert bm.block_table(h) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert bm.num_free_blocks() == 0
    try:
        bm.allocate(1)
    except MemoryError:
        pass
    else:
        raise AssertionError("应当 OOM")
    bm.free(h)
    assert bm.num_free_blocks() == 8
    h2 = bm.allocate(129)
    assert bm.block_table(h2) == [1, 2]
    print("block_manager: PASS")


def _manager_buffers():
    """池形状/维数自证（防止 alloc 成 2D 导致 K 缓存整段错——2026-10-23 踩过）。"""
    cfg = {"model": {"compress_ratios": [0] * 40, "kv_source_layers": [],
                     "index_source_layers": [], "candidate_source_layer": 20}}
    mgr = kv.KVCacheManager(cfg)
    dev = torch.device("cpu")
    hold = {}
    k, s = mgr.idx_k(hold, 2, 65, dev)
    assert tuple(k.shape) == (66, 128, 1, 64), k.shape
    assert tuple(s.shape) == (66, 128, 1, 2, 2), s.shape
    kc = mgr.qsli(hold, 1, 127, 16, dev)
    assert tuple(kc.shape) == (127, 16, 1, 544), kc.shape
    assert hold["_idx_qslis_r1"] == (16, 127)
    stc = mgr.cmp_state(hold, 2, 2, 512, dev)
    assert tuple(stc.shape) == (2, 2, 2, 512), stc.shape
    p = mgr.get_pool(hold, "_win_pool", 384, 544, dev, torch.float8_e4m3fn)
    assert tuple(p.shape) == (384, 544)
    assert mgr.get_pool(hold, "_win_pool", 100, 544, dev, torch.float8_e4m3fn) is p
    print("manager buffers: PASS")


def main():
    geo = kv.KVGeometry()
    _cmp_inline(geo)
    _cmp_slots(geo)
    print("geometry/slots: PASS")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _layout(os.path.join(here, "configs", "mini_ds41_flash.yaml"))
    _block_manager()
    _manager_buffers()
    print("kv_manager: PASS")


if __name__ == "__main__":
    main()

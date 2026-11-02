"""`--points`：一次进程、一次预热，连续跑多个点位（hit 矩阵 / 扫描）。

用户口径（2026-10-20）："新增一次预热跑多点位的功能" —— 不同点位**不需要重复重启**
（常驻启动 5.5 min，其中权重打包/首跑占大头；每个点位重启一遍纯浪费）。

每个点位做与单点 `runner.py` 完全相同的事：
  · 重建层栈（L / hit / batches / replay 都会改几何）
  · 重造输入（每个 batch 独立 seed；packed 时拼接）
  · `--warmup-iters` 轮预热（不进 trace）
  · `--iters` 轮采集（trace 按点位落到 `<out>/L<L>/h<hhh>_b<N>_r<0|1>/prefill`）

点与点之间必须清掉**一切依赖几何（T/L/cut/池地址）的进程级缓存**：
`_CTX_CACHE`（ctx 里的 `T`/`win_abs` 由点位决定）、`exec_ops` 的 META（常量元数据）。
漏清 = 新点位复用旧 T 的池 → 数值错、越界甚至设备侧异常。

本模块**不碰**权重打包缓存（`_PACK_CACHE`/`_PACKED_LAYER_CACHE`）——与几何无关，
跨点位复用正是"不重复预热"的收益来源。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ds41_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    _q = _os.path.join(_ds41_root, _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)


import gc
import json
import os
import shutil
import statistics
import sys
import threading
import time


# ───────────────── 结果指纹（2026-11-01）─────────────────
# 换树 / 换开关后 `results.tsv` 里的数会混在一起，事后无法分辨是哪个口径算的。
# 指纹算法在 `assemble`（纯 CPU 可测，见 tools/test_hit_geometry.py）；这里只落盘（缺陷 G6）。
_RESULT_COLS = ("L", "M", "R", "H", "S", "keep", "actual_hit", "plan_hash", "backends",
                # ★ 2026-11-01（B1 吸收 / P3-1）：三段命名 + 三段行数（只增列，老表照读）
                "prefix_hit", "enc_replay", "dec_keep", "rows_q0_19", "rows_q21_39",
                # ★ 2026-11-01（P3-2 吸收）：两级 metadata 复用/重建计数（geo=进程级几何缓存，
                #   fwd=forward 级整表重建）。与 [meta-reuse] 打印同源。
                "meta_geo_hit", "meta_geo_miss", "meta_fwd_hit", "meta_fwd_miss")


class _SkipEmptyOrigin(Exception):
    """A5：空 origin（1 行虚拟请求）不写 results.tsv —— 内部信号，由写表处的
    `except Exception: pass` 吞掉（不改变既有行为，也不静默丢真实点位）。"""


# ───────────────── 点位级失败隔离（2026-10-29）─────────────────
def _load_done_tags(root: str) -> set:
    """`results.tsv` 里已完成的点位 tag（续跑跳过用）。`DS41M_POINT_REDO=1` 强制重采。"""
    if os.environ.get("DS41M_POINT_REDO", "").strip().lower() in ("1", "on", "true", "yes"):
        return set()
    out = set()
    try:
        with open(os.path.join(root, "results.tsv"), encoding="utf-8") as fh:
            for ln in fh:
                t = ln.split("\t", 1)[0].strip()
                if t and t != "tag":
                    out.add(t)
    except OSError:
        pass
    return out


def _load_fail_counts(root: str) -> dict:
    """`points_failed.tsv`：`tag -> 失败次数`（跨进程累计，防"坏点永远堵在前面"）。

    没有它的话：某个点位**确定性失败**（几何非法/必 OOM）→ 熔断退出 → 重跑又卡在同一点，
    后面的点永远轮不到。有了它，失败到 `DS41M_POINT_MAX_ATTEMPTS`（默认 3）次后跳过。
    """
    out = {}
    try:
        with open(os.path.join(root, "points_failed.tsv"), encoding="utf-8") as fh:
            for ln in fh:
                parts = ln.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0]:
                    out[parts[0]] = int(parts[1] or 0)
    except (OSError, ValueError):
        pass
    return out


def _note_fail(root: str, tag: str, rank: int) -> None:
    """记一次失败（**只 rank0 写**：多个 rank 同时写会互相覆盖）。"""
    if int(rank) != 0:
        return
    p = os.path.join(root, "points_failed.tsv")
    rows = _load_fail_counts(root)
    rows[tag] = int(rows.get(tag, 0)) + 1
    tmp = p + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            for t, c in sorted(rows.items()):
                fh.write(f"{t}\t{c}\n")
        os.replace(tmp, p)
    except OSError:
        pass


def _watchdog(state: dict, limit_s: float, rank: int) -> None:
    """单点超时 → 强制退进程。

    设备侧卡死（HCCL barrier 超时、aicore fault 后等对端）**无法**用 try/except 打断，
    只能退出；驱动脚本重跑同一条命令时，`results.tsv` 会把已完成的点位跳过。
    """
    while True:
        time.sleep(5.0)
        t0 = state.get("t0")
        if t0 is None:
            continue
        dt = time.time() - t0
        if dt <= limit_s:
            continue
        try:
            if rank == 0:
                print(f"✗ [watchdog] 点位 {state.get('pi')} 已跑 {dt:.0f}s > {limit_s:.0f}s "
                      f"→ 强制退出进程（重跑同命令会自动跳过已完成点位）", flush=True)
                sys.stdout.flush()
        except Exception:                    # noqa: BLE001
            pass
        os._exit(9)


def _prof_here(a, rank):
    pr = (getattr(a, "prof_ranks", None) or "all").strip().lower()
    return pr in ("", "all") or rank in {int(x) for x in pr.split(",") if x.strip()}


def _report_rank(a, world):
    """稳态数字/结果落盘的 rank（2026-10-27 用户规则）。

    prof 轮里被采集的 rank 会被 msprof 拖慢（动辄 10–20×），不能当性能数；
    取**第一个未被采集**的 rank（默认 prof 只采 rank0 ⇒ 报 rank1）。
    `--prof-ranks all` 时退回 0（全采口径，仅历史对照）。
    """
    pr = (getattr(a, "prof_ranks", None) or "all").strip().lower()
    if pr in ("", "all"):
        return 0
    prof = {int(x) for x in pr.split(",") if x.strip()}
    for r in range(int(world)):
        if r not in prof:
            return r
    return 0


def _fmt_L(L: int) -> str:
    """序列长的标准写法：**具体数字**（`262144` 就写 `262144`，不用 256k/1m）。"""
    return str(int(L))


def point_tag(L, h, nb, rep):
    """点位输出子目录名（驱动脚本按同一规则核对 trace）。"""
    return f"L{int(L)}/h{int(round(float(h) * 1000)):03d}_b{int(nb)}_r{1 if rep else 0}"


def _reset_between_points(R, measure=False):
    """清掉所有随几何变化的进程级缓存。

    ★ 2026-10-29：**统一登记表**（`core/memlife.py`）取代以前的"手写三行"。
      手写清表漏一处 = 一个点位后开始累积（099 实测 point 192 仍钉着 65 GiB 活张量，
      OOM 现场 `allocated == active`，不是碎片）。各模块把自己的几何缓存写进
      `POINT_CACHES` / `POINT_RELEASE()`，这里一次扫完。
    R 必须是**正在运行的** runner 模块（脚本入口下是 `__main__`，不在 `sys.modules`
    的包名集合里 ⇒ memlife 扫不到它，必须显式清）。
    """
    R._CTX_CACHE.clear()
    R._SEQ_CACHE.clear()          # 层内算子序列（按层号缓存；与点位几何解耦，但清了最稳）
    try:
        R._MODULE_PLAN_CACHE.clear()          # Phase 2 模块计划（按层号；同理由几何驱动）
    except Exception:                    # noqa: BLE001
        pass
    try:
        R.POINT_RELEASE()            # CP plan LRU + gather 固定缓冲（按 L 分配，天量级）
    except Exception:                    # noqa: BLE001
        pass
    try:
        import memlife as _memlife
        _memlife.sweep(measure=measure)   # 其余模块的登记表（exec_ops/exec_indexer/mq_outbuf/…）
        _gb = _memlife.unknown_guard_gb()  # 兜底：未登记但长个儿的容器（默认 2 GiB）
        if _gb > 0:
            _memlife.sweep_unknown(_gb)
    except Exception as _e:              # noqa: BLE001
        print(f"  [memlife] ⚠️ sweep 失败：{type(_e).__name__}: {_e}", flush=True)


def _free_device():
    # DeepInfer 式阶段边界：点与点之间 gc + sync + empty_cache（**不在 forward 循环里**）
    try:
        gc.collect()
    except Exception:                    # noqa: BLE001
        pass
    try:
        import torch
        torch.npu.synchronize()
        torch.npu.empty_cache()
    except Exception:                    # noqa: BLE001
        pass


def run_points(cfg, source, a, dev, runner_mod=None):
    """执行 `a.points` 指定的全部点位；返回退出码（0=全成功）。

    `runner_mod` 必须是**正在运行的** runner 模块（runner.main 传 `sys.modules[__name__]`）。
    不能在这里 `import runner`：runner.py 以脚本运行时 `__main__` 与 `runner` 是两份模块，
    后者的 `_RANK`=0 / 缓存对象也是另一套（实测 8 rank 全开 profiler + 点 2 崩）。
    """
    import torch
    import assemble
    if runner_mod is None:                    # 仅交互式调试兜底（脚本路径必须传）
        import runner as runner_mod
    R = runner_mod
    # DeepInfer 式 GC：worker 里 gc.disable，靠阶段边界（每点位 _free_device）collect。
    if os.environ.get("DS41M_GC_OFF", "1").strip().lower() not in (
            "0", "off", "false", "no"):
        gc.disable()

    pts = json.load(open(a.points))
    if not isinstance(pts, list) or not pts:
        raise SystemExit(f"--points 文件为空或格式不对（要 list）：{a.points}")
    if a.prompt or a.ids_file or a.decode or a.layers or a.collect_layers:
        raise SystemExit("--points 只支持合成输入路径（不能与 --prompt/--ids-file/--decode/"
                         "--layers/--collect-layers 同用）")
    rank = int(getattr(R, "_RANK", 0))
    prof_here = _prof_here(a, rank)
    # ★ 2026-10-22 用户规则：每个点位至少 4 个 pass（多 pass 取平均/中位）
    iters = max(4, int(a.iters or cfg["run"].get("iters", 1) or 1))
    warm = max(0, int(a.warmup_iters or 0))
    base_hit = dict(cfg.get("hit") or {})
    seed0 = int(getattr(a, "input_seed", 0) or 0)
    root = os.path.abspath(a.out or os.path.join(os.path.dirname(os.path.abspath(R.__file__)), "prof"))
    print(f"== --points：{len(pts)} 个点位 · 每点 {warm} 预热 + {iters} 采集轮 · "
          f"rank={rank} prof={'on' if (a.prof and prof_here) else 'off'} · root={root}", flush=True)

    # ★ 2026-10-29：点位边界的显存审计（`DS41M_MEM_AUDIT=1`）。留存超 `DS41M_MEM_GUARD_GB`
    #   时自动按「模块容器 / 张量形状」归因 —— 有人新加了没登记的几何缓存，日志直接点名。
    import memlife as _memlife
    _audit = _memlife.audit_on()
    _audit_every = _memlife.audit_every()
    _base_alloc = None                  # 各点位边界 allocated 的低水位（= 无泄漏时的基线）
    _prev_alloc = None

    # ★ 2026-10-29：点位级**失败隔离**（用户口径：中间一个点出问题不能整轮挂死）。
    #   · 续跑：`results.tsv` 里已有的点位跳过（进程重启后只补没跑完的）；
    #   · 单点异常：只标该点失败，清显存后继续下一点（跨 rank 对齐靠点位开头的 barrier）；
    #   · 卡死：watchdog 线程，单点超 `DS41M_POINT_TIMEOUT`（默认 1800s，0=关）直接退出，
    #     由驱动脚本重跑同命令（续跑会跳过已完成的点）；
    #   · 连续失败 `DS41M_POINT_MAX_FAILS`（默认 2）次 → 退出，别在坏掉的设备上文里空烧。
    _done_tags = _load_done_tags(root)
    _fail_counts = _load_fail_counts(root)
    _max_attempts = max(1, int(os.environ.get("DS41M_POINT_MAX_ATTEMPTS", "3") or 3))
    if _done_tags:
        print(f"   续跑：results.tsv 已有 {len(_done_tags)} 个点位 → 跳过"
              f"（DS41M_POINT_REDO=1 可强制重采）", flush=True)
    if _fail_counts:
        _dead = sorted(t for t, c in _fail_counts.items() if c >= _max_attempts)
        print(f"   失败记录：{len(_fail_counts)} 个点位失败过，其中 {len(_dead)} 个已达"
              f" {_max_attempts} 次上限 → 本轮跳过（DS41M_POINT_MAX_ATTEMPTS 可调）",
              flush=True)
    _pt_timeout = float(os.environ.get("DS41M_POINT_TIMEOUT", "1800") or 0)
    _max_fails = max(1, int(os.environ.get("DS41M_POINT_MAX_FAILS", "2") or 2))
    _wd = {"t0": None, "pi": 0}
    if _pt_timeout > 0:
        threading.Thread(target=_watchdog, args=(_wd, _pt_timeout, rank),
                         daemon=True, name="point-watchdog").start()
    _fail_rows = []
    n_fail = 0                          # 跳过的点位数（异常隔离）
    n_consec = 0                        # 连续失败（熔断用）

    all_rows = []
    n_bad = 0
    def _as_ratio(v):
        """hit 兼容 0.97 / 97 / "97%" → [0,1) 小数。"""
        if isinstance(v, str):
            v = v.strip().rstrip("%")
        fv = float(v or 0)
        return fv / 100.0 if fv > 1 else fv

    for pi, pt in enumerate(pts, 1):
        try:
            tag = None                    # 点位产物名；解析前抛异常时保持 None
            _wd["pi"] = pi
            _wd["t0"] = time.time()          # watchdog 从进点开始计时（含预热/采集/finalize）
            # ★ 跨 rank 对齐必须在点位体**最前面**：某个 rank 若在建输入/预热阶段就抛
            #   （OOM 最常见），它也必须先调过这次 barrier —— 否则各 rank 的 barrier
            #   次数错一拍，后面每次集合通信都错位 → 整轮挂死。
            if pi > 1:
                R.pass_barrier_sync(pi)
            # ★ 2026-10-21 点位/批次 v2：`batches` 可以是
            #   · 列表 [{L, hit}, …] → 每批自己的 L/命中率（同一次 packed pass 内打包）
            #   · 整数 N（旧格式）→ N 批同 L/同 hit
            #   点位只保留一个可选的 `tag`（分组名），L 全在批次上。
            # ★ 2026-10-21 per-DP：`{"tag":..., "dps":[{rank, batches:[{L,hit}]}, …]}`
            #   每个 rank 只取自己那条；同时用全部条目做 **span 一致性** 前置校验
            #   （MegaMoE 同步 → 各 rank 的集体通信轮数必须相同；小 L/少批时天然 1 span）。
            _bs = pt.get("batches")
            _dps = pt.get("dps")
            per_dp = isinstance(_dps, list) and bool(_dps)
            _all_dps = []
            if per_dp:
                def _specs_of(entry, who):
                    out = []
                    for bi, b in enumerate(entry.get("batches") or [], 1):
                        _L = int(b.get("L", 0) or 0)
                        _h = _as_ratio(b.get("hit", 0) or 0)
                        if _L <= 0:
                            raise SystemExit(f"✗ 点位 {pi} {who} 第 {bi} 批缺 L")
                        if not (0.0 <= _h < 1.0):
                            raise SystemExit(f"✗ 点位 {pi} {who} 第 {bi} 批 hit ∈ [0,1)：{b.get('hit')!r}")
                        out.append({"L": _L, "hit": _h})
                    if not out:
                        # ★ A5：`DS41M_ALLOW_EMPTY_DP=1` 允许该 DP 无 token（T=0 空 origin）
                        if os.environ.get("DS41M_ALLOW_EMPTY_DP", "0").strip().lower() in (
                                "1", "on", "true", "yes"):
                            print(f"  [empty-dp] 点位 {pi} {who}：0 token（A5 空 origin 测试）",
                                  flush=True)
                            return []
                        raise SystemExit(f"✗ 点位 {pi} {who} 的 batches 为空")
                    return out
                _all_dps = [(int(d.get("rank", i)), _specs_of(d, f"dp{i}"))
                            for i, d in enumerate(_dps)]
                _mine = [sp for _r, sp in _all_dps if _r == int(rank)]
                if not _mine:
                    raise SystemExit(f"✗ 点位 {pi}: per-DP 里没有 rank={rank} 的条目"
                                     f"（有的 rank：{[r for r, _ in _all_dps]}）")
                specs = _mine[0]
            elif isinstance(_bs, list) and _bs and isinstance(_bs[0], dict):
                specs = []
                for bi, b in enumerate(_bs, 1):
                    _L = int(b.get("L", pt.get("L", 0)) or 0)
                    _h = _as_ratio(b.get("hit", pt.get("hit", 0) or 0))
                    if _L <= 0:
                        raise SystemExit(f"✗ 点位 {pi} 第 {bi} 批缺 L")
                    if not (0.0 <= _h < 1.0):
                        raise SystemExit(f"✗ 点位 {pi} 第 {bi} 批 hit ∈ [0,1)：{b.get('hit')!r}")
                    specs.append({"L": _L, "hit": _h})
            else:
                L0 = int(pt.get("L") or 0)
                if L0 <= 0:
                    raise SystemExit(f"✗ 点位 {pi} 缺 L")
                h0 = _as_ratio(pt.get("hit", 0) or 0)
                if not (0.0 <= h0 < 1.0):
                    raise SystemExit(f"✗ 点位 {pi} 的 hit 必须 ∈ [0,1)（也接受 96/96%）：{pt.get('hit')!r}")
                nb0 = max(1, int(pt.get("batches", 1) or 1))
                specs = [{"L": L0, "hit": h0}] * nb0

            nb = len(specs)
            # ★ 2026-11-01（A5 验证脚手架）：DS41M_ALLOW_EMPTY_DP=1 时允许某 DP **0 token**
            #   （EP8 下"7 个 rank 有 token、1 个 rank T=0"的真实场景）。此时 L/h 取同点位
            #   其它 DP 的值（池/元数据几何照样按最长序列建，只是本 rank 没有行要算）。
            if nb == 0:
                _ref = next((sp for _r, sp in _all_dps if sp), [])
                if not _ref:
                    raise SystemExit(f"✗ 点位 {pi}: 所有 DP 都空")
                L = max(int(b["L"]) for b in _ref)
                h = max(float(b["hit"]) for b in _ref)
                _empty_origin = True
            else:
                L = max(int(b["L"]) for b in specs)          # 池按最长序列开
                h = max(float(b["hit"]) for b in specs)      # meta.hit 只用于日志/抑制判断
                _empty_origin = False
            rep = bool(pt.get("replay", True))
            rtail = int(pt.get("replay_tail", base_hit.get("replay_tail", 0)) or 0)
            # ★ DeepInfer KEEP_TOKENS 对齐：与 assemble.replay_tail_of 同口径（V4.1 140）。
            rtail = assemble.replay_tail_of({"model": cfg["model"],
                                             "hit": {"replay_tail": rtail}})
            # ★ 2026-11-01（铁律 0）：命中几何**显式预检**——`M>0` 且 `R<=L`。
            #   以前非法几何（如 1k@0.1：L=1024, M=922, R=1062>L）只能靠生成脚本
            #   `tools/finish_cp2cp4.sh:164` 的 `continue` 静默丢点 ⇒ 点位矩阵缺测且没人知道。
            #   现在整轮直接报错并点名"哪个点位/哪一批、L/M/R 各是多少"。
            for _bi, _b in enumerate(specs, 1):
                _g = {"model": cfg["model"],
                      "hit": {"ratio": float(_b.get("hit", 0) or 0),
                              "non_cached_tokens": int(_b.get("non_cached_tokens", 0) or 0),
                              "replay_tail": rtail}}
                if assemble.hit_is_on(_g):
                    try:
                        assemble.check_hit_geometry(_g, int(_b["L"]),
                                                    where=f"点位 {pi} 第 {_bi} 批")
                    except ValueError as _e:
                        raise SystemExit(f"✗ {_e}")

            def _m_of(b):
                return assemble.hit_m({"hit": {
                    "ratio": float(b.get("hit", 0) or 0),
                    "non_cached_tokens": int(b.get("non_cached_tokens", 0) or 0),
                }}, int(b["L"]))

            def _hit_on(b):
                return (float(b.get("hit", 0) or 0) > 0
                        or int(b.get("non_cached_tokens", 0) or 0) > 0)

            rows = [(_m_of(b) + rtail) if _hit_on(b) else int(b["L"]) for b in specs]
            # ★ 2026-11-01（A5 空 origin 正解）：本 DP 无 token 时不"跳过算子"，而是用
            #   **1 行虚拟请求**走**完整算子栈** —— 集合通信（engram 的逐层 2-rank 域 +
            #   MegaMoE 的 all2all）轮次与形状才能和正常 rank 对齐（实测"跳过算子"会让
            #   对端 barrier 超时：`rank barrier timeout target=2 signal=1`）。
            #   虚拟请求 = 关 hit（1 行 plain prefill）、几何 cu=[0,1]/row_lens=[1]；
            #   跑完结果丢弃（只参与通信，不产生自身行）。
            if _empty_origin:
                rows = [1]
            cu = [0]
            for r in rows:
                cu.append(cu[-1] + int(r))
            _nsp_pre = _nsp_post = 0
            if per_dp:
                # ★ 2026-10-21 用户口径：**per-DP 与多点位分开**——per-DP 只允许单点位；
                #   多点位场景所有 DP 用同一套规格（不要带 dps）。
                if len(pts) > 1:
                    raise SystemExit(
                        f"✗ per-DP 模式只能有 1 个点位（当前 {len(pts)} 个）。\n"
                        f"  多点位请去掉 `dps`、所有 DP 用同一套 batches；"
                        f"per-DP 请单独跑一次。")
                _chunk = int((cfg.get("run") or {}).get("moe_chunk_max_len", 8192) or 8192)
                _w = assemble.replay_keep_tokens(cfg)   # DeepInfer KEEP：window + 最大回退
                _rep_on = bool((cfg.get("features") or {}).get("decoder_replay")) and rep
                def _rows_of(sp_list):
                    return [(_m_of(b) + rtail) if _hit_on(b) else int(b["L"])
                            for b in sp_list]
                _pre_spans, _post_spans, _trims = [], [], []
                for _r, _sp in _all_dps:
                    _pre = sum(_rows_of(_sp))
                    _pre_spans.append(-(-_pre // _chunk))
                    _tr = bool(_rep_on and _pre > _w)
                    _trims.append(_tr)
                    _post_spans.append(-(-(_w * len(_sp)) // _chunk) if _tr else None)
                # ★ 2026-11-01（A5）：空 DP（0 token）不参与"裁不裁尾"的裁决 —— 它没有行，
                #   也不该把自己的 `_tr=False` 当成反例；按有 token 的 DP 的多数派补齐。
                _nz_trims = [t for (_r, _sp), t in zip(_all_dps, _trims) if _sp]
                if len(set(_nz_trims)) != 1:
                    raise SystemExit(
                        "✗ per-DP：有的 DP 会裁尾、有的不会（各自 pre T 与 W 的关系不同）——不支持；"
                        "请把各 DP 的 T 调到同一侧")
                _tr0 = bool(_nz_trims[0]) if _nz_trims else False
                _trims = [(_tr0 if not _sp else t)
                          for (_r, _sp), t in zip(_all_dps, _trims)]
                _post_spans = [(-(-(_w * len(_sp)) // _chunk) if t else None)
                               for (_r, _sp), t in zip(_all_dps, _trims)]
                _nsp_pre = max(_pre_spans)
                _nsp_post = max(int(x) for x in _post_spans if x) if _tr0 else 0
                _dummy_ok = os.environ.get("DS41M_MOE_DUMMY_SPAN", "1").strip().lower() not in (
                    "0", "off", "false", "no")
                _uneq = (len(set(_pre_spans)) > 1
                         or (len(set(x for x in _post_spans if x)) > 1))
                if _uneq and not _dummy_ok:
                    raise SystemExit(
                        f"✗ per-DP 的 MoE span 数不一致（chunk={_chunk}）：pre={_pre_spans} "
                        f"post={_post_spans}。\n  MegaMoE 同步 → 各 rank 轮数必须相同；"
                        f"设 DS41M_MOE_DUMMY_SPAN=1（默认）用 dummy 轮对齐，或把各 DP 的 L/批数调一致。")
                if _uneq:
                    print(f"  [per-DP] span 不一致 → dummy 对齐：pre={_pre_spans} → {_nsp_pre} 轮，"
                          f"post={_post_spans} → {_nsp_post or '-'} 轮（chunk={_chunk}）", flush=True)
            # ★ 默认 packed（统一口径）；需要旧单序列路径做 A/B 时 DS41M_UNIFY_PACKED=0
            _unify = os.environ.get("DS41M_UNIFY_PACKED", "1").strip().lower() not in (
                "0", "off", "false", "no")
            pkd = (nb > 1) or _unify
            # 点位几何写回 cfg（层栈/池/元数据全按它建）
            cfg["run"]["seq_len"] = L
            # ★ A5 空 origin：虚拟请求是**关 hit** 的 1 行 plain prefill（否则 runner 会按
            #   hit 几何去切 R 行，而本 rank 只有 1 行可切）。
            # ★ C5 收尾：命中策略的写入也走 metadata 单源（整块 dict 形态，未知字段留在镜像里）
            import metadata as _md_h
            _md_h.set_hit_dict(cfg, dict(base_hit,
                                        ratio=(0.0 if _empty_origin else h),
                                        replay_tail=rtail, source="fake"))
            # ★ 2026-11-01：本点位几何（H=前缀行、E=重算的 replay 行、S=query 起点）落盘用。
            #   · 命中关闭：H=0、E=0、S=0、M=L、R=L；
            #   · 命中开启：E=keep（本仓命中段**不用**额外 encoder replay，固定 0）、
            #     M=non_cached、R=M+keep、H=L-M、S=L-R（= 尾部起点，与 runner 的 pos_off 同口径）。
            _hb = bool(h) or any(int(b.get("non_cached_tokens", 0) or 0) > 0 for b in specs)
            if _empty_origin:
                # 虚拟请求：1 行、无 hit ⇒ M=0/R=1/H=L/S=L-1（照着写，便于日志辨认）
                _L_tot = int(L)
                _M_geo = 0
                _R_geo = 1
                _hb = False
            else:
                _L_tot = sum(int(b["L"]) for b in specs)    # 本 pass 全部逻辑 token 数
                _M_geo = sum(int(_m_of(b)) for b in specs) if _hb else int(_L_tot)
                _R_geo = sum(int(r) for r in rows)
            _H_geo = int(_L_tot) - _M_geo
            _S_geo = (max(0, int(L) - int(rows[0]))          # 首段 query 起点（多批看首段）
                      if rows else 0)                        # A5：空 DP 没有 query 行
            _geo = dict(L=int(L), L_tot=int(_L_tot), M=int(_M_geo), R=int(_R_geo),
                        H=int(_H_geo), S=int(_S_geo), keep=int(rtail) if _hb else 0,
                        rows=[int(r) for r in rows], nb=int(nb),
                        actual_hit=(1.0 - _M_geo / max(1, int(_L_tot))) if _hb else 0.0)
            # ★ 2026-11-01（B1 吸收，GPT §4.2 改写口径）：三段**显式命名**。
            #   本仓语义：prefix_hit = 前缀命中行（= H = L-M）；enc_replay = 额外 encoder
            #   replay（本仓命中段**固定为 0**，只做尾段重算）；dec_keep = DeepInfer KEEP
            #   （window + 最大回退，默认 140）。**只命名与打印，不改计算**。
            _geo["prefix_hit"] = int(_H_geo)
            _geo["enc_replay"] = 0
            _geo["dec_keep"] = int(_geo["keep"])
            # ★ 2026-11-01（P3-1，PERF_RULES 口径）：三段**行数**——层 0..19 与层 20 吃 R 行
            #   （层 20 = 边界层，cut 前），层 21..39 吃裁后行数（min(keep, R)；replay 关时 = R）。
            _geo["rows_q0_19"] = int(_R_geo)
            _geo["rows_q21_39"] = int(min(int(_geo["keep"]) or int(_R_geo), int(_R_geo)))
            if pkd:
                import metadata as _md
                # varlen 唯一口径：布局进 metadata（`_packed` 只是它的镜像）
                cfg.pop("_meta_obj", None)
                cfg["_packed"] = {"n": nb, "lens": [int(b["L"]) for b in specs],
                                  "row_lens": [int(r) for r in rows], "cu": cu}
                if per_dp and _nsp_pre:
                    cfg["_packed"]["nspan_pre"] = int(_nsp_pre)
                    cfg["_packed"]["nspan_post"] = int(_nsp_post)
                _md.ensure(cfg)                     # 由点位几何建 metadata + 镜像
            else:
                cfg.pop("_packed", None)
                cfg.pop("_meta_obj", None)
            # ── 产物命名（提前算：续跑跳过、失败标记都要用）─────────────────────
            # ★★ 2026-10-29 用户口径：**采集时就生成标准扁平名**（不再用 `(L_hit)_r1`）——
            #   `cp<CP>dp<DP>ep<EP>_L<Lk>_hit<h>_batch<批>_replay<on|off>_prefill`
            #   规则见 PERF_RULES §5 / PIPELINE.md；解析/打包工具直接认这个名字。
            #   多批且规格不同：L 与 hit 用 `+` 连接
            #   （`L81920+65536_hit0.97+0.9_batch2_replayon_prefill`）。
            #   per-DP 各 rank 规格不同时用点位自带的 `tag` 分组名（各 rank 必须得到同一个目录名）。
            _tag_specs = "&".join(f"{int(b['L'])}_{float(b['hit']):.4g}" for b in specs)
            _same_all = per_dp and all(sp == specs for _r, sp in _all_dps)
            _pc = (cfg.get("parallel_config") or {})
            _w = int(_pc.get("world_size", 1) or 1)
            _c = int(os.environ.get("DS41M_CP_SIZE") or _pc.get("cp_size", 1) or 1)
            _e = int(_pc.get("moe_ep_size", _w) or _w)
            _d = max(1, _w // max(1, _c))
            _flat_pfx = f"cp{_c}dp{_d}ep{_e}"
            # ★ 2026-11-01：结果指纹（几何 + 并行 + 实际后端）。换树/换开关 → 变（缺陷 G6）。
            _bk = assemble.backends_of(_c)
            _ph = assemble.plan_hash(int(L), _M_geo, _R_geo, rep, _c, _w, _bk)
            if per_dp and not _same_all:
                tag = (f"{_flat_pfx}_{pt.get('tag') or 'perdp'}_batch{len(specs)}"
                       f"_replay{'on' if rep else 'off'}_prefill")
            else:
                _l_tags = "+".join(_fmt_L(int(b["L"])) for b in specs)
                _h_tags = "+".join(f"{float(b['hit']):g}" for b in specs)
                tag = (f"{_flat_pfx}_L{_l_tags}_hit{_h_tags}"
                       f"_batch{len(specs)}_replay{'on' if rep else 'off'}_prefill")
            # ★ 去重/续跑：`results.tsv` 里已有的点位直接跳过（进程被 watchdog/崩溃打断后，
            #   重跑同一条命令只补没跑完的点）。`DS41M_POINT_REDO=1` 强制重采。
            if tag in _done_tags:
                print(f"  ↷ [point {pi}/{len(pts)}] 跳过（results.tsv 已有）  {tag}", flush=True)
                _wd["t0"] = None                 # 跳过的点位不算进 watchdog
                continue
            # 确定性失败的点位：失败次数到上限就跳过，否则会一直堵在最前面（见 _load_fail_counts）
            if _fail_counts.get(tag, 0) >= _max_attempts:
                print(f"  ↷ [point {pi}/{len(pts)}] 跳过（已失败 {_fail_counts[tag]} 次，"
                      f"达上限 {_max_attempts}）  {tag}", flush=True)
                _wd["t0"] = None
                continue
            # replay 是**点位级**开关：两个方向都必须显式写 env（run_forward 会把未设的默认成 1）
            os.environ["DS41M_DECODER_REPLAY"] = "1" if rep else "0"
            # ★ 2026-10-29：顺序 = 先排空、再释放、再回收。
            #   先 sync 是安全前提：上一点位可能还有在飞 kernel 引用着 `_shared` 池 /
            #   side stream 的 gather 缓冲；现在把 Python 引用一放，块会被后续分配复用
            #   ⇒ 必须先确认设备队列排空（use-after-free 的最短路径）。
            # ★ 上一点位的 forward **返回值**也要放：`r["x"]` 是 merge_global 后的
            #   **全局 [L,H]**（256k = 2.7 GiB），`r["cp"]` 带着那个 plan 的固定 gather 缓冲。
            #   它会被 `r = R.run_prefill_auto(...)` 覆盖 —— 但那是本点位第一个 forward
            #   **跑完**之后，整段 build/预热都在它头上顶着。边界上显式置空。
            r = None
            _free_device()
            _reset_between_points(R, measure=_audit)
            _free_device()
            if _audit and (pi % _audit_every == 0):
                # 标签只用点号：`tag` 在下面才算（这里要抢在 make_input/build_stack 之前取数）
                _r = _memlife.report(f"[point {pi}/{len(pts)}]",
                                     base=_base_alloc, prev=_prev_alloc)
                if _r:
                    _a = _r["alloc"]
                    _base_alloc = _a if _base_alloc is None else min(_base_alloc, _a)
                    _prev_alloc = _a
            try:
                torch.npu.reset_peak_memory_stats()
            except Exception:                 # noqa: BLE001
                pass

            stack, meta = assemble.build_stack(cfg)
            n_stack = len(stack)
            # 逐批 seed：可用 DS41M_PACKED_SEEDS 显式指定（序列隔离对拍：让两轮的同一批同种子）
            _seed_env = [int(v) for v in
                         os.environ.get("DS41M_PACKED_SEEDS", "").replace(",", " ").split()]
            _sd = [(_seed_env[b] if b < len(_seed_env)
                    else seed0 + 7919 * b + (1009 * int(rank) if per_dp else 0))
                   for b in range(nb)]
            seqs = [R.make_input(cfg, int(rows[b]), seed=_sd[b]) for b in range(nb)]
            # ★ 2026-11-01（A5 空 origin）：本 DP 无 token 时**不走"跳过算子"**，而是造
            #   **1 行虚拟请求**跑完整栈 —— engram（逐层 2-rank HCCL 域）与 MegaMoE 的
            #   all2all 都必须**每个 rank 都到场**，否则对端 `rank barrier timeout`。
            #   虚拟请求关 hit（1 行 plain prefill），结果在下面落盘前丢弃。
            if _empty_origin:
                batch_inputs = [R.make_input(cfg, 1, seed=0)]
                T_run = 1
            elif pkd:
                batch_inputs = [torch.cat(seqs, dim=0)]
                T_run = int(sum(rows))
            else:
                batch_inputs = seqs
                T_run = int(rows[0])
            npass = 1 if (pkd or _empty_origin) else max(1, nb)
            pdir = os.path.join(root, tag, "prefill")
            if per_dp:
                os.makedirs(os.path.join(root, tag), exist_ok=True)
                with open(os.path.join(root, tag, "dps.json"), "w", encoding="utf-8") as _df:
                    json.dump(_dps, _df, ensure_ascii=False, indent=2)
            if a.prof and prof_here:
                # ★ 重跑遗留：本点位**不在** results.tsv 里（否则上面已跳过）⇒ pdir 里的
                #   `PROF_*` 一定是上次被 watchdog/崩溃打断的半成品。清掉再采，避免解析
                #   时同一形状被重复计数（msprof 的 export 也不幂等）。
                try:
                    for _d in os.listdir(pdir):
                        if _d.startswith("PROF_"):
                            shutil.rmtree(os.path.join(pdir, _d), ignore_errors=True)
                except OSError:
                    pass
                os.makedirs(pdir, exist_ok=True)
            _rows_s = "+".join(str(int(r)) for r in rows)
            print(f"\n──── [point {pi}/{len(pts)}] {tag}  T_pass={T_run}（各行 {_rows_s} × {nb} 段）"
                  f"  specs={_tag_specs}{' per-DP' if per_dp else ''} replay={'on' if rep else 'off'}"
                  f"  T_main={meta['T_main']} cut={meta['cut']}"
                  f"  prof={'y' if (a.prof and prof_here) else 'n'}", flush=True)
            # ★ 2026-11-01：几何 + 指纹 + 实际后端一并打印（与 results.tsv 新列同源）。
            print(f"  [geo] L={_geo['L']} M={_geo['M']} R={_geo['R']} H={_geo['H']}"
                  f" S={_geo['S']} keep={_geo['keep']}"
                  f"  target_hit={h:.4g} actual_hit={_geo['actual_hit']:.4g}"
                  f"  plan_hash={_ph}", flush=True)
            # ★ 2026-11-01（B1 / P3-1）：三段命名 + 三段行数（与 results.tsv 新列同源）
            print(f"  [segs] prefix_hit={_geo['prefix_hit']} enc_replay={_geo['enc_replay']}"
                  f" dec_keep={_geo['dec_keep']}"
                  f" | rows Q0-19={_geo['rows_q0_19']} Q20={_geo['R']}"
                  f" Q21-39={_geo['rows_q21_39']}", flush=True)
            print(f"  [backend] {_bk}", flush=True)
            if rank == 0:                    # pure 模式也能看到：第几个点 / 规格 / 开始时间
                print(f"[point {pi}/{len(pts)}] 开始 {time.strftime('%H:%M:%S')}  {tag}"
                      f"  specs={_tag_specs} replay={'on' if rep else 'off'}"
                      f"  T_pass={T_run}", flush=True)
            profiler = R.make_profiler(pdir, a.alop_config) if (a.prof and prof_here) else None

            # ★ 2026-10-29：跨 rank 对齐的 barrier 已挪到**点位体最开头**（见 `try:` 之后）。
            #   原因：它原来在 make_input / build_stack **之后**，一旦某个 rank 在前面就抛异常
            #   （OOM 最常见），该 rank 会少调一次 barrier ⇒ 全体错位 → 后面每一次集合通信
            #   都错一拍 → 整轮挂死（"一个点出问题，全部挂死"）。现在每个 rank 每个点位
            #   无条件先对齐一次，跳过的点位也照样对齐。
            t_p0 = time.time()
            # ── 预热（不进 trace）──
            for w in range(warm):
                for b in range(npass):
                    R.run_prefill_auto(cfg, source, T_run, None, None,
                                       allow_placeholder=a.allow_placeholder, prog=None,
                                       inputs=batch_inputs[b], input_ids=None,
                                       shared=None, hit_stage=None)
            # ★ 2026-10-21 用户口径：预热跑完先 **排空设备**，之后才开采；
            #   否则预热尾部还在队列里，trace 的第一个 pass 会混入前一轮的尾巴。
            torch.npu.synchronize()
            print(f"  [warmup] {warm} 轮预热完成 → torch.npu.synchronize() 已排空，开始采集",
                  flush=True)
            if R.start_line_sync():
                print("  [起跑线] 跨 rank barrier ×1 完成 → 开 profiler 采集", flush=True)
            if profiler is not None:
                dstr = cfg["run"].get("device", "npu:0")
                did = int(dstr.split(":")[-1]) if ":" in dstr else 0
                profiler.start(device=did)

            # ★ 2026-10-21：去掉 **pass 之间的空泡**。旧口径每个 pass 末尾 `torch.npu.synchronize()`
            #   → 设备排空后，下一 pass 的 host 收尾/取权重/建 ctx（实测 ~4.6ms，见 80k trace 的
            #   pass 边界 idle gap）期间设备空转。官方 hit-inner-reps 是**流水**跑法：pass 之间不排空。
            #   这里默认切流水（DS41M_PIPE_PASSES=0 回旧口径），用 NPU event 在设备侧给每个 pass 打点：
            #   · 稳态数 = event 差（纯设备区间，跨 pass 的 host 气泡被剔除）
            #   · 另打印总墙钟/N_pass（含所有开销的吞吐，用来看 host 是否真的跟上）
            #   trace 轮（profiler != None）保持隔离口径，trace 结构不变。
            # ★ 2026-10-21：**prof 轮也流水**（用户口径：trace 里 pass 之间不要间隔）。
            #   profiler 的 step() 仍按层打点，pass 边界由时间线自然分开；最后一次 pass 由
            #   stop_profiler 强制排空。DS41M_PIPE_PASSES=0 可回旧的"每 pass 排空"口径。
            _pipe = (os.environ.get("DS41M_PIPE_PASSES", "1").strip().lower()
                     not in ("0", "off", "false", "no"))
            # ★ 2026-10-21 用户口径：**纯 pass 开关**（DS41M_PURE_PASSES，默认开）。
            #   被测量区间里任何 host 动作（time.time()/Event 创建与 record/print/记账）都是
            #   一次下发断点：host 线程从"连续喂 pass"切去干别的，device 侧就在 pass 边界
            #   出现空泡。纯模式下区间内**只有 `run_prefill_auto` 本身**：
            #     · 不建 NPU event、不取墙钟、不打印、不做额外记账；
            #     · 所有 pass 连续下发（no_sync=True），最后一次性 synchronize；
            #     · 度量 = 总墙钟 / (iters×npass) = 真实吞吐（含 host 跟不上时的气泡）。
            #   DS41M_PURE_PASSES=0 回旧口径（每 pass NPU event 打点 + 逐轮打印）。
            _pure = (os.environ.get("DS41M_PURE_PASSES", "1").strip().lower()
                     not in ("0", "off", "false", "no"))
            if _pure:
                _pipe = True                 # 纯 pass 必须流水；每 pass 排空本身就是空泡
            it_secs, it_evs, r = [], [], None
            if _pure:
                # ── 纯 pass：测量区间零 host 操作（连续下发 → 一次 sync）────────────
                _t0 = time.time()
                for it in range(iters):
                    for b in range(npass):
                        _last = (it == iters - 1 and b == npass - 1)
                        r = R.run_prefill_auto(cfg, source, T_run, None, profiler,
                                               allow_placeholder=a.allow_placeholder, prog=None,
                                               inputs=batch_inputs[b], input_ids=None,
                                               stop_profiler=_last,
                                               shared=None, hit_stage=None,
                                               no_sync=(not _last))
                torch.npu.synchronize()      # 唯一一次排空（在测量区间之后）
                _loop_s = time.time() - _t0
                _pipe_ms = _loop_s * 1000.0 / max(1, iters * npass)
                _per_it = _loop_s / max(1, iters)          # 每轮 = npass 个 pass
                it_secs = [_per_it] * iters                # 无逐轮采样，按吞吐折算
                it_dev = it_secs
                print(f"  [纯 forward] {iters}×{npass}={iters * npass} 个 forward 连续下发"
                      f"（区间内无计时/打印/event）→ 一次 sync：{_loop_s:.3f}s，"
                      f"{_pipe_ms:.1f} ms/forward", flush=True)
            else:
                t_loop0 = time.time()
                for it in range(iters):
                    t_it = time.time()
                    _evs = []
                    for b in range(npass):
                        _last = (it == iters - 1 and b == npass - 1)
                        if _pipe:
                            _e0 = torch.npu.Event(enable_timing=True)
                            _e0.record()
                        r = R.run_prefill_auto(cfg, source, T_run, None, profiler,
                                               allow_placeholder=a.allow_placeholder, prog=None,
                                               inputs=batch_inputs[b], input_ids=None,
                                               stop_profiler=_last,
                                               shared=None, hit_stage=None,
                                               no_sync=(_pipe and not _last))
                        if _pipe:
                            _e1 = torch.npu.Event(enable_timing=True)
                            _e1.record()
                            _evs.append((_e0, _e1))
                    it_secs.append(time.time() - t_it)
                    it_evs.append(_evs)
                    if iters > 1 and not _pipe:
                        print(f"  [iter {it + 1}/{iters}] {it_secs[-1]:.3f}s"
                              f"（{it_secs[-1] * 1000 / max(1, n_stack):.2f} ms/层）", flush=True)
                if _pipe:
                    torch.npu.synchronize()          # 一次性排空；之后 elapsed_time 才有效
                    t_loop1 = time.time()
                    it_dev = [sum(a.elapsed_time(b) for a, b in evs) / 1000.0 for evs in it_evs]
                    _pipe_ms = (t_loop1 - t_loop0) * 1000.0 / max(1, iters * npass)
                    if iters > 1:
                        print(f"  [iter] 设备 {['%.3f' % x for x in it_dev]}s"
                              f"  流水吞吐 {_pipe_ms:.1f} ms/forward（含全部开销）", flush=True)
                else:
                    it_dev = it_secs
            if profiler is not None:
                profiler.finalize()

            # ── 稳态口径（PERF_RULES §0）：剔除冷启首轮，≥3 样本取中位 ──
            # ★ 2026-10-29：跑过预热（warm>0）时冷启已被预热吃掉 → 默认**不再剔除**
            #   （用户口径「2 轮预热 + 8 轮计时」= 8 个样本全算）。显式覆盖：
            #   DS41M_STEADY_KEEP_FIRST=1 保留首轮 / =0 仍剔除首轮。
            _kf = os.environ.get("DS41M_STEADY_KEEP_FIRST", "").strip().lower()
            if _kf in ("1", "on", "true", "yes"):
                _excl = set()
            elif _kf in ("0", "off", "false", "no"):
                _excl = {0}
            else:
                _excl = set() if warm > 0 else {0}
            _base = it_dev if _pipe else it_secs   # 流水口径 = event 设备区间；旧口径 = 墙钟
            warm_s = [s for i, s in enumerate(_base) if i not in _excl]
            if not warm_s:
                warm_s = _base[1:] or _base
            if len(warm_s) >= 3:
                _val, _kind = statistics.median(warm_s), "中位"
            else:
                _val, _kind = sum(warm_s) / len(warm_s), "均值"
            try:
                peak = torch.npu.max_memory_allocated() / 2 ** 30
            except Exception:                 # noqa: BLE001
                peak = float("nan")
            nerr = len((r or {}).get("errors") or [])
            n_bad += 1 if nerr else 0
            # ★ 2026-10-21：结果落盘（DS41M_LOG_LEVEL=pure 时 stdout 只放行进度行，点位稳态数
            #   不再出现在控制台/日志里；这里保证数据不丢）。rank0 写，一点一行。
            if rank == _report_rank(a, getattr(a, "world", 1)):
                try:
                    _rf = os.path.join(root, "results.tsv")
                    if _empty_origin:
                        # ★ A5：空 origin 的"耗时"没有意义（只算 1 行虚拟请求），不写进结果表，
                        #   免得污染矩阵；几何与后端照样打印在上面的 [geo]/[backend]。
                        print(f"  [empty-dp] 点位 {pi}：空 origin（1 行虚拟请求）跑完，"
                              f"不写 results.tsv", flush=True)
                        raise _SkipEmptyOrigin()
                    _ms_pass = _pipe_ms if _pipe else (_val * 1000.0 / max(1, npass))
                    _new = not os.path.exists(_rf)
                    # ★ 2026-11-01（缺陷 G6）：除稳态数外，落盘**实际几何**（L/M/R/H/S/实际
                    #   hit）与**指纹**（plan_hash + 实际后端）。旧表只有 tag 与耗时，
                    #   换树/换开关后数字混用无法辨认。新列一律追加在 `ts` 之后，
                    #   读取方都是按列名取（csv.DictReader / 首列 tag），旧表不动也读得通。
                    _ms_meta = {}
                    try:
                        _ms_meta = R.exec_ops.meta_snapshot()
                    except Exception:              # noqa: BLE001 —— 诊断列，缺了不致命
                        _ms_meta = {}
                    _extra = (f"\t{_geo['L']}\t{_geo['M']}\t{_geo['R']}\t{_geo['H']}"
                              f"\t{_geo['S']}\t{_geo['keep']}\t{_geo['actual_hit']:.6f}"
                              f"\t{_ph}\t{_bk}"
                              f"\t{_geo['prefix_hit']}\t{_geo['enc_replay']}"
                              f"\t{_geo['dec_keep']}\t{_geo['rows_q0_19']}"
                              f"\t{_geo['rows_q21_39']}"
                              f"\t{_ms_meta.get('geo_hit', -1)}"
                              f"\t{_ms_meta.get('geo_miss', -1)}"
                              f"\t{_ms_meta.get('fwd_hit', -1)}"
                              f"\t{_ms_meta.get('fwd_miss', -1)}")
                    with open(_rf, "a", encoding="utf-8") as _fh:
                        if _new:
                            _fh.write("tag\tsteady_s\tms_per_forward\terrors\tts\t"
                                      + "\t".join(_RESULT_COLS) + "\n")
                        _fh.write(f"{tag}\t{_val:.4f}\t{_ms_pass:.2f}\t{nerr}\t"
                                  f"{time.strftime('%Y-%m-%dT%H:%M:%S')}{_extra}\n")
                except Exception:              # noqa: BLE001
                    pass
            if nerr:                       # 只计数看不出原因 → 打前 3 条（层/算子/异常）
                # ★ 2026-10-31：异常正文默认截到 180 字会把**编译器报错**切掉（deep-jit
                #   的 "command failed ...: / cd ... / <stderr>" 全在第 2 行之后）⇒ 查不了根因。
                #   `DS41M_ERR_FULL=1` 时不截断（铁律 0：报错要能定位）。
                _ef = os.environ.get("DS41M_ERR_FULL", "").strip().lower()
                _lim = 8000 if _ef not in ("", "0", "off", "false", "no") else 180
                for _li, _k, _e in ((r or {}).get("errors") or [])[:3]:
                    print(f"    ⚠️ [L{_li}] {_k}: {_e[:_lim]}", flush=True)
            if _pure:
                # 纯 pass 无逐轮采样：只报吞吐（_val = 每轮 = npass 个 pass 的折算值）
                print(f"  纯 forward 吞吐：{_pipe_ms:.1f} ms/forward × {iters * npass} forward ="
                      f"{_pipe_ms * iters * npass / 1000:.3f}s（一次 sync，无逐轮采样）"
                      f"  均={_val:.3f}s/轮  峰值 {peak:.2f} GiB  用时 {time.time() - t_p0:.0f}s"
                      + (f"  ⚠️ 出错 {nerr} 处" if nerr else ""), flush=True)
            else:
                _wall = (f"  墙钟{['%.3f' % x for x in it_secs]}s"
                         f" 流水{_pipe_ms:.1f}ms/forward") if _pipe else ""
                print(f"  多轮耗时：{['%.3f' % x for x in _base]}s  均(除首轮)={_val:.3f}s"
                      f"  最快={min(_base):.3f}s  [稳态 {len(warm_s)} 轮{_kind}；剔除轮={sorted(_excl)}]"
                      f"{_wall}  峰值 {peak:.2f} GiB  用时 {time.time() - t_p0:.0f}s"
                      + (f"  ⚠️ 出错 {nerr} 处" if nerr else ""), flush=True)
            if a.prof and prof_here:
                print(f"  trace → {pdir}", flush=True)
            if rank == 0:                    # pure 模式小结：本点用时 + 预热后的每 pass
                #   每 pass 口径 = 稳态设备区间（_pipe_ms）；预热/首轮都在计时之外。
                print(f"[point {pi}/{len(pts)}] 完成 {time.strftime('%H:%M:%S')}  {tag}"
                      f"  用时 {time.time() - t_p0:.1f}s  每forward={_pipe_ms:.1f}ms"
                      f"（iters={iters}×{npass} forward，除预热）", flush=True)
            all_rows.append((tag, _val, nerr))
            _wd["t0"] = None
        except Exception as _exc:                 # noqa: BLE001 —— 单点失败不许拖垮整轮
            _msg = f"{type(_exc).__name__}: {_exc}"
            print(f"  ✗ [point {pi}/{len(pts)}] {tag or f'#{pi}'} 失败 → 跳过（不中断整轮）："
                  f"{_msg[:400]}", flush=True)
            try:
                import traceback as _tb
                _tb.print_exc(limit=6)
            except Exception:                     # noqa: BLE001
                pass
            _wd["t0"] = None
            _fail_rows.append((tag or f"#{pi}", _msg[:160]))
            if tag:                             # 解析阶段就失败（无产物名）不记账，免得脏 tag
                try:
                    _note_fail(root, tag, rank)     # 跨进程累计（rank0 写）
                except Exception:                   # noqa: BLE001
                    pass
            n_fail += 1
            n_consec += 1
            try:                                  # 别把坏状态带进下一个点
                _free_device()
            except Exception:                     # noqa: BLE001
                pass
            if n_consec >= _max_fails:
                print(f"  ✗ 连续 {n_consec} 个点位失败 → 退出进程"
                      f"（重跑同命令会从下一个未完成点位续跑）", flush=True)
                sys.stdout.flush()
                return 9
            continue
        else:
            n_consec = 0

    print(f"\n== --points 结束：{len(all_rows)} 个点位完成"
          + (f"，{n_fail} 个点位失败被跳过" if n_fail else "")
          + (f"，{n_bad} 个点位有算子错误" if n_bad else "，无算子错误"), flush=True)
    for tag, val, nerr in all_rows:
        print(f"   {tag:<28} 稳态 {val:.3f}s" + (f"  ⚠️ {nerr} 错" if nerr else ""), flush=True)
    if _fail_rows:
        print(f"\n== 失败点位（重跑同命令会自动补这些；已完成的会跳过）：", flush=True)
        for t, m in _fail_rows:
            print(f"   ✗ {t}: {m}", flush=True)
        return 9
    if n_bad:
        # ★★ 2026-10-31（主控，铁律 0）：**算子错误必须判失败**。
        #   旧行为：只打一行 "⚠️ N 处"，末尾仍打 "✅ 全部完成"、rc=0 ⇒ 实测被坑两次：
        #   ①dg 路径 deep-jit 失败 8 处；②CP1 packed 断言 NameError 3 处 —— 都被当成"通过"。
        #   有错就是失败：rc=8 让 run_hit.sh 打 ❌。
        print(f"\n== ❌ {n_bad} 个点位有算子错误（见上面的 ⚠️ 行）——按铁律 0 判失败，"
              f"不许当成功继续。rc=8", flush=True)
        return 8
    return 0

"""`cp_plan` —— CP 的**计划对象**：把"这一轮 CP 要做什么"显式声明成数据。

★ 2026-11-01（C4 吸收，GPT §14「新 `cp/cp_plan.py`：carry/halo/KV/boundary/
  terminal-state routes 及 MoE 计划」）。

落点说明（为什么先做 façade 而不是另起几何）：
  · 本仓的几何真源是 `cp_layout.build_cp_plan`（纯 CPU、`tools/test_cp_layout.py`
    13 lens × cp1/2/4/8 覆盖）+ `cp_state`（plan→设备索引/缓冲）。
  · 直接新建"第三套分块公式"会立刻产生两套几何（GPT 同一条 §14 也写了
    "不要建立并存的第三套几何"）。所以 cp_plan 先做**声明层**：
    把 carry / halo / KV / boundary / terminal-state / MoE 六条路由写成显式字段，
    并**委派** `cp_layout` 出实际 plan；对外保证 `CpPlan` 与原实现**逐字段相同**。

后续（C5 一起）：把 `cp_state` 的绑定也收进来，让"计划 → 设备缓冲"只有一条路径。

用法：
    from cp.cp_plan import build_plan
    p = build_plan(seq_lens=[8192], cp_size=4, cp_rank=1,
                   window=128, keep=140, replay=True, moe_span=1)
    p.layout            # 就是 cp_layout.build_cp_plan(...) 的返回（同一对象语义）
    p.routes            # 六条路由的显式声明（dict）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CpRoute:
    """一条路由的声明。`kind` 只用于记账/断言，不改变执行路径。"""
    name: str
    enabled: bool = True
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CpPlanView:
    """计划视图：`layout` 是真几何（cp_layout 产出），`routes` 是显式声明。"""
    layout: Any
    routes: Dict[str, CpRoute]
    seq_lens: Tuple[int, ...]
    cp_size: int
    cp_rank: int

    # ── 便捷访问（避免调用方到处拿 layout 的属性）──
    @property
    def total_q(self) -> int:
        return int(self.layout.total_q)

    @property
    def local_lens(self) -> Tuple[int, ...]:
        return tuple(self.layout.local_lens)

    @property
    def local_rows(self) -> int:
        return int(sum(self.layout.local_lens))

    @property
    def mirror_blocks(self) -> Tuple[int, int]:
        """本 rank 负责的两个块号：head=`rank`、tail=`2·cp-1-rank`（mirror 定义）。"""
        return (int(self.cp_rank), int(2 * self.cp_size - 1 - self.cp_rank))


def _halo_blocks(layout, seq_lens, window: int, keep: int) -> Tuple[int, List[int]]:
    """尾部窗口的 KV 可见块（**按几何计算**，不是常量表）。

    口径：最后 `keep` 行的 query，其窗口覆盖 `[q-(window-1), q]`（q ∈ [L-keep, L)），
    取并集后在 mirror 分块（每序列 `ps = local_lens/2` 行一块、共 `2·cp` 块）上投影。

    ★ 2026-11-01 纠正：旧口径曾写"tail 140 @ CP8 需 15 个前驱段（CP4 为 7）"。
      按本仓 mirror 分块实测**复现不出**该数字：`L=16384` 时 cp2/4/8 都只需 **1** 块
      （`ps` 分别为 4096/2048/1024 ≥ window+keep）。该说法已在下文标为**存疑并撤回**，
      这里只保留可计算的定义，避免把未核实的数字写进代码。
    """
    if not seq_lens:
        return 0, []
    blocks: List[int] = []
    for i, sq in enumerate(seq_lens):
        ps = int(layout.local_lens[i]) // 2
        if ps <= 0:
            continue
        nblk = 2 * int(layout.cp_size)
        lo = max(0, int(sq) - int(keep) - (int(window) - 1))
        hi = int(sq)
        base = sum(2 * (int(x) // 2) for x in seq_lens[:i])   # 序列在 rank 内的块偏移
        for p in range(lo, hi):
            b = min(nblk - 1, p // ps)
            if (base + b) not in blocks:
                blocks.append(base + b)
    return len(blocks), sorted(blocks)


def _routes(cp_size: int, window: int, keep: int, replay: bool,
            moe_span: int, block_rows: List[int],
            layout=None, seq_lens=()) -> Dict[str, CpRoute]:
    """六条路由的声明（数值都是"这一轮实际会用到"的口径，不是常量表）。"""
    tail_rows = int(block_rows[-1]) if block_rows else 0
    n_halo, halo_ids = _halo_blocks(layout, seq_lens, window, keep) if layout is not None else (0, [])
    return {
        "carry": CpRoute("carry", enabled=(cp_size > 1),
                         detail={"rows": 1, "why": "压缩器窗口跨段回退 1 行"}),
        "halo": CpRoute("halo", enabled=(cp_size > 1 and tail_rows > 0),
                        detail={"lookback": max(0, int(window) - 1),
                                "pred_segments": int(n_halo),
                                "pred_blocks": list(halo_ids),
                                "why": "尾部 query 需看 window-1 行；跨 rank 需覆盖全部相交块",
                                "legacy_claim_2cp_1": "存疑并撤回（见 _halo_blocks 注释）"}),
        "kv_gather": CpRoute("kv_gather", enabled=(cp_size > 1),
                             detail={"mode": "full_current_ag"}),
        "boundary": CpRoute("boundary", enabled=bool(replay),
                            detail={"cut_rows": int(keep), "triggers_if_rows_gt": int(keep)}),
        "terminal_state": CpRoute("terminal_state", enabled=(cp_size > 1),
                                  detail={"per_segment": True}),
        "moe": CpRoute("moe", enabled=True,
                       detail={"spans": int(moe_span), "must_match_across_ranks": True}),
    }


def build_plan(seq_lens, cp_size: int, cp_rank: int, window: int = 128,
               keep: int = 140, replay: bool = True, moe_span: int = 1) -> CpPlanView:
    """声明 + 委派：几何仍由 `cp_layout.build_cp_plan` 产出（**不做第二套公式**）。"""
    import cp_layout                      # 同目录；调用方已把 cp/ 放进 PYTHONPATH

    lens = tuple(int(x) for x in seq_lens)
    layout = cp_layout.build_cp_plan(lens, int(cp_size), int(cp_rank))
    routes = _routes(int(cp_size), int(window), int(keep), bool(replay),
                     int(moe_span), [int(v) for v in layout.local_lens],
                     layout=layout, seq_lens=lens)
    return CpPlanView(layout=layout, routes=routes, seq_lens=lens,
                      cp_size=int(cp_size), cp_rank=int(cp_rank))


@dataclass
class HaloSpan:
    """halo 需要可见的一段 KV：`(块号, 持有该块的 rank, 序列内位置区间)`。"""
    block: int
    rank: int
    pos_lo: int
    pos_hi: int            # 不含

    def covered(self, lo: int, hi: int) -> bool:
        return self.pos_lo <= lo and hi <= self.pos_hi


def _block_owner(b: int, cp_size: int) -> int:
    """mirror 布局：块 b 由 rank `min(b, 2·cp−1−b)` 持有（每 rank 两段：head/tail）。"""
    return int(min(int(b), 2 * int(cp_size) - 1 - int(b)))


def halo_manifest(view: CpPlanView, seq_index: int = 0, window: int = 128,
                  keep: int = 140) -> List[HaloSpan]:
    """把 `halo` 从"声明"变成**可执行清单**：尾部 query 窗口必须可见的 KV 段。

    口径（与 `_halo_blocks` 同一几何）：尾部 `keep` 行 query 的窗口是
    `[q-(window-1), q]`，q ∈ [L-keep, L)；对每个相交块给出
    `(块号, 持有 rank, 位置区间)`，供 CP runtime 做"按清单取"（B5）。
    """
    lens = list(view.seq_lens)
    if not (0 <= seq_index < len(lens)):
        raise ValueError(f"seq_index 越界：{seq_index} / len={len(lens)}")
    L = int(lens[seq_index])
    ps = int(view.layout.local_lens[seq_index]) // 2
    if ps <= 0:
        return []
    nblk = 2 * int(view.cp_size)
    lo = max(0, L - int(keep) - (int(window) - 1))
    hi = L
    out: List[HaloSpan] = []
    b0 = min(nblk - 1, lo // ps)
    b1 = min(nblk - 1, (hi - 1) // ps)
    for b in range(b0, b1 + 1):
        out.append(HaloSpan(block=b, rank=_block_owner(b, view.cp_size),
                            pos_lo=max(lo, b * ps), pos_hi=min(hi, (b + 1) * ps)))
    return out


def halo_covered(manifest: List[HaloSpan], lo: int, hi: int) -> bool:
    """清单是否**覆盖** `[lo,hi)`（无洞）—— 运行时断言与 CPU 测试共用。"""
    if hi <= lo:
        return True
    spans = sorted((s.pos_lo, s.pos_hi) for s in manifest)
    cur = lo
    for a, b in spans:
        if a > cur:
            return False
        cur = max(cur, b)
        if cur >= hi:
            return True
    return cur >= hi


def gather_stats(view: CpPlanView, window: int = 128, keep: int = 140,
                 seq_index: int = 0, mode: str = "replay") -> Dict[str, Any]:
    """全量 AG vs 按清单取（B5 的**收益量化**，纯 CPU 可算，不用占卡）。

    口径（★ 修正：全量 AG 搬的是**所有 cp 个 rank** 的行，不是单 rank）：
      · `rows_total`  = `sum(local_lens) × cp` —— 全量 AG 的搬运量；
      · `rows_needed` = 尾部窗口相交块的块内行之和（只这些块必须可见）；
      · `saved_pct`   = 节省比例。

    ★ `mode` 决定"这一层的 query 覆盖多少行"，直接决定收益是否存在：
      · `"replay"`：层 21+ 只算 KEEP=140 行 ⇒ 只需要尾部窗口那一段（收益可达 90%+）；
      · `"full"`  ：全长段每行都有自己的窗口 ⇒ 并集就是全部行，收益 **= 0**。
      所以在**全长段**上做选择性 gather 没有意义，只在 replay/compact 段有意义。
    """
    lens = list(view.seq_lens)
    if not (0 <= seq_index < len(lens)):
        raise ValueError(f"seq_index 越界：{seq_index} / len={len(lens)}")
    if mode not in ("replay", "full"):
        raise ValueError(f"mode 只支持 replay|full：{mode!r}")
    ps = int(view.layout.local_lens[seq_index]) // 2
    rows_total = int(sum(int(v) for v in view.layout.local_lens)) * int(view.cp_size)
    man = halo_manifest(view, seq_index, window=window, keep=keep)
    rows_needed = int(sum(s.pos_hi - s.pos_lo for s in man))
    if mode == "full":
        rows_needed = rows_total            # 全长段：每行都要自己的窗口 ⇒ 并集=全部
    blocks_total = int(2 * view.cp_size)
    return {
        "cp_size": int(view.cp_size),
        "L": int(lens[seq_index]),
        "mode": mode,
        "ps": int(ps),
        "blocks_total": blocks_total,
        "blocks_needed": len(man),
        "rows_total": rows_total,
        "rows_needed": rows_needed,
        "saved_rows": max(0, rows_total - rows_needed),
        "saved_pct": (0.0 if rows_total <= 0
                      else round(100.0 * max(0, rows_total - rows_needed) / rows_total, 2)),
        "all_gathered": rows_needed >= rows_total,
        "manifest": man,
    }


def halo_gather_plan(view: CpPlanView, seq_index: int = 0, window: int = 128,
                     keep: int = 140) -> List[Dict[str, Any]]:
    """B5 的"谁发哪些行"清单：把 `halo_manifest` 翻译成**每 rank 的本地切片**。

    返回 `[{rank, block, local_row0, nrows, pos_lo, pos_hi}]`：
      · `rank`      = 该块持有者（mirror: `min(b, 2·cp−1−b)`）；
      · `local_row0`= 该块在**持有 rank 的本地行轴**上的起点（= `块内偏移`）；
      · `nrows`     = 必须发送的行数（已被窗口裁到 `[pos_lo,pos_hi)`）。

    为什么需要它：`all_gather_global` 是**全量**交换（每 rank 发 `sum(local_lens)` 行）；
    B5 只想发这几段。清单本身是纯几何，可在 CPU 上验：

      · 各段互不重叠；· 并集 = `halo_manifest` 的位置区间；· 每段落在该 rank 自己的块内。
    """
    lens = list(view.seq_lens)
    if not (0 <= seq_index < len(lens)):
        raise ValueError(f"seq_index 越界：{seq_index} / len={len(lens)}")
    ps = int(view.layout.local_lens[seq_index]) // 2
    if ps <= 0:
        return []
    man = halo_manifest(view, seq_index, window=window, keep=keep)
    out: List[Dict[str, Any]] = []
    for sp in man:
        owner = sp.rank
        # 本 rank 的两个块：head=rank（块 rank）、tail=（块 2·cp−1−rank）
        b = sp.block
        # 块内偏移：head 块起点 = 0；tail 块起点 = 0 也成立（块内局部坐标从 0 起）
        local0 = sp.pos_lo - b * ps
        n = sp.pos_hi - sp.pos_lo
        if local0 < 0 or local0 + n > ps:
            raise ValueError(
                f"halo 段越出块边界：block={b} pos=[{sp.pos_lo},{sp.pos_hi}) ps={ps}")
        out.append({"rank": int(owner), "block": int(b), "local_row0": int(local0),
                    "nrows": int(n), "pos_lo": int(sp.pos_lo), "pos_hi": int(sp.pos_hi)})
    return out


def scatter_plan(view: CpPlanView, seq_index: int = 0, window: int = 128,
                 keep: int = 140) -> List[Dict[str, int]]:
    """B5 的**搬运三元组**：`{dst_pos, src_rank, src_local_row}`。

    语义：目标池的全局位置 `dst_pos`（序列内）应当写入 "持有者 rank 的本地第
    `src_local_row` 行" 的值。runtime 只需按这份清单做 P2P/all_gather + 散写，
    不需要理解 mirror 分块。

    保证（CPU 可验）：
      · `dst_pos` 覆盖 `halo_manifest` 的并集、**无重复**；
      · `0 ≤ src_local_row < 2·ps`（该 rank 的本地行轴长度）；
      · `src_rank == _block_owner(块号, cp)`，且 `src_local_row` 落在该块的半区
        （head 块 → `[0, ps)`；tail 块 → `[ps, 2·ps)`）。
    """
    lens = list(view.seq_lens)
    if not (0 <= seq_index < len(lens)):
        raise ValueError(f"seq_index 越界：{seq_index} / len={len(lens)}")
    ps = int(view.layout.local_lens[seq_index]) // 2
    if ps <= 0:
        return []
    out: List[Dict[str, int]] = []
    for g in halo_gather_plan(view, seq_index, window=window, keep=keep):
        b = int(g["block"])
        # head 块（b == rank）在本 rank 的 [0, ps)；tail 块（b == 2cp-1-rank）在 [ps, 2ps)
        _base = 0 if b < int(view.cp_size) else ps
        for i in range(int(g["nrows"])):
            out.append({"dst_pos": int(g["pos_lo"] + i),
                        "src_rank": int(g["rank"]),
                        "src_local_row": int(_base + g["local_row0"] + i)})
    return out


def describe(view: CpPlanView) -> str:
    r = view.routes
    return (f"cp{view.cp_size}/r{view.cp_rank} len={view.seq_lens} "
            f"local={view.local_lens} blocks={view.mirror_blocks} "
            f"routes: carry={'on' if r['carry'].enabled else 'off'} "
            f"halo={'on' if r['halo'].enabled else 'off'} "
            f"kv={r['kv_gather'].detail.get('mode')} "
            f"boundary={'on' if r['boundary'].enabled else 'off'} "
            f"moe_spans={r['moe'].detail.get('spans')}")
# coding=utf-8
"""Decoder replay：策略解析 + 尾段裁剪（单序列 / packed 统一入口）。

背景
----
本模型 `kv_source_layers=[2,8,14,20]`、`sliding_window=128`：cut=21 之后的层只写滑窗 KV，
不写任何压缩/index cache，最终只有最后一个 token 进 head → 更早的 token 在这些层是白算。
decoder replay 把 cut 之后的前向裁到**每条请求最后 W 行**，逻辑位置仍是绝对坐标。

W 的口径（DeepInfer KEEP_TOKENS 对齐）
------------------------------------
`W = sliding_window + max_backtracking`（默认 128+6=134，见 `assemble.replay_keep_tokens`）：
保留后缀必须覆盖下一 token 的完整 SWA 窗口，并在 DSpark/next-n 回退后仍够用；
持久窗口池也按 `window + 回退` 多存（`exec_ops._kv_epilog` 的 `win_keep`）。
可见域仍按 `rect_lo = L - W` clamp（尾段之前没有 KV 的行 block 0/−1），与 DeepInfer
`yoco_swa_visibility` 的 `max(pos-window+1, L-KEEP, valid_start)` 同式。

数值语义（重要）
----------------
尾段最前面的若干行（窗口越过 `rect_lo` 的那些）仍是**有界截断**，这是裁剪本身的语义，
DeepInfer 的 mask/compact 也一样 clamp 到 `rect_lo`；**不承诺与全量逐位一致**。
能保证的是：最后 token 与 decode/回退所需的窗口完整（W >= window + 回退）；
图↔eager 的 max|Δ|=0 是**同一 replay 口径内**的结论。
旧 128 口径已删除（全按 DeepInfer 口径：V4.1 KEEP_TOKENS=140）。
见 `NOTES/2026-10-19-decoder-replay-semantics.md`、`NOTES/2026-10-22-deepinfer-replay-logic.md`。

设计
----
- `ReplayPolicy`：只解析开关（env / JSON 热切），不做张量操作；
- `apply()`：纯裁剪，返回 `TrimResult`（不打印、不碰 profiler）；runner 负责日志与图逻辑。
- packed 的逐请求切片复用 `batch.tail_slice()`，本模块不自己拼 cu。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import batch as batch_layout

# 随行状态里 "dim0 == 原 T" 的行账；裁尾必须同步切，否则 qsli 形状校验/静默错值。
ROW_ACCOUNT_PREFIXES = ("_li_cand", "_cmp_sparse", "_cmp_topklen")

__all__ = ["ReplayPolicy", "TrimResult", "apply", "ROW_ACCOUNT_PREFIXES"]

# ★ 2026-10-26（B1 §2 D3 / T-056）：**实测无收益，默认关**（保留代码备查）。
#   两台机器交错 4 臂实测：本机 off 106.28/114.14（含 1 个异常臂）vs on 107.73/105.92；
#   邻机（干净）off **106.27/106.71** vs on **107.56/106.36** ⇒ **+0.47 ms（反而略慢）**。
#   digest 全同 ⇒ 语义安全，但那一次 `idx.cpu().tolist()` 的 D2H 并不是瓶颈
#   （idx 只有 140×n 个 int64，且前一层的同步已经把它盖住）。
#   结论：尾段的空泡不在这条 D2H 上；`DS41M_TAIL_IDX_CACHE=1` 可手动打开对照。
#
# 以下是原始设计说明（保留）：尾段裁剪的行号索引是**点位不变量**——
#   `tail_slice(cu, W)` 只依赖 (cu, W)，而同一请求内 cu 每 forward 都一样。
#   原实现每 forward 都做一次 `idx.cpu().tolist()`，那是**阻塞 D2H**（要等设备队列排空），
#   而尾段正是 host-bound 段（尾段 idle 16.7ms/52%）。这里按 (cu, W, device) 缓存
#   device 侧 idx 与 host 侧 list，稳态零 D2H。值不变 ⇒ 数值逐位不变。
#   kill switch：`DS41M_TAIL_IDX_CACHE=0`（回到每次重算的原行为）。
_ENV_TAIL_IDX_CACHE = os.environ.get(
    "DS41M_TAIL_IDX_CACHE", "0").strip().lower() in ("1", "on", "true", "yes")
_TAIL_IDX_CACHE: dict = {}



@dataclass(frozen=True)
class ReplayPolicy:
    on: bool = False
    w: int = 0
    cut: int = 0
    force: bool = False
    source: str = "off"

    @classmethod
    def from_env(cls, cfg, default_cut: int, default_w: int) -> "ReplayPolicy":
        """`DS41M_DECODER_REPLAY`：1/on/true，或一个 JSON 文件路径（热切 A/B）。

        JSON 字段 {on,w,cut,force}，与官方 `DS41L_DECODER_REPLAY` 完全一致。
        """
        raw = os.environ.get("DS41M_DECODER_REPLAY", "")
        on = raw.strip().lower() in ("1", "on", "true", "yes")
        w = 0
        cut = 0
        force = False
        src = "env" if on else "off"
        if not on and raw and os.path.isfile(raw):
            try:
                with open(raw, encoding="utf-8") as fh:
                    j = json.load(fh)
                on = bool(j.get("on", True))
                w = int(j.get("w", 0))
                cut = int(j.get("cut", 0))
                force = bool(j.get("force", False))
                src = raw
            except Exception as exc:                       # noqa: BLE001
                print(f"  [replay] 配置读取失败（按关处理）：{type(exc).__name__}: {exc}",
                      flush=True)
                on = False
                src = "off(bad-json)"
        return cls(on=on, w=int(w or default_w), cut=int(cut or default_cut),
                   force=force, source=src)

    def describe(self, n_layers: int, T: int) -> Optional[str]:
        if not self.on:
            return None
        if self.cut >= n_layers:
            return (f"  [replay] 开启（cut={self.cut} ≥ 实际层数 {n_layers}：全程不裁，"
                    f"等价回归路径）")
        note = "；W≥本轮 T → 不裁，等价回归路径" if self.w >= T else ""
        return (f"  [replay] 开启：层 {self.cut}..{n_layers - 1} 只跑最后 W={self.w} 个 token"
                f"（本轮 T={T}，逻辑位置从 L-W 起算{note}）")


@dataclass
class TrimResult:
    applied: bool = False
    x: Any = None
    input_ids: Any = None
    pos_off: int = 0
    pos_offs: list = field(default_factory=list)   # 逐序列起始位置（packed 多批时）
    n_cut: int = 0
    packed: bool = False
    dropped: int = 0
    message: str = ""
    extra: dict = field(default_factory=dict)


def _trim_rows(t, index):
    """按行取（packed）；index 是 device long 张量。"""
    return t.index_select(0, index).contiguous()


def _trim_shared(shared, index, T_full, packed: bool, W: int) -> int:
    """把 shared 里 dim0 == 原 T 的行账/随行状态切到尾部；返回切了几个。"""
    if not isinstance(shared, dict):
        return 0
    n = 0
    for k in [k for k in shared
              if isinstance(k, str) and k.startswith(ROW_ACCOUNT_PREFIXES)]:
        t = shared.get(k)
        if hasattr(t, "dim") and getattr(t, "dim")() >= 1 and t.shape[0] == T_full:
            shared[k] = _trim_rows(t, index) if packed else t[-W:].contiguous()
            n += 1
    return n


def apply(policy: ReplayPolicy, *, x, input_ids, shared, layer_idx: int, pos_off: int,
          packed_ctx: Optional[dict], pos_offs=None) -> TrimResult:
    """在 `layer_idx >= policy.cut` 的层入口调用一次；未到 cut 时返回 applied=False。

    packed_ctx：runner 的 `cfg["_packed"]`（含 cu/lens/n）；裁剪会**原地**改写它的 cu。
    pos_offs：逐序列起始位置（packed 多批；单序列时为空列表）。
    """
    pos_offs = list(pos_offs or [])
    if not policy.on or layer_idx < policy.cut:
        return TrimResult(x=x, input_ids=input_ids, pos_off=pos_off, pos_offs=pos_offs)

    pn = int((packed_ctx or {}).get("n", 1) or 1)
    lens = [int(v) for v in ((packed_ctx or {}).get("lens") or [])]
    # ★★ 2026-10-20（packed-hit）：尾段几何必须用**本 pass 输入行数**（cu 差分），
    #   不能用逻辑长 lens。hit 时每序列只喂 R 行（R<L），pos_offset 的增量是 R-W，
    #   不是 L-W。用 lens 会把 pos_off 推到 2L-R-W（实测 4k@0.96：7808 而不是 3968）
    #   → 压缩池按错位置开/写、索引越界（MqsmlaKernel MTE trap，层 31）。
    #   plain packed 时 rows == lens，与旧行为逐位相同。
    _cu = (packed_ctx or {}).get("cu") if isinstance(packed_ctx, dict) else None
    rows = ([int(_cu[i + 1]) - int(_cu[i]) for i in range(len(_cu) - 1)]
            if _cu and len(_cu) >= pn + 1 else [])
    packed_tail = pn > 1 and rows and policy.w < min(rows)
    single_tail = pn <= 1 and policy.w < x.shape[0]
    if not (packed_tail or single_tail):
        return TrimResult(x=x, input_ids=input_ids, pos_off=pos_off, pos_offs=pos_offs)

    T_full = x.shape[0]
    pm = shared.get("_ffn_pre") if isinstance(shared, dict) else None

    if packed_tail:
        # 逐请求取自己的尾部 W 行（不是全局尾 W 行）；cu 重登记，seq_lens 保持原始 S。
        _ck = None
        _hit = None
        if _ENV_TAIL_IDX_CACHE:
            _ck = (tuple(int(v) for v in (packed_ctx.get("cu") or ())),
                   int(policy.w), str(x.device))
            _hit = _TAIL_IDX_CACHE.get(_ck)
        if _hit is not None:
            idx, new_cu, idx_host = _hit            # 点位不变量：稳态零阻塞 D2H
        else:
            idx, new_cu = batch_layout.tail_slice(
                {"cu_seqlens": packed_ctx.get("cu"), "n_seqs": pn,
                 "seq_lens": packed_ctx.get("lens")}, policy.w, x.device)
            idx_host = idx.cpu().tolist()
            if _ENV_TAIL_IDX_CACHE and _ck is not None:
                _TAIL_IDX_CACHE.clear()
                _TAIL_IDX_CACHE[_ck] = (idx, new_cu, idx_host)
        x = _trim_rows(x, idx)
        if input_ids is not None and len(input_ids) >= T_full:
            input_ids = [input_ids[i] for i in idx_host]
        if pm is not None and pm.shape[0] == T_full:
            shared["_ffn_pre"] = _trim_rows(pm, idx)
        n_cut = _trim_shared(shared, idx, T_full, True, policy.w)
        if packed_ctx is not None:
            packed_ctx["cu"] = new_cu
        S0 = rows[0]                                   # 本段输入行数（不是逻辑长 lens[0]）
        # ★ 2026-10-21 逐序列几何：每请求起始位置各自前移 (R_i - W)
        if pos_offs and len(pos_offs) == len(rows):
            pos_offs = [int(pos_offs[i]) + int(rows[i]) - policy.w for i in range(len(rows))]
            pos_off = max(pos_offs) if pos_offs else pos_off
        else:
            pos_offs = []
            pos_off += S0 - policy.w
        msg = (f"  [replay] 层 {layer_idx} 起裁剪（packed {pn} 条）：每请求尾 {policy.w} 行，"
               f"T {T_full} → {x.shape[0]}，逻辑位置 "
               + (("逐序列 " + ", ".join(f"{o}..{o + policy.w - 1}" for o in pos_offs))
                  if pos_offs else f"{pos_off}..{pos_off + S0 - 1}")
               + ("（随行 pre_mix 同步裁剪 ✓）" if pm is not None else "")
               + (f"（索引行账 ×{n_cut} 同步裁剪 ✓）" if n_cut else ""))
        return TrimResult(applied=True, x=x, input_ids=input_ids, pos_off=pos_off,
                          pos_offs=pos_offs, n_cut=n_cut, packed=True,
                          dropped=S0 - policy.w, message=msg)

    # 单序列：全局尾 W 行
    drop = T_full - policy.w
    x = x[-policy.w:].contiguous()
    if input_ids is not None and len(input_ids) > policy.w:
        input_ids = input_ids[-policy.w:]
    if pm is not None and pm.shape[0] > policy.w:
        shared["_ffn_pre"] = pm[-policy.w:].contiguous()
    n_cut = _trim_shared(shared, None, T_full, False, policy.w)
    # ★ 2026-10-21（packed b=1 根因）：b=1 的 packed 也走本分支 —— cu 必须同步登记成
    #   **行边界**。以前只有 pn>1 的 packed_tail 才改写 cu；b=1 时 cu 仍是全长 L
    #   → QSLI 的 metadata AICPU 按 L 行调度、实际只有 W 行 → AICPU exception 507018
    #   （实测 cu=[0,2048] vs T=128）。单请求的行边界就是 [0, W]。
    if isinstance(packed_ctx, dict) and packed_ctx.get("cu"):
        packed_ctx["cu"] = [0, x.shape[0]]
    pos_off += drop
    msg = (f"  [replay] 层 {layer_idx} 起裁剪：T {T_full} → {x.shape[0]}，"
           f"逻辑位置 {pos_off}..{pos_off + x.shape[0] - 1}"
           + ("（随行 pre_mix 同步裁剪 ✓）" if pm is not None else "")
           + (f"（索引行账 ×{n_cut} 同步裁剪 ✓）" if n_cut else ""))
    if pos_offs and len(pos_offs) == 1:
        pos_offs = [int(pos_offs[0]) + drop]
    return TrimResult(applied=True, x=x, input_ids=input_ids, pos_off=pos_off,
                      pos_offs=pos_offs, n_cut=n_cut, packed=False,
                      dropped=drop, message=msg)

# ── 点位边界（core/memlife.py）：尾段索引缓存（按 cu/T/W）──
POINT_CACHES = (_TAIL_IDX_CACHE,)

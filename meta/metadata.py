# coding=utf-8
"""InputMetadata —— 一次 forward 的**唯一**元数据（2026-10-26，对齐 DeepInfer 分层）。

设计目标（用户口径）：推理 = ①构造 metadata → ②按 metadata 决定并行策略/batch/命中 →
③按 metadata 调计算（或让计算内核按 metadata 规则做）。

现状与缺口：
  · 现在我们只有**碎片**：`batch.py`（cu/seq_lens/row_lens）、runner 里临时拼的 `ctx`
    （T/pos_offset/ratio/hit_stage/is_candidate…）、`cp_state`（CP 计划）、`replay.py`
    （裁剪 cut/w）、`kv_manager`（池几何）——同一条几何信息在多处重复推导，任一处
    口径变了其余不会自动跟随（今天修过的 `_packed` 必填、seq_k=off+T 都是这个病）。
  · 本模块把"这段/这层的几何与策略"收敛成一个对象；上层只填它，下层只读它。

字段分三层（照 DeepInfer `input_metadata` / `model_metadata` 的拆法）：
  1) **BatchLayout**：n / cu / seq_lens / row_lens（tokenwise 打包；b=1 也是 packed）
  2) **Geometry**：L_full / T / pos_offset / ratio / n_layers / cut / W（replay 裁剪）
  3) **Policy**：parallel（ep/tp/cp/dp）、hit_stage（A/B/none）、replay on/off、
     graph on/off、backend（moe/idx）
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

_META_TRACE_ON = os.environ.get("DS41M_META_TRACE", "").strip().lower() in (
    "1", "on", "true", "yes")
# ★ 过渡口：允许 `cfg["hit"]` 绕过 set_hit 生效（默认**不允许**，见 `hit_of` 的漂移报错）
_HIT_DRIFT_OK = os.environ.get("DS41M_HIT_DRIFT_OK", "").strip().lower() in (
    "1", "on", "true", "yes")


@dataclass
class BatchLayout:
    n: int = 1                     # 请求数（b=1 也是 1 个 packed 序列）
    cu: List[int] = field(default_factory=list)          # 行边界（长度 n+1）
    seq_lens: List[int] = field(default_factory=list)    # 每序列逻辑长度 L_i
    row_lens: List[int] = field(default_factory=list)    # 每序列本次要算的行数 R_i

    @property
    def T(self) -> int:
        return self.cu[-1] if self.cu else sum(self.row_lens or [0])


@dataclass
class Geometry:
    L_full: int = 0                # 逻辑全序列长
    T: int = 0                     # 本 pass 行数（= layout.T）
    pos_offset: int = 0            # 段起点（命中 stage B / decode）
    ratio: int = 1                 # 压缩比（逐层复制）
    cut: Optional[int] = None      # replay 裁剪起始层
    W: int = 0                     # 裁剪后每序列行数


@dataclass
class Policy:
    ep: int = 1
    tp: int = 1
    cp: int = 1
    dp: int = 1
    hit_stage: str = "none"        # none / A / B
    replay: bool = False
    graph: bool = False            # 2026-10-26 起默认关
    moe_backend: str = "deepgemm"  # deepgemm / ascendc
    idx_variant: str = "varlen"    # varlen（唯一口径；CP 内部暂走分块）


@dataclass
class HitPolicy:
    """命中口径（C5 收尾：从 `cfg["hit"]` 搬进 metadata 的**单源**）。

    `ratio` / `non_cached_tokens` / `replay_tail` / `alignment` / `source` 全部在这里；
    `cfg["hit"]` 只作为**镜像**（老代码仍读它），由 `set_hit`/`attach` 同步。
    """
    ratio: float = 0.0
    non_cached_tokens: int = 0
    replay_tail: int = 0
    alignment: str = ""            # ★ 空 = "未显式指定"，让 DS41M_HIT_ALIGN 仍能生效
    source: str = "fake"

    def to_dict(self) -> dict:
        return {"ratio": float(self.ratio),
                "non_cached_tokens": int(self.non_cached_tokens),
                "replay_tail": int(self.replay_tail),
                "alignment": str(self.alignment),
                "source": str(self.source)}

    def is_on(self) -> bool:
        return float(self.ratio) > 0 or int(self.non_cached_tokens) > 0


@dataclass
class InputMetadata:
    layout: BatchLayout = field(default_factory=BatchLayout)
    geom: Geometry = field(default_factory=Geometry)
    policy: Policy = field(default_factory=Policy)
    hit: HitPolicy = field(default_factory=HitPolicy)

    # ---- 便捷视图（下层只读这些，不再各自从 cfg 推导）----
    @property
    def T(self) -> int:
        return self.geom.T or self.layout.T

    def layer_T(self, idx: int) -> int:
        """某层本 pass 的行数：cut 之后按 W×n（裁剪），否则全量。"""
        if self.geom.cut is not None and idx >= self.geom.cut:
            return int(self.geom.W) * max(1, self.layout.n)
        return self.T

    def to_packed(self) -> dict:
        """兼容旧接口：`cfg["_packed"]` 的等价 dict。"""
        # 兼容旧 key：`lens` 是历史字段名（runner/points_runner 仍按它读）
        return {"n": self.layout.n, "cu": list(self.layout.cu),
                "lens": list(self.layout.seq_lens),
                "seq_lens": list(self.layout.seq_lens),
                "row_lens": list(self.layout.row_lens)}

    def as_dict(self) -> dict:
        return asdict(self)

    # ---- 构造 ----
    @classmethod
    def from_cfg(cls, cfg: dict) -> "InputMetadata":
        run = cfg.get("run") or {}
        pk = cfg.get("_packed") or {}
        feat = cfg.get("features") or {}
        hit = cfg.get("hit") or {}
        n = int(pk.get("n", 1) or 1)
        rows = [int(x) for x in (pk.get("row_lens") or [])] or [int(run.get("seq_len", 0))]
        lens = [int(x) for x in (pk.get("lens") or [])] or [int(run.get("seq_len", 0))]
        cu = [int(x) for x in (pk.get("cu") or [])]
        if not cu:
            cu, acc = [0], 0
            for r in rows:
                acc += int(r); cu.append(acc)
        cp = int(run.get("cp", 1) or 1)
        world = int(run.get("world", 8) or 8)
        try:
            import policy as _policy          # 规则来自 configs/policy.yaml + env
            _pol = _policy.resolve(cfg)
        except Exception:
            _pol = None
        # ★ 2026-11-01（C5 收尾）：命中策略进 metadata（单源），`cfg["hit"]` 继续作镜像。
        _hp = HitPolicy(ratio=float(hit.get("ratio", 0.0) or 0.0),
                        non_cached_tokens=int(hit.get("non_cached_tokens", 0) or 0),
                        replay_tail=int(hit.get("replay_tail", 0) or 0),
                        alignment=str(hit.get("alignment") or "token"),
                        source=str(hit.get("source") or "fake"))
        if _pol is not None:
            return cls(layout=BatchLayout(n=n, cu=cu, seq_lens=lens, row_lens=rows),
                       geom=Geometry(L_full=int(run.get("seq_len", 0) or 0), T=cu[-1],
                                     pos_offset=int(run.get("pos_offset", 0) or 0),
                                     ratio=int(run.get("compress_ratio", 1) or 1)),
                       policy=_pol, hit=_hp)
        return cls(
            layout=BatchLayout(n=n, cu=cu, seq_lens=lens, row_lens=rows),
            geom=Geometry(L_full=int(run.get("seq_len", 0) or 0), T=cu[-1],
                          pos_offset=int(run.get("pos_offset", 0) or 0),
                          ratio=int(run.get("compress_ratio", 1) or 1)),
            policy=Policy(cp=cp, ep=world // max(1, cp) if world >= cp else 1,
                          tp=1, dp=world // max(1, cp) if world >= cp else world,
                          hit_stage=("none" if not hit.get("ratio") else "A"),
                          replay=bool(feat.get("decoder_replay")),
                          graph=False,
                          moe_backend=str((cfg.get("backends") or {}).get("megamoe", "deepgemm"))),
            hit=_hp,
        )


def attach(cfg: dict) -> InputMetadata:
    """把 metadata 挂到 cfg（幂等）；上层构造一次，下层只读。"""
    meta = InputMetadata.from_cfg(cfg)
    cfg["_meta"] = meta.as_dict()
    cfg["_meta_obj"] = meta            # 运行时对象（dry-run/工具用）
    return meta


# ── 兼容层：`_packed` 从"源"降级为 metadata.layout 的**镜像**（2026-10-26）──
def layout_of(cfg: dict) -> "BatchLayout":
    """取本次 forward 的 batch 布局（唯一来源 = InputMetadata）。"""
    m = cfg.get("_meta_obj")
    if m is None:
        m = attach(cfg)
    return m.layout


def set_layout(cfg: dict, **kw) -> "BatchLayout":
    """改布局（CP/replay 会改 cu/row_lens）；同步镜像并返回。"""
    lay = layout_of(cfg)
    for k, v in kw.items():
        setattr(lay, k, v)
    cfg["_meta"] = cfg["_meta_obj"].as_dict()
    cfg["_packed"] = cfg["_meta_obj"].to_packed()      # 镜像（兼容旧读点）
    return lay


def ensure(cfg: dict) -> "InputMetadata":
    """确保 cfg 带 metadata（并从它生成 `_packed` 镜像）。"""
    if "_meta_obj" not in cfg:
        attach(cfg)
    cfg["_packed"] = cfg["_meta_obj"].to_packed()
    return cfg["_meta_obj"]


def hit_of(cfg: dict) -> HitPolicy:
    """命中策略的唯一读取入口。

    ★ 两种形态都要正确：
      · `cfg` 只是 `{"hit": {...}}` 的最小 dict（`points_runner._m_of`、`assemble.hit_m`
        的调用方会这样传）—— 现场构造一个 `HitPolicy` 返回，**不写回**；
      · `cfg` 有 `_meta_obj` —— 以 `cfg["hit"]`（**输入侧仍是最新的**）为准**即时刷新**
        `m.hit` 再返回。为什么刷新：大量代码仍是整块 `cfg["hit"] = dict(...)` 赋值，
        若直接读缓存对象就会拿到陈旧值（那会变成静默的行为改变）。
        等写入侧全部改成 `set_hit` 之后，可以把这个刷新去掉、翻转优先级。
    """
    m = cfg.get("_meta_obj")
    h = cfg.get("hit")
    if m is None:
        h = h or {}
        return HitPolicy(ratio=float(h.get("ratio", 0.0) or 0.0),
                         non_cached_tokens=int(h.get("non_cached_tokens", 0) or 0),
                         replay_tail=int(h.get("replay_tail", 0) or 0),
                         alignment=str(h.get("alignment") or ""),
                         source=str(h.get("source") or "fake"))
    if isinstance(h, dict):
        _before = m.hit.to_dict()
        _now = HitPolicy(
            ratio=float(h.get("ratio", _before["ratio"]) or 0.0),
            non_cached_tokens=int(h.get("non_cached_tokens",
                                         _before["non_cached_tokens"]) or 0),
            replay_tail=int(h.get("replay_tail", _before["replay_tail"]) or 0),
            alignment=str(h.get("alignment") or _before["alignment"] or ""),
            source=str(h.get("source") or _before["source"] or "fake"),
        )
        _after = _now.to_dict()
        # ★ C5 收尾（2026-11-01 20:xx）：写入侧已全部收敛到 set_hit/set_hit_dict
        #   （`tools/test_meta_single_source.py` 守着"cfg['hit'] 直写只允许出现在 metadata"）。
        #   所以这里**不再静默刷新**：若检出 `cfg["hit"]` 与 `metadata.hit` 不一致，
        #   说明有人绕过 set_hit 直接改配置 —— 按铁律 0 **显式报错**（不许静默回退）。
        #   调试/过渡期可用 `DS41M_HIT_DRIFT_OK=1` 显式放行（会打印告警）。
        if _before != _after:
            _msg = (f"[hit-drift] cfg['hit'] 与 metadata.hit 不一致："
                    f"{_before} → {_after}（写入应走 metadata.set_hit/set_hit_dict）")
            if _META_TRACE_ON:
                import sys as _s
                _s.stderr.write("  " + _msg + "\n")
            if not _HIT_DRIFT_OK:
                raise RuntimeError(_msg + "；如确需放行请设 DS41M_HIT_DRIFT_OK=1")
            m.hit = _now                      # 显式放行：按 cfg 覆盖（旧行为）
    return m.hit


def set_hit(cfg: dict, **kw) -> HitPolicy:
    """改命中策略（同步 `cfg["hit"]` 镜像）。"""
    hp = hit_of(cfg)
    if cfg.get("_meta_obj") is None:
        hp = HitPolicy(**hp.to_dict())
        cfg["_meta_obj"] = InputMetadata(hit=hp)
        cfg["_packed"] = cfg["_meta_obj"].to_packed()
    for k, v in kw.items():
        if k not in hp.to_dict():
            raise ValueError(f"未知的 hit 字段：{k}")
        setattr(hp, k, v)
    cfg["hit"] = hp.to_dict()
    return hp


def set_hit_dict(cfg: dict, d: dict) -> HitPolicy:
    """按**整块 dict** 改命中策略（写入侧收敛用）。

    ★ 2026-11-01（C5 收尾）：仓库里仍有 `cfg["hit"] = dict(base_hit, …)` 的整块赋值。
    这里把它收敛成一个入口：
      · 已知字段（ratio/non_cached_tokens/replay_tail/alignment/source）写进 `HitPolicy`；
      · **未知字段原样保留在 `cfg["hit"]` 镜像里**（老代码可能读它们，不能悄悄丢）；
      · 返回更新后的 `HitPolicy`。
    """
    d = dict(d or {})
    hp = hit_of(cfg)
    for k in ("ratio", "non_cached_tokens", "replay_tail", "alignment", "source"):
        if k in d:
            if k == "alignment":
                setattr(hp, k, str(d[k] or ""))
            elif k == "source":
                setattr(hp, k, str(d[k] or "fake"))
            elif k == "ratio":
                setattr(hp, k, float(d[k] or 0.0))
            else:
                setattr(hp, k, int(d[k] or 0))
    m = cfg.get("_meta_obj")
    if m is None:
        m = InputMetadata(hit=hp)
        cfg["_meta_obj"] = m
        cfg["_packed"] = m.to_packed()
    m.hit = hp
    _mir = hp.to_dict()
    _mir.update({k: v for k, v in d.items() if k not in _mir})
    cfg["hit"] = _mir
    return hp


def packed_view(cfg: dict) -> dict:
    v = cfg.get("_packed")
    return v if isinstance(v, dict) else {}


# ── ★ 2026-11-01（C5 第一小步）：**有作用域**的布局覆盖 ────────────────────────
#   以前 compact/replay 直接 `set_layout()` 改配置里的持久布局，谁忘了还原就会漏到
#   下一个 pass（实测：b=2 的第二轮在 L0 就按 512/请求 摆行 ⇒ moe gate 吃 nan）。
#   runner 当时用「入口存档、出口还原」手写兜住；这里把同一语义做成**唯一入口**：
#     · `snapshot_layout / restore_layout`：显式存档/还原（runner 用）；
#     · `stage_view(**kw)`：上下文管理器，进块改、出块必还原（新代码一律用它）。
#   语义上这就是"不可变 stage view"的雏形：**布局只在作用域内可见**，越界即失效。
_SNAP_FIELDS = ("cu", "row_lens", "seq_lens", "n")


def snapshot_layout(cfg: dict):
    """存档当前 layout 的可变字段；没有 metadata 时返回 None（调用方按"无存档"处理）。"""
    m = cfg.get("_meta_obj")
    if m is None:
        return None
    lay = m.layout
    return {k: (int(getattr(lay, k)) if k == "n" else list(getattr(lay, k) or []))
            for k in _SNAP_FIELDS}


def restore_layout(cfg: dict, snap) -> bool:
    """还原到存档；返回**是否真的改过**（诊断用，便于发现漏还原）。"""
    if not snap:
        return False
    lay = layout_of(cfg)
    changed = any((int(getattr(lay, k)) if k == "n" else list(getattr(lay, k) or [])) != snap[k]
                  for k in _SNAP_FIELDS)
    if changed:
        set_layout(cfg, **{k: snap[k] for k in _SNAP_FIELDS})
    return changed


@contextlib.contextmanager
def stage_view(cfg: dict, **kw):
    """作用域内覆盖布局（`cu/row_lens/seq_lens/n`），退出时**必定**还原。"""
    snap = snapshot_layout(cfg)
    try:
        if kw:
            set_layout(cfg, **kw)
        yield layout_of(cfg)
    finally:
        restore_layout(cfg, snap)


# ── ★ 2026-11-01（C5 第二步）：栈式 push/pop ─────────────────────────────────
#   `stage_view` 适合"进入一段代码就改、出来就还原"；但 compact/replay 的布局变化
#   发生在**层循环中间**、一直持续到 pass 结束。用显式栈表达：
#     · `push_layout(cfg, **kw)`：把当前值压栈后再应用（返回栈深）；
#     · `pop_layout(cfg)`：出栈还原（返回是否真的改过）；
#     · `pop_all_layouts(cfg)`：把本 pass 压的都还原（出口用，保证不漏）。
#   好处：漏还原在结构上不可能（除非有人忘了 pop_all，而 pop_all 在出口无条件调用）。
_STACK_KEY = "_lay_stack"


def push_layout(cfg: dict, **kw) -> int:
    snap = snapshot_layout(cfg)
    stk = cfg.setdefault(_STACK_KEY, [])
    stk.append(snap)
    if kw:
        set_layout(cfg, **kw)
    return len(stk)


def pop_layout(cfg: dict) -> bool:
    stk = cfg.get(_STACK_KEY) or []
    if not stk:
        return False
    snap = stk.pop()
    return restore_layout(cfg, snap)


def pop_all_layouts(cfg: dict) -> int:
    n = 0
    while pop_layout(cfg):
        n += 1
    return n


def layout_depth(cfg: dict) -> int:
    return len(cfg.get(_STACK_KEY) or [])

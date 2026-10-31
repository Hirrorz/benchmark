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

from dataclasses import dataclass, field, asdict
from typing import List, Optional


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
class InputMetadata:
    layout: BatchLayout = field(default_factory=BatchLayout)
    geom: Geometry = field(default_factory=Geometry)
    policy: Policy = field(default_factory=Policy)

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
        if _pol is not None:
            return cls(layout=BatchLayout(n=n, cu=cu, seq_lens=lens, row_lens=rows),
                       geom=Geometry(L_full=int(run.get("seq_len", 0) or 0), T=cu[-1],
                                     pos_offset=int(run.get("pos_offset", 0) or 0),
                                     ratio=int(run.get("compress_ratio", 1) or 1)),
                       policy=_pol)
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


def packed_view(cfg: dict) -> dict:
    """`_packed` 的**唯一读取入口**（过渡期=透传；未来可由 InputMetadata 生成后彻底删 key）。"""
    v = cfg.get("_packed")
    return v if isinstance(v, dict) else {}

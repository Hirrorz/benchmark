# coding=utf-8
"""Policy —— 规则/策略的唯一入口（2026-10-26，照 DeepInfer `config.py`/`environ.py`）。

分层：
  · `configs/policy.yaml`：**声明式规则**（并行、MoE 后端、replay、图、索引器变体…）
  · 本模块：加载 + 环境变量覆盖 + 推导（如 dp = ep // cp）+ 类型化环境变量包装
  · `metadata.Policy`：某次 forward 实际生效的**结果**（由本模块 resolve 填充）

原则：代码里不再出现"策略常量"；要改规则改 yaml（或环境变量 A/B）。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import metadata

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 树根（分包后）
DEFAULT_PATH = os.path.join(HERE, "configs", "policy.yaml")


# ── 类型化环境变量（照 DeepInfer EnvVar）──────────────────────────────────
class EnvBool:
    def __init__(self, name: str, default: bool):
        self.name, self.default = name, default

    def get(self) -> bool:
        v = os.environ.get(self.name)
        if v is None or v == "":
            return self.default
        return v.strip().lower() in ("1", "on", "true", "yes")


class EnvInt:
    def __init__(self, name: str, default: int):
        self.name, self.default = name, default

    def get(self) -> int:
        v = os.environ.get(self.name)
        try:
            return int(v) if v not in (None, "") else self.default
        except ValueError:
            return self.default


class EnvStr:
    def __init__(self, name: str, default: str):
        self.name, self.default = name, default

    def get(self) -> str:
        v = os.environ.get(self.name)
        return v if v not in (None, "") else self.default


# 入图（ACL Graph）已永久删除（2026-10-27 用户规则）：无 GRAPH 开关；Policy.graph 恒 False。
REPLAY = EnvBool("DS41M_DECODER_REPLAY", True)
MOE_BACKEND = EnvStr("DS41M_MOEMODE", "")           # 空 = 用 yaml / 模型默认
CP = EnvInt("DS41M_CP_SIZE", 0)                     # 0 = 用 yaml


def load(path: Optional[str] = None) -> Dict[str, Any]:
    """读 policy.yaml；缺文件返回空 dict（走代码兜底）。"""
    p = path or os.environ.get("DS41M_POLICY_YAML") or DEFAULT_PATH
    try:
        import yaml
        with open(p) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def resolve(cfg: Optional[dict] = None, pol: Optional[Dict[str, Any]] = None) -> metadata.Policy:
    """把「yaml 规则 + 环境变量覆盖 + 推导」落成某次 forward 的 Policy。"""
    pol = load() if pol is None else pol
    par = (pol.get("parallel") or {})
    moe = (pol.get("moe") or {})
    rep = (pol.get("replay") or {})
    gra = (pol.get("graph") or {})
    idx = (pol.get("indexer") or {})

    cp = CP.get() or int(par.get("cp", 1) or 1)
    ep = int(par.get("ep", 8) or 8)
    dp_cfg = par.get("dp", "auto")
    dp = (max(1, ep // cp) if str(dp_cfg) == "auto" else int(dp_cfg))
    backend = MOE_BACKEND.get() or str(moe.get("backend", "deepgemm"))
    return metadata.Policy(
        ep=ep, tp=int(par.get("tp", 1) or 1), cp=cp, dp=dp,
        hit_stage=str(((cfg or {}).get("hit") or {}).get("stage", "none")),
        replay=REPLAY.get(), graph=False,
        moe_backend=backend,
        idx_variant=str(idx.get("variant", "varlen")),
    )


def rule(path: str, default: Any = None) -> Any:
    """按点号取规则值，如 `rule("graph.allow_if.max_T")`。"""
    node: Any = load()
    for k in path.split("."):
        if not isinstance(node, dict):
            return default
        node = node.get(k, None)
        if node is None:
            return default
    return node

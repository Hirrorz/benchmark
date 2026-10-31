"""上下文键的规范命名空间——**配置驱动**（2026-10-25 用户要求）。

唯一事实源 = `modules/specs/key_names.yaml`（可用 `DS41M_KEY_NAMES` 覆盖）：
  · `legacy`    : 原始键 → 规范键（`H` → `model.hidden` …）；
  · `overloads` : 同名多义的说明（`position.offset` 等）；
  · `tiers`     : 命名空间默认 tier（model→const、geometry→pass、cache→cache…）。

文件缺失/解析失败（`DS41M_SPEC_STRICT=0`）才回退 `modules/names_builtin.py`。
本模块不依赖 torch；加载在 import 期完成一次。

命名空间（第一个点之前）：
  model.*     模型常量（hidden/head_dim/ratio 的静态部分）
  geometry.*  本次 forward 的几何（行数/全长/分块）
  position.*  位置语义（rope/窗口/命中起点/decode 位置）
  batch.*     packed 批次（cu/seq_lens/row_lens/n_seqs）
  layer.*     层身份与角色
  hidden.*    残差流（x/residual）
  attn.*      注意力激活与中间量
  cmp.*       压缩器激活
  idx.*       索引器激活
  moe.*       MoE 激活与路由
  engram.*    engram 激活
  hc.*        hyper-connection 瞬态
  cp.*        上下文并行（本地/全局位置）
  cache.*     跨层共享池与行账（不走词法作用域）
  process.*   进程级（device/src/backends）
  external.*  进程级全局（CP_STATE 等）
  input.*     输入（input_ids）
"""
from __future__ import annotations

import os
import sys

SCHEMA = "ds41_mini/key-names@1"
DEFAULT_SPEC = os.path.join(os.path.dirname(__file__), "specs", "key_names.yaml")
SPEC_PATH = os.environ.get("DS41M_KEY_NAMES") or DEFAULT_SPEC


def _load_builtin():
    from modules import names_builtin as b
    return b.LEGACY, b.OVERLOADS, b._TIER_BY_PREFIX


def _load_yaml(path: str):
    import yaml
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: 顶层必须是映射")
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema={doc.get('schema')!r}，期望 {SCHEMA!r}")
    legacy = doc.get("legacy") or {}
    over = doc.get("overloads") or {}
    tiers = doc.get("tiers") or {}
    for name, d in (("legacy", legacy), ("overloads", over), ("tiers", tiers)):
        if not isinstance(d, dict):
            raise ValueError(f"{path}: {name} 必须是映射")
    return ({str(k): str(v) for k, v in legacy.items()},
            {str(k): tuple(v) for k, v in over.items()},
            {str(k): str(v) for k, v in tiers.items()})


def _load():
    strict = os.environ.get("DS41M_SPEC_STRICT", "1").strip().lower() not in (
        "0", "off", "false", "no")
    if SPEC_PATH and os.path.exists(SPEC_PATH):
        try:
            return _load_yaml(SPEC_PATH)
        except Exception as exc:                         # noqa: BLE001
            if strict:
                raise
            print(f"  ⚠️ [key-names] {SPEC_PATH} 解析失败，回退 Python 表："
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    else:
        print(f"  ⚠️ [key-names] 未找到 {SPEC_PATH}，回退 Python 表",
              file=sys.stderr, flush=True)
    return _load_builtin()


LEGACY, OVERLOADS, _TIER_BY_PREFIX = _load()

# ── 规范名 → 全部 legacy 别名（反向索引，运行期 O(1)）───────────────
ALIASES_BY_CANON: dict[str, tuple[str, ...]] = {}
for _raw, _canon in LEGACY.items():
    ALIASES_BY_CANON.setdefault(_canon, []).append(_raw)
ALIASES_BY_CANON = {k: tuple(v) for k, v in ALIASES_BY_CANON.items()}


def aliases_for(canonical_name: str) -> tuple[str, ...]:
    """规范键对应的所有 legacy 原始键（不含自身；自身由调用方先查）。"""
    return ALIASES_BY_CANON.get(canonical_name, ())


def canonical(key: str) -> str:
    """legacy/规范键 → 规范键。"""
    return LEGACY.get(key, key)


def namespace(key: str) -> str:
    return canonical(key).split(".", 1)[0]


def default_tier(key: str) -> str:
    return _TIER_BY_PREFIX.get(namespace(key), "module")


def is_overloaded(key: str) -> bool:
    return canonical(key) in OVERLOADS


def overload_roles(key: str) -> tuple[str, ...]:
    return OVERLOADS.get(canonical(key), ())

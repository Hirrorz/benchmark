"""分层上下文（Phase 1.5/2 的运行时骨架；纯 Python，无 torch）。

模型（与查看器展示的一致）：
  · `Scope` 是一条作用域链：process → pass → layer → container → leaf；
  · 每个作用域只存**本节点新增/改动的键**（自己的"字典"）；
  · 读到不存在的键 → 自动向最近祖先查询（`resolve` 返回解析到的 provider）；
  · 模块完成后，声明的 `provides` 由调用方 `promote()` 到父作用域，后续兄弟可见；
  · `cache.*` 不走词法链（跨层池由 `CacheIndex` 按"最近前序写者"解析）。

`Ctx` 是 dict 兼容适配器：现有 op 代码的 `ctx.get/[]/in/update/pop` 可零改动跑在
作用域链上（Phase 1.5 接入 runner 时用）。Phase 1 只被自检与未来阶段使用。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from modules import names


class MissingContext(KeyError):
    """查询链上找不到键（带查询轨迹，便于定位幽灵键）。"""

    def __init__(self, key: str, trail: list[str]):
        super().__init__(key)
        self.key = key
        self.trail = trail

    def __str__(self) -> str:
        return f"上下文缺少 {self.key!r}；查询轨迹：{' → '.join(self.trail) or '(空)'}"


@dataclass(frozen=True)
class Provenance:
    provider: str
    tier: str


@dataclass
class Resolution:
    value: object
    prov: Provenance
    status: str = "ok"          # ok|in_place|defaulted|optional


class Scope:
    """一个命名空间节点。`enter` 里的键只在本作用域可见（如容器提供的 hc.stage）。"""

    def __init__(self, name: str, parent: "Scope | None" = None, tier: str = "module"):
        self.name = name
        self.parent = parent
        self.tier = tier
        self.local: dict[str, object] = {}
        self.prov: dict[str, Provenance] = {}

    # ── 写入（只写本地）──────────────────────────────────────────
    def define(self, key: str, value, provider: str | None = None,
               tier: str | None = None) -> None:
        k = names.canonical(key)
        self.local[k] = value
        self.prov[k] = Provenance(provider or self.name, tier or self.tier)

    # ── 读取（本地 → 祖先）───────────────────────────────────────
    def resolve(self, key: str, _trail: list[str] | None = None) -> Resolution | None:
        k = names.canonical(key)
        trail = _trail if _trail is not None else []
        trail.append(self.name)
        if k in self.local:
            p = self.prov.get(k) or Provenance(self.name, self.tier)
            return Resolution(self.local[k], p, "ok")
        if self.parent is not None:
            return self.parent.resolve(k, trail)
        return None

    def get(self, key: str, default=None):
        r = self.resolve(key)
        return default if r is None else r.value

    def require(self, key: str):
        r = self.resolve(key)
        if r is None:
            raise MissingContext(names.canonical(key), [self.name])
        return r.value

    def child(self, name: str, tier: str = "module") -> "Scope":
        return Scope(name, parent=self, tier=tier)

    # ── 模块完成：provides 上浮到父作用域（后续兄弟可见）─────────
    def promote(self, provides, provider: str | None = None,
                shadow: str = "single_writer") -> list[str]:
        """把本模块声明的 provides 发布到父作用域；返回被遮蔽的键列表。"""
        if self.parent is None:
            raise RuntimeError("根作用域没有父级，不能 promote")
        shadowed = []
        for key in provides:
            k = names.canonical(key)
            if k in self.parent.local and shadow not in ("stream", "allow_multi"):
                shadowed.append(k)
            self.parent.define(k, self.local.get(k), provider or self.name,
                               tier=self.tier)
        return shadowed

    def __repr__(self) -> str:
        return f"<Scope {self.name} tier={self.tier} keys={len(self.local)}>"


class Ctx:
    """dict 兼容的上下文视图（读走作用域链，写只写当前作用域）。

    现有 op 代码常见的读取方式都覆盖：
        ctx["T"] / ctx.get("T", d) / "T" in ctx / dict(ctx) / ctx.update(...) / ctx.pop(...)
    规范键与 legacy 别名互通（`names.canonical`）。
    """

    def __init__(self, scope: Scope):
        self.scope = scope

    # ── 读 ──────────────────────────────────────────────────────
    def __getitem__(self, key):
        r = self.scope.resolve(key)
        if r is None:
            raise MissingContext(names.canonical(key), [self.scope.name])
        return r.value

    def get(self, key, default=None):
        return self.scope.get(key, default)

    def __contains__(self, key) -> bool:
        return self.scope.resolve(key) is not None

    def keys(self):
        seen = set()
        s = self.scope
        while s is not None:
            for k in s.local:
                if k not in seen:
                    seen.add(k)
                    yield k
            s = s.parent

    def items(self):
        for k in self.keys():
            yield k, self.scope.get(k)

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.keys())

    def __bool__(self):
        return True

    # ── 写（只写当前作用域）─────────────────────────────────────
    def __setitem__(self, key, value):
        self.scope.define(key, value)

    def update(self, *args, **kw):
        data = dict(*args, **kw)
        for k, v in data.items():
            self.scope.define(k, v)

    def pop(self, key, default=None):
        k = names.canonical(key)
        if k in self.scope.local:
            self.scope.prov.pop(k, None)
            return self.scope.local.pop(k)
        return default


class CanonicalView(Mapping):
    """只读视图：把作用域里可见的键按**规范名**暴露（legacy 别名合并）。

    例：`{"T": 8192, "H": 5120}` → `{"geometry.rows": 8192, "model.hidden": 5120}`。
    `raw(canonical)` 可反查最近的原始键名（排障用）。
    """

    def __init__(self, ctx):
        self.ctx = ctx

    def __getitem__(self, key):
        return self.ctx[key]

    def __iter__(self):
        seen = set()
        for raw in self.ctx.keys():
            c = names.canonical(raw)
            if c not in seen:
                seen.add(c)
                yield c

    def __len__(self):
        return sum(1 for _ in self)

    def raw(self, canonical: str):
        for raw in self.ctx.keys():
            if names.canonical(raw) == canonical:
                return raw
        return None


def self_test() -> None:
    """最小自检：祖先解析 / 别名 / provides 上浮 / 影子检测。"""
    root = Scope("process", tier="const")
    root.define("model.hidden", 5120)
    p = root.child("pass", tier="pass")
    p.define("geometry.rows", 8192)
    layer = p.child("layer", tier="layer")
    layer.define("layer.ratio", 2)
    attn = layer.child("attn", tier="container")
    attn.define("hc.stage", "attn", provider="container:attn", tier="container")
    wq = attn.child("attn.q_proj.wq_a")
    assert wq.get("H") == 5120
    assert wq.get("T") == 8192
    assert wq.get("ratio") == 2
    assert wq.get("_norm_stage") == "attn"
    wq.define("attn.qr", "QR")
    assert wq.promote(["attn.qr", "attn.qr_scale"], shadow="stream") == []
    norm = attn.child("attn.q_proj.q_norm")
    assert norm.get("qr") == "QR"
    # 影子：单写者键被第二次发布
    second = attn.child("attn.other")
    second.define("attn.qr", "QR2")
    assert second.promote(["attn.qr"]) == ["attn.qr"]
    # 缺失键
    try:
        wq.require("cache.nope")
    except MissingContext as e:
        assert "cache.nope" in str(e)
    else:
        raise AssertionError("应当抛 MissingContext")
    # Ctx dict 兼容
    ctx = Ctx(layer)
    assert ctx["T"] == 8192 and ctx.get("nope", 7) == 7 and "H" in ctx
    ctx["custom.k"] = 1
    assert layer.local["custom.k"] == 1
    assert dict(ctx.items())["model.hidden"] == 5120
    # 规范视图：T→geometry.rows / H→model.hidden，且能反查原始键
    view = CanonicalView(ctx)
    assert view["geometry.rows"] == 8192 and view["model.hidden"] == 5120
    assert "geometry.rows" in view and view.raw("geometry.rows") == "geometry.rows"
    assert len(view) >= 4

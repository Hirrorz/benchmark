"""Phase 2 最小模块运行时：由模块树给出层内执行计划（默认关闭）。

定位（与 `MODULE_DEV.md` 一致）：
- **不是**新执行引擎——叶子仍然调用现有 `exec_ops.run_op(key, st, ctx, w)`；
- 本模块只把 `recipes.yaml` → 容器树（`modules.compose`）变成**可执行的叶子序**，
  并在启用时与 legacy `run.expand_layer` 做严格对拍（顺序 / 次数 / 视图）；
- 容器 `enter/exit` 钩子已预留，Phase 2 目前容器不持有代码，钩子默认空。

开关：`DS41M_MODULE_RUNTIME=1`（runner 里接）。DSpark / decode 由 runner 强制 legacy，
不在这里判断。
"""
from __future__ import annotations

from dataclasses import dataclass

from modules import compose, names


class ModuleRuntimeOrderError(RuntimeError):
    """模块树 DFS 序与 legacy 执行序不一致。"""


@dataclass(frozen=True)
class LayerPlan:
    idx: int
    kind: str
    view: str
    ops: tuple[str, ...]
    steps: tuple
    root: dict

    def as_dict(self) -> dict:
        return dict(idx=self.idx, kind=self.kind, view=self.view,
                    ops=list(self.ops), count=len(self.ops))


def is_enabled(env=None) -> bool:
    """默认开（2026-10-24 用户拍板）；`=0/off/false/no` 回退 legacy。"""
    import os
    e = os.environ if env is None else env
    return e.get("DS41M_MODULE_RUNTIME", "1").strip().lower() not in (
        "0", "off", "false", "no")


def build_plan(cfg, recipes, layer, view: str = "default") -> LayerPlan:
    """由 recipes + 层角色装配执行计划；纯 CPU，不碰设备。

    `view="split"` 对应层 cut-1 的 `c1a_source_split`（边界切分变体）。
    """
    if view not in ("default", "split"):
        raise ValueError(f"未知 view: {view!r}")
    comp = compose.compose_layer(cfg, recipes, layer, view)
    ops = tuple(s.op for s in comp.steps)
    return LayerPlan(idx=int(layer["idx"]), kind=str(comp.kind), view=view,
                     ops=ops, steps=tuple(comp.steps), root=comp.tree)


def verify_order(plan: LayerPlan, legacy_ops) -> None:
    """严格对拍：模块树叶子序必须与 legacy 序列逐项相同（含重复次数）。"""
    legacy = list(legacy_ops)
    got = list(plan.ops)
    if got == legacy:
        return
    n = min(len(got), len(legacy))
    for i in range(n):
        if got[i] != legacy[i]:
            raise ModuleRuntimeOrderError(
                f"L{plan.idx} view={plan.view} 第 {i} 项不一致："
                f"module={got[i]!r} legacy={legacy[i]!r}")
    raise ModuleRuntimeOrderError(
        f"L{plan.idx} view={plan.view} 长度不一致："
        f"module={len(got)} legacy={len(legacy)}；"
        f"尾部 module={got[n:n + 4]} legacy={legacy[n:n + 4]}")


def ops_of(plan: LayerPlan) -> list[str]:
    return list(plan.ops)


def walk(plan: LayerPlan, leaf, enter=None, exit=None) -> int:
    """按树 DFS 访问叶子；`leaf(node)` / `enter(node)` / `exit(node)`。

    返回访问到的叶子数。容器回调只在真正进入/离开容器时调用一次，
    与 HTML 里的树形展开一一对应（Phase 2 的扩展点）。
    """
    count = 0

    def _visit(node):
        nonlocal count
        if node.get("kind") == "container":
            if enter is not None:
                enter(node)
            for child in node.get("children") or []:
                _visit(child)
            if exit is not None:
                exit(node)
        else:
            leaf(node)
            count += 1

    _visit(plan.root)
    return count


def signature(plan: LayerPlan) -> str:
    """计划指纹（路径 + 算子 + 行段），用于缓存/诊断，不参与执行。"""
    return "|".join(f"{s.op}@{s.path}@{s.rows_phase}" for s in plan.steps)


# ── 配置参数（modules/specs/*.yaml 的 params）─────────────────────────
def _ctx_getter(ctx):
    """按 raw 名 / 规范名 / legacy 别名在扁平 ctx 里查值（祖先语义）。

    别名走 `names.ALIASES_BY_CANON` 反向索引，避免每个参数都线性扫 LEGACY。
    """
    def _get(key):
        c = names.canonical(key)
        if key in ctx:
            return ctx[key]
        if c in ctx:
            return ctx[c]
        for alias in names.aliases_for(c):
            if alias in ctx:
                return ctx[alias]
        return None
    return _get


def _key_present(ctx, key) -> bool:
    c = names.canonical(key)
    if key in ctx or c in ctx:
        return True
    return any(alias in ctx for alias in names.aliases_for(c))


def params_for(op: str, get) -> dict:
    """解析某模块的配置参数（不写入 ctx）；`get` 通常是 `_ctx_getter(ctx)`。"""
    from modules.op_iface import OP_IFACE
    iface = OP_IFACE.get(op)
    if iface is None or not iface.params:
        return {}
    from modules.spec_loader import resolve_params
    return resolve_params(iface.params, get)


def inject_params(op: str, ctx) -> tuple[str, ...]:
    """执行模块前把配置参数中**运行配置未提供**的键临时注入 ctx。

    返回注入的键元组；调用方必须在 finally 里 `pop` 掉（不污染兄弟模块）。
    优先级：运行配置（cfg→ctx）> 模块规格 YAML 的 params（默认值）。

    ★ T-042（host 瘦身③）：**编译成扁平计划**再逐次执行。
      每次 forward 826 次下发里有 574 次带 params；旧实现每次都重算
      `canonical()` / `aliases_for()` / `resolve_params()`（还会建临时 dict），
      CPU 微基准实测 **2.25 ms/forward**。计划里每个参数只留
      「键 + 候选查找名元组 + 引用候选名（或 None）+ 常量」，剩下每次只是
      几次 dict 成员判断 ⇒ 实测降到 0.35 ms/forward。语义与旧路径逐项相同
      （查找顺序 raw → 规范名 → legacy 别名；引用取不到时抛同样的 KeyError）。
    """
    plan = _INJ_PLAN.get(op)
    if plan is None:
        plan = _compile_params(op)
        if plan is None:                       # 规格里有不认识的写法 → 退回旧实现
            return _inject_params_slow(op, ctx)
        _INJ_PLAN[op] = plan
    if not plan:
        return ()
    injected = []
    for key, kvars, ref_vars, const in plan:
        for vk in kvars:
            if vk in ctx:
                break
        else:                                  # 运行配置没给 → 用规格里的值
            if ref_vars is not None:
                for rk in ref_vars:
                    if rk in ctx:
                        const = ctx[rk]
                        break
                else:
                    raise KeyError(
                        f"params[{key!r}] 引用 {ref_vars[0]!r} 在祖先作用域中不存在"
                        f"（检查 modules/specs 或运行配置）")
            ctx[key] = const
            injected.append(key)
    return tuple(injected)


_INJ_PLAN: dict = {}


def _compile_params(op: str):
    """把 op 的 params 规格编译成注入计划；规格不认识 → None（走旧路径）。

    计划项 `(key, kvars, ref_vars, const)`：
      · kvars   = 判断「运行配置是否已提供该键」要查的名字（raw/规范名/全部别名）；
      · ref_vars= 非 None 时表示该参数是 `${ref}` 引用，值从 ctx 里按这些名字取；
      · const   = 非引用参数的常量值。
    """
    from modules.op_iface import OP_IFACE
    iface = OP_IFACE.get(op)
    if iface is None or not iface.params:
        return ()
    plan = []
    for k, v in iface.params.items():
        ref_vars, const = None, v
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            ref = v[2:-1].strip()
            if not ref:                        # 空引用：交给旧实现抛错
                return None
            c = names.canonical(ref)
            names_seq = [ref] if c == ref else [ref, c]
            names_seq.extend(a for a in names.aliases_for(c) if a not in names_seq)
            ref_vars, const = tuple(names_seq), None
        ck = names.canonical(k)
        kvars = [k] if ck == k else [k, ck]
        kvars.extend(a for a in names.aliases_for(ck) if a not in kvars)
        plan.append((k, tuple(kvars), ref_vars, const))
    return tuple(plan)


def _inject_params_slow(op: str, ctx) -> tuple[str, ...]:
    """旧实现（计划编译失败时的兜底，行为与 T-042 前逐字节相同）。"""
    params = params_for(op, _ctx_getter(ctx))
    injected = []
    for k, v in params.items():
        if _key_present(ctx, k):
            continue
        ctx[k] = v
        injected.append(k)
    return tuple(injected)

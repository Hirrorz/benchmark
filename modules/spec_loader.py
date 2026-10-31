"""模块规格配置加载器（YAML → OpInterface）。

约定（2026-10-25 用户要求）：
  · **模块与其参数不在 Python 里硬编码**；唯一事实源是 `modules/specs/model_modules.yaml`
    （可用 `DS41M_MODULE_SPEC` 指向别的文件）；
  · 每个模块只声明**自己的** requires/provides/weights/params；容器/父节点不汇总子节点
    的 I/O，运行时缺失的键由 `modules.context.Scope` 沿祖先链解析；
  · `params` 支持字面量与 `"${key}"` 引用（`resolve_params` 用祖先作用域求值）。

纯 CPU（只依赖 yaml），不导入 torch/exec_*。
"""
from __future__ import annotations

import os
from typing import Any, Callable

from modules.spec import KeySpec, OpInterface, WeightRef

SCHEMA = "ds41_mini/module-spec@1"
_HERE = os.path.dirname(__file__)
DEFAULT_SPEC = os.path.join(_HERE, "specs", "model_modules.yaml")
# 开发者可把小模块放在 modules.d/*.yaml，不必改 3800 行主文件；
# 同名 op 需显式 `override: true` 才允许覆盖主文件。
DEFAULT_OVERLAY_DIR = os.path.join(_HERE, "specs", "modules.d")

_KEY_FIELDS = {"key", "type", "shape", "storage", "tier", "optional", "mutation",
               "shadow", "condition", "legacy", "roles", "doc"}
_W_FIELDS = {"suffix", "via", "packed_key", "condition", "role", "doc"}
_I_FIELDS = {"op", "impl", "source", "requires", "provides", "weights", "flags",
             "calls", "dispatch", "rows_phase", "doc", "intrinsics", "category",
             "container", "after", "kernel", "module_cls", "params", "override"}


def _err(path, msg):
    raise ValueError(f"{path}: {msg}")


def _check_fields(path, what, d, allowed):
    bad = set(d) - allowed
    if bad:
        _err(path, f"{what} 含未知字段 {sorted(bad)}（允许 {sorted(allowed)}）")


def _key(path, d):
    _check_fields(path, f"key {d.get('key')!r}", d, _KEY_FIELDS)
    if "key" not in d:
        _err(path, f"key 缺少 'key': {d}")
    return KeySpec(
        name=str(d["key"]), type=str(d.get("type", "int")),
        shape=d.get("shape"), storage=d.get("storage", "ctx"),
        tier=d.get("tier", "module"), optional=bool(d.get("optional", False)),
        mutation=d.get("mutation", "new"), shadow=d.get("shadow", "single_writer"),
        condition=d.get("condition"), legacy=tuple(d.get("legacy") or ()),
        roles=tuple(d.get("roles") or ()), doc=d.get("doc", ""),
    )


def _weight(path, d):
    _check_fields(path, f"weight {d.get('suffix')!r}", d, _W_FIELDS)
    if "suffix" not in d:
        _err(path, f"weight 缺少 'suffix': {d}")
    return WeightRef(suffix=str(d["suffix"]), via=d.get("via", "raw"),
                     packed_key=d.get("packed_key"), condition=d.get("condition"),
                     role=d.get("role", "read"), doc=d.get("doc", ""))


def _iface(path, d):
    _check_fields(path, f"module {d.get('op')!r}", d, _I_FIELDS)
    if "op" not in d:
        _err(path, f"module 缺少 'op': {d}")
    src = d.get("source")
    if isinstance(src, dict):
        src = (src.get("file", ""), int(src.get("line", 0)), int(src.get("end_line", 0)))
    params = d.get("params") or {}
    if not isinstance(params, dict):
        _err(path, f"{d['op']}.params 必须是映射，得到 {type(params).__name__}")
    return OpInterface(
        op=str(d["op"]), impl=d.get("impl"), source=src,
        requires=tuple(_key(path, k) for k in (d.get("requires") or ())),
        provides=tuple(_key(path, k) for k in (d.get("provides") or ())),
        weights=tuple(_weight(path, w) for w in (d.get("weights") or ())),
        flags=frozenset(d.get("flags") or ()), calls=int(d.get("calls", 1)),
        dispatch=d.get("dispatch"), rows_phase=d.get("rows_phase", "same"),
        doc=d.get("doc", ""), intrinsics=tuple(d.get("intrinsics") or ()),
        category=d.get("category", "misc"), container=d.get("container"),
        after=tuple(d.get("after") or ()), kernel=d.get("kernel"),
        module_cls=d.get("module_cls"), params=dict(params),
    )


def _read_doc(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        _err(path, "顶层必须是映射")
    return doc


def _entries(path: str, doc: dict, require_schema: bool = False) -> list[dict]:
    """主文件用 `modules: [...]`；modules.d 小文件允许单个 `op:` 映射。

    主配置必须声明 schema；overlay 小文件可省略，但写了就必须匹配。
    """
    schema = doc.get("schema")
    if require_schema and schema != SCHEMA:
        _err(path, f"schema={schema!r}，主配置要求 {SCHEMA!r}")
    if schema is not None and schema != SCHEMA:
        _err(path, f"schema={schema!r}，期望 {SCHEMA!r}")
    if "modules" in doc:
        mods = doc["modules"]
        if not isinstance(mods, list) or not mods:
            _err(path, "modules 必须是非空列表")
        return mods
    if "op" in doc:                       # 单模块小文件
        return [doc]
    _err(path, "需要 `modules:` 列表或单个 `op:` 映射")


def _overlay_dir() -> str | None:
    d = os.environ.get("DS41M_MODULE_SPEC_DIR", DEFAULT_OVERLAY_DIR)
    return d or None


def load_spec_file(path: str | None = None,
                   overlay_dir: str | None = None) -> list[OpInterface]:
    """解析 YAML → 模块接口列表（纯 CPU）。

    `path` 为主配置；`overlay_dir`（默认 `specs/modules.d/`）里每个 `*.yaml`
    可放单个模块或 `modules:` 片段。同名 op 默认报错，需 `override: true` 才覆盖。
    """
    path = path or os.environ.get("DS41M_MODULE_SPEC") or DEFAULT_SPEC
    by_op: dict[str, OpInterface] = {}

    def merge(src: str, entries: list[dict], allow_override: bool) -> None:
        for d in entries:
            if not isinstance(d, dict):
                _err(src, f"modules 项必须是映射：{d!r}")
            op = d.get("op")
            if not op:
                _err(src, "module 缺少 'op'")
            if op in by_op:
                if not (allow_override and d.get("override") is True):
                    _err(src, f"重复模块 {op!r}（覆盖主配置需 `override: true`）")
            by_op[op] = _iface(src, d)

    merge(path, _entries(path, _read_doc(path), require_schema=True),
          allow_override=False)
    odir = overlay_dir if overlay_dir is not None else _overlay_dir()
    explicit_overlay = overlay_dir is not None or "DS41M_MODULE_SPEC_DIR" in os.environ
    if odir and not os.path.isdir(odir):
        # 显式指定的覆盖目录不存在时必须报错：静默退回主配置会让
        # "--out-dir /tmp/x 生成后忘了设 DS41M_MODULE_SPEC_DIR" 的门禁假绿。
        if explicit_overlay:
            _err(odir, "modules.d 覆盖目录不存在（DS41M_MODULE_SPEC_DIR / overlay_dir "
                       "指向了不存在的路径；不报错会静默退回主配置，形成沙盒假绿）")
    elif odir:
        import glob
        for fp in sorted(glob.glob(os.path.join(odir, "*.yaml"))):
            if os.path.basename(fp).startswith("_"):
                continue
            merge(fp, _entries(fp, _read_doc(fp)), allow_override=True)
    return list(by_op.values())


def resolve_params(params: dict, get: Callable[[str], Any]) -> dict:
    """把 `"${key}"` 引用替换为祖先作用域的值（`get(key)` 由 Scope 提供）。

    只解析字符串引用；其余值原样返回。找不到的键抛 KeyError（带键名），
    避免静默用默认值掩盖配置错误。
    """
    out = {}
    for k, v in (params or {}).items():
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            ref = v[2:-1].strip()
            if not ref:
                raise KeyError(f"params[{k!r}] 的空引用")
            val = get(ref)
            if val is None:
                raise KeyError(
                    f"params[{k!r}] 引用 {ref!r} 在祖先作用域中不存在"
                    f"（检查 modules/specs 或运行配置）")
            out[k] = val
        else:
            out[k] = v
    return out

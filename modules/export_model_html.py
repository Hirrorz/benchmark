"""导出单文件离线模型查看器（Phase 1，CPU-only）。

用法：
  python3.11 -m modules.export_model_html --config configs/mini_ds41_flash.yaml \
          --template tools/model_viewer.template.html --out tools/model_viewer.html
  python3.11 -m modules.export_model_html --config ... --check     # 与已提交文件比对
  python3.11 -m modules.export_model_html --config ... --emit-dryrun-json /tmp/spec.json

约定：
  · JSON 注入到模板的 `/* MODEL-DATA:START */ … /* MODEL-DATA:END */` 之间；
  · JSON 排序键 + 紧凑分隔符（两次导出逐字节一致，无时间戳）；
  · 不 import torch / exec_*。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import assemble                                  # noqa: E402
import opgraph                                   # noqa: E402
import run as runmod                             # noqa: E402

from modules import tree as treemod              # noqa: E402
from modules.op_iface import OP_IFACE             # noqa: E402
from modules import shapes as shapemod           # noqa: E402

MARK_START = "/* MODEL-DATA:START */"
MARK_END = "/* MODEL-DATA:END */"


def _sha(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return "missing"


def _git_rev() -> str:
    try:
        out = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "nogit"
    except Exception:                            # noqa: BLE001
        return "nogit"


def _sha_specs() -> str:
    """模块/装配/键名 YAML（含 modules.d）的内容指纹。"""
    root = os.path.join(HERE, "specs")
    h = hashlib.sha256()
    for base, _dirs, names in os.walk(root):
        for n in sorted(names):
            if not n.endswith((".yaml", ".yml")):
                continue
            p = os.path.join(base, n)
            h.update(os.path.relpath(p, root).encode())
            try:
                with open(p, "rb") as f:
                    h.update(f.read())
            except OSError:
                h.update(b"missing")
    return h.hexdigest()[:16]


def fingerprint(cfg_path: str) -> dict:
    return dict(
        config=_sha(cfg_path),
        recipes=_sha(os.path.join(ROOT, "recipes.yaml")),
        opgraph=_sha(os.path.join(ROOT, "opgraph.py")),
        op_iface=_sha(os.path.join(HERE, "op_iface.py")),
        specs=_sha_specs(),
        git=_git_rev(),
    )


def shape_vectors(tbuild, cfg) -> list[dict]:
    """给 JS 求值器的对拍向量：(expr, preset, ratio, rows_phase, expected)。"""
    import model_spec
    spec = model_spec.build(cfg)
    out = []
    seen = set()
    presets = tbuild.tree.presets
    for op in OP_IFACE.values():
        for k in list(op.requires) + list(op.provides):
            if not k.shape:
                continue
            for pname in ("prefill", "prefill_replay", "replay_1024", "cp8_L8192"):
                p = presets[pname]
                idxs = (0, int(p["cut"]) + 1) if p.get("cut") is not None else (0,)
                for phase, ratio in (("same", 1), ("same", 2), ("compact", 1)):
                    if phase == "compact" and pname not in ("prefill", "prefill_replay"):
                        continue
                    for lidx in idxs:
                        key = (k.shape, pname, ratio, phase, lidx)
                        if key in seen:
                            continue
                        seen.add(key)
                        env = treemod.env_for(spec, p, ratio, phase, layer_idx=lidx)
                        try:
                            val = shapemod.eval_shape(k.shape, env)
                        except shapemod.ShapeError as e:
                            val = {"error": str(e)}
                        out.append(dict(expr=k.shape, preset=pname, ratio=ratio,
                                        phase=phase, layer_idx=lidx, expected=val))
    return out


def build_data(cfg_path: str) -> tuple[dict, object]:
    cfg = assemble.load_yaml(cfg_path)
    recipes = runmod.load_recipes()
    tbuild = treemod.build(cfg, recipes)
    data = treemod.to_json(tbuild)
    data["fingerprint"] = fingerprint(cfg_path)
    data["cfg_path"] = os.path.relpath(cfg_path, ROOT)
    data["shape_vectors"] = shape_vectors(tbuild, cfg)
    # 环境开关记录（默认树=全开；被改过的树要显式标注）
    env_flags = {}
    for k in ("DS41M_SKIP_MOE", "DS41M_SKIP_ENGRAM", "DS41M_PAD_OPS",
              "DS41M_BOUNDARY_SPLIT", "DS41M_DECODER_REPLAY"):
        v = os.environ.get(k)
        if v is not None:
            env_flags[k] = v
    data["env_flags"] = env_flags
    return data, (cfg, recipes, tbuild)


def spec_dryrun(cfg, recipes, tbuild) -> dict:
    """模块树 → 与 run.dry_run 同结构的内核直方图（等价性证明）。"""
    stack, meta = assemble.build_stack(cfg)
    layers_out = []
    hist = collections.Counter()
    by_idx = {int(l["idx"]): l for l in stack}
    for row in tbuild.tree.layers:
        layer = by_idx[int(row["idx"])]
        ctx = runmod.ctx_for(cfg, layer)
        counts = collections.Counter()
        comp = tbuild.templates_raw[row["template"]]
        for step in comp.steps:
            spec = opgraph.OPS.get(step.op)
            if spec is None:
                continue
            n = spec["count"](ctx)
            if n:
                counts[spec["kernel"]] += n
        layers_out.append(dict(idx=int(row["idx"]), attn=row["kind"], T=layer["T"],
                               ops=dict(counts)))
        hist.update(counts)
    extras = collections.Counter()
    for name, n in recipes.get("extras") or []:
        spec = opgraph.OPS.get(name)
        if spec is not None:
            extras[spec["kernel"]] += int(n)
    hist.update(extras)
    return dict(meta=meta, layers=layers_out, hist=dict(hist), extras=dict(extras))


def render(template_path: str, data: dict) -> str:
    with open(template_path, encoding="utf-8") as f:
        html = f.read()
    blob = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    blob = blob.replace("</", "<\\/")
    i = html.find(MARK_START)
    j = html.find(MARK_END)
    if i < 0 or j < 0 or j < i:
        raise SystemExit(f"模板缺少注入标记：{MARK_START} … {MARK_END}")
    return html[:i + len(MARK_START)] + "\n" + blob + "\n" + html[j:]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--template", default=os.path.join(ROOT, "tools",
                                                       "model_viewer.template.html"))
    ap.add_argument("--out", default=os.path.join(ROOT, "tools", "model_viewer.html"))
    ap.add_argument("--check", action="store_true", help="只比对，不写文件")
    ap.add_argument("--emit-dryrun-json", default="",
                    help="把模块树的 dry-run 直方图写成 json")
    a = ap.parse_args(argv)

    data, (cfg, recipes, tbuild) = build_data(a.config)
    if a.emit_dryrun_json:
        payload = spec_dryrun(cfg, recipes, tbuild)
        with open(a.emit_dryrun_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True, indent=1)
        print(f"spec dry-run → {a.emit_dryrun_json}")
    html = render(a.template, data)

    if a.check:
        try:
            with open(a.out, encoding="utf-8") as f:
                cur = f.read()
        except OSError:
            raise SystemExit(f"--check 失败：{a.out} 不存在")
        if cur != html:
            raise SystemExit(f"--check 失败：{a.out} 与模板/数据不一致（请重新导出）")
        print(f"--check OK：{a.out}")
        return 0

    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"导出 {a.out}（{len(html) / 1024:.0f} KiB，"
          f"{len(data['layers'])} 层 / {len(data['templates'])} 模板）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

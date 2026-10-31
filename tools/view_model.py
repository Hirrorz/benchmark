#!/usr/bin/env python3



"""终端里的模型结构查看器（SSH 友好；HTML 之外的第二入口）。

    python3.11 tools/view_model.py layers
    python3.11 tools/view_model.py tree 20 --view split --keys
    python3.11 tools/view_model.py op 2 attn.compressor.kv_compress
    python3.11 tools/view_model.py plugin kv_compress
    python3.11 tools/view_model.py catalog --category attention
    python3.11 tools/view_model.py blueprint
    python3.11 tools/view_model.py search kv_compress
    python3.11 tools/view_model.py tui                    # 全屏交互（↑↓/Enter/Tab）
    python3.11 tools/view_model.py serve --port 8765      # 端口转发 + 自动刷新

数据与 HTML 完全同源（`modules.tree`），因此终端里看到的就是代码的投影。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ds41_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    _q = _os.path.join(_ds41_root, _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)

import argparse
import curses
import json
import os
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import assemble                                   # noqa: E402
import model_spec                                 # noqa: E402
import run as runmod                              # noqa: E402

from modules import shapes as shapemod            # noqa: E402
from modules import tree as treemod               # noqa: E402

COLOR = True


# ── 颜色/宽度 ─────────────────────────────────────────────────────
def c(code: str, s: str) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if COLOR else s


def dim(s):   return c("2", s)
def bold(s):  return c("1", s)
def red(s):   return c("31", s)
def green(s): return c("32", s)
def yellow(s):return c("33", s)
def cyan(s):  return c("36", s)
def mag(s):   return c("35", s)

_KIND_COLOR = {
    "win": cyan, "c2a_source": yellow, "c2a_consumer": (lambda s: c("34", s)),
    "c1a_source": mag, "c1a_consumer": (lambda s: c("95", s)),
    "c1a_indexsource": green, "c1a_source_split": (lambda s: c("91", s)),
    "unfused": dim,
}


def kind_str(kind: str) -> str:
    f = _KIND_COLOR.get(kind, lambda s: s)
    return f(kind)


def dwidth(s: str) -> int:
    w = 0
    for ch in s:
        if unicodedata.combining(ch):
            continue
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def truncate_w(s: str, width: int) -> str:
    if dwidth(s) <= width:
        return s
    out, w = [], 0
    for ch in s:
        cw = 0 if unicodedata.combining(ch) else (2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1)
        if w + cw > width - 1:
            break
        out.append(ch)
        w += cw
    return "".join(out) + "…"


def pad_w(s: str, width: int) -> str:
    return s + " " * max(0, width - dwidth(s))


# ── 数据 ──────────────────────────────────────────────────────────
def load(cfg_path: str):
    cfg = assemble.load_yaml(cfg_path)
    recipes = runmod.load_recipes()
    tb = treemod.build(cfg, recipes)
    data = treemod.to_json(tb)
    spec = model_spec.build(cfg)
    return dict(cfg=cfg, recipes=recipes, tb=tb, data=data, spec=spec,
                preset=next(iter(data["geometry_presets"])))


def layer_of(ctx, idx: int) -> dict | None:
    return next((l for l in ctx["data"]["layers"] if int(l["idx"]) == int(idx)), None)


def template_for(ctx, layer: dict, view: str):
    if view == "unfused":
        tid = "tpl:unfused"
    elif view == "split" and layer.get("split_template"):
        tid = layer["split_template"]
    else:
        tid = layer["template"]
    return tid, ctx["data"]["templates"].get(tid)


def walk(node):
    yield node
    if node.get("kind") == "container":
        for ch in node.get("children") or []:
            yield from walk(ch)


def flat(node, collapsed=frozenset(), depth=0, out=None):
    out = out if out is not None else []
    out.append((depth, node))
    if node.get("kind") == "container" and node["id"] not in collapsed:
        for ch in node.get("children") or []:
            flat(ch, collapsed, depth + 1, out)
    return out


def find_node(root, node_id):
    for n in walk(root):
        if n.get("id") == node_id:
            return n
    return None


def env_of(ctx, preset, ratio, phase="same", layer_idx=None):
    return treemod.env_for(ctx["spec"], ctx["data"]["geometry_presets"][preset],
                           max(1, int(ratio or 1)), phase, layer_idx=layer_idx)


def eval_shape(ctx, shape, preset, layer, phase="same"):
    if not shape:
        return ""
    try:
        v = shapemod.eval_shape(shape, env_of(ctx, preset, layer.get("ratio", 1),
                                              phase, layer.get("idx")))
        if isinstance(v, list):
            return "[" + ", ".join(str(x) for x in v) + "]"
        return str(v)
    except Exception as e:                        # noqa: BLE001
        return f"<{e}>"


def layer_T(ctx, layer, preset):
    return env_of(ctx, preset, layer.get("ratio", 1), "same", layer.get("idx")).get("T")


# ── 命令：layers ─────────────────────────────────────────────────
def cmd_layers(ctx, a):
    rows = []
    for l in ctx["data"]["layers"]:
        roles = "".join(k for k, v in [("K", l["roles"].get("kv_source")),
                                       ("I", l["roles"].get("index_source")),
                                       ("C", l["roles"].get("candidate_source")),
                                       ("E", l["roles"].get("engram"))] if v) or "-"
        warns = len(l["coverage"].get("unmapped", [])) + len(l["coverage"].get("missing", []))
        rows.append((l["idx"], l["kind"], l["ratio"], layer_T(ctx, l, a.preset),
                     roles, l["template"], len(l.get("weights", [])), warns))
    head = " ".join([pad_w("层", 3), pad_w("层型", 16), pad_w("ratio", 5),
                     pad_w("T", 8), pad_w("角色", 4), pad_w("模板", 28),
                     pad_w("权重", 4), pad_w("告警", 4)])
    print(bold(head))
    for idx, kind, ratio, T, roles, tid, nw, warn in rows:
        cells = [pad_w(str(idx), 3), kind_str(pad_w(kind, 16)), pad_w(str(ratio), 5),
                 pad_w(str(T), 8), pad_w(roles, 4), pad_w(tid, 28), pad_w(str(nw), 4),
                 red(pad_w(str(warn), 4)) if warn else pad_w("-", 4)]
        print(" ".join(cells))
    print(dim(f"\n共 {len(rows)} 层；T 按预设 {a.preset}（replay 层带 ↺ 语义）"))


# ── 命令：tree ──────────────────────────────────────────────────
def render_tree_lines(ctx, layer, view, opts):
    tid, tpl = template_for(ctx, layer, view)
    if not tpl:
        print(red(f"层 {layer['idx']} 没有视图 {view!r}（可用：default"
                  + ("/split" if layer.get("split_template") else "") + "/unfused）"))
        return
    root = tpl["root"]
    lines = flat(root)

    def rec(node, prefix="", last=True, depth=0):
        out = []
        if node.get("kind") == "container":
            label = bold(node.get("name") or node["id"])
            sub = dim(f"  ({node.get('_path') or node['id']})")
            out.append(f"{prefix}{'└─' if last else '├─'} {label}{sub}")
        else:
            m = ctx["data"]["op_catalog"].get(node["op"], {})
            kernel = m.get("kernel")
            parts = [node["op"]]
            if kernel:
                parts.append(dim(f" [{kernel}]"))
            flags = [f for f in (m.get("flags") or [])
                     if f in ("implicit", "fused", "boundary_only", "runner_hook",
                              "no_impl", "placeholder", "packed_variant")]
            if flags:
                parts.append(" " + yellow(",".join(flags)))
            if opts.keys and node.get("iface"):
                req = ",".join(k["key"] for k in node["iface"].get("requires", []))
                prov = ",".join(k["key"] for k in node["iface"].get("provides", []))
                if req or prov:
                    parts.append(dim(f"  req[{req}] → prov[{prov}]"))
            out.append(f"{prefix}{'└─' if last else '├─'} " + "".join(parts))
        kids = node.get("children") or [] if node.get("kind") == "container" else []
        for i, ch in enumerate(kids):
            out += rec(ch, prefix + ("   " if last else "│  "), i == len(kids) - 1, depth + 1)
        return out

    print(bold(f"层 {layer['idx']} · {kind_str(layer['kind'])} · {tid} · "
               f"T={layer_T(ctx, layer, opts.preset)}"))
    for line in rec(root):
        print(line)
    if opts.full:
        print(dim("\n接口明细用：view_model.py op <层的idx> <路径>"))


def cmd_tree(ctx, a):
    layer = layer_of(ctx, a.layer)
    if not layer:
        raise SystemExit(f"没有层 {a.layer}")
    render_tree_lines(ctx, layer, a.view, a)


# ── 命令：op ────────────────────────────────────────────────────
def node_detail_lines(ctx, node, layer, preset):
    d = ctx["data"]["op_catalog"].get(node["op"], {})
    iface = node.get("iface") or {}
    out = [
        bold(f"{node['id']}") + dim(f"  ({node['op']})"),
        f"  Module  {d.get('module') or '-'}   impl {d.get('impl') or '-'}"
        + (f"   src {d['source']['file']}:{d['source']['line']}-{d['source']['end_line']}"
           if d.get("source") else ""),
        f"  kernel  {d.get('kernel') or '-'}   flags {','.join(d.get('flags') or []) or '-'}",
        "",
        bold("requires"),
    ]
    for k in iface.get("requires", []):
        ev = eval_shape(ctx, k.get("shape"), preset, layer, node.get("rows_phase", "same"))
        src = k.get("from") or "-"
        st = k.get("status", "")
        st_s = green("ok") if st == "ok" else (yellow(st) if st in ("in_place", "optional")
                                               else red(st or "?"))
        extra = f"  {k['key']}"
        if k.get("shape"):
            extra += dim(f" {k['shape']} → {ev}")
        extra += f"  {dim('←')} {src}  [{st_s}]"
        if k.get("optional"):
            extra += dim(" (可选)")
        out.append("  " + extra)
    out.append("")
    out.append(bold("provides"))
    for k in iface.get("provides", []):
        ev = eval_shape(ctx, k.get("shape"), preset, layer, node.get("rows_phase", "same"))
        extra = f"  {k['key']}"
        if k.get("shape"):
            extra += dim(f" {k['shape']} → {ev}")
        if k.get("mutation") and k["mutation"] != "new":
            extra += yellow(f"  [{k['mutation']}]")
        if k.get("consumers"):
            extra += dim("  → " + ", ".join(k["consumers"]))
        out.append("  " + extra)
    ws = d.get("weights") or []
    out += ["", bold(f"weights（{len(ws)}）")]
    for w in ws:
        hit = next((x for x in layer.get("weights", []) if x["key"] == w["suffix"]), None)
        shape = f"[{', '.join(map(str, hit['shape']))}]" if hit else ""
        out.append("  " + f"{w['suffix']:<46} {dim((hit or {}).get('dtype', '')):<12} "
                          f"{shape:<18} {w.get('via', 'raw')}")
    if d.get("note"):
        out += ["", dim("note: " + d["note"])]
    return out


def cmd_op(ctx, a):
    layer = layer_of(ctx, a.layer)
    if not layer:
        raise SystemExit(f"没有层 {a.layer}")
    tid, tpl = template_for(ctx, layer, a.view)
    node = find_node(tpl["root"], a.path) if tpl else None
    if not node:
        # 允许只给算子名：在模板里找第一个
        cand = [n for n in walk(tpl["root"]) if n.get("op") == a.path] if tpl else []
        if cand:
            node = cand[0]
        else:
            print(red(f"层 {a.layer} 的 {tid} 里找不到 {a.path!r}"))
            for alt in ("split", "unfused"):
                if alt == a.view:
                    continue
                tid2, tpl2 = template_for(ctx, layer, alt)
                if not tpl2:
                    continue
                hit2 = find_node(tpl2["root"], a.path) or next(
                    (n for n in walk(tpl2["root"]) if n.get("op") == a.path), None)
                if hit2:
                    print(yellow(f"提示：该模块在 {alt} 视图里；用 --view {alt}（{tid2}）"))
                    break
            print(dim("可用路径："))
            for n in walk(tpl["root"]) if tpl else []:
                if n.get("kind") != "container":
                    print("  " + n["id"])
            return
    for line in node_detail_lines(ctx, node, layer, a.preset):
        print(line)


# ── 命令：plugin / catalog ──────────────────────────────────────
def cmd_plugin(ctx, a):
    d = ctx["data"]["op_catalog"].get(a.op)
    if not d:
        raise SystemExit(f"没有插件 {a.op!r}（用 catalog 看全部）")
    layer = layer_of(ctx, a.layer) if a.layer is not None else ctx["data"]["layers"][0]
    print(bold(a.op) + dim(f"  · {d.get('category')} · Module {d.get('module')}"))
    print(f"  kernel {d.get('kernel') or '-'}   impl {d.get('impl') or '-'}")
    print(f"  flags  {','.join(d.get('flags') or []) or '-'}")
    if d.get("doc"):
        print(dim("  " + d["doc"]))
    print()
    print(bold("requires"))
    for k in d.get("requires", []):
        ev = eval_shape(ctx, k.get("shape"), a.preset, layer)
        print("  " + k["key"] + (dim(f"  {k['shape']} → {ev}") if k.get("shape") else "")
              + (dim("  (可选)") if k.get("optional") else ""))
    print(bold("provides"))
    for k in d.get("provides", []):
        ev = eval_shape(ctx, k.get("shape"), a.preset, layer)
        print("  " + k["key"] + (dim(f"  {k['shape']} → {ev}") if k.get("shape") else ""))
    ws = d.get("weights") or []
    if ws:
        print(bold(f"weights（{len(ws)}）"))
        for w in ws:
            hit = next((x for x in layer.get("weights", []) if x["key"] == w["suffix"]), None)
            print("  " + w["suffix"] + dim(f"  {w.get('via', 'raw')}")
                  + (dim(f"  {hit['dtype']} [{', '.join(map(str, hit['shape']))}]") if hit else ""))
    u = d.get("used_in") or {}
    print(bold("装在哪"))
    if u.get("templates"):
        print("  模板：" + ", ".join(u["templates"]))
        print("  层  ：" + range_str(u.get("layers", [])))
        print("  路径：" + ", ".join(u.get("paths", [])))
    else:
        print(dim("  当前配方未使用（写进 recipes.yaml 后重新生成即可）"))


def range_str(layers):
    if not layers:
        return ""
    out, a, b = [], layers[0], layers[0]
    for x in layers[1:]:
        if x == b + 1:
            b = x
            continue
        out.append(str(a) if a == b else f"{a}-{b}")
        a = b = x
    out.append(str(a) if a == b else f"{a}-{b}")
    return ", ".join(out)


def cmd_catalog(ctx, a):
    plugs = ctx["data"]["plugin_catalog"]
    if a.category:
        plugs = [p for p in plugs if p["category"] == a.category]
    if a.unused:
        plugs = [p for p in plugs if not p["used"]]
    if a.q:
        q = a.q.lower()
        plugs = [p for p in plugs if q in " ".join(str(p.get(k, "")) for k in
                 ("op", "module", "kernel", "impl", "doc", "category")).lower()]
    print(bold(f"{'插件':<20} {'分类':<12} {'kernel':<28} {'req':>4} {'prov':>5} "
               f"{'w':>3}  用在哪"))
    for p in plugs:
        used = "未使用" if not p["used"] else f"{len(p['used_in'].get('layers', []))} 层"
        print(f"{p['op']:<20} {p['category']:<12} {(p.get('kernel') or '-'):<28} "
              f"{p['requires']:>4} {p['provides']:>5} {p['weights']:>3}  "
              + (dim(used) if not p["used"] else green(used)))
    print(dim(f"\n共 {len(plugs)} 个；用 view_model.py plugin <op> 看接口"))


# ── 命令：blueprint / search ────────────────────────────────────
def _seg_item(x) -> str:
    if isinstance(x, str):
        return cyan(x) if x.startswith("_") else x
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return f"{x[0]}×{x[1]}"
    return str(x)


def cmd_blueprint(ctx, a):
    bp = ctx["data"].get("blueprint") or {}
    print(bold("配方段（recipes.yaml）"))
    for name, val in (bp.get("segments") or {}).items():
        if isinstance(val, list):
            print(f"  {name:<22} " + " ".join(_seg_item(x) for x in val))
        elif isinstance(val, dict):
            for k, arr in val.items():
                if isinstance(arr, list):
                    print(f"  {name + '.' + k:<22} " + " ".join(_seg_item(x) for x in arr))
    print(bold("\n模板 → 层"))
    for tid, tpl in ctx["data"]["templates"].items():
        ls = [l["idx"] for l in ctx["data"]["layers"]
              if l["template"] == tid or l.get("split_template") == tid]
        print(f"  {tid:<30} {dim(range_str(ls) or '对照模板')}")
        print("    " + dim(" → ".join(tpl["steps"])))


def cmd_search(ctx, a):
    q = a.q.lower()
    hits = []
    for op, d in ctx["data"]["op_catalog"].items():
        fields = [op, d.get("module") or "", d.get("kernel") or "", d.get("impl") or "",
                  d.get("doc") or ""]
        fields += [k["key"] for k in d.get("requires", [])]
        fields += [k["key"] for k in d.get("provides", [])]
        fields += [w["suffix"] for w in d.get("weights", [])]
        if q in " ".join(fields).lower():
            u = d.get("used_in") or {}
            hits.append((op, d.get("category", ""), d.get("kernel") or "",
                         range_str(u.get("layers", [])) or "未使用"))
    print(bold(f"匹配 {len(hits)} 个插件"))
    for op, cat, kernel, layers in hits:
        print(f"  {op:<20} {cat:<12} {kernel:<28} 层 {layers}")


# ── TUI ─────────────────────────────────────────────────────────
class TUI:
    def __init__(self, ctx, layer_idx=0):
        self.ctx = ctx
        self.preset_keys = list(ctx["data"]["geometry_presets"])
        self.preset_i = 0
        self.preset = self.preset_keys[0]
        self.lidx = max(0, min(layer_idx, len(ctx["data"]["layers"]) - 1))
        self.view = "default"
        self.pane = 0                     # 0=层 1=树 2=详情
        self.tree_sel = 0
        self.tree_scroll = 0
        self.collapsed = set()
        self.message = ""
        self.show_detail = False

    # 数据
    def layer(self):
        return self.ctx["data"]["layers"][self.lidx]

    def tpl(self):
        return template_for(self.ctx, self.layer(), self.view)[1]

    def lines(self):
        tpl = self.tpl()
        return flat(tpl["root"], self.collapsed) if tpl else []

    def selected_node(self):
        ls = self.lines()
        if not ls:
            return None
        self.tree_sel = max(0, min(self.tree_sel, len(ls) - 1))
        return ls[self.tree_sel][1]

    def detail(self, width):
        node = self.selected_node()
        if not node:
            return ["(空)"]
        if node.get("kind") == "container":
            return [bold(node["id"]),
                    dim("容器：接口是子模块接口的差集；Enter 展开/折叠"),
                    ] + [x for x in self._container_detail(node, width)]
        out = node_detail_lines(self.ctx, node, self.layer(), self.preset)
        return out

    def _container_detail(self, node, width):
        kids = [n["op"] for n in walk(node) if n.get("kind") != "container"]
        return ["子积木：" + ", ".join(kids)]

    # 绘制
    def draw(self, std):
        std.erase()
        h, w = std.getmaxyx()
        # 顶栏
        title = (f" DS4.1 mini · 层 {self.layer()['idx']} {self.layer()['kind']} · "
                 f"{self.preset} · {self.view} ")
        std.attron(curses.A_REVERSE)
        std.addnstr(0, 0, pad_w(title, w - 1), w - 1)
        std.attroff(curses.A_REVERSE)
        # 左栏
        rail_w = 24
        self._draw_rail(std, 1, rail_w, h - 2)
        # 主区
        main_x = rail_w + 1
        detail_w = 44 if w >= 150 else 0
        tree_w = w - main_x - detail_w - (1 if detail_w else 0)
        tree_h = (h - 2) if not self.show_detail or detail_w else h - 2
        self._draw_tree(std, 1, main_x, tree_h, tree_w)
        if detail_w:
            self._draw_detail(std, 1, main_x + tree_w + 1, h - 2, detail_w)
        # 底栏
        hint = (" ↑↓ 移动  Enter 展开/选择  Tab 切换栏  1/2/3 视图  p 预设  "
                "/ 搜索  d 详情  q 退出")
        std.addnstr(h - 1, 0, pad_w(hint, w - 1), w - 1)
        if self.message:
            std.addnstr(h - 1, 0, pad_w(" " + self.message, w - 1), w - 1,
                        curses.color_pair(4))
        std.refresh()

    def _draw_rail(self, std, y0, width, height):
        layers = self.ctx["data"]["layers"]
        start = max(0, min(self.lidx - height // 2, len(layers) - height))
        for i in range(height):
            li = start + i
            if li >= len(layers):
                break
            l = layers[li]
            sel = li == self.lidx
            T = layer_T(self.ctx, l, self.preset)
            text = f"{l['idx']:>2} {l['kind'][:12]:<12} T{T}"
            text = truncate_w(text, width - 2)
            pair = curses.color_pair(1) if (sel and self.pane == 0) else 0
            if sel:
                std.attron(curses.A_REVERSE if self.pane != 0 else curses.A_BOLD)
            std.addnstr(y0 + i, 0, pad_w(" " + text, width - 1), width - 1, pair)
            if sel:
                std.attroff(curses.A_REVERSE if self.pane != 0 else curses.A_BOLD)

    def _draw_tree(self, std, y0, x0, height, width):
        ls = self.lines()
        if not ls:
            std.addnstr(y0, x0, "（没有模板）", width)
            return
        if self.pane == 1:
            self.tree_sel = max(0, min(self.tree_sel, len(ls) - 1))
            if self.tree_sel < self.tree_scroll:
                self.tree_scroll = self.tree_sel
            if self.tree_sel >= self.tree_scroll + height:
                self.tree_scroll = self.tree_sel - height + 1
        for i in range(height):
            k = self.tree_scroll + i
            if k >= len(ls):
                break
            depth, node = ls[k]
            if node.get("kind") == "container":
                mark = "▸ " if node["id"] in self.collapsed else "▾ "
                text = "  " * depth + mark + (node.get("name") or node["id"])
            else:
                m = self.ctx["data"]["op_catalog"].get(node["op"], {})
                flags = ",".join(f for f in (m.get("flags") or [])
                                 if f in ("implicit", "fused", "boundary_only",
                                          "no_impl", "placeholder", "runner_hook"))
                text = "  " * depth + node["op"]
                if m.get("kernel"):
                    text += f" [{m['kernel']}]"
                if flags:
                    text += f" ({flags})"
            sel = self.pane == 1 and k == self.tree_sel
            std.addnstr(y0 + i, x0, pad_w(truncate_w(text, width), width), width,
                        curses.A_REVERSE if sel else 0)

    def _draw_detail(self, std, y0, x0, height, width):
        lines = self.detail(width)
        for i, line in enumerate(lines[:height]):
            # 去掉 ANSI（curses 不认识颜色转义）
            plain = _strip_ansi(line)
            std.addnstr(y0 + i, x0, pad_w(truncate_w(plain, width), width), width)

    # 交互
    def loop(self, std):
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_BLUE, -1)
        curses.init_pair(3, curses.COLOR_MAGENTA, -1)
        curses.init_pair(4, curses.COLOR_YELLOW, -1)
        curses.init_pair(5, curses.COLOR_RED, -1)
        curses.init_pair(6, curses.COLOR_GREEN, -1)
        curses.curs_set(0)
        std.keypad(True)
        while True:
            self.draw(std)
            ch = std.getch()
            if ch in (ord("q"), 27):
                break
            elif ch == curses.KEY_RESIZE:
                continue
            elif ch in (curses.KEY_UP, ord("k")):
                self._move(-1)
            elif ch in (curses.KEY_DOWN, ord("j")):
                self._move(+1)
            elif ch == curses.KEY_PPAGE:
                self._move(-10)
            elif ch == curses.KEY_NPAGE:
                self._move(+10)
            elif ch == 9:                      # Tab
                self.pane = (self.pane + 1) % 2
            elif ch in (curses.KEY_LEFT,):
                self._collapse(+1)
            elif ch in (curses.KEY_RIGHT, curses.KEY_ENTER, 10, 13):
                self._activate()
            elif ch in (ord("1"), ord("2"), ord("3")):
                self.view = {"1": "default", "2": "split", "3": "unfused"}[chr(ch)]
                self.tree_sel = 0
                self.collapsed.clear()
            elif ch == ord("p"):
                self.preset_i = (self.preset_i + 1) % len(self.preset_keys)
                self.preset = self.preset_keys[self.preset_i]
            elif ch == ord("d"):
                self.show_detail = not self.show_detail
            elif ch == ord("/"):
                self._search(std)

    def _move(self, delta):
        if self.pane == 0:
            self.lidx = max(0, min(len(self.ctx["data"]["layers"]) - 1,
                                   self.lidx + delta))
            self.tree_sel = 0
            self.tree_scroll = 0
            self.collapsed.clear()
            self.view = "default"
        else:
            self.tree_sel += delta

    def _collapse(self, _):
        if self.pane != 1:
            return
        node = self.selected_node()
        if node and node.get("kind") == "container":
            self.collapsed.add(node["id"])

    def _activate(self):
        if self.pane == 0:
            self.pane = 1
            return
        node = self.selected_node()
        if node and node.get("kind") == "container":
            if node["id"] in self.collapsed:
                self.collapsed.discard(node["id"])
            else:
                self.collapsed.add(node["id"])
        elif node:
            self.pane = 1
            self.show_detail = True

    def _search(self, std):
        h, w = std.getmaxyx()
        curses.echo()
        std.addnstr(h - 1, 0, pad_w(" 搜索（键/权重/算子，空=取消）: ", w - 1), w - 1)
        std.refresh()
        try:
            raw = std.getstr(h - 1, 26, 40).decode("utf-8", "replace")
        finally:
            curses.noecho()
        q = raw.strip().lower()
        if not q:
            return
        hit = self._find(q)
        if hit:
            layer_idx, path = hit
            self.lidx = layer_idx
            self.view = "default"
            self.collapsed.clear()
            ls = self.lines()
            for i, (_d, n) in enumerate(ls):
                if n.get("id") == path:
                    self.tree_sel = i
                    self.pane = 1
                    self.message = f"命中：层 {layer_idx} · {path}"
                    return
        self.message = f"没有找到 {q!r}"

    def _find(self, q):
        # 先在当前层模板里找路径/算子/键
        tpl = self.tpl()
        if tpl:
            for n in walk(tpl["root"]):
                if n.get("kind") == "container":
                    continue
                if q in n["id"].lower() or q in n["op"].lower():
                    return self.layer()["idx"], n["id"]
                for k in (n.get("iface") or {}).get("requires", []) + \
                        (n.get("iface") or {}).get("provides", []):
                    if q in k["key"].lower():
                        return self.layer()["idx"], n["id"]
        # 再跨层搜（找到第一层使用匹配插件的）
        for op, d in self.ctx["data"]["op_catalog"].items():
            if q in op.lower() or any(q in w["suffix"].lower() for w in d.get("weights", [])):
                u = d.get("used_in") or {}
                if u.get("layers"):
                    li = u["layers"][0]
                    path = (u.get("paths") or [None])[0]
                    return li, path
        return None


def _strip_ansi(s: str) -> str:
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def cmd_tui(ctx, a):
    if not sys.stdout.isatty():
        raise SystemExit("tui 需要终端（stdout 是 tty）；脚本里请用 tree/op 子命令")
    try:
        curses.wrapper(TUI(ctx, a.layer).loop)
    except KeyboardInterrupt:
        pass
    except curses.error as e:
        raise SystemExit(f"curses 初始化失败：{e}\n（TERM={os.environ.get('TERM')}）")


# ── main ────────────────────────────────────────────────────────
def main(argv=None):
    global COLOR
    ap = argparse.ArgumentParser(description="终端里的模型结构查看器")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "mini_ds41_flash.yaml"))
    ap.add_argument("--preset", default="", help="几何预设（默认第一个）")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--json", action="store_true", help="输出完整 JSON（脚本用）")
    # 允许把全局参数写在子命令后面（子解析器用 SUPPRESS 避免覆盖主解析器的默认值）
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS)
    common.add_argument("--preset", default=argparse.SUPPRESS)
    common.add_argument("--no-color", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("layers", parents=[common], help="40 层总览")
    p.set_defaults(fn=cmd_layers)

    p = sub.add_parser("tree", parents=[common], help="层内模块树")
    p.add_argument("layer", type=int)
    p.add_argument("--view", default="default", choices=["default", "split", "unfused"])
    p.add_argument("--keys", action="store_true", help="显示每个叶子的 req/prov 键")
    p.add_argument("--full", action="store_true")
    p.set_defaults(fn=cmd_tree)

    p = sub.add_parser("op", parents=[common], help="单个模块的接口/权重/形状")
    p.add_argument("layer", type=int)
    p.add_argument("path")
    p.add_argument("--view", default="default", choices=["default", "split", "unfused"])
    p.set_defaults(fn=cmd_op)

    p = sub.add_parser("plugin", parents=[common], help="插件详情（含装在哪）")
    p.add_argument("op")
    p.add_argument("--layer", type=int, default=None)
    p.set_defaults(fn=cmd_plugin)

    p = sub.add_parser("catalog", parents=[common], help="积木库")
    p.add_argument("--category", default="")
    p.add_argument("--unused", action="store_true")
    p.add_argument("--q", default="")
    p.set_defaults(fn=cmd_catalog)

    p = sub.add_parser("blueprint", parents=[common], help="配方段与模板")
    p.set_defaults(fn=cmd_blueprint)

    p = sub.add_parser("search", parents=[common], help="全局搜索插件/键/权重")
    p.add_argument("q")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("tui", parents=[common], help="全屏交互（curses）")
    p.add_argument("--layer", type=int, default=0)
    p.set_defaults(fn=cmd_tui)

    p = sub.add_parser("serve", parents=[common], help="HTTP 服务 + 自动刷新（转发端口）")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--bind", default="127.0.0.1")
    p.set_defaults(fn=None, serve=True)

    a = ap.parse_args(argv)
    COLOR = not a.no_color and os.environ.get("NO_COLOR") in (None, "") \
        and sys.stdout.isatty()
    if not getattr(a, "cmd", None):
        ap.print_help()
        return 0
    if getattr(a, "serve", False):
        from tools.serve_viewer import main as serve_main
        return serve_main(["--config", a.config, "--port", str(a.port),
                           "--bind", a.bind])
    ctx = load(a.config)
    if not a.preset:
        a.preset = ctx["preset"]
    elif a.preset not in ctx["data"]["geometry_presets"]:
        raise SystemExit(f"未知预设 {a.preset!r}；可选："
                         f"{', '.join(ctx['data']['geometry_presets'])}")
    if a.json:
        json.dump(ctx["data"], sys.stdout, ensure_ascii=False, indent=1)
        print()
        return 0
    a.fn(ctx, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

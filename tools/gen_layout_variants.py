#!/usr/bin/env python3
"""从 bench_command_builder.html 生成「布局 + 主题」变体（功能/JS/DOM id 全不动）。

用法：
    python3 tools/gen_layout_variants.py              # 生成 2 个变体 + 对比页
    python3 tools/gen_layout_variants.py --check      # 只校验磁盘上的变体是否最新（CI 用）
    python3 tools/gen_layout_variants.py --gallery    # 只重生成对比页
    python3 tools/gen_layout_variants.py --merge studio --out tools/bench_command_builder.html --force

产物：
    tools/bench_command_builder.layout_studio.html   工作台 Studio（三栏 + 左导航）
    tools/bench_command_builder.layout_bento.html    Bento 面板（12 栏磁贴 + 底部操作条）
    tools/bench_command_builder.layouts.html         对比选择页

原理：变体 = 基础文件的完整拷贝 + 在 </body> 前注入一段带标记的
`<style id="layout-pack">` + `<script id="layout-pack">`（shared + archetype）。
脚本位于主脚本之后，所以执行时 init()/bind() 已完成，可直接用 window.BUILDER。
"""
import argparse
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE / "bench_command_builder.html"
PACKS = HERE / "layout_packs"

ARCHES = {
    "studio": {
        "file": "bench_command_builder.layout_studio.html",
        "name": "工作台 · Studio",
        "tag": "三栏 + 左导航",
        "desc": "左：区块导航（随场景/高级模式自动变化，滚动联动高亮，底部常驻校验状态）；"
                "中：配置卡；右：实时预览。深色为「极光玻璃」，切浅色为「糖果渐变」。",
        "theme": "深色（极光玻璃）",
        "alt": "浅色（糖果渐变）",
        "swatch_dark": ["#070b16", "#8b5cf6", "#22d3ee"],
        "swatch_light": ["#f7f7fb", "#7c3aed", "#a78bfa"],
    },
    "bento": {
        "file": "bench_command_builder.layout_bento.html",
        "name": "Bento · 面板",
        "tag": "不对称磁贴 + 底部操作条",
        "desc": "通用 7 栏 + 场景 5 栏并排、多点/队列整行、高级工具 6+6；每块一个渐变色头；"
                "底部常驻「校验状态 / 复制当前 / 生成命令」。浅色为「糖果渐变」，"
                "切深色为「霓虹终端」。",
        "theme": "浅色（糖果渐变）",
        "alt": "深色（霓虹终端）",
        "swatch_dark": ["#04070c", "#00e6a0", "#22d3ee"],
        "swatch_light": ["#fdf4ff", "#db2777", "#7c3aed"],
    },
}

BLOCK_RE = re.compile(
    r"\n?<!-- LP-PACK:([a-z0-9_-]+):CSS:START -->.*?<!-- LP-PACK:\1:CSS:END -->\n?",
    re.S,
)
JS_BLOCK_RE = re.compile(
    r"\n?<!-- LP-PACK:([a-z0-9_-]+):JS:START -->.*?<!-- LP-PACK:\1:JS:END -->\n?",
    re.S,
)
# 兼容旧版单块标记（LP-PACK:<arch>:START/END），迁移时自动清掉
LEGACY_BLOCK_RE = re.compile(
    r"\n?<!-- LP-PACK:([a-z0-9_-]+):START -->.*?<!-- LP-PACK:\1:END -->\n?",
    re.S,
)


def read_pack(name: str) -> str:
    p = PACKS / name
    if not p.exists():
        raise SystemExit(f"✗ 缺文件：{p}")
    return p.read_text(encoding="utf-8")


def build(arch: str) -> str:
    base = BASE.read_text(encoding="utf-8")
    # 先剥旧包（含旧版单块标记），保证幂等
    base = BLOCK_RE.sub("\n", base)
    base = JS_BLOCK_RE.sub("\n", base)
    base = LEGACY_BLOCK_RE.sub("\n", base)
    if base.count("</body>") != 1:
        raise SystemExit("✗ 基础文件里 </body> 不是恰好一处")
    if base.count("</head>") != 1:
        raise SystemExit("✗ 基础文件里 </head> 不是恰好一处")

    css = read_pack("shared.css") + "\n" + read_pack(f"{arch}.css")
    js = read_pack("shared.js") + "\n" + read_pack(f"{arch}.js")
    if "</style" in css.lower():
        raise SystemExit(f"✗ {arch}.css 里出现了 </style")
    if "</script" in js.lower():
        raise SystemExit(f"✗ {arch}.js 里出现了 </script")

    css_block = (
        f"<!-- LP-PACK:{arch}:CSS:START -->\n"
        f'<style id="lp-pack-style" data-layout="{arch}">\n{css}\n</style>\n'
        f"<!-- LP-PACK:{arch}:CSS:END -->\n"
    )
    js_block = (
        f"<!-- LP-PACK:{arch}:JS:START -->\n"
        f'<script id="lp-pack-script" data-layout="{arch}">\n'
        f'window.__LP_ARCHETYPE__ = "{arch}";\n{js}\n</script>\n'
        f"<!-- LP-PACK:{arch}:JS:END -->\n"
    )
    out = base.replace("</head>", css_block + "</head>", 1)
    return out.replace("</body>", js_block + "</body>", 1)


def swatches(colors):
    return "".join(f'<span class="sw" style="background:{c}"></span>' for c in colors)


def gallery_html() -> str:
    cards = []
    for arch, m in ARCHES.items():
        cards.append(f"""
      <a class="card" href="{m['file']}" target="_blank" rel="noopener">
        <div class="swrow">
          <div class="swset"><span class="swlbl">默认</span>{swatches(m['swatch_dark'])}</div>
          <div class="swset"><span class="swlbl">切换</span>{swatches(m['swatch_light'])}</div>
        </div>
        <h2>{m['name']}</h2>
        <p class="tag">{m['tag']}</p>
        <p>{m['desc']}</p>
        <p class="themes">默认 <b>{m['theme']}</b> · 点右上角 ◐ 切到 <b>{m['alt']}</b></p>
        <span class="open">打开预览 →</span>
      </a>""")
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>命令构建器 · 两套布局方案</title>
<style>
  :root{{--ink:#141118;--muted:#6b6477;--line:#e8e3ee;--bg:#faf8fc}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:
      radial-gradient(900px 520px at 0% 0%, rgba(251,207,232,.55), transparent 60%),
      radial-gradient(900px 520px at 100% 0%, rgba(186,230,253,.55), transparent 60%),
      var(--bg);
    color:var(--ink);font-family:system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}}
  .wrap{{max-width:1080px;margin:0 auto;padding:52px 24px 72px}}
  h1{{font-size:30px;margin:0 0 8px;letter-spacing:-.02em}}
  .sub{{color:var(--muted);margin:0 0 30px;font-size:14px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}}
  .card{{display:block;background:#fff;border:1px solid var(--line);border-radius:20px;
    padding:20px;text-decoration:none;color:inherit;transition:.18s;
    box-shadow:0 1px 2px rgba(20,17,24,.04),0 14px 34px rgba(20,17,24,.06)}}
  .card:hover{{transform:translateY(-3px);box-shadow:0 22px 54px rgba(20,17,24,.14)}}
  .card h2{{font-size:20px;margin:14px 0 4px}}
  .tag{{color:#7c3aed;font-weight:700;font-size:12.5px;margin:0 0 8px}}
  .card p{{color:var(--muted);font-size:13px;line-height:1.65;margin:0 0 10px}}
  .themes{{font-size:12.5px}}
  .themes b{{color:var(--ink)}}
  .swrow{{display:flex;gap:18px;flex-wrap:wrap}}
  .swset{{display:flex;align-items:center;gap:6px}}
  .swlbl{{font-size:10.5px;color:var(--muted);letter-spacing:.06em}}
  .sw{{width:30px;height:30px;border-radius:9px;border:1px solid rgba(0,0,0,.06)}}
  .open{{font-size:13px;font-weight:800;color:#db2777}}
  .tip{{margin-top:30px;color:var(--muted);font-size:12.5px;line-height:1.8}}
  code{{background:#f1ecf7;padding:1px 6px;border-radius:5px;font-size:.92em}}
  kbd{{background:#141118;color:#fff;border-radius:5px;padding:1px 6px;font-size:11px}}
</style></head><body><div class="wrap">
  <h1>命令构建器 · 两套布局方案</h1>
  <p class="sub">功能完全一致（同一份 HTML/JS，只是外面多套一层「布局包」）；两套都自带明暗双主题。</p>
  <div class="grid">{''.join(cards)}
  </div>
  <p class="tip">
    两套都是**单文件离线可用**：直接把这个 HTML 拷走就能开。<br>
    场景/侧栏折叠记在浏览器 localStorage 里，两套共用；想从干净状态对比，开无痕窗口。<br>
    选定一套后，用 <code>python3 tools/gen_layout_variants.py --merge &lt;studio|bento&gt; --out tools/bench_command_builder.html --force</code>
    合并进主文件，再跑 <code>node tools/test_bench_builder.js</code> 回归。
  </p>
</div></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验磁盘产物是否最新")
    ap.add_argument("--gallery", action="store_true", help="只重生成对比页")
    ap.add_argument("--merge", choices=sorted(ARCHES), help="把某套布局合并进指定文件")
    ap.add_argument("--out", help="--merge 的输出路径")
    ap.add_argument("--force", action="store_true", help="--merge 时允许覆盖已存在文件")
    a = ap.parse_args()

    if a.merge:
        if not a.out:
            raise SystemExit("✗ --merge 需要 --out")
        out = pathlib.Path(a.out)
        if out.exists() and not a.force:
            raise SystemExit(f"✗ {out} 已存在；确认要覆盖请加 --force")
        out.write_text(build(a.merge), encoding="utf-8")
        print(f"✓ merged {a.merge} → {out}")
        return 0

    if a.check:
        bad = 0
        for arch, m in ARCHES.items():
            p = HERE / m["file"]
            want = build(arch)
            have = p.read_text(encoding="utf-8") if p.exists() else None
            if have != want:
                print(f"✗ 过期：{p.name}（重跑 gen_layout_variants.py）")
                bad += 1
            else:
                print(f"✓ 最新：{p.name}")
        return 1 if bad else 0

    if a.gallery:
        p = HERE / "bench_command_builder.layouts.html"
        p.write_text(gallery_html(), encoding="utf-8")
        print(f"✓ {p.name}")
        return 0

    for arch, m in ARCHES.items():
        p = HERE / m["file"]
        p.write_text(build(arch), encoding="utf-8")
        print(f"✓ {p.name}  ({m['name']})")
    p = HERE / "bench_command_builder.layouts.html"
    p.write_text(gallery_html(), encoding="utf-8")
    print(f"✓ {p.name}  (对比页)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

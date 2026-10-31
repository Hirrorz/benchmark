#!/usr/bin/env python3
"""从 bench_command_builder.html 生成多套**纯视觉**风格变体（功能/JS/DOM id 全不动）。

用法：
    python3 tools/gen_style_variants.py

产物（同目录）：
    bench_command_builder.style1.html  墨白 · 极简（Linear 风，浅色）
    bench_command_builder.style2.html  石墨 · 深色（IDE 风，默认深色）
    bench_command_builder.style3.html  海盐 · 清爽（Apple 风，大圆角）
    bench_command_builder.style4.html  仪表 · 高密度（Dashboard 风，紧凑）
    bench_command_builder.styles.html  四套风格的选择页

原理：原文件已有完整的 CSS 变量令牌（:root / [data-theme=dark]），变体只在
`</head>` 前追加一个覆盖层 `<style>`，并把需要的组件规则（顶栏/卡片/按钮/终端）
一起重写；不改任何 id/事件/逻辑。
"""
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE / "bench_command_builder.html"

# ── 每套风格：dark=True 时把 <html data-theme> 改成 dark ─────────────────────
VARIANTS = {
    "style1": {
        "file": "bench_command_builder.style1.html",
        "name": "墨白 · 极简",
        "desc": "白底细线、Indigo 点缀、零阴影；顶栏全白与页面融为一体。接近 Linear / Vercel 的克制感。",
        "dark": False,
        "css": r"""
/* ===== 风格 1：墨白 · 极简 ===== */
:root{
  --font-sans:"Inter",system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
  --bg:#ffffff; --bg-subtle:#f6f6f7; --card:#ffffff; --card-2:#fafafa;
  --ink:#18181b; --ink-2:#3f3f46; --muted:#71717a; --faint:#8f8f98;
  --line:#ebebef; --line-2:#f3f3f5; --line-strong:#dedee3;
  --topbar-bg:rgba(255,255,255,.82);
  --primary:#5b5bd6; --primary-2:#4b4bc4; --primary-hover:#4b4bc4;
  --primary-weak:#f1f1fd; --primary-line:#dcdcf7; --on-primary:#ffffff;
  --focus-ring:rgba(91,91,214,.16);
  --success:#15803d; --success-bg:#f2fbf5; --success-line:#bbf7d0;
  --warn:#a16207; --warn-bg:#fefce8; --warn-line:#fde68a;
  --danger:#b91c1c; --danger-bg:#fef2f2; --danger-line:#fecaca;
  --radius:10px; --radius-s:8px; --radius-xs:6px;
  --shadow-1:0 1px 2px rgba(24,24,27,.04);
  --shadow-2:0 12px 32px rgba(24,24,27,.08); --shadow:var(--shadow-1);
  --navy:#18181b; --navy-2:#3f3f46; --navy-3:#52525b;
  --teal:#5b5bd6; --teal-2:#4b4bc4; --teal-l:#f1f1fd;
  --blue:#4b4bc4; --blue-l:#f1f1fd;
  --v3-purple-l:#f1f1fd; --v3-purple-2:#4b4bc4;
}
.topbar{background:rgba(255,255,255,.86)!important;color:var(--ink)!important;
  border-bottom:1px solid var(--line)!important;box-shadow:none!important;
  backdrop-filter:saturate(180%) blur(14px);padding:11px 26px!important}
.brand .tag{color:#5b5bd6!important;letter-spacing:.2em!important}
.brand .sub{color:var(--muted)!important}
.brand h1 .ver{background:var(--primary-weak)!important;color:var(--primary-2)!important}
.segctl{background:var(--bg-subtle)!important;border:1px solid var(--line)!important}
.segctl button{color:var(--muted)!important}
.segctl button:hover{color:var(--ink)!important}
.segctl button.on{background:#fff!important;color:var(--ink)!important;
  box-shadow:0 1px 2px rgba(24,24,27,.08),0 0 0 1px var(--line)!important}
.topbtn{background:#fff!important;border:1px solid var(--line)!important;color:var(--ink-2)!important}
.topbtn:hover{background:var(--bg-subtle)!important;border-color:var(--line-strong)!important}
.iconbtn{background:#fff!important;border-color:var(--line)!important;color:var(--ink-2)!important}
.card{border-color:var(--line)!important;box-shadow:none!important}
.card:hover{border-color:var(--line-strong)!important}
.stepbadge{background:var(--primary-weak)!important;color:var(--primary-2)!important;box-shadow:none!important}
pre.term{background:#0f1115!important;border-color:#22252c!important;color:#d4d4d8!important;
  border-radius:10px!important;box-shadow:none!important}
.tab{background:transparent!important;border:1px solid transparent!important}
.tab:hover{background:var(--bg-subtle)!important;color:var(--ink)!important}
.tab.on{background:var(--ink)!important;color:#fff!important;box-shadow:none!important}
.chip{background:var(--bg-subtle)!important;border-color:var(--line)!important;color:var(--ink-2)!important}
.chip b{color:var(--primary-2)!important}
.btn.primary{box-shadow:none!important;background:var(--primary-2)!important}
.btn.primary:hover:not(:disabled){background:var(--primary-hover)!important}
.btn.secondary{background:var(--ink)!important}
.btn.ghost{background:#fff!important}
""",
    },
    "style2": {
        "file": "bench_command_builder.style2.html",
        "name": "石墨 · 深色",
        "desc": "默认深色、细边框、无阴影；顶栏与页面同色，蓝紫点缀 + 等宽标题，偏 IDE / 终端气质。",
        "dark": True,
        "css": r"""
/* ===== 风格 2：石墨 · 深色 ===== */
html[data-theme="dark"]{
  --bg:#0b0d10; --bg-subtle:#111419; --card:#14181f; --card-2:#181d25;
  --ink:#e8eaed; --ink-2:#c3c8d0; --muted:#8f98a5; --faint:#77808d;
  --line:#232a34; --line-2:#1a2028; --line-strong:#333c49;
  --topbar-bg:rgba(11,13,16,.82);
  --primary:#7aa2f7; --primary-2:#9ab8ff; --primary-hover:#5f8ef0;
  --primary-weak:#16233b; --primary-line:#2b4670; --on-primary:#0b0d10;
  --focus-ring:rgba(122,162,247,.28);
  --success:#7ee787; --success-bg:#0f2a17; --success-line:#1f5130;
  --warn:#e3b341; --warn-bg:#2a2110; --warn-line:#5c4a1c;
  --danger:#ff7b72; --danger-bg:#2d1214; --danger-line:#6b2529;
  --radius:8px; --radius-s:6px; --radius-xs:5px;
  --shadow-1:none; --shadow-2:0 16px 40px rgba(0,0,0,.5); --shadow:var(--shadow-1);
  --navy:#e8eaed; --navy-2:#c3c8d0; --navy-3:#9aa3af;
  --teal:#7aa2f7; --teal-2:#5f8ef0; --teal-l:#16233b;
  --blue:#9ab8ff; --blue-l:#16233b;
  --v3-purple-l:#16233b; --v3-purple-2:#5f8ef0;
}
html[data-theme="dark"]{color-scheme:dark}
.topbar{background:#0b0d10!important;color:var(--ink)!important;
  border-bottom:1px solid var(--line)!important;box-shadow:none!important;padding:11px 24px!important}
.brand .tag{color:#7aa2f7!important;letter-spacing:.2em!important}
.brand h1{font-family:var(--font-mono)!important;font-size:19px!important;letter-spacing:-.01em}
.brand .sub{color:var(--muted)!important}
.brand h1 .ver{background:#16233b!important;color:#9ab8ff!important}
.segctl{background:#111419!important;border:1px solid var(--line)!important}
.segctl button{color:var(--muted)!important}
.segctl button:hover{color:var(--ink)!important}
.segctl button.on{background:#1d2733!important;color:#e8eaed!important;
  box-shadow:inset 0 0 0 1px #2b3644!important}
.topbtn{background:#14181f!important;border:1px solid var(--line)!important;color:var(--ink-2)!important}
.topbtn:hover{background:#1a2028!important;border-color:var(--line-strong)!important}
.iconbtn{background:#14181f!important;border-color:var(--line)!important;color:var(--ink-2)!important}
.card{border-color:var(--line)!important;box-shadow:none!important;background:var(--card)!important}
.stepbadge{background:#1d2733!important;color:#9ab8ff!important;box-shadow:none!important}
pre.term{background:#08090b!important;border-color:#1e242c!important;color:#c9d1d9!important;
  border-radius:8px!important;box-shadow:none!important}
.tab{background:transparent!important;border:1px solid transparent!important}
.tab:hover{background:#1a2028!important;color:var(--ink)!important}
.tab.on{background:#1d2733!important;color:#e8eaed!important;box-shadow:inset 0 0 0 1px #2b3644!important}
.chip{background:#111419!important;border-color:var(--line)!important;color:var(--ink-2)!important}
.chip b{color:#9ab8ff!important}
.btn.primary{background:#2f6feb!important;color:#fff!important;box-shadow:none!important}
.btn.primary:hover:not(:disabled){background:#3b7cf5!important}
.btn.secondary{background:#1d2733!important;color:#e8eaed!important}
.btn.ghost{background:#14181f!important;color:var(--ink-2)!important}
""",
    },
    "style3": {
        "file": "bench_command_builder.style3.html",
        "name": "海盐 · 清爽",
        "desc": "浅灰蓝底 + 纯白大圆角卡片 + 柔和投影，Apple 式留白与胶囊按钮；最“轻”的一套。",
        "dark": False,
        "css": r"""
/* ===== 风格 3：海盐 · 清爽 ===== */
:root{
  --font-sans:-apple-system,BlinkMacSystemFont,"SF Pro Text","PingFang SC","Microsoft YaHei",
    system-ui,sans-serif;
  --bg:#f5f5f7; --bg-subtle:#ececf0; --card:#ffffff; --card-2:#fbfbfd;
  --ink:#1d1d1f; --ink-2:#424245; --muted:#6e6e73; --faint:#86868b;
  --line:#e8e8ed; --line-2:#f0f0f4; --line-strong:#d7d7de;
  --topbar-bg:rgba(245,245,247,.78);
  --primary:#0071e3; --primary-2:#0063c6; --primary-hover:#0063c6;
  --primary-weak:#eaf3ff; --primary-line:#c3dcfa; --on-primary:#ffffff;
  --focus-ring:rgba(0,113,227,.18);
  --success:#1a7f45; --success-bg:#eefaf2; --success-line:#bfe8cf;
  --warn:#9a6700; --warn-bg:#fff8e6; --warn-line:#f2d68a;
  --danger:#c0392b; --danger-bg:#fdeeec; --danger-line:#f2c0b9;
  --radius:18px; --radius-s:12px; --radius-xs:9px;
  --shadow-1:0 1px 2px rgba(0,0,0,.03),0 8px 24px rgba(0,0,0,.05);
  --shadow-2:0 24px 60px rgba(0,0,0,.14); --shadow:var(--shadow-1);
  --navy:#1d1d1f; --navy-2:#424245; --navy-3:#55555a;
  --teal:#0071e3; --teal-2:#0063c6; --teal-l:#eaf3ff;
  --blue:#0063c6; --blue-l:#eaf3ff;
  --v3-purple-l:#f0edff; --v3-purple-2:#0063c6;
}
body{background:var(--bg)!important}
.topbar{background:rgba(245,245,247,.78)!important;color:var(--ink)!important;
  border-bottom:1px solid rgba(0,0,0,.06)!important;box-shadow:none!important;
  backdrop-filter:saturate(180%) blur(20px);padding:12px 28px!important}
.brand .tag{color:#0071e3!important}
.brand .sub{color:var(--muted)!important}
.brand h1 .ver{background:#eaf3ff!important;color:#0063c6!important}
.segctl{background:rgba(120,120,128,.10)!important;border:1px solid transparent!important}
.segctl button{color:var(--muted)!important}
.segctl button.on{background:#fff!important;color:var(--ink)!important;
  box-shadow:0 1px 3px rgba(0,0,0,.10)!important}
.topbtn{background:rgba(255,255,255,.75)!important;border:1px solid rgba(0,0,0,.06)!important;
  color:var(--ink-2)!important;border-radius:999px!important}
.topbtn:hover{background:#fff!important}
.iconbtn{background:rgba(255,255,255,.75)!important;border-color:rgba(0,0,0,.06)!important;
  color:var(--ink-2)!important}
.card{border:none!important;box-shadow:var(--shadow-1)!important;border-radius:var(--radius)!important}
.cardhead h2{letter-spacing:-.01em}
.stepbadge{background:#eaf3ff!important;color:#0063c6!important;box-shadow:none!important;
  border-radius:999px!important}
pre.term{background:#1c1c1e!important;border:none!important;border-radius:14px!important;
  color:#e8e8ed!important;box-shadow:none!important}
.tab{background:rgba(120,120,128,.08)!important;border:none!important;border-radius:999px!important}
.tab:hover{background:rgba(120,120,128,.14)!important;color:var(--ink)!important}
.tab.on{background:var(--ink)!important;color:#fff!important;box-shadow:none!important}
.chip{background:rgba(120,120,128,.08)!important;border:none!important;color:var(--ink-2)!important}
.chip b{color:#0063c6!important}
.btn{border-radius:999px!important}
.btn.primary{background:#0071e3!important;box-shadow:0 2px 10px rgba(0,113,227,.25)!important}
.btn.primary:hover:not(:disabled){background:#0063c6!important}
.btn.secondary{background:var(--ink)!important}
.btn.ghost{background:#fff!important;border-color:rgba(0,0,0,.08)!important}
""",
    },
    "style4": {
        "file": "bench_command_builder.style4.html",
        "name": "仪表 · 高密度",
        "desc": "13px 小字号、6px 圆角、零阴影、蓝色 2px 顶边；信息密度最高，像监控台/仪表盘。",
        "dark": False,
        "css": r"""
/* ===== 风格 4：仪表 · 高密度 ===== */
:root{
  --font-sans:system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
  --font-mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --bg:#f4f6f9; --bg-subtle:#eef1f5; --card:#ffffff; --card-2:#fafbfc;
  --ink:#0f172a; --ink-2:#334155; --muted:#64748b; --faint:#7c8899;
  --line:#dde3ea; --line-2:#e9eef3; --line-strong:#cbd5e1;
  --topbar-bg:#ffffff;
  --primary:#2563eb; --primary-2:#1d4ed8; --primary-hover:#1d4ed8;
  --primary-weak:#eff4ff; --primary-line:#c7d8f9; --on-primary:#ffffff;
  --focus-ring:rgba(37,99,235,.18);
  --success:#15803d; --success-bg:#f0fdf4; --success-line:#bbf7d0;
  --warn:#b45309; --warn-bg:#fffbeb; --warn-line:#fde68a;
  --danger:#b91c1c; --danger-bg:#fef2f2; --danger-line:#fecaca;
  --radius:6px; --radius-s:5px; --radius-xs:4px;
  --shadow-1:none; --shadow-2:0 4px 14px rgba(15,23,42,.10); --shadow:var(--shadow-1);
  --navy:#0f172a; --navy-2:#334155; --navy-3:#475569;
  --teal:#2563eb; --teal-2:#1d4ed8; --teal-l:#eff4ff;
  --blue:#1d4ed8; --blue-l:#eff4ff;
  --v3-purple-l:#eff4ff; --v3-purple-2:#1d4ed8;
}
body{font-size:13px!important}
.topbar{background:#fff!important;color:var(--ink)!important;
  border-bottom:2px solid #2563eb!important;box-shadow:none!important;padding:8px 18px!important}
.brand .tag{color:#2563eb!important;font-size:10px!important}
.brand h1{font-size:17px!important}
.brand .sub{color:var(--muted)!important;font-size:11px!important}
.brand h1 .ver{background:#eff4ff!important;color:#1d4ed8!important}
.segctl{background:var(--bg-subtle)!important;border:1px solid var(--line)!important;border-radius:6px!important}
.segctl button{border-radius:4px!important;padding:4px 12px!important;color:var(--muted)!important}
.segctl button.on{background:#2563eb!important;color:#fff!important;box-shadow:none!important}
.topbtn{background:#fff!important;border:1px solid var(--line)!important;color:var(--ink-2)!important;
  border-radius:5px!important;padding:5px 11px!important}
.iconbtn{background:#fff!important;border-color:var(--line)!important;color:var(--ink-2)!important;
  border-radius:5px!important}
.card{padding:12px 14px!important;border-color:var(--line)!important;box-shadow:none!important;
  border-radius:var(--radius)!important}
.cardhead{margin-bottom:8px!important;gap:8px!important}
.cardhead h2{font-size:14px!important}
.stepbadge{width:22px!important;height:22px!important;font-size:11.5px!important;
  background:#2563eb!important;color:#fff!important;box-shadow:none!important;border-radius:4px!important}
pre.term{background:#0f172a!important;border-color:#1e293b!important;color:#e2e8f0!important;
  border-radius:6px!important;font-size:11.5px!important;box-shadow:none!important}
.tab{border-radius:4px!important;background:#f1f5f9!important}
.tab.on{background:#2563eb!important;color:#fff!important;box-shadow:none!important}
.chip{background:#f1f5f9!important;border-color:var(--line)!important;color:var(--ink-2)!important}
.chip b{color:#1d4ed8!important}
.btn{border-radius:5px!important;padding:6px 11px!important;font-size:12px!important}
.btn.primary{background:#2563eb!important;box-shadow:none!important}
.btn.secondary{background:#0f172a!important}
.btn.ghost{background:#fff!important}
.field input,.field select{padding:5px 8px!important}
""",
    },
}


def build():
    base = BASE.read_text(encoding="utf-8")
    if "</head>" not in base:
        raise SystemExit("✗ 基础文件里找不到 </head>")
    out = []
    for key, v in VARIANTS.items():
        html = base
        if v["dark"]:
            html = html.replace('<html lang="zh-CN" data-theme="light">',
                                '<html lang="zh-CN" data-theme="dark">', 1)
        style = ("\n<!-- ===== 风格覆盖层（由 gen_style_variants.py 生成；功能不动） ===== -->\n"
                 "<style id=\"style-pack\">" + v["css"] + "</style>\n")
        html = html.replace("</head>", style + "</head>", 1)
        p = HERE / v["file"]
        p.write_text(html, encoding="utf-8")
        out.append((key, v, p))
        print(f"✓ {p.name}  ({v['name']})")
    return out


def gallery(items):
    cards = []
    swatches = {
        "style1": ["#ffffff", "#18181b", "#5b5bd6"],
        "style2": ["#0b0d10", "#e8eaed", "#7aa2f7"],
        "style3": ["#f5f5f7", "#1d1d1f", "#0071e3"],
        "style4": ["#f4f6f9", "#0f172a", "#2563eb"],
    }
    for key, v, p in items:
        sw = "".join(f'<span class="sw" style="background:{c}"></span>' for c in swatches[key])
        cards.append(f"""
      <a class="card" href="{p.name}" target="_blank" rel="noopener">
        <div class="swrow">{sw}</div>
        <h2>{v['name']}</h2>
        <p>{v['desc']}</p>
        <span class="open">打开预览 →</span>
      </a>""")
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>命令构建器 · 风格方案</title>
<style>
  :root{{--ink:#111318;--muted:#6b7280;--line:#e7e8ec;--bg:#f7f7f9}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink);
    font-family:system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}}
  .wrap{{max-width:1180px;margin:0 auto;padding:48px 24px 64px}}
  h1{{font-size:28px;margin:0 0 6px;letter-spacing:-.02em}}
  .sub{{color:var(--muted);margin:0 0 28px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}}
  .card{{display:block;background:#fff;border:1px solid var(--line);border-radius:16px;
    padding:18px;text-decoration:none;color:inherit;transition:.18s;box-shadow:0 1px 2px rgba(0,0,0,.03)}}
  .card:hover{{transform:translateY(-2px);box-shadow:0 12px 32px rgba(0,0,0,.10);border-color:#d6d8de}}
  .card h2{{font-size:17px;margin:12px 0 6px}}
  .card p{{color:var(--muted);font-size:13px;line-height:1.6;margin:0 0 12px}}
  .swrow{{display:flex;gap:6px}}
  .sw{{width:34px;height:34px;border-radius:9px;border:1px solid rgba(0,0,0,.06)}}
  .open{{font-size:12.5px;font-weight:700;color:#4f46e5}}
  .tip{{margin-top:26px;color:var(--muted);font-size:12.5px;line-height:1.7}}
  code{{background:#eef0f4;padding:1px 6px;border-radius:5px;font-size:.92em}}
</style></head><body><div class="wrap">
  <h1>命令构建器 · 4 套风格</h1>
  <p class="sub">功能完全一致（同一份 HTML/JS，仅覆盖视觉层）；点开任意一套即可使用。</p>
  <div class="grid">{''.join(cards)}
  </div>
  <p class="tip">想微调：改 <code>tools/gen_style_variants.py</code> 里对应风格的 CSS 变量后重跑，
  会重新生成这 4 个文件。<br>偏好确定后，我可以把选中的那套合并回主文件
  <code>bench_command_builder.html</code>。</p>
</div></body></html>
"""
    p = HERE / "bench_command_builder.styles.html"
    p.write_text(html, encoding="utf-8")
    print(f"✓ {p.name}  (风格选择页)")


if __name__ == "__main__":
    gallery(build())

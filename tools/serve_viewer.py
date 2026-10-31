#!/usr/bin/env python3
"""在线版模型查看器：SSH 下推荐用法（端口转发 + 自动重载）。

    # 远端
    python3.11 tools/serve_viewer.py --config configs/mini_ds41_flash.yaml
    # 本地另开一个终端（或 VS Code 的端口转发）
    ssh -L 8765:127.0.0.1:8765 <user>@<host>
    # 本地浏览器打开 http://localhost:8765/

特性：
  · 源码（modules/、recipes.yaml、opgraph.py、模板）改动后，浏览器**自动刷新**；
  · `/api/data` 直接给 JSON（脚本/diff 用）；
  · 绑定默认 127.0.0.1（只经 SSH 隧道访问，不暴露到网络）；
  · 无第三方依赖（stdlib http.server）。

    python3.11 tools/serve_viewer.py --bind 0.0.0.0 --port 8765   # 可信内网才这么用
"""
from __future__ import annotations

import argparse
import hashlib
import html as html_mod
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.export_model_html import build_data, render   # noqa: E402

AUTO_RELOAD = """
<script>
(function(){
  let known = null;
  async function tick(){
    try{
      const r = await fetch('/api/version', {cache:'no-store'});
      const j = await r.json();
      if (known === null) known = j.version;
      else if (j.version !== known) location.reload();
    }catch(e){}
  }
  tick(); setInterval(tick, 1500);
})();
</script>
"""


def watch_files(cfg: str, template: str, root: str | None = None) -> list[str]:
    """自动刷新监听的源文件：代码 + **配置 YAML** + 环境变量指向的外部配置。

    modules/specs/*.yaml（以及 modules.d/）是模块定义的事实源；只监听 .py 会让
    "改 YAML 立刻看 HTML" 失效（DX P0-5）。
    """
    root = root or ROOT
    files = [cfg, os.path.join(root, "recipes.yaml"),
             os.path.join(root, "opgraph.py"),
             os.path.join(root, "model_spec.py"),
             os.path.join(root, "weights_manifest.py"),
             template]
    watched = (".py", ".yaml", ".yml")
    for base, _dirs, names in os.walk(os.path.join(root, "modules")):
        for n in sorted(names):
            if n.endswith(watched):
                files.append(os.path.join(base, n))
    # DS41M_MODULE_SPEC* 可以指向仓库外（沙盒/临时配置）；它们也影响 HTML。
    extra = [os.environ.get("DS41M_MODULE_SPEC"),
             os.environ.get("DS41M_RECIPE_MAP"),
             os.environ.get("DS41M_KEY_NAMES")]
    extra_dir = os.environ.get("DS41M_MODULE_SPEC_DIR")
    if extra_dir and os.path.isdir(extra_dir):
        for base, _dirs, names in os.walk(extra_dir):
            for n in sorted(names):
                if n.endswith((".yaml", ".yml")):
                    extra.append(os.path.join(base, n))
    files.extend(f for f in extra if f and os.path.exists(f))
    return files


class State:
    def __init__(self, cfg: str, template: str):
        self.cfg = cfg
        self.template = template
        self.lock = threading.Lock()
        self.html = ""
        self.data_json = "{}"
        self.built_version = ""
        self.error = ""
        self.built_at = 0.0
        self.build_ms = 0.0

    # ── 源文件指纹（mtime+size）─────────────────────────────────
    def source_version(self) -> str:
        h = hashlib.sha256()
        for f in sorted(set(watch_files(self.cfg, self.template))):
            try:
                st = os.stat(f)
                h.update(f.encode())
                h.update(str(st.st_mtime_ns).encode())
                h.update(str(st.st_size).encode())
            except OSError:
                pass
        return h.hexdigest()[:16]

    def rebuild(self) -> None:
        t0 = time.time()
        try:
            data, _ = build_data(self.cfg)
            page = render(self.template, data)
            if "</body>" in page:
                page = page.replace("</body>", AUTO_RELOAD + "</body>")
            blob = json.dumps(data, ensure_ascii=False, sort_keys=True)
            with self.lock:
                self.html = page
                self.data_json = blob
                self.error = ""
                self.built_version = self.source_version()
                self.built_at = time.time()
                self.build_ms = (time.time() - t0) * 1000
        except Exception as e:                       # noqa: BLE001
            tb = traceback.format_exc()
            with self.lock:
                self.error = tb
                self.html = _error_page(e, tb)
                self.built_version = self.source_version()
                self.built_at = time.time()
                self.build_ms = (time.time() - t0) * 1000

    def ensure_fresh(self) -> None:
        if self.source_version() != self.built_version:
            self.rebuild()

    def snapshot(self):
        with self.lock:
            return self.html, self.data_json, self.error, self.built_version, \
                self.built_at, self.build_ms


def _error_page(exc, tb: str) -> str:
    return f"""<!doctype html><meta charset="utf-8">
<title>模型查看器 · 构建失败（会自动重试）</title>
<body style="font:13px/1.6 ui-monospace,monospace;background:#0a0e14;color:#dbe6f3;padding:24px">
<h2 style="color:#f87171">构建失败：{html_mod.escape(type(exc).__name__)}</h2>
<p>修好代码后本页会自动刷新（服务端每 1.5s 检查源文件）。</p>
<pre style="white-space:pre-wrap;color:#fca5a5">{html_mod.escape(tb)}</pre>
<script>setTimeout(()=>location.reload(), 2000)</script></body>"""


def make_handler(state: State, verbose: bool):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if verbose:
                print(f"  [{time.strftime('%H:%M:%S')}] " + fmt % args, flush=True)

        def _send(self, code, body: bytes, ctype: str, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path == "/api/version":
                    state.ensure_fresh()
                    _h, _d, err, ver, built_at, ms = state.snapshot()
                    body = json.dumps(dict(version=ver, error=bool(err),
                                           built_at=built_at, build_ms=round(ms, 1)))
                    return self._send(200, body.encode(), "application/json")
                if path == "/api/data":
                    state.ensure_fresh()
                    _h, d, _e, _v, _b, _m = state.snapshot()
                    return self._send(200, d.encode(), "application/json")
                if path in ("/", "/index.html", "/model_viewer.html"):
                    state.ensure_fresh()
                    page, _d, err, _v, _b, ms = state.snapshot()
                    return self._send(200, page.encode(),
                                      "text/html; charset=utf-8",
                                      {"X-Build-Ms": str(round(ms, 1)),
                                       "X-Build-Error": "1" if err else "0"})
                if path == "/health":
                    return self._send(200, b"ok", "text/plain")
            except Exception as e:                   # noqa: BLE001
                tb = traceback.format_exc()
                return self._send(500, _error_page(e, tb).encode(),
                                  "text/html; charset=utf-8")
            return self._send(404, b"not found", "text/plain")

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "mini_ds41_flash.yaml"))
    ap.add_argument("--template", default=os.path.join(HERE, "model_viewer.template.html"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="默认只允许本机（配合 ssh -L）；可信内网可 0.0.0.0")
    ap.add_argument("--verbose", action="store_true", help="打印每个请求")
    a = ap.parse_args(argv)

    state = State(os.path.abspath(a.config), os.path.abspath(a.template))
    t0 = time.time()
    state.rebuild()
    _h, _d, err, ver, _b, ms = state.snapshot()
    print(f"构建 {'失败' if err else '完成'}：{ms:.0f} ms · version={ver}")
    if err:
        print(err)
    try:
        httpd = ThreadingHTTPServer((a.bind, a.port), make_handler(state, a.verbose))
    except OSError as e:
        print(f"端口 {a.port} 不可用（{e}）。换一个：--port 8766", file=sys.stderr)
        return 2
    host = "localhost" if a.bind in ("127.0.0.1", "0.0.0.0") else a.bind
    print(f"""
服务已启动：http://{a.bind}:{a.port}/   （源文件改动后浏览器自动刷新）

SSH 下（推荐）：
  1) 本地终端执行：ssh -L {a.port}:127.0.0.1:{a.port} <user>@<host>
  2) 本地浏览器打开：http://localhost:{a.port}/
  （VS Code Remote-SSH 直接在「端口」面板转发 {a.port} 即可）

其他入口：
  /api/data      完整 JSON（脚本/diff）
  /api/version   源文件指纹（自动刷新用）
  Ctrl-C 退出
""", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

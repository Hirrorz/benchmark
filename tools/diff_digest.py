#!/usr/bin/env python3
"""对比两份日志里的逐算子 digest（Phase 1.5/2 的 A/B 闸门）。

    python3.11 tools/diff_digest.py base.log mod.log [--strict]

约定：
  · 只比较包含 `[op-digest]` / `[packed-digest]` 的行；
  · 忽略时间戳/进度等前缀，只取 digest 行本身；
  · 同模式对比（compact 只与 compact 比）；行数/顺序/内容任一不同 → 失败；
  · `--strict`：有任何差异退出码 1（默认也返回 1，该参数用于显式声明意图）。
"""
from __future__ import annotations

import argparse
import re
import sys

_PAT = re.compile(r"\[(op-digest|packed-digest)\](.*)$")


def digest_lines(path: str) -> list[str]:
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _PAT.search(line.rstrip("\n"))
            if m:
                out.append(m.group(0).strip())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("mod")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--limit", type=int, default=20, help="最多打印多少条差异")
    a = ap.parse_args(argv)

    base = digest_lines(a.base)
    mod = digest_lines(a.mod)
    if not base or not mod:
        print(f"FAIL：digest 行数为 0（base={len(base)}, mod={len(mod)}）——"
              f"检查 DS41M_OP_DIGEST=1 / DS41M_DIGEST_TAIL=1 是否设置")
        return 1

    n = min(len(base), len(mod))
    diffs = []
    for i in range(n):
        if base[i] != mod[i]:
            diffs.append((i, base[i], mod[i]))
    if len(base) != len(mod):
        diffs.append(("len", f"base={len(base)}", f"mod={len(mod)}"))

    if diffs:
        print(f"FAIL：{len(diffs)} 处差异（base {len(base)} 行 / mod {len(mod)} 行）")
        for i, b, m in diffs[:a.limit]:
            print(f"  [{i}]\n    base: {b}\n    mod : {m}")
        return 1
    print(f"PASS：{len(base)} 条 digest 逐字节一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

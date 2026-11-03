#!/usr/bin/env python3.11
"""生成网格 points.json：按实测硬约束过滤非法/必崩格子。

四条约束（全部实测）：
  ① `R = M + replay_tail` 超过 **32628** → DeepSelect 抢不到 workspace（8/16 GB）
     → `Insufficient_Device_Memory`。**CP1/CP2/CP4 上限一样**（按 cp_size 放大会崩：
     实测 CP2 的 L49152@hit0 每次必崩，runner 逐个死掉）。
  ② `hit > 0` 时 `R` 不能超过 `L`，否则 runner 抛 `命中几何非法` 并 SystemExit
     → **整轮网格直接死**（不是单点跳过）。
  ③ `hit == 0` 且 `L > 32768`：全算路径，实测 CP2 上必崩（先占满显存再要 workspace）。
     这条等价于 ①`R=M+140 ≤ 32628`，但额外挡 `L` 本身太大、KV 池就吃满的情况。
  ④ **每卡新增（重算）token 数 `(1-hit)*L/cp` < `--min-per-card`（默认 128）
     不采集**：`(1-hit)*L` = M（新增 token），`/cp` = 每卡分到的量；实测这类
     极小重算段（如 CP4 的 hit0.999，~36 行/卡）会稳定触发 Mqsmla 的 aicore trap
     （507015 / code 341）。`0` 关闭。

用法：
  python3.11 tools/gen_grid.py --out points.json --cp 2
  python3.11 tools/gen_grid.py --out points.json --cp 4 --min-per-card 0   # 关闭 ④
"""
import argparse
import json
import sys

L_UNITS = {"": 1, "k": 1024, "m": 1024 * 1024}
R_LIMIT = 32628                 # 实测硬上限（所有 CP 相同）
FULL_L_LIMIT = 32768            # hit=0 时 L 的上限（更大必崩）


def parse_l(s: str) -> int:
    s = s.strip().lower()
    num = "".join(c for c in s if c.isdigit())
    unit = "".join(c for c in s if c.isalpha())
    if not num or unit not in L_UNITS:
        sys.exit(f"✗ L 不合法：{s!r}")
    return int(num) * L_UNITS[unit]


def parse_hit(s: str) -> float:
    t = s.strip().rstrip("%")
    try:
        v = float(t)
    except ValueError:
        sys.exit(f"✗ hit 不合法：{s!r}")
    v = v / 100.0 if v > 1 else v
    if not (0.0 <= v < 1.0):
        sys.exit(f"✗ hit ∈ [0,1)：{s!r}")
    return v


DEFAULT_LS = ("1k", "2k", "4k", "8k", "12k", "16k", "20k", "24k", "32k", "40k",
              "48k", "56k", "64k", "80k", "96k", "112k", "128k", "256k", "512k", "1M")
DEFAULT_HITS = ("0", "0.1", "0.6", "0.7", "0.8", "0.9", "0.95", "0.96", "0.97", "0.98",
                "0.99", "0.991", "0.992", "0.993", "0.994", "0.995", "0.996", "0.997",
                "0.998", "0.999")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cp", type=int, default=1,
                    help="CP 大小；用于 ④ 的每卡口径 hit*L/cp（其余上限与 CP 无关）")
    ap.add_argument("--ls", default=",".join(DEFAULT_LS))
    ap.add_argument("--hits", default=",".join(DEFAULT_HITS))
    ap.add_argument("--rtail", type=int, default=140)
    ap.add_argument("--r-limit", type=int, default=0, help="覆盖 R 上限")
    ap.add_argument("--full-l-limit", type=int, default=0, help="覆盖 hit=0 的 L 上限")
    ap.add_argument("--min-per-card", type=int, default=128,
                    help="每卡新增 token 数 (1-hit)*L/cp 低于此值不采集（0=关闭，默认 128）")
    ap.add_argument("--report-skipped", action="store_true")
    args = ap.parse_args()

    rlim = args.r_limit or R_LIMIT
    flim = args.full_l_limit or FULL_L_LIMIT
    min_pc = max(0, int(args.min_per_card or 0))
    ls = [x for x in args.ls.split(",") if x.strip()]
    hits = [x for x in args.hits.split(",") if x.strip()]

    pts, skipped = [], []
    for Ls in ls:
        L = parse_l(Ls)
        if L % 128:
            sys.exit(f"✗ L 必须是 128 的倍数：{Ls}")
        for hs in hits:
            h = parse_hit(hs)
            M = int(round((1.0 - h) * L))
            R = M + args.rtail
            reason = ""
            if R > rlim:
                reason = f"R{rlim}<={R}"
            elif h == 0 and L > flim:
                reason = f"hit0&L>{flim}"
            elif h > 0 and R > L:
                reason = "R>L(裁尾超界)"
            elif min_pc and ((1.0 - h) * L / max(1, args.cp)) < min_pc:
                reason = f"(1-hit)*L/cp={(1.0 - h) * L / max(1, args.cp):.1f}<{min_pc}"
            if reason:
                skipped.append((L, h, M, R, reason))
                continue
            pts.append({"L": L, "hit": h, "batches": 1, "replay": True})

    if not pts:
        sys.exit("✗ 网格为空")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(pts, f, ensure_ascii=False, indent=1)
    print(f"  ✓ CP{args.cp}: {len(pts)}/{len(pts) + len(skipped)} 个点位 → {args.out}"
          f"（R≤{rlim}，hit0 且 L≤{flim}，(1-hit)*L/cp≥{min_pc}）")
    if args.report_skipped and skipped:
        with open(args.out + ".skipped.tsv", "w", encoding="utf-8") as f:
            f.write("seq_len\thit_rate\tper_die_tokens\tR\treason\n")
            for L, h, M, R, why in skipped:
                f.write(f"{L}\t{h}\t{M}\t{R}\t{why}\n")
        print(f"  ✓ 跳过清单 → {args.out}.skipped.tsv")


if __name__ == "__main__":
    main()

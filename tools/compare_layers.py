#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare_layers.py — 官方 bench 与 mini 的逐层 hidden 对拍工具。

用法:
  python3 tools/compare_layers.py --ref <目录或文件> --ours <目录或文件> \
      [--rank 0] [--top 10] [--logits]

输入约定（两边一致，不做兼容猜测）:
  collect_rank{R}.pt = torch.save({"layers": {层号: Tensor[T,H] fp32}, "rank": R, "n": 层数}, path)
  dump_rank{R}.pt    = torch.save({"hidden": Tensor, "logits": Tensor, ...}, path)   # --logits 时使用

  --ref/--ours 给目录时按上述文件名自动寻找；给文件时直接用该文件
  （--logits 时若给的是文件但没有 logits，会退到同目录的 dump_rank{R}.pt）。

输出:
  1) 逐层表: 层号 | ours max/mean abs | ref max/mean abs | 绝对误差 max/mean | 相对误差 | 状态
     相对误差 = mean(|ours-ref|) / mean(|ref|)
     状态: ✅ <1e-3   ⚠️ <5e-2   ❌ 其余（含 NaN）
  2) 末尾给"第一处分歧层"的建议行。
  3) 层集合/形状不一致时明确报错（列出缺哪些层/哪些层形状不同），绝不按位截断。
  4) --logits: 比 dump_rank*.pt 的 logits（top1/top5 token id+分数、top-K 重合度、max/mean 绝对误差）。
"""

import argparse
import os
import sys

import torch

OK_THRESHOLD = 1e-3      # ✅ 相对误差下限
WARN_THRESHOLD = 5e-2    # ⚠️ 上限，超过即 ❌


def die(msg, code=2):
    print("错误: %s" % msg, file=sys.stderr)
    raise SystemExit(code)


# --------------------------------------------------------------------------
# 加载
# --------------------------------------------------------------------------

def torch_load(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        # 老/新 torch 对 weights_only 默认值不同；dump 里可能有非基本类型
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            die("无法加载 %s: %s" % (path, e))


def load_payload(path, what):
    if not os.path.exists(path):
        die("%s 路径不存在: %s" % (what, path))
    return torch_load(path)


def resolve_collect(arg, rank):
    """--ref/--ours 的层 dump 路径解析。"""
    if os.path.isdir(arg):
        p = os.path.join(arg, "collect_rank%d.pt" % rank)
        if not os.path.exists(p):
            die("目录 %s 下找不到 collect_rank%d.pt" % (arg, rank))
        return p
    return arg


def resolve_dump(arg, rank):
    """--logits 时的 dump_rank{R}.pt 路径解析。"""
    if os.path.isdir(arg):
        p = os.path.join(arg, "dump_rank%d.pt" % rank)
        if not os.path.exists(p):
            die("目录 %s 下找不到 dump_rank%d.pt（--logits 需要）" % (arg, rank))
        return p
    payload = load_payload(arg, "输入")
    if isinstance(payload, dict) and "logits" in payload:
        return arg
    sibling = os.path.join(os.path.dirname(os.path.abspath(arg)),
                           "dump_rank%d.pt" % rank)
    if os.path.exists(sibling):
        return sibling
    die("%s 里没有 logits，且同目录下没有 dump_rank%d.pt（--logits 需要）"
        % (arg, rank))


def norm_key(k):
    if isinstance(k, bool):
        return k
    try:
        return int(k)
    except (TypeError, ValueError):
        return str(k)


def sort_key(k):
    return (0, k) if isinstance(k, int) else (1, str(k))


def fmt_list(keys):
    return ", ".join(str(k) for k in keys)


def load_layers(path, label):
    payload = load_payload(path, label)
    if not isinstance(payload, dict) or "layers" not in payload:
        die("%s %s 缺少 'layers' 字段（是否把 dump_rank 文件当成 collect 用了？）"
            % (label, path))
    raw = payload["layers"]
    if not isinstance(raw, dict) or len(raw) == 0:
        die("%s %s 的 'layers' 为空或不是 dict" % (label, path))
    layers = {}
    for k, v in raw.items():
        layers[norm_key(k)] = v
    n = payload.get("n", None)
    if isinstance(n, int) and n != len(layers):
        print("[警告] %s 的 n=%d 与实际层数 %d 不一致（%s）"
              % (label, n, len(layers), path), file=sys.stderr)
    return layers, payload


def as_f64(label, layer, v):
    if not torch.is_tensor(v):
        die("%s 层 %s 不是 Tensor（type=%s）" % (label, layer, type(v).__name__))
    if not v.is_floating_point():
        die("%s 层 %s dtype=%s 不是浮点" % (label, layer, v.dtype))
    return v.detach().to(device="cpu", dtype=torch.float64)


def abs_mm(t):
    if t.numel() == 0:
        return 0.0, 0.0
    a = t.abs()
    return float(a.max().item()), float(a.mean().item())


def status_of(rel_err):
    if rel_err != rel_err:  # NaN
        return "❌"
    if rel_err < OK_THRESHOLD:
        return "✅"
    if rel_err < WARN_THRESHOLD:
        return "⚠️"
    return "❌"


def common_shape(layers, keys):
    shapes = {tuple(layers[k].shape) for k in keys}
    if len(shapes) == 1:
        return str(next(iter(shapes)))
    return "各层不同"


# --------------------------------------------------------------------------
# 逐层对拍
# --------------------------------------------------------------------------

def compare_layers(ref_arg, ours_arg, rank):
    ref_path = resolve_collect(ref_arg, rank)
    ours_path = resolve_collect(ours_arg, rank)
    ref_raw, ref_payload = load_layers(ref_path, "ref ")
    ours_raw, ours_payload = load_layers(ours_path, "ours")
    declared = [p.get("rank") for p in (ref_payload, ours_payload)
                if isinstance(p.get("rank"), int)]
    disp_rank = declared[0] if declared and all(d == declared[0] for d in declared) else rank

    # 1) 层集合必须一致 —— 不一致直接报错，列出缺哪些层
    ref_keys = sorted(ref_raw, key=sort_key)
    ours_keys = sorted(ours_raw, key=sort_key)
    missing_in_ours = [k for k in ref_keys if k not in ours_raw]
    missing_in_ref = [k for k in ours_keys if k not in ref_raw]
    if missing_in_ours or missing_in_ref:
        lines = ["ref 与 ours 的层集合不一致，拒绝按位截断对拍:"]
        if missing_in_ours:
            lines.append("  ours 缺少（ref 有 %d 层，ours 有 %d 层）: %s"
                         % (len(ref_keys), len(ours_keys), fmt_list(missing_in_ours)))
        if missing_in_ref:
            lines.append("  ref  缺少（ours 有 %d 层，ref 有 %d 层）: %s"
                         % (len(ours_keys), len(ref_keys), fmt_list(missing_in_ref)))
        die("\n".join(lines), code=2)

    ref = {k: as_f64("ref ", k, ref_raw[k]) for k in ref_keys}
    ours = {k: as_f64("ours", k, ours_raw[k]) for k in ref_keys}

    # 2) 形状必须一致 —— 不广播、不截断
    bad_shape = []
    for k in ref_keys:
        if tuple(ref[k].shape) != tuple(ours[k].shape):
            bad_shape.append((k, tuple(ref[k].shape), tuple(ours[k].shape)))
    if bad_shape:
        lines = ["层形状不一致（不做广播/截断）:"]
        for k, rs, os_ in bad_shape:
            lines.append("  层 %s: ref=%s ours=%s" % (k, list(rs), list(os_)))
        die("\n".join(lines), code=2)

    print("=" * 108)
    print("逐层 hidden 对拍  (rank %s)" % disp_rank)
    print("  ref : %s" % ref_path)
    print("  ours: %s" % ours_path)
    print("  层数: %d    张量形状: %s (T,H)" % (len(ref_keys), common_shape(ref, ref_keys)))
    print("  相对误差 = mean(|ours-ref|) / mean(|ref|)    状态阈值: ✅ <1e-3, ⚠️ <5e-2, ❌ ≥5e-2")
    print("=" * 108)
    header = ("%4s | %21s | %21s | %21s | %12s | %s"
              % ("层号", "ours max/mean abs", "ref max/mean abs",
                 "绝对误差 max/mean", "相对误差", "状态"))
    print(header)
    print("-" * 108)

    rows = []
    for k in ref_keys:
        r, o = ref[k], ours[k]
        d = (o - r).abs()
        e_max, e_mean = abs_mm(d)
        r_max, r_mean = abs_mm(r)
        o_max, o_mean = abs_mm(o)
        if r_mean > 0:
            rel = e_mean / r_mean
        else:
            rel = 0.0 if e_max == 0 else float("inf")
        st = status_of(rel)
        rows.append((k, o_max, o_mean, r_max, r_mean, e_max, e_mean, rel, st))
        print("%4s | %10.3e/%10.3e | %10.3e/%10.3e | %10.3e/%10.3e | %12.3e | %s"
              % (k, o_max, o_mean, r_max, r_mean, e_max, e_mean, rel, st))
    print("-" * 108)

    n_ok = sum(1 for row in rows if row[8] == "✅")
    n_warn = sum(1 for row in rows if row[8] == "⚠️")
    n_bad = sum(1 for row in rows if row[8] == "❌")
    worst = max(rows, key=lambda row: (row[7] if row[7] == row[7] else float("inf"),
                                       row[5]))
    print("统计: ✅ %d 层, ⚠️ %d 层, ❌ %d 层    最差: 层 %s (相对误差 %.3e)"
          % (n_ok, n_warn, n_bad, worst[0], worst[7]))

    # 3) 第一处分歧层建议行
    first_div = next((row for row in rows if row[8] != "✅"), None)
    if first_div is None:
        print("第一处分歧层: 无 ✅ —— 全部 %d 层相对误差 < 1e-3" % len(rows))
    else:
        layer, _, _, _, _, _, _, rel, st = first_div
        prev = [row[0] for row in rows if sort_key(row[0]) < sort_key(layer)]
        tail = "（层 %s 及以下一致）" % prev[-1] if prev else "（最底层就分歧）"
        print("%s 第一处分歧层: 层 %s 相对误差 %.3e %s → 从这里往上查" % (st, layer, rel, tail))
        first_bad = next((row for row in rows if row[8] == "❌"), None)
        if first_bad is not None and first_bad[0] != layer:
            print("❌ 首次严重分歧: 层 %s 相对误差 %.3e"
                  % (first_bad[0], first_bad[7]))
    return first_div


# --------------------------------------------------------------------------
# logits 对拍
# --------------------------------------------------------------------------

def last_token_vec(t, name, path):
    if not torch.is_tensor(t):
        die("%s 的 logits 不是 Tensor（%s）" % (path, type(t).__name__))
    if t.dim() == 1:
        return t.detach().to(device="cpu", dtype=torch.float64), "1-D"
    if t.dim() == 0:
        die("%s 的 logits 是 0 维" % path)
    return (t.detach().reshape(-1, t.shape[-1])[-1].to(device="cpu", dtype=torch.float64),
            "末 token（原 shape=%s）" % list(t.shape))


def topk_ids_scores(v, k):
    k = min(k, v.numel())
    vals, ids = torch.topk(v, k)
    return [int(i) for i in ids.tolist()], [float(s) for s in vals.tolist()]


def compare_logits(ref_arg, ours_arg, rank, topk):
    ref_path = resolve_dump(ref_arg, rank)
    ours_path = resolve_dump(ours_arg, rank)
    ref_payload = load_payload(ref_path, "ref ")
    ours_payload = load_payload(ours_path, "ours")
    if not isinstance(ref_payload, dict) or "logits" not in ref_payload:
        die("ref %s 里没有 logits 字段" % ref_path)
    if not isinstance(ours_payload, dict) or "logits" not in ours_payload:
        die("ours %s 里没有 logits 字段" % ours_path)

    ref_v, ref_desc = last_token_vec(ref_payload["logits"], "ref", ref_path)
    ours_v, ours_desc = last_token_vec(ours_payload["logits"], "ours", ours_path)
    if ref_v.shape != ours_v.shape:
        die("logits 形状不一致（不做截断）: ref=%s ours=%s"
            % (list(ref_v.shape), list(ours_v.shape)))

    print()
    print("=" * 108)
    print("logits 对拍  (rank %d)" % rank)
    print("  ref : %s  logits 取 %s" % (ref_path, ref_desc))
    print("  ours: %s  logits 取 %s" % (ours_path, ours_desc))
    print("=" * 108)

    e_max, e_mean = abs_mm(ours_v - ref_v)
    r_max, r_mean = abs_mm(ref_v)
    rel = (e_mean / r_mean) if r_mean > 0 else (0.0 if e_max == 0 else float("inf"))
    print("logits 绝对误差: max=%.3e  mean=%.3e   相对误差 mean(|Δ|)/mean(|ref|)=%.3e  %s"
          % (e_max, e_mean, rel, status_of(rel)))
    print("  (ref |logits| max=%.3e mean=%.3e, vocab=%d)" % (r_max, r_mean, ref_v.numel()))

    r_ids, r_scores = topk_ids_scores(ref_v, max(5, topk))
    o_ids, o_scores = topk_ids_scores(ours_v, max(5, topk))

    ok1 = r_ids[0] == o_ids[0]
    print("top1 一致: %s  id ref=%d ours=%d   score ref=%.6e ours=%.6e  Δscore=%.3e"
          % ("✅" if ok1 else "❌", r_ids[0], o_ids[0],
             r_scores[0], o_scores[0], abs(r_scores[0] - o_scores[0])))

    set5 = len(set(r_ids[:5]) & set(o_ids[:5]))
    same_order5 = r_ids[:5] == o_ids[:5]
    print("top5 一致: %s  重合 %d/5  顺序一致: %s"
          % ("✅" if set5 == 5 else "❌", set5, "✅" if same_order5 else "❌"))
    print("  ref  top5: %s" % ", ".join("%d(%.4f)" % (i, s) for i, s in zip(r_ids[:5], r_scores[:5])))
    print("  ours top5: %s" % ", ".join("%d(%.4f)" % (i, s) for i, s in zip(o_ids[:5], o_scores[:5])))
    if topk != 5:
        setk = len(set(r_ids[:topk]) & set(o_ids[:topk]))
        print("top%d 重合度: %d/%d %s" % (topk, setk, topk, "✅" if setk == topk else "❌"))


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="官方 bench 与 mini 逐层 hidden（及 logits）对拍工具")
    ap.add_argument("--ref", required=True,
                    help="参考侧（官方 bench）的 collect/dump 目录或文件")
    ap.add_argument("--ours", required=True,
                    help="对比侧（mini）的 collect/dump 目录或文件")
    ap.add_argument("--rank", type=int, default=0, help="rank 号（默认 0）")
    ap.add_argument("--top", type=int, default=10,
                    help="logits top-K 重合度里的 K（默认 10；top1/top5 总是报）")
    ap.add_argument("--logits", action="store_true",
                    help="顺便比 dump_rank{R}.pt 的 logits")
    args = ap.parse_args()

    if args.rank < 0:
        die("--rank 必须 >= 0")
    if args.top < 1:
        die("--top 必须 >= 1")

    compare_layers(args.ref, args.ours, args.rank)
    if args.logits:
        compare_logits(args.ref, args.ours, args.rank, args.top)


if __name__ == "__main__":
    main()

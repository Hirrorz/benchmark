#!/usr/bin/env python3
"""从真权重索引生成 `weights_manifest.py` —— "每层该有哪些权重"的唯一事实来源。

真权重目录（476 GB / 48 分片 / 96085 tensor）里的 `model.safetensors.index.json` 给出
**名字 → 分片**；每个分片的 safetensors 头（前 8 字节 = 头长度）给出 **dtype + shape**。
本工具把 96k 个 tensor 归纳成：

    COMMON   : 40 层都有的 tensor（名字用 {L} 占位）
    COND     : 只在部分层有的（indexer/compressor …），带层号列表
    EXPERTS  : 每层 × n_experts 的专家权重（40×384×6 = 92160 个，不逐条列出）

用法：
  python3 tools/gen_manifest.py <权重目录> [-o weights_manifest.py]
  只读索引 + 分片头，不读权重数据本身（秒级）。
"""
import argparse
import collections
import json
import os
import re
import struct


def read_index(root):
    idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))
    return idx["weight_map"]


def read_headers(root, weight_map, shards=None):
    """返回 {tensor_name: (dtype, shape)}；只读分片头。"""
    by_file = collections.defaultdict(list)
    for n, f in weight_map.items():
        by_file[f].append(n)
    meta = {}
    files = sorted(by_file) if shards is None else sorted(by_file)[:shards]
    for f in files:
        with open(os.path.join(root, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k in by_file[f]:
            v = hdr.get(k) or {}
            if v:
                meta[k] = (v.get("dtype"), tuple(v.get("shape") or ()))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights_dir")
    ap.add_argument("-o", "--out", default="weights_manifest.py")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--experts", type=int, default=384)
    a = ap.parse_args()

    wm = read_index(a.weights_dir)
    meta = read_headers(a.weights_dir, wm)
    missing = len(wm) - len(meta)

    # 按层归集：layers.<i>.<suffix>
    per_layer = collections.defaultdict(dict)          # layer → {suffix: (dtype, shape)}
    experts = collections.defaultdict(dict)            # layer → {expert suffix: (dtype, shape)}
    for name, (dt, shp) in meta.items():
        m = re.match(r"layers\.(\d+)\.(.+)$", name)
        if not m:
            continue
        li, suffix = int(m.group(1)), m.group(2)
        if re.match(r"ffn\.experts\.\d+\.", suffix):
            experts[li][re.sub(r"ffn\.experts\.\d+\.", "ffn.experts.{E}.", suffix)] = (dt, shp)
        else:
            per_layer[li][suffix] = (dt, shp)

    layers = sorted(per_layer)
    allsuf = set().union(*per_layer.values()) if per_layer else set()
    common, cond = {}, collections.defaultdict(list)
    for suf in sorted(allsuf):
        holders = [li for li in layers if suf in per_layer[li]]
        shape = per_layer[holders[0]][suf]
        if len(holders) == len(layers):
            common[suf] = shape
        else:
            cond[suf] = (shape, holders)

    exp_spec = {}
    for li in sorted(experts):
        for suf, spec in experts[li].items():
            exp_spec.setdefault(suf, spec)

    def fmt(spec):  # (dtype, shape) → 源码字面量
        if spec is None:
            return "None"
        dt, shp = spec
        return f"({dt!r}, {tuple(shp)!r})"

    out = [
        '"""weights_manifest —— 每层该有哪些权重（**自动生成**，勿手改）。',
        '',
        f'源：{a.weights_dir}',
        f'    model.safetensors.index.json + {len(set(wm.values()))} 个分片头（{len(wm)} tensor）',
        f'生成：python3 tools/gen_manifest.py <权重目录>    （读头不读数据，秒级）',
        '',
        '用法：SyntheticSource 按本表造"本 rank 该有的那一份"；RealSource 按本表去索引里取。',
        '两条模式共用同一张表 + 同一套切分代码（PLAN §9 纪律）。',
        '"""',
        '',
        '# 40 层都有的 tensor：suffix → (dtype, shape)',
        'COMMON = {',
    ]
    for suf, spec in common.items():
        out.append(f"    {suf!r}: {fmt(spec)},")
    out += ['}', '', '# 只在部分层有的：suffix → ((dtype, shape), [层号…])', 'COND = {']
    for suf, (spec, holders) in cond.items():
        out.append(f"    {suf!r}: ({fmt(spec)}, {holders}),")
    out += ['}', '', '# 每层 × n_experts 的专家权重：suffix（{E}=专家号）→ (dtype, shape)', 'EXPERTS = {']
    for suf in sorted(exp_spec):
        out.append(f"    {suf!r}: {fmt(exp_spec[suf])},")
    out += ['}', '', '', 'def cond_holders(cfg):',
            '    """按 config 推导 COND 的 holder 层表；cfg=None → None（回退字面表）。"""',
            '    if cfg is None:',
            '        return None',
            '    import model_spec as _ms',
            '    s = _ms.build(cfg)',
            '    kv = list(s.kv_source_layers)',
            '    wg = [l for l in kv if 0 <= l < len(s.compress_ratios)',
            '          and int(s.compress_ratios[l]) == 2]',
            '    idx = list(s.index_source_layers)',
            '    eng = list(s.engram_layers)',
            '    return {',
            '        "attn.compressor.norm.weight": kv,',
            '        "attn.compressor.wkv.weight": kv,',
            '        "attn.compressor.wgate.weight": wg,',
            '        "attn.indexer.k_norm.weight": kv,',
            '        "attn.indexer.wk.weight": kv,',
            '        "attn.indexer.wq_b.scale": idx,',
            '        "attn.indexer.wq_b.weight": idx,',
            '        "attn.indexer.weights_proj.weight": idx,',
            '        "engram.embed.scale": eng,',
            '        "engram.embed.weight": eng,',
            '        "engram.k_weight": eng,',
            '        "engram.q_weight": eng,',
            '        "engram.wkv.scale": eng,',
            '        "engram.wkv.weight": eng,',
            '    }',
            '', '',
            'def layer_manifest(layer_idx, n_experts=None, e_range=None, cfg=None):',
            '    """本层该有哪些权重；cfg 非空 → COND 层表按 config 推导。"""',
            '    _holders = cond_holders(cfg)',
            '    out = [(s, dt, shp) for s, (dt, shp) in COMMON.items()]',
            '    for s, ((dt, shp), holders) in COND.items():',
            '        hs = _holders.get(s, holders) if _holders is not None else holders',
            '        if layer_idx in hs:',
            '            out.append((s, dt, shp))',
            '    if n_experts:',
            '        lo, hi = e_range or (0, n_experts)',
            '        for s, (dt, shp) in EXPERTS.items():',
            '            for e in range(lo, min(hi, n_experts)):',
            '                out.append((s.replace("{E}", str(e)), dt, shp))',
            '    return out', '']
    open(a.out, "w").write("\n".join(out) + "\n")
    print(f"写 {a.out}：COMMON {len(common)} 项 / COND {len(cond)} 项 / EXPERTS {len(exp_spec)} 项"
          + (f"；⚠️ {missing} 个 tensor 没读到头（--shards 限制？）" if missing else ""))
    print(f"层集合 {layers[0]}..{layers[-1]}（{len(layers)} 层）；专家后缀 {sorted(exp_spec)}")


if __name__ == "__main__":
    main()

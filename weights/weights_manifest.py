"""weights_manifest —— 每层该有哪些权重（**自动生成**，勿手改）。

源：/mnt/xds/sfs/DeepSeek-V4.1-Flash-cann-mxfp8-mxfp4
    model.safetensors.index.json + 48 个分片头（96085 tensor）
生成：python3 tools/gen_manifest.py <权重目录>    （读头不读数据，秒级）

用法：SyntheticSource 按本表造"本 rank 该有的那一份"；RealSource 按本表去索引里取。
两条模式共用同一张表 + 同一套切分代码（PLAN §9 纪律）。
"""

# 40 层都有的 tensor：suffix → (dtype, shape)
COMMON = {
    'attn.attn_sink': ('F32', (64,)),
    'attn.kv_norm.weight': ('BF16', (512,)),
    'attn.q_norm.weight': ('BF16', (1280,)),
    'attn.wkv.scale': ('F8_E8M0', (512, 160)),
    'attn.wkv.weight': ('F8_E4M3', (512, 5120)),
    'attn.wo_a.scale': ('F8_E8M0', (8192, 128)),
    'attn.wo_a.weight': ('F8_E4M3', (8192, 4096)),
    'attn.wo_b.scale': ('F8_E8M0', (5120, 256)),
    'attn.wo_b.weight': ('F8_E4M3', (5120, 8192)),
    'attn.wq_a.scale': ('F8_E8M0', (1280, 160)),
    'attn.wq_a.weight': ('F8_E4M3', (1280, 5120)),
    'attn.wq_b.scale': ('F8_E8M0', (32768, 40)),
    'attn.wq_b.weight': ('F8_E4M3', (32768, 1280)),
    'attn_norm.weight': ('BF16', (5120,)),
    'ffn.gate.bias': ('F32', (384,)),
    'ffn.gate.bias_vl': ('F32', (384,)),
    'ffn.gate.weight': ('BF16', (384, 5120)),
    'ffn.shared_experts.w1.scale': ('F8_E8M0', (2304, 160)),
    'ffn.shared_experts.w1.weight': ('F8_E4M3', (2304, 5120)),
    'ffn.shared_experts.w2.scale': ('F8_E8M0', (5120, 72)),
    'ffn.shared_experts.w2.weight': ('F8_E4M3', (5120, 2304)),
    'ffn.shared_experts.w3.scale': ('F8_E8M0', (2304, 160)),
    'ffn.shared_experts.w3.weight': ('F8_E4M3', (2304, 5120)),
    'ffn_norm.weight': ('BF16', (5120,)),
    'hc_attn_base': ('F32', (24,)),
    'hc_attn_fn': ('F32', (24, 20480)),
    'hc_attn_scale': ('F32', (3,)),
    'hc_ffn_base': ('F32', (24,)),
    'hc_ffn_fn': ('F32', (24, 20480)),
    'hc_ffn_scale': ('F32', (3,)),
}

# 只在部分层有的：suffix → ((dtype, shape), [层号…])
COND = {
    'attn.compressor.norm.weight': (('BF16', (512,)), [2, 8, 14, 20]),
    'attn.compressor.wgate.weight': (('BF16', (512, 5120)), [2, 8, 14]),
    'attn.compressor.wkv.weight': (('BF16', (512, 5120)), [2, 8, 14, 20]),
    'attn.indexer.k_norm.weight': (('BF16', (128,)), [2, 8, 14, 20]),
    'attn.indexer.weights_proj.weight': (('BF16', (32, 5120)), [2, 8, 14, 20, 24, 28, 32, 36]),
    'attn.indexer.wk.weight': (('BF16', (128, 512)), [2, 8, 14, 20]),
    'attn.indexer.wq_b.scale': (('F8_E8M0', (4096, 40)), [2, 8, 14, 20, 24, 28, 32, 36]),
    'attn.indexer.wq_b.weight': (('F8_E4M3', (4096, 1280)), [2, 8, 14, 20, 24, 28, 32, 36]),
    'engram.embed.scale': (('F8_E8M0', (384006168, 8)), [1, 14]),
    'engram.embed.weight': (('F8_E4M3', (384006168, 256)), [1, 14]),
    'engram.k_weight': (('BF16', (4, 5120)), [1, 14]),
    'engram.q_weight': (('BF16', (4, 5120)), [1, 14]),
    'engram.wkv.scale': (('F8_E8M0', (25600, 192)), [1, 14]),
    'engram.wkv.weight': (('F8_E4M3', (25600, 6144)), [1, 14]),
}

# 每层 × n_experts 的专家权重：suffix（{E}=专家号）→ (dtype, shape)
EXPERTS = {
    'ffn.experts.{E}.w1.scale': ('F8_E8M0', (2304, 160)),
    'ffn.experts.{E}.w1.weight': ('I8', (2304, 2560)),
    'ffn.experts.{E}.w2.scale': ('F8_E8M0', (5120, 72)),
    'ffn.experts.{E}.w2.weight': ('I8', (5120, 1152)),
    'ffn.experts.{E}.w3.scale': ('F8_E8M0', (2304, 160)),
    'ffn.experts.{E}.w3.weight': ('I8', (2304, 2560)),
}


_COND_CACHE: dict = {}


def cond_holders(cfg):
    """按 config 推导 COND 的 holder 层表（2026-10-19：不再依赖字面列表）。

    ★ 按签名缓存：`layer_manifest` 每层每 pass 都会被调用一次（40 层 × 每轮），
      不缓存的话每次 build 一个 ModelSpec ≈ 77 µs → 3 ms/pass 的纯 host 开销。

    规则（与官方 checkpoint 的权重分布一致，见 tools/gen_manifest.py）：
      · compressor norm/wkv   = kv_source_layers
      · compressor wgate      = kv_source_layers 中 compress_ratios==2 的层（c2a）
      · indexer k_norm/wk     = kv_source_layers（跑 qli 的层）
      · indexer wq_b/weights_proj = index_source_layers
      · engram.*              = engram_layers

    返回 {suffix: [layer_id…]}；`cfg=None` 时返回 None（调用方回退到字面 COND）。
    默认配置的推导结果与下方字面表**逐项相同**（tools/check_specs.py 会断言）。
    """
    if cfg is None:
        return None
    _m = cfg.get("model") or {}
    _key = (tuple(int(x) for x in (_m.get("kv_source_layers") or [])),
            tuple(int(x) for x in (_m.get("index_source_layers") or [])),
            tuple(int(x) for x in (_m.get("compress_ratios") or [])),
            tuple(int(x) for x in (_m.get("engram_layers") or [])))
    _hit = _COND_CACHE.get(_key)
    if _hit is not None:
        return _hit
    import model_spec as _ms
    s = _ms.build(cfg)
    kv = list(s.kv_source_layers)
    wg = [l for l in kv if 0 <= l < len(s.compress_ratios)
          and int(s.compress_ratios[l]) == 2]
    idx = list(s.index_source_layers)
    eng = list(s.engram_layers)
    _out = {
        'attn.compressor.norm.weight': kv,
        'attn.compressor.wkv.weight': kv,
        'attn.compressor.wgate.weight': wg,
        'attn.indexer.k_norm.weight': kv,
        'attn.indexer.wk.weight': kv,
        'attn.indexer.wq_b.scale': idx,
        'attn.indexer.wq_b.weight': idx,
        'attn.indexer.weights_proj.weight': idx,
        'engram.embed.scale': eng,
        'engram.embed.weight': eng,
        'engram.k_weight': eng,
        'engram.q_weight': eng,
        'engram.wkv.scale': eng,
        'engram.wkv.weight': eng,
    }
    _COND_CACHE[_key] = _out
    return _out


def mtp_layer_manifest(stage, n_stages=3, n_experts=128, e_range=None):
    """DSpark MTP 草稿层（`mtp.{stage}.*`）的权重清单。

    - 复用主模型 COMMON 的 attention/hc/norm，但 gate 宽度改成 n_experts（128）；
    - **不包含 COND**（MTP 没有 compressor/indexer/engram）；
    - 专家形状复用 EXPERTS（2304 intermediate，与主模型相同）；
    - stage 0 带 main_proj/main_norm，最后一个 stage 带 norm/markov/confidence。
    """
    out = []
    for s, (dt, shp) in COMMON.items():
        if s.startswith("ffn.gate."):
            shp = (int(n_experts), 5120) if s.endswith(".weight") else (int(n_experts),)
        out.append((s, dt, tuple(shp)))
    lo, hi = e_range or (0, int(n_experts))
    for s, (dt, shp) in EXPERTS.items():
        for e in range(lo, min(hi, int(n_experts))):
            out.append((s.replace("{E}", str(e)), dt, shp))
    if int(stage) == 0:
        out += [
            ("main_proj.weight", "F8_E4M3", (5120, 15360)),
            ("main_proj.scale", "F8_E8M0", (5120, 480)),
            ("main_norm.weight", "BF16", (5120,)),
        ]
    if int(stage) == int(n_stages) - 1:
        out += [
            ("norm.weight", "BF16", (5120,)),
            ("markov_head.embed.weight", "BF16", (129280, 256)),
            ("markov_head.head.weight", "BF16", (129280, 256)),
            ("confidence_head.proj.weight", "BF16", (1, 5376)),
        ]
    return out


def layer_manifest(layer_idx, n_experts=None, e_range=None, cfg=None):
    """本层该有哪些权重：[(name, dtype, shape)…]（name 里的 {E} 会按专家展开）。

    `cfg` 非空 → COND 的层表按 config 推导（`cond_holders`）；否则用字面表
    （离线工具 `tools/gen_manifest.py` 不需要 cfg）。
    """
    _holders = cond_holders(cfg)
    out = [(s, dt, shp) for s, (dt, shp) in COMMON.items()]
    for s, ((dt, shp), holders) in COND.items():
        hs = _holders.get(s, holders) if _holders is not None else holders
        if layer_idx in hs:
            out.append((s, dt, shp))
    if n_experts:
        lo, hi = e_range or (0, n_experts)      # EP 连续切分：本 rank 只认领 [lo,hi) 号专家
        for s, (dt, shp) in EXPERTS.items():
            for e in range(lo, min(hi, n_experts)):
                out.append((s.replace("{E}", str(e)), dt, shp))
    return out


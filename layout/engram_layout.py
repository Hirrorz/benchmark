"""engram 哈希布局与 n-gram id —— 逐字复刻官方 `modules/engram.py`（NgramHashState/EngramLayout）。

官方链条（prefill + tilelang，= 精度对拍用的那条）：
  NgramHashState.forward:
      ids → token_map[·]（压缩词表，大小写/空白归一后同形 token 合并）
      → shift_stack[T, max_ngram_size]（shift 0..3，越界/跨 image 的位置填 pad_id）
      → tile_kernels.engram.engram_hash([T,4] int32, multipliers, primes, offsets)
        → [n_layers, T, 24]（24 = (max_ngram_size-1) × n_heads = 3×8）
  再按本层取 [T,24] 作为 MultiHeadEmbedding 的取表 id。

本模块只做**主机侧**部分（不碰 NPU），可离线自检：
    python3.11 engram_layout.py --check
"""
from __future__ import annotations

import os

import numpy as np
import torch

# 官方 config（DeepSeek-V4.1-Flash）里的 engram 常数
# 官方 config（DeepSeek-V4.1-Flash）里的 engram 常数 —— 默认值 = 改造前的硬编码值。
# 2026-10-19：`configure(cfg)` 会按 model_spec 覆盖（缺省键 = 这里）；`DEFAULT_L`
# 保留出厂值，用于“参数是否被改过”的判断（缓存文件名/等价断言）。
DEFAULT_L = dict(
    layer_ids=(1, 14),
    num_embeddings=(384006168, 384016682),
    max_ngram_size=4,
    vocab_size=16000000,
    n_heads=8,
    head_dim=256,
    pad_token_id=2,
    compressed_vocab_size=99092,
)
L = dict(DEFAULT_L)


def configure(cfg=None):
    """按 model_spec 覆盖 engram 布局参数（缺省 = DEFAULT_L，行为逐位不变）。

    原地更新 `L`（exec_engram 以 `el.L[...]` 读，引用保持有效）。
    """
    import model_spec as _ms
    s = _ms.set_spec(cfg, strict=False) if cfg is not None else _ms.get()
    L.clear()
    L.update(DEFAULT_L)
    L.update(
        layer_ids=tuple(int(x) for x in s.engram_layers),
        num_embeddings=tuple(int(x) for x in s.engram_num_embeddings),
        max_ngram_size=int(s.engram_max_ngram),
        vocab_size=int(s.engram_vocab),
        n_heads=int(s.engram_n_heads),
        head_dim=int(s.engram_head_dim),
        pad_token_id=int(s.pad_token_id),
        compressed_vocab_size=int(s.compressed_vocab),
    )
    return L


def _params_digest() -> str:
    import hashlib
    import json
    blob = json.dumps({k: L[k] for k in sorted(L)}, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def _default_cache(name: str) -> str:
    """默认参数 → 原缓存文件名（行为不变）；改过参数 → 带 digest 后缀，避免读到旧布局。"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 树根（分包后）
    if L == DEFAULT_L:
        return os.path.join(base, name)
    stem, ext = os.path.splitext(name)
    return os.path.join(base, f"{stem}.{_params_digest()}{ext}")


# ─────────────────────── ① 压缩词表 ───────────────────────
def compressed_token_map(model_dir: str, cache: str = "") -> list:
    """官方 `build_compressed_token_map`：每个 token 解码→归一化→同形合并成新 id。"""
    cache = cache or _default_cache("cache_engram_token_map.pt")
    if os.path.exists(cache):
        return torch.load(cache, map_location="cpu", weights_only=True)
    from tokenizers import normalizers, Regex          # 官方 engram.py:38 同款导入
    from transformers import AutoTokenizer

    sentinel = ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    backend = tok.backend_tokenizer
    key_to_new, lookup = {}, [0] * len(tok)
    for token_id in range(len(tok)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            norm = normalizer.normalize_str(text)
            key = norm if norm else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    lookup = torch.tensor(lookup, dtype=torch.int32)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(lookup, cache)
    return lookup


# ─────────────────────── ② 桶布局/乘子（primes/offsets/multipliers）───────────────────────
def _is_prime(n: int) -> bool:
    """确定性 Miller-Rabin（n < 3.3e24 时用这组底数无误判）。"""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _next_prime(start: int, seen: set) -> int:
    c = start + 1
    while True:
        if c not in seen and _is_prime(c):
            return c
        c += 1


def build_layout(model_dir: str = "/mnt/xds/sfs/DeepSeek-V4.1-Flash-cann-mxfp8-mxfp4",
                 cache: str = "") -> dict:
    """primes / offsets / multipliers —— 与官方 `EngramLayout.from_args` +
    `compute_hash_multipliers` 逐字同口径（prime 全局不重复、按层×n-gram×head 顺序取）。"""
    cache = cache or _default_cache("cache_engram_layout.pt")
    if os.path.exists(cache):
        return torch.load(cache, map_location="cpu", weights_only=True)

    primes, seen = [], set()
    for _ in L["layer_ids"]:
        per_ngram = []
        for _ in range(L["max_ngram_size"] - 1):
            sizes, current = [], L["vocab_size"] - 1
            for _ in range(L["n_heads"]):
                current = _next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        primes.append(tuple(per_ngram))

    # offsets：每层把 24 个桶（3 n-gram × 8 head）按前缀和铺开
    flat = [[p for per_ngram in lay for p in per_ngram] for lay in primes]
    offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]

    # multipliers：每层一个 RNG（seed = 10007×layer_id），奇数化防溢出。
    # ★ bound 用的是**压缩词表**大小（官方 NgramHashState.__init__ 把
    #   build_compressed_token_map 返回的 vocab_size 传进来），不是 engram_vocab_size！
    bound = max(1, (np.iinfo(np.int64).max // L["compressed_vocab_size"]) // 2)
    rows = []
    for layer_id in L["layer_ids"]:
        g = np.random.default_rng(10007 * layer_id)
        rows.append(torch.tensor(g.integers(0, bound, size=(L["max_ngram_size"],),
                                            dtype=np.int64) * 2 + 1))
    out = dict(primes=torch.tensor(primes, dtype=torch.int32),              # [L,3,8]
               offsets=torch.tensor(np.array(offsets), dtype=torch.int32),  # [L,24]
               multipliers=torch.stack(rows))                               # [L,4]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(out, cache)
    return out


# ─────────────────────── ③ n-gram id ───────────────────────
def ngram_hash_ids(model_dir: str, input_ids: torch.Tensor, device="cpu") -> torch.Tensor:
    """→ [T, n_layers, 24] int32（先按 token_map 压缩，再走官方 engram_hash 内核）。

    单请求、无 image token（对拍场景）下 shift mask 退化为 `t >= k`：第 k 个 shift 只在
    t≥k 时可用，否则填 pad_id。
    """
    from tile_kernels.engram import engram_hash

    tmap = compressed_token_map(model_dir)
    lay = build_layout(model_dir)
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    ids = input_ids.reshape(-1).to(torch.int64)
    x = tmap.to(ids.device)[ids].view(1, -1)                       # [1,T]
    T = x.shape[1]
    # ★ 官方 pad_id = **token_map[engram_pad_token_id]**（压缩后的 pad），不是原始 id
    pad = int(tmap[L["pad_token_id"]].item())
    shifts = []
    for k in range(L["max_ngram_size"]):
        if k == 0:
            shifts.append(x)
        else:
            sh = torch.full_like(x, pad)
            sh[:, k:] = x[:, :T - k]
            shifts.append(sh)
    stack = torch.stack(shifts, dim=-1)                            # [1,T,4]
    pos = torch.arange(T, device=stack.device)
    mask = (pos[:, None] >= torch.arange(L["max_ngram_size"], device=stack.device)[None, :])
    stack = torch.where(mask.unsqueeze(0), stack, torch.full_like(stack, pad))
    # ★ 内核要求 NPU 张量：整体搬到 device 上再调（之前漏搬 → device_type mismatch，
    #   异常被逐算子容错吞掉后回落到零占位，层 1 就全错了）
    stack = stack.to(dev)
    h = engram_hash(stack.reshape(-1, L["max_ngram_size"]).to(torch.int32),
                    lay["multipliers"].to(dev),
                    lay["primes"].to(dev),
                    lay["offsets"].to(dev))                        # [L, T, 24]
    return h.permute(1, 0, 2).contiguous()                         # [T, L, 24]


def layer_index(layer_id: int) -> int:
    return L["layer_ids"].index(layer_id)


# ─────────────────────── 自检 ───────────────────────
def _check():
    # 仅离线自检用：模型目录默认取环境变量（换机器时不再改代码）
    model_dir = os.environ.get("DS41M_MODEL_DIR",
                               "/mnt/xds/sfs/DeepSeek-V4.1-Flash-cann-mxfp8-mxfp4")
    print("① 压缩词表 …")
    tmap = compressed_token_map(model_dir)
    n_comp = int(tmap.max()) + 1
    print(f"   token_map: {tuple(tmap.shape)} 压缩词表大小={n_comp} "
          f"（官方 config engram_compressed_vocab_size={L['compressed_vocab_size']}）"
          f" {'✅' if n_comp == L['compressed_vocab_size'] else '❌ 不一致！'}")
    print("② 桶布局 …")
    lay = build_layout(model_dir)
    pr = lay["primes"]
    print(f"   primes {tuple(pr.shape)} 版本={tuple(L['layer_ids'])} 唯一性={int(torch.unique(pr).numel())}/48"
          f" 全部 >16e6: {bool((pr > 16000000).all())}")
    print(f"   offsets 首层前 4 个 = {lay['offsets'][0][:4].tolist()}（应 0,16384159,32768891,49154197…）")
    print(f"   multipliers[0] = {lay['multipliers'][0].tolist()}")
    print("③ n-gram id（CPU 上也要能算，用于核对范围）…")
    ids = torch.arange(512, dtype=torch.long)
    try:
        h = ngram_hash_ids(model_dir, ids)
        print(f"   hash ids {tuple(h.shape)} min={int(h.min())} max={int(h.max())} "
              f"layer1 表行数={L['num_embeddings'][0]} 合法={bool((h < L['num_embeddings'][0]).all())}")
    except Exception as exc:                                   # noqa: BLE001
        print(f"   （需要 NPU 内核，跳过：{type(exc).__name__}: {exc}）")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        _check()

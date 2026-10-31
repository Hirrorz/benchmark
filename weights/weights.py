"""权重来源：合成 / 真实 —— **共用同一套切分与布局**（PLAN §9）。

两条硬纪律：
  1. 合成模式也走同一套 EP/TP 切分（造出"**本 rank 该有的那一份**"，形状/dtype/分布与真实一致）；
  2. 切分、布局转换、后端打包的代码只有一份 —— Source 只负责"tensor 从哪来"。

于是"合成能跑通"几乎等价于"真实能跑通"，两条路互为回归。

清单来自 `weights_manifest.py`（由真权重目录自动生成）；本文件不 import 任何官方仓库代码。

内存策略：专家权重每层 ~6.8 GB（mxfp4），**按层流式**（iter_layer / release），
单卡也能跑 —— 与 PLAN §9.0 的 `iter_layer()` 对应。

dtype 速查（真权重实况）：
  F8_E4M3  注意力 / 共享专家（mxfp8，权重 1 字节 + F8_E8M0 scale，32 元素一个 scale）
  I8       路由专家（mxfp4：一个 int8 装 2 个 fp4；如 [2304,2560] = [2304,5120]/2）
  BF16/F32 norm、gate、hc_*、engram.q/k_weight、compressor.wkv/wgate、indexer.wk
"""
from __future__ import annotations

import collections
import os
import re
import json
import struct

import torch

import weights_manifest as M

# ★ 2026-10-26（T-020 P1 / T2）：专家重切（EP↔TP、PD 重分布）时，除了清本模块的缓存，
#   还要通知下游缓存（如 exec_moe._PREP_GIDS 的“本 rank 全局专家号”）。
#   用注册表避免 weights → exec_moe 的循环导入；exec_moe 在导入时注册自己的清空函数。
_EXPERT_CACHE_CLEARERS: list = []


def register_expert_cache_clearer(fn) -> None:
    """注册一个「专家布局已失效」回调（幂等：同一个 fn 只登记一次）。"""
    if fn not in _EXPERT_CACHE_CLEARERS:
        _EXPERT_CACHE_CLEARERS.append(fn)


def _notify_expert_cache_cleared() -> None:
    for fn in list(_EXPERT_CACHE_CLEARERS):
        try:
            fn()
        except Exception:                    # noqa: BLE001 —— 回调失败不能影响权重失效
            pass


DTYPE = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.uint8,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "I8": torch.int8,
}
ELEM = {"F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2, "F32": 4, "I8": 1}


# ─────────────────────────── 切分（合成/真实共用）───────────────────────────
def ep_slice(n_experts: int, ep_rank: int, ep_size: int):
    """专家维**连续**切分（官方口径）：rank r 拿 [r*e_local, (r+1)*e_local)。"""
    e_local = n_experts // ep_size
    return ep_rank * e_local, (ep_rank + 1) * e_local


# 头维切分表：suffix → (维号, 该维总长)（只有 attn_tp>1 时才真的切）；未列出的 = 复制
HEAD_SPLIT = {
    "attn.wq_b.weight": (0, 32768), "attn.wq_b.scale": (0, 32768),   # 64 头 × 512
    "attn.wo_a.weight": (0, 8192), "attn.wo_a.scale": (0, 8192),     # 64 头 × 128
    "attn.wo_b.weight": (1, 8192), "attn.wo_b.scale": (1, 8192),
    "attn.attn_sink": (0, 64),
}


def _skip_moe_env() -> bool:
    return os.environ.get("DS41M_SKIP_MOE", "").strip().lower() in ("1", "on", "true", "yes")


def _skip_engram_env() -> bool:
    return os.environ.get("DS41M_SKIP_ENGRAM", "").strip().lower() in ("1", "on", "true", "yes")


def _filter_manifest(man):
    """跳 MoE / engram 时**不加载对应权重**（只影响性能口径）。"""
    if _skip_moe_env():
        man = [e for e in man if not e[0].startswith("ffn.")]
    if _skip_engram_env():
        man = [e for e in man if not e[0].startswith("engram.")]
    return man


def moe_tp_on(cfg=None) -> bool:
    """decode 低时延 TP 模式（官方 moe_infer_tp）：本 rank 拥有**所有**专家的 1/tp 中间维分片。

    ★ 判定看 **config 的 moe_tp_size**（而不是 env）：PD 分离时 prefill 用 EP（8），
      decode 前把 config 改成 tp=8/ep=1 → 同一进程内权重按新布局重载。
    """
    if cfg is not None:
        return int(((cfg.get("parallel_config") or {}).get("moe_tp_size", 1)) or 1) > 1
    return os.environ.get("DS41M_MOE_TP", "0") == "1"


def split_intermediate_aligned(total: int, parts: int, align: int) -> list:
    """官方 FusedMoEGMM.split_intermediate_aligned（module/fuse_moe_gmm.py:181）：
    按 align 为单位尽量均分，余数给前面的 rank。

    为什么必须对齐：`swiglu_group_quant` 要求末维（= 2×本 rank 的 intermediate 宽）
    能被 **256** 整除。2304/8=288 → 576 ✗；128 对齐后 rank0-1 各 384（768 ✓）、
    rank2-7 各 256（512 ✓），合计仍 2304。
    """
    units = total // align
    base, rem = divmod(units, parts)
    return [(base + 1) * align if r < rem else base * align for r in range(parts)]


def _tp_slice_expert(t: torch.Tensor, name: str, cfg, dim_offset=0) -> torch.Tensor:
    """专家权重按 **intermediate 维** 切（TP，非均匀 128 对齐，官方口径）：
        w1/w3.weight [I,H/2] / scale [I,H/32]  → dim 0
        w2.weight  [H,I/2]  / scale [H,I/32]   → dim 1
    w2 的行/列是**打包过**的（I/2、I/32），偏移与宽度按同一比例缩放。
    """
    pc = cfg["parallel_config"]
    tp = int(pc.get("moe_tp_size", 1) or 1)
    r = int(pc.get("_moe_tp_rank", 0) or 0)
    if tp <= 1:
        return t
    m = cfg["model"]
    I_full = int(m.get("moe_intermediate") or m.get("moe_intermediate_size") or 0)
    align = int(pc.get("moe_tp_align", 128) or 128)
    sizes = split_intermediate_aligned(I_full, tp, align)
    off_i, per_i = sum(sizes[:r]), sizes[r]
    dim = (1 if name.split(".")[-2].startswith("w2") else 0) + int(dim_offset)
    total = t.shape[dim]
    # 打包比例：w1/w3 是 I 维本身；w2 是 I/2；scale 是 I/32
    off = off_i * total // I_full
    per = per_i * total // I_full
    if per <= 0 or off + per > total:
        raise RuntimeError(f"TP 专家切分越界：{name} dim{dim}={total} off={off} per={per} tp={tp}")
    return t.narrow(dim, off, per).contiguous()


def shard(name: str, t: torch.Tensor, cfg) -> torch.Tensor:
    """把**全局** tensor 切成"本 rank 该有的那一份"。合成/真实都走这里。"""
    pc = cfg["parallel_config"]
    m = cfg["model"]
    if re.search(r"ffn\.experts\.\d+\.", name):
        if moe_tp_on(cfg):
            return _tp_slice_expert(t, name, cfg)
        return t          # 专家权重**整块属于某个 rank**：在名字级选择（见 expert_range），这里不切
    suf = re.sub(r"^layers\.\d+\.", "", name)
    hs = HEAD_SPLIT.get(suf)
    if hs and pc.get("attn_tp_size", 1) > 1:
        dim, total = hs
        tp, n = pc.get("_tp_rank", 0), pc["attn_tp_size"]
        per = total // n
        idx = torch.arange(tp * per, (tp + 1) * per, device=t.device)
        return t.index_select(dim, idx)
    if suf.startswith("engram.") and pc.get("engram_tp_size", 1) > 1:
        return t   # engram 表按 engram_tp_size 切（M4 落地）
    return t


# ─────────────────────────── 合成 ───────────────────────────
def _rand(dtype: str, shape, device):
    """按 dtype 造"分布合理"的随机数据（避开 NaN/Inf 位型）。

    用**各设备自己的全局 RNG**（NPU 上不接受 CPU generator）；复现性靠调用方
    `torch.manual_seed` / `torch.npu.manual_seed` 播种。
    """
    if dtype == "F8_E4M3":
        b = torch.randint(0, 127, shape, dtype=torch.uint8, device=device)   # 0x7F = NaN
        return b.view(torch.float8_e4m3fn)
    if dtype == "F8_E8M0":                       # uint8 存 2^(x-127)
        return torch.randint(120, 136, shape, dtype=torch.uint8, device=device)
    if dtype == "I8":
        return torch.randint(-120, 121, shape, dtype=torch.int8, device=device)
    if dtype == "BF16":
        return torch.randn(shape, dtype=torch.float32, device=device).to(torch.bfloat16)
    if dtype == "F32":
        return torch.randn(shape, dtype=torch.float32, device=device)
    raise KeyError(dtype)


# 常驻**主机内存**的权重（engram 表 ~98 GB/层，真实系统同样走 host-offload / ElasticBuffer）
HOST_RESIDENT = {"engram.embed.weight", "engram.embed.scale"}


def apply_manifest_override(cfg, rows):
    """把清单里 engram 表的第一维改成 rows（0 = 不改）。返回改了几项。

    表大小**不进算子图**（只改 gather 的输入，不改输出形状），所以联调时可以缩小；
    真权重模式忽略此项（按真实文件读）；表其余维不变。
    """
    if not rows:
        return 0
    n = 0
    for k, v in list(M.COMMON.items()):
        if k in HOST_RESIDENT:
            dt, shp = v
            M.COMMON[k] = (dt, (rows,) + tuple(shp[1:]))
            n += 1
    for k, v in list(M.COND.items()):
        if k in HOST_RESIDENT:
            (dt, shp), holders = v
            M.COND[k] = ((dt, (rows,) + tuple(shp[1:])), holders)
            n += 1
    return n


class SyntheticSource:
    """按 weights_manifest 造本 rank 的权重。`lazy_experts=True` 时专家权重按层生成/释放。"""

    kind = "synthetic"

    def __init__(self, cfg, device="npu:0", seed=0, lazy_experts=True, verbose=False):
        apply_manifest_override(cfg, int((cfg.get("weights") or {}).get("engram_table_rows", 0) or 0))
        self.cfg = cfg
        self.device = device
        torch.manual_seed(seed)
        if device.startswith("npu"):
            torch.npu.manual_seed(seed)
        self.lazy_experts = lazy_experts
        # 与 RealSource 同名同义（iter_layer 换层时会读它）：resident=true 时全部权重常驻、不释放
        self.resident = bool((cfg.get("weights") or {}).get("resident", False))
        # 跨层共用同一份假权重（缓存键去掉 `layers.<i>.` 前缀）：
        # 合成权重本身没有"层"的含义，40 层各造一份纯属浪费——共用后生成只发生一次、
        # 也不需要在换层时 offload/重载（性能采集时每轮前向的耗时才是干净的）。
        self.share = bool((cfg.get("weights") or {}).get("share_across_layers", False))
        self.verbose = verbose
        self._layer = None
        self._cache: dict[str, torch.Tensor] = {}
        self.count = collections.Counter()

    # — 公开接口 —
    def expert_range(self):
        """本 rank 认领的专家区间（EP 连续切分；单卡 = 全部）。"""
        pc = self.cfg["parallel_config"]
        if moe_tp_on(self.cfg):
            return 0, int(self.cfg["model"]["n_routed_experts"])   # TP：全部专家（按中间维切片）
        return ep_slice(self.cfg["model"]["n_routed_experts"],
                        pc.get("_ep_rank", 0), pc.get("moe_ep_size", 1))

    def get(self, name, dtype, shape, layer_idx=None):
        key = suf_module(name) if self.share else name     # 共用时按"去层号"的名字缓存
        if key in self._cache:
            return self._cache[key]
        dev = "cpu" if suf_module(name) in HOST_RESIDENT else self.device
        t = _rand(dtype, shape, dev)
        t = shard(name, t, self.cfg)
        self._cache[key] = t
        self.count[dtype] += 1
        if self.verbose:
            print(f"    [synth] {name:<44} {dtype:<8} {tuple(t.shape)}")
        return t

    def iter_layer(self, layer_idx, n_experts=None):
        """本层所有权重：[(name, dtype, shape)] → dict[name] = tensor。

        重复调用同一层会命中缓存；换层时释放上一层（lazy_experts）。
        """
        if (self._layer is not None and self._layer != layer_idx
                and self.lazy_experts and not self.resident and not self.share):
            self.release()          # resident=true / 跨层共用时不释放
        self._layer = layer_idx
        # ★ 2026-10-18 修复：lazy_experts 只决定“换层是否释放”，**不决定本层有哪些权重**。
        #   以前 resident 时传 None → layer_manifest 认为“没有专家权重”→ mega_moe 报
        #   “本 rank 权重里没有 ffn.experts.*”。EP 切分永远按 [lo,hi) 生效。
        man = _filter_manifest(M.layer_manifest(layer_idx, n_experts, e_range=self.expert_range(), cfg=self.cfg))
        prefix = f"layers.{layer_idx}."
        out = {}
        # 专家权重：同 (dtype, shape) 的合成**一个大 tensor 再切片** —— 否则每层 2300 个
        # RNG 内核对 trace 是灾难（实测 40 层 92k 个 StatelessRandom 淹没流水图）。
        bulk = collections.defaultdict(list)
        for suf, dtype, shape in man:
            if "ffn.experts." in suf:
                bulk[(dtype, tuple(shape))].append(suf)
        done = set()
        for (dtype, shape), sufs in bulk.items():
            ck = ("__experts_bulk__", dtype, tuple(shape)) if self.share else None
            if ck is not None and ck in self._cache:
                big = self._cache[ck]
            else:
                _eng_dev, _eng_tp, _eng_per = engram_placement(self.cfg)
                _is_eng = suf_module(prefix + sufs[0]) in HOST_RESIDENT
                if _is_eng and _eng_dev:
                    # 设备常驻 + 按 tp 分片：**每张卡只造自己那 1/tp 行**（与官方加载口径一致）
                    dev = self.device
                    shape = (_eng_per,) + tuple(shape[1:])
                else:
                    dev = "cpu" if _is_eng else self.device
                big = _rand(dtype, (len(sufs),) + shape, dev)
                if moe_tp_on(self.cfg):
                    # ★ 批量 tensor 的 dim0 = 专家堆叠；TP 要切的是权重维（整体 +1）
                    big = _tp_slice_expert(big, prefix + sufs[0], self.cfg, dim_offset=1)
                else:
                    big = shard(prefix + sufs[0], big, self.cfg)
                if ck is not None:
                    self._cache[ck] = big
            for i, suf in enumerate(sufs):
                out[prefix + suf] = big[i]
                done.add(suf)
        for suf, dtype, shape in man:
            if suf not in done:
                out[prefix + suf] = self.get(prefix + suf, dtype, shape, layer_idx)
        return out

    def invalidate_experts(self):
        """丢掉合成权重里的专家缓存（EP↔TP 切换时形状会变，必须重造）。"""
        import re as _re
        pat = _re.compile(r"(^|\.)ffn\.experts\.")
        gone = [k for k in self._cache
                if (isinstance(k, str) and (pat.search(k) or k == "__experts_bulk__"))
                or (isinstance(k, tuple) and k and k[0] == "__experts_bulk__")]
        for k in gone:
            del self._cache[k]
        _notify_expert_cache_cleared()
        return len(gone)

    def release(self):
        n = len(self._cache)
        self._cache.clear()
        if self.device.startswith("npu"):
            torch.npu.empty_cache()
        if self.verbose and n:
            print(f"    [synth] 释放 {n} 个 tensor")

    def summary(self):
        tot = sum(v for v in self.count.values())
        return [f"合成权重：{tot} 个 tensor（dtype 分布 {dict(self.count)}）",
                "  按层流式生成，形状/dtype 取自 weights_manifest（= 真权重清单）"]

class RealSource:
    """从真权重目录按需读（safetensors 头 + 分片，lazy）。"""

    kind = "real"

    def __init__(self, cfg, root=None, device="npu:0", verbose=False, lazy_experts=True):
        self.dropped: set = set()      # decode NZ 缓存接管后不再向磁盘重取的专家 key
        _w = cfg.get("weights") or {}
        # lazy_experts=false（多卡默认）→ 等同 resident：换层不释放、权重一直留在卡上
        # （真权重 EP8 下每卡只有 ~4.6 GB，98 GB 的卡装得下；省掉每层重新 H2D 的活）
        self.resident = bool(_w.get("resident", False)) or not bool(lazy_experts)
        # ★ host 缓存：第一次从 NFS 读进**主机内存**（pinned），之后每层从内存拷到卡
        #   （NFS ~1 GB/s → 内存拷贝 10+ GB/s；EP8 全模型 ~37 GB，本机 1.5 TB 内存足够）
        self.host_cache_on = bool(_w.get("host_cache", True))
        # 常驻服务：AC/NZ 已是唯一副本，逐 rank pin 33.6 GiB 专家权重纯属浪费；
        # bench_server 会设 DS41M_WEIGHT_HOST_CACHE=0（单跑默认不变）。
        _hc_env = os.environ.get("DS41M_WEIGHT_HOST_CACHE", "").strip().lower()
        if _hc_env in ("0", "off", "false", "no"):
            self.host_cache_on = False
        self._host: dict = {}
        self.host_bytes = 0
        self.cfg = cfg
        self.device = device
        self.verbose = verbose
        self.root = root or cfg["weights"]["path"]
        idx = json.load(open(os.path.join(self.root, "model.safetensors.index.json")))
        self.map = idx["weight_map"]
        self._handles = {}
        self._layer = None
        self._cache = {}
        try:
            from safetensors import safe_open
            self._open = safe_open
        except ImportError as exc:                       # pragma: no cover
            raise SystemExit(f"真权重模式需要 safetensors：{exc}")
        # ── 跨进程 MoE 常驻缓存（tools/moe_host_cache.py）─────────────────
        self._mc_mm = None
        self._mc_man = None
        self._mc_hits = 0
        self._mc_attach()

    def _mc_attach(self):
        """连接 `DS41M_MOE_HOST_CACHE` 指定的 daemon，mmap 其 memfd（只读）。

        缓存里是全量/部分 MoE 专家权重；`get()` 发现缺专家权重时优先从这里取，
        不读共享盘、也不再把整份权重 pin 进本进程的 `_host`（页由所有 rank 共享）。
        """
        path = os.environ.get("DS41M_MOE_HOST_CACHE", "").strip()
        if not path or path.lower() in ("0", "off", "false", "no"):
            return
        if path.lower() in ("1", "on", "true", "yes", "auto"):
            path = "/tmp/ds41_moe_host_cache.sock"
        try:
            import array
            import mmap as _mmap
            import socket as _socket
            import struct
            import zlib

            c = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            c.settimeout(60)
            c.connect(path)
            c.sendall(b"GET\n")
            hdr = b""
            while len(hdr) < 8:
                b = c.recv(8 - len(hdr))
                if not b:
                    raise RuntimeError("socket closed (header)")
                hdr += b
            n = struct.unpack("<Q", hdr)[0]
            blob = b""
            while len(blob) < n:
                b = c.recv(min(1 << 20, n - len(blob)))
                if not b:
                    raise RuntimeError("socket closed (manifest)")
                blob += b
            meta = json.loads(zlib.decompress(blob))
            _msg, anc, _fl, _ad = c.recvmsg(
                1, _socket.CMSG_SPACE(array.array("i", [0]).itemsize))
            fds = array.array("i")
            for level, typ, data in anc:
                if level == _socket.SOL_SOCKET and typ == _socket.SCM_RIGHTS:
                    fds.frombytes(data[:len(data) - len(data) % fds.itemsize])
            if not fds:
                raise RuntimeError("未收到 memfd")
            fd = fds[0]
            total = int(meta["total"])
            mm = _mmap.mmap(fd, total, access=_mmap.ACCESS_READ)
            os.close(fd)
            c.close()
            if os.path.realpath(str(meta.get("model", ""))) != os.path.realpath(self.root):
                print(f"  [moe-host-cache] ⚠️ 模型路径不一致（cache={meta.get('model')} "
                      f"run={self.root}）→ 忽略缓存", flush=True)
                mm.close()
                return
            self._mc_mm = mm
            self._mc_man = dict(meta["keys"])
            print(f"  [moe-host-cache] 已连接 {path}：{len(self._mc_man)} 个专家张量 / "
                  f"{total / 2**30:.1f} GiB（缺 MoE 权重时优先取这里）", flush=True)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [moe-host-cache] 连接失败（{type(exc).__name__}: {exc}）→ 走原路径",
                  flush=True)

    # ── 常驻服务：host cache 热挂接 / 解绑（idle 安全点调用）──────────────
    def hostcache_attach(self, path=None):
        """可重入地连接 moe_host_cache daemon（不重启 rank）。

        `--serve` 下由 bench_cli hostcache-apply 触发；只应在 idle 时调用。
        """
        if path:
            os.environ["DS41M_MOE_HOST_CACHE"] = str(path)
        if self._mc_mm is not None:
            return {"ok": True, "already": True, "keys": len(self._mc_man or {}),
                    "path": os.environ.get("DS41M_MOE_HOST_CACHE", "")}
        self._mc_attach()
        return {"ok": self._mc_mm is not None,
                "keys": len(self._mc_man or {}),
                "hits": int(getattr(self, "_mc_hits", 0)),
                "path": os.environ.get("DS41M_MOE_HOST_CACHE", "")}

    def hostcache_detach(self):
        """解绑 memfd（所有 rank 都解绑后 daemon 的页才会真正释放）。"""
        mm = self._mc_mm
        self._mc_mm = None
        self._mc_man = None
        if mm is not None:
            try:
                mm.close()
            except Exception:                          # noqa: BLE001
                pass
        return {"ok": True, "hits": int(getattr(self, "_mc_hits", 0))}

    def _handle(self, shard_file):
        if shard_file not in self._handles:
            self._handles[shard_file] = self._open(
                os.path.join(self.root, shard_file), framework="pt", device="cpu")
        return self._handles[shard_file]

    def expert_range(self):
        pc = self.cfg["parallel_config"]
        if moe_tp_on(self.cfg):
            return 0, int(self.cfg["model"]["n_routed_experts"])   # TP：全部专家（按中间维切片）
        return ep_slice(self.cfg["model"]["n_routed_experts"],
                        pc.get("_ep_rank", 0), pc.get("moe_ep_size", 1))

    def get(self, name, dtype, shape, layer_idx=None):
        if name in self._cache:
            return self._cache[name]
        # ★ 跨进程常驻缓存优先：MoE 专家权重缺失时先从 memfd（moe_host_cache daemon）取。
        #   零拷贝视图 → 按需 TP 切片 → 上卡；不读共享盘，也不 pin 进本进程 `_host`。
        mc = self._mc_man.get(name) if self._mc_man is not None else None
        if mc is not None:
            dt_s, shp_c, off_c, nb_c = mc
            t = torch.frombuffer(self._mc_mm, dtype=torch.uint8,
                                 count=int(nb_c), offset=int(off_c))
            dt_c = DTYPE.get(str(dt_s))
            if dt_c is not None and dt_c is not torch.uint8:
                t = t.view(dt_c)
            t = t.reshape(tuple(int(x) for x in shp_c))
            self._mc_hits += 1
            if self._mc_hits == 1:
                print(f"  [moe-host-cache] 命中：{name} {tuple(t.shape)}", flush=True)
            t = shard(name, t, self.cfg)
            t = t.to(self.device)
            self._cache[name] = t
            return t
        f = self.map.get(name)
        if f is None:
            raise KeyError(f"真权重里没有 {name}（清单与目录不一致？）")
        # 注意：不能用 `with self._handle(f)` —— safe_open 的上下文管理退出会关掉文件，
        # 而句柄是缓存复用的（第二次取同分片的 tensor 就报 "File is closed"）。
        t = self._handle(f).get_tensor(name)
        # ★ 归一化 scale dtype：真权重的 E8M0 scale 从这里读出来是 torch.float8_e8m0fnu，
        # 而对该 dtype 做 contiguous()/transpose 走的是 AICPU fallback，会直接崩设备
        # （实测 507018 aicpu exception）。统一成 uint8（1 字节，语义不变）：
        # 下游 reshape_mx_scale/转置都在 uint8 上做，喂算子时再 .view(float8_e8m0fnu)，
        # 与官方调用点一致。
        if t.dtype == getattr(torch, "float8_e8m0fnu", None):
            t = t.view(torch.uint8)
        # ★ 先切片再进 host 缓存 / 上卡：TP 专家权重每卡只要 1/8（safetensors 是 mmap，
        #   切片只触需要的页）。以前顺序是「读整张 → pin 整张 → 再切」→ EP→TP 换形态时
        #   每 rank 要把 384 专家的**全量**权重读进 pinned 内存（8 rank 合计 ~1.9 TB），
        #   实测 40 层 decode 转换 >6 min 还卡在前 10 层。切完再 pin 后，每 rank 只读
        #   自己那 1/8（合计 ~1× 模型大小）。
        t = shard(name, t, self.cfg)
        # host 缓存命中 → 直接从内存拷到卡；未命中 → 读盘后放进去（host-resident 的
        # engram 表本来就是 mmap，零拷贝，不 pin 也不搬）
        host_res = suf_module(name) in HOST_RESIDENT
        if not host_res:
            cached = self._host.get(name)
            if cached is not None:
                t = cached
            elif self.host_cache_on:
                try:
                    t = t.pin_memory()          # 固定页：H2D 更快（只 pin 本 rank 的分片）
                except Exception:               # noqa: BLE001
                    pass
                self._host[name] = t
                self.host_bytes += t.numel() * t.element_size()
        t = t if host_res else t.to(self.device)
        self._cache[name] = t
        if self.verbose:
            print(f"    [real] {name:<44} {dtype:<8} {tuple(t.shape)} ← {f}")
        return t

    def iter_layer(self, layer_idx, n_experts=None):
        if self._layer is not None and self._layer != layer_idx and not self.resident:
            self.release()
        self._layer = layer_idx
        man = _filter_manifest(M.layer_manifest(layer_idx, n_experts, e_range=self.expert_range(), cfg=self.cfg))
        prefix = f"layers.{layer_idx}."
        out = {}
        for suf, dt, shp in man:
            full = prefix + suf
            if full in self.dropped:      # decode NZ 缓存已接管，别向磁盘重取
                continue
            out[full] = self.get(full, dt, shp, layer_idx)
        return out

    def invalidate_experts(self, clear_host: bool = False):
        """PD 分离：丢掉**所有层**的原始专家权重缓存（含 dropped 记录），下次 iter_layer 按新
        布局（EP↔TP）重新加载。注意力/engram 等其它权重不受影响。

        `clear_host`：host 缓存里存的是**按当前形态切好的分片**（EP=整专家 / TP=中间维 1/tp）。
          · EP→TP / TP→EP 换形态：必须 True，否则会把旧布局的分片当新布局用；
          · EP→EP（保持布局）：False，保留 host 缓存 → 免去 34 GB/rank 的重读
            （实测 EP 的 P→D 因此从 47 s 回到 ~4 s）。
        """
        pat = re.compile(r"\.ffn\.experts\.")
        gone = [k for k in self._cache if pat.search(k)]
        for k in gone:
            del self._cache[k]
        if clear_host:
            hgone = [k for k in self._host if pat.search(k)]
            for k in hgone:
                t = self._host.pop(k)
                self.host_bytes -= t.numel() * t.element_size()
        self.dropped.clear()
        _notify_expert_cache_cleared()
        if gone and self.device.startswith("npu"):
            torch.npu.empty_cache()
        if self.verbose:
            print(f"    [real] PD 换形态：清空 {len(gone)} 个专家权重缓存")
        return len(gone)

    def mark_all_experts_dropped(self, layer_idx, n_experts, cfg=None):
        """把本层**全部**专家的所有键标记为 dropped（decode 步不再回读磁盘）。

        TP all2all 专用：a2a 在 EP 布局下只加载了本 rank 的 48 个专家；切到 TP 后
        manifest 会列出全部 384 个。若只 drop 已加载的 48 个，decode 步的 `iter_layer`
        会为另外 336 个重新按 TP 布局读盘（实测第二步就重读 ~30 GB/rank → 极慢/OOM）。
        """
        man = M.layer_manifest(int(layer_idx), int(n_experts),
                               e_range=(0, int(n_experts)), cfg=cfg)
        pre = f"layers.{int(layer_idx)}."
        n = 0
        for suf, _dt, _shp in man:
            if "ffn.experts." in suf:
                self.dropped.add(pre + suf)
                n += 1
        return n

    def drop_layer_experts(self, layer_idx):
        """decode：释放本层原始专家权重（NZ 缓存同尺寸替代 ⇒ 显存净增 0）。

        被释放的名字记入 `self.dropped`，`iter_layer` 不再重取（否则每步从盘上重载 = 灾难）。
        """
        pat = re.compile(rf"^layers\.{int(layer_idx)}\.ffn\.experts\.")
        gone = [k for k in self._cache if pat.match(k)]
        for k in gone:
            del self._cache[k]
            self.dropped.add(k)
        if gone and self.device.startswith("npu"):
            torch.npu.empty_cache()
        if gone and self.verbose:
            print(f"    [real] 层 {layer_idx}：释放原始专家权重 {len(gone)} 项（NZ 缓存接管）")
        return len(gone)

    def drop_all_experts(self, n_layers=None):
        """一次性释放**所有层**的原始专家权重（常驻服务 W2 后：AC + decode NZ 已接管）。

        与 `drop_layer_experts` 同一套账：名字记入 `dropped` → `iter_layer` 不再重取。
        只调用一次 `empty_cache`，避免 40 层逐层 flush 的停顿。
        """
        pat = re.compile(r"^layers\.\d+\.ffn\.experts\.")
        gone = [k for k in self._cache if pat.match(k)]
        for k in gone:
            del self._cache[k]
            self.dropped.add(k)
        if gone and self.device.startswith("npu"):
            torch.npu.empty_cache()
        if self.verbose:
            print(f"    [real] 释放全部原始专家权重 {len(gone)} 项（AC/NZ 接管）", flush=True)
        return len(gone)

    def drop_names(self, names, empty_cache=False):
        """释放**指定名字**的原始权重（转置/NZ/融合副本已接管时用）。

        与 `drop_layer_experts` 同一套账：名字记入 `self.dropped` → `iter_layer` 不再取，
        避免下一 pass 从盘上重读。`empty_cache` 默认关：PyTorch 会把释放的块直接复用，
        真要归还驱动时由已有的 AC 释放路径统一 flush（避免每层多一次 flush 引发停顿）。
        """
        host = getattr(self, "_host", None)
        gone = 0
        for n in names:
            if n in self._cache:
                del self._cache[n]
                self.dropped.add(n)
                gone += 1
            if host is not None:
                host.pop(n, None)          # --host-cache 时主机侧那份也一起放掉
        if gone and empty_cache and self.device.startswith("npu"):
            torch.npu.empty_cache()
        if gone and self.verbose:
            print(f"    [real] 释放原始权重 {gone} 项（转换产物接管）")
        return gone

    def release(self):
        self._cache.clear()
        self.dropped.clear()
        if self.device.startswith("npu"):
            torch.npu.empty_cache()

    def summary(self):
        return [f"真权重：{self.root}", f"  {len(self.map)} 个 tensor / {len(set(self.map.values()))} 分片",
                "  只读本 rank 切片（lazy，按层释放）"]


def suf_module(name: str) -> str:
    """去掉 `layers.<i>.` 前缀（HOST_RESIDENT 等表按这个查）。"""
    return re.sub(r"^layers\.\d+\.", "", name)


# ─────────────────────────── 工厂 ───────────────────────────
def lazy_experts_of(cfg):
    """`weights.lazy_experts`：专家权重**逐层生成/释放**（单卡省显存）还是全程常驻。

    ★ 默认按卡数自动（2026-10-15 用户口径）：**多卡默认不逐层加载**（性能采集要的是干净
    稳态，逐层重造专家是纯开销），**单卡才默认逐个来**（显存装不下 40 层专家）。
    ★ 2026-10-18 用户口径：**EP8 默认不开按层加载（显存足够）** —— 多卡（world>=2）即常驻；
    各 EP8 配置里的显式 `lazy_experts: true` 已删除，让本函数按运行卡数决定。
    需要 A/B 时在 YAML 里显式写 `lazy_experts: true/false` 覆盖。
    """
    w = cfg.get("weights") or {}
    v = w.get("lazy_experts", None)
    if v is not None:
        return bool(v)
    return int((cfg.get("parallel_config") or {}).get("world_size", 1) or 1) <= 1


_ENGRAM_ROWS_CACHE: dict = {}


def engram_table_rows(cfg, root=None) -> dict:
    """真权重里**所有 engram 层的表行数** → `{layer_idx: rows}`（只读 safetensors 头，秒级）。

    为什么需要：厂商 ElasticBuffer 的主机窗口是**进程级共享池、按第一个创建者定尺寸、重建不得超首建**；
    而两个 engram 层的真表行数并不相同（384006168 vs 384016682，见 `docs/WEIGHTS_INVENTORY.md`）——
    `exec_engram._vendor_buffer` 必须按**最大层**申请，否则第二层建时会
    "engram buffer size exceeds the existing buffer of this group … rolled back"。

    合成模式返回 {}（表是现造的，行数从配置走）。
    """
    root = root or (cfg.get("weights") or {}).get("path") or ""
    key = os.path.abspath(str(root))
    if key in _ENGRAM_ROWS_CACHE:
        return _ENGRAM_ROWS_CACHE[key]
    out: dict = {}
    try:
        with open(os.path.join(key, "model.safetensors.index.json")) as fh:
            wmap = json.load(fh)["weight_map"]
        from safetensors import safe_open
        handles: dict = {}
        for name, shard in wmap.items():
            m = re.match(r"^layers\.(\d+)\.engram\.embed\.weight$", name)
            if not m:
                continue
            h = handles.get(shard)
            if h is None:
                h = handles[shard] = safe_open(os.path.join(key, shard),
                                               framework="pt", device="cpu")
            out[int(m.group(1))] = int(h.get_slice(name).get_shape()[0])
    except Exception as exc:                              # noqa: BLE001
        print(f"  [engram] 读真表行数失败（{type(exc).__name__}: {exc}）→ 池按当前层申请", flush=True)
    _ENGRAM_ROWS_CACHE[key] = out
    return out


def engram_placement(cfg):
    """engram 表放哪：(on_device, tp_size, rows_per_rank)。

    ★ 官方 bench（light_8card_ep8_eager_128k.yaml:25-26）= `enable_engram_offload: False`
      + `engram_tp_size: 8`：表**常驻设备**、按 8 卡分片（每卡 1/8 行），取行在设备上做 +
      跨卡 all_reduce 合并。我们以前一律 host-resident + 主机取整表 → 每层 302 MB + 33 ms。
    """
    w = cfg.get("weights") or {}
    m = cfg.get("model") or {}
    on_dev = not bool(m.get("engram_offload", True))
    tp = max(1, int(m.get("engram_tp_size", 1) or 1))
    rows = int(w.get("engram_table_rows", 0) or 0)
    per = (rows // tp) if rows else 0
    return on_dev, tp, per


def make_source(cfg, device="npu:0", verbose=False):
    mode = (cfg.get("weights") or {}).get("mode", "synthetic")
    lazy = lazy_experts_of(cfg)
    if mode == "synthetic":
        return SyntheticSource(cfg, device=device, lazy_experts=lazy, verbose=verbose)
    if mode == "real":
        return RealSource(cfg, device=device, lazy_experts=lazy, verbose=verbose)
    raise SystemExit(f"未知 weights.mode={mode!r}（synthetic | real）")


def layer_bytes(manifest_rows):
    """本层权重总字节数（预算/进度用）。"""
    import math
    return sum(ELEM[dt] * int(math.prod(shp)) for _, dt, shp in manifest_rows)

"""exec_moe.py —— MegaMoE（DS4.1 的融合 MoE 算子）插件：**单卡 / 单 rank / EP=1** 形态。

接缝见 exec_ops.py 末尾的 `_load_plugins()`：本文件被 `register(OPS, SKIP, core)` 调一次，
`core` = exec_ops 模块（可借 `_qmm` / `_hold` 等工具）。

官方调用点（两个后端，都是真算子）：
  ① ascendc（`cann_ops_transformer.ops.mega_moe`）
       op_impls/megamoe_backend.py:66   get_symm_buffer_for_mega_moe(group, num_experts=…, …)
       op_impls/megamoe_backend.py:246  mega_moe(x=, topk_ids=, topk_weights=, sym_buffer=, **static_kwargs)
       static_kwargs（:209-229）= l1_weights[fp4] / l1_weights_sf[E8M0] / weight1_type=
       torch_npu.float4_e2m1fn_x2 / activation="swiglu" / shared_l1_weights[fp8] …
  ② deepgemm（`deep_gemm.fp8_fp4_mega_moe`）
       op_impls/deepgemm_backend.py:65   deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts=…)
       op_impls/deepgemm_backend.py:265  fp8_fp4_mega_moe(y, l1_weights, l2_weights, buffer,
                                                          shared_l1_weights=, shared_l2_weights=)

单卡结论（tools/probe_megamoe.py 实测，原文见该文件输出）：
  **ascendc 版在 Ascend950 上结构上不支持 ep_world_size=1**：
    · get_symm_buffer_for_mega_moe → get_mega_moe_ccl_buffer_size：
        RuntimeError: ep_world_size only support in [2, 1024], but got 1
        （csrc/mc2/mega_moe.cpp:587，纯 host 硬校验，与设备无关）
    · 绕过它继续走，卡在 HCCL channel：rankDim=1 → channelNum = rankDim-1 = 0
        RuntimeError: Acquire HCCL channel failed, ret: 2
        （csrc/mc2/comm_context.cpp:555；Ascend950 的 CommContextManager 走 "channel" 后端）
    · 官方自己也是 `if self.enable_mega_moe and self.moe_ep_size > 1:` 才注册 megamoe EP 组
      （models/modeling_dspark.py:750），即本就不打算在 EP=1 上用。
  而 **deepgemm 版单卡可用**：对称缓冲 rank=1 能建（T=1024 → 246 MB），算子真执行、出真输出
  （`num_experts=384 / num_topk=6 / hidden=5120 / intermediate=2304`，mma_type='fp8xfp4'）。

  → 本插件默认 `auto`：先按 ascendc 口径建缓冲，失败**大声回退** deepgemm（打一行
    [mega_moe] 警告，不做静默降级）。`DS41M_MOEMODE=ascendc|deepgemm` 可强制。
    EP≥2 的多卡场景（下一轮）把 MOEMODE 设成 ascendc 即可走官方主路径。

实测到的两个环境坑（与 mega_moe 无关，但会挡住 runner）：
  · 进程组端口：多会话共用一台机器时固定 29517 会 EADDRINUSE → `_free_port()` 动态挑端口
    （`DS41M_MOE_PORT` 可强制指定）。
  · 逻辑卡号 ≥1：`--device npu:2` 下 layer 0 的第一个算子 hc_pre（torch.ops.custom.npu_hc_pre_v2）
    就 aicore fault（507035/507015），最小复现见报告；`ASCEND_RT_VISIBLE_DEVICES=2` 把物理卡 2
    映射成逻辑 npu:0 后同一张卡一切正常 → 是本机 device 索引层面的问题，不是算子图的问题。

权重口径（真权重目录 = /mnt/xds/sfs/DeepSeek-V4.1-Flash-cann-mxfp8-mxfp4）：
  · 路由专家 mxfp4：`ffn.experts.{e}.w1.weight` I8 [I, H/2]（一字节两个 fp4）+ `.w1.scale`
    E8M0 [I, H/32]；w13 = **[w1 ; w3] 沿输出维拼接**（官方 module/fuse_moe_gemm.py:369
    `_load_w13`：w1 → 偏移 0、w3 → 偏移 I）。w2 [H, I/2] / scale [H, I/32]。
  · 共享专家 mxfp8：w1/w3 [I, H] fp8 + scale [I, H/32]（同样 w1 在前），w2 [H, I] / [H, I/32]。
  · deepgemm 要的 scale 是**两个 E8M0 打成一个 int16**（低字节在前）：
    官方 `_pack_ue8m0_exponents`（deepgemm_backend.py:25）= s[..., ::2] + s[..., 1::2]*256。
  · 打包布局（`_interleave_weights` / `_reshape_weights` / `_scale_fp4_weight_sf`）
    由 `deep_gemm.transform_weights_for_mega_moe()` 做，本文件不重造。
"""
from __future__ import annotations

import os
import time

import os
import torch
import torch_npu

import decode_layout
import ds41_memlog as _memlog_mod
# ★ 2026-10-23 显存优化：分块输出预分配，避免 `_outs` 列表 + `torch.cat` 双份。
# ★ 2026-10-26（T-058，B2 §3 第 2 块）：`os.environ.get` 被 torch_npu 打了补丁
#   （1.75 µs/次，普通 dict.get 0.065）。`DS41M_DEBUG` 在 `_mega_moe` 热路径里被读了
#   二十次/层，而它是**进程内静态**开关 ⇒ 提到模块级常量。语义不变（改它本来就要重启）。
_DEBUG_ON = os.environ.get("DS41M_DEBUG")     # 进程内静态开关（原 21 处 os.environ.get 都读它）

_ENV_MOE_PREALLOC = os.environ.get(
    "DS41M_MOE_PREALLOC", "").strip().lower() in ("1", "on", "true", "yes")

# ★ 2026-10-19：原来的 HIDDEN/INTER/N_EXPERTS/TOPK 是**死常量**（全文件无引用，
#   真实值一律走 ctx H/I/topk）→ 删除。下面两个由 model_spec 在 register 时刷新。
QUANT_BLOCK = 32         # mxfp8 量化块（model_spec.quant_block）
N_SHARED = 1             # 共享专家数（model_spec.n_shared_experts；当前只支持 1）


def _apply_spec(cfg=None):
    """按 model_spec 刷新模块常量 + `_CACHE["n_shared"]`（register 时一次）。"""
    import model_spec as _ms
    s = _ms.set_spec(cfg, strict=False) if cfg is not None else _ms.get()
    global QUANT_BLOCK, N_SHARED
    QUANT_BLOCK = int(s.quant_block)
    N_SHARED = int(s.n_shared_experts)
    _CACHE["n_shared"] = N_SHARED
    return s

_CORE = None             # register 时注入（exec_ops 模块）


def _oenv(k, d=None):
    """env 读取走 exec_ops 的启动期快照（★ A4-4）。kill switch 与 exec_ops 共用 `DS41M_ENV_HOIST=0`。
    只用于 `_mega_moe` 里的**启动期常量键**；`DS41M_MOEMODE` / `DS41M_MOE_CHUNK` 保持现场读
    （core/runner.py:607,949,2113,2820 会在运行期改）。"""
    c = _CORE
    return c._oenv(k, d) if c is not None else os.environ.get(k, d)
ADDED = "mega_moe"

# ★ 2026-10-18 host 优化：mega_moe 的 import 从热路径提到模块级缓存（40 层/轮）
_MM_OP = None


def _mm_op():
    global _MM_OP
    if _MM_OP is None:
        from cann_ops_transformer.ops import mega_moe as _m
        _MM_OP = _m
    return _MM_OP

# 设备 → 单次调用用的上下文（进程组 / 对称缓冲 / 已转换权重），跨层复用
_SUB: dict = {}      # DS41M_MOE_SUBTIME=1 时的子计时（barrier / 调用本体）

_CACHE: dict = {
    "pg": None, "group": None, "sb": None, "cap": 0, "dev": None, "enum": 0,
    "backend": None, "ascendc_err": None, "src": None, "w": None, "n_shared": 0,
}
# ★ 2026-10-22：对称缓冲**按 (后端, 全局专家数) 多槽缓存**。
#   单槽时 DSpark 草稿（E=128, T=6）会把主模型（E=384, cap=chunk）的缓冲挤掉，
#   下一轮 decode 再重建 10 GiB 级缓冲（每轮两次）→ 旧缓冲仍被在飞 kernel 引用，
#   实测第 2 轮 propose 在 H2D 处 507015（aicore exception）。
_SB: dict = {}          # key=(backend, E) -> (sb, cap)
_BACKEND_LAST = [None]  # mode=auto 时沿用上次成功的后端


# ─────────────────────────── 基础设施 ───────────────────────────
def _expert_ids(w) -> list:
    """本 rank 权重里出现的**全局**专家号（升序）。EP 连续切分 → 就是这个 rank 的 [lo,hi)。"""
    import re
    ids = set()
    for k in w:
        m = re.match(r"ffn\.experts\.(\d+)\.", k)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def _free_port() -> int:
    """挑一个空闲端口（多进程/多会话共用一台机器时，固定 29517 会 EADDRINUSE）。"""
    import socket
    for p in range(29517, 29597):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", p))
                return p
            except OSError:
                continue
    raise RuntimeError("mega_moe：29517~29596 全部被占用，找不到空闲端口")


_EFF_MODE = None        # _mega_moe 里临时覆盖（DS41M_MOEMODE=decode 时 prefill 仍走 ascendc）


def _init_pg(device) -> "object":
    """单卡 1-rank HCCL 进程组（进程内只建一次）。

    实测：`init_process_group("hccl", rank=0, world_size=1)` 可建；
    但 `get_hccl_comm_name(rank, init_comm=False)` 首次返回**空串**，要先 init_comm=True
    预热一次（deep_gemm 的 SymmBuffer 直接拿 init_comm 默认值 True，不受影响；
    ascendc 的 SymmBuffer 用的是 init_comm=False，所以这一步必须做）。
    """
    import torch.distributed as dist
    if _CACHE["pg"] is not None and _CACHE["dev"] == str(device):
        return _CACHE["pg"]
    if dist.is_initialized():
        g = dist.group.WORLD
    else:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if not os.environ.get("DS41M_MOE_PORT"):     # 未显式指定 → 动态挑一个空闲端口
            os.environ["MASTER_PORT"] = str(_free_port())
        else:
            os.environ["MASTER_PORT"] = os.environ["DS41M_MOE_PORT"]
        torch.npu.set_device(device)                 # 先设当前卡，否则 HCCL 报 indexFromRank≠device
        dist.init_process_group("hccl", rank=0, world_size=1)
        g = dist.group.WORLD
    # ★★ 预热**必须无条件做**（2026-10-15 踩实）：launch_mini 下 PG 已经存在 → 以前这里整段
    #   被跳过 → ascendc 的 SymmBuffer（内部用 init_comm=False 读 comm name）拿到**未初始化**的
    #    通信域 → 设备侧地址错乱 → 第一次调用就 aicore MTE 异常（所有核同一 PC）。
    #    厂商自测 test_mega_moe.py:576 就是无条件预热的；deepgemm 的 SymmBuffer 默认 init_comm=True
    #    所以单卡/旧路径看不出来。
    try:
        _rank = dist.get_rank() if dist.is_initialized() else 0
        g._get_backend(torch.device("npu")).get_hccl_comm_name(_rank, init_comm=True)
    except Exception as exc:                          # noqa: BLE001
        print(f"  [mega_moe] 警告：HCCL comm name 预热失败（{type(exc).__name__}: {exc}）")
    _CACHE["pg"] = g
    _CACHE["dev"] = str(device)
    return g


def _try_ascendc(group, T, dev, E, K, H, I):
    """按官方 megamoe_backend.py:66 的口径建 ascendc 对称缓冲；单卡必失败（见模块 docstring）。"""
    from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe
    return get_symm_buffer_for_mega_moe(
        group, num_experts=E, num_max_tokens_per_rank=T, num_topk=K,
        hidden=H, intermediate_hidden=I, max_recv_token_num=0,
        dispatch_quant_mode=4, dispatch_quant_out_dtype=torch.float8_e4m3fn,
        combine_quant_mode=0, comm_alg="", topk_weights_type=0)


def _symm_buffer(T, dev, E, K, H, I, n_shared):
    """deepgemm 对称缓冲。

    ★ E 必须是**全局**专家数（384），不是本 rank 的 e_local：算子自己按
      `num_experts_per_rank = num_experts / num_ranks` 推每 rank 的专家数并校验权重行数
      （csrc/apis/mega_moe.hpp:171-181，vendor 自测 tests/test_mega_moe.py 同口径）。
      EP=1 时两者相同，EP≥2 时传 e_local 会让权重行数校验炸（实测 Assertion mega_moe.hpp:123）。
    """
    # ★ 2026-10-21：`import deep_gemm` **只在真走 deepgemm 回退时才需要**。
    #   以前放在函数开头 → ascendc 口径在没有 deep_gemm 的机器上（新机 -00003）也直接
    #   ModuleNotFoundError；ascendc 融合算子根本不碰 deep_gemm（缓冲来自
    #   cann_ops_transformer.ops.get_symm_buffer_for_mega_moe）。挪到回退分支前。
    mode = _EFF_MODE or os.environ.get("DS41M_MOEMODE", "auto").lower()
    # ★ 缓冲"建一次、跨层复用"（指南 §2.1/§9）：以前这里写死 `backend == "deepgemm"`，
    #   ascendc 路径因此**每次调用都重建一个对称缓冲**（旧的还在用）→ 设备侧地址错乱，
    #   实测第一次调用后就 aicore MTE 异常（所有核同一 PC）。改成"按当前模式复用"。
    _want = _BACKEND_LAST[0] if mode == "auto" else mode
    _ent = _SB.get((_want, int(E)))
    if _ent is not None and _ent[1] >= T:
        return _ent[0]
    if _want in ("ascendc", "deepgemm") and (_want, int(E)) in _SB:
        _SB.pop((_want, int(E)))       # 容量不够 → 重建该槽（其他槽保留）
    group = _init_pg(dev)
    if mode in ("auto", "ascendc"):
        try:
            sb = _try_ascendc(group, T, dev, E, K, H, I)
            # ★ ascendc 的厂家缓冲**没有** deepgemm 那套字段（.x/.x_sf/.topk_idx/.topk_weights）
            #   —— 它的调用口径完全不同：输入/权重/输出都走 `mega_moe(...)` 的实参，
            #   缓冲只提供 context/ccl_buffer_size/quant 参数（见 _mega_moe 的 ascendc 分支）。
            #   过去这条路径没接，EP≥2 建成缓冲后必挂；现在默认走这里（DS41M_MOEMODE=deepgemm 可退回）。
            if not hasattr(sb, "ccl_buffer_size"):
                raise RuntimeError(f"ascendc 缓冲形态不认识（{type(sb).__name__}）")
            # ★ 缓存字段必须写全（尤其 `enum`/`dev`）：以前只写了 sb/backend/cap，`enum` 恒 None
            #   → 上面的复用判据永不成立 → **每次调用都重建一个对称缓冲**（实测第二次暴涨到
            #   400 MB，与第一次的 10.8 MB 不同；旧的还活着）→ 设备侧地址错乱 → aicore MTE。
            _CACHE.update(sb=sb, backend="ascendc", cap=T, dev=str(dev), enum=E)
            _SB[("ascendc", int(E))] = (sb, int(T))
            _BACKEND_LAST[0] = "ascendc"
            print(f"  [mega_moe] 后端 = ascendc 融合算子（cann_ops_transformer.ops.mega_moe）"
                  f"T={T} 专家={E}×{group.size()}rank | ccl_buffer_size={sb.ccl_buffer_size} "
                  f"topo={sb.topo_type} rank/server={sb.rank_num_per_server} "
                  f"max_recv={sb.max_recv_token_num} dq_mode={sb.dispatch_quant_mode} "
                  f"topk_w_type={sb.topk_weights_type} combine_mode={sb.combine_quant_mode}")
            return _CACHE["sb"]
        except Exception as exc:                      # noqa: BLE001
            _CACHE["ascendc_err"] = f"{type(exc).__name__}: {exc}"
            if mode == "ascendc":
                raise
            print(f"  [mega_moe] ascendc 不可用（{_CACHE['ascendc_err']}）"
                  f"→ 回退 deepgemm（DS41M_MOEMODE=ascendc 可强制）")
    import deep_gemm                      # 仅 deepgemm 回退路径需要（见函数头注释）
    sb = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts=E, num_max_tokens_per_rank=T, num_topk=K,
        hidden=H, intermediate_hidden=I, num_shared_experts=n_shared)
    _CACHE.update(sb=sb, backend="deepgemm", cap=T, dev=str(dev), enum=E)
    _SB[("deepgemm", int(E))] = (sb, int(T))
    _BACKEND_LAST[0] = "deepgemm"
    print(f"  [mega_moe] 后端 = deepgemm（deep_gemm.fp8_fp4_mega_moe）T={T} "
          f"对称缓冲 {sb.num_bytes / 2**20:.1f} MB（num_experts={E} × {group.size()} rank，"
          f"本 rank 持有 {E // group.size()} 个专家）")
    return sb


# ─────────────────────────── 权重打包 ───────────────────────────
def _u8(t: torch.Tensor) -> torch.Tensor:
    """E8M0 scale → uint8 视图（真权重可能是 F8_E8M0 / uint8 两种取法）。"""
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


# ★ T-042：pack 的 dtype 视图快路（默认开；PERF_RULES §0）。`DS41M_PACK_PAIRS_VIEW=0` 回退。
_PACK_VIEW_ON = os.environ.get("DS41M_PACK_PAIRS_VIEW", "1").strip().lower() in (
    "1", "on", "true", "yes")

# ═══════════════ A2 探针（2026-10-26，DeepInfer 对比；**默认全关**）═══════════════
#  ① `DS41M_MOE_ALL_LOCAL=1`：全部路由限定在本 rank 的专家窗口内（时间账 A/B 用）。
#     DeepInfer 没有这个开关；它的价值是**把 all2all 从 mega_moe 的时间里剥掉**：
#     每专家路由数、M-block 数、权重流量、GEMM 量都与 balroute 逐项相同，唯一差别是
#     跨 rank 的 dispatch/combine 字节归零 ⇒ Δ(时长) 就是 all2all 项的量级。
#     会改数值（路由不同），只能用于时延 A/B，**不能**与逐位基线混比。
_ALL_LOCAL_ON = os.environ.get("DS41M_MOE_ALL_LOCAL", "0").strip().lower() in (
    "1", "on", "true", "yes")
_ALL_LOCAL_LOGGED = [False]
#  ② `DS41M_MOE_CAST_INTO_SB=1`：`tile_kernels.quant.per_token_cast(..., out=(x, x_sf))`
#     直写对称缓冲（DeepInfer 的 cast 口径：norm.py:65-215 / mlp.py:274-301 的
#     `out=(sym_buffer.x[:n], sym_buffer.x_sf[:n])` + `use_packed_ue8m0=True`）。
#     相对现状省掉 `npu_dynamic_mx_quant` 的中间张量 + 一次 HBM 往返
#     （fp8 9.1 MB 读 + 9.1 MB 写 / 重层）。**sf 取整口径未对拍 ⇒ 默认关**。
_CAST_INTO_SB_ON = os.environ.get("DS41M_MOE_CAST_INTO_SB", "0").strip().lower() in (
    "1", "on", "true", "yes")
_CAST_INTO_SB_OK = [True]      # 首次失败后置 False → 之后每层直接走旧路（只报一次）

def _quant_into_sb_legacy(x, sb, n: int) -> None:
    """原路径：npu_dynamic_mx_quant → 两次 copy_（官方 deepgemm_backend.py:245-251）。"""
    xq, xs = torch_npu.npu_dynamic_mx_quant(_contig(x), dst_type=torch.float8_e4m3fn)
    xs_u8 = xs.view(torch.uint8)
    if xs_u8.dim() > 2:
        xs_u8 = xs_u8.flatten(1)
    sb.x[:n].copy_(xq)
    sb.x_sf[:n].copy_(_pack_pairs(xs_u8))


def _all_local_ids(ids, lo: int, e_local: int, K: int) -> torch.Tensor:
    """把全局 topk_idx 换成「只选集内专家」的强均分布（**时延 A/B 专用，改数值**）。

    与 balroute 同构：`id = lo + (t*K + k) % e_local`
      · 每条路由都落在本 rank 的专家窗口 [lo, lo+e_local) ⇒ 零跨 rank 字节；
      · 每专家路由数 = T*K/e_local（与 balroute 的 E=384 强均同值 ⇒ M-block 数不变）；
      · 同一 token 的 K 条路由互不相同（K=6 ≤ e_local=48）。
    """
    T = int(ids.shape[0])
    K = int(ids.shape[1]) if ids.dim() > 1 else int(K)
    dev = ids.device
    idx = ((torch.arange(T, device=dev).unsqueeze(1) * K
            + torch.arange(K, device=dev)) % int(e_local)) + int(lo)
    if not _ALL_LOCAL_LOGGED[0]:
        _ALL_LOCAL_LOGGED[0] = True
        print(f"[moe/all-local] ★ A/B 探针 ON：全部路由限定本 rank 专家 "
              f"[{lo},{lo + e_local})，T={T} K={K}（改数值，仅时延 A/B）", flush=True)
    return idx.to(torch.int32)


def _contig(t):
    """T-042：连续则原样返回（与 `exec_ops._contig` 同义，本文件独立可用）。"""
    return t if t.is_contiguous() else t.contiguous()


def _pack_pairs(sf_u8: torch.Tensor) -> torch.Tensor:
    """两个 E8M0 打成一个 int16（低字节在前）——官方 `_pack_ue8m0_exponents` 同式。

    ★ T-042：prefill 热路径（每层 megamoe 1 次）改成**零设备算子**的 dtype 视图 ——
      little-endian 下相邻两字节 uint8 就是 `lo + hi*256`（本机 aarch64/NPU 实测
      与旧式逐位相同，tmp/t042 报告）。连续且末维偶数才走视图；否则退回旧式
      （权重打包期的小表 / 非连续切片）。
    """
    if (_PACK_VIEW_ON and sf_u8.dim() > 0 and sf_u8.shape[-1] % 2 == 0
            and sf_u8.is_contiguous()):
        try:
            return sf_u8.view(torch.int16)
        except RuntimeError:                      # 非 2 字节对齐等 → 退回逐字节式
            pass
    x = sf_u8.to(torch.int32)
    return (x[..., ::2] + x[..., 1::2] * 256).to(torch.int16).contiguous()


# ★ T-042：`sb.topk_idx` 是 int64、`ids` 是 int32 —— 以前先 `.to(torch.int64)` 造一份
#   临时张量再 copy_（trace 实测 25 µs/层，含一次全量分配），现在直接在 copy_ 里带
#   dtype 转换（**一个 kernel、零临时分配**；int32→int64 精确无损，逐位不变）。
#   实测两 dtype 的 copy_ 在 NPU 上均支持（tmp/t042 报告）。


def _routed_weights(w, gids, I, H, dev):
    """本 rank 的路由专家 → deepgemm 的 (l1 = w13, l2 = w2) 原始对（未经 transform）。

    ★ `gids` = 本 rank 权重里的**全局**专家号（升序）；打包出的第 i 行 = 全局专家 gids[i]。
      算子侧口径：topk_idx 是全局号，分发时 dest_rank = id // e_local、本地行号 = id % e_local
      （e_local = 全局专家数 / rank 数，连续切分），所以本地行必须按全局号升序排，
      vender 自测 tests/test_mega_moe.py 就是这么喂的（make_topk 出全局号直接拷进 buffer）。

    l1: w13 [E, 2I, H/2] int8（**w1 在前、w3 在后**）+ scale [E, 2I, H/32] 打对 → int16
    l2: w2  [E, H, I/2] int8 + scale [E, H, I/32] 打对 → int16
    """
    E = len(gids)
    l1 = torch.empty((E, 2 * I, H // 2), dtype=torch.int8, device=dev)
    l1s = torch.empty((E, 2 * I, H // 32), dtype=torch.uint8, device=dev)
    l2 = torch.empty((E, H, I // 2), dtype=torch.int8, device=dev)
    l2s = torch.empty((E, H, I // 32), dtype=torch.uint8, device=dev)
    for i, g in enumerate(gids):                       # i = 本地行号，g = 全局专家号
        p = f"ffn.experts.{g}."
        l1[i, :I].copy_(w[p + "w1.weight"]); l1[i, I:].copy_(w[p + "w3.weight"])
        l1s[i, :I].copy_(_u8(w[p + "w1.scale"])); l1s[i, I:].copy_(_u8(w[p + "w3.scale"]))
        l2[i].copy_(w[p + "w2.weight"]); l2s[i].copy_(_u8(w[p + "w2.scale"]))
    return (l1, _pack_pairs(l1s)), (l2, _pack_pairs(l2s))


def _shared_weights(w, I, H, dev):
    """共享专家（mxfp8，n_shared=1）→ deepgemm 的 (l1, l2) 原始对。"""
    p = "ffn.shared_experts."
    w1 = w[p + "w1.weight"]; w3 = w[p + "w3.weight"]
    l1 = torch.cat((w1, w3), dim=0).contiguous()                       # [2I, H] fp8
    l1s = torch.cat((_u8(w[p + "w1.scale"]), _u8(w[p + "w3.scale"])), dim=0)   # [2I, H/32]
    l2 = w[p + "w2.weight"].contiguous()                               # [H, I] fp8
    l2s = _u8(w[p + "w2.scale"])                                       # [H, I/32]
    return (l1, _pack_pairs(l1s)), (l2, _pack_pairs(l2s))


def _src_keys(gids):
    """缓存身份键：本 rank **第一个**专家（不能写死 0 号——EP≥2 的 rank 没有 0 号专家，
    写死会让身份比对退化成 None==None 而永远命中旧权重）。"""
    g = gids[0]
    return (f"ffn.experts.{g}.w1.weight", f"ffn.experts.{g}.w2.weight",
            "ffn.shared_experts.w1.weight")


def _src_ref(t):
    """缓存身份：切片取基张量（见 _prepared 的注释），非视图取自身。"""
    return t._base if getattr(t, "_base", None) is not None else t


_PREP_L: dict = {}      # 按层缓存 deepgemm 的 weights transform（真权重 decode 跨步复用）
_PREP_AC_L: dict = {}   # 按层缓存 ascendc 的融合权重（40 层 ≈ 34.5 GiB；DS41M_AC_CACHE_N 限层数）
_PREP_GIDS: dict = {}   # 按层缓存本 rank 的全局专家号（raw 释放后 w 里扫不到 experts.*）
_AC_RELEASED: set = set()  # 已释放 raw 的层号（ascendc，避免每 pass 重复全表扫描）
_DG_RELEASED: set = set()  # 同上，deepgemm（`DS41M_DG_RELEASE_RAW`）

# ★ 2026-10-26（T-020 P1 / T2）：`_expert_ids(w)` 每层用 re.match 扫 ~293 个权重键
#   （CPU 实测 85.8 µs/次 ⇒ 3.4 ms/forward）。本 rank 的专家号在一个进程内恒定，
#   直接按层缓存；专家重切（EP↔TP / PD）时由 weights.invalidate_experts 回调清空。
#   ★ T-042：按 PERF_RULES §0 把默认从 0 翻成 **1**（T-020 eager A/B 实测 −5.4 ms@64k97）；
#   逐位由 T-042 的 digest A/B 复核（L0/1/2/20 + 末行，64k@0.97 与 256k@0.60）。
_GIDS_CACHE_ON = os.environ.get("DS41M_MOE_GIDS_CACHE", "1").strip().lower() in (
    "1", "on", "true", "yes")
try:                                  # 注册失效回调（weights 不 import exec_moe，无循环）
    import weights as _weights_mod
    _weights_mod.register_expert_cache_clearer(_PREP_GIDS.clear)
except Exception:                     # noqa: BLE001 —— 独立脚本里没有 weights 也能用
    pass


def _prepared_ac_cached(w, gids, I, H, dev, layer_idx=None):
    """ascendc 融合权重（`_prepared_ac`）+ 两级缓存。

    ★ 缓存顺序**不能反**：按层缓存必须先于 `_src` 身份缓存 —— raw 专家权重被释放后
      （DS41M_AC_RELEASE_RAW）`w.get(k)` 全为 None，身份键退化成 (None,None,None)，
      会跨层误命中、返回**上一层的 ac**（静默错权重）。
    """
    # ① 按层缓存（真权重：融合 ~0.5 s/40 层；DS41M_AC_CACHE_N=40 ⇒ 每层只融合一次）
    _ac_n = int(os.environ.get("DS41M_AC_CACHE_N", "0") or 0)
    _li = int(layer_idx) if layer_idx is not None else -1
    if 0 <= _li < _ac_n:
        _hit = _PREP_AC_L.get(_li)
        if _hit is not None:
            if _DEBUG_ON:
                print(f"      [mega_moe] prep_ac 命中缓存（层 {_li}）", flush=True)
            return _hit
    # ② 身份缓存（合成权重 share_across_layers：40 层同一 base tensor → 一次融合全命中）
    src = tuple(_src_ref(w.get(k)) for k in _src_keys(gids))
    cached = _CACHE["src"]
    if (_CACHE.get("ac") is not None and cached is not None
            and all(a is b for a, b in zip(cached, src))):
        if _DEBUG_ON:
            print("      [mega_moe] prep 命中缓存", flush=True)
        return _CACHE["ac"]
    if _DEBUG_ON:
        _t0 = time.time()
    ac = _prepared_ac(w, gids, I, H, dev)
    if _DEBUG_ON:
        print(f"      [mega_moe] prep_ac 重算（{time.time() - _t0:.2f}s）", flush=True)
    if 0 <= _li < _ac_n:
        _PREP_AC_L[_li] = ac
        _PREP_GIDS[_li] = tuple(int(g) for g in gids)   # raw 释放后靠它恢复 gids
    _CACHE["src"] = src           # 持源引用：下一层换 tensor 时这里自然失效（对比 is）
    _CACHE["ac"] = ac
    return ac


def _prepared_dg_cached(w, gids, I, H, dev, layer_idx=None):
    """deepgemm 的融合布局（`transform_weights_for_mega_moe`）+ 按层/身份缓存。

    ★ 这里**不再**顺手算 ascendc 的 `ac`：那是给 ascendc 算子用的另一套物理布局
      （fp4 format29 + e8m0 成对 scale），deepgemm 分支根本读不到 → 旧代码每层白拷贝
      ~0.85 GB + 一次 format_cast。
    ★ `DS41M_PREP_CACHE_N` 默认 0：transform 是 34 GB/卡的实体拷贝，叠在常驻 raw 权重上
      会 OOM（实测 81.84 GiB 已分配 + 5.12 GiB 申请失败）；要常驻缓存必须同时释放 raw
      （见 `_DeepGemmBackend.release_raw` 的说明）。
    """
    _cache_n = int(os.environ.get("DS41M_PREP_CACHE_N", "0") or 0)
    _do_cache = layer_idx is not None and int(layer_idx) < _cache_n
    if _do_cache:
        _hit = _PREP_L.get(int(layer_idx))
        if _hit is not None:
            return _hit
    # ★ 身份键取**基张量**：share_across_layers 下合成权重是"一个大 tensor 切片"，
    #   每次 iter_layer 返回新的视图对象 → 直接比 `is` 永远不命中（实测每次调用都重打包 +
    #   重做 format_cast，凭空多出一串 TransData，还把 host 时间顶起来）。
    #   非视图（真权重/非 share）取自己 → 仍是逐层重算，语义安全。
    src = tuple(_src_ref(w.get(k)) for k in _src_keys(gids))
    cached = _CACHE["src"]
    if (_CACHE.get("w") is not None and cached is not None
            and all(a is b for a, b in zip(cached, src))):
        if _DEBUG_ON:
            print("      [mega_moe] prep 命中缓存", flush=True)
        return _CACHE["w"]
    if _DEBUG_ON:
        _t0 = time.time()
    import deep_gemm
    routed = _routed_weights(w, gids, I, H, dev)
    shared = _shared_weights(w, I, H, dev)
    l1, l2 = deep_gemm.transform_weights_for_mega_moe(*routed)
    sh1, sh2 = deep_gemm.transform_weights_for_mega_moe(*shared)
    out = dict(l1=l1, l2=l2, sh1=sh1, sh2=sh2)
    if _DEBUG_ON:
        print(f"      [mega_moe] prep **重算**（{time.time() - _t0:.2f}s）", flush=True)
    _CACHE["src"] = src           # 持源引用：下一层换 tensor 时这里自然失效（对比 is）
    _CACHE["w"] = out
    if layer_idx is not None:
        # ★ 2026-10-21：与 ascendc 同一套账——raw 释放后 `w` 里扫不到 experts.*，
        #   `_mega_moe` 靠这里的 gids 恢复 EP 切分（否则二次 pass 报"权重里没有 experts"）。
        _PREP_GIDS[int(layer_idx)] = tuple(int(g) for g in gids)
    if _do_cache:
        _PREP_L[int(layer_idx)] = out
    return out


# 官方 ascendc 口径（megamoe_backend.py:186-227 的 static_kwargs / build_shared_args）：
#   路由 l1 (E,2I,H/2) float4_e2m1fn_x2 + sf (E,2I,H/32) e8m0；l2 (E,H,I/2) + sf (E,H,I/32)
#   共享 l1 (1,2I,H)  e4m3 + sf (1,2I,H/64,2)；           l2 (1,H,I)  + sf (1,H,I/64,2)
# 都是**视图**（int8/uint8 与 fp4/e8m0 同字节宽），不搬数据。
# ★ 踩过：`torch_npu.float4_e2m1fn_x2` 是 **int 296**（dtype 类型码，op 的 `weight1_type` 参数要它），
#   不是 torch.dtype —— 拿它去 `.view()` 会报 `shape '[296]' is invalid for input of size …`。
#   张量视图要用 torch 侧的 `torch.float4_e2m1fn_x2` / `torch.float8_e8m0fnu`（官方同一写法）。
def _fp4x2(t):
    """packed FP4 权重 → 算子要的形态：**uint8 字节 + format_cast 到定制 FP4 format(29)**。

    踩过两次：① 直接交 `float4_e2m1fn_x2` → "l1_weights[0] dtype must be uint8 (packed FP4)"；
    ② 只交裸 uint8 → aclnn 层报 "dtype or format … inconsistent with OpDef"（缺 format）。
    厂商自测 megamoe/test_mega_moe.py:368-371 的写法就是这个 format_cast。
    """
    u8 = t if t.dtype == torch.uint8 else t.view(torch.uint8)
    return torch_npu.npu_format_cast(u8, 29, customize_dtype=torch.float8_e4m3fn,
                                     input_dtype=torch_npu.float4_e2m1fn_x2)


def _e8m0(t):
    return t if t.dtype == torch.float8_e8m0fnu else t.view(torch.float8_e8m0fnu)


def _prepared_ac(w, gids, I, H, dev):
    """ascendc 融合算子要的权重列表（`mega_moe(..., l1_weights=[...], l1_weights_sf=[...])`）。"""
    E = len(gids)
    l1 = torch.empty((E, 2 * I, H // 2), dtype=torch.int8, device=dev)
    l1s = torch.empty((E, 2 * I, H // 32), dtype=torch.uint8, device=dev)
    l2 = torch.empty((E, H, I // 2), dtype=torch.int8, device=dev)
    l2s = torch.empty((E, H, I // 32), dtype=torch.uint8, device=dev)
    for i, g in enumerate(gids):
        p = f"ffn.experts.{g}."
        l1[i, :I].copy_(w[p + "w1.weight"]); l1[i, I:].copy_(w[p + "w3.weight"])
        l1s[i, :I].copy_(_u8(w[p + "w1.scale"])); l1s[i, I:].copy_(_u8(w[p + "w3.scale"]))
        l2[i].copy_(w[p + "w2.weight"]); l2s[i].copy_(_u8(w[p + "w2.scale"]))
    p = "ffn.shared_experts."
    sh1 = torch.cat((w[p + "w1.weight"], w[p + "w3.weight"]), dim=0).contiguous()     # (2I, H) e4m3
    sh1s = torch.cat((_u8(w[p + "w1.scale"]), _u8(w[p + "w3.scale"])), dim=0)         # (2I, H/32)
    sh2 = w[p + "w2.weight"].contiguous()                                             # (H, I)
    sh2s = _u8(w[p + "w2.scale"])                                                     # (H, I/32)
    # ★ scale 要**配成对**：(E,2I,H/64,2) / (E,H,I/64,2)（厂商自测 megamoe/test_mega_moe.py:350-353
    #   的 s13/s2 就是这个形状；给扁平的 (…,H/32) 会在 aclnn 层报 "dtype or format … inconsistent"）。
    #   字节序不变（只是把最后一维两两分组）→ reshape 即可，与共享专家同款。
    return dict(
        l1=_fp4x2(l1), l1s=_e8m0(l1s.reshape(E, 2 * I, H // 64, 2)),
        l2=_fp4x2(l2), l2s=_e8m0(l2s.reshape(E, H, I // 64, 2)),
        sh1=sh1.reshape(1, *sh1.shape), sh1s=_e8m0(sh1s.reshape(1, sh1s.shape[0], -1, 2)),
        sh2=sh2.reshape(1, *sh2.shape), sh2s=_e8m0(sh2s.reshape(1, sh2s.shape[0], -1, 2)),
    )


def _shared_expert(x, ctx, w, H):
    """共享专家（官方 DeepseekV41SharedExpert.forward_a8float8 逐字口径）：

        gate_up_proj → swiglu_group_quant(clamp_limit=swiglu_limit, round_scale=True, quant_mode=1)
        → down_proj(pergroup scale)
    ★ clamp 作用在 silu **之前**（A=Min(A,limit)、B=Clamp(B,±limit)），不是 silu 之后。
    """
    lim = float(ctx.get("swiglu_limit") or 10.0)
    x2 = x.reshape(-1, H)
    g1 = _CORE._qmm(x2, "ffn.shared_experts.w1.weight", w, ctx)
    g3 = _CORE._qmm(x2, "ffn.shared_experts.w3.weight", w, ctx)
    merged = torch.cat([g1, g3], dim=-1)                      # [T, 2I]，gate 在前
    inter, ps = _swiglu_quant(merged, lim)
    W2, WS2 = w["_packed"]["ffn.shared_experts.w2.weight"]
    return torch_npu.npu_quant_matmul(
        inter, W2, WS2, pertoken_scale=ps, output_dtype=torch.bfloat16,
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        group_sizes=[1, 1, QUANT_BLOCK])


# ─────────────── decode 口径 MoE（官方 moe_infer_dispatch_combine，2026-10-17）───────────────
# 官方 modeling_deepseek.py:806：**decode 不走 MegaMoE**，而是
#   npu_moe_distribute_dispatch_v2 → npu_grouped_matmul×2(+swiglu_quant)
#   → npu_moe_distribute_combine_v2（共享专家另走一条流再相加）。
# 这就是 decode 单步 MoE 从 ~5.8 ms/层 降到亚毫秒量级的关键。
def _swiglu_quant(x, lim, group_index=None, block_size=None, group_list_type=1):
    """decode 的 swiglu+fp8 量化（官方 low_latency_tp 路：`torch.ops.custom.npu_swiglu_group_quant`）。

    ★ 为什么不用 `torch.ops.cann_ops_nn.swiglu_group_quant`：本机 CANN 9.2 的该算子 Python
      schema 是 `int dst_type`（>=256 当 acl 枚举），但它的 aclnn 内核把 attr 收成 **0**
      → 设备侧 `attr dst_type only support (...), got 0`（2026-10-17 实测；
      传 torch.dtype / 292 / 24 都一样，是 wrapper↔kernel 版本不匹配）。
      custom 包的 schema 是规范的 `ScalarType dst_type`，官方低时延 decode 走的就是它。
    """
    block_size = int(block_size or QUANT_BLOCK)   # 缺省 = model_spec.quant_block（旧硬编码 32）
    kw = dict(dst_type=torch.float8_e4m3fn, quant_mode=1, block_size=block_size,
              round_scale=True, output_origin=False, group_list_type=group_list_type)
    if lim is not None:
        kw["clamp_limit"] = float(lim)
    if group_index is not None:
        kw["group_index"] = group_index
    y, ps, _ = torch.ops.custom.npu_swiglu_group_quant(x, **kw)
    return y, ps


def _reshape_mx_scale(s):
    """官方 quant_utils.reshape_mx_scale：末维两两成对（奇数补 0）。"""
    n = int(s.size(-1))
    if n % 2:
        s = torch.cat((s, torch.zeros((*s.shape[:-1], 1), dtype=s.dtype, device=s.device)), dim=-1)
        n += 1
    return s.reshape(*s.shape[:-1], n // 2, 2)


def _decode_weights(w, gids, I, H, dev, ctx=None):
    """兼容包装：decode-GMM NZ 布局的实现在 `decode_layout.py`（单机 P→D 共用）。"""
    return decode_layout.ensure_decode_gmm(w, gids, I, H, dev, ctx)


_DEC_N = {"n": 0}
_MC2 = {}
_DEC_W = decode_layout.DECODE_CACHE   # 兼容别名（同一张表；实现在 decode_layout）


def _mc2_group(dev):
    """官方 `moe_ep_group_mc2` 通信域（dispatch_v2/combine_v2 专用）。

    官方 CommManager.register_group("moe_ep_group_mc2", ...) 对 **同样的 world ranks**
    另建一个独立 HCCL 域（allow_physical_reuse=False），并带 MC2 专属配置：
      · Ascend 950：`group_type=3`（hccl_op_expansion_mode=aiv）、`is_full_mesh_v2=False`；
      · 其它平台：`comm_alg="fullmesh_v2"`，不设 group_type。
    buffer size = calc_moe_hccl_buffer_size(...)：decode 单 batch 时 ≈19 MB < 默认 200 → 取 200。
    ★ 用 world 组的 comm name 直接跑 dispatch 在本机（950DT）会"收到 0 行"（ep_recv/grp 全 0）。
    """
    if _MC2.get("name"):
        return _MC2["name"]
    import torch.distributed as dist
    world = dist.get_world_size()
    if os.environ.get("DS41M_MC2_USE_WORLD", "0") == "1":
        # 对照实验：直接用 world 组（不建 MC2 专属域）
        g0 = _init_pg(dev)
        rk0 = int(os.environ.get("RANK", dist.get_rank()))
        name0 = g0._get_backend(torch.device("npu")).get_hccl_comm_name(rk0)
        _MC2["group"], _MC2["name"] = g0, name0
        if _DEBUG_ON and rk0 == 0:
            print(f"      [mc2] 使用 world 组 name={name0}", flush=True)
        return name0
    opts = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    cfg = {"hccl_buffer_size": int(os.environ.get("DS41M_MC2_BUFSIZE", "200"))}
    gt = os.environ.get("DS41M_MC2_GTYPE", "3")
    if gt not in ("", "none"):
        cfg["hccl_op_expansion_mode"] = int(gt)
    opts.hccl_config = cfg
    g = dist.new_group(list(range(world)), pg_options=opts)
    rk = dist.get_rank()
    name = g._get_backend(torch.device("npu")).get_hccl_comm_name(rk)
    _MC2["group"], _MC2["name"] = g, name
    if _DEBUG_ON and rk == 0:
        print(f"      [mc2] 建域 OK world={world} name={name} cfg={cfg}", flush=True)
    return name


def _moe_decode(st, ctx, w, x, gids):
    """官方 decode MoE 全链（含共享专家）。x: [T,H] bf16；topk 已在 st 里。"""
    import torch.distributed as dist
    T, H, I, K = x.shape[0], ctx["H"], ctx["I"], ctx["topk"]
    dev = x.device
    # ★ 首个 decode 步：把 prefill 最后一层的 MegaMoE transform 缓存放掉（~GB 级，decode 用不到）
    if _DEC_N["n"] == 0 and _CACHE.get("w") is not None:
        _CACHE["src"], _CACHE["w"] = None, None
        if dev.type == "npu":
            torch.npu.empty_cache()
    group = _init_pg(dev)
    world = group.size()
    E = int(ctx["e_local"]) * world
    rank = int(os.environ.get("RANK", getattr(group, "rank", lambda: 0)()))
    gname = _mc2_group(dev)
    # ★ 官方口径：expert_ids / expert_scales 都是 [T, K] **二维**（踩过：摊平成 [T*K] →
    #   `RuntimeError: The x and expert_ids should be 2D`）
    topk_idx = st["topk_idx"].reshape(T, K).to(torch.int32)         # [T,K] 全局专家号
    topk_w = st["topk_weight"].reshape(T, K).float()                # [T,K]
    common = dict(x_active_mask=None, expert_shard_type=0, shared_expert_rank_num=0,
                  moe_expert_num=E, global_bs=0, group_ep=gname,
                  ep_world_size=world, ep_rank_id=int(rank),
                  group_tp=gname, tp_world_size=1, tp_rank_id=0)
    # ① dispatch（quant_mode=4：w4a8/w8a8 mx 路；y_dtype=fp8）
    #   ★ 官方 set_mc2_kwargs：只有**非 950** 平台才加 comm_alg="fullmesh_v2"；950 留空。
    _alg = os.environ.get("DS41M_COMM_ALG", "")
    out = torch_npu.npu_moe_distribute_dispatch_v2(
        x=x, expert_ids=topk_idx, scales=None, quant_mode=4,
        y_dtype=torch.float8_e4m3fn,
        **({"comm_alg": _alg} if _alg else {}), **common)
    expand_x, dyn_scale, expand_idx, expert_tokens = out[0], out[1], out[2], out[3]
    # ★ 官方 moe_infer_dispatch_combine：mxfloat 路要把 dispatch 回来的 per-token scale
    #   两两成对 reshape 后再喂 GMM（`dynamic_scale = reshape_mx_scale(dynamic_scale)`）
    if dyn_scale is not None and dyn_scale.dim() >= 2:
        dyn_scale = _reshape_mx_scale(dyn_scale)
    ep_recv, tp_recv = out[4], out[5]
    # ② 专家 GMM×2（w13 → swiglu_quant → w2）
    w13, w13s, w2, w2s = _decode_weights(w, gids, I, H, dev, ctx)
    if _DEBUG_ON and int(os.environ.get("RANK", "0")) == 0 and _DEC_N["n"] <= 1:
        print(f"      [gmm/dbg] expand_x{tuple(expand_x.shape)}/{expand_x.dtype} "
              f"exp_idx{tuple(expand_idx.shape)} w13{tuple(w13.transpose(1,2).shape)} "
              f"s13{tuple(w13s.transpose(1,2).shape)} dyn{tuple(dyn_scale.shape)} "
              f"grp{tuple(expert_tokens.shape)}/{expert_tokens.dtype} "
              f"grp[-1]={int(expert_tokens.flatten()[-1])} ep_recv.sum={int(ep_recv.sum())} "
              f"tp_recv.sum={int(tp_recv.sum())} xx_u8sum={int(expand_x.view(torch.uint8).sum())} "
              f"topk={topk_idx.tolist()}", flush=True)
    mm1 = torch_npu.npu_grouped_matmul(
        [expand_x], [w13.transpose(1, 2)], antiquant_scale=[w13s.transpose(1, 2)],
        per_token_scale=[dyn_scale], group_list=expert_tokens, split_item=3,
        output_dtype=torch.bfloat16, group_type=0,
        weight_dtype=torch_npu.float4_e2m1fn_x2,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        group_list_type=1, tuning_config=[0])[0]
    mid, ps = _swiglu_quant(mm1, ctx.get("swiglu_limit"), group_index=expert_tokens)
    y = torch_npu.npu_grouped_matmul(
        [mid], [w2.transpose(1, 2)], bias=None,
        antiquant_scale=[w2s.transpose(1, 2)], per_token_scale=[ps],
        group_list=expert_tokens, split_item=3, output_dtype=torch.bfloat16,
        group_type=0, weight_dtype=torch_npu.float4_e2m1fn_x2,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        group_list_type=1, tuning_config=[0])[0]
    # ③ combine（按 topk 权重求和）
    y = torch_npu.npu_moe_distribute_combine_v2(
        expand_x=y, expert_ids=topk_idx, assist_info_for_combine=expand_idx,
        expert_scales=topk_w, ep_send_counts=ep_recv, tp_send_counts=tp_recv, **common)
    _DEC_N["n"] += 1
    if _DEBUG_ON and int(os.environ.get("RANK", "0")) == 0 and _DEC_N["n"] % 10 == 1:
        print(f"      [moe/decode] 第 {_DEC_N['n']} 次调用 OK（层 {ctx.get('idx')}，T={T}）", flush=True)
    # ④ 共享专家（官方另走一条流；这里串行算，**逐步照抄** forward_a8float8 口径）
    sh = _shared_expert(x, ctx, w, H)
    out = (y.view(T, H) + sh.view(T, H)).to(torch.bfloat16)
    if _DEBUG_ON and int(os.environ.get("RANK", "0")) == 0 and _DEC_N["n"] <= 2:
        _o = out.float()
        print(f"      [moe/out] decode 层{ctx.get('idx')} y_absmax={_o.abs().max().item():.2f} "
              f"y_std={_o.std().item():.3f} sh_absmax={sh.float().abs().max().item():.2f} "
              f"行0[:4]={_o.view(-1)[:4].tolist()}", flush=True)
    return out


def _moe_tp(st, ctx, w, x, gids):
    """官方低时延 TP 口径（modeling_deepseek.py:600 `moe_infer_tp`）：

        npu_moe_init_routing_v2 → GMM×2(+swiglu) → npu_moe_finalize_routing
        → dist.all_reduce(moe_tp_group) → 加共享专家

    为什么快：没有 dispatch/combine 的 384 专家通信（EP 路径 3.5 ms/层），本 rank 只算
    自己 1/tp 的 intermediate 分片，一次 [T,H] all_reduce 汇合（T=1 时 10 KB）。
    ★ 共享专家在 all_reduce **之后**加：各 rank 复制计算，若加在前面会被 sum tp 次。
    """
    import torch.distributed as dist
    T, H, I, K = x.shape[0], ctx["H"], ctx["I"], ctx["topk"]
    E = len(gids)
    dev = x.device
    topk_idx = st["topk_idx"].reshape(T, K).to(torch.int32)
    topk_w = st["topk_weight"].reshape(T, K).float()
    # ★ 稀疏 grouplist（官方 enable_grouplist_type2 / 技术报告"稀疏 Grouplist"）：
    #   type=2 → 返回 [E,2] (expert_id, count)，**有效专家在前、零 token 专家在后**；
    #   GMM/swiglu 用 group_list_type=2 只遍历非空专家。T=1 时 384→6 组，
    #   否则 GMM 的 per-group 标量开销按 384 个专家算（实测 prefill ~20 s/层）。
    # DS41M_TP_GL=2（默认，稀疏 [E,2] 对）/ =1（密集计数，官方 YAML 默认口径）
    _gl = int(os.environ.get("DS41M_TP_GL", "1") or 1)   # 稀疏(type=2)在本机 decode 第 2 步 NaN，默认用密集
    expanded_x, row_idx, expert_tokens, pscale = torch_npu.npu_moe_init_routing_v2(
        x, expert_idx=topk_idx, active_num=T * K, expert_num=E,
        expert_tokens_num_type=(2 if _gl == 2 else 1), expert_tokens_num_flag=True,
        active_expert_range=[0, E], quant_mode=3)
    glist = expert_tokens.to(torch.int64) if _gl == 2 else expert_tokens
    if pscale is not None and pscale.dim() >= 2:
        pscale = _reshape_mx_scale(pscale)
    w13, w13s, w2, w2s = _decode_weights(w, gids, I, H, dev, ctx)
    if _DEBUG_ON and int(os.environ.get("RANK", "0")) == 0 and _DEC_N["n"] <= 1:
        print(f"      [tp/dbg] T={T} expanded{tuple(expanded_x.shape)}/{expanded_x.dtype} "
              f"row_idx{tuple(row_idx.shape)} etok{tuple(expert_tokens.shape)} "
              f"ps{None if pscale is None else tuple(pscale.shape)} "
              f"etok{tuple(expert_tokens.shape)} nz={(expert_tokens[:,1] > 0).sum().item() if expert_tokens.dim() == 2 else -1}", flush=True)
    mm1 = torch_npu.npu_grouped_matmul(
        [expanded_x], [w13.transpose(1, 2)], antiquant_scale=[w13s.transpose(1, 2)],
        per_token_scale=[pscale], group_list=glist, split_item=3,
        output_dtype=torch.bfloat16, group_type=0,
        weight_dtype=torch_npu.float4_e2m1fn_x2,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        group_list_type=_gl, tuning_config=[0])[0]
    mid, ps = _swiglu_quant(mm1, ctx.get("swiglu_limit"), group_index=glist,
                            group_list_type=_gl)
    y = torch_npu.npu_grouped_matmul(
        [mid], [w2.transpose(1, 2)], bias=None,
        antiquant_scale=[w2s.transpose(1, 2)], per_token_scale=[ps],
        group_list=glist, split_item=3, output_dtype=torch.bfloat16,
        group_type=0, weight_dtype=torch_npu.float4_e2m1fn_x2,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        group_list_type=_gl, tuning_config=[0])[0]
    out = torch_npu.npu_moe_finalize_routing(
        y, skip1=None, skip2=None, bias=None, scales=topk_w.to(y.dtype),
        expanded_src_to_dst_row=row_idx, export_for_source_row=None, drop_pad_mode=2)
    dist.all_reduce(out, group=_init_pg(dev))
    _DEC_N["n"] += 1
    if _DEBUG_ON and int(os.environ.get("RANK", "0")) == 0 and _DEC_N["n"] % 10 == 1:
        print(f"      [moe/tp] 第 {_DEC_N['n']} 次调用 OK（层 {ctx.get('idx')}，T={T}）", flush=True)
    sh = _shared_expert(x, ctx, w, H)
    return (out.view(T, H) + sh.view(T, H)).to(torch.bfloat16)


# ─────────────────── MoE 后端适配器（一个调度器 + 两个薄适配器）───────────────────
# 两个后端都是真算子，但**契约不同** —— 差异全部收进适配器，`_mega_moe` 只做编排
# （专家号 / 分块循环 / 对称缓冲复用 / 输出写回 / raw 释放时机都只有一份）：
#   · 权重布局  ascendc = fp4(format29) + e8m0 成对 scale（`_prepared_ac`）
#               deepgemm = blocked/NZ + int16 偏置 scale（`transform_weights_for_mega_moe`）
#   · 调用形态  ascendc 返回 (y, expert_tokens) 新张量；deepgemm 先把 x 量化进对称缓冲，
#               再把 y 当**出参**写满
#   · raw 释放  ascendc 的 ac 可当唯一副本（融合后释放 raw）；deepgemm 不释放（见下）
class _MoeBackend:
    """MoE 后端协议：inputs / prepare / call(单 span) / release_raw。"""
    name = ""
    supports_chunk = True

    def inputs(self, st, x, ctx, e_global):
        T, K = x.shape[0], int(ctx["topk"])
        return (x, st["topk_idx"].reshape(T, K).to(torch.int32),
                st["topk_weight"].reshape(T, K).float())

    def prepare(self, w, gids, I, H, dev, layer_idx=None):
        raise NotImplementedError

    def call(self, x, ids, tw, sb, prep, ctx):
        """算**一个 span**（x: [n,H]）；返回 [n,H] bf16。n 必须 ≤ 对称缓冲容量。"""
        raise NotImplementedError

    def release_raw(self, ctx, prep):
        """融合布局建好后释放 raw 权重（可选；返回释放项数）。"""
        return 0


class _AscendcBackend(_MoeBackend):
    name = "ascendc"

    def inputs(self, st, x, ctx, e_global):
        # 排查用：不要 gate 产出，直接造 ids/tw/x（旧口径，逐位不变）
        if os.environ.get("DS41M_MOE_FAKEIN"):
            T, K, dev = x.shape[0], int(ctx["topk"]), x.device
            ids = ((torch.arange(T, device=dev).unsqueeze(1) * K
                    + torch.arange(K, device=dev)) % e_global).int()
            tw = torch.softmax(torch.randn(T, K, device=dev), dim=1)
            x = torch.randn(T, x.shape[1], dtype=torch.bfloat16, device=dev) * 0.5
        else:
            x, ids, tw = super().inputs(st, x, ctx, e_global)
        if _DEBUG_ON:
            print(f"      [mega_moe/ac] ids 范围 [{int(ids.min())},{int(ids.max())}] "
                  f"tw 范围 [{float(tw.min()):.3g},{float(tw.max()):.3g}] "
                  f"x 范围 ±{float(x.abs().max()):.3g}", flush=True)
        return x, ids, tw

    def prepare(self, w, gids, I, H, dev, layer_idx=None):
        return {"ac": _prepared_ac_cached(w, gids, I, H, dev, layer_idx)}

    def call(self, x, ids, tw, sb, prep, ctx):
        ac = prep["ac"]
        n, H = x.shape[0], int(ctx["H"])
        _mm = _mm_op()
        try:
            _kw = dict(
                l1_weights_sf=[ac["l1s"]], l2_weights_sf=[ac["l2s"]],
                weight1_type=torch_npu.float4_e2m1fn_x2,
                weight2_type=torch_npu.float4_e2m1fn_x2,
                activation="swiglu", activation_clamp=ctx.get("swiglu_limit"),
            )
            _stry = os.environ.get("DS41M_MOE_SHARED_TRY", "")
            _NZ = getattr(torch_npu, "FRACTAL_NZ", 29)
            if _stry == "u8":        # 排查：共享 scale 用 uint8 传（不走 e8m0 视图）
                ac = dict(ac, sh1s=ac["sh1s"].view(torch.uint8), sh2s=ac["sh2s"].view(torch.uint8))
            if _stry == "fresh":     # 排查：把配对后的 4D scale 实体化（新分配 + copy_）
                _s1 = torch.empty_like(ac["sh1s"]); _s1.copy_(ac["sh1s"])
                _s2 = torch.empty_like(ac["sh2s"]); _s2.copy_(ac["sh2s"])
                ac = dict(ac, sh1s=_s1, sh2s=_s2)
            if _stry == "nz":        # 排查：共享权重转 NZ 再传（看 0.5s/call 是不是格式转换）
                _sh1 = torch_npu.npu_format_cast(ac["sh1"].contiguous(), _NZ)
                _sh2 = torch_npu.npu_format_cast(ac["sh2"].contiguous(), _NZ)
                ac = dict(ac, sh1=_sh1, sh2=_sh2)
            if not os.environ.get("DS41M_MOE_NOSHARED"):   # 排查用：不带共享专家（厂商自测口径）
                _kw.update(
                    shared_l1_weights=[ac["sh1"]], shared_l2_weights=[ac["sh2"]],
                    shared_weight1_type=torch.float8_e4m3fn,
                    shared_weight2_type=torch.float8_e4m3fn,
                    shared_expert_quant_out_dtype=torch.float8_e4m3fn,
                )
                if _stry != "noscales":   # 排查：只传权重、不传 scale
                    _kw.update(shared_l1_weights_sf=[ac["sh1s"]],
                               shared_l2_weights_sf=[ac["sh2s"]])
            # ★ 官方同款**分块**在调度器里做（`_mega_moe` 的 _spans）；这里只算一块。
            #   厂商融合算子的内部 workspace 按每 call 的 token 数走（T=16384 单次 ≈20 GiB，
            #   40 层常驻权重下会顶穿 91.8 GiB）；MoE 逐 token 独立 ⇒ 分块数值等价。
            y, _expert_tokens = _mm(x.contiguous(), ids, tw, [ac["l1"]], [ac["l2"]],
                                    sym_buffer=sb, **_kw)
        except Exception:
            if _DEBUG_ON:
                import traceback
                traceback.print_exc()
                print(f"      [mega_moe/ascendc] 入参：x{tuple(x.shape)} ids{tuple(ids.shape)}"
                      f" l1{tuple(ac['l1'].shape)}{ac['l1'].dtype} l1s{tuple(ac['l1s'].shape)}"
                      f" l2{tuple(ac['l2'].shape)} sh1{tuple(ac['sh1'].shape)}"
                      f" sh1s{tuple(ac['sh1s'].shape)}", flush=True)
            raise
        return y.view(n, H)

    def release_raw(self, ctx, prep):
        # ★ ascendc + 融合缓存：融合布局成为**唯一副本**，立即释放该层 raw per-expert 权重
        #   （官方模型本来就只有 `moe.experts.w13_weight` 一份；我们以前留着 raw + 每 pass 重新融合）。
        #   AC 与 raw 同尺寸（40 层 ≈34.5 vs 33.6 GiB）→ 显存净变化≈0；收益是**不再每 pass
        #   重融合**（TransData 570→9.8 ms），且两者不能同时常驻（68 GiB 会顶穿 91.8）。
        #   仅 prefill：decode 走 `decode_layout`（NZ 反转换），本进程不做 decode。
        #   raw 释放后 iter_layer 不再重取（RealSource.dropped），重跑会命中 `_PREP_AC_L`。
        _acn_rel = int(os.environ.get("DS41M_AC_CACHE_N", "0") or 0)
        _rel_on = os.environ.get("DS41M_AC_RELEASE_RAW",
                                 "1" if _acn_rel > 0 else "0").strip().lower()
        # ★ 常驻服务（--serve）：W1 建 AC 后 raw 还要留给 W2 建 decode NZ；由 serve_loop
        #   在 W2 结束后统一 drop（见 weights.drop_all_experts），这里跳过自动释放。
        _serve_warm = os.environ.get("DS41M_SERVER_WARMUP", "").strip().lower() in (
            "1", "on", "true", "yes")
        _li = int(ctx.get("idx", -1))
        if (_rel_on in ("1", "on", "true", "yes") and not _serve_warm
                and not ctx.get("is_decode_step")
                and 0 <= _li < _acn_rel and _li not in _AC_RELEASED):
            # ★ 每层只为**首次**释放扫描一次 `RealSource._cache`：drop_layer_experts 会对全表
            #   正则匹配（几千个键），以前每次 mega_moe 调用都扫（40 层 × 每 pass）→ 纯 host 浪费。
            _src = ctx.get("_src")
            if _src is not None and hasattr(_src, "drop_layer_experts"):
                try:
                    _AC_RELEASED.add(_li)
                    _ng = _src.drop_layer_experts(_li)
                    # ★ 2026-10-19：共享专家的 raw 同样只在 `_prepared_ac` 里用一次（fp8 转置/NZ 版
                    #   在 `_packed`，decode 的 `_shared_expert` 读它）→ 融合完成后一并释放
                    #   （3 个权重 + 3 个 scale ≈ 35 MB/层）。放这里是因为 `prep` 已在上方构建完成。
                    _ng += _src.drop_names(
                        [f"layers.{_li}.ffn.shared_experts.{n}" for n in
                         ("w1.weight", "w2.weight", "w3.weight",
                          "w1.scale", "w2.scale", "w3.scale")])
                    if _ng and _DEBUG_ON:
                        print(f"      [mega_moe] 层 {_li}：释放 raw 专家+共享权重 {_ng} 项"
                              f"（ascendc 融合布局接管）", flush=True)
                    return _ng
                except Exception as _e:                    # noqa: BLE001
                    if _DEBUG_ON:
                        print(f"      [mega_moe] raw 释放失败：{type(_e).__name__}: {_e}", flush=True)
        return 0


class _DeepGemmBackend(_MoeBackend):
    name = "deepgemm"

    def prepare(self, w, gids, I, H, dev, layer_idx=None):
        return _prepared_dg_cached(w, gids, I, H, dev, layer_idx)

    def call(self, x, ids, tw, sb, prep, ctx):
        import deep_gemm
        n, H = x.shape[0], int(ctx["H"])
        y = torch.empty((n, H), dtype=torch.bfloat16, device=x.device)
        # ① 量化输入 → 缓冲（官方 deepgemm_backend.py:245-251 逐字）
        # ★ T-042：`x` 是 st["x"].view(-1,H) 的连续视图 → contiguous 短路；
        #   ids 的 int64 转换并进下面的 copy_（见 `_pack_pairs` 后的注释）。
        if _CAST_INTO_SB_ON and _CAST_INTO_SB_OK[0]:
            # ★ A2 探针：DeepInfer 的 cast 口径（norm.py:65-215 / mlp.py:274-301）——
            #   预分配 out=(x_fp8, sf_packed) 直写对称缓冲，省中间张量 + 一次 HBM 往返。
            #   本树只搬「cast 直写」这一段（上游 rmsnorm 融合见 NOTES §4）。
            try:
                from tile_kernels.quant import per_token_cast
                per_token_cast(_contig(x), "e4m3", 32, round_sf=True,
                               use_packed_ue8m0=True, out=(sb.x[:n], sb.x_sf[:n]))
            except Exception as _e:                    # noqa: BLE001
                _CAST_INTO_SB_OK[0] = False
                print(f"  [moe/cast-into-sb] 不可用（{type(_e).__name__}: {_e}）"
                      f"→ 本进程回退 npu_dynamic_mx_quant", flush=True)
                _quant_into_sb_legacy(x, sb, n)
        else:
            _quant_into_sb_legacy(x, sb, n)
        sb.topk_idx[:n].copy_(ids)          # T-042：int32→int64 由 copy_ 内部转（无临时张量）
        sb.topk_weights[:n].copy_(tw)
        # ★ activation_clamp 必须传（官方 deepgemm_backend.py:212/263）：
        #   `"activation_clamp": moe.swiglu_limit`（config.swiglu_limit = 10.0）。
        #   官方参考实现见 modeling_deepseek.py:298-300：gate.clamp(max=limit)、
        #   up.clamp(±limit)（**非对称**：gate 只截上界）。之前传 None = 完全不截断，
        #   每层 MoE 都少这一步 → 层 0 起数值即偏（实测层 0 相对误差 0.49）。
        deep_gemm.fp8_fp4_mega_moe(
            y, prep["l1"], prep["l2"], sb,
            shared_l1_weights=prep["sh1"], shared_l2_weights=prep["sh2"],
            activation="swiglu", activation_clamp=ctx.get("swiglu_limit"))
        return y

    def release_raw(self, ctx, prep):
        # ★ 2026-10-21：deepgemm 的等价优化（默认关；run_real 的 prefill-only 口径打开）。
        #   transform 是 1:1 实体拷贝（40 层 ≈34 GB），与 raw 同量级 → 只有**同时**
        #   `DS41M_PREP_CACHE_N=40`（按层缓存 transform）+ `DS41M_DG_RELEASE_RAW=1`
        #   （释放本层 raw）才能：① 显存峰值不涨 ② 后续 pass 不再重做 transform。
        #   decode/TP/P→D 需要 raw，所以只在非 decode 步、且调用方显式开启时释放。
        _rel = os.environ.get("DS41M_DG_RELEASE_RAW", "").strip().lower() in (
            "1", "on", "true", "yes")
        if not _rel or ctx.get("is_decode_step"):
            return 0
        _li = int(ctx.get("idx", -1))
        _src = ctx.get("_src")
        if (_li < 0 or _li in _DG_RELEASED or _src is None
                or not hasattr(_src, "drop_layer_experts")):
            return 0
        _DG_RELEASED.add(_li)
        try:
            _n = _src.drop_layer_experts(_li)
            _n += _src.drop_names(
                [f"layers.{_li}.ffn.shared_experts.{x}" for x in
                 ("w1.weight", "w2.weight", "w3.weight", "w1.scale", "w2.scale", "w3.scale")])
            if _DEBUG_ON:
                print(f"      [mega_moe/dg] 层 {_li}：释放 raw {_n} 项（DG 融合布局接管）",
                      flush=True)
            return _n
        except Exception as _e:                    # noqa: BLE001
            if _DEBUG_ON:
                print(f"      [mega_moe/dg] 释放 raw 失败：{type(_e).__name__}: {_e}", flush=True)
            return 0


_BACKENDS = {"ascendc": _AscendcBackend(), "deepgemm": _DeepGemmBackend()}


def _moe_cap(T, L_full, chunk, mode):
    """对称缓冲容量（**跨调用必须稳定**；两个后端都允许 y.size(0) ≤ cap）。

    · 分块开 → 固定块容量（decode 每步 T 变化也绝不重建；deepgemm 源码
      csrc/apis/mega_moe.hpp:157 `y.size(0) <= num_max_tokens_per_rank` 明确支持）；
    · 分块关 + 显式 deepgemm → max(T, L_full)（旧口径：整段喂 T，decode 逐步 +1）；
    · 分块关 + ascendc/auto → T。
    """
    if chunk:
        return int(chunk)
    if mode == "deepgemm":
        return max(int(T), int(L_full))
    return int(T)


# ─────────────────────────── 算子 ───────────────────────────
def _mega_moe(st, ctx, w):
    """MegaMoE 融合算子：gate/topk 之后一条算子算完 **路由专家 + 共享专家**。

    官方（megamoe_backend.py:120 注释）：返回值已含共享专家贡献，模型侧不再另加。

    编排（两个后端的差异全在 `_BACKENDS` 的适配器里，这里只有一份）：
      gids → cap/对称缓冲 → backend.prepare → backend.release_raw
      → backend.inputs → span 循环 backend.call → 写回 st
    """
    x = st["x"]
    if x.dim() == 3:
        x = x.reshape(x.shape[0], -1)
    T = x.shape[0]
    H, I, K = ctx["H"], ctx["I"], ctx["topk"]
    dev = x.device

    # 专家口径（EP1 与 EP≥2 走**同一套**）：
    #   gids  = 本 rank 权重里出现的全局专家号（weights.ep_slice 连续切分 → [lo, lo+e_local)）
    #   e_local / e_global → 本地权重行数 / 缓冲要的全局专家数
    #   topk_idx 原样透传（gate 出的是全局号；分配/换号由算子自己按 id//e_local、id%e_local 做）
    _dw = _DEC_W.get(int(ctx.get("idx") or 0))
    _li_g = int(ctx.get("idx") or 0)
    _gcache = _PREP_GIDS.get(_li_g) if _GIDS_CACHE_ON else None
    if _dw:
        gids = list(_dw["gids"])
    elif _gcache:
        gids = list(_gcache)
    else:
        gids = _expert_ids(w)
    if not gids:
        # ★ ascendc + DS41M_AC_RELEASE_RAW：raw 专家权重已释放，w 里扫不到 experts.*
        #   → 用首次融合时缓存的 gids（同一进程内 EP 切片固定）。
        gids = list(_PREP_GIDS.get(int(ctx.get("idx") or 0), ()))
    if _GIDS_CACHE_ON and gids and not _PREP_GIDS.get(_li_g):
        _PREP_GIDS[_li_g] = tuple(int(g) for g in gids)
    if not gids:
        raise RuntimeError("mega_moe：本 rank 权重里没有 ffn.experts.*（EP 切分与权重源对不上？）")
    group = _init_pg(dev)
    world = group.size()
    e_local, lo = len(gids), gids[0]
    e_global = e_local * world
    if gids != list(range(lo, lo + e_local)) or lo % e_local != 0:
        raise RuntimeError(f"mega_moe：专家号不是连续切分（lo={lo}, e_local={e_local}, "
                           f"首尾={gids[0]}/{gids[-1]}），算子要求 id//e_local 能算出 rank")

    # ★★ 低时延 TP 口径（DS41M_MOE_TP=1）：prefill/decode 都走官方 moe_infer_tp
    #   （本 rank 全专家 × I/tp + 一次 all_reduce），与 EP 路径二选一。
    _tp_here = _oenv("DS41M_MOE_TP", "0") == "1" or (
        _oenv("DS41M_TP_DECODE_ONLY", "0") == "1" and ctx.get("is_decode_step"))
    if _tp_here:
        st["moe_out"] = st["x"] = _moe_tp(st, ctx, w, x, gids)
        return

    # ★★ decode 口径（官方 moe_infer_dispatch_combine）：dispatch→GMM×2→combine
    #   官方只在 **decode 步**用这条（prefill 仍是 MegaMoE）⇒ 这里按 ctx["is_decode_step"] 分流。
    global _EFF_MODE
    _mode_env = os.environ.get("DS41M_MOEMODE", "auto").lower()
    if _mode_env == "decode":
        # ★ split_tail：奇数 T 拆分出来的最后 1 token 属于 **prefill 阶段**，不是真 decode。
        #   仅对 **TP decode 布局** 改走 ascendc：tp8 的 P→D 需要 EP raw，若这里走 EP
        #   decode 口径，`ensure_decode_gmm` 会建 EP NZ 并把该层 raw 标记 dropped →
        #   紧随其后的 P→D 报"需要 EP raw 专家权重"。ep8 保持原 decode 口径不变
        #   （已验证过的逐位基线就来自那条路径，不要动）。
        _split_keep_prefill = bool(ctx.get("split_tail")) and (
            _oenv("DS41M_DECODE_LAYOUT", "").strip().lower() == "tp")
        if ctx.get("is_decode_step") and not _split_keep_prefill:
            try:
                y = _moe_decode(st, ctx, w, x, gids)
            except Exception as _exc:                  # noqa: BLE001
                if int(_oenv("RANK", "0")) == 0:
                    import traceback
                    print(f"      [moe/decode] 失败（层 {ctx.get('idx')} T={T}）：{type(_exc).__name__}: {_exc}", flush=True)
                    traceback.print_exc()
                raise
            st["moe_out"] = y
            st["x"] = y
            if _oenv("DS41M_DEBUG"):
                print(f"      [mega_moe/decode] T={T} → y_absmax={y.float().abs().max().item():.3f}", flush=True)
            return
        _EFF_MODE = "auto"          # prefill：按默认后端（ascendc 可用就用它）
    else:
        _EFF_MODE = None

    # ★ MoE 分块长度（官方 cann-recipes `get_moe_num_chunks` / `moe_chunk_max_len` 同口径）
    #   必须在建对称缓冲**之前**算出来：窗口按 `num_max_tokens_per_rank` 定尺寸，
    #   拿整 T 建 = 每次 call 都留 ~20 GiB workspace（实测 T=16384 单次申请 20.12 GiB，
    #   40 层常驻权重下必然把 91.8 GiB 顶穿）。分块 + 按块建缓冲才真正省下这 4 倍。
    #   ★ 2026-10-18 用户口径：一律默认打开。`runner.run_forward` 在入口设
    #   `DS41M_MOE_CHUNK=auto`（→ 用 config 的 moe_chunk_max_len，通常 8192）；
    #   显式 `DS41M_MOE_CHUNK=0` 可关（A/B 对照）。注意：分块后大 L 的 trace 里
    #   每层会有 ceil(T/chunk) 次 MoE，与旧"每层 1 次"的合成 trace 形状不同。
    _ch = os.environ.get("DS41M_MOE_CHUNK")
    _chunk = (int(ctx.get("chunk") or 0) if _ch == "auto" else int(_ch)) if _ch else 0
    # ★★ 对称缓冲的**容量必须跨调用稳定**（某 rank 先到下一步、别的 rank 还在用旧缓冲
    #   → HCCL/设备侧 aicore 错乱；decode 逐 token 时 T 每步 +1 必踩 ✗）。规则见 `_moe_cap`：
    #   分块开 → 固定块容量；分块关 → T（ascendc/auto）或 L_full（显式 deepgemm）。
    #   ★ 2026-10-21：deepgemm 也走分块（算子 `y.size(0) <= num_max_tokens_per_rank` 支持；
    #   旧"按 chunk 建而整段喂 T → aclnnInplaceCopy failed"是调用方没切 call 的老 bug）。
    _moemode = os.environ.get("DS41M_MOEMODE", "auto").lower()
    _Lfull = int(ctx.get("L_full") or 0) or int(T)
    _cap = _moe_cap(T, _Lfull, _chunk, _moemode)

    _ml_idx = ctx.get("idx")
    _memlog_mod.mem(f"mega_moe L{_ml_idx} T={T} cap={_cap} pre-symm")
    sb = _symm_buffer(_cap, dev, e_global, K, H, I, _CACHE["n_shared"] or 1)
    _memlog_mod.mem(f"mega_moe L{_ml_idx} T={T} cap={_cap} post-symm")
    if _cap > _CACHE["cap"]:
        raise RuntimeError(f"mega_moe 收到 {_cap} token（块容量），超过对称缓冲容量 {_CACHE['cap']}")
    be = _BACKENDS[_CACHE["backend"]]
    if _chunk and not be.supports_chunk:
        raise RuntimeError(f"mega_moe：后端 {be.name} 不支持分块，但 DS41M_MOE_CHUNK={_chunk}")
    prep = be.prepare(w, gids, I, H, dev, ctx.get("idx"))

    # ★ raw 释放：后端自己决定（ascendc = 融合布局成为唯一副本后释放；deepgemm = 空操作）。
    #   必须在 `prepare` **之后**：融合布局要先从 raw 建出来。
    be.release_raw(ctx, prep)
    x, ids, tw = be.inputs(st, x, ctx, e_global)
    # ★ A2 探针：把 all2all 从时间账里剥掉（详见文件头 `_all_local_ids` 的注释）
    if _ALL_LOCAL_ON:
        ids = _all_local_ids(ids, lo, e_local, K)

    # ★ 跨 rank 同步：**默认关**（2026-10-15 用户口径：官方图里没有 HcclLaunchAicpuKernel /
    #   EventWait 这两类 op，要"图逐 op 与官方一致"就不能插；prof 轮同样不插）。
    #   起跑线对齐改在 **runner.start_line_sync()**：整个 run 只在开 profiler **之前**
    #   barrier 一次，之后各位各跑。
    #   背景：mini 是 Python 逐算子发、各 rank 会漂；不 sync 时 MegaMoe 的 kernel 时长会夹带
    #   "等最慢 rank"的等待（16k 实测：中位 577 µs → 1.4~3.4 ms，首轮还有 200 ms 尖峰）——
    #   那不是算子慢。要"逐算子归因诚实"时开 DS41M_MOE_BARRIER=1（把等待挪到 kernel 外面）。
    _t_b = time.perf_counter()
    if _oenv("DS41M_MOE_BARRIER"):     # 仅调试归因时开：默认关，prof 轮也不开
        import torch.distributed as _d
        if _d.is_initialized():
            _d.barrier()
    if _oenv("DS41M_MOE_SUBTIME"):
        _SUB["barrier"] = _SUB.get("barrier", 0.0) + (time.perf_counter() - _t_b)

    # ★ 唯一的执行循环：按**实际建成的缓冲容量**切 span（末块可小于容量；两个后端都接受
    #   `y.size(0) ≤ cap`）。分块循环只有这一份，不再各写一套。
    _cap_use = int(_CACHE["cap"])
    _spans = [(i, min(i + _cap_use, T)) for i in range(0, T, _cap_use)]
    # ★ 2026-10-21 per-DP dummy span：MegaMoE 是**同步**的 —— 各 rank 的 chunk 轮数必须相同。
    #   per-DP 时由 points_runner 把「全 DP 的最大 span 数」写进 ctx["moe_nspan"]；
    #   T 小的 rank 在多余轮次喂 **1 行占位**（y.size(0) ≤ cap 厂商支持），通信照常参与，
    #   输出直接丢弃。这样不同 DP 的 batch 组合才能在同一 pass 里共存。
    _nspan_target = int((ctx or {}).get("moe_nspan") or 0)
    _ndummy = max(0, _nspan_target - len(_spans)) if _nspan_target > 0 else 0
    _t_c = time.perf_counter() if _oenv("DS41M_MOE_SUBTIME") else 0.0
    _memlog_mod.mem(f"mega_moe L{_ml_idx} T={T} nspan={len(_spans)} pre-spans")
    if _ENV_MOE_PREALLOC:
        # 预分配整段输出，逐 span 写入；不再持有全部 span 输出，也不做 cat。
        y = torch.empty((T, H), dtype=x.dtype, device=x.device)
        for _i, _e in _spans:
            _yi = be.call(x[_i:_e], ids[_i:_e], tw[_i:_e], sb, prep, ctx)
            y[_i:_e].copy_(_yi)
            del _yi
    else:
        _outs = [be.call(x[_i:_e], ids[_i:_e], tw[_i:_e], sb, prep, ctx)
                 for _i, _e in _spans]
    _memlog_mod.mem(f"mega_moe L{_ml_idx} T={T} nspan={len(_spans)} post-spans")
    if _ndummy:
        _d1 = min(1, int(T))                       # 占位行数：1（0 行厂商未验证）
        for _ in range(_ndummy):
            _ = be.call(x[:_d1], ids[:_d1], tw[:_d1], sb, prep, ctx)   # 结果丢弃
        if _oenv("DS41M_DEBUG"):
            print(f"      [mega_moe] dummy span ×{_ndummy}（本 rank {len(_spans)} 轮，"
                  f"全 DP 最大 {_nspan_target} 轮；每轮喂 {_d1} 行占位）", flush=True)
    if _oenv("DS41M_MOE_SUBTIME"):
        _SUB["call"] = _SUB.get("call", 0.0) + (time.perf_counter() - _t_c)
        _SUB["n"] = _SUB.get("n", 0) + 1
    if not _ENV_MOE_PREALLOC:
        y = _outs[0] if len(_outs) == 1 else torch.cat(_outs, 0)
    _memlog_mod.mem(f"mega_moe L{_ml_idx} T={T} post-cat")
    y = y.view(T, H)
    st["moe_out"] = y
    st["x"] = y                     # 残差流：hc_post 吃它（官方模型侧同样直接替换 hidden）
    if _oenv("DS41M_DEBUG"):
        if len(_spans) > 1:
            print(f"      [mega_moe] 分块 {_cap_use}/块 ×{len(_spans)}"
                  f"（官方 get_moe_num_chunks 同口径）", flush=True)
        print(f"      [mega_moe] T={T} E={e_global}(本 rank {e_local}) K={K} 后端={be.name} "
              f"x_absmax={x.float().abs().max().item():.3f} "
              f"y_absmax={y.float().abs().max().item():.3f}", flush=True)


def register(OPS, SKIP, core, cfg=None):
    global _CORE
    _CORE = core
    _apply_spec(cfg)                     # ★ 常量/共享专家数对齐 config（默认值不变）
    OPS["mega_moe"] = _mega_moe
    SKIP.pop("mega_moe", None)


# 注入 decode_layout 需要的张量工具（避免循环导入）
decode_layout.configure(_u8, _reshape_mx_scale)

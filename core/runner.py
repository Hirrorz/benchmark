"""runner.py —— M2 单卡真跑：按 recipes 的顺序把算子发到 NPU，可选出 msprof trace。

与 dry-run 共用同一份 recipes/ops → **图与执行是一条路**（改了配方两边立刻一致）。

用法：
  python3 runner.py configs/mini_single.yaml --seq 1024 --layers 2      # 冒烟（不落 trace）
  python3 runner.py configs/mini_single.yaml --seq 1024 --prof --out <共享盘目录>
"""
from __future__ import annotations
import os as _os, sys as _sys
_ds41_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    _q = _os.path.join(_ds41_root, _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)

import metadata as _md


import argparse
import collections
import json
import os
import socket
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch                                   # noqa: E402
import torch_npu                               # noqa: E402

import assemble                                # noqa: E402
import ds41_log                                # noqa: E402
import exec_ops                                # noqa: E402
import ds41_memlog as _memlog_mod              # noqa: E402
import model_spec                              # noqa: E402
import opgraph as ops_mod                          # noqa: E402
import decode_layout                              # noqa: E402
import profiling                                  # noqa: E402
import replay                                     # noqa: E402
import weights as weight_src                   # noqa: E402
from progress import Progress, STAGES_MINI             # noqa: E402
from run import ctx_for, expand_layer, load_recipes   # noqa: E402
from modules import runtime_probe as _probe       # noqa: E402 —— Phase 1.5 上下文探针
from modules import runtime as _mod_runtime        # noqa: E402 —— Phase 2 模块树执行计划

# ★ 2026-10-23（Phase 1.5）：默认关；开启只记录读写，不改数值。
_PROBE_ON = _probe.maybe_enable()
# 作用域注入 exec_ops：只包算子实现体，不含 packed 分派/CP 判断（否则 batch.* 假依赖）。
exec_ops.PROBE_SET = _probe.set_current if _PROBE_ON else None

# ★ 2026-10-18 host 优化：热循环里的 env 查询提到模块级（进程启动后不变）
_ENV_DEBUG = bool(os.environ.get("DS41M_DEBUG"))
_ENV_HOSTTIME = bool(os.environ.get("DS41M_HOSTTIME"))
_ENV_SYNC_BEFORE = bool(os.environ.get("DS41M_SYNC_BEFORE"))
_ENV_SILENT = bool(os.environ.get("DS41M_SILENT"))   # 无日志模式：不打印逐层进度（launch_mini 传）
_ENV_SYNC = bool(os.environ.get("DS41M_SYNC"))
_ENV_DUMP_OPS = os.environ.get("DS41M_DUMP_OPS")
# 排查：decode 步逐算子检查 x 是否出现 NaN（层 < DS41M_NAN_TRACE_LAYERS，默认 4）。
# 每算子一次设备同步 → 很慢，只在定位"第几步/哪个算子先 NaN"时开。
_ENV_NAN_TRACE = bool(os.environ.get("DS41M_NAN_TRACE"))
# ★ 2026-10-21 极致性能：逐算子/逐层的**诊断开关**也全提到模块级。这些在热循环里
#   每算子查一次 os.environ（20~63 算子 × 40 层 ≈ 每 pass 数千次 dict 查询），
#   值全在进程启动时定死 → 提升后热循环里只剩布尔/整数比较，且默认路径全是 False。
_ENV_STATS = bool(os.environ.get("DS41M_STATS"))
_ENV_MOE_ONLY = bool(os.environ.get("DS41M_MOE_ONLY"))
_ENV_DUMP_LAYERS_CAP = int(os.environ.get("DS41M_DUMP_LAYERS", "2") or 2)
_ENV_DUMP_FROM = int(os.environ.get("DS41M_DUMP_FROM", "0") or 0)
_ENV_DUMP_SLICE = os.environ.get("DS41M_DUMP_SLICE")
_ENV_DUMP_SLICE_MAX = int(os.environ.get("DS41M_DUMP_SLICE_MAX", "2") or 2)
_ENV_DUMP_SLICE_N = int(os.environ.get("DS41M_DUMP_SLICE_N", "512") or 512)
_ENV_DUMP_SLICE_OFF = int(os.environ.get("DS41M_DUMP_SLICE_OFF", "0") or 0)
# ★ 2026-10-21：回放裁尾后不同层的 T 不同（L21 只有 W=128 行），固定 OFF=128 会把
#   被裁层切空（假"无差异"）。TAIL=1 时 token 级张量按"本层最后 N 行"取（各层对齐到
#   同一批逻辑位置），层内 T=N 时 OFF 自然为 0。
_ENV_DUMP_SLICE_TAIL = os.environ.get("DS41M_DUMP_SLICE_TAIL", "0") == "1"


def _ranks_spec(name):
    """`DS41M_DUMP_RANKS`（all / 逗号列表）→ (all?, set)。只在模块导入时解析一次。
    非法项跳过（解析在 import 期，不能因为一个拼错的 rank 号让整进程起不来）。"""
    s = (os.environ.get(name, "all") or "all").strip().lower()
    if s in ("all", ""):
        return True, frozenset()
    out = set()
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            out.add(int(t))
        except ValueError:
            pass
    return False, frozenset(out)


_ENV_DUMP_OPS_ALL, _ENV_DUMP_OPS_RANKS = _ranks_spec("DS41M_DUMP_RANKS")
_ENV_DUMP_SLICE_ALL, _ENV_DUMP_SLICE_RANKS = _ranks_spec("DS41M_DUMP_RANKS")
_ENV_SYNC_LAYERS = os.environ.get("DS41M_SYNC_LAYERS", "").strip().lower() in (
    "1", "on", "true", "yes")
# ★ host 抢跑上限（2026-10-25，CP）：去掉逐层阻塞 H2D 后 host 可以整层整层抢跑，
#   而 MegaMoE 的对称缓冲 barrier 是**跨 rank 同步**的（exec_moe.py:1004 注释：
#   "某 rank 先到下一步、别的 rank 还在用旧缓冲 → 设备侧错乱"）。实测 CP2 64K
#   无限抢跑：稳态 2.60→3.11s（每层 MoE 等最慢 rank），且 8 个 forward 内可能
#   触发 "MegaMoE rank barrier timeout epoch=960 signal=6 target=8" → HCCL watchdog。
#   事件环把"设备落后 host"限制在 K 层以内：K 层前的事件没完成就等它。
#   K=0 关闭（=旧行为，不限；A/B 用）；只对 CP>1 生效，其余路径零变化。
_RUNAHEAD_K = int(os.environ.get("DS41M_RUNAHEAD_LAYERS", "2") or 0)
# ★ 2026-10-26：非 CP 深度单独一个旋钮，**默认也是 2（实测安全上界）**。
#   128K@70（T=39462，崩溃点位）扫深度：K=2 稳定；**K=4 / K=8 / K=16 都复现
#   MegaMoE barrier timeout**（K=16 在 epoch=485 signal=7，K=4 同样 7 次）。
#   深度代价（64K@95）：关 158.0 / K=2 194.6（+23%）/ K=8 173.8 / K=16 159.7
#   ——K 越大越便宜，但只有 2 是安全的；多出来的 ~36ms 是**正确性代价**，
#   不能靠加深 K 省掉（要做只能从 host 每层耗时下手，T-020 P1）。
_RUNAHEAD_K_NONCP = int(os.environ.get("DS41M_RUNAHEAD_LAYERS_NONCP", "2") or 0)
# 非 CP 的**所有**前向都套同一道有界流水（DeepInfer 式：请求内不无限抢跑）。
# ★ 2026-10-26 实测：阈值 40000 留了个洞——128K@70% 的 T_pass=39462 刚好在阈值下，
#   noprof 连排 6 个 forward（240 层 barrier）→ "MegaMoE rank barrier timeout
#   epoch=1365 signal=3 target=8"（r2/r4/r7），随后 allocator 里 segfault。
#   阈值本质是"host 抢跑上限"，与 T 大小无关 → 默认 0 = 一律生效。
#   DS41M_RUNAHEAD_MIN_T 仍可显式设阈值（A/B 用）；0=对所有前向生效。
#
# ★★ 2026-10-26（T-046，邻机 099 交错 A/B，每臂 0-error、两次重复差 ≤0.23ms）：
#   **小 T 上关环比有环快**：64k@0.97（T_pass=2106）
#     K=0 **122.9** / K=1 137.4 / K=2 133.5 / K=3 130.0
#   ⇒ 环在这个形状上净亏 10.6 ms；K=1 最差（浅到 1 层时 host 完全无法覆盖设备抖动）。
#   规则声明进 `configs/policy.yaml: runahead.min_t_pass`（默认 8192）：
#   T_pass <  min_t_pass ⇒ 不建环（= K=0 行为，事件记录 1158→998）；T_pass >= min_t_pass
#   ⇒ 仍按 K 建环（大 T 的 MegaMoE barrier 保险不撤）。env 覆盖优先。
#   依据/风险：NOTES/2026-10-26-runahead-k0-and-host-serial.md
try:
    import policy as _policy_mod                       # meta/ 在 sys.path（bootstrap 已加）
    _RU_MIN_T_DEFAULT = int(_policy_mod.rule("runahead.min_t_pass", 8192) or 0)
except Exception:                                      # noqa: BLE001
    _RU_MIN_T_DEFAULT = 8192
_RUNAHEAD_MIN_T = int(os.environ.get("DS41M_RUNAHEAD_MIN_T", _RU_MIN_T_DEFAULT) or 0)
_RUNAHEAD_RING: list = []
_RUNAHEAD_POS = [0]          # 环形槽位（事件对象复用；见层循环里的注释）
_RUNAHEAD_K_ACTIVE = [0]     # 当前环容量；CP/非 CP 深度不同，切换时排空重建
# ★★ 2026-10-27（B6 / DeepInfer 口径，默认开）：**层内不做任何 host 同步，每 forward 末尾排空一次**。
#   DeepInfer 在**同一份** mega_moe kernel 上层内一次 host 同步都没有（逐层路径 0 个
#   synchronize/.item()/.cpu()；整个 forward 被 forbid_cuda_sync() 包住，强制抛错），
#   缓冲不被写坏靠设备侧三层协议（入口 rank_barrier / 收尾 combine_ready_prefix==~0 /
#   环内 per-block credit）——三段在我们的树里与 DeepInfer **diff = 0 行**（B6 笔记 §2/§3）。
#   所以 `_RUNAHEAD_RING` 的每层 `Event.synchronize()` 是纯冗余（64k@0.60 实测
#   aclrtSynchronizeEvent 160 次 / 4 forward = 810 ms/forward 的 host 阻塞）。
#   =2：**一个 forward 在飞**（DeepInfer 真口径）：层内零同步；forward 末尾 record 事件，
#       只等**上一个** forward 的事件（2 槽轮换）⇒ host/设备仍重叠，抢跑上界 = 1 个 forward
#       （≤40 层 MoE，落在 DeepInfer 的 ≤2 forward 验证域内）。128K@70 实测 =1 硬同步慢 22%
#       （1719.9 vs 1405.7），就是因为丢掉了这段重叠。
#   =1：每个 forward 末尾无条件 `torch.npu.synchronize()`（最严；A/B 用）。
#   =0：回到旧口径（有界抢跑 K + pass 间可不排空），A/B 用。
#   ⚠️ 风险 R1（B6 §3.4）：我们 K≥4 在 128K@70 崩过，而 DeepInfer 层内 0 同步不崩 ⇒
#      "2 层安全窗口"不是 kernel 性质。E1（128K@70 + 本开关，6 forward）通过后才算定稿。
#      E1 已过：=1 与 =2 在 128K@70（T=39462）各 6 fwd 均 0 错。
_DF_RING: list = []           # 2 个 Event（=2 模式；进程内复用）
_DF_POS = [0]                 # 当前要 record 的槽位
_DF_ARMED = [False]           # 第一个 forward 没有"上一个"可等
try:
    _DRAIN_PER_FORWARD = int(os.environ.get("DS41M_DRAIN_PER_FORWARD", "0") or 0)
except ValueError:
    _DRAIN_PER_FORWARD = 1 if os.environ.get(
        "DS41M_DRAIN_PER_FORWARD", "").strip().lower() in ("1", "on", "true", "yes") else 0
if _DRAIN_PER_FORWARD not in (0, 1, 2):
    _DRAIN_PER_FORWARD = 1
# ★ CP 压缩层 hidden gather 预取（2026-10-25）：kv_compress 的 CP 路径要 gather 全序列
#   hidden（total_q×H×2B，64K=671MB）后才能跑厂商压缩；原实现在 kv_compress 入口
#   **同步等** gather（4 个压缩层合计暴露 2.8/4.9/2.1 ms/层）。现在 rmsnorm(attn)
#   之后就在**侧流**发起 gather（hidden 此时已定稿、到 kv_compress 之间只读），
#   kv_compress 消费时 `wait_event` —— 数值同一条 all-gather，逐位不变。
#   显存：预取张量寿命从"压缩算子内"提前到"注意力量子段"，峰值 +1 份 gather
#   （64K=671MB）；> `DS41M_CP_CMP_PREFETCH_MAX_GB`（默认 3 GiB，256K 2.68GB 可、
#   1M 10.7GB 关）自动跳过。`DS41M_CP_CMP_PREFETCH=0` 关闭。
_CP_PREFETCH_ENV = os.environ.get("DS41M_CP_CMP_PREFETCH", "auto").strip().lower()
_CP_PREFETCH_MAX_GB = float(os.environ.get("DS41M_CP_CMP_PREFETCH_MAX_GB", "3.0") or 3.0)
_CP_SIDE_STREAM = [None]
_CP_PREFETCH_ERR = [0]


def _cp_side_stream():
    s = _CP_SIDE_STREAM[0]
    if s is None:
        s = torch.npu.Stream()
        _CP_SIDE_STREAM[0] = s
    return s


def _cp_maybe_prefetch_cmp(st, ctx):
    """层内预取 kv_compress 的 CP hidden gather（见 _CP_PREFETCH_ENV 注释）。"""
    if _CP_PREFETCH_ENV in ("0", "off", "false", "no"):
        return
    cp = getattr(exec_ops, "CP_STATE", None)
    if cp is None:
        return
    # ★ 回放开启时只预取 replay 起点之前的层（2/8/14），层号 ≥20 跳过：
    #   那里的侧流 gather 会与 compact 段的同步 H2D 冲突（实测 2026-10-25）。
    if (os.environ.get("DS41M_DECODER_REPLAY", "").strip().lower()
            in ("1", "on", "true", "yes")
            and int(ctx.get("idx", -1)) >= 20):
        return
    if st.get("_cp_cmp_pre") is not None:
        return
    x = st.get("x")
    if not isinstance(x, torch.Tensor) or x.dim() != 2:
        return
    H = int(ctx.get("H") or 0)
    if H <= 0 or int(x.shape[1]) != H:
        return
    nbytes = int(cp.plan.total_q) * H * int(x.element_size())
    if nbytes > _CP_PREFETCH_MAX_GB * 2 ** 30:
        return
    try:
        side = _cp_side_stream()
        cur = torch.npu.current_stream()
        # 索引张量在**主流**建好（跨流复用要向 allocator 登记，免得侧流池提前复用）
        import cp_runtime as _cpr
        _ok = (len(cp.plan.inverse_reverse) == cp.plan.cp_size * cp.plan.local_rows
               and all(v >= 0 for v in cp.plan.inverse_reverse))
        _nm, _vals = (("inv", cp.plan.inverse_reverse) if _ok
                      else ("reverse", cp.plan.reverse_index))
        _idx = _cpr._plan_tensor(cp.plan, _nm, _vals, x.device)
        side.wait_stream(cur)
        try:
            _idx.record_stream(side)
        except Exception:                          # noqa: BLE001
            pass
        with torch.npu.stream(side):
            # ★ 单块 gather（chunk=local_rows）：侧流上如果分多块，多个 collective
            #   会和主流的 kv_epilog gather 交错，实测 CP2 64K 下一层出现 14ms 互等；
            #   单块后消失。字节数不变 ⇒ 数值逐位不变。
            full = cp.merge_global(x.reshape(-1, H), chunk=int(cp.plan.local_rows))
            ev = torch.npu.Event()
            ev.record()
        st["_cp_cmp_pre"] = (full, ev, x, cp)
    except Exception as exc:                       # noqa: BLE001
        _CP_PREFETCH_ERR[0] += 1
        if _CP_PREFETCH_ERR[0] <= 3:
            print(f"  ⚠️ [cp-prefetch] 关闭（{type(exc).__name__}: {exc}）", flush=True)
_ENV_OP_DIGEST = os.environ.get("DS41M_OP_DIGEST", "").strip().lower() in (
    "1", "on", "true", "yes")
_ENV_OP_DIGEST_LAYERS = frozenset(
    int(t) for t in os.environ.get("DS41M_OP_DIGEST_LAYERS", "0").replace(",", " ").split()
    if t.strip().lstrip("-").isdigit())
# ★ 2026-10-24（Phase 2，用户拍板默认开）：模块树驱动层内执行。
#   只换"执行序的来源"（modules.compose → 叶子序），叶子仍调 exec_ops.run_op；
#   首次使用与 legacy expand_layer 严格对拍，不一致默认直接报错（可显式回退）。
#   `DS41M_MODULE_RUNTIME=0/off/false/no` 一键回退 legacy。
_ENV_MODULE_RUNTIME = os.environ.get(
    "DS41M_MODULE_RUNTIME", "1").strip().lower() not in ("0", "off", "false", "no")
_ENV_MODULE_RUNTIME_FALLBACK = os.environ.get(
    "DS41M_MODULE_RUNTIME_FALLBACK", "").strip().lower() in ("1", "on", "true", "yes")


def _module_runtime_here(ctx, hit_stage=None, is_hit=False) -> bool:
    """Phase 2 初期边界：DSpark 与 decode 强制 legacy（顺序之外还有额外状态机）。

    ★ `is_decode_step` 在 hit stage B 里也因 `pos_offset>0` 为真（见 ctx_for），
    不能拿它单独判 decode——hit 段（真前缀带 `hit_stage`，fake 前缀靠
    `meta["hit"]>0`）必须放行，否则 hit 口径的 A/B 会静默退回 legacy。
    """
    if not _ENV_MODULE_RUNTIME:
        return False
    try:
        if int(os.environ.get("DS41M_DSPARK", "0") or 0) > 0:
            return False
    except ValueError:
        return False
    if hit_stage is not None or is_hit:
        return True
    return not ctx.get("is_decode_step")
# ★ 显存实测日志（默认关；见 memlog.py）。逐层 + 可选逐算子，只查询不改分配。
_ENV_MEMLOG = _memlog_mod.ENABLED
_ENV_MEMLOG_OPS = _memlog_mod.OPS
_ENV_MEMLOG_FROM = _memlog_mod.FROM
_ENV_MEMLOG_TO = _memlog_mod.TO
# ★ 显存自动阀（2026-10-25，默认 auto；阈值 90000）：非 CP 前向 token 数 ≥ AUTO_T
#   才**逐层** `empty_cache()`，以下完全不开。A/B（eager，noprof 纯 forward 6 轮，
#   max_split + 有界抢跑 K=2 口径）：
#   · T≤39462（峰 ≤20.0 GiB）：不开稳定且快 21–39%（64K@97 280→171；256K@90 1366→1080）；
#   · T=52569（25.8 GiB）：有界抢跑后不开也稳，2149 vs 开 2499（不开快 14%）；
#   · T=78783（37.1 GiB）：不开 3188 vs 开 3700（不开快 14%）；
#   · T=104998（48.45 GiB）：不开 5876 vs 开 4784（**开快 19%**，仍 allocator thrash）。
#   ⇒ 阈值取 90000（夹在 78783/104998 之间）。显式数字仍可强制（=0 关、=1 逐层）。
_EMPTY_CACHE_RAW = os.environ.get("DS41M_EMPTY_CACHE_EVERY", "auto").strip().lower()
_EMPTY_CACHE_AUTO = _EMPTY_CACHE_RAW in ("", "auto")
_ENV_EMPTY_CACHE_EVERY = 0 if _EMPTY_CACHE_AUTO else int(_EMPTY_CACHE_RAW or 0)
_EMPTY_CACHE_AUTO_T = int(os.environ.get("DS41M_EMPTY_CACHE_AUTO_T", "90000") or 0)
# ★ 2026-10-23 显存优化：residual 直接别名层输入（原来 x.clone() 是第二份 3D hidden）。
# ★★ 2026-10-26（T-036，本机 8 卡跑完 A/B）：**默认改成开**。
#   64k@0.97 交错 4 臂（各 8 轮）：关 115.21/115.42 vs 开 114.77/114.55 ⇒ **−0.66 ms（−0.57%）**，
#   四臂 `[packed-digest] sha1=08443723284dd9ed` 完全相同 ⇒ 逐位不变。
#   逐位安全性来自这里本来就是"只读别名"：attn 段 hc_post 之后 `st["residual"]` 会被刷新成新张量，
#   不会原地改别名的那份 x。`DS41M_NO_RESIDUAL_CLONE=0` 可回退（=1 等价默认）。
_ENV_NO_RESIDUAL_CLONE = os.environ.get(
    "DS41M_NO_RESIDUAL_CLONE", "1").strip().lower() in ("1", "on", "true", "yes")


def make_profiler(prof_dir, alop_config_path=""):
    """兼容入口：实现已移到 `profiling.py`（ALOP 是唯一采集后端，PERF_RULES §0）。"""
    return profiling.make_profiler(prof_dir, alop_config_path)


_RANK = 0        # main() 里设置；run_forward 里用


def start_line_sync() -> bool:
    """★ 起跑线（2026-10-15 用户口径）：整个 run **只在最最开始**做**一次**跨 rank barrier。

    作用：把 8 个 rank 拉到同一起跑线，**然后**才 `profiler.start()` 开采；采集区内
    （含 mega_moe 前后）**一律不插** barrier —— 插了既会把"等最慢 rank"的等待混进算子时长，
    trace 里还会多出 HcclLaunchAicpuKernel / EventWait 两类官方图里没有的 op。
    需要逐算子归因诚实时再显式 `DS41M_MOE_BARRIER=1`（调试开关，默认关）。
    `DS41M_START_NOBARRIER=1` 可把起跑线这一次也关掉；world<2 / PG 未建时自动跳过。
    """
    if os.environ.get("DS41M_START_NOBARRIER"):
        return False
    try:
        import torch.distributed as _d
        if not _d.is_initialized() or _d.get_world_size() < 2:
            return False
        _d.barrier()
        return True
    except Exception as exc:                       # noqa: BLE001
        print(f"  [起跑线] 跨 rank barrier 跳过（{type(exc).__name__}: {exc}）", flush=True)
        return False


def pass_barrier_sync(idx: int) -> bool:
    """★ 每 pass 前的跨 rank 对齐（2026-10-16 用户口径，`DS41M_PASS_BARRIER=1` 开，默认关）。

    与"起跑线只在最开始一次"的区别：这里在**每个 pass 边界**对齐一次，且放在
    `t_it = time.time()` **之前** → "等最慢 rank"的时间**不记进任何 pass 的账** ✓。
    它治的是 rank 漂移：先到的 rank 会在**下一个 pass 的 MoE/集合通信 kernel 里干等**
    （等待落进 kernel 时长 → MoE 虚高/逐点波动/离群，实测与本轮采集的离群同源），
    边界 barrier 把这笔等待赶到干净处 ✓。
    `torch.distributed.barrier()` 只对齐主机/HCCL 流 → 随后补 `torch.npu.synchronize()`
    排空设备，才是真正的 pass 边界 ✓。
    代价：总墙钟 = Σ(最慢 rank)；trace 在 pass **之间**多出 3 条 HcclLaunchAicpuKernel
    （不进任何 pass 的算子表 ✓）。与 DS41M_PIPELINE 语义互斥（barrier 会把流水排空）。
    """
    try:
        import torch.distributed as _d
        if _d.is_initialized() and _d.get_world_size() >= 2:
            _d.barrier()
    except Exception as exc:                       # noqa: BLE001
        print(f"  [pass {idx}] barrier 跳过（{type(exc).__name__}: {exc}）", flush=True)
        return False
    torch.npu.synchronize()
    return True


def strip_prefix(w, layer_idx):
    """把 `layers.<i>.` 去掉 —— exec_ops 里用短名（attn.wq_a.weight 这样）。"""
    pre = f"layers.{layer_idx}."
    return {(k[len(pre):] if k.startswith(pre) else k): v for k, v in w.items()}


_PACK_CACHE: dict = {}      # 跨层共用权重（share_across_layers）时的打包结果缓存


# ★ T-043 ④（实验，默认 0 = 关）：wo_a 走 deep_gemm.fp8_einsum 的权重对。
#   与 `pack_wo_a`（transpose+NZ，供 transposeQMM）**并存**：这里额外存一份
#   `(W[h,d,r] 原始 ND view, SF 按 k-major 配对的 int16)`，运行期由 `_wo_a` 选用。
#   打包失败（后端/布局不支持）只打印一次并回退，不影响原路径。
_ENV_WOA_EINSUM = os.environ.get("DS41M_WOA_EINSUM", "1").strip().lower() in (   # 2026-10-26 T-054 默认开
    "1", "on", "true", "yes")


_STRIP_CACHE: dict = {}
_PACKED_LAYER_CACHE: dict = {}
# ★ 2026-10-17：真权重也要**跨 pass 缓存每层的布局转换**（transpose + NZ cast + fp4 打包）。
#   以前只有 share_across_layers（合成权重）才缓存；真权重每层每 pass 重做 →
#   decode 每次前向白烧 ~336 ms（TransData）。缓存键带层号（同名的层间值不同）。
_PACK_CACHE_L: dict = {}
# ★ 2026-10-19：真权重每 pass 还要**重扫一遍 ~320 个键**重建 `_packed` 字典（实测层前
#   0.02s/pass vs 合成 0.01s）。打包结果本身已在 `_PACK_CACHE_L` 里常驻，这里只缓存
#   字典本身（零额外显存）→ 同层第二次起直接返回。
_PACKED_SUB_CACHE: dict = {}


_CTX_CACHE: dict = {}
_SEQ_CACHE: dict = {}
_MODULE_PLAN_CACHE: dict = {}          # Phase 2：layer idx → LayerPlan（False = 已回退 legacy）


# ★ 2026-10-21：META 的"每 pass 清一次"改成 **几何变了才清**。
#   原实现（每 pass 清空元数据缓存）下一层的
#   `_meta` 重建就是一次 `torch.tensor(list, device=npu)` = **阻塞 H2D**；流水跑法下要等
#   整条设备队列排空，实测单次 54–64ms（sync-trace 栈：_mqsmla 的 cmp_seqlen）。
#   prefill 各 pass 的几何完全相同（T/pos/cu/lens 都不变）→ 缓存跨 pass 复用是安全的；
#   decode（pos_offset 每步变）仍会按新几何清，语义不变。
_META_GEO = [None]


def pack_layer_weights(w, cfg, layer_idx=None):
    """给本层权重做**后端布局转换**（mxfp8 线性层：transpose + NZ；scale 原样）。
    合成/真实两条路共用这里（PLAN §9 纪律 2）。

    `weights.share_across_layers=true` 时各层的源张量本来就是同一份 → 打包结果也按
    "去层号的名字 + 形状"缓存（transpose + NZ cast 都是每秒 GB 级的活，别每层重做；
    多卡长跑/采集时这一步会直接吃掉可观的时间）。

    ★ 缓存键必须是**完整名**：踩过 —— `k.split(".", 2)[-1]` 对**已经 strip_prefix 过**的键
    （`attn.wq_a.weight` 只有两段）会取到 `"weight"`，于是 wq_a/wq_b/wkv/wo_b 全挤到同一个键上，
    全都拿到第一份打包结果（输出宽度变成 512=head_dim）→ q_norm/wq_b/mqsmla/wo_a/wo_b 整块级联挂。
    """
    share = bool((cfg.get("weights") or {}).get("share_across_layers", False))
    if not share and layer_idx is not None:
        _cached_packed = _PACKED_SUB_CACHE.get(int(layer_idx))
        if _cached_packed is not None:          # 同层跨 pass：布局转换都已做过（见 _PACK_CACHE_L）
            w["_packed"] = _cached_packed
            return w
    packed = {}
    for k in list(w):
        if not k.endswith(".weight") or w[k].device.type != "npu":
            continue                    # CPU 上的（engram 表）不做 NPU 布局转换
        if w[k].dtype == torch.float8_e4m3fn:
            scale = w.get(k.replace(".weight", ".scale"))
            if scale is None:
                continue
            if k.startswith("attn.indexer."):
                # 索引器权重走 `_nz()`（原始 ND→NZ，形状 [4096,1280]）；这里 pack_qmm 会再打
                # 一份**转置过的、没人用**的 NZ（8 层 × 5.2 MB），跳过。
                continue
            wo_a = k.endswith("attn.wo_a.weight")       # wo_a 走专用打包（官方 permute）
            name = k.split(".", 2)[-1] if k.startswith("layers.") else k
            # ★ 真权重：键 = (层号, 名, 形状, wo_a)；合成：键不含层号（各层同源）
            if share:
                ck, cache = (name, tuple(w[k].shape), wo_a), _PACK_CACHE
            elif layer_idx is not None:
                ck, cache = (int(layer_idx), name, tuple(w[k].shape), wo_a), _PACK_CACHE_L
            else:
                ck, cache = None, None
            if ck is not None and ck in cache:
                packed[k] = cache[ck]
                # ★ T-043 ④：缓存命中也必须把 einsum 权重对带出来 —— 否则第二个
                #   pass 起 `_packed` 里缺键 → 热路径静默回退（踩过：首 pass 有、
                #   次 pass 没有，host 表看不出、只有 [woa-einsum] fallback 计数会涨）。
                if wo_a and _ENV_WOA_EINSUM:
                    _ep = cache.get(ck + ("einsum",))
                    if _ep is not None:
                        packed["attn.wo_a.weight.einsum"] = _ep
                continue
            p = exec_ops.pack_wo_a(w[k], scale) if wo_a else exec_ops.pack_qmm(w[k], scale)
            if ck is not None:
                cache[ck] = p
            packed[k] = p
            # ★ T-043 ④（实验）：einsum 版权重对，与上面同一份缓存（键多一个 "einsum"）。
            if wo_a and _ENV_WOA_EINSUM:
                eck = (ck + ("einsum",)) if ck is not None else None
                if eck is not None and eck in cache:
                    ep = cache[eck]
                else:
                    try:
                        ep = exec_ops.woa_einsum_pack(w[k], scale)
                    except Exception as _e:          # 后端/布局不支持 → 热路径回退
                        ep = None
                        print(f"  ⚠ wo_a einsum 打包失败（回退 transposeQMM）："
                              f"{type(_e).__name__}: {str(_e)[:120]}")
                    if eck is not None:
                        cache[eck] = ep
                if ep is not None:
                    packed["attn.wo_a.weight.einsum"] = ep
    # ★ 非 QMM 的两处"加载期布局转换"（以前在算子热路径里每层现做，白送 kernel）：
    #   gate 权重转置（官方 npu_fused_matmul 要 [K,N]，checkpoint 存的是 [N,K]）
    #   gate bias 的 fp32（官方 gate_topk 的 bias 走 float32）
    # ★★ 真权重：这几处以前**每 pass 重做**（每个小转换 1 TransData/Transpose，40 层累计 ~10ms/pass）。
    #   值只依赖本层权重（跨 pass 不变）→ 按层缓存。合成模式按"去层号"缓存（各层同源）。
    _pcache = _PACK_CACHE if share else _PACK_CACHE_L

    def _pget(name, shape, build):
        if not share and layer_idx is None:
            return build()
        ck = (name, tuple(shape)) if share else (int(layer_idx), name, tuple(shape))
        v = _pcache.get(ck)
        if v is None:
            v = build()
            _pcache[ck] = v
        return v

    if w.get("ffn.gate.weight") is not None and w["ffn.gate.weight"].device.type == "npu":
        _gw = w["ffn.gate.weight"]
        packed["ffn.gate.weight.T"] = _pget(
            "gateT", _gw.shape, lambda: _gw.t().contiguous())
    if w.get("ffn.gate.bias") is not None and w["ffn.gate.bias"].device.type == "npu":
        _gb = w["ffn.gate.bias"]
        packed["ffn.gate.bias.f32"] = _pget(
            "gateBiasF32", _gb.shape, lambda: _gb.float().contiguous())
    #   indexer 的 k_norm 权重（官方 modeling_deepseek.py:2648-2650 把它 cast 成 fp32）——
    #   以前每个"压缩源层"现做 to(float32).contiguous()（1 Cast + 1 Copy × 8 层/轮）
    _s_wqb = w.get("attn.indexer.wq_b.scale")
    if _s_wqb is not None and _s_wqb.device.type == "npu":
        _np = (_s_wqb.shape[-1] + 1) // 2      # ★ 是**末维**（列数）的一半，不是行数
        packed["attn.indexer.wq_b.scale.pairs"] = _pget(
            "wqbPairs", _s_wqb.shape,
            lambda: _s_wqb.view(torch.uint8).reshape(_s_wqb.shape[0], _np, 2).contiguous())
    if w.get("attn.indexer.k_norm.weight") is not None:
        _kn = w["attn.indexer.k_norm.weight"]
        packed["attn.indexer.k_norm.weight.f32"] = _pget(
            "kNormF32", _kn.shape, lambda: _kn.to(torch.float32).contiguous())
    #   compressor 的 norm 权重同理（官方 modeling_deepseek.py:2646-2649 也 cast 成 fp32）——
    #   以前**每次 kv_compress 现做** to(float32) + contiguous（bf16[512]，1 Cast + 1 Copy），
    #   真权重 16k 下是 4 层/轮 × 每 pass。
    _cn = w.get("attn.compressor.norm.weight")
    if _cn is not None and _cn.device.type == "npu":
        packed["attn.compressor.norm.weight.f32"] = _pget(
            "cmpNormF32", _cn.shape, lambda: _cn.to(torch.float32).contiguous())
    w["_packed"] = packed
    if not share and layer_idx is not None:
        _PACKED_SUB_CACHE[int(layer_idx)] = packed
    return w


_RAW_PACK_RELEASED: set = set()


def _release_packed_raw(source, w, layer_idx):
    """打包完成后释放 **raw（ND）** 的 fp8 权重（转换产物已接管；真权重专用）。

    覆盖：`attn.wq_a/wq_b/wkv/wo_a/wo_b`（`_qmm` 全读 `_packed`）+ `engram.wkv`
    （侧流 `core_qmm` 同样读 `_packed`）。每层 ~127 MB（engram 层再多 157 MB），
    40 层合计 ≈ **5.4 GiB** —— 之前 raw 与转置/NZ 副本同时常驻。

    不动：`ffn.shared_experts.*`（ascendc 融合要 raw，由 exec_moe 的 AC 释放路径接管）、
    gate/norm/compressor（bf16，本来就不打包）、专家权重（I8，不打包，另有 drop_layer_experts）。
    """
    if layer_idx is None:
        return 0
    _li = int(layer_idx)
    if _li in _RAW_PACK_RELEASED:                 # 每层只做一次（跨 pass）
        return 0
    _RAW_PACK_RELEASED.add(_li)
    if os.environ.get("DS41M_PACK_RELEASE_RAW", "1").strip().lower() in ("0", "off", "false", "no"):
        return 0
    if source is None or not hasattr(source, "drop_names"):
        return 0                                   # 合成权重（share）不释放：base tensor 是跨层共享的
    # ★ 必须是**白名单**：indexer 的 wq_b 虽然也被 pack_qmm 打了一份（其实没人用），
    #   但 `_nz()` 每次仍要 `w.get(name)` 拿原始 ND 张量 —— 释放它会让下一个 pass KeyError。
    _releasable = ("attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight",
                   "attn.wo_a.weight", "attn.wo_b.weight", "engram.wkv.weight")
    _pk = w.get("_packed") or {}
    names = []
    for k in _releasable:
        if k in _pk:                       # 该权重确实打包成功 → raw 的权重与 scale 都死了
            names.append(f"layers.{_li}.{k}")
            names.append(f"layers.{_li}.{k.replace('.weight', '.scale')}")
    _n = source.drop_names(names) if names else 0
    if _n and os.environ.get("DS41M_DEBUG"):
        print(f"  [pack] 层 {_li}：释放 raw 权重 {_n} 项（transpose/NZ 副本接管）", flush=True)
    return _n


def _tokenizer(cfg):
    """DS4.1 的 tokenizer（weights.path 下有 tokenizer.json）。"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg["weights"]["path"], trust_remote_code=True)


def _chat_ids(cfg, tok, text, thinking="chat"):
    """官方 chat 模板：`<model_dir>/encoding/encoding.py::encode_messages`。

    该权重目录的 tokenizer_config.json **没有 chat_template**；官方 `inference/generate.py`
    的做法就是 `tokenizer.encode(encode_messages(messages, thinking_mode=...))`。
    encoding.py 自包含（只用标准库），直接按文件路径加载即可。
    """
    import importlib.util as _ilu
    mp = str((cfg.get("weights") or {}).get("path") or "")
    fp = os.path.join(mp, "encoding", "encoding.py")
    if not os.path.exists(fp):
        raise SystemExit(f"--chat 需要官方 encoding 包：{fp} 不存在")
    spec = _ilu.spec_from_file_location("dsv41_encoding", fp)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prompt = mod.encode_messages([{"role": "user", "content": text}],
                                 thinking_mode=thinking)
    return tok.encode(prompt, add_special_tokens=False)


def embed_ids(cfg, ids, source, device):
    """词嵌入：x = embed.weight[ids]（真权重 [129280,5120] BF16）。"""
    name = "embed.weight"
    w = source.get(name, "BF16", (model_spec.get().vocab, cfg["model"]["hidden"]))
    idx = torch.tensor(ids, dtype=torch.long, device=w.device)
    return torch.index_select(w, 0, idx).to(torch.bfloat16).contiguous()


def head_logits(cfg, x, pre_mix, source):
    """末层 hidden [T,M,H] → hc_pre_mix 塌缩 → 最终 norm → lm head → logits。

    官方 modeling_deepseek.py:2321-2322：`hc_pre_mix(hidden_states, pre_mix)` 用的
    pre_mix 是**最后一层 ffn 段 hc_pre 返回的那个**（逐层往下传的那个）。
    """
    h_last = x[-1]                                       # [M, H]
    if pre_mix is not None and pre_mix.dim() == 2 and pre_mix.shape[1] == h_last.shape[0]:
        pm = pre_mix[-1].float()                          # [M]
        h_last = (pm.unsqueeze(-1) * h_last.float()).sum(0).to(x.dtype)
    w = source.get("norm.weight", "BF16", (cfg["model"]["hidden"],))
    h = torch_npu.npu_rms_norm(h_last.view(1, -1), w, 1e-20)[0]
    hw = source.get("head.weight", "BF16",
                    (model_spec.get().vocab, cfg["model"]["hidden"]))
    return torch.matmul(h.view(1, -1), hw.t())[0].float()


def head_logits_t(cfg, x, pre_mix, source, norm_name="norm.weight"):
    """T>1 的 head：逐行 hc_pre_mix → norm → lm head → [T,V] float32。

    `norm_name` 允许草稿模型用 `mtp.2.norm.weight`；主模型验证用默认 `norm.weight`。
    与 `head_logits` 的 T=1 结果逐位一致（同一 rms_norm/matmul 路径）。
    """
    if x.dim() == 3:
        if pre_mix is not None and pre_mix.dim() == 2 and pre_mix.shape[1] == x.shape[1]:
            h = (pre_mix.float().unsqueeze(-1) * x.float()).sum(1).to(x.dtype)
        else:
            h = x.mean(dim=1).to(x.dtype)
    else:
        h = x
    w = source.get(norm_name, "BF16", (cfg["model"]["hidden"],))
    h = torch_npu.npu_rms_norm(h, w, 1e-20)[0]
    hw = source.get("head.weight", "BF16",
                    (model_spec.get().vocab, cfg["model"]["hidden"]))
    return torch.matmul(h, hw.t()).float()


def make_input(cfg, T, device=None, seed=0):
    """本 rank 的输入：合成/真实都用同一形状（[T,H] bf16）。

    `seed` 供 `--batches` 给每个 batch 造**独立**输入（默认 0 = 与旧行为逐位一致）。
    """
    device = device or cfg["run"].get("device", "npu:0")
    torch.manual_seed(seed)
    return torch.randn(T, cfg["model"]["hidden"], dtype=torch.float32).to(torch.bfloat16).to(device)


def _decode_step_barrier(world):
    """decode 的**每步边界 barrier**（DS41M_DECODE_BARRIER=1）。

    为什么要它：decode 每步工作量很小，各 rank 在 Python 侧会漂；厂商 engram 侧流
    （独立 HCCL 通信域）与 mega_moe 的集合通信对"步序"敏感 → 实测漂移时出现过
    rank1 `vector core exception`（Muls 核）。加 barrier 后步序严格对齐（边界等待
    不计入步耗时；要"纯流水"计时时把开关关掉）。
    """
    if not world or int(world) <= 1:
        return
    try:
        import torch.distributed as _d
        if _d.is_initialized():
            _d.barrier()
    except Exception:                      # noqa: BLE001
        pass


def decode_greedy(cfg, source, ids, n_new, tok, stop_ids=()):
    """贪心续写 N 个 token —— **全量重算版**（每步把整段 prefix 重新前向一遍）。

    为什么先要这一版：
      · 它走的就是**已验证与官方逐位一致**的那条 prefill 代码路径 ⇒ 生成文本可信，
        同时给出每一步的 logits 金标准（后面接**增量 decode** 时逐步对拍用）；
      · 代价 O(N²)：16k prompt 每步 ~2.4s，32 步约 80s（短 prompt 则每步 ~0.2s）。

    用法：runner.py <config> --ids-file <ids.pt> --decode 32
    输出：每步 id/文本 + 最终整句（tokenizer.decode）。
    """
    dev = cfg["run"].get("device", "npu:0")
    # ★ 固定 MoE 对称缓冲容量（不随每步 T 变 → 不会中途重建撞 HCCL/aicore）
    if not os.environ.get("DS41M_MOE_CHUNK"):
        os.environ["DS41M_MOE_CHUNK"] = "auto"
    shared = {"_win_cache_A": None, "_win_cache_B": None, "_cmp_cache": None}
    out_ids = list(int(i) for i in ids)
    pieces = []
    _bar = os.environ.get("DS41M_DECODE_BARRIER", "").strip().lower() in ("1", "on", "true", "yes")
    _world = int(cfg["parallel_config"].get("world_size", 1) or 1)
    _step_secs = []
    t_start = time.time()
    for step in range(n_new):
        T = len(out_ids)
        cfg["run"]["seq_len"] = T            # 池/元数据按当前全序列长度开（全量重算 ⇒ 每步重写）
        emb = embed_ids(cfg, out_ids, source, dev)
        r = run_prefill_auto(cfg, source, T, None, profiler=None, verbose=False,
                        allow_placeholder=False, prog=None, inputs=emb,
                        collect_dir="", input_ids=out_ids, stop_profiler=True,
                        shared=shared, hit_stage=None)
        logits = head_logits(cfg, r["x"], r.get("pre_mix"), source)
        _step_secs.append(time.time() - t_start)      # 本步耗时（= 一次整段前向）
        t_start = time.time()
        top = torch.topk(logits, 3)
        nxt = int(top.indices[0])
        piece = tok.decode([nxt])
        pieces.append(piece)
        if _RANK == 0:
            print(f"  [decode {step + 1}/{n_new}] pos={T} id={nxt} {piece!r}  "
                  f"top3={[(int(i), round(float(s), 2)) for s, i in zip(top.values, top.indices)]}"
                  f"  本步 {_step_secs[-1]:.2f}s", flush=True)
        if stop_ids and nxt in stop_ids:
            print("  [decode] 命中 EOS，停止", flush=True)
            break
        out_ids.append(nxt)
        if _bar:
            _decode_step_barrier(_world)
        t_start = time.time()
    text = tok.decode(out_ids[len(ids):])
    if _RANK == 0:
        print(f"\n=== decode 输出（{len(pieces)} 个新 token）===")
        print(text)
        print("=" * 60)
        if _step_secs:
            _w = _step_secs[1:] or _step_secs
            print(f"  耗时：首步(含 prefill/装载) {_step_secs[0]:.2f}s；后续 {len(_w)} 步均 "
                  f"{sum(_w) / len(_w):.2f}s（≈ {1000 * sum(_w) / len(_w):.0f} ms/token）", flush=True)
    return out_ids


def _cp_redistribute_rows(t, contrib_idx, boundary_idx, group, has_pad):
    """DP-free CP 行重分布：本地行 → 截尾保留行 → all-gather → 本 rank compact-mirror 行。

    与 DeepInfer `yoco_boundary_compact` 同三步：
      ① `gather(x, yoco_cp_contrib_index)`：各 rank 挑出"截尾后仍存在"的本地行；
      ② `cp.all_gather(contrib)`：rank 序拼接；
      ③ `gather(gathered, yoco_cp_boundary_index)`：取本 rank compact-mirror 的行。
    pad 行（boundary_idx=-1）填 0。
    """
    import torch.distributed as _d
    sel = t.index_select(0, contrib_idx.clamp_min(0)).contiguous()
    world = _d.get_world_size(group) if _d.is_initialized() else 1
    if world <= 1:
        buf = sel
    else:
        parts = [torch.empty_like(sel) for _ in range(world)]
        _d.all_gather(parts, sel, group=group)
        buf = torch.cat(parts, 0)
    out = buf.index_select(0, boundary_idx.clamp_min(0))
    if has_pad:
        shape = (-1,) + (1,) * (out.dim() - 1)
        out = torch.where((boundary_idx >= 0).view(shape), out, torch.zeros_like(out))
    return out


def _boundary_cut_state_cp(st, ctx, W, pos_off, cfg=None):
    """CP 版层 20 边界：**截尾 + 跨 rank 重分布**（DeepInfer YOCO compaction 同口径）。

    前置：本层 KV/压缩/索引 K 已按**全长**写完（CP 分支内部 gather）。
    之后：本地行 = 各 rank 的 mirror 行 → 保留最后 W 行 → 按 **compact-mirror**
    重新切分（DeepInfer `prepare_cp_mirror(compact)`），继续算 21..39。
    """
    import cp_layout as _cpl
    import cp_state as _cps_mod
    cp = exec_ops.CP_STATE
    if cp is None:
        raise RuntimeError("_boundary_cut_state_cp 需要 CP_STATE")
    lens = [int(x) for x in cp.plan.seq_lens]
    if len(lens) != 1:
        raise RuntimeError("CP 边界拆分目前只支持单序列（packed 多请求待接）")
    tot = int(cp.plan.total_q)
    keep = [max(1, min(int(W), lens[0]))]
    if keep[0] >= lens[0]:
        return ctx, int(pos_off), 0
    # ★ 2026-10-28（命中 + CP）：plan 的行数是**本段**行数（命中 stage B 只有 R 行），
    #   绝对起点 = 段起点 + 段内偏移（= CP1 的 pos_off + drop）；seq_len 用**绝对终点**
    #   （= 原序列长 cb.seq_len），否则 compact 段的 rope/seqused/索引器宽度全错。
    _seq_abs = int(getattr(cp, "seq_len", 0) or tot)
    pos_off2 = int(pos_off) + int(lens[0] - keep[0])

    contribs, gathered, padn = _cpl.boundary_indices(lens, keep, cp.cp_size, cp.cp_rank)
    mine = list(contribs[cp.cp_rank])
    cidx = torch.tensor(mine + [-1] * (padn - len(mine)), dtype=torch.long,
                        device=cp.device)
    cplan = cp.with_compact(keep, pos_offset=pos_off2, seq_len=_seq_abs)
    bvals = list(_cpl.compact_boundary_index(cplan.plan, gathered))
    bidx = torch.tensor(bvals, dtype=torch.long, device=cp.device)
    has_pad = any(v < 0 for v in bvals)

    n_rows = int(st["x"].shape[0])
    _keys = ("x", "residual", "hc_pre_t", "q", "qr", "idx_q", "idx_descale_q",
             "idx_w", "idx_sparse", "cmp_sparse_indices", "cmp_topk_length")
    for k in _keys:
        v = st.get(k)
        if torch.is_tensor(v) and v.dim() >= 1 and int(v.shape[0]) == n_rows:
            st[k] = _cp_redistribute_rows(v, cidx, bidx, cp.group, has_pad)
    hct = st.get("hc_post_t")
    if isinstance(hct, tuple):
        st["hc_post_t"] = tuple(
            (_cp_redistribute_rows(t, cidx, bidx, cp.group, has_pad)
             if torch.is_tensor(t) and t.dim() >= 1 and int(t.shape[0]) == n_rows else t)
            for t in hct)
    for k in ("kv", "k"):
        v = st.get(k)
        if torch.is_tensor(v) and v.dim() >= 1 and int(v.shape[0]) == n_rows:
            st[k] = None
    # 窗口 scratch：全序列池的最后 W 行搬到 0..W-1（与 CP1 的 rebase 同口径）
    moved = exec_ops.rebase_window_pool(st, ctx, tot, keep[0])
    exec_ops.set_cp_state(cplan)
    # 后续层的 ctx 用 `_packed.cu` 作行边界；compact 段本地行数变了要同步
    # （_packed 恒存在：varlen 唯一口径，2026-10-26）
    import metadata as _md
    _md.set_layout(cfg, cu=[0, int(cplan.local_rows)], row_lens=[int(cplan.local_rows)])
    _hc_cap = int(os.environ.get("DS41M_HC_PAD", "8192") or 0)
    cu = ctx.get("cu_seqlens")
    ctx2 = dict(ctx, T=keep[0], pos_offset=pos_off2,
                cu_seqlens=([0, keep[0]] if cu is not None else None),
                seq_lens=([_seq_abs] if ctx.get("seq_lens") else None),
                row_lens=([keep[0]] if ctx.get("row_lens") else None),
                n_seqs=1, moe_nspan=0, pos_offsets=None,
                # K 池已由 cut 前的 idx_k_write 按全长写好：compact 段只读不重写
                idx_k_ready=True,
                hc_pad_ref=(tot if _hc_cap > 0 else 0), hc_pad_cap=_hc_cap,
                cp_positions=cplan.local_positions,
                cp_abs_positions=cplan.abs_positions,
                # seqused_ori_kv / rope 表宽都用**绝对终点**（CP1 的 pos_offset+T = L）
                cp_total_q=_seq_abs, cp_seq_len=_seq_abs)
    if _RANK == 0:
        print(f"  [cp-boundary] 层截尾：全序列 {tot} → 保留 {keep[0]}（每请求），"
              f"compact-mirror 本地 {cplan.local_rows} 行（pad={int(has_pad)}），"
              f"pos_offset={pos_off2}", flush=True)
    if _PROBE_ON:
        ctx2 = _probe.wrap_ctx(ctx2)
    return ctx2, pos_off2, moved


def _cp_replay_trim(x, input_ids, shared, cfg, cp, W):
    """层 21 入口：compact 段从 `_split_pad`(512) 再裁到 replay W(140)，**跨 rank 重分布**。

    与 DeepInfer `yoco_boundary_compact` 同一套 contrib/boundary 索引，只不过这次的
    输入是"compact-512 的 mirror 布局"，输出是"compact-140 的 mirror 布局"。
    """
    import cp_layout as _cpl
    cur = [int(v) for v in cp.plan.seq_lens]
    keep = [max(1, min(int(W), v)) for v in cur]
    if all(k >= c for k, c in zip(keep, cur)):
        return None
    if len(cur) != 1:
        raise RuntimeError("CP replay 裁剪目前只支持单序列")
    contribs, gathered, padn = _cpl.boundary_indices(cur, keep, cp.cp_size, cp.cp_rank)
    mine = list(contribs[cp.cp_rank])
    cidx = torch.tensor(mine + [-1] * (padn - len(mine)), dtype=torch.long,
                        device=cp.device)
    new_ofs = int(cp.pos_offset) + (cur[0] - keep[0])
    cplan = cp.with_compact(keep, pos_offset=new_ofs, seq_len=cp.seq_len)
    bvals = list(_cpl.compact_boundary_index(cplan.plan, gathered))
    bidx = torch.tensor(bvals, dtype=torch.long, device=cp.device)
    has_pad = any(v < 0 for v in bvals)
    T_full = int(x.shape[0])
    x = _cp_redistribute_rows(x, cidx, bidx, cp.group, has_pad)
    if input_ids is not None and len(input_ids) >= T_full:
        ids_t = torch.as_tensor(input_ids, dtype=torch.long, device=cp.device)
        ids_t = _cp_redistribute_rows(ids_t, cidx, bidx, cp.group, has_pad)
        input_ids = ids_t.tolist()
    if isinstance(shared, dict) and _RANK == 0 and os.environ.get("DS41M_CP_TRIM_DBG"):
        for _k in list(shared):
            if isinstance(_k, str) and _k.startswith(replay.ROW_ACCOUNT_PREFIXES):
                _t = shared.get(_k)
                if torch.is_tensor(_t) and _t.dim() >= 1:
                    _n = int((_t.reshape(_t.shape[0], -1) >= 0).sum(-1).min()) \
                        if _t.dtype == torch.int32 and "sparse" in _k else -1
                    print(f"  [cp-trim-dbg] {_k} shape={tuple(_t.shape)} min_valid={_n}", flush=True)
        print(f"  [cp-trim-dbg] cur={cur} keep={keep} cidx={cidx.tolist()[:40]} "
              f"bidx={bidx.tolist()[:20]} padn={padn}", flush=True)
    if isinstance(shared, dict):
        pm = shared.get("_ffn_pre")
        if torch.is_tensor(pm) and int(pm.shape[0]) == T_full:
            shared["_ffn_pre"] = _cp_redistribute_rows(pm, cidx, bidx, cp.group, has_pad)
        n_cut = 0
        for k in list(shared):
            if not (isinstance(k, str) and k.startswith(replay.ROW_ACCOUNT_PREFIXES)):
                continue
            t = shared.get(k)
            if torch.is_tensor(t) and t.dim() >= 1 and int(t.shape[0]) == T_full:
                shared[k] = _cp_redistribute_rows(t, cidx, bidx, cp.group, has_pad)
                n_cut += 1
    else:
        n_cut = 0
    exec_ops.set_cp_state(cplan)
    import metadata as _md
    _md.set_layout(cfg, cu=[0, int(cplan.local_rows)], row_lens=[int(cplan.local_rows)])
    pos_off = int(new_ofs)
    msg = (f"  [replay:cp] 层入口裁剪：compact {cur[0]} → {keep[0]}（每请求），"
           f"本地 {T_full} → {cplan.local_rows} 行，逻辑位置 "
           f"{pos_off}..{pos_off + keep[0] - 1}"
           + (f"（行账 ×{n_cut} 同步重分布 ✓）" if n_cut else ""))
    return {"x": x, "input_ids": input_ids, "pos_off": pos_off, "message": msg}


def _boundary_cut_state(st, ctx, W, pos_off):
    """DeepInfer 层 20 边界：KV 已按全长写完 → 把行状态截到 W 行。

    - `st["x"]` / `residual` / `hc_pre_t` / `hc_post_t` 同步裁尾；
    - `st["kv"]` 已写池，丢弃；
    - 窗口 scratch 的每条序列最后 W 行搬到序列起始槽（rebase），后续仍用局部 0..W-1；
    - `ctx` 换成 compact 视图（pos_offset 前移 T-W，cu/row_lens 同步）。
    """
    T = int(st["x"].shape[0])
    if W >= T:
        return ctx, int(pos_off), 0
    st["x"] = st["x"][-W:].contiguous()
    for k in ("residual", "hc_pre_t", "q", "qr", "idx_q", "idx_descale_q", "idx_w",
              "idx_sparse", "cmp_sparse_indices", "cmp_topk_length"):
        v = st.get(k)
        if torch.is_tensor(v) and v.dim() >= 1 and int(v.shape[0]) == T:
            st[k] = v[-W:].contiguous()
    hct = st.get("hc_post_t")
    if isinstance(hct, tuple):
        st["hc_post_t"] = tuple(
            (t[-W:].contiguous()
             if torch.is_tensor(t) and t.dim() >= 1 and int(t.shape[0]) == T else t)
            for t in hct)
    for k in ("kv", "k"):
        v = st.get(k)
        if torch.is_tensor(v) and v.dim() >= 1 and int(v.shape[0]) == T:
            st[k] = None
    # 随行/行账（若有）同步裁尾
    replay._trim_shared(st.get("_shared"), None, T, False, W)
    moved = exec_ops.rebase_window_pool(st, ctx, T, W)
    drop = T - W
    pos_off2 = int(pos_off) + drop
    cu = ctx.get("cu_seqlens")
    # `npu_hc_pre_v2` 的 post/comb/pre 按 M 分块（y 不受影响）：实测等价类为
    # ≤512 / 652 / 1024 / 2048 / 4096 / ≥6144（6144..32768 同一类，见
    # tmp/hc_ab/classes.log）。compact 段把输入零补到**参考段 M**（M≤上限时逐位
    # 精确）；参考 M 超过上限时补到上限（默认 8192，落在 ≥6144 类）。
    _hc_cap = int(os.environ.get("DS41M_HC_PAD", "8192") or 0)
    ctx2 = dict(ctx, T=W, pos_offset=pos_off2,
                cu_seqlens=([0, W] if cu is not None else None),
                seq_lens=([int(ctx.get("L_full", T))] if ctx.get("seq_lens") else None),
                row_lens=([W] if ctx.get("row_lens") else None),
                n_seqs=1, moe_nspan=0,
                # compact 段是单序列视图：清掉 packed 的逐序列起始位置（否则 mqsmla/qli
                # 的 packed 分支会拿 cut 前的 pos_offsets，和裁后的 pos_offset 不一致）。
                pos_offsets=None,
                # 索引 K 池已在 cut 前由 `idx_k_write` 按全长 latent 写好：compact 段
                # 的 qli 只读池，不再重写（否则 full latent + row0=pos_offset 写错行）。
                idx_k_ready=True,
                hc_pad_ref=(T if _hc_cap > 0 else 0), hc_pad_cap=_hc_cap)
    if _PROBE_ON:
        ctx2 = _probe.wrap_ctx(ctx2)
    return ctx2, pos_off2, moved


def _run_forward_impl(cfg, source, T, n_layers=None, profiler=None, verbose=True,
                      allow_placeholder=False, prog=None, inputs=None, collect_dir="",
                      input_ids=None, stop_profiler=True, shared=None, hit_stage=None,
                      no_sync=False, pos_offset=None, l_full=None, engram_ctx=0,
                      stage_prefix="", post_layer_cb=None, split_tail=False):
    """`hit_stage`（只在 hit.ratio>0 时有意义）：

      · None  → 单段跑。`hit.source=fake`（合成权重默认）走这里：前缀 = 池子初值（零），
                **各层共用同一份前缀池**（压缩 KV/索引缓存本来就按 ratio 共享，窗口池是
                每层自己重写的 scratch）→ 形状/次数/耗时口径与官方 stage B 一致。
      · "A"   → 真前缀 stage A：把前缀 [0,H) 真算一遍，KV/压缩 KV/索引器缓存/压缩器 state
                全部落进 `shared` 池（调用方传入同一个 dict，跨段存活）。
      · "B"   → stage B（被测量的那一段）：只算尾部 R 个 token，读 stage A 留下的前缀缓存。
    """
    # ★ 2026-10-21 诊断：pass 计数，供 `DS41M_META_TRACE=1` 标记 `_meta` 未命中发生在哪一轮
    #   （预热轮 vs 采集轮）。纯计数，无额外 host 开销。
    exec_ops.PASS_NO = getattr(exec_ops, "PASS_NO", 0) + 1
    if _ENV_MEMLOG:
        try:
            torch.npu.reset_peak_memory_stats()
        except Exception:                          # noqa: BLE001
            pass
        _memlog_mod.mem(f"forward start T={T}", rank=_RANK)
    # ★ 2026-10-21 pure：日志等级进程内不变 → 每 pass 取一次，层/算子循环里只比布尔。
    #   pure 下 runner 侧 Progress 也关掉：进度条由 launch_mini 解析层行来画（stdout 一条），
    #   stderr 那份 tick 每次要算百分比/ETA/strftime —— 纯 host 开销，trace 里还是干扰。
    _pure_mode = ds41_log.pure()
    _prog_on = (prog is not None) and not _pure_mode
    # ★ 显存自动阀（见文件头 _EMPTY_CACHE_* 注释）：非 CP 前向按 token 数决定是否逐层回收。
    #   CP 路径有自己的一套显存开关（RELEASE_Q_ATTN 等），不掺和自动阀。
    _ec_every = _ENV_EMPTY_CACHE_EVERY
    if (_EMPTY_CACHE_AUTO and exec_ops.CP_STATE is None
            and _EMPTY_CACHE_AUTO_T > 0 and int(T) >= _EMPTY_CACHE_AUTO_T):
        _ec_every = 1
    # 有界抢跑：CP 用 K_CP（默认 2，恒开）；非 CP 用 K_NONCP（默认 16），默认也恒开
    # （MIN_T=0），只有显式设阈值时才按 T 过滤——DeepInfer 式有界流水，
    # 用来代替逐层 empty_cache 压住在途缓冲代数。
    if exec_ops.CP_STATE is not None:
        _k_here = _RUNAHEAD_K
        _ru_here = _k_here > 0
    else:
        _k_here = _RUNAHEAD_K_NONCP
        _ru_here = _k_here > 0 and (
            _RUNAHEAD_MIN_T <= 0 or int(T) >= _RUNAHEAD_MIN_T)
    if _DRAIN_PER_FORWARD:
        # B6：每 forward 只在末尾排空一次 → 层内不建环（事件记录省掉，host 直接抢跑整段）。
        _ru_here = False
    # ★ 2026-10-18 用户口径：对性能/显存有利的开关一律默认打开。
    #   MoE 分块（DS41M_MOE_CHUNK=auto → config 的 moe_chunk_max_len，通常 8192）默认开启：
    #   对称缓冲按块容量建，不再按整 T 建（T=16384 单次 ~20 GiB workspace；1M 会顶穿 HBM）。
    #   显式 DS41M_MOE_CHUNK=0 可关（A/B 对照用）。
    # packed：replay 裁尾会改写 `cfg["_packed"]["cu"]`（逐请求行数），但下一个 pass 的输入
    # 又是全长 → 每次前向开始按 lens 重建 cu（lens 不变；无 packed 时 no-op）。
    _pkd0 = _md.packed_view(cfg)
    # ★ 2026-10-22：**n==1 的 packed 几何按本次调用重算**（T/pos_offset 是唯一事实源）。
    #   踩过：serve 的 main() 启动时按 --seq（默认 1024）建 `_packed`，之后每个请求的
    #   prefill/decode 都沿用它 → build_stack 的 T / seq_lens 仍是 1024，而实际只有
    #   prompt 那几十行 → 层 2 的 mxfp8 QMM aicore 507015 / topk gate NaN。
    #   单序列时：rows=T；逻辑长度 = T（prefill）或 pos_offset+T（decode/hit-B/verify）。
    if (_pkd0 and int(_pkd0.get("n", 1) or 1) <= 1
            and pos_offset is not None):
        #   · decode/verify/hit-B：逻辑长度 = pos+T；行数 = T。
        #   · prefill（pos_offset=None）由调用方决定 lens（serve 每请求重置；bench 的
        #     hit stage A 仍沿用已验证的 lens=L）。
        _pkd0["row_lens"] = [int(T)]
        _pkd0["lens"] = [int(pos_offset) + int(T)]
    # ★ 2026-10-21：**n==1 也要重建**。以前限定 n>1，b=1 packed 在回放裁尾后
    #   replay 把 cu 改成 [0,W]，下一个全长 pass 就再也恢复不回来（且裁尾 pass 里
    #   cu 若没改就是 [0,L] 与实际行数不符 → QSLI metadata AICPU 异常）。
    _pk_rows = list(_pkd0.get("row_lens") or _pkd0.get("lens") or [])
    if _pkd0 and _pk_rows:
        _cu0 = [0]
        # ★ 必须用**本 pass 输入行数**（row_lens）重建 cu，不能用逻辑长 lens：
        #   hit 时每序列只喂 R 行（R<L），replay 还会把 cu 改写成每序列 W 行；
        #   用 lens（=L）会把 cu 撑到 N×L 行 → 压缩器/索引按不存在的行数算（实测层 2 行数错）。
        #   plain packed 时 row_lens == lens，与旧行为逐位相同。
        for _v in _pk_rows:
            _cu0.append(_cu0[-1] + int(_v))
        _pkd0["cu"] = _cu0
    if not os.environ.get("DS41M_MOE_CHUNK"):
        os.environ["DS41M_MOE_CHUNK"] = "auto"
    #   decoder-side SWA bounded replay（vLLM#58132 同款；等价回归逐位一致）同样默认开：
    #   长序列尾段只算最后 W 个 token（≥256k 实测省 ~35%）。显式 =0 可关。
    if not os.environ.get("DS41M_DECODER_REPLAY"):
        os.environ["DS41M_DECODER_REPLAY"] = "1"
    stack, meta = assemble.build_stack(cfg)
    # ── DSpark：主模型 aux hidden（target layer 的 attention 输入，层执行前 mean(hc_mult)）──
    _dspark_on = str(os.environ.get("DS41M_DSPARK", "0")).strip().lower() not in (
        "", "0", "off", "false", "no")
    _aux_ids = set(int(v) for v in (
        cfg["model"].get("dspark_target_layer_ids")
        or cfg.get("dspark_target_layer_ids") or [37, 38, 39]))
    _aux = []
    # 诊断（因果泄漏定位）：逐层打印 row0 的 hidden 校验和；fake id 0/5 两次请求
    # 第一个出现差异的层 = 泄漏进入点（eager/无图时用）。
    _row0chk = os.environ.get("DS41M_ROW0_CHECK", "").strip().lower() in (
        "1", "on", "true", "yes") and int(os.environ.get("RANK", "0")) == 0
    if _dspark_on:
        # 必须在 assemble.build_stack 之前关：它按 replay cut 把尾层 T 设成 W，
        # 之后只关 _rep_on 会留下 T=W 的 ctx 与全长 x 不匹配。
        os.environ["DS41M_DECODER_REPLAY"] = "0"
    _ENG = weight_src.engram_placement(cfg)      # (on_device, tp_size, rows_per_rank)
    if n_layers:
        stack = stack[:n_layers]
    recipes = load_recipes()

    exec_cnt, skip_cnt, errs = collections.Counter(), collections.Counter(), []
    last_pre, collected = None, {}
    snap_list = []
    if verbose:
        print(f"  [stats] DS41M_STATS={os.environ.get('DS41M_STATS')!r} "
              f"DS41M_DUMP_OPS={os.environ.get('DS41M_DUMP_OPS')!r}", flush=True)
    implicit_cnt = collections.Counter()
    ph_used, w_sec = set(), 0.0
    # ── engram 预取：engram 层是 config 固定的 [1,14]，其主机取表 ~33 ms/层（单层算只有 ~5 ms），
    #    所以在**前 2 层**就把取表发到后台线程（官方靠设备/DMA 取行才藏得住，我们只能提前发起）。
    try:
        import exec_engram as _eg
        _eng_list = [_s["idx"] for _s in stack if _s.get("engram")]
        _eng_pref = []                                 # [(触发层, 目标 engram 层), ...]
        # ★★ 2026-10-16 深夜：`DS41M_NO_ENGRAM_PREFETCH=1` → **完全关掉 engram 预取**
        #   （厂商路径 = 无侧流 ✓ / 主机路径 = 无后台线程 ✓），engram 层在层内**内联**算
        #   （慢 ~33 ms/层 ✗ 但无任何跨线程/跨流共享状态 ✓）。
        #   动因：主机路径的 run3 出现 `pre_mix 15872 vs 640`（stage A 的 pre_mix 串进 stage B ✗）
        #   + `KeyError: hc_post_t`，而 runner 里每段开头**确实**清了 `_ffn_pre` ✓
        #   ⇒ 清掉之后被**预取机制**（后台线程/侧流）写回 ✗；且 NaN 恒出现在 **L14**，
        #     正是 engram 层 ✓✓ —— 两条线索都指向预取。
        _no_pref = os.environ.get("DS41M_NO_ENGRAM_PREFETCH", "").strip().lower() in ("1","on","true","yes")
        _skip_eng = os.environ.get("DS41M_SKIP_ENGRAM", "").strip().lower() in ("1","on","true","yes")
        if _no_pref or _skip_eng:
            pass                                       # 预取全关（见上；跳 engram 时也不预取）
        else:
            # 主机取表路径沿用"提前 2 层"：再多没用——取表是**主机 CPU** 的活（~33 ms/层），
            #   提前太早会和驱动 NPU 的主线程抢 CPU/GIL（实测提前 6 层反而 0.213→0.405 s）。
            for _l in _eng_list:
                if _l >= 2:
                    _eng_pref.append((_l - 2, _l))
    except Exception:                                  # noqa: BLE001
        _eg, _eng_pref = None, []
    # ★ 2026-10-21：只在**几何变化**时清 META（见 _META_GEO 注释）。
    _pk_geo = _md.packed_view(cfg)
    _geo = (int(T), None if pos_offset is None else int(pos_offset),
            int(_pk_geo.get("n", 1) or 1), tuple(_pk_geo.get("cu") or ()),
            tuple(_pk_geo.get("lens") or ()), int(l_full or 0),
            int(engram_ctx or 0))
    if _META_GEO[0] != _geo:
        exec_ops.meta_clear()
        _META_GEO[0] = _geo
    if _ENV_HOSTTIME:
        exec_ops.HOSTTIME.clear()          # 每轮一张表（首轮含预热/编译，看后面几轮才准）
        exec_ops.PHASE.clear()
    # PA cache 跨层复用（否则每层都要 ZerosLike/Fill，污染 trace）；
    # 命中两段跑时由调用方传入同一个 dict → 前缀缓存跨 stage 存活
    if shared is None:
        shared = {"_win_cache_A": None, "_win_cache_B": None, "_cmp_cache": None}
    # ★★ 每段/每轮开头必须清掉 `_ffn_pre`：它是"**本段内**上一层 ffn 段的 pre_mix"（官方
    #   `hidden_states, pre_mix = _run_decoder_layer(...)` 逐层往下传，modeling_deepseek.py:2311），
    #   **不是**跨段/跨轮的缓存。漏清会出两种事故：
    #     · hit 两段（stage A 真前缀 H=15872 → stage B 尾段 640）：stage B 的层 0/1 拿到 A 段的
    #       pre_mix → `hc_pre` 形状不符（实测量到 `pre_mix.shape[0] … 15872 vs 640` ✗）；不校验的
    #       算子会越界读写 → 设备侧 aicpu 异常/挂死（今晚追的 layer 2 挂死就是这个 ✗）。
    #     · 同形状的多轮 pass：形状对得上、不报错，但 layer 0 的 pre_mix 是**上一轮残留** →
    #       数值悄悄偏（末位 logits 的分布尾部漂移，top8 6/8、|Δscore|≈4 ✗）。
    #   跨段要保的只有 `_win_cache_A/_win_cache_B/_cmp_cache` 那几件真缓存 ✓。
    # ★★ 2026-10-17 00:05 定位：**上面这段注释一直在、清它的语句却从来没落进文件** ✗（23:38 版实测
    #   仍无 ⇒ 两段真前缀（stage A 15872 → stage B 640）**每跑必中**）：
    #     stage B 层 0 第一次 hc_pre 拿到 A 段 15872 行的 `_ffn_pre` → 报
    #     `pre_mix.shape[0] should equal x.shape[0], actual 15872 vs 640` ✗
    #     → 该 op 被 runner 记错并**跳过** ⇒ `hc_pre_t` 没写上 → **同层 ffn 段**的 hc_pre 又找不到
    #       `hc_pre_t`、回落又吃 15872 行的 `_ffn_pre` → 再报一次 ✗ ⇒ `_ffn_pre` **永不刷新** ✗
    #     → 之后 40 层每层第一次 hc_pre 全错 ✗（实测单跑 `⚠️ 出错 409 处` ≈ 40 层 × 10 算子 ✓）
    #     → x 全链没经过 hc 处理、末位 logits 全脏（`取 top1` 0/8 ✗）。
    #   实证对照（同一份日志体系）：单段（无 stage A）的 `winfix/hit8` 从无此错、16/16 打出 ' that' ✓；
    #   两段（`winfix/cmpreset` 30 份、`winfix/noprefetch/run1` 48 处）全中 ✗。
    _stale_pm = shared.pop("_ffn_pre", None)
    if _stale_pm is not None and verbose:
        print(f"  [stage] 清掉上一段/上一轮残留的 _ffn_pre（{int(_stale_pm.shape[0])} 行）", flush=True)
    # ★★ 2026-10-16 深夜（病C 的候选修，**env 开关默认关**）：
    #   定位证据：rank7 的首个 NaN 快照 = L14_kv_compress 写出的 `cmp_kv`/`cmp_latent`
    #   （恰好一行 512 维压缩 KV 全 NaN ✗，而 rank0 全程有限 ⇒ 按 rank/数据出现）。
    #   嫌疑：命中 stage B 用 `start_pos=段基址` 去消费一个**已被 stage A 推进过的
    #   压缩器 ring 状态**（`_cmp_state_L*` 跨段复用 ✗）→ 相位/槽位错位 → 读到 ring 之外。
    #   prefill 从不崩正是因为"没有状态前进后又回到段起点再消费"的场景 ✓。
    #   修（候选）：每段开头把 ring 状态清零 —— 段起点与 ratio 组对齐时数值无害 ✓。
    if os.environ.get("DS41M_CMPSTATE_RESET", "").strip().lower() in ("1", "on", "true", "yes"):
        for _k in [k for k in list(shared) if isinstance(k, str) and k.startswith("_cmp_state_L")]:
            shared.pop(_k, None)
    t0 = time.time()
    _ti0 = time.perf_counter() if _ENV_HOSTTIME else 0.0
    x = make_input(cfg, T) if inputs is None else inputs
    if _ti0:
        exec_ops.PHASE["input"] = exec_ops.PHASE.get("input", 0.0) + (time.perf_counter() - _ti0)
    # ── 命中（hit.ratio>0）──
    # 单段（fake）：只重算 replay 尾，逻辑位置从 L-T_main 起算，前缀 = 池子初值（零、各层共用）；
    # 两段（real）：stage A 真算前缀 [0,H)、stage B 只算尾部；两段共用同一批池子（调用方传 shared）。
    # 池子一律按 L 开 → 尾部的稀疏索引/槽位都用**绝对位置**。
    # （以前这里只改了 ctx["T"] 记账、张量仍是全长 → "命中跑"其实是全量 prefill，2026-10-15 采集时发现。）
    pos_off = 0
    _pos_offs = []          # 逐序列起始位置（packed 且各批命中率不同时非空）
    if pos_offset is not None:
        # ── 增量 decode（单步）：绝对位置显式给出，池/元数据按 l_full（全序列）开 ──
        #   与 prefill 的关系：off==0 时两者逐位相同；这里 off>0 且 T 很小，
        #   所有池的写/读都走绝对坐标（见 _kv_epilog / _kv_compress / _idx_k_cache）。
        pos_off = int(pos_offset)
        if verbose:
            print(f"  [decode-step] 绝对位置 {pos_off}（T={T}，池按 L={l_full or meta.get('L', T)} 开）",
                  flush=True)
    elif meta.get("hit"):
        L = int(meta["L"])
        if hit_stage == "A":
            H = min(int(T), L)
            if x.shape[0] > H:
                x = x[:H].contiguous()
            if input_ids is not None and len(input_ids) > H:
                input_ids = input_ids[:H]
            print(f"  [hit-A] 真算前缀 {H} 个 token（位置 0..{H - 1}；池/缓存按 L={L} 开，"
                  f"压缩 KV 与索引缓存跨层共用）", flush=True)
        else:
            # ★ 2026-10-21（逐序列几何）：packed 各批可不同 L/hit → 起始位置逐序列算；
            #   标量 pos_off 只用于日志/兼容（取最大），真正的逐 token 位置走 ctx["pos_offsets"]。
            _packed_run = bool(_md.packed_view(cfg))
            if _packed_run and meta.get("pos_offs"):
                _pos_offs = [int(v) for v in meta["pos_offs"]]
                pos_off = max(_pos_offs) if _pos_offs else 0
            else:
                _pos_offs = []
                pos_off = max(0, L - int(meta["T_main"]))
            if pos_off > 0:
                if not _packed_run:
                    x = x[-int(meta["T_main"]):].contiguous()
                    if input_ids is not None and len(input_ids) > x.shape[0]:
                        input_ids = input_ids[-x.shape[0]:]      # engram 的 hash 吃尾部 ids（同长度契约）
                # packed：x 已经是 ΣR_i 行（每条序列的 tail），**不要**再切；
                #   每 token 的绝对位置 = 本序列 off_i + 序列内局部位置（batch.py 的 local_positions）。
                tag = "hit-B" if hit_stage == "B" else "hit"
                src = ("前缀 = stage A 的真缓存（共享池）" if hit_stage == "B"
                       else "前缀 = fake 零值池（逐请求块表隔离）")
                _nseq_h = int(_md.packed_view(cfg).get("n", 1) or 1)
                _shape = (f"{_nseq_h} 段 × 各自 R 行（packed，Σ={int(x.shape[0])}）"
                          if _packed_run else f"最后 {x.shape[0]} 个 token")
                if _packed_run and _pos_offs:
                    _rng = ", ".join(f"{o}..{o + int(r) - 1}"
                                     for o, r in zip(_pos_offs, meta.get("T_mains") or []))
                    print(f"  [{tag}] 重算 {_shape}；各序列逻辑位置 {_rng}；{src}", flush=True)
                else:
                    print(f"  [{tag}] h={meta['hit']:.4f}：重算 {_shape}，"
                          f"逻辑位置 {pos_off}..{L - 1}；{src}", flush=True)
    # ── decoder-side SWA bounded replay（对齐官方 ds41_light/adaptlib/replay.py；vLLM#58132 同款）──
    #   全量 prefill 时，**最后一个 kv_source 之后**的层只写滑窗 KV、不写任何压缩/index cache
    #   （24/28/32/36 是 index_source 但 owns_k=False，只读 layer 20 的 cache 重算 topk），
    #   而最终只有 head 取最后一行 → 更早的 token 在 tail 层全是白算。
    #   → cut 之后的层只跑最后 W 个 token（逻辑位置 L-W..L-1，mini 的池/缓存本来就按绝对位置开 ✓）。
    #   开关：DS41M_DECODER_REPLAY=1/on/true（静态）或 =<json 路径>（热切，字段 {on,w,cut,force}
    #   与官方 DS41L_DECODER_REPLAY 完全一致）；w=0→sliding_window，cut=0→最后一个 kv_source+1。
    #   等价回归：w >= 序列长时**不裁**（走原路）→ 应与关掉开关逐位一致。
    #   ★ 对 **hit 口径也生效**：官方 hit 的实测段（stage B）本身"就是一段短 prefill"，
    #     replay 照常裁它的尾层（stage A/前缀段不测 → 裁不裁都一样，但走了同一条码路）。
    #   位置口径：裁剪只把 token 数从 T 降到 W，**逻辑位置继续用绝对坐标** → `pos_off` 前移
    #     `(裁掉的 token 数)` 即可（窗口/rope/槽位/压缩 KV 的 index 全按绝对位置算 ✓）。
    _def_cut, _def_w = assemble.replay_cut_w(cfg)      # 默认几何统一在 assemble（值不变）
    _rep = replay.ReplayPolicy.from_env(cfg, _def_cut, _def_w)
    # ★ DeepInfer KEEP_TOKENS 对齐：显式 JSON 的 w 若小于 KEEP_TOKENS，拉回 keep。
    if _rep.on and _rep.w < _def_w:
        print(f"  [replay] w={_rep.w} < keep={_def_w}（DeepInfer KEEP_TOKENS）"
              f" → 按 keep 执行", flush=True)
        _rep = replay.ReplayPolicy(on=True, w=_def_w, cut=_rep.cut,
                                   force=_rep.force, source=_rep.source)
    _rep_W, _rep_cut, _rep_on = _rep.w, _rep.cut, _rep.on
    # ★ 2026-10-22（层 20 边界 / DeepInfer）：compact 段的**形状对齐宽度**。
    #   `npu_quant_matmul` 在 M≈140 时选到与 full-T 不同的分块（实测层 20：
    #   wq_a/wkv/shared w1/w3 与全长尾部逐位不同；M=512 时 wq_a/w1/w3/wo_b 与
    #   full-T（16384/652）逐位一致）。层 20 的 compact 段只此一层，补到 512 行
    #   （真实行，来自 cut 前的全长 x）仍远小于全长，换取"拆分前后逐位相同"。
    #   层 21 入口的 replay 再把它裁到 `_rep_W`=140。`DS41M_BOUNDARY_PAD=0` 关闭补行
    #   （=官方 140 行口径，QMM 走小 M 分块，数值与全长基线有 ~1e-4 级差异）。
    _split_pad = int(os.environ.get("DS41M_BOUNDARY_PAD", "512") or 0)
    if _dspark_on:
        # DSpark 需要 layers 37..39 的**全前缀** aux；decoder replay 会把尾段裁成 W 行，
        # 导致 aux 只有最后 W 行 → 草稿上下文 KV 不完整。v1 直接关 replay。
        _rep_on, _rep_W = False, 0
    if _rep_on:
        _line = _rep.describe(len(stack), int(x.shape[0]))
        if _line and not _pure_mode:          # ★ 采集窗口内零日志（2026-10-27 用户规则）
            print(_line, flush=True)
    _rep_done = False
    # 逐层摘要（序列隔离排查）：`DS41M_PACKED_DIGEST_LAYERS=0,1,2` → 打印这些层**入口处**
    # 每条序列末行 hidden 的 sha1。层 0 若有差异 = 真泄漏；只在末层差异 = 数值混沌/求和顺序。
    _pl_dig = {int(v) for v in os.environ.get("DS41M_PACKED_DIGEST_LAYERS", "")
               .replace(",", " ").split() if v.strip()}
    for n, layer in enumerate(stack):
        if _pl_dig and int(layer["idx"]) in _pl_dig and _RANK == 0:
            try:
                _pkd_l = _md.packed_view(cfg)
                _nl = int(_pkd_l.get("n", 1) or 1)
                _cul = [int(v) for v in (_pkd_l.get("cu") or [])]
                import hashlib as _hl
                _ds = []
                # ★ CP：本地行是 head-tail 两块，非整除 L 时 **末行是 pad**——序列末
                #   token 要用 cp_state.last_local_rows 定位（否则摘要对的是垃圾行）。
                _cps_d = getattr(exec_ops, "CP_STATE", None)
                _clr = (list(_cps_d.last_local_rows)
                        if _cps_d is not None and len(_cps_d.last_local_rows) == _nl else None)
                if _nl > 1 and len(_cul) >= _nl + 1:
                    for _i in range(_nl):
                        _a, _b = _cul[_i], _cul[_i + 1]
                        if _b <= _a:
                            continue
                        _rr = x[_clr[_i] if _clr is not None else _b - 1].detach().float().cpu()
                        _ds.append(f"seq{_i}[{_a}:{_b}] "
                                   f"sha1={_hl.sha1(_rr.numpy().tobytes()).hexdigest()[:16]} "
                                   f"sum={float(_rr.sum()):+.6e} absmax={float(_rr.abs().max()):.4g}")
                else:
                    _rr = x[_clr[0] if _clr is not None else -1].detach().float().cpu()
                    _ds.append(f"single[{int(x.shape[0])}] "
                               f"sha1={_hl.sha1(_rr.numpy().tobytes()).hexdigest()[:16]} "
                               f"sum={float(_rr.sum()):+.6e} absmax={float(_rr.abs().max()):.4g}")
                print(f"  [packed-digest] L{int(layer['idx'])} " + " | ".join(_ds), flush=True)
            except Exception as _de:                   # noqa: BLE001
                print(f"  [packed-digest] L{int(layer['idx'])} 跳过（{type(_de).__name__}: {_de}）",
                      flush=True)
        if _row0chk and 1 < int(T) <= 32:
            try:
                print(f"  [row0-chk] L{int(layer['idx'])} "
                      f"sum={float(x[0].float().sum()):.6f}", flush=True)
            except Exception:                          # noqa: BLE001
                pass
        # DSpark aux：在**执行本层之前**抓（= 该 target layer 的 attention 输入）
        if _dspark_on and int(layer["idx"]) in _aux_ids:
            _aux.append(x.mean(dim=1) if x.dim() == 3 else x)
        if _rep_on and not _rep_done and int(layer["idx"]) >= _rep_cut:
            _rep_done = True
            _cp_rep = exec_ops.CP_STATE
            if _cp_rep is not None:
                # ★ CP：按**全局 compact 长度**裁（本地只有 mirror 行），跨 rank 重分布
                _trc = _cp_replay_trim(x, input_ids, shared, cfg, _cp_rep, int(_rep_W))
                if _trc is not None:
                    x, input_ids = _trc["x"], _trc["input_ids"]
                    pos_off = _trc["pos_off"]
                    _pos_offs = []
                    print(_trc["message"], flush=True)
                _tr = None                      # CP 不走下方 replay.apply 的图/记账分支
            else:
                _tr = replay.apply(_rep, x=x, input_ids=input_ids, shared=shared,
                                   layer_idx=int(layer["idx"]), pos_off=pos_off,
                                   packed_ctx=(_md.packed_view(cfg) or None),
                                   pos_offs=_pos_offs)
            if _tr is not None and _tr.applied:
                x, input_ids, pos_off = _tr.x, _tr.input_ids, _tr.pos_off
                _pos_offs = list(_tr.pos_offs or [])
                if not _pure_mode:            # ★ 采集窗口内零日志（2026-10-27 用户规则）
                    print(_tr.message, flush=True)
        ctx = _CTX_CACHE.get(layer["idx"]) or ctx_for(cfg, layer)   # 层内常量：按层缓存
        if _PROBE_ON:                                  # Phase 1.5：读写探针（默认关）
            ctx = _probe.wrap_ctx(ctx)
        _CTX_CACHE[layer["idx"]] = ctx
        ctx["_src"] = source      # decode MoE 的 NZ 缓存用它释放原始专家权重（内存中性）
        # ★ CP：本地 head-tail 行的全局位置（rope / 窗口索引 / 池槽位用）。
        #   每层都刷新；CP=1 时清掉，避免上一个 CP forward 的缓存串到普通路径。
        _cp_here = getattr(exec_ops, "CP_STATE", None)
        if _cp_here is not None:
            # ★ compact 段（层 20 截尾后）：`seq_len` 是**原序列长**（rope/压缩/ref-M
            #   都按它），本地位置是 compact 局部 0..W-1；绝对位置 = 局部 + pos_offset。
            _cp_seqlen = int(getattr(_cp_here, "seq_len", 0) or _cp_here.plan.total_q)
            ctx["cp_positions"] = _cp_here.local_positions
            ctx["cp_abs_positions"] = _cp_here.abs_positions
            ctx["cp_total_q"] = _cp_seqlen
            # hc_pre 对 M 分块敏感：本地行必须按**当前计划的全局行数**补零再取尾部，
            # 否则与 CP1 同段基线不在同一个 M 等价类（classes.log：≤512/652/1024/
            # 2048/4096/≥6144）。超过 8192 时补到 8192 —— 尾部行在 ≥6144 类里与任意
            # 更大参考 M 逐位一致（hc_ab/tile2.log），且不必为 1M 开 total_q 行缓冲。
            # ★ compact 段（层 21+ 裁到 W 后）用 plan.total_q=W，与 CP1 的 M=W 一致；
            #   层 20 的 512 compact 由 `_boundary_cut_state(_cp)` 自己设 hc_pad_ref=全长 T。
            _hc_ref = min(8192, int(_cp_here.plan.total_q))
            ctx["hc_pad_ref"] = _hc_ref
            ctx["hc_pad_cap"] = _hc_ref
        else:
            ctx.pop("cp_positions", None)
            ctx.pop("cp_abs_positions", None)
            ctx.pop("cp_total_q", None)
            ctx.pop("hc_pad_ref", None)
            ctx.pop("hc_pad_cap", None)
        # ★ 2026-10-21 per-DP dummy span：本层 MoE 的 span 数目标（全 DP 最大）。
        #   0 = 不启用（普通多点/单跑）；只有 per-DP 且各 DP span 不同才非 0。
        _pk_nsp = _md.packed_view(cfg)
        _nsp_pre_t = int(_pk_nsp.get("nspan_pre") or 0)
        _nsp_here = 0
        if _nsp_pre_t:
            _t_pre_t = sum(int(x) for x in (_pk_nsp.get("row_lens") or []))
            _nsp_here = (_nsp_pre_t if int(layer["T"]) >= _t_pre_t
                         else int(_pk_nsp.get("nspan_post") or _nsp_pre_t))
        ctx.update(hc_mult=cfg["model"].get("hc_mult", 4),
                   hc_sinkhorn_iters=cfg["model"].get("hc_sinkhorn_iters", 20),
                   hc_eps=cfg["model"].get("hc_eps", 1e-6),
                   rope_dim=cfg["model"].get("qk_rope_head_dim", 64),
                   topk=cfg["model"]["topk"],
                   routed_scaling_factor=cfg["model"].get("routed_scaling_factor", 1.5),
                   # SwiGLU 截断（官方 config.swiglu_limit=10.0 → mega_moe 的 activation_clamp）
                   swiglu_limit=cfg["model"].get("swiglu_limit", 10.0),
                   eps=cfg["model"].get("rms_norm_eps", 1e-20),
                   # engram 的 n-gram hash 必须吃**真实输入 id**（缺失就退化成全零 → 取表全错）
                   input_ids=input_ids,
                   # 命中：本层 token 的**绝对起始位置**与全序列长度（池子/槽位/rope 都用绝对位置）
                   pos_offset=pos_off, L_full=int(l_full or meta.get("L", T)),
                   # ★ 2026-10-21 逐序列几何：packed 各批不同 L/hit 时，逐序列起始位置/行数
                   pos_offsets=(list(_pos_offs) if _pos_offs else None),
                   row_lens=(list(meta.get("T_mains") or []) if (_md.packed_view(cfg) or None) else None),
                   # decode 步标记（pos_offset>0）→ MoE 选"官方 decode 口径"而不是 MegaMoE
                   is_decode_step=bool(pos_offset is not None and int(pos_offset) > 0),
                   # ★ 奇数 T 拆分的最后 1 token：形状是 decode，但**仍属于 prefill 阶段**。
                   #   tp8 的 P→D 需要 EP raw；若这里走 EP decode 口径，ensure_decode_gmm
                   #   会建 EP NZ 并 drop 该层 raw → 紧随其后的 P→D 报"需要 EP raw"。
                   #   置真时 MoE 仍走 prefill 的 ascendc 融合路径。
                   split_tail=bool(split_tail),
                   # 增量 decode：input_ids 比 T 多带几个"上文 id" 给 engram 的 n-gram hash
                   engram_ctx=int(engram_ctx or 0),
                   # ★ 命中两段：重放段长度 → 压缩行写入抑制用
                   #   （官方 hit.py：**只对 hit replay 行**的 compressed-KV/indexer 写 slot=-1）
                   # ★★ 2026-10-20 根因修复：以前只要 `pos_off > 0` 就设 128 →
                   #   普通增量 decode 的每一步都命中 `_kv_compress` 的抑制分支
                   #   （slot=-1），压缩 KV 永远不写；第一组完成时读到 prefill 的
                   #   carry/部分行 → gate NaN（rank2–5 vector core exception）。
                   #   hit 抑制只在 hit replay 段（fake hit 或 stage B）生效；普通 decode
                   #   的 `meta["hit"]` 为 false（mini_real: hit.ratio=0）→ 不抑制。
                   hit_replay_tail=(assemble.replay_tail_of(cfg)
                                    if (pos_off > 0 and (hit_stage == "B" or meta.get("hit")))
                                    else 0),
                   # engram 分片口径（与 weights.engram_placement 一致；表在设备上时用）
                   engram_tp_size=_ENG[1], engram_tp_rank=_RANK, engram_rows=_ENG[2] * _ENG[1],
                   # packed 真 batch：每 token 属于哪条序列 / 各序列长度（单序列时 n_seqs=1）
                   n_seqs=int(_md.packed_view(cfg).get("n", 1) or 1),
                   cu_seqlens=_md.packed_view(cfg).get("cu"),
                   seq_lens=_md.packed_view(cfg).get("lens"),
                   moe_nspan=_nsp_here)
        if (bool(_ENV_DUMP_OPS) and layer["idx"] < _ENV_DUMP_LAYERS_CAP
                and layer["idx"] >= _ENV_DUMP_FROM):
            try:
                d = _ENV_DUMP_OPS
                os.makedirs(d, exist_ok=True)
                torch.save({"x_in": (x.mean(dim=1) if x.dim() == 3 else x).detach().float().cpu()},
                           os.path.join(d, f"rank{_RANK}_L{layer['idx']}_x_in.pt"))
            except Exception:                              # noqa: BLE001
                pass
        if collect_dir:                     # 与官方 DS41L_COLLECT_LAYERS 同一表达式/同一挂点
            h = x
            # DS41M_COLLECT_LAYERS="0,1,2,14,39"：只落指定层（packed 对拍时 40 层 × 4096 行
            # 的 float32 dump ≈ 3.4 GB/份，按需裁剪）
            _cl = os.environ.get("DS41M_COLLECT_LAYERS", "").strip()
            if not _cl or layer["idx"] in {int(t) for t in _cl.split(",") if t.strip()}:
                collected[layer["idx"]] = (h.mean(dim=1) if h.dim() == 3 else h).detach().float().cpu()
        if _prog_on:
            prog.stage("S1 逐层：权重装载")
        if _ENV_HOSTTIME:
            _ph0 = time.perf_counter()
            _phA = _ph0
        # ★ 2026-10-21 pure：w_sec 只喂进度条（prog），没有 prog 时**不取墙钟**
        #   （旧版每层 2 次 time.time()，40 层/pass 在 bench 里纯属多余 host 调用）
        t_w = time.time() if _prog_on else 0.0
        # ★ 每层都要把 ~2300 个权重键扫两遍（strip_prefix 建新 dict + pack 循环）≈0.75 ms/层。
        #   `share_across_layers` 下各层拿到的是**同一批张量**，这两步的结果按层号缓存即可
        #   （非 share 模式不缓存：真权重逐层不同）。
        _share_w = bool((cfg.get("weights") or {}).get("share_across_layers", False))
        _li = layer["idx"]
        _mem_here = bool(_ENV_MEMLOG and _ENV_MEMLOG_FROM <= int(_li) <= _ENV_MEMLOG_TO)
        if _mem_here:
            _memlog_mod.mem(f"L{_li} pre T={layer['T']}", rank=_RANK)
        if _share_w and _li in _PACKED_LAYER_CACHE:
            w = _PACKED_LAYER_CACHE[_li]
        else:
            w = source.iter_layer(_li, cfg["model"]["n_routed_experts"])
            _phB = time.perf_counter() if _ENV_HOSTTIME else 0.0
            if _share_w and _li in _STRIP_CACHE:
                w = _STRIP_CACHE[_li]
            else:
                w = strip_prefix(w, _li)
                if _share_w:
                    _STRIP_CACHE[_li] = w
            w = pack_layer_weights(w, cfg, layer_idx=_li)
            if _ENV_HOSTTIME:                   # 细分：取权重 vs strip+布局打包
                exec_ops.PHASE["pack"] = exec_ops.PHASE.get("pack", 0.0) + (
                    time.perf_counter() - _phB)
            if not _share_w:                    # ★ 打包完 → raw 释放（见 _release_packed_raw）
                _release_packed_raw(source, w, _li)
            if _share_w:
                _PACKED_LAYER_CACHE[_li] = w
        if _ENV_HOSTTIME:
            exec_ops.PHASE["w_load"] = exec_ops.PHASE.get("w_load", 0.0) + (
                time.perf_counter() - _phA)
        if _prog_on:
            w_sec += time.time() - t_w
            prog.stage("S2 逐层：前向执行")
            prog.tick((n + 1) / len(stack),
                      f"层 {layer['idx']}/{len(stack)-1} {layer['attn']} · 权重累计 {w_sec:.1f}s")
        _res0 = x if _ENV_NO_RESIDUAL_CLONE else x.clone()
        st = {"x": x, "residual": _res0, "_w": w, "_shared": shared,
              "_backends": (cfg.get("backends") or {}),
              "_allow_placeholder": allow_placeholder, "_placeholder_used": ph_used}
        if _PROBE_ON:                                  # Phase 1.5：st 读写探针
            st = _probe.wrap_st(st)
        del _res0        # 不能留局部引用：否则 clone 的寿命被延长到下一层（多 4 GiB）
        for _trig, _tgt in _eng_pref:              # 到触发层：提前把 engram 层的活发出去
            if layer["idx"] != _trig:
                continue
            try:
                _lay2 = next(x for x in stack if x["idx"] == _tgt)
                _w2 = strip_prefix(source.iter_layer(_tgt, cfg["model"]["n_routed_experts"]), _tgt)
                # ★ 无论 share 与否都必须 pack（写 `_packed`）：真权重（share=false）以前跳过这一步
                #   → 侧流里的 `_engram_wkv`（core_qmm 读 `w["_packed"]`）直接 KeyError('_packed')
                #   → 侧流半路崩、`pending` 从未写入 → 主干把整条链又跑一遍（trace 里 EngramFetch 双份）。
                if _share_w:                       # 打包结果按层缓存（和主循环同一份，别打两遍）
                    _w2 = _PACKED_LAYER_CACHE.get(_tgt) or pack_layer_weights(_w2, cfg)
                    _PACKED_LAYER_CACHE[_tgt] = _w2
                else:
                    # ★ 2026-10-19：以前这里**没传 layer_idx** → `_PACK_CACHE_L` 不生效
                    #   （ck=None 直接跳过缓存）→ engram 层的 QMM 权重**每 pass 重新
                    #   transpose + NZ cast**（层 1/14，每层 ~120 MB 的 TransData）。
                    #   传 layer_idx 后与主循环用同一个按层缓存（打包结果本来就常驻）。
                    _w2 = pack_layer_weights(_w2, cfg, layer_idx=_tgt)
                    _release_packed_raw(source, _w2, _tgt)   # 侧流也走同一条 raw 释放
                _c2 = dict(ctx_for(cfg, _lay2))
                if _PROBE_ON:                          # 侧流 ctx 也记录（线程局部 op 名）
                    _c2 = _probe.wrap_ctx(_c2)
                _c2.update(_engram_T=int(layer["T"]),
                           hc_mult=cfg["model"].get("hc_mult", 4),          # wkv/gate 要用
                           eps=cfg["model"].get("rms_norm_eps", 1e-20),
                           input_ids=input_ids,                             # 真 ids 时 ngram hash 要吃
                           engram_tp_size=_ENG[1], engram_tp_rank=_RANK,
                           engram_rows=_ENG[2] * _ENG[1])
                _eg.prefetch_layer(_tgt, _c2, _w2, input_ids)      # host 取表：后台线程预取
            except Exception as _e2:                           # noqa: BLE001
                # ★ 必须**总是**可见：侧流半路失败会让主干把整条链重跑一遍 → trace 里
                #   EngramFetch 双份、侧流白占设备（曾经静默发生，靠 DS41M_DEBUG 才挖出来）。
                print(f"      ⚠️ [engram] 侧流发起失败（层 {_tgt}）：{type(_e2).__name__}: {_e2}"
                      f" —— 本层将退回单流重算（trace 里该层会有两份 EngramFetch！）", flush=True)
        if os.environ.get("DS41M_HID_SHA") and int(layer["idx"]) <= int(
                os.environ.get("DS41M_HID_SHA_MAX", "3")):
            # 诊断：层入口 hidden 指纹（判定 CP1 vs CP2 的 hidden 从哪一层开始分叉）
            try:
                import hashlib as _hh
                _xv = x.reshape(-1, x.shape[-1])
                _hs = _hh.sha1(_xv.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]
                print(f"  [hid-sha] L{int(layer['idx'])} rows={int(_xv.shape[0])} sha={_hs}",
                      flush=True)
            except Exception as _he:                  # noqa: BLE001
                print(f"  [hid-sha] 失败 {type(_he).__name__}: {_he}", flush=True)
        # ★ DeepInfer 层 20 边界拆分（PERF_RULES §2；默认开，`=0` 关闭）：
        #   KV 全长算完 → _boundary_cut（截 512）→ q/indexer/attn/MoE 只跑 512 行；
        #   compact 段 + 21..39 全程 eager（2026-10-26 起不再入图）。
        #   ★ CP：本地只有 mirror 行，"还有多少可裁"要按**全局序列长**判断。
        _cp_split = exec_ops.CP_STATE
        _t_glob = (int(_cp_split.plan.total_q) if _cp_split is not None
                   else int(x.shape[0]))
        _split_here = (bool(os.environ.get("DS41M_BOUNDARY_SPLIT", "1").strip().lower()
                            in ("1", "on", "true", "yes"))
                       and _rep_on and not _rep_done
                       and int(layer["idx"]) == _rep_cut - 1
                       and str(layer.get("attn")) == "c1a_source"
                       and int(ctx.get("n_seqs", 1) or 1) == 1
                       and _rep_W < _t_glob)
        if _split_here:
            _seq_key = (int(layer["idx"]), "split")
            _legacy_seq = _SEQ_CACHE.get(_seq_key) or expand_layer(
                cfg, recipes, dict(layer, attn="c1a_source_split"))
            _SEQ_CACHE[_seq_key] = _legacy_seq
        else:
            _seq_key = layer["idx"]
            _legacy_seq = _SEQ_CACHE.get(_seq_key) or expand_layer(cfg, recipes, layer)
            _SEQ_CACHE[_seq_key] = _legacy_seq
        seq = _legacy_seq
        # ★ Phase 2（默认关）：执行序改由模块树给出；首次与 legacy 严格对拍。
        if _module_runtime_here(ctx, hit_stage, bool(meta.get("hit"))):
            _plan = _MODULE_PLAN_CACHE.get(_seq_key)
            if _plan is None:
                _plan = _mod_runtime.build_plan(
                    cfg, recipes, layer, "split" if _split_here else "default")
                try:
                    _mod_runtime.verify_order(_plan, _legacy_seq)
                    _MODULE_PLAN_CACHE[_seq_key] = _plan
                    print(f"  [module-runtime] L{int(layer['idx'])} "
                          f"view={_plan.view} ops={len(_plan.ops)} order=legacy-ok",
                          flush=True)
                except _mod_runtime.ModuleRuntimeOrderError as _mre:
                    if not _ENV_MODULE_RUNTIME_FALLBACK:
                        raise
                    print(f"  ⚠️ [module-runtime] 顺序不一致 → 回退 legacy：{_mre}",
                          flush=True)
                    _MODULE_PLAN_CACHE[_seq_key] = False
                    _plan = False
            if _plan:
                seq = _mod_runtime.ops_of(_plan)
        if _ENV_HOSTTIME:
            _ph1 = time.perf_counter()
            exec_ops.PHASE["pre"] = exec_ops.PHASE.get("pre", 0.0) + (_ph1 - _ph0)
        hc_seen = 0
        # ★★ 2026-10-27 用户规则：**逐层进度行已永久删除**（`[ 12/40] layer … 算子 N`）。
        #   这是早期调试日志：每层一次字符串拼接 + print/flush + 8 rank × NFS 写盘，
        #   在 host-bound 点位直接计入 forward 时间。性能数只由 points_runner 的
        #   汇总行（"多轮耗时/纯 forward 吞吐"）给出；要诊断请用 trace（ALOP）+ hosttime。
        # 本层诊断开关一次算清（全在模块导入时定死；默认全 False，热循环里只剩布尔判断）
        _li_now = layer["idx"]
        _do_stats = _ENV_STATS and _li_now < 2
        _do_dump_ops = (bool(_ENV_DUMP_OPS) and _li_now < _ENV_DUMP_LAYERS_CAP
                        and _li_now >= _ENV_DUMP_FROM
                        and (_ENV_DUMP_OPS_ALL or _RANK in _ENV_DUMP_OPS_RANKS))
        _do_dump_slice = (bool(_ENV_DUMP_SLICE) and _li_now < _ENV_DUMP_SLICE_MAX
                          and _li_now >= _ENV_DUMP_FROM
                          and (_ENV_DUMP_SLICE_ALL or _RANK in _ENV_DUMP_SLICE_RANKS))
        _do_digest = _ENV_OP_DIGEST and _RANK == 0 and _li_now in _ENV_OP_DIGEST_LAYERS
        # CP 压缩层：rmsnorm(attn) 之后预取 kv_compress 的 hidden gather（见 helper）
        _has_kvc = "kv_compress" in seq
        for key in seq:
            # ★ Phase 1.5 归因口径：只在**算子实现体内**挂当前算子名。
            #   旧版在循环头就 set_current(key)，导致算子前后的诊断代码（stats/dump/digest）
            #   对 st/ctx 的读取被算到该算子头上 → 报告里出现大量假“读漏”
            #   （如 hc_pre 读 moe.out）。现在 run_op / boundary_cut 各自局部挂/摘。
            if key in ("hc_pre", "rmsnorm"):
                # hc_pre → rmsnorm → 计算 → hc_post 是一个"阶段"；attn 段先，ffn 段后
                if key == "hc_pre":
                    st["_norm_stage"] = "attn" if hc_seen == 0 else "ffn"
                    hc_seen += 1
            if _do_stats:
                try:
                    v = st.get("x")
                    if isinstance(v, torch.Tensor):
                        f = v.float()
                        snap_list.append((layer["idx"], key, tuple(v.shape),
                                          float(f.mean()), float(f.abs().max())))
                except Exception as _e:                    # noqa: BLE001
                    if _ENV_DEBUG:
                        print(f"      [dump-slice] 落盘 {key} 失败: {type(_e).__name__}: {_e}",
                              flush=True)
            if _do_dump_ops:
                # 逐算子落盘关键张量（EP 自洽排查用）：只存 [0,1] 层
                # ★ 2026-10-16 扩表：原来只落 topk_idx/weight/logits → 命中口径的层 0
                #   首错只能定位到"门控输入之前"；补齐注意力侧整链（q/kv/attn_out/hc_* 等）。
                #   超大张量按"保 token 轴的行均值"压缩（dim0 保持行数 → 尾段切片对照仍成立）。
                try:
                    snap = {}
                    # ★ DS41M_DUMP_TAIL=N：二维张量只保**尾 N 行**、不再做行均值降维
                    #   （命中口径的对照都在尾段；均值降维会把"只前 127 行错"这种
                    #    结构性差异抹平，实测误导过一次）。默认 0 = 老行为（不变 ✓）。
                    _tail = int(os.environ.get("DS41M_DUMP_TAIL", "0") or 0)
                    need = ("x", "residual", "qr", "q", "kv", "cmp_kv", "cmp_latent",
                            "attn_out", "o_lora", "hc_pre_t", "hc_post_t",
                            "topk_idx", "topk_weight", "logits")
                    for k in need:
                        v = st.get(k)
                        if v is None and isinstance(st.get("_shared"), dict):
                            v = st["_shared"].get(k)      # 池类张量在 _shared 里（跨层复用）
                        if not isinstance(v, torch.Tensor):
                            continue
                        t = v.detach().float()
                        if _tail > 0 and t.dim() >= 2 and t.shape[0] >= _tail:
                            # 全保真：只取尾 _tail 行，**不做**行均值降维
                            # （行均值是"跨 4 万个元素取平均"的投影，会把逐元素大误差抹平
                            #   —— 实测被误导过一次：行均值只差 7.8e-4，下游却差 0.38）
                            t = t[-_tail:].contiguous()
                        elif _tail <= 0 and t.numel() > 4_000_000 and t.dim() >= 2:
                            t = t.reshape(t.shape[0], -1).mean(dim=1)
                        if t.numel() <= 40_000_000:
                            snap[k] = t.cpu()
                    snap["_stage"] = st.get("_norm_stage")      # hc 段（attn/ffn）——口径自证
                    snap["_rows"] = int(x.shape[0])             # 本段实际行数（stage A/B 自证）
                    if snap:
                        d = _ENV_DUMP_OPS
                        os.makedirs(d, exist_ok=True)
                        torch.save(snap, os.path.join(d, f"rank{_RANK}_L{layer['idx']}_{key}.pt"))
                except Exception:                          # noqa: BLE001
                    pass
            if key == "boundary_cut":
                if _PROBE_ON:
                    _probe.set_current(key)            # 内联钩子：只算 boundary_cut 自己
                if not _split_here:
                    raise RuntimeError("边界切分算子 boundary_cut 出现在非边界层/未开 DS41M_BOUNDARY_SPLIT")
                # CP：本地是 mirror 行，"裁到多少"按**全局序列长**算（compact 宽度是
                # 每条请求保留的 token 数），再按 compact-mirror 重新切到各 rank。
                _t_before = int(_cp_split.plan.total_q) if _cp_split is not None \
                    else int(st["x"].shape[0])
                # compact 段宽度：补到 _split_pad（默认 512）以避开 QMM 小 M 分块；
                # 本段行数不足时保持原样（等于不裁）。
                _w_cut = min(max(int(_rep_W), int(_split_pad)), _t_before)
                # 诊断（DS41M_QMM_AB=1）：同一份 x/权重，全 T vs 尾部 M 的 QMM 是否逐位一致。
                if os.environ.get("DS41M_QMM_AB", "").strip().lower() in (
                        "1", "on", "true", "yes"):
                    _m_ab = int(os.environ.get("DS41M_QMM_AB_M", _w_cut) or _w_cut)
                    _m_ab = min(max(_m_ab, 1), _t_before)
                    _xf = st["x"]
                    _xc = _xf[-_m_ab:].contiguous()
                    _xq_f, _xs_f = torch_npu.npu_dynamic_mx_quant(
                        _xf, dst_type=torch.float8_e4m3fn)
                    _xq_c, _xs_c = torch_npu.npu_dynamic_mx_quant(
                        _xc, dst_type=torch.float8_e4m3fn)
                    print(f"  [qmm-ab] L{int(layer['idx'])} M={_m_ab} dynquant xq_equal="
                          f"{torch.equal(_xq_f[-_m_ab:].view(torch.uint8), _xq_c.view(torch.uint8))}"
                          f" xs_equal={torch.equal(_xs_f[-_m_ab:], _xs_c)}", flush=True)
                    for _nm in ("attn.wq_a.weight", "ffn.shared_experts.w1.weight"):
                        if _nm not in (w.get("_packed") or {}):
                            continue
                        _yf = exec_ops._qmm(_xf, _nm, w, ctx)
                        _yc = exec_ops._qmm(_xc, _nm, w, ctx)
                        _d = (_yf[-_m_ab:].float() - _yc.float()).abs()
                        print(f"  [qmm-ab] {_nm.split('.')[-2]} full={tuple(_yf.shape)} "
                              f"M={_m_ab} equal={torch.equal(_yf[-_m_ab:], _yc)} "
                              f"max={float(_d.max()):.6g} "
                              f"neq={int((_yf[-_m_ab:] != _yc).sum())}/{_yc.numel()}",
                              flush=True)
                        del _yf, _yc
                if _cp_split is not None and hasattr(_cp_split, "with_compact"):
                    ctx, pos_off, _mvd = _boundary_cut_state_cp(
                        st, ctx, _w_cut, pos_off, cfg)
                else:
                    ctx, pos_off, _mvd = _boundary_cut_state(st, ctx, _w_cut, pos_off)
                _pos_offs = [pos_off]
                # ★ 不设 `_rep_done`：层 21 入口的 replay 还要把本段裁到 `_rep_W`=140
                #   （pos_offset 已前移 T-_w_cut，replay 会用 512→140 的 drop 再前移）。
                if os.environ.get("DS41M_SPLIT_DUMP", "").strip().lower() in (
                        "1", "on", "true", "yes") and _RANK == 0:
                    _rows = []
                    for _k, _v in st.items():
                        if torch.is_tensor(_v):
                            _rows.append(f"{_k}={tuple(_v.shape)}:"
                                         f"{str(_v.dtype).replace('torch.', '')}")
                        elif isinstance(_v, tuple) and _v and torch.is_tensor(_v[0]):
                            _rows.append(f"{_k}=tuple[{tuple(_v[0].shape)}]")
                    print("  [split-dump] st: " + " ".join(_rows), flush=True)
                    _sh = (st.get("_shared") if isinstance(st.get("_shared"), dict)
                           else {})
                    _rows = [f"{_k}={tuple(_v.shape)}" for _k, _v in _sh.items()
                             if torch.is_tensor(_v)]
                    print("  [split-dump] shared: " + " ".join(_rows), flush=True)
                print(f"  [boundary-cut] L{int(layer['idx'])}：KV 全长 {_t_before} 行 → "
                      f"compact {_w_cut} 行（pool rebase {_mvd} 段，pos_offset {pos_off}）；"
                      f"层 {int(layer['idx']) + 1} 入口 replay 再裁到 {_rep_W} 行",
                      flush=True)
                if _PROBE_ON:
                    _probe.set_current(None)
                continue
            if _ENV_MOE_ONLY and key != "mega_moe":
                continue                       # 排查用：只发 mega_moe（其余全跳过）
            if key in exec_ops.IMPLICIT:          # 由别的算子内部产生，不单独发
                implicit_cnt[key] += 1
                continue
            try:
                if _ENV_DEBUG:
                    print(f"      → {key}", flush=True)
                # DS41M_HOSTTIME=1：逐算子 host 墙钟计时（异步下发时是"下发耗时"，不是设备耗时）
                _ht0 = time.perf_counter() if _ENV_HOSTTIME else 0.0
                if _ENV_SYNC_BEFORE:
                    torch.npu.synchronize()      # 队列清空后再计时 → 得到"纯 host 成本"
                    _ht0 = time.perf_counter()
                if _ENV_MEMLOG_OPS and _mem_here:
                    _memlog_mod.mem(f"L{_li} op={key} pre", rank=_RANK)
                # Phase 2：模块参数来自 modules/specs/*.yaml（运行配置优先），
                #   执行前临时注入、执行后撤掉，不污染兄弟模块。
                _inj = _mod_runtime.inject_params(key, ctx)
                try:
                    # Phase 1.5：探针作用域在 exec_ops.run_op 内部（只包实现体）
                    ok = exec_ops.run_op(key, st, ctx, w)
                finally:
                    for _ik in _inj:
                        ctx.pop(_ik, None)
                # ★ 侧流预取：attn 段 rmsnorm 之后 hidden 定稿，到 kv_compress 之间只读
                if (_has_kvc and key == "rmsnorm"
                        and st.get("_norm_stage") == "attn"):
                    _cp_maybe_prefetch_cmp(st, ctx)
                if _ENV_MEMLOG_OPS and _mem_here:
                    _memlog_mod.mem(f"L{_li} op={key} post", rank=_RANK)
                if _ht0:
                    exec_ops.HOSTTIME[key] = exec_ops.HOSTTIME.get(key, 0.0) + (
                        time.perf_counter() - _ht0)
                if _ENV_SYNC:
                    torch.npu.synchronize()      # 逐算子同步：异步错误当场定位到算子
                # 诊断（DS41M_OP_DIGEST=1）：层 0 rank0 每个算子后打印关键张量 sha1，
                #   用于 packed/单序列两条实现逐算子定位首个分歧点。
                if _do_digest:
                    import hashlib as _hl2
                    _s = []
                    for _k in ("x", "attn_out", "moe_out", "qr", "q", "kv",
                               "cmp_kv", "cmp_latent", "idx_q", "idx_descale_q",
                               "idx_w", "idx_sparse"):
                        _v = st.get(_k)
                        if isinstance(_v, torch.Tensor) and _v.numel():
                            _t = _v.detach().float().contiguous()
                            _dt = int(os.environ.get("DS41M_DIGEST_TAIL", "0") or 0)
                            # ★ CP：本地末行可能是 pad（非整除 L）→ 用序列末 token 的本地行号；
                            #   CP1 对照时把 DS41M_DIGEST_TAIL 设 1（x[-1] 就是全局末 token）。
                            if os.environ.get("DS41M_DIGEST_FULL"):
                                _s.append(f"{_k}=" + _hl2.sha1(_t.cpu().numpy().tobytes()).hexdigest()[:12])
                                continue
                            _cps_o = getattr(exec_ops, "CP_STATE", None)
                            _lr = (_cps_o.last_local_rows[0]
                                   if _cps_o is not None and _cps_o.last_local_rows
                                   and _cps_o.last_local_rows[0] >= 0 else -1)
                            if _lr >= 0 and _t.dim() >= 1 and _t.shape[0] > _lr:
                                _t = _t[_lr:_lr + 1].contiguous()
                            elif _dt and _t.dim() >= 2 and _t.shape[0] >= _dt:
                                _t = _t[-_dt:].contiguous()
                            _s.append(f"{_k}={_hl2.sha1(_t.cpu().numpy().tobytes()).hexdigest()[:12]}")
                    print(f"  [op-digest] L{int(layer['idx'])} {key}: " + " ".join(_s), flush=True)
                if (_ENV_NAN_TRACE
                        and (ctx.get("is_decode_step")
                             or os.environ.get("DS41M_NAN_TRACE_PREFILL") == "1")
                        and layer["idx"] < int(os.environ.get("DS41M_NAN_TRACE_LAYERS", "4"))):
                    # 同时看注意力中间量：x 只在 hc_post 类算子更新，NaN 往往更早在
                    # attn_out/o_lora/q/kv 里出现（这些键不写 x，必须单独查）。
                    _bad_keys = []
                    for _k in ("x", "attn_out", "o_lora", "q", "kv", "cmp_kv",
                               "idx_q", "idx_sparse", "topk_idx", "topk_weight"):
                        _v = st.get(_k)
                        if not isinstance(_v, torch.Tensor) or not _v.numel():
                            continue
                        if _v.dtype in (torch.float32, torch.float16, torch.bfloat16,
                                        torch.float8_e4m3fn):
                            _bad = int(torch.isnan(_v).sum())
                            if _bad:
                                # ★ 2026-10-31（主控）：给出**哪些行**是 NaN（定位"整行坏"还是"散点"）
                                _r = torch.isnan(_v).reshape(int(_v.shape[0]), -1).any(1)
                                _ridx = torch.nonzero(_r).flatten()[:8].cpu().tolist()
                                _bad_keys.append(f"{_k}={_bad}/{int(_v.numel())}"
                                                 f" rows={_ridx}")
                    if _bad_keys:
                        print(f"      [nan-trace] 层{layer['idx']}/{key} "
                              + " ".join(_bad_keys), flush=True)
            except Exception as exc:                       # noqa: BLE001
                tb, best = exc.__traceback__, None
                while tb is not None:                     # 找 exec_ops 里最深的帧
                    if "exec_ops" in tb.tb_frame.f_code.co_filename:
                        best = tb
                    tb = tb.tb_next
                loc = f"{best.tb_frame.f_code.co_filename.split('/')[-1]}:{best.tb_lineno}" if best else "?"
                errs.append((layer["idx"], key, f"{type(exc).__name__}: {exc} @ {loc}"))
                if _ENV_DEBUG:
                    print(f"      ✗ {key}: {type(exc).__name__}: {str(exc)[:700]} @ {loc}",
                          flush=True)
                # ★ decode 步的 MoE 不能按"可选算子"静默跳过：跳过后 st["x"] 保持原值，
                #   输出退化成"没算 MoE"的乱码、多步后还会 NaN（host-NZ 格式丢失曾因此
                #   伪装成"decode 跑通 16 token"）。默认严格；排查时可 DS41M_DECODE_STRICT=0。
                if (key == "mega_moe" and ctx.get("is_decode_step")
                        and os.environ.get("DS41M_DECODE_STRICT", "1") != "0"):
                    raise
                continue
            (exec_cnt if ok else skip_cnt)[key] += 1
            # ★ 2026-10-21：默认 8 rank 全落（512 行 × 每层 ~63 算子 ≈ 92GB）。
            #   逐算子对拍一般只看 rank0 → DS41M_DUMP_RANKS=0，与 DUMP_OPS 同语义。
            if _do_dump_slice:
                # 逐算子存**前 N 行切片**（同 T 逐算子对拍用：官方侧 dump 也是前 512 行切片）
                try:
                    n = _ENV_DUMP_SLICE_N
                    d = _ENV_DUMP_SLICE
                    os.makedirs(d, exist_ok=True)
                    snap = {}
                    _hold_d = st.get("_shared") if isinstance(st.get("_shared"), dict) else {}
                    for k in ("x", "residual", "q", "kv", "qr", "attn_out", "o_lora",
                              "logits", "moe_out", "topk_idx", "topk_weight",
                              "engram_kv", "engram_proj_out", "engram_out",
                              "cmp_kv", "cmp_latent", "idx_sparse", "_full_pool",
                              # 索引器缓存/候选（住在 st["_shared"]，见 exec_ops._hold）
                              "_idx_k_r1", "_idx_k_r2", "_idx_bt_r1", "_idx_bt_r2",
                              "_idx_qsli_r1", "_idx_qsli_r2", "_idx_qsli_bt_r1", "_idx_qsli_bt_r2",
                              "_li_cand_r1", "_li_candlen_r1", "_cmp_sparse_r1",
                              # mqsmla 的窗口/压缩入参（与官方 dump 同名对拍；_bt/_cu 是块表/cu）
                              "_win_idx", "_win_len", "_bt", "_cu",
                              "_cmp_sparse_in", "_cmp_len_in",
                              "_cmp_pool_r1", "_cmp_pool_r2",
                              "_cmp_nblk_r1", "_cmp_nblk_r2", "_cmp_pc_r1", "_cmp_pc_r2",
                              "_cmp_len_r1", "_cmp_len_r2", "_full_pool_nblk",
                              # 索引器算子输入（与官方 qli 钩子同名对拍）
                              "idx_q", "idx_w", "idx_descale_q"):
                        v = st.get(k)
                        if v is None:
                            v = _hold_d.get(k)
                        # ★ 2026-10-21：压缩池的数据落在**末尾块**（槽位 = 绝对压缩行号
                        #   → 块 31/32，块 1 永远空）。只落写/读两个算子、且落**整池**，
                        #   否则比的是空块（假"一致"）。
                        if k.startswith("_cmp_pool_") and key not in ("kv_compress", "mqsmla"):
                            continue
                        # ★ 池类张量的 dim0 是**块号**不是 token 行：行偏移对它们无意义
                        #   （曾用 512 偏移把 129 块的索引器缓存整段切空 → 假"缺 dump"）
                        # 池类/索引器中间量按"从头看"（与官方钩子同约定）；token 级张量才套行偏移
                        _zero_off = (k.startswith("_idx_") or k.startswith("_cmp_pool_")
                                     or k in ("idx_q", "idx_w", "idx_descale_q", "_full_pool"))
                        # ★ 2026-10-21：窗口/压缩池与索引 K 缓存是**按块号**存的（dim0=块），
                        #   且**块 0 恒空**（官方池约定）→ 从头 dump 出来全是 0（假"一致"）。
                        #   这些键从块 1 起看；`_idx_qsli_*`/`_li_cand_*` 的 dim0 是 token 行，不在此列。
                        #   注意 `_idx_k_*` 的 dim0 是**块数**（18/34），不是行号 → 不在此列。
                        _blk_pool = (k.startswith("_cmp_pool_") or k == "_full_pool")
                        min_rows = 1 if (k.startswith("_idx_") or k.startswith("idx_")) else n
                        if isinstance(v, torch.Tensor) and v.shape and v.shape[0] >= min_rows:
                            # ★ off 必须在 isinstance 之后算：TAIL 分支要读 v.shape，
                            #   而本字典里很多键在本算子/本层根本不存在（v=None）——
                            #   曾因此在 v 上取属性抛异常，被外层 `except: pass` 吞掉 →
                            #   整个算子的快照全不落盘（假"dump 没开"）。
                            _nn = n
                            if k.startswith("_cmp_pool_"):
                                off = 0                    # 整池（数据在末尾块）
                                _nn = int(v.shape[0])
                            elif _blk_pool:
                                off = 128            # BS：跳过恒空的块 0
                            elif _zero_off:
                                off = 0
                            elif _ENV_DUMP_SLICE_TAIL:
                                off = max(0, int(v.shape[0]) - n)
                            else:
                                off = _ENV_DUMP_SLICE_OFF
                            snap[k] = (v[off:off + _nn] if off else v[:_nn]).detach().float().cpu()
                    if snap:
                        # 同名算子在一层里会调多次（rmsnorm/hc_pre/hc_post 各有 attn/ffn 两段）
                        # → 文件名带上阶段，避免后一次覆盖前一次（对拍时曾因此误判）
                        stg = st.get("_norm_stage", "") if key in ("rmsnorm", "hc_pre", "hc_post") else ""
                        torch.save(snap, os.path.join(
                            d, f"rank{_RANK}_L{layer['idx']}_{key}{('_' + stg) if stg else ''}.pt"))
                except Exception as _exc:                  # noqa: BLE001
                    # 不再全静默：曾因 `except: pass` 吞掉 NoneType.shape，导致
                    # "dump 一个文件都没有"排查两轮。DS41M_DUMP_DEBUG=1 时打印。
                    if os.environ.get("DS41M_DUMP_DEBUG"):
                        print(f"      [dump] 层{layer['idx']}/{key} 快照失败: "
                              f"{type(_exc).__name__}: {_exc}", flush=True)
        if _PROBE_ON:                                  # 层内算子循环结束：其余写入记 <runner>
            _probe.set_current("<runner>")
        if "x" in st:
            x = st["x"]
        last_pre = st.get("hc_pre_t")
        if _ENV_SYNC_LAYERS:
            torch.npu.synchronize()          # 诊断：把异步设备错误钉在具体层
        if _ENV_HOSTTIME:
            _ph2 = time.perf_counter()
            exec_ops.PHASE["ops"] = exec_ops.PHASE.get("ops", 0.0) + (_ph2 - _ph1)
        if profiler is not None:
            profiler.step()
        # ★ 有界抢跑（见 _RUNAHEAD_K 注释）：默认对所有前向生效（MIN_T=0）。
        #   ★ 2026-10-26：事件对象**复用**（环形槽位），每层只剩 record + 对 K 层前
        #   事件的 synchronize（已完成时就是一次 query）；CP/非 CP 深度切换时排空重建。
        if _ru_here:
            if _RUNAHEAD_K_ACTIVE[0] != _k_here:
                for _e in _RUNAHEAD_RING:
                    _e.synchronize()
                _RUNAHEAD_RING.clear()
                _RUNAHEAD_POS[0] = 0
                _RUNAHEAD_K_ACTIVE[0] = _k_here
            if len(_RUNAHEAD_RING) < _k_here:
                _RUNAHEAD_RING.append(torch.npu.Event())
            else:
                _RUNAHEAD_RING[_RUNAHEAD_POS[0]].synchronize()
            _RUNAHEAD_RING[_RUNAHEAD_POS[0]].record()
            _RUNAHEAD_POS[0] = (_RUNAHEAD_POS[0] + 1) % _k_here
        # 常驻 ep8 的 W1：本层 AC 已建、raw 还在 → 立刻建本层 decode NZ 并释放 raw。
        if post_layer_cb is not None:
            post_layer_cb(layer, w, ctx, st)
        if (_ec_every > 0
                and (n + 1) % _ec_every == 0):
            # B6 R3：无界层内抢跑下，EC 前必须先把在途 kernel 排空（否则 host 已 free、
            # 设备还在写）。EC 只在超大 T（≥阈值）才开，代价可忽略。
            if _DRAIN_PER_FORWARD:
                try:
                    torch.npu.synchronize()
                except Exception:                      # noqa: BLE001
                    pass
            try:
                torch.npu.empty_cache()
            except Exception:                          # noqa: BLE001
                pass
            if _mem_here:
                _memlog_mod.mem(f"L{_li} empty_cache", rank=_RANK)
    if _prog_on:
        prog.stage("S3 trace 汇总")
    # B6：DRAIN_PER_FORWARD>=1 时 no_sync 不再抑制排空（层内零同步）：
    #   =1 → forward 末尾一次 `torch.npu.synchronize()`（最严）；
    #   =2 → 只等**上一个** forward 的 event（一个 forward 在飞，host/设备仍重叠）。
    _no_sync = bool(no_sync) and not stop_profiler and not _DRAIN_PER_FORWARD
    if _no_sync:
        pass                                   # 流水口径：pass 之间不排空（= 官方 hit-inner-reps 的跑法）
    elif profiler is not None and stop_profiler:
        profiler.stop()
    if _ENV_HOSTTIME:
        exec_ops.hosttime_table(time.time() - t0)      # 本轮（=一个 pass）的 host 预算
        print(f"  ── 造输入 {exec_ops.PHASE.get('input', 0):.2f}s | "
              f"层前(取权重/打包/建 ctx) {exec_ops.PHASE.get('pre', 0):.2f}s "
              f"[取权重 {exec_ops.PHASE.get('w_load', 0):.2f}s + 打包 {exec_ops.PHASE.get('pack', 0):.2f}s] | "
              f"算子循环 {exec_ops.PHASE.get('ops', 0):.2f}s | "
              f"合计 {exec_ops.PHASE.get('pre', 0) + exec_ops.PHASE.get('ops', 0):.2f}s", flush=True)
    if _DRAIN_PER_FORWARD == 2:
        # 一个 forward 在飞：record 本 forward；等上一个 forward 的 event（2 槽轮换）。
        # record 在**当前流**上，能捕获本 forward 已下发的全部 kernel。
        if len(_DF_RING) < 2:
            _DF_RING.extend(torch.npu.Event() for _ in range(2 - len(_DF_RING)))
        _p = _DF_POS[0]
        _DF_RING[_p].record()
        if _DF_ARMED[0]:
            _DF_RING[_p ^ 1].synchronize()     # 上一个 forward 完成即可，不排空当前
        _DF_ARMED[0] = True
        _DF_POS[0] = _p ^ 1
    elif not _no_sync:
        torch.npu.synchronize()
    _aux_out = _aux
    # ── packed 序列隔离验证（DS41M_PACKED_DIGEST=1，rank0）──────────────────────────
    #   对每条序列打印**末行 hidden** 的 sha1 + 统计量。判据：
    #     packed [A,B] 与 packed [A,C] 里 seq0 的摘要**逐位相同**（序列间无泄漏）；
    #     packed seq_i 与同 seed 的单序列跑也相同（打包不改变单序列结果）。
    #   cu 用**本轮实际**行边界（decoder replay 会把它改写成每序列 W 行）。
    if (os.environ.get("DS41M_PACKED_DIGEST", "").strip().lower() in ("1", "on", "true", "yes")
            and _RANK == 0):
        import hashlib as _hl
        _pk = _md.packed_view(cfg)
        _n = int(_pk.get("n", 1) or 1)
        _cu = [int(v) for v in (_pk.get("cu") or [])]
        _dig = []
        try:
            # ★ CP：末行可能是 pad（非整除 L）→ 用 cp_state.last_local_rows 定位序列末 token。
            _cps_f = getattr(exec_ops, "CP_STATE", None)
            _clr_f = (list(_cps_f.last_local_rows)
                      if _cps_f is not None and len(_cps_f.last_local_rows) == _n else None)
            if _n > 1 and len(_cu) >= _n + 1:
                for _i in range(_n):
                    _a, _b = _cu[_i], _cu[_i + 1]
                    if _b - _a <= 0:
                        continue
                    _r = x[_clr_f[_i] if _clr_f is not None else _b - 1].detach().float().cpu()
                    _dig.append(f"seq{_i}[{_a}:{_b}] sha1={_hl.sha1(_r.numpy().tobytes()).hexdigest()[:16]}"
                                f" sum={float(_r.sum()):+.6f} absmax={float(_r.abs().max()):.6g}")
            else:
                _r = x[_clr_f[0] if _clr_f is not None else -1].detach().float().cpu()
                _dig.append(f"single sha1={_hl.sha1(_r.numpy().tobytes()).hexdigest()[:16]}"
                            f" sum={float(_r.sum()):+.6f} absmax={float(_r.abs().max()):.6g}")
            print("  [packed-digest] " + " | ".join(_dig), flush=True)
        except Exception as _de:                       # noqa: BLE001 —— 诊断不许带崩主流程
            print(f"  [packed-digest] 跳过（{type(_de).__name__}: {_de}）", flush=True)
    if _ENV_MEMLOG:
        _memlog_mod.mem(f"forward end T={T}", rank=_RANK)
    return dict(exec=exec_cnt, skip=skip_cnt, errors=errs, seconds=time.time() - t0,
                x=x, meta=meta, placeholder=sorted(ph_used), weights_sec=w_sec,
                implicit=implicit_cnt, pre_mix=last_pre, collected=collected,
                aux=(_aux_out or None), snap=snap_list, pos_offset=pos_off)


# ★ CP 状态缓存（2026-10-25 空泡 A/B）：同一几何的连续 forward 复用 `CpState`
#   （plan + device 索引张量）。每 forward 重建 `build_cp_plan` 实测 **15.5ms 纯 Python**
#   （L=65536 单序列），外加 4 次 `torch.tensor(list, device=npu)` 阻塞 H2D ——
#   稳态 trace 里 forward 边界因此有 18–21ms **全设备空泡**（CP8 64K 实测：
#   边界 18.3ms，其中 14ms 无任何 host aten 事件）。plan 不可变，缓存安全。
#   `DS41M_NO_CP_PLAN_CACHE=1` 回退旧行为（A/B 用）。
_CP_STATE_CACHE: "collections.OrderedDict" = collections.OrderedDict()
_CP_STATE_MAX = 4


def _cp_state_cached(lens, world, dev, cp_size):
    import cp_state as _cps
    if os.environ.get("DS41M_NO_CP_PLAN_CACHE", "").strip().lower() in (
            "1", "on", "true", "yes"):
        return _cps.CpState.from_env(lens, world, dev, cp_size=cp_size)
    key = (int(cp_size), int(world), tuple(int(x) for x in lens), str(dev))
    cp = _CP_STATE_CACHE.get(key)
    if cp is None:
        cp = _cps.CpState.from_env(lens, world, dev, cp_size=cp_size)
        if cp is not None:
            _CP_STATE_CACHE[key] = cp
            while len(_CP_STATE_CACHE) > _CP_STATE_MAX:
                _old_key, _old = _CP_STATE_CACHE.popitem(last=False)
                try:                     # 释放被淘汰 plan 上的固定缓冲（DeepInfer 式池）
                    import cp_runtime as _cpr
                    _cpr.release_plan_buffers(_old.plan)
                except Exception:        # noqa: BLE001
                    pass
    else:
        _CP_STATE_CACHE.move_to_end(key)
    return cp


def run_forward(cfg, source, T, n_layers=None, profiler=None, verbose=True,
                allow_placeholder=False, prog=None, inputs=None, collect_dir="",
                input_ids=None, stop_profiler=True, shared=None, hit_stage=None,
                no_sync=False, pos_offset=None, l_full=None, engram_ctx=0,
                stage_prefix="", post_layer_cb=None, split_tail=False):
    """CP 入口包装（Step 2）：`DS41M_CP_SIZE>1` 且 plain prefill 时走 CP。

    - CP=1 / decode / （CP>1 时）`DS41M_CP_HIT=0` 的 hit：原样调用
      `_run_forward_impl`，零行为变化。
    - CP>1：全量 inputs/ids → 本 rank 的 head-tail 行；本地 `T`、`_packed`；
      跑完把 x / pre_mix all-gather 回全局顺序。
    - 命中 + CP（`DS41M_CP_HIT=1`，默认关）：本函数负责把 stage A/B 的**全局**
      几何（L / R / 段起点 off）落成 `_meta_override` + `CpState.pos_offset/seq_len`，
      行切片在 `split_local` 之前完成；见下方 2026-10-28 注释块。
    - Step 2 只接 SWA：`exec_ops` 里 `_kv_epilog` all-gather full KV、`_mqsmla`
      用 full pool + 本地 q 全局窗口；压缩/indexer 的 CP 在 Step 3。
    """
    _cp_size = int(os.environ.get("DS41M_CP_SIZE", "1") or 1)
    # ★ 2026-10-27：命中路径接 CP（实验开关 DS41M_CP_HIT=1）。默认关 = 旧行为零变化。
    _cp_hit = os.environ.get("DS41M_CP_HIT", "0").strip().lower() in ("1", "on", "true", "yes")
    if _cp_size <= 1 or pos_offset is not None or (hit_stage is not None and not _cp_hit):
        _r_plain = _run_forward_impl(
            cfg, source, T, n_layers=n_layers, profiler=profiler, verbose=verbose,
            allow_placeholder=allow_placeholder, prog=prog, inputs=inputs,
            collect_dir=collect_dir, input_ids=input_ids, stop_profiler=stop_profiler,
            shared=shared, hit_stage=hit_stage, no_sync=no_sync, pos_offset=pos_offset,
            l_full=l_full, engram_ctx=engram_ctx, stage_prefix=stage_prefix,
            post_layer_cb=post_layer_cb, split_tail=split_tail)
        # ★ 2026-10-27：CP=1 也落同一份"全局末行"，才能做 CP1/2/4 三方对拍
        _fr0 = os.environ.get("DS41M_DUMP_FINAL_ROW", "").strip()
        if _fr0 and _RANK == 0 and torch.is_tensor(_r_plain.get("x")):
            try:
                _row0 = _r_plain["x"].detach().float().cpu().contiguous()
                _d0 = os.path.dirname(_fr0)
                if _d0:
                    os.makedirs(_d0, exist_ok=True)
                torch.save(_row0, _fr0)
                print(f"  [cp-dump] final row sha1="
                      f"{__import__('hashlib').sha1(_row0.numpy().tobytes()).hexdigest()[:12]} → {_fr0}",
                      flush=True)
            except Exception:                       # noqa: BLE001
                pass
        return _r_plain

    # ★★ 2026-10-28：CP 分支的**命中口径**（仅 DS41M_CP_HIT=1 允许 hit_stage）。
    #   历史：这段只支持 plain prefill（内层写死 hit_stage=None）⇒ 命中进来会静默丢掉
    #   hit 语义（NOTES/2026-10-27-hit-cp-not-correct.md：CP2 vs CP4 末行 max|Δ|≈4e13）。
    #   口径（与 CP1 的命中几何逐项对齐；设计见
    #   TEAM/notes/chief-architect/2026-10-28-cp-hit-stage-plan.md）：
    #     · 池一律按**全局 L** 开（内层 `l_full=L` + `_meta_override.L=L`）⇒ stage A/B
    #       共享同一批池（压缩池 / 索引 K / 组合缓存 / 窗口 scratch 的 nblk 都由 L 决定）；
    #     · CpState.pos_offset = 本段绝对起点（stage A=0，stage B/fake=L-R）、
    #       seq_len = **原序列长 L**（= CP1 的 L_full 口径；rope 表宽 / seqused_ori_kv /
    #       压缩可见长度都按它，不能用段内行数）；
    #     · 行数：stage A 取**头部 H 行**，stage B 取**尾部 R 行**（真实前缀 stage B 传 L 行，
    #       这里按全局 R 先切，再 head-tail 分片）；
    #     · 内层仍传 hit_stage=None，但 `_meta_override.pos_offs=[off]` 让命中段
    #       pos_off>0 → `hit_replay_tail` 生效（压缩/索引缓存抑制 replay 行重写）。
    if hit_stage is not None and not _cp_hit:
        raise RuntimeError(
            "CP 分支的命中口径需要 DS41M_CP_HIT=1（默认关，回滚开关）：否则 hit_stage 会被"
            "丢弃（内层固定传 None）。请用 DS41M_CP_SIZE=1，或设 DS41M_CP_HIT=1。")
    import cp_state as cp_state_mod
    pk = _md.packed_view(cfg)
    lens = [int(x) for x in (pk.get("lens") or [T])]
    world = int(cfg["parallel_config"].get("world_size", 1) or 1)
    full_inputs = inputs if inputs is not None else make_input(cfg, T)
    if not torch.is_tensor(full_inputs):
        raise TypeError("CP 目前只支持单个 packed hidden tensor")
    dev = full_inputs.device
    _hit_cfg0 = (cfg.get("hit") or {})
    _hit_on0 = (float(_hit_cfg0.get("ratio", 0) or 0) > 0
                or int(_hit_cfg0.get("non_cached_tokens", 0) or 0) > 0)
    # ★ 2026-10-30（D2，正确性守卫）：CP>1 且是命中 case、却没开 DS41M_CP_HIT ⇒
    #   CP 分支会把它当 **plain prefill**（pos_offset=0、压缩/索引缓存写到行 0.. 而不是尾部）
    #   ⇒ 静默算错（2026-10-27 实测 CP2 vs CP4 末行 max|Δ|≈4e13）。这里直接拒绝。
    if _hit_on0 and not _cp_hit:
        raise RuntimeError(
            "CP>1 + 命中口径必须显式 DS41M_CP_HIT=1：否则命中段会被当成 plain prefill"
            "（缓存写到错误的行，静默算错 —— 2026-10-27 实测 CP2 vs CP4 末行 max|Δ|≈4e13）。"
            "要么设 DS41M_CP_HIT=1，要么用 DS41M_CP_SIZE=1。"
            "（2026-10-30：原逃生开关 DS41M_ALLOW_CP_HIT_OFF 已删除——它只会让人跑出错数。）")
    _hit_new = bool(_cp_hit and _hit_on0)
    _L_glob, _off_glob, _rows_glob = int(T), 0, int(T)
    _R_glob = int(T)
    if _hit_new:
        # ── 命中几何（全局量）：L=原序列长、R=本段要重算的行数、off=段起点 ──
        _L_run = int((cfg.get("run") or {}).get("seq_len") or 0)
        _L_glob = max(int(lens[0]) if lens else 0, _L_run, int(T))
        _R_glob = int(assemble.hit_m(cfg, _L_glob) + assemble.replay_tail_of(cfg))
        if len(lens) != 1:
            raise RuntimeError(
                f"命中 + CP 只支持单序列（_packed.n={len(lens)}）；packed 多序列命中未接"
                "（见 2026-10-28-cp-hit-stage-plan.md「不做」）")
        if hit_stage == "A":
            _rows_glob, _off_glob = int(T), 0                  # 前缀 [0, H)
        else:                                                  # stage B / 单段 fake
            if int(T) < _R_glob:
                raise RuntimeError(
                    f"命中 + CP：输入 {int(T)} 行 < 全局 R={_R_glob}（L={_L_glob}）")
            _rows_glob = _R_glob
            _off_glob = max(0, _L_glob - _R_glob)
        if int(full_inputs.shape[0]) > _rows_glob:              # 真实前缀：按全局行数切
            full_inputs = (full_inputs[:_rows_glob] if _off_glob == 0
                           else full_inputs[-_rows_glob:]).contiguous()
            if input_ids is not None and len(input_ids) > _rows_glob:
                input_ids = (input_ids[:_rows_glob] if _off_glob == 0
                             else input_ids[-_rows_glob:])
        lens = [_rows_glob]
    else:
        # ★ 非命中（含 DS41M_CP_HIT=0 回滚）：几何不一致时按单序列处理 = **旧行为逐位不变**
        #   （命中口径的几何校验都在上面 `_hit_new` 分支里；这里不许新增报错/退化）。
        if sum(lens) != int(T):
            lens = [int(T)]
    cp = _cp_state_cached(lens, world, dev, _cp_size)
    if cp is not None and _hit_new:
        # 绝对位置：pos_offset = **本段起点**（A=0 / B=L-R），seq_len = **原序列长 L**
        #   （= CP1 的 L_full 口径，见 `_boundary_cut_state` 的 seq_lens=[L_full]；
        #    stage A 的有效行只到 H，但 CP1 也声明 L，逐位对拍以 CP1 为准）。
        from cp.cp_state import CpState as _CpState
        cp = _CpState.from_plan(cp.plan, group=cp.group, device=cp.device,
                                pos_offset=int(_off_glob),
                                seq_len=int(_L_glob))
    if cp is None:
        return _run_forward_impl(
            cfg, source, T, n_layers=n_layers, profiler=profiler, verbose=verbose,
            allow_placeholder=allow_placeholder, prog=prog, inputs=inputs,
            collect_dir=collect_dir, input_ids=input_ids, stop_profiler=stop_profiler,
            shared=shared, hit_stage=hit_stage, no_sync=no_sync, pos_offset=pos_offset,
            l_full=l_full, engram_ctx=engram_ctx, stage_prefix=stage_prefix,
            post_layer_cb=post_layer_cb, split_tail=split_tail)

    local_inputs = cp.split_local(full_inputs, pad=0)
    local_ids = None
    if input_ids is not None:
        ids_t = torch.as_tensor(input_ids, dtype=torch.long, device=dev)
        local_ids = cp.split_local(ids_t, pad=0).tolist()

    cfg2 = dict(cfg)
    pk2 = dict(pk)
    pk2.update(n=len(lens), cu=list(cp.plan.local_cu),
               lens=list(cp.plan.local_lens), row_lens=list(cp.plan.local_lens))
    cfg2["_packed"] = pk2
    if _hit_new:
        # ★ 内层按**全局**几何：L=原序列长（池/rope/ref-M 宽）、T_main=全局 R、
        #   pos_offs=[段起点]（packed 路径的 pos_off 只认它，见 assemble.build_stack）。
        cfg2["_meta_override"] = {"L": int(_L_glob), "T_main": int(_R_glob),
                                  "pos_offs": [int(_off_glob)]}
        # ★ 2026-10-28：`cfg2 = dict(cfg)` 是浅拷贝 ⇒ `_meta_obj` 与父 cfg **共享**；
        #   内层 `assemble.build_stack` 的 `metadata.ensure(cfg2)` 会用父布局重写
        #   `cfg2["_packed"]`（实测：局部 [512] 行被父 [L] 覆盖）⇒ 上面的 pk2 失效。
        #   命中口径每段布局都不同（stage A=H / stage B=R，均为**段内局部**），必须丢父
        #   元数据让 ensure 从 pk2 重建；顺带避免 `_cp_replay_trim` 的 set_layout 改到父对象。
        cfg2.pop("_meta_obj", None)
        cfg2.pop("_meta", None)
    run2 = dict(cfg.get("run") or {})
    run2["seq_len"] = cp.local_rows
    cfg2["run"] = run2

    exec_ops.set_cp_state(cp)
    _cp_final = cp
    try:
        r = _run_forward_impl(
            cfg2, source, cp.local_rows, n_layers=n_layers, profiler=profiler,
            verbose=verbose, allow_placeholder=allow_placeholder, prog=prog,
            inputs=local_inputs, collect_dir=collect_dir, input_ids=local_ids,
            stop_profiler=stop_profiler, shared=shared, hit_stage=None,
            no_sync=no_sync, pos_offset=None,
            l_full=(int(_L_glob) if _hit_new else
                    (int(l_full) if l_full is not None else sum(lens))),
            engram_ctx=engram_ctx, stage_prefix=stage_prefix,
            post_layer_cb=post_layer_cb, split_tail=split_tail)
        # ★ 层 20/21 截尾会把 CP 状态换成 compact 布局（行数变少）→ 合并要用**最终**状态
        _cp_final = exec_ops.CP_STATE or cp
    finally:
        exec_ops.set_cp_state(None)

    r["x"] = _cp_final.merge_global(r["x"])
    if r.get("pre_mix") is not None:
        r["pre_mix"] = _cp_final.merge_global(r["pre_mix"])
    r["cp"] = _cp_final
    # ★ 跨 CP 尺寸对拍（诊断，默认关）：逐算子 digest 会被本地 T 的分块差异影响，
    #   全局末行 hidden 才是 cp-size 无关的等价口径。落盘一行（float32）供容差比较。
    _fr = os.environ.get("DS41M_DUMP_FINAL_ROW", "").strip()
    if _fr and _RANK == 0 and torch.is_tensor(r.get("x")):
        try:
            _row = r["x"].detach().float().cpu().contiguous()
            _dir = os.path.dirname(_fr)
            if _dir:
                os.makedirs(_dir, exist_ok=True)
            torch.save(_row, _fr)
            _h = __import__("hashlib").sha1(_row.numpy().tobytes()).hexdigest()[:12]
            print(f"  [final-row] shape={tuple(_row.shape)} sha1={_h} -> {_fr}",
                  flush=True)
        except Exception as _e:                       # noqa: BLE001
            print(f"  ⚠️ [final-row] 落盘失败：{type(_e).__name__}: {_e}", flush=True)
    if _RANK == 0 and os.environ.get("DS41M_CP_AG_TIME", "").strip().lower() in (
            "1", "on", "true", "yes"):
        import cp_runtime as _cpr
        _st = _cpr.ag_stats()
        _tot = sum(v["t"] for v in _st.values())
        _mk = " ".join(f"cols{k}:×{v['n']} {v['t']*1000:.1f}ms "
                       f"{v['bytes']/2**30:.2f}GiB"
                       for k, v in sorted(_st.items(), key=lambda kv: -kv[1]["t"]))
        print(f"  [cp-ag] {_mk} | 合计 {_tot*1000:.1f}ms", flush=True)
        _cpr.ag_stats_reset()
    return r


def _guard_hit_perlayer(hit_stage):
    """★ 2026-10-30（D1，正确性守卫）：真实命中段必须开 PERLAYER_CACHE。

    不开时所有池/缓存**只按 ratio 分**（一个 pass 内多个写者共用一块）——prefill 靠
    "读紧跟自己的写"侥幸成立，但命中 **stage B 读的是前缀**，池里是 stage A 的**最后一个
    写者**（别的层）⇒ 数值静默错（`core/run.py:63-72` 记录的根因：末 token `' that'`
    19.25 vs prefill 33.75）。decode/serve 早已强制 =1（本文件 :2286/:3452），
    **真实命中的 prefill 一直没强制**。这里补上：开它，并明说一次。
    """
    if hit_stage is None:
        return
    if os.environ.get("DS41M_PERLAYER_CACHE", "").strip().lower() in (
            "1", "on", "true", "yes"):
        return
    os.environ["DS41M_PERLAYER_CACHE"] = "1"
    print("  [hit] 真实命中段：强制 DS41M_PERLAYER_CACHE=1"
          "（否则 stage B 会读到别的层写的前缀，数值静默错）", flush=True)


def run_prefill_auto(cfg, source, T, n_layers=None, profiler=None, verbose=True,
                     allow_placeholder=False, prog=None, inputs=None, collect_dir="",
                     input_ids=None, stop_profiler=True, shared=None, hit_stage=None,
                     no_sync=False, l_full=None, stage_prefix="", post_layer_cb=None):
    """前向入口：对 T%4==1（T>1）自动拆成「prefill(T-1) + decode 步(1)」。

    背景（2026-10-19 实测）：厂商 mqsmla 在 **cmp 索引路径** 上，q 行数 T%4==1 且 T>1
    时稳定 aicore trap（T=5/9/329/333 崩；T=1/330/331/332/512 正常；DS41M_NO_CMP=1 不崩）。
    在算子内部拆段（[0,T-1]+[T-1,T]）实测仍崩（同一份 q/元数据直接跑 T=328 却正常）。

    整段前向拆分在语义上**完全等价**：
      · prefill 每行只读自己的因果历史 → 前 T-1 行与前向总行数无关；
      · 池按 l_full 开、绝对位置寻址 → 第二段从 pos_offset=T-1 续写；
      · 压缩器 ring state 由第一段推进，第二段继续消费（不会重置）。
    只对单序列、无 hit、无 profiler、有显式 inputs 的路径生效；其余保持单次调用。
    """
    _guard_hit_perlayer(hit_stage)
    _force = os.environ.get("DS41M_FORCE_SPLIT", "").strip().lower() in ("1", "on", "true", "yes")
    if os.environ.get("DS41M_DEBUG"):
        print(f"  [prefill-auto] T={T} inputs={'y' if inputs is not None else 'n'} "
              f"prof={'y' if profiler is not None else 'n'} hit={float((cfg.get('hit') or {}).get('ratio', 0) or 0)} "
              f"packed={int((cfg.get('_packed') or {}).get('n', 1) or 1)} "
              f"collect={collect_dir!r} no_sync={no_sync} force={_force}", flush=True)
    # ★ CP（Step 2）：CP 的本地 head-tail 行数是 2*ps（偶数），不做奇偶拆分；
    #   直接走 CP 包装（内部会 split 全量输入）。CP 只支持 plain prefill。
    if int(os.environ.get("DS41M_CP_SIZE", "1") or 1) > 1:
        return run_forward(cfg, source, T, n_layers=n_layers, profiler=profiler,
                           verbose=verbose, allow_placeholder=allow_placeholder,
                           prog=prog, inputs=inputs, collect_dir=collect_dir,
                           input_ids=input_ids, stop_profiler=stop_profiler,
                           shared=shared, hit_stage=hit_stage, no_sync=no_sync,
                           pos_offset=None, l_full=l_full, engram_ctx=0,
                           stage_prefix=stage_prefix, post_layer_cb=post_layer_cb)

    # ★ 2026-10-19 常驻服务：奇数 T 都拆（前 T-1 偶 + 最后 1 token decode）。
    #   原因：压缩器 ratio=2 的 ring state 只在**偶数位置**可精确续写；奇数 prompt
    #   （如 1 token 的"你好"）直接 decode 会在 topk gate 读到部分组的 NaN。
    #   同时覆盖原 mqsmla T%4==1 的 aicore trap（子集）。
    _split = (T > 1 and ((T % 2) == 1 or _force)
              and hit_stage is None and not no_sync and not collect_dir
              and profiler is None
              and float((cfg.get("hit") or {}).get("ratio", 0) or 0) == 0
              and int(_md.packed_view(cfg).get("n", 1) or 1) <= 1)
    if not _split:
        return run_forward(cfg, source, T, n_layers, profiler, verbose,
                           allow_placeholder, prog, inputs, collect_dir, input_ids,
                           stop_profiler, shared, hit_stage, no_sync, None, l_full, 0,
                           stage_prefix, post_layer_cb)
    if inputs is None:
        # 与 run_forward 的默认输入一致（--layers 调试路径不预造输入）
        inputs = make_input(cfg, T)
    # ★ 2026-10-22：奇数 prefill 拆成「T-1 + 1」时，packed 几何必须跟着第一段缩到 T-1，
    #   否则 serve 每请求设置的 cu=[0,T] 会与第一段 T-1 行不一致（qsli: cu[-1]=17 vs T=16）。
    _pk_s = _md.packed_view(cfg)
    if _pk_s and int(_pk_s.get("n", 1) or 1) <= 1:
        _pk_s["row_lens"] = [int(T) - 1]
        _pk_s["lens"] = [int(T) - 1]
        _pk_s["cu"] = [0, int(T) - 1]

    _L = int(l_full if l_full is not None else T)
    _seq0 = cfg["run"]["seq_len"]
    if verbose:
        print(f"  [prefill-split] T={T}（奇数）：前 {T - 1} token 走 prefill，"
              f"最后 1 token 走 decode 步（pos_offset={T - 1}；压缩/索引对齐）",
              flush=True)
    cfg["run"]["seq_len"] = T - 1
    _r0 = run_forward(cfg, source, T - 1, n_layers, None, verbose,
                      allow_placeholder, prog, inputs[:T - 1].contiguous(), collect_dir,
                      (list(input_ids)[:T - 1] if input_ids is not None else None),
                      stop_profiler, shared, None, False, None, _L, 0, stage_prefix,
                      post_layer_cb)
    cfg["run"]["seq_len"] = 1
    # 最后 1 token：与增量 decode 单步同口径（engram 吃最近 8 个 id + ctx=3）
    r = run_forward(cfg, source, 1, n_layers, None, False,
                    allow_placeholder, prog, inputs[T - 1:].contiguous(), collect_dir,
                    (list(input_ids)[-8:] if input_ids is not None else None),
                    stop_profiler, shared, None, False, T - 1, _L, 3,
                    "", post_layer_cb, split_tail=True)
    # DSpark：奇数拆分时两段 aux 必须顺序拼接（前缀 L 行的草稿上下文）
    if (r.get("aux") is not None and _r0.get("aux") is not None
            and len(r["aux"]) == len(_r0["aux"])):
        r["aux"] = [torch.cat([a, b], dim=0)
                    for a, b in zip(_r0["aux"], r["aux"])]
    cfg["run"]["seq_len"] = _seq0
    return r


def decode_incremental(cfg, source, ids, n_new, tok, stop_ids=(), profiler=None, prof_dir="",
                       layers=None):
    """增量 decode：prefill 一次 + 每步只算 1 个新 token（读/写各层持久 KV 池）。

    与 `decode_greedy`（全量重算）是**同一数学**的两种算法 ⇒ 两边 logits 应逐位一致；
    对不上就是增量路径的池/槽位/索引/state 有 bug（这正是最好的自检）。

    强制 `DS41M_PERLAYER_CACHE=1`：decode 时 40 层各自的窗口池/压缩池/索引缓存必须同时存活
    （prefill 里"各层共用一个池、读写紧邻"的侥幸不再成立）。
    池/元数据一律按 `L_total = len(ids) + n_new` 开 → 全程不重分配。
    """
    dev = cfg["run"].get("device", "npu:0")
    os.environ["DS41M_PERLAYER_CACHE"] = "1"
    # ★ 同上：对称缓冲容量固定（decode 每步 T 不变=1，但首步 prefill 的 T 不同 → 统一按块）
    if not os.environ.get("DS41M_MOE_CHUNK"):
        os.environ["DS41M_MOE_CHUNK"] = "auto"
    L_total = len(ids) + n_new
    shared = {"_win_cache_A": None, "_win_cache_B": None, "_cmp_cache": None}
    # ★★ 2026-10-19：`cfg["run"]["seq_len"]` 是**本次前向实际行数**（run_forward 用它建层栈），
    #   不是池容量！以前这里写 L_total → 层栈按 L_total 建（13 token 的 prompt + 8 步 decode
    #   = T=21），而 `inputs` 只有 13 行 → 层 2 的 kv_compress 按 cu_seqlens=[0,21] 读 13 行
    #   张量 ⇒ VEC UB 越界(341)/MTE 非法 GM(95)/507015。窗口层 0/1 的越界读恰好落在分配块内，
    #   所以只有第一个压缩层（层 2）炸 —— 与"小 prompt 必炸层 2"完全吻合。
    #   池容量由 run_forward(l_full=L_total) 单独传入（ctx.L_full），与层栈几何解耦。
    cfg["run"]["seq_len"] = len(ids)
    t0 = time.time()
    emb = embed_ids(cfg, ids, source, dev)
    r = run_prefill_auto(cfg, source, len(ids), (layers or None), profiler=None, verbose=True,
                         allow_placeholder=False, prog=None, inputs=emb, collect_dir="",
                         input_ids=list(ids), stop_profiler=True, shared=shared,
                         hit_stage=None, l_full=L_total)
    logits = head_logits(cfg, r["x"], r.get("pre_mix"), source)
    out_ids = list(ids)
    pieces = []
    # ── decode 采集：prefill 当 warmup（不进 trace），只采 decode 步 ──
    _dev_id = 0
    _prof_from = int(os.environ.get("DS41M_PROF_FROM_STEP", "0") or 0)
    if profiler is not None:
        _ds = str(cfg["run"].get("device", "npu:0"))
        if ":" in _ds:
            _dev_id = int(_ds.split(":")[-1])
        if _prof_from <= 0:
            profiler.start(device=_dev_id)
            if _RANK == 0:
                print(f"  [prof] decode 采集开始（prefill 不计入）→ {prof_dir}", flush=True)
        elif _RANK == 0:
            print(f"  [prof] 跳过前 {_prof_from} 个 decode 步（含 NZ 建缓存）→ {prof_dir}", flush=True)
    if _RANK == 0:
        print(f"  [inc] prefill {len(ids)} token 完成 {time.time() - t0:.1f}s"
              f"（池按 L={L_total} 开）", flush=True)
    # ★ 单机 P→D：prefill 跑完后**一次**把专家权重切成 decode 形态（decode_layout.py）。
    #   触发：DS41M_PD_MERGE=1（mode 由 DS41M_DECODE_LAYOUT 指定，默认 ep）
    #        或兼容旧开关 DS41M_TP_DECODE_ONLY=1（mode=tp）。
    #   DS41M_DECODE_PREBUILD=0 可只切模式/清缓存，按层懒构建。
    _pd_on = bool(os.environ.get("DS41M_PD_MERGE", "").strip()) or \
        os.environ.get("DS41M_TP_DECODE_ONLY", "0") == "1"
    if _pd_on:
        _pd_mode = os.environ.get("DS41M_DECODE_LAYOUT", "").strip().lower()
        if not _pd_mode:
            _pd_mode = "tp" if os.environ.get("DS41M_TP_DECODE_ONLY", "0") == "1" else "ep"
        decode_layout.prepare_decode(
            source, cfg, _RANK,
            int(cfg["parallel_config"].get("world_size", 1) or 1),
            mode=_pd_mode,
            prebuild=(os.environ.get("DS41M_DECODE_PREBUILD", "1") != "0"),
            n_layers=int(layers or len(assemble.build_stack(cfg)[0])),
            verbose=(_RANK == 0))
    _bar = os.environ.get("DS41M_DECODE_BARRIER", "").strip().lower() in ("1", "on", "true", "yes")
    _world = int(cfg["parallel_config"].get("world_size", 1) or 1)
    _step_secs = []
    t_step = time.time()
    for step in range(n_new):
        nxt = int(torch.argmax(logits))
        top = torch.topk(logits, 3)
        piece = tok.decode([nxt])
        pieces.append(piece)
        if _RANK == 0:
            print(f"  [inc {step + 1}/{n_new}] 预测位置 {len(out_ids)} → id={nxt} {piece!r}  "
                  f"top3={[(int(i), round(float(s), 2)) for s, i in zip(top.values, top.indices)]}"
                  f"  本步 {time.time() - t_step:.2f}s", flush=True)
        t_step = time.time()
        if stop_ids and nxt in stop_ids:
            if _RANK == 0:
                print("  [inc] 命中 EOS，停止", flush=True)
            break
        out_ids.append(nxt)
        if step + 1 >= n_new:
            break
        # 下一步：把刚生成的 token 以**绝对位置 pos** 送进模型（只算这 1 个 token）
        emb = embed_ids(cfg, [nxt], source, dev)
        if profiler is not None and _prof_from > 0 and step == _prof_from:
            profiler.start(device=_dev_id)
            if _RANK == 0:
                print(f"  [prof] 第 {step + 1} 步开始采集 → {prof_dir}", flush=True)
        _pf = profiler if (profiler is not None and step >= _prof_from) else None
        # 层栈几何 = 本步实际行数（1）；池仍按 l_full=L_total。（与上面 prefill 段同一修法）
        cfg["run"]["seq_len"] = 1
        r = run_forward(cfg, source, 1, (layers or None), profiler=_pf, verbose=False,
                        allow_placeholder=False, prog=None, inputs=emb, collect_dir="",
                        input_ids=out_ids[-8:], stop_profiler=False, shared=shared,
                        hit_stage=None, pos_offset=len(out_ids) - 1, l_full=L_total,
                        engram_ctx=3)
        logits = head_logits(cfg, r["x"], r.get("pre_mix"), source)
        _step_secs.append(time.time() - t_step)
        t_step = time.time()
        if _bar:
            _decode_step_barrier(_world)
    if profiler is not None:
        profiler.stop()
        try:
            profiler.finalize()
        except Exception as _exc:                      # noqa: BLE001
            print(f"  [prof] finalize 告警：{type(_exc).__name__}: {_exc}", flush=True)
        if _RANK == 0:
            print(f"  [prof] decode 采集结束 → {prof_dir}", flush=True)
    if _RANK == 0:
        print(f"\n=== decode(inc) 输出（{len(pieces)} 个新 token）===")
        print(tok.decode(out_ids[len(ids):]))
        print("=" * 60)
        if _step_secs:
            print(f"  增量步耗时：{['%.2f' % x for x in _step_secs]}s；"
                  f"均 {sum(_step_secs) / len(_step_secs):.2f}s/step"
                  f"（≈ {1000 * sum(_step_secs) / len(_step_secs):.0f} ms/token）", flush=True)
    return out_ids


# ═══════════════════════════ 常驻服务（--serve） ═══════════════════════════
# 设计：初始化只做一次；之后 rank 在自己的 ctl Unix socket 上收命令。
# 进度继续用**人类可读行**（[W1 12/40]、[ 12/40] layer、[inc 3/32]），
# server/client 用 bench_protocol.py 的正则渲染一行进度条，不侵入 run_forward 热路径。
_SERVE_CANCEL = threading.Event()


def _serve_zero_pools(obj, depth=0):
    """把常驻池里的旧数据清零（不换地址，图尾段可继续复用）。

    为什么必须清：池跨请求复用 ⇒ 本次 prompt 比上一轮短时，压缩 ratio=2 的**部分组**
    会读到上一轮留下的"未来行"，prefill 首 token 与单跑（新池）不一致。清零后未写
    的行是 0，与单跑新开池的语义一致。
    """
    if depth > 4:
        return
    if torch.is_tensor(obj):
        try:
            if obj.numel() and obj.device.type == "npu":
                obj.zero_()
        except Exception:                               # noqa: BLE001
            pass
    elif isinstance(obj, dict):
        for v in list(obj.values()):
            _serve_zero_pools(v, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _serve_zero_pools(v, depth + 1)


def reset_request_state(shared, clear_ctx=True, clear_req=True):
    """请求级重置：清几何/ring 状态，**不碰** AC、decode NZ、packed 等常驻资产。

    `clear_req=False`：W1 prefill → W2 decode 的同一请求延续（ring 状态必须保留）。
    新请求（clear_req=True）额外把池数据清零，避免短 prompt 读到上一轮的未来行。
    """
    if clear_ctx:
        _CTX_CACHE.clear()
        _SEQ_CACHE.clear()
        _MODULE_PLAN_CACHE.clear()
    if clear_req and isinstance(shared, dict):
        for k in list(shared.keys()):
            if not isinstance(k, str):
                continue
            if (k.startswith("_ffn_pre") or k.startswith("_cmp_state_")
                    or k.startswith("_cmp_sparse") or k.startswith("_li_cand")
                    or k in ("_cmp_len_in", "_win_idx")):
                shared.pop(k, None)
    if clear_req:
        _serve_zero_pools(shared)
        # 跨请求清掉"上一请求还在路上的 engram 预取"和 MegaMoE 的 staging 引用：
        # 两者都按层号全局缓存，若残留会让新请求拿到上一轮 ids 的取表结果
        # （症状：第 1 个请求正常、第 2 个请求 decode 第 2 步 layer1/2 gate NaN）。
        try:
            import exec_engram as _eg
            _eg._PREFETCH.clear()
        except Exception:                              # noqa: BLE001
            pass
        try:
            import exec_moe as _em
            _em._CACHE["src"] = None
            _em._CACHE["w"] = None
        except Exception:                              # noqa: BLE001
            pass
        # ★ 2026-10-19：元数据张量缓存（_meta / indexer metadata）原本只按形状缓存，
        #   第二个请求复用同一批 tensor 对象。实测第 2 个请求 decode 第 2 步在
        #   layer2 的 wo_b 出现整行 NaN（第一个请求正常）⇒ 先按请求清掉，排除
        #   "元数据被算子原地改写/跨请求污染"。
        try:
            import exec_ops as _eo
            _eo.META.clear()
            import exec_indexer as _ei
            _ei._MD_CACHE.clear()
            _ei._QSLI_TB.clear()
        except Exception:                          # noqa: BLE001
            pass
        # 诊断开关：请求之间**丢掉全部池**（重新分配），而不是清零复用。
        # 用于定位"第 2 个请求才 NaN"是否来自某个未覆盖的 shared 池/缓存。
        if os.environ.get("DS41M_RESET_ALL_POOLS") == "1":
            _keep = {"_win_cache_A", "_win_cache_B", "_cmp_cache"}
            for _k in list(shared.keys()):
                if _k not in _keep:
                    shared.pop(_k, None)
        # 诊断开关 2：连模块级缓存（rope/元数据/索引器 NZ）一起清掉。
        # 权重类缓存（_PREP_AC_L / _PACK_* / DECODE_CACHE）**不动**。
        if os.environ.get("DS41M_RESET_ALL_CACHES") == "1":
            try:
                import exec_ops as _eo2
                import exec_indexer as _ei2
                import exec_engram as _eg2
                _eo2._CS.clear()
                _eo2.META.clear()
                _ei2._CS_CACHE.clear()
                _ei2._MD_CACHE.clear()
                _ei2._QSLI_TB.clear()
                _ei2._NZ_CACHE.clear()
                _eg2._SYNTH_IDS.clear()
                _eg2._PREFETCH.clear()
            except Exception:                          # noqa: BLE001
                pass


def _serve_send(conn, obj):
    conn.sendall((json.dumps(obj, ensure_ascii=False, default=str) + "\n").encode())


def _serve_gib():
    try:
        return {
            "allocated_gib": round(torch.npu.memory_allocated() / 2 ** 30, 2),
            "max_allocated_gib": round(torch.npu.max_memory_allocated() / 2 ** 30, 2),
            "reserved_gib": round(torch.npu.memory_reserved() / 2 ** 30, 2),
            "total_gib": round(torch.npu.get_device_properties(0).total_memory / 2 ** 30, 1),
        }
    except Exception:                                   # noqa: BLE001
        return {}


def _serve_broadcast_token(cfg, token):
    """rank0 的 next token 广播给所有 rank，杜绝逐 rank argmax 漂移。"""
    world = int(cfg["parallel_config"].get("world_size", 1) or 1)
    if world <= 1:
        return int(token)
    import torch.distributed as dist
    t = torch.tensor([int(token)], dtype=torch.int32,
                     device=cfg["run"].get("device", "npu:0"))
    dist.broadcast(t, src=0)
    return int(t.item())


def _serve_cancel_vote(cfg):
    """decode 每步做一次 all_reduce(MAX)，保证 cancel 时 8 rank 同步退出。

    不能只查本地 Event：server 的 cancel 到达各 rank 有先后，先退的 rank 会让
    下一步 HCCL 集合通信缺人 → 整组挂死。
    """
    world = int(cfg["parallel_config"].get("world_size", 1) or 1)
    local = 1 if _SERVE_CANCEL.is_set() else 0
    if world <= 1:
        return bool(local)
    import torch.distributed as dist
    t = torch.tensor([local], dtype=torch.int32,
                     device=cfg["run"].get("device", "npu:0"))
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(int(t.item()))


def _warm_ids(n):
    """确定性的合法 token id（vocab 129280），预热用；不依赖 tokenizer。"""
    return [((i * 7919 + 13) % 120000) + 100 for i in range(int(n))]


def _serve_ac_layers():
    try:
        from exec_moe import _PREP_AC_L
        return len(_PREP_AC_L)
    except Exception:                                   # noqa: BLE001
        return -1


def _serve_nz_layers():
    try:
        import decode_layout
        return len(decode_layout.DECODE_CACHE)
    except Exception:                                   # noqa: BLE001
        return -1


_ENGRAM_PREFAULT = {"done": False}


def _prefault_engram_tables(cfg, source):
    """`DS41M_ENGRAM_PREFAULT=1`：warmup 开头把两张 engram 表的页**读一遍**。

    背景（2026-10-20）：host 取表的正常耗时 0.1–4 ms，但表格页冷时单次取行会尖刺到
    200–775 ms（verify 直接从 26 ms 顶到 0.5–1.3 s）。表格是 host mmap，页在共享盘上；
    逐 token 的随机 24 行无法承受冷页。这里在 warmup 里顺序读整表（sum 只为触发缺页），
    让页常驻 page cache，之后 host gather 稳定在亚毫秒级。
    """
    if _ENGRAM_PREFAULT["done"]:
        return
    _ENGRAM_PREFAULT["done"] = True
    if os.environ.get("DS41M_ENGRAM_PREFAULT", "").strip().lower() not in (
            "1", "on", "true", "yes"):
        return
    import torch.distributed as _d
    # ★ 只让 rank0 读：两层的表文件相同、page cache 跨进程共享；8 rank 同时读同一批
    #   冷页会互相踩（实测 8 rank 并发 2 min 才进 37 GB）。其余 rank 在 barrier 等。
    try:
        if _RANK == 0:
            stack = assemble.build_stack(cfg)[0]
            for _s in stack:
                if not _s.get("engram"):
                    continue
                li = int(_s["idx"])
                t0 = time.time()
                try:
                    _w = source.iter_layer(li, cfg["model"]["n_routed_experts"])
                    _tbl = _w.get(f"layers.{li}.engram.embed.weight")
                    if _tbl is None or _tbl.device.type != "cpu":
                        continue
                    _u8 = _tbl.view(torch.uint8).reshape(-1)
                    _chunk = 256 << 20
                    _n = int(_u8.numel())
                    for _i in range(0, _n, _chunk):
                        _u8[_i:_i + _chunk].sum(dtype=torch.int64)
                    print(f"  [engram] 预取表页：层 {li} {_n / 2**30:.1f} GiB "
                          f"{time.time() - t0:.1f}s（page cache 预热）", flush=True)
                except Exception as _e:                # noqa: BLE001
                    print(f"  [engram] 层 {li} 预取表页失败：{type(_e).__name__}: {_e}",
                          flush=True)
                finally:
                    try:
                        source.release()
                    except Exception:                  # noqa: BLE001
                        pass
    finally:
        if _d.is_initialized():
            try:
                _d.barrier()
            except Exception:                          # noqa: BLE001
                pass


_ENGRAM_PRECREATE = {"done": False}


def _precreate_engram_vendor(cfg, source, dev):
    """DSpark/verify 前把所有 engram 层的 ElasticBuffer **按顺序**建好（+ 预热一次取行）。

    2026-10-20 实测：层 1 的 buffer 建好、且第一次 `vendor_fetch` 还在飞（65 s）时，
    层 14 的 `ElasticBuffer(...)` 就报 `Acquire HCCL channel failed, ret: 11`。
    把两层都提前到 warmup 开头、每层建完 `synchronize()` 再建下一层，并各做一次
    跨 rank 预热取行，可避免首次请求里的建链竞争与 200–775 ms host gather 尖刺。
    """
    if _ENGRAM_PRECREATE["done"]:
        return
    _ENGRAM_PRECREATE["done"] = True
    if os.environ.get("DS41M_ENGRAM_PRECREATE", "1").strip().lower() in (
            "0", "off", "false", "no"):
        return
    if os.environ.get("DS41M_ENGRAM_VENDOR", "").strip().lower() not in (
            "1", "on", "true", "yes"):
        return
    try:
        import exec_engram as _ege
        import weights as _wsrc
        on_dev, tp, per = _wsrc.engram_placement(cfg)
        if on_dev or tp < 2:
            return
        stack = assemble.build_stack(cfg)[0]
        for _s in stack:
            if not _s.get("engram"):
                continue
            li = int(_s["idx"])
            try:
                _w = source.iter_layer(li, cfg["model"]["n_routed_experts"])
                _w = strip_prefix(_w, li)
                # ★ 只保留 engram 的两张表：`_vendor_buffer` 不需要本层 attention/MoE
                #   权重，也**不要** pack（以前顺带 pack 整层，P→D 前多占 ~2 GB/层，
                #   2026-10-20 T=1 基线在 P→D all2all 上直接 OOM）。
                _w = {k: v for k, v in _w.items() if k.startswith("engram.")}
                _c = dict(ctx_for(cfg, _s))
                _c.update(engram_tp_size=tp, engram_tp_rank=_RANK,
                          engram_rows=per * tp, input_ids=None, engram_ctx=0,
                          _engram_T=1)
                _buf = _ege._vendor_buffer(_w, dev, _c, force=True)
                if _buf is None:
                    print(f"  [engram] 预建 ElasticBuffer 层 {li} 失败（已回退 host）", flush=True)
                    continue
                torch.npu.synchronize()
                # 预热取行：每个 rank 的片都打一下（跨 RDMA），把首次 65 s 的建链放进 warmup。
                try:
                    _ids = (torch.arange(int(tp), dtype=torch.int32, device=dev) * int(per))
                    _h = _buf.engram_fetch(_ids)
                    _h()
                    torch.npu.synchronize()
                except Exception as _fe:               # noqa: BLE001
                    print(f"  [engram] 层 {li} 预热取行失败（不影响后续）："
                          f"{type(_fe).__name__}: {_fe}", flush=True)
                if _RANK == 0:
                    print(f"  [engram] 预建完成：层 {li}（buffer + 预热取行）", flush=True)
            except Exception as _e:                    # noqa: BLE001
                print(f"  [engram] 层 {li} 预建异常（不影响主流程）："
                      f"{type(_e).__name__}: {_e}", flush=True)
            finally:
                # 本层临时权重（EP 布局 48 专家）在预建后立刻释放；W1 会按需重取。
                try:
                    source.release()
                except Exception:                      # noqa: BLE001
                    pass
    except Exception as _e:                            # noqa: BLE001
        print(f"  [engram] 预建入口异常（不影响主流程）：{type(_e).__name__}: {_e}",
              flush=True)


def serve_warmup(cfg, source, tok, shared, lengths, l_pool, mode):
    """W1 AC → W2 decode NZ → W3 形状阶梯。返回 timings/显存。"""
    global _SERVE_CANCEL
    _SERVE_CANCEL.clear()
    dev = cfg["run"].get("device", "npu:0")
    lengths = [max(1, min(int(x), l_pool)) for x in (lengths or [128])]
    out = {"ok": True, "warm_lengths": [], "buckets": [], "mode": mode}
    _prefault_engram_tables(cfg, source)
    _precreate_engram_vendor(cfg, source, dev)

    # ── W1：40 层 AC（raw 暂不释放，W2 还要用）──
    reset_request_state(shared, clear_ctx=True)
    T1 = max(1, min(lengths[0], l_pool))
    ids = _warm_ids(T1)
    emb = embed_ids(cfg, ids, source, dev)
    cfg["run"]["seq_len"] = T1
    # ★ 2026-10-22：W1/W3 的 packed 几何同样要按“本轮长度”重置（serve 启动时
    #   main() 的 _packed 是 --seq 默认 1024；不改的话 W1 实际跑 1024 行 →
    #   层 2 mxfp8 QMM aicore 507015）。
    def _set_packed(n):
        n = int(n)
        cfg["_packed"] = {"n": 1, "lens": [n], "row_lens": [n], "cu": [0, n]}
    _set_packed(T1)
    # ★ W1 只做干净 prefill：**不缓存 AC**（raw 全量 + AC 缓存会顶到 88 GiB），
    #   也不在层循环里建 NZ（会扰动厂商 engram 侧流/异步流，短 prompt decode 出 NaN）。
    #   AC 与 NZ 统一在 W2 之后逐层建（raw 还在，建完 AC+NZ 再释放 raw）。
    _dual_cb = None
    _acn_prev = os.environ.get("DS41M_AC_CACHE_N")
    os.environ["DS41M_AC_CACHE_N"] = "0"

    t0 = time.time()
    try:
        run_prefill_auto(cfg, source, T1, None, None, verbose=(_RANK == 0),
                         allow_placeholder=False, prog=None, inputs=emb, input_ids=ids,
                         stop_profiler=True, shared=shared, hit_stage=None, l_full=l_pool,
                         stage_prefix="W1", post_layer_cb=_dual_cb)
    finally:
        if _acn_prev is None:
            os.environ.pop("DS41M_AC_CACHE_N", None)
        else:
            os.environ["DS41M_AC_CACHE_N"] = _acn_prev
    out["w1_s"] = time.time() - t0
    out["ac_layers"] = _serve_ac_layers()
    out["nz_layers"] = _serve_nz_layers()
    out.update(_serve_gib())
    if _RANK == 0:
        print(f"  [W1 done] {out['w1_s']:.3f}s  AC={out['ac_layers']} 层  "
              f"max_alloc={out.get('max_allocated_gib')} GiB", flush=True)

    # ── W2：decode 布局 ──
    world = int(cfg["parallel_config"].get("world_size", 1) or 1)
    n_stack = len(assemble.build_stack(cfg)[0])
    if mode in ("tp8-fidelity", "ep8-fidelity"):
        _pd_mode = "tp" if mode == "tp8-fidelity" else "ep"
        # 完整 P→D → 1 步 decode → D→P；结束后回到 EP raw（AC 不常驻）。
        # ★ 2026-10-19 实测：40 层 AC 融合权重 = **34.5 GiB**（不是 18），而 TP decode NZ
        #   = 44.8 GiB。两者 + raw 33.6 + 注意力/packed 12.8 远超 91.75 → 必须二选一：
        #   P→D 前释放 AC（keep_prefill=False），下一次 prefill 再按层现融合。
        #   AC 现融合实测 **~0.2s/40 层**（HBM 内拷贝），prefill 仍 ≤3s，不必常驻。
        #   ep8-fidelity 同理：EP decode NZ ~36 GiB 也不与 AC/raw 共存，P→D 建 NZ、
        #   drop raw；D→P 从 host cache 回 raw。decode 不再每 token 回拷。
        t0 = time.time()
        decode_layout.prepare_decode(
            source, cfg, _RANK, world, mode=_pd_mode, prebuild=True,
            n_layers=n_stack, verbose=(_RANK == 0), keep_prefill=False)
        out["p2d_s"] = time.time() - t0
        # ★ 只清 ctx/seq（T 从 T1 变 1），保留 _cmp_state_*：W2 是 W1 同一请求的下一步
        reset_request_state(shared, clear_ctx=True, clear_req=False)
        tok_id = ids[-1]
        emb1 = embed_ids(cfg, [tok_id], source, dev)
        cfg["run"]["seq_len"] = 1
        t0 = time.time()
        run_forward(cfg, source, 1, None, None, verbose=(_RANK == 0),
                    allow_placeholder=False, prog=None, inputs=emb1, collect_dir="",
                    input_ids=ids[-8:], stop_profiler=False, shared=shared,
                    hit_stage=None, pos_offset=T1, l_full=l_pool, engram_ctx=3,
                    stage_prefix="W2")
        out["w2_s"] = time.time() - t0
        out["nz_layers"] = _serve_nz_layers()
        t0 = time.time()
        # AC 已在 P→D 之前建好常驻 → 这里可以正常回载 EP raw（W3 的 prefill 会命中 AC，
        # 不再现场融合，因此不会再出现"建 AC + raw 共存"的 OOM-retry）。
        decode_layout.restore_ep_raw(source, cfg, _RANK, world, n_stack,
                                     verbose=(_RANK == 0))
        out["d2p_s"] = time.time() - t0
        out["dropped_raw"] = 0
    else:
        # ep8：W1 没缓存 AC、raw 全量还在。这里逐层：
        #   ① 从 raw 建 AC 并缓存（prefill 用）→ ② 建 decode NZ → pinned host
        #   → ③ 释放本层 raw。干净 prefill 之后再动，避免扰动 engram 异步流。
        import exec_moe as _em
        E = int(cfg["model"]["n_routed_experts"])
        _I = int(cfg["model"].get("moe_intermediate",
                                  cfg["model"].get("moe_intermediate_size", 0)))
        _H = int(cfg["model"]["hidden"])
        t0 = time.time()
        n_build = 0
        # ★ ep8 全量重算 decode（DS41M_SERVE_DECODE=full，MOEMODE=auto）不需要 decode NZ：
        #   decode 和 prefill 共用同一套 AC，raw 可以直接释放。省掉 36 GiB host pin
        #   和 ~12s 构建，也彻底避开 host-NZ 的每层 H2D/cast。
        _need_nz = (os.environ.get("DS41M_MOEMODE", "decode").strip().lower()
                    == "decode")
        for li in range(n_stack):
            w = source.iter_layer(li, E)
            ws = strip_prefix(w, li)
            gids = decode_layout._gids_of(ws)
            if not gids:
                continue
            _ac = _em._prepared_ac(ws, gids, _I, _H, dev)
            _em._PREP_AC_L[li] = _ac
            _em._PREP_GIDS[li] = tuple(int(x) for x in gids)
            if _need_nz:
                nz = decode_layout.ensure_decode_gmm(
                    ws, gids, _I, _H, dev,
                    {"idx": li, "is_decode_step": True, "_src": source})
                if os.environ.get("DS41M_DECODE_HOST_NZ", "").strip() == "1":
                    host = []
                    for v in nz:
                        h = torch.empty(tuple(v.shape), dtype=v.dtype,
                                        device="cpu").pin_memory()
                        h.copy_(v)
                        host.append(h)
                    decode_layout.DECODE_HOST[li] = tuple(host)
                    decode_layout.DECODE_CACHE[li] = {"gids": list(gids), "host": True}
            elif hasattr(source, "drop_layer_experts"):
                # 全量 decode：AC 已接管本层 → 立即释放 raw，W2 峰值从 80.6 → ~49 GiB
                source.drop_layer_experts(li)
            n_build += 1
            if _RANK == 0:
                # ★ NZ 放 host 不是审美选择：prefill 的 ascendc 对称缓冲按 chunk=8192
                #   常驻 ~10.05 GiB，加上 AC 18 + EP NZ 36 + 注意力/engram ≈ 80 GiB，
                #   下一次 prefill 再要 10 GiB 就 OOM（实测 device-NZ：W3 反复 OOM-retry）。
                _how = ("host offload"
                        if os.environ.get("DS41M_DECODE_HOST_NZ", "").strip() == "1"
                        else "device NZ")
                print(f"  [W2 {n_build}/{n_stack}] build AC"
                      + (f"+NZ（{_how}）" if _need_nz else "（全量 decode，无 NZ）")
                      + f" ({time.time() - t0:.1f}s)", flush=True)
        out["ac_layers"] = _serve_ac_layers()
        out["nz_layers"] = _serve_nz_layers()
        # 1 步 decode 编译 kernel（NZ 已在 host；verbose=False 不重置 W2 进度条）
        reset_request_state(shared, clear_ctx=True, clear_req=False)
        tok_id = ids[-1]
        emb1 = embed_ids(cfg, [tok_id], source, dev)
        cfg["run"]["seq_len"] = 1
        run_forward(cfg, source, 1, None, None, verbose=False,
                    allow_placeholder=False, prog=None, inputs=emb1, collect_dir="",
                    input_ids=ids[-8:], stop_profiler=False, shared=shared,
                    hit_stage=None, pos_offset=T1, l_full=l_pool, engram_ctx=3,
                    stage_prefix="W2")
        out["w2_s"] = time.time() - t0
        n_drop = 0
        if hasattr(source, "drop_all_experts"):
            n_drop = source.drop_all_experts()
        out["dropped_raw"] = n_drop
    out.update(_serve_gib())
    if _RANK == 0:
        print(f"  [W2 done] {out['w2_s']:.3f}s  NZ={out['nz_layers']} 层  "
              + (f"P→D {out.get('p2d_s', 0):.2f}s  D→P {out.get('d2p_s', 0):.2f}s  "
                 if mode in ("tp8-fidelity", "ep8-fidelity")
                 else f"drop_raw={out.get('dropped_raw')}  ")
              + f"max_alloc={out.get('max_allocated_gib')} GiB", flush=True)

    # ── W3：形状阶梯（每个长度跑 2 遍：首遍预热，次遍报数）──
    try:
        torch.npu.empty_cache()
    except Exception:                                   # noqa: BLE001
        pass
    try:
        for i, L in enumerate(lengths, 1):
            _t_eager = _t_cap = 0.0
            for _rep in range(2):
                reset_request_state(shared, clear_ctx=True)
                # 每遍 prefill 前把 reserved 还给驱动：W3 首遍要建 vendor 对称缓冲
                # （1.26/10 GiB），D→P 后 reserved 碎片会导致 OOM-retry（实测 13.9s）。
                try:
                    torch.npu.empty_cache()
                except Exception:                      # noqa: BLE001
                    pass
                ids = _warm_ids(L)
                emb = embed_ids(cfg, ids, source, dev)
                cfg["run"]["seq_len"] = L
                _set_packed(L)
                t0 = time.time()
                run_prefill_auto(cfg, source, L, None, None, verbose=False,
                                 allow_placeholder=False, prog=None, inputs=emb,
                                 input_ids=ids, stop_profiler=True, shared=shared,
                                 hit_stage=None, l_full=l_pool)
                _sec = time.time() - t0
                if _rep == 0:
                    _t_eager = _sec
                else:
                    _t_cap = _sec
                if _SERVE_CANCEL.is_set():
                    out["canceled"] = True
                    break
            out["warm_lengths"].append(L)
            out["buckets"].append({"L": L, "prefill_s": _t_cap,
                                   "eager_s": _t_eager})
            if _RANK == 0:
                print(f"  [W3 {i}/{len(lengths)}] L={L} {_t_cap:.3f}s"
                      f"（eager {_t_eager:.3f}s）", flush=True)
            if out.get("canceled"):
                break
    finally:
        try:
            torch.npu.empty_cache()      # 清 W3 的 reserved 碎片
        except Exception:                               # noqa: BLE001
            pass
    reset_request_state(shared, clear_ctx=True)
    out.update(_serve_gib())
    out["ok"] = True
    return out


def _serve_stop_ids(tok):
    eos = getattr(tok, "eos_token_id", None)
    return {int(eos)} if eos is not None else set()


def _dspark_bcast(cfg, vals):
    """rank0 的草稿 ids / 接受决策广播到所有 rank（保证 collective 步序一致）。"""
    import torch.distributed as dist
    if not dist.is_initialized():
        return list(vals)
    try:
        dev = cfg["run"].get("device", "npu:0")
        t = torch.tensor(list(vals), dtype=torch.int64, device=dev)
        dist.broadcast(t, src=0)
        return [int(x) for x in t.cpu().tolist()]
    except Exception:                                  # noqa: BLE001
        return list(vals)


def _serve_decode_dspark(cfg, source, tok, shared, ids, out_ids, pieces, step_secs,
                         n_new, l_pool, first_token, stop_ids, draft, aux0, prefill_s):
    """DSpark-3 decode：strict-all 接受。

    out_ids/pieces/step_secs 原地更新；返回 canceled。
    每个 round：propose(seed) → verify(T=B+1) → 全中则提交 B 个草稿 + 主模型
    bonus；否则恢复压缩器 ring state，用 T=1 图走一步（主模型正确 token）。
    """
    import exec_dspark
    dev = cfg["run"].get("device", "npu:0")
    B = int(draft.B)
    limit = len(ids) + int(n_new)
    pending = [int(first_token)]
    aux_ctx, aux_start = aux0, 0
    canceled = False
    first_emit = True
    emit_dt = prefill_s
    n_round = 0
    acc_hist = [0] * (B + 1)
    # 诊断：`DS41M_DSPARK_FAKE_FILE` 指向一个文本文件，**每个请求**读一次：
    #   内容 "real"/空 → 真实草稿；数字 N → fake 草稿全 N。
    # 这样一次常驻启动可以连跑 fake 0 / fake 5 / 真实三组（同 prompt ⇒ 同位置对拍）。
    _fakereq = None
    _ff = os.environ.get("DS41M_DSPARK_FAKE_FILE", "").strip()
    if _ff:
        try:
            _t = open(_ff).read().strip().lower()
            _fakereq = (False, 0) if _t in ("", "real", "off", "none") else (True, int(_t))
        except Exception:                              # noqa: BLE001
            _fakereq = None
    while len(out_ids) < limit:
        if canceled:
            break
        if pending:
            nxt = pending.pop(0)
        else:
            seed = int(out_ids[-1])
            pos = len(out_ids) - 1
            _fake = os.environ.get("DS41M_DSPARK_FAKE", "").strip().lower() in (
                "1", "on", "true", "yes")
            _fid = int(os.environ.get("DS41M_DSPARK_FAKE_ID", "0") or 0)
            if _fakereq is not None:
                _fake, _fid = _fakereq
            t0 = time.time()
            if _fake:
                # 各 rank 必须拿同一个 id（文件可能被 rank 间不同时刻读到）
                _fid = int(_dspark_bcast(cfg, [_fid if _RANK == 0 else 0])[0])
                draft_ids = [_fid] * B
            else:
                draft_ids, _ = draft.propose(aux_ctx, aux_start, seed)
                # rank0 的 proposal 为所有 rank 的 verify 输入（避免 collective 分叉）
                draft_ids = _dspark_bcast(cfg, draft_ids if _RANK == 0 else [0] * B)
            t_draft = time.time() - t0
            verify_ids = [seed] + list(draft_ids)
            cfg["run"]["seq_len"] = len(verify_ids)
            emb = embed_ids(cfg, verify_ids, source, dev)
            hist = list(out_ids) + list(draft_ids)
            snap = exec_dspark.snapshot_cmp(shared)
            t1 = time.time()
            _dpv = os.environ.get("DS41M_DSPARK_VERBOSE", "").strip().lower() in (
                "1", "on", "true", "yes")
            # ★ 2026-10-22：ep8 的 verify 是 **prefill 形状**（T=B+1），走融合 MegaMoE
            #   （AC 已常驻）而不是 decode 特化路径（host-NZ 每层回拷 ~36 GiB，
            #   实测 verify 2397→ 数十 ms）。tp8 仍按原路径（MOE_TP 由布局决定）。
            _mm_prev = os.environ.get("DS41M_MOEMODE")
            if os.environ.get("DS41M_MOEMODE", "").strip().lower() == "decode":
                os.environ["DS41M_MOEMODE"] = "auto"
            try:
                r4 = run_forward(cfg, source, len(verify_ids), None, None, verbose=_dpv,
                                 allow_placeholder=False, prog=None, inputs=emb, collect_dir="",
                                 input_ids=hist, stop_profiler=False, shared=shared,
                                 hit_stage=None, pos_offset=pos, l_full=l_pool, engram_ctx=3)
            finally:
                if _mm_prev is None:
                    os.environ.pop("DS41M_MOEMODE", None)
                else:
                    os.environ["DS41M_MOEMODE"] = _mm_prev
            t_fwd = time.time() - t1
            logits4 = head_logits_t(cfg, r4["x"], r4.get("pre_mix"), source)
            t_ver = time.time() - t1
            if _RANK == 0 and r4.get("errors"):
                print(f"  [dspark] ⚠️ verify 有 {len(r4['errors'])} 个算子报错："
                      f"{r4['errors'][:3]}", flush=True)
            main_ids = [int(x) for x in logits4.argmax(dim=-1).cpu().tolist()]
            if (_RANK == 0 and os.environ.get("DS41M_DSPARK_LOGITS", "").strip().lower()
                    in ("1", "on", "true", "yes")):
                _v3, _i3 = torch.topk(logits4[0].float(), 3)
                _pairs = [(int(a), round(float(b), 5))
                          for a, b in zip(_i3.cpu().tolist(), _v3.cpu().tolist())]
                print(f"  [dspark-logits] round {n_round + 1} row0 top3={_pairs} "
                      f"pos={pos}", flush=True)
            k = 0
            while k < B and int(draft_ids[k]) == int(main_ids[k]):
                k += 1
            k = int(_dspark_bcast(cfg, [k if _RANK == 0 else 0])[0])
            if (k < B and os.environ.get("DS41M_DSPARK_FORCE_ACCEPT", "").strip().lower()
                    in ("1", "on", "true", "yes")):
                k = B            # 诊断：强制全接受，跳过 T=1 fallback
            if os.environ.get("DS41M_DSPARK_FORCE_REJECT", "").strip().lower() in (
                    "1", "on", "true", "yes"):
                k = 0            # 诊断：强制拒绝 → 每轮走回滚 + T=1 fallback
            # 默认开：2026-10-20 32 token 与 T=1 基线逐 token 相同（不 rollback 也安全：
            # 被拒位置的窗口/压缩/索引行会在真实序列走到"该组完成"时原地重写，
            # 读侧只读已完成的组；压缩器 ring 块 = B+1 保住接受前缀末尾的 carry）。
            _partial = os.environ.get("DS41M_DSPARK_PARTIAL", "1").strip().lower() not in (
                "0", "off", "false", "no")
            if k == B:
                bonus = int(_dspark_bcast(cfg, [main_ids[B] if _RANK == 0 else 0])[0])
                new_tokens = list(draft_ids) + [bonus]
                next_aux = [a[:B + 1] for a in (r4.get("aux") or [])]
            elif _partial:
                # ★ 部分接受（2026-10-20）：提交 k 个已核对草稿 + 主模型 bonus。
                #   不 restore、不跑 T=1：
                #   · 窗口/压缩/索引器行：被拒位置的行会在真实序列走到"该组完成"时
                #     由压缩器**原地重写**，而读侧只读已完成的组 ⇒ 旧值不会被读到；
                #   · 压缩器 ring 块 = B+1（next_n+1），接受前缀末尾的 carry 仍在环里；
                #   · 草稿上下文只写**已接受**的 k+1 行 aux（seed..第 k 个草稿）。
                bonus = int(_dspark_bcast(cfg, [main_ids[k] if _RANK == 0 else 0])[0])
                new_tokens = list(draft_ids[:k]) + [bonus]
                next_aux = [a[:k + 1] for a in (r4.get("aux") or [])]
            else:
                exec_dspark.restore_cmp(shared, snap)
                cfg["run"]["seq_len"] = 1
                emb1 = embed_ids(cfg, [seed], source, dev)
                r1 = run_forward(cfg, source, 1, None, None, verbose=_dpv,
                                 allow_placeholder=False, prog=None, inputs=emb1,
                                 collect_dir="", input_ids=list(out_ids)[-8:],
                                 stop_profiler=False, shared=shared, hit_stage=None,
                                 pos_offset=pos, l_full=l_pool, engram_ctx=3)
                l1 = head_logits(cfg, r1["x"], r1.get("pre_mix"), source)
                new_tokens = [int(torch.argmax(l1))]
                next_aux = [a[:1] for a in (r1.get("aux") or [])]
                if _RANK == 0 and r1.get("errors"):
                    print(f"  [dspark] ⚠️ fallback 有 {len(r1['errors'])} 个算子报错："
                          f"{r1['errors'][:3]}", flush=True)
            if not _fake:
                draft.update_context(next_aux, pos)
            aux_ctx, aux_start = next_aux, pos
            pending.extend(new_tokens)
            emit_dt = (time.time() - t0) / max(1, len(new_tokens))
            n_round += 1
            acc_hist[min(B, k)] += 1
            if _RANK == 0:
                print(f"  [dspark] round {n_round}: accept {k}/{B}  "
                      f"draft {t_draft * 1000:.1f}ms  verify {t_ver * 1000:.1f}ms"
                      f"（fwd {t_fwd * 1000:.1f} / head {(t_ver - t_fwd) * 1000:.1f}）  "
                      f"→ +{len(new_tokens)} tok", flush=True)
            if os.environ.get("DS41M_DSPARK_EMPTY", "").strip().lower() in (
                    "1", "on", "true", "yes"):
                try:
                    torch.npu.empty_cache()
                except Exception:                      # noqa: BLE001
                    pass
            continue

        nxt = _serve_broadcast_token(cfg, nxt)
        if _serve_cancel_vote(cfg):
            canceled = True
            break
        piece = tok.decode([nxt])
        pieces.append(piece)
        if _RANK == 0:
            print(f"  [dspark {len(out_ids) - len(ids) + 1}/{n_new}] "
                  f"预测位置 {len(out_ids)} → id={nxt} {piece!r}  本步 {emit_dt:.3f}s",
                  flush=True)
        if not first_emit:
            step_secs.append(emit_dt)
        first_emit = False
        if nxt in stop_ids:
            if _RANK == 0:
                print("  [dspark] 命中 EOS，停止", flush=True)
            break
        out_ids.append(nxt)
    if _RANK == 0 and n_round:
        tot = sum(acc_hist)
        distb = "/".join(str(acc_hist[i]) for i in range(B + 1))
        print(f"  [dspark] 汇总：{n_round} round，accept 分布 k=0..{B}: {distb}；"
              f"平均接受 {sum(i * acc_hist[i] for i in range(B + 1)) / max(1, tot):.2f}/{B}",
              flush=True)
    return canceled


def serve_run_once(cfg, source, tok, shared, prompt, n_new, l_pool, mode,
                   chat=False, thinking="chat"):
    """一次完整请求：prefill →（P→D）→ decode →（D→P）。返回结构化 timings。"""
    global _SERVE_CANCEL
    dev = cfg["run"].get("device", "npu:0")
    if chat:
        ids = _chat_ids(cfg, tok, prompt, thinking)
    else:
        ids = tok.encode(prompt, add_special_tokens=True)
    if not ids:
        raise ValueError("prompt 为空")
    n_new = max(0, int(n_new))
    # ★ prompt 长度约定（2026-10-19 定稿）：
    #   ① prompt ≥16 token（太短时索引器只有个位数压缩行，边界不稳）；
    #   ② **full decode 用偶数长度**：单次 prefill（不拆分），也天然避开 mqsmla 的
    #      T%4==1 aicore trap（偶数 T 只可能是 0/2 mod 4）；全量重算每步清 state，
    #      不再依赖"奇数 ring 相位"。
    #   增量 decode（DS41M_SERVE_DECODE=inc）仍要用奇数（部分组边界），故按模式选默认。
    _MIN_DEC_TOKENS = int(os.environ.get("DS41M_MIN_DEC_TOKENS", "16") or 16)
    _dec_mode_env = os.environ.get("DS41M_SERVE_DECODE", "full").strip().lower()
    _PAD_PARITY = os.environ.get(
        "DS41M_PAD_PARITY", "odd" if _dec_mode_env == "inc" else "even").strip().lower()
    if n_new > 0:
        _n0 = len(ids)
        _n_pad = max(0, _MIN_DEC_TOKENS - _n0)
        if _PAD_PARITY == "even":
            if (_n0 + _n_pad) % 2 == 1:
                _n_pad += 1
        elif (_n0 + _n_pad) % 2 == 0:
            _n_pad += 1                      # 奇数（见上 ②）
        if _n_pad:
            _pad = tok.encode(" ", add_special_tokens=False)
            if _pad:
                _pt = int(_pad[0])
            else:
                _pt = int(getattr(tok, "bos_token_id", None)
                          or getattr(tok, "eos_token_id", None) or ids[0])
            _bos = getattr(tok, "bos_token_id", None)
            if _bos is not None and int(ids[0]) == int(_bos):
                # chat：BOS 之后插入（保持 BOS 在首位）
                ids = list(ids[:1]) + [_pt] * _n_pad + list(ids[1:])
            else:
                # raw：**末尾**补。前置空格会让模型顺着开头继续吐空格
                # （实测 raw prompt 首 token = 空格，文本为空）。
                ids = list(ids) + [_pt] * _n_pad
            if _RANK == 0:
                print(f"  [serve] prompt {_n0} token → 补 {_n_pad} 个中性 token {_pt}"
                      f"（decode 需 ≥{_MIN_DEC_TOKENS} 且奇数；prefill {len(ids)} token）",
                      flush=True)
    if len(ids) + max(1, n_new) > l_pool:
        raise ValueError(
            f"prompt {len(ids)} + decode {n_new} > 常驻池容量 {l_pool}；"
            f"请用更大的 --serve-l-max 重启服务（当前池不重分配）")

    # prefill
    cfg["run"]["seq_len"] = len(ids)
    # ★ 2026-10-22：serve 的 packed 几何必须**按本请求**重置。main() 启动时按 --seq
    #   （默认 1024）建过 cfg["_packed"]；不重置的话 build_stack 仍按 1024 行/长度算
    #   → 层 2 的 mxfp8 QMM aicore 507015 / topk gate NaN（deduce：b=1 packed 改造后
    #   serve 一直带着启动几何）。decode/verify 的 lens 由 run_forward 按 pos+T 更新。
    cfg["_packed"] = {"n": 1, "lens": [int(len(ids))], "row_lens": [int(len(ids))],
                      "cu": [0, int(len(ids))]}
    # ★ tp8 的 D→P 把 reserved 顶到 ~90 GiB（P→D 的 all2all 瞬时），下一次 prefill
    #   的首批分配会反复 OOM-retry（实测 128 形状 3.7s）。开跑前先把缓存块还给驱动。
    if mode == "tp8-fidelity":
        try:
            torch.npu.empty_cache()
        except Exception:                              # noqa: BLE001
            pass
    t0 = time.time()
    emb = embed_ids(cfg, ids, source, dev)
    r = run_prefill_auto(cfg, source, len(ids), None, None, verbose=(_RANK == 0),
                         allow_placeholder=False, prog=None, inputs=emb,
                         input_ids=list(ids), stop_profiler=True, shared=shared,
                         hit_stage=None, l_full=l_pool)
    prefill_s = time.time() - t0
    logits = head_logits(cfg, r["x"], r.get("pre_mix"), source)
    first_token = int(torch.argmax(logits))
    out_ids = list(ids)
    pieces = []
    step_secs = []
    canceled = False
    p2d_s = 0.0
    d2p_s = 0.0
    canceled = _serve_cancel_vote(cfg)                  # prefill 结束后的同步检查
    world = int(cfg["parallel_config"].get("world_size", 1) or 1)
    n_stack = len(assemble.build_stack(cfg)[0])
    p2d_done = False
    # P→D（ep8 同布局=0；tp8/ep8-fidelity 才真正转换到 decode NZ）
    if (not canceled) and n_new > 0 and mode in ("tp8-fidelity", "ep8-fidelity"):
        t0 = time.time()
        # P→D 前把 prefill 的 reserved 碎片还掉（1.31 GiB vendor 缓冲 + all2all out）
        try:
            torch.npu.empty_cache()
        except Exception:                              # noqa: BLE001
            pass
        decode_layout.prepare_decode(
            source, cfg, _RANK, world,
            mode=("tp" if mode == "tp8-fidelity" else "ep"), prebuild=True,
            n_layers=n_stack, verbose=(_RANK == 0), keep_prefill=False)
        p2d_s = time.time() - t0
        p2d_done = True
        # 布局/并行配置变了 → 层 ctx/op 序列必须重建（T 也从 prompt 变 1）
        _CTX_CACHE.clear()
        _SEQ_CACHE.clear()
        _MODULE_PLAN_CACHE.clear()
        # AC 在 P→D 时已释放（AC+TP NZ 不能共存）；D→P 后下一次 prefill 现融合 ~0.2s
    # decode
    stop_ids = _serve_stop_ids(tok)
    # ★ 2026-10-20：decode 分支在**循环外**读一次，日志标签也用同一个值。
    #   - ep8 + full：正确性基线（每步重跑整段 prefix，O(L)）；
    #   - ep8 + inc ：pinned host EP NZ，每 token 回拷 ~36 GiB（~2.2s/token，能跑但慢）；
    #   - tp8 + inc ：TP NZ 常驻设备，T=1 增量，低时延路径（full T>1 数值错误，已被
    #                 serve_loop 强制改成 inc）。
    _serve_dec = os.environ.get("DS41M_SERVE_DECODE", "full").strip().lower()
    _dec_tag = "inc" if _serve_dec == "inc" else "full"
    # ── DSpark 草稿模型（必须在 P→D 之后建：权重按 TP 布局切）──
    _dspark_B = int(os.environ.get("DS41M_DSPARK", "0") or 0)
    _dspark_draft = None
    # ★ 2026-10-21：ep8 / ep8-fidelity 也支持 DSpark（草稿 MoE 走 EP，不用 P→D 切权重）。
    #   前提是增量 decode（inc）：verify/fallback 都复用同一份 shared 前缀状态。
    #   ep8 允许 full：full decode 本身已验证（0.21s/tok），DSpark 的 verify 仍是
    #   pos_offset 的增量前向；只有被拒后的 T=1 fallback 走全量重算（O(L)）。
    if (_dspark_B > 0 and mode in ("tp8-fidelity", "ep8", "ep8-fidelity")
            and (_serve_dec == "inc" or (mode == "ep8" and _serve_dec == "full"))
            and not canceled):
        if r.get("aux") is None:
            if _RANK == 0:
                print("  [dspark] prefill 没返回 aux → 本请求退回 T=1", flush=True)
        else:
            import exec_dspark
            _dspark_draft = exec_dspark.DSparkDraft(
                cfg, source, dev, l_pool=l_pool, n_block=_dspark_B,
                rank=_RANK, world=world, verbose=(_RANK == 0))
            _fake0 = os.environ.get("DS41M_DSPARK_FAKE", "").strip().lower() in (
                "1", "on", "true", "yes")
            if not _fake0:
                _dspark_draft.reset()
                _dspark_draft.prefill_context(r["aux"], 0)
    t_step = time.time()
    try:
        _loop_range = range(n_new)
        if _dspark_draft is not None:
            canceled = _serve_decode_dspark(
                cfg, source, tok, shared, ids, out_ids, pieces, step_secs,
                n_new, l_pool, first_token, stop_ids, _dspark_draft,
                r["aux"], prefill_s)
            _loop_range = ()
        for step in _loop_range:
            if canceled:
                break
            nxt = int(torch.argmax(logits))
            nxt = _serve_broadcast_token(cfg, nxt)          # rank0 → 所有 rank
            # ★ cancel 必须在 broadcast 之后做 all_reduce：8 rank 同步退出，避免少人挂死
            if _serve_cancel_vote(cfg):
                canceled = True
                break
            piece = tok.decode([nxt])
            pieces.append(piece)
            if _RANK == 0:
                # 本步耗时 = **产出这个 token 的那次 forward**（上一步 step_secs），
                # 不是采样/broadcast 的几毫秒；第 1 个 token 由 prefill 产出。
                _dt = step_secs[-1] if step_secs else prefill_s
                print(f"  [{_dec_tag} {step + 1}/{n_new}] 预测位置 {len(out_ids)} → id={nxt} "
                      f"{piece!r}  本步 {_dt:.3f}s", flush=True)
            t_step = time.time()
            if nxt in stop_ids:
                if _RANK == 0:
                    print(f"  [{_dec_tag}] 命中 EOS，停止", flush=True)
                break
            out_ids.append(nxt)
            if step + 1 >= n_new:
                break
            if _serve_dec == "full":
                # 压缩器 ring state 必须清零（全量重算的语义：从位置 0 重写整段）。
                # clear_req=True 才会 pop `_cmp_state_*`；同时清池/元数据，避免
                # 上一步 T 更短时读到"未来行"。
                reset_request_state(shared, clear_ctx=True, clear_req=True)
                cfg["run"]["seq_len"] = len(out_ids)
                emb = embed_ids(cfg, out_ids, source, dev)
                _mtp_prev = os.environ.get("DS41M_MOE_TP")
                if mode == "tp8-fidelity":
                    os.environ["DS41M_MOE_TP"] = "1"     # 任意 T 都走 TP MoE
                try:
                    r = run_prefill_auto(cfg, source, len(out_ids), None, None,
                                         verbose=False, allow_placeholder=False, prog=None,
                                         inputs=emb, input_ids=list(out_ids),
                                         stop_profiler=False, shared=shared,
                                         hit_stage=None, l_full=l_pool)
                finally:
                    if _mtp_prev is None:
                        os.environ.pop("DS41M_MOE_TP", None)
                    else:
                        os.environ["DS41M_MOE_TP"] = _mtp_prev
            else:
                emb = embed_ids(cfg, [nxt], source, dev)
                cfg["run"]["seq_len"] = 1
                r = run_forward(cfg, source, 1, None, None, verbose=False,
                                allow_placeholder=False, prog=None, inputs=emb, collect_dir="",
                                input_ids=out_ids[-8:], stop_profiler=False, shared=shared,
                                hit_stage=None, pos_offset=len(out_ids) - 1, l_full=l_pool,
                                engram_ctx=3)
            logits = head_logits(cfg, r["x"], r.get("pre_mix"), source)
            step_secs.append(time.time() - t_step)
            t_step = time.time()
    finally:
        # D→P：即使 decode 被取消/抛错也回 EP，否则下一请求的 prefill 仍是 TP 布局。
        # ★ 必须放 finally：decode 算子失败（严格模式会 raise）后若停在 TP，服务虽活着
        #   但下一次 prefill 会按 TP 几何算 → 全错。
        if p2d_done:
            t0 = time.time()
            # D→P 要分配 6 类 pinned staging + H2D；decode 刚结束时的 reserved 碎片
            # 会让大块分配反复 retry（实测同一路径 4.3s vs 7.5s）。先还给驱动。
            try:
                torch.npu.empty_cache()
            except Exception:                          # noqa: BLE001
                pass
            decode_layout.restore_ep_raw(source, cfg, _RANK, world, n_stack,
                                         verbose=(_RANK == 0))
            d2p_s = time.time() - t0
            _CTX_CACHE.clear()
            _SEQ_CACHE.clear()
            _MODULE_PLAN_CACHE.clear()
            # 把 D→P 腾出的 reserved 还给驱动：下一次 prefill 要建 1.26/10 GiB 的
            # vendor 对称缓冲，不还缓存块会因 reserved 碎片反复 OOM-retry（实测 13.9s）。
            try:
                torch.npu.empty_cache()
            except Exception:                          # noqa: BLE001
                pass
    text = tok.decode(out_ids[len(ids):]) if pieces else ""
    dec_total = sum(step_secs)
    dec_ms = (1000.0 * dec_total / len(step_secs)) if step_secs else 0.0
    res = {
        "ok": True, "canceled": bool(canceled), "mode": mode,
        "prompt_tokens": len(ids), "n_new": len(pieces),
        "prefill_s": round(prefill_s, 4),
        "p2d_s": round(p2d_s, 4), "d2p_s": round(d2p_s, 4),
        "decode_total_s": round(dec_total, 4),
        "decode_ms_per_token": round(dec_ms, 2),
        "first_token": first_token, "ids": out_ids, "text": text,
        "rank": int(_RANK),
    }
    res.update(_serve_gib())
    return res


def _serve_median(vals):
    vals = sorted(float(v) for v in vals)
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0


def serve_run_command(cfg, source, tok, shared, cmd, l_pool, mode):
    """同一 prompt 跑 repeat 次；第 1 次清 ctx，之后复用缓存（真稳态）。"""
    global _SERVE_CANCEL
    _SERVE_CANCEL.clear()
    req_mode = (cmd.get("mode") or "").strip()
    if req_mode and req_mode != mode:
        raise ValueError(f"请求 mode={req_mode} 与服务 mode={mode} 不一致；"
                         f"换布局需重启服务")
    n_rep = max(1, int(cmd.get("repeat", 1) or 1))
    runs = []
    for i in range(n_rep):
        if _SERVE_CANCEL.is_set():
            break
        # ★ 必须每次清 ctx/seq：decode 步会把同层号的 ctx 覆盖成 T=1，
        #   不清的话第 2 次 repeat 的 prefill 会拿到 T=1 的几何（错）。
        reset_request_state(shared, clear_ctx=True)
        runs.append(serve_run_once(
            cfg, source, tok, shared, cmd.get("prompt", ""),
            int(cmd.get("decode", 0) or 0), l_pool, mode,
            chat=bool(cmd.get("chat")), thinking=cmd.get("thinking", "chat")))
        if runs[-1].get("canceled"):
            break
    steady = runs[1:] if len(runs) > 1 else runs
    summary = {
        "prefill_s": _serve_median([r["prefill_s"] for r in steady]),
        "p2d_s": _serve_median([r["p2d_s"] for r in steady]),
        "d2p_s": _serve_median([r["d2p_s"] for r in steady]),
        "decode_ms_per_token": _serve_median([r["decode_ms_per_token"] for r in steady]),
    }
    return {"ok": True, "mode": mode, "runs": runs, "summary": summary,
            "n_new": runs[0]["n_new"] if runs else 0,
            "ids": runs[0]["ids"] if runs else [],
            "text": runs[0]["text"] if runs else "",
            "logdir": os.environ.get("DS41_BENCH_LOGDIR", "")}


def serve_status(cfg, source, mode, l_pool, warmed):
    _full = (mode == "ep8"
             and os.environ.get("DS41M_SERVE_DECODE", "full").strip().lower() == "full")
    st = {
        "ok": True, "rank": int(_RANK), "pid": os.getpid(),
        "mode": mode, "l_pool": int(l_pool), "warmed": bool(warmed),
        "ac_layers": _serve_ac_layers(), "nz_layers": _serve_nz_layers(),
        "prefill_ready": _serve_ac_layers() > 0,
        # ep8 全量 decode 直接复用常驻 AC；tp8/增量 decode 才需要 NZ
        "decode_ready": bool(_serve_nz_layers() > 0
                             or (_full and _serve_ac_layers() > 0)),
        "hostcache_attached": bool(getattr(source, "_mc_mm", None) is not None),
        "hostcache_hits": int(getattr(source, "_mc_hits", 0)),
        "logdir": os.environ.get("DS41_BENCH_LOGDIR", ""),
    }
    st.update(_serve_gib())
    return st


def serve_loop(cfg, source, a, dev):
    """初始化完成后的常驻命令循环（每个 rank 一个 ctl socket）。"""
    mode = a.serve_mode or "auto"
    hc = os.environ.get("DS41M_MOE_HOST_CACHE", "").strip()
    hc_on = bool(hc) and hc.lower() not in ("0", "off", "false", "no")
    if mode == "auto":
        _hc_live = bool(hc_on) and getattr(source, "_mc_mm", None) is not None
        # ★ 2026-10-19：auto 一律选 ep8（+全量 decode，已 5 连跑稳定）。
        #   tp8-fidelity 的 TP 增量 decode 在 T=1 边界仍不稳（首步 gate NaN），
        #   需要显式 --serve-mode tp8-fidelity 才启用；host cache 只作为它的前提检查。
        mode = "ep8"
        if _RANK == 0:
            print(f"  [bench] mode=auto → ep8（host cache "
                  f"{'attached' if _hc_live else 'off'}；tp8 需显式指定）", flush=True)
    if mode == "tp8-dual":
        raise SystemExit("[bench] tp8-dual 尚未实现；先用 tp8-fidelity 或 ep8")
    if mode in ("tp8-fidelity", "ep8-fidelity"):
        if getattr(source, "_mc_mm", None) is None:
            raise SystemExit(
                f"[bench] {mode} 需要 host cache（D→P 从内存回载 EP raw）；"
                "先 python3.11 tools/moe_host_cache.py start，或改用 --serve-mode ep8")
        if mode == "tp8-fidelity":
            os.environ["DS41M_PD_TP_XFER"] = "all2all"
            os.environ["DS41M_DECODE_LAYOUT"] = "tp"
        else:
            # EP 同布局，直接建 EP decode NZ（device），不做 all2all
            os.environ.pop("DS41M_PD_TP_XFER", None)
            os.environ["DS41M_DECODE_LAYOUT"] = "ep"
            os.environ["DS41M_DECODE_HOST_NZ"] = "0"
        os.environ["DS41M_PD_MERGE"] = "1"

    os.environ["DS41M_PERLAYER_CACHE"] = "1"
    os.environ.setdefault("DS41M_MOE_CHUNK", "auto")
    # 允许外部覆盖：DS41M_MOEMODE=auto → decode 也走 prefill 的 ascendc 融合算子
    # （不建 MC2/dispatch 缓冲、不需要 decode NZ；代价是每层 MoE 从亚毫秒到 ~5.8ms）
    # ★ 2026-10-20 用户口径：**不要全量重算**。默认两种模式都走增量（inc）：
    #   - ep8 + inc：pinned host EP NZ（AC 与 device NZ 不能共存），~2.2 s/token；
    #   - tp8 + inc：TP NZ 常驻设备，T=1 增量，低时延路径（~0.23 s/step，launch 实测）。
    #   只有显式设 DS41M_SERVE_DECODE=full 才走全量重算（数值基线，O(L)）。
    _serve_dec = os.environ.get("DS41M_SERVE_DECODE", "inc").strip().lower()
    # ★ 2026-10-20 根因修复：以前这里只算了局部变量 `_serve_dec`，**没有写回环境**；
    #   `serve_run_once` 里 `os.environ.get("DS41M_SERVE_DECODE", "full")` 读到的是
    #   "full" → tp8 实际一直在跑**全量重算**（T>1 的 TP MoE 数值错误 → 第 2 个 token
    #   起乱码）。日志里每步同时出现 T=18（全量）和 T=1（尾段）两次 forward 就是证据。
    #   这里统一写回，后面所有读取（decode 分支、pad parity）都用同一个值。
    if mode in ("tp8-fidelity", "ep8-fidelity") and _serve_dec != "inc":
        _serve_dec = "inc"
        if _RANK == 0:
            print(f"  [bench] {mode} 需要增量 decode（P→D/D→P 只对 inc 有意义）；强制 "
                  "DS41M_SERVE_DECODE=inc", flush=True)
    os.environ["DS41M_SERVE_DECODE"] = _serve_dec
    if _RANK == 0:
        print(f"  [bench] serve decode={_serve_dec}"
              f"（{'增量：每步 T=1' if _serve_dec == 'inc' else '全量重算'}）", flush=True)
    if mode == "ep8" and _serve_dec == "full":
        os.environ["DS41M_MOEMODE"] = "auto"
    else:
        os.environ.setdefault("DS41M_MOEMODE", "decode")
    if mode == "ep8" and _serve_dec == "inc":
        # ep8 增量：AC（34.5 GiB）与 device EP NZ（~36 GiB）不能共存 → 只能把 NZ 放
        # pinned host，每 token 回拷 ~36 GiB（实测 ~2.2 s/token）。这能跑，但不是
        # 低时延路径；要快的增量解码请用 tp8-fidelity（TP NZ 常驻设备）。
        os.environ.setdefault("DS41M_DECODE_HOST_NZ", "1")
        # 增量 decode 的 W2 会 drop 全部 raw；prefill 只能吃常驻 AC。AC_CACHE_N=0
        # 会让 prefill 既没有 raw 也没有 AC → 直接坏。强制回 40。
        if int(os.environ.get("DS41M_AC_CACHE_N", "40") or 0) <= 0:
            os.environ["DS41M_AC_CACHE_N"] = "40"
            if _RANK == 0:
                print("  [bench] ep8 增量需要常驻 AC：DS41M_AC_CACHE_N=0 已改为 40"
                      "（否则 raw 释放后 prefill 无权重）", flush=True)
        if _RANK == 0:
            if os.environ.get("DS41M_DECODE_HOST_NZ", "1").strip() == "0":
                print("  [bench] ⚠️ ep8 增量用 device NZ：AC+NZ 可能 OOM（A/B 口径）", flush=True)
            else:
                print("  [bench] ⚠️ ep8 增量 decode 走 pinned host NZ：每 token 回拷 ~36 GiB，"
                      "实测 ~2.2 s/token；要免回拷请用 --serve-mode ep8-fidelity", flush=True)
    if mode == "ep8-fidelity":
        # P→D 建 device EP NZ、drop raw；decode 走 _moe_decode（device NZ），
        # 不再有每 token 36 GiB H2D；D→P 从 host cache 回 raw。
        os.environ["DS41M_DECODE_HOST_NZ"] = "0"
        os.environ["DS41M_AC_CACHE_N"] = "0"
        if _RANK == 0:
            print("  [bench] ep8-fidelity：EP P→D（device NZ）→ 增量 decode；"
                  "D→P 从 host cache 回 raw", flush=True)
    # ★ 2026-10-19：常驻服务默认**关 engram 侧流**。8 rank 下 using_side() 恒真（world≥2），
    #   侧流的 pending/event 状态跨请求不清；实测第 2 个请求的 decode 第 1 步在 layer2
    #   gate NaN（层 1 正是 engram 层），而 DS41M_ENGRAM_NOSIDE=1 后稳定。表取行仍走
    #   主机预取线程（DS41M_ENGRAM_HOST=1），只是 wkv 改为主干流串行。
    os.environ.setdefault("DS41M_ENGRAM_NOSIDE", "1")
    # ★ DSpark 的 verify 是 T=B+1（tail-sliced ids）：`_engram_prepare` 的预取按
    #   `input_ids[:T]` 取，T>1 时与 `_ids_of`（带 engram_ctx 上文）不一致 →
    #   关闭预取，改为层内内联 `_ids_of`（逐位确定的 engram 值）。
    _dspark_n = int(os.environ.get("DS41M_DSPARK", "0") or 0)
    if _dspark_n > 0:
        os.environ["DS41M_NO_ENGRAM_PREFETCH"] = "1"
        # 官方 RingCache 的 state block = next_n+1（DSpark-5 → 6 行），
        # 否则 T=B+1 的 verify 会按 pos%ratio 重写 carry 行。
        os.environ["DS41M_CMP_STATE_BLOCK"] = str(_dspark_n + 1)
        # ★ 2026-10-20（DSpark 性能关键）：**engram 厂商设备取行默认开**。
        #   engram 取表原本走 host mmap（98 GB/层、NFS 随机行）：常态 0.2 ms、
        #   尖刺 100–350 ms/verify（长上下文 fwd 184–251 ms）。厂商 ElasticBuffer
        #   （主机窗口 + 设备侧 RDMA）实测取行 0.3 ms，verify fwd 降到数十 ms，
        #   且与 host 取表**逐元素完全一致**（ENGRAM_CHECK max=0），32 token 与
        #   T=1 基线逐 token 相同。代价：每层 ~11–13 GiB 池 → 只在 l_max ≤ 16k 时默认开；
        #   大池（128k 等）显式 DS41M_ENGRAM_VENDOR=0（或按需 =1 强开）。
        if int(getattr(a, "serve_l_max", 0) or 0) <= 16384:
            os.environ.setdefault("DS41M_ENGRAM_VENDOR", "1")
    if mode in ("tp8-fidelity", "ep8-fidelity"):
        # raw 在每次 D→P 都会回载 → prefill 可以按层现融合 AC（~0.2s/40 层），
        # 不需要常驻 AC（34.5 GiB）。常驻反而让 prefill 期间 raw+AC+其他 = 82 GiB，
        # 只剩 ~9 GiB 给 vendor 缓冲/池，易 fragmentation 假 OOM。
        os.environ["DS41M_AC_CACHE_N"] = "0"
    else:
        # ep8：W2 之后 raw 已被 decode NZ 替换，prefill 只能吃常驻的融合 AC（必须缓存）
        os.environ.setdefault("DS41M_AC_CACHE_N", "40")
    os.environ["DS41M_SERVER_WARMUP"] = "1"
    # ★ 不要设 DS41M_CMPSTATE_RESET=1：run_forward 每段都会清 ring state，
    #   会把 prefill→decode 之间的压缩器状态冲掉（短 prompt decode 第二步即 NaN）。
    #   请求级重置由 reset_request_state() 在 prefill 之前统一做。
    os.environ.pop("DS41M_CMPSTATE_RESET", None)

    l_pool = max(16, int(a.serve_l_max))
    shared = {"_win_cache_A": None, "_win_cache_B": None, "_cmp_cache": None}
    tok = _tokenizer(cfg)
    warmed = False

    # ★ 2026-10-20：**预热前**把 PyTorch/HCCL 的 WORLD 通信域初始化一次。
    #   `all_reduce`/`broadcast` 是 lazy init：第一次调用发生在第一个 run（或取消投票）。
    #   这时 40 层 vendor 对称缓冲和大量 stream 已经建好，在共享卡上偶发
    #   `RunAicpuCommInit ... get kernel failed`（device_id=2/7、stream_id=125），
    #   整组被杀。提前建 comm 时资源充足；万一环境真有问题，也会在启动阶段明确报错，
    #   而不是等预热完再挂。注意：vendor MegaMoE 用的是它自己的 HCCL 域，覆盖不到这个。
    try:
        import torch.distributed as _d
        if _d.is_initialized():
            _wz = torch.zeros(1, dtype=torch.float32, device=dev)
            _d.all_reduce(_wz)
            _d.barrier()
            torch.npu.synchronize()
            if _RANK == 0:
                print("  [bench] HCCL WORLD 通信域已预热", flush=True)
    except Exception as _e:                            # noqa: BLE001
        print(f"  [bench] ⚠️ HCCL WORLD 预热失败：{type(_e).__name__}: {_e}",
              flush=True)
        raise

    ctl = a.control or os.path.join("/tmp/ds41_bench", f"rank{_RANK}.ctl")
    os.makedirs(os.path.dirname(ctl), exist_ok=True)
    try:
        os.unlink(ctl)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(ctl)
    srv.listen(1)
    os.chmod(ctl, 0o600)
    print(f"[bench] rank {_RANK} serve ready mode={mode} l_pool={l_pool} ctl={ctl}",
          flush=True)
    if a.serve_warmup:
        lengths = [int(x) for x in str(a.serve_warmup).split(",") if x.strip()]
        if _RANK == 0:
            print(f"  [bench] serve-warmup 自动预热：{lengths}", flush=True)
        serve_warmup(cfg, source, tok, shared, lengths, l_pool, mode)
        warmed = True

    conn, _ = srv.accept()
    srv.close()
    f = conn.makefile("r", encoding="utf-8", errors="replace", newline="\n")
    send_lock = threading.Lock()
    busy = threading.Event()

    def _send(obj):
        with send_lock:
            try:
                _serve_send(conn, obj)
            except Exception:                           # noqa: BLE001
                pass

    def _worker(op, cid, cmd):
        nonlocal warmed
        try:
            if op == "warmup":
                res = serve_warmup(cfg, source, tok, shared,
                                   cmd.get("lengths") or [128], l_pool, mode)
                warmed = True
                res["logdir"] = os.environ.get("DS41_BENCH_LOGDIR", "")
            else:
                res = serve_run_command(cfg, source, tok, shared, cmd, l_pool, mode)
            _send({"type": "result", "cmd_id": cid, "result": res})
        except ValueError as exc:
            _send({"type": "error", "cmd_id": cid, "error": str(exc),
                   "recoverable": True})
        except Exception as exc:                        # noqa: BLE001
            traceback.print_exc()
            _send({"type": "error", "cmd_id": cid,
                   "error": f"{type(exc).__name__}: {exc}", "recoverable": False})
            # 集合通信可能已失步：退出，让 supervisor 整组重启
            os._exit(3)
        finally:
            busy.clear()

    try:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError:
                continue
            op = cmd.get("op")
            cid = cmd.get("cmd_id")
            if op == "stop":
                _SERVE_CANCEL.set()
                _send({"type": "bye", "cmd_id": cid})
                break
            if op == "cancel":
                _SERVE_CANCEL.set()
                _send({"type": "ack", "cmd_id": cid, "op": "cancel"})
                continue
            if busy.is_set() and op in ("run", "warmup", "hostcache", "status"):
                _send({"type": "error", "cmd_id": cid,
                       "error": "rank 忙（有命令在跑）", "recoverable": True})
                continue
            _send({"type": "ack", "cmd_id": cid, "op": op})
            try:
                if op == "status":
                    st = serve_status(cfg, source, mode, l_pool, warmed)
                    _send({"type": "result", "cmd_id": cid, "result": st})
                elif op == "hostcache":
                    action = (cmd.get("action") or "").strip()
                    if not hasattr(source, "hostcache_attach"):
                        raise ValueError("当前权重源不是 RealSource；host cache 仅真权重可用")
                    if action == "apply":
                        res = source.hostcache_attach(cmd.get("path"))
                    elif action == "detach":
                        res = source.hostcache_detach()
                    else:
                        raise ValueError(f"hostcache action={action!r}（apply|detach）")
                    _send({"type": "result", "cmd_id": cid,
                           "result": {"ok": True, "hostcache": res}})
                elif op in ("run", "warmup"):
                    busy.set()
                    threading.Thread(target=_worker, args=(op, cid, cmd),
                                     name=f"serve-{op}", daemon=True).start()
                else:
                    raise ValueError(f"未知 op {op!r}")
            except ValueError as exc:
                _send({"type": "error", "cmd_id": cid,
                       "error": str(exc), "recoverable": True})
            except Exception as exc:                    # noqa: BLE001
                traceback.print_exc()
                _send({"type": "error", "cmd_id": cid,
                       "error": f"{type(exc).__name__}: {exc}",
                       "recoverable": False})
                break
    finally:
        try:
            f.close()
        except Exception:                               # noqa: BLE001
            pass
        try:
            conn.close()
        except Exception:                               # noqa: BLE001
            pass
        try:
            os.unlink(ctl)
        except Exception:                               # noqa: BLE001
            pass
    return 0


def _install_sync_trace():
    """DS41M_SYNC_TRACE=1：打印任何 >DS41M_SYNC_TRACE_MS（默认 5ms）的 host 同步，附 Python 栈。

    覆盖 `.to/.cpu/.item/.tolist` —— hostbound 定位用（稳态报数时**不要**开：
    包装器本身给每次调用加 ~1µs 的 Python 开销）。
    """
    if not (os.environ.get("DS41M_SYNC_TRACE", "").strip().lower()
            in ("1", "on", "true", "yes")):
        return
    import time as _t
    import traceback as _tb
    thr = float(os.environ.get("DS41M_SYNC_TRACE_MS", "5") or 5)

    def _wrap(name, orig):
        def f(self, *a, **k):
            t0 = _t.perf_counter()
            r = orig(self, *a, **k)
            dt = (_t.perf_counter() - t0) * 1000.0
            if dt >= thr:
                stack = "".join(_tb.format_stack(limit=14)[:-1])
                print(f"[sync-trace] {name} {dt:.1f}ms shape={tuple(self.shape)} "
                      f"dtype={self.dtype} dev={self.device}\n{stack}", flush=True)
            return r
        return f

    for _n in ("to", "cpu", "item", "tolist"):
        try:
            setattr(torch.Tensor, _n, _wrap(_n, getattr(torch.Tensor, _n)))
        except Exception as _e:                        # noqa: BLE001
            print(f"[sync-trace] patch {_n} 失败：{type(_e).__name__}: {_e}", flush=True)
    # ★ 2026-10-21：`torch.tensor(list, device=npu)` / `torch.as_tensor(...)` 走的是 C++ 的
    #   `aten::to`，**不会**经过上面的 Tensor.to 包装 → 之前漏抓（replay 后每 pass 一次的
    #   53ms 阻塞 H2D 就属于这种）。这里把工厂函数也包上（仅在诊断开关下）。
    for _n in ("tensor", "as_tensor"):
        try:
            _orig = getattr(torch, _n)

            def _wf(*a, _o=_orig, _nm=_n, **k):
                t0 = _t.perf_counter()
                r = _o(*a, **k)
                dt = (_t.perf_counter() - t0) * 1000.0
                if dt >= thr:
                    stack = "".join(_tb.format_stack(limit=14)[:-1])
                    dev = getattr(r, "device", "?")
                    shp = tuple(r.shape) if hasattr(r, "shape") else "?"
                    print(f"[sync-trace] torch.{_nm} {dt:.1f}ms shape={shp} dev={dev}\n{stack}",
                          flush=True)
                return r

            setattr(torch, _n, _wf)
        except Exception as _e:                        # noqa: BLE001
            print(f"[sync-trace] patch torch.{_n} 失败：{type(_e).__name__}: {_e}", flush=True)
    print(f"[sync-trace] 已开启：>{thr}ms 的 .to/.cpu/.item/.tolist/torch.tensor 打印栈",
          flush=True)


# ★ T-042 诊断（host 瘦身）：`DS41M_CPROFILE=1` → 用 cProfile 包住 `run_forward`，
#   跳过第 1 次（含权重装载/编译/预热），退出时 dump 到 `DS41M_CPROFILE_OUT`
#   （默认 /tmp/ds41m_host.prof），用 `python -m pstats` 按 tottime 排。
#   用途：`DS41M_HOSTTIME` 只到算子粒度，算子内部（aten 视图/元数据/环境变量）的
#   Python 开销要靠这个才能定位。**只影响 host 计时，不改变数值**。
if os.environ.get("DS41M_CPROFILE", "").strip().lower() in ("1", "on", "true", "yes"):
    import atexit as _atexit
    import cProfile as _cPr

    _HOST_PROF = _cPr.Profile()
    _HOST_PROF_N = [0]
    _HOST_PROF_ORIG = run_forward

    def run_forward(*a, **kw):                          # noqa: F811
        _HOST_PROF_N[0] += 1
        if _HOST_PROF_N[0] < 2:                         # 第 1 次 = 预热/编译，不采
            return _HOST_PROF_ORIG(*a, **kw)
        _HOST_PROF.enable()
        try:
            return _HOST_PROF_ORIG(*a, **kw)
        finally:
            _HOST_PROF.disable()

    def _host_prof_dump():
        _p = os.environ.get("DS41M_CPROFILE_OUT", "/tmp/ds41m_host.prof")
        _p = _p.replace(".prof", f"_r{_RANK}.prof")      # 8 个 rank 各写各的
        try:
            _HOST_PROF.dump_stats(_p)
            print(f"  [cprofile] → {_p}", flush=True)
        except Exception as _e:                         # noqa: BLE001
            print(f"  [cprofile] dump 失败: {_e}", flush=True)

    _atexit.register(_host_prof_dump)


def main():
    # ★ 2026-10-21：日志等级（DS41M_LOG_LEVEL=pure 时 stdout 全部丢弃，只留进度条）。
    #   必须在任何 print 之前安装。`--log-level` 在 argparse 之前先扫一遍 argv
    #   （launch_mini 用 parse_known_args 直接透传，所以三种入口都能用同一个 flag）。
    import ds41_log
    _lv = None
    for _i, _a in enumerate(sys.argv):
        if _a == "--log-level" and _i + 1 < len(sys.argv):
            _lv = sys.argv[_i + 1]
        elif _a.startswith("--log-level="):
            _lv = _a.split("=", 1)[1]
    if _lv:
        os.environ["DS41M_LOG_LEVEL"] = str(_lv).strip().lower()
    ds41_log.install()
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--log-level", default=None,
                    choices=["pure", "error", "warn", "info", "debug"],
                    help="日志等级：pure 只留进度条；默认 info（也可用 DS41M_LOG_LEVEL）")
    ap.add_argument("--seq", type=int, default=1024,
                    help="序列长度；--prompt 时传 0 = 按 prompt 全量（不截断）")
    ap.add_argument("--batches", type=int, default=1,
                    help="每个 iteration 的 batch 数：sequential=顺序跑 N 个独立 pass；"
                         "packed=把 N 条序列拼成**一次前向**（varlen cu_seqlens）")
    ap.add_argument("--batch-mode", default="auto", choices=["auto", "sequential", "packed"],
                    help="auto（默认，规则口径）：batches>1 → packed（tokenwise 打包 + attention "
                         "逐序列恢复）；sequential 仅显式 A/B（默认关，不得用于报数）")
    ap.add_argument("--input-seed", type=int, default=0,
                    help="合成输入第一个 batch 的随机种子（默认 0；对拍/复现用）")
    ap.add_argument("--iters", type=int, default=0,
                    help="重复前向轮数（性能采集用；默认取 config 的 run.iters；profile step = layers×iters）")
    ap.add_argument("--layers", type=int, default=0, help="只跑前 N 层（冒烟用）")
    ap.add_argument("--prof", action="store_true",
                    help="用 ALOP 采集 profiling 数据（替代原 torch_npu.profiler）")
    ap.add_argument("--alop-config", default="", help="ALOP 配置文件路径（默认用内置默认配置）")
    ap.add_argument("--out", default="", help="trace 输出目录（建议放共享盘）")
    ap.add_argument("--progress", default="on", choices=["on", "off"], help="阶段进度（stderr）")
    ap.add_argument("--progress-every", type=float, default=1.0, help="进度刷新间隔(秒)")
    ap.add_argument("--device", default="", help="指定卡，如 npu:3（多人并行时各用各的卡）")
    ap.add_argument("--prompt", default="", help="给了就走词嵌入 → 40 层 → head，贪心出下一个 token")
    ap.add_argument("--dump-hidden", default="", help="把末层 hidden 存到该路径（对拍用）")
    ap.add_argument("--collect-layers", default="",
                    help="逐层 dump 目录（与官方 DS41L_COLLECT_LAYERS 同格式/同表达式）")
    ap.add_argument("--collect-ranks", default="all",
                    help="逐层 dump 落盘哪些 rank：all（默认）| 0 | 0,3（真权重对拍只收 rank0 省 100+ GB）")
    ap.add_argument("--warmup-iters", type=int, default=0,
                    help="先跑 K 个**不进 profiler** 的 pass（真权重首 pass 含权重装载，别写进 trace）")
    ap.add_argument("--points", default="",
                    help="多点位预热跑（JSON list：{L,hit,batches,replay[,replay_tail]}）："
                         "一个进程内逐点重建几何/池/输入并各自预热+采集（不重复重启）；"
                         "trace 落 <out>/L<L>/h<hhh>_b<N>_r<0|1>/prefill")
    ap.add_argument("--prof-ranks", default="all",
                    help="哪些 rank 开 profiler：all（默认）| 0 | 0,3（只落一张卡的 trace，其余 rank 照跑）")
    ap.add_argument("--ids-file", default="", help="从文件读 input_ids（与官方 bench 同一份，对拍用）")
    ap.add_argument("--ids-offset", type=int, default=None,
                    help="从 --ids-file 的第 N 个 token 开始取（显式给 0 也会按本次 T 切片；"
                         "packed 对拍：seqB 用 2048）")
    ap.add_argument("--decode", type=int, default=0,
                    help="贪心续写 N 个 token（**全量重算版**：每步重跑整段，数值等价 prefill；慢）")
    ap.add_argument("--chat", action="store_true",
                    help="用官方 chat 模板包装 --prompt（该权重无 chat_template，见 _chat_ids）")
    ap.add_argument("--thinking", default="chat", choices=["chat", "thinking"])
    ap.add_argument("--decode-mode", default="full", choices=["full", "inc"],
                    help="full=每步全量重算（金标准）；inc=增量（prefill 1 次 + 每步 1 token，读持久池）")
    ap.add_argument("--weights-path", default="",
                    help="覆盖 config 里的 weights.path（真权重模式换机器/换版本用）")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--master", default="", help="多 rank 时的 MASTER_ADDR:PORT")
    ap.add_argument("--allow-placeholder", action="store_true",
                    help="允许用占位输入跑（缺 Tier B 算子时）；会在报告里标出")
    ap.add_argument("--serve", action="store_true",
                    help="常驻服务模式：初始化一次后进入命令循环（不被 bench_server 直接调用时勿用）")
    ap.add_argument("--control", default="", help="常驻模式的控制 Unix socket（rank 自己 listen）")
    ap.add_argument("--serve-l-max", type=int, default=4096,
                    help="常驻池容量：prompt+decode 的最大总 token 数")
    ap.add_argument("--serve-mode", default="auto",
                    choices=["auto", "ep8", "ep8-fidelity", "tp8-fidelity", "tp8-dual"],
                    help="常驻布局模式（ep8=host NZ 增量；ep8-fidelity=P→D device NZ；"
                         "tp8-fidelity=TP P→D；auto 一律 ep8）")
    ap.add_argument("--serve-warmup", default="",
                    help="启动后自动预热（逗号分隔的长度阶梯；空=不自动）")
    a = ap.parse_args()
    _install_sync_trace()                                  # DS41M_SYNC_TRACE=1 才生效
    global _RANK
    _RANK = a.rank

    cfg = assemble.load_yaml(a.config)
    # ★ 2026-10-22：把 `cfg.backends.megamoe` 接到 `DS41M_MOEMODE`（唯一真源）。
    #   exec_moe 的后端选择只看 env；没接之前 config 写 deepgemm、但 EP≥2 时
    #   `auto` 会静默优先 ascendc（实测 run_hit 多点位日志：banner=deepgemm、
    #   实际 `[mega_moe] 后端 = ascendc`）。只在 **prefill-only**（无 --decode）时接；
    #   decode 走 `decode_layout` 自己的 `DS41M_MOEMODE=decode`，不能在这里钉死。
    _be_moe = str((cfg.get("backends") or {}).get("megamoe", "") or "").lower()
    if (not int(getattr(a, "decode", 0) or 0)
            and _be_moe in ("ascendc", "deepgemm")
            and not os.environ.get("DS41M_MOEMODE")):
        os.environ["DS41M_MOEMODE"] = _be_moe
    cfg["run"]["seq_len"] = a.seq
    # ── 真 batch（packed）：N 条序列拼成一次前向（varlen cu_seqlens）──────────────
    #   ★ 规则（PERF_RULES §0）：batch 的唯一实现 = tokenwise 打包 seq 维 + attention 按
    #   序列边界恢复。`--batches N` 默认（auto）N>1 走 packed；sequential 仅显式 A/B。
    _bmode = str(getattr(a, "batch_mode", "auto") or "auto").lower()
    _nseq = max(1, int(a.batches or 1))
    # ★ 2026-10-21（统一 packed，b=1 也对齐完成：40/40 层逐位）：auto 的 b=1 也走 packed。
    #   只有 packed **不支持**的组合才回退 sequential（decode / --layers / --prompt /
    #   real 前缀命中）——此时保持旧 b1 路径，不报错。显式 --batch-mode sequential 仍可 A/B。
    _hitc0 = cfg.get("hit") or {}
    _auto_pack_ok = (not (a.decode or a.layers or a.prompt)
                     and not (float(_hitc0.get("ratio", 0) or 0) > 0
                              and str(_hitc0.get("source", "fake")).lower() == "real"))
    _packed = (_bmode == "packed") or (_bmode == "auto" and (_nseq > 1 or _auto_pack_ok))
    if _packed:
        _hitc = cfg.get("hit") or {}
        _hr = float(_hitc.get("ratio", 0) or 0)
        if a.decode or a.layers or a.prompt:
            raise SystemExit("--batch-mode packed 不支持 --decode / --layers / --prompt")
        if _hr > 0 and str(_hitc.get("source", "fake")).lower() == "real":
            raise SystemExit("packed hit 暂不支持 real prefix（只支持 fake 零值前缀）")
        _L = int(a.seq)
        if _hr > 0:
            _R = assemble.hit_m(cfg, _L) + assemble.replay_tail_of(cfg)
        else:
            _R = _L
        # run.seq_len = **每序列逻辑长度 L**（t_main 按它算每序列 R）；
        # 本 pass 实际行数 = R×N（build_stack 的 T_pass）；
        # cu = 输入行边界（每请求 R 行）；lens = 每请求**逻辑**长度 L（池/索引器可见长度）。
        cfg["run"]["seq_len"] = _L
        #   lens = 每请求**逻辑**长度（池/索引器/rope 的可见长度）；row_lens = 本 pass
        #   实际输入行数（hit: R<L）。run_forward 每个 pass 开头按 row_lens 重建 cu。
        cfg["_packed"] = {"n": _nseq, "lens": [_L] * _nseq, "row_lens": [_R] * _nseq,
                          "cu": [_R * i for i in range(_nseq + 1)]}
        print(f"  [packed] 真 batch：{_nseq} 条 × L={_L}（本 pass {_R}×{_nseq} 行）"
              f" hit={_hr:.2f} src={_hitc.get('source', 'fake')} "
              f"cu={cfg['_packed']['cu'][:4]}…", flush=True)
    cfg["parallel_config"]["_ep_rank"] = _RANK
    cfg["parallel_config"]["moe_ep_size"] = a.world        # EP = world（cp1 · attn_tp1）
    cfg["parallel_config"]["world_size"] = a.world
    if os.environ.get("DS41M_MOE_TP", "0") == "1":
        # 低时延 TP 口径：本 rank 拥有**全部**专家的 1/world 中间维分片（EP 关闭）
        cfg["parallel_config"]["moe_ep_size"] = 1
        cfg["parallel_config"]["moe_tp_size"] = a.world
        cfg["parallel_config"]["_moe_tp_rank"] = _RANK
    if a.device:
        cfg["run"]["device"] = a.device
    if a.weights_path:
        cfg["weights"]["path"] = a.weights_path
    # ★ 结构参数的单一事实源：先构建 + 校验（fail-fast；DS41M_CONFIG_VALIDATE=0 可跳过）。
    #   所有 exec_* 的模块常量随后由 ensure_ops(cfg) 一次性对齐（默认值 = 旧硬编码值）。
    _strict = os.environ.get("DS41M_CONFIG_VALIDATE", "1").strip().lower() not in (
        "0", "off", "false", "no")
    model_spec.set_spec(cfg, strict=_strict)
    prog = Progress(enabled=(a.progress != "off"), every=a.progress_every, stages=STAGES_MINI)
    prog.stage("S0 环境与算子注册")
    print("== 加载 vendor 算子命名空间")
    exec_ops.ensure_ops(cfg)
    dev = cfg["run"].get("device", "npu:0")
    # ★ 先设当前卡：否则 current_device() 停在 0 而张量在别的卡 → hc_pre 等算子 507035
    # （实测：不设时 layer0 第一个 custom 算子就 aicore fault）
    torch.npu.set_device(dev)
    if a.world > 1:                                        # 多 rank：真 EP，建 HCCL 进程组
        import torch.distributed as dist
        os.environ["RANK"] = str(_RANK)
        os.environ["WORLD_SIZE"] = str(a.world)
        os.environ.setdefault("MASTER_ADDR", (a.master or "127.0.0.1:29517").split(":")[0])
        os.environ.setdefault("MASTER_PORT", (a.master or "127.0.0.1:29517").split(":")[1])
        if not dist.is_initialized():
            dist.init_process_group("hccl", rank=_RANK, world_size=a.world)
            dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(
                _RANK, init_comm=True)
        print(f"  进程组就绪：rank {_RANK}/{a.world}")
    source = weight_src.make_source(cfg, device=dev, verbose=False)

    prof_dir = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prof")
    # profiler 的 step 边界 = **每层一次**（run_forward 里逐层 step()）→ 一个 pass 打 n_layers 个 step，
    # iters 个 pass 就是 iters×n_layers 个 step（MindStudio 里按 iteration 逐段看）。
    n_stack = len(assemble.build_stack(cfg)[0])
    iters = max(1, a.iters or int(cfg["run"].get("iters", 1) or 1))
    # --prof-ranks：只让指定 rank 开 profiler（8 卡照跑；trace 只落这几张，省盘/省带宽）
    _pr = (getattr(a, "prof_ranks", None) or "all").strip().lower()
    _prof_here = (_pr in ("", "all")
                  or _RANK in {int(x) for x in _pr.split(",") if x.strip()})
    if a.prof and _prof_here and not a.points:
        profiler = make_profiler(prof_dir, a.alop_config)
    else:
        profiler = None
    if a.prof and not _prof_here:
        print(f"  [prof] rank {_RANK} 不开 profiler（--prof-ranks {_pr}）", flush=True)
    # --collect-ranks：本 rank 要不要落逐层 dump（真权重 8 rank 全落 100+ GB，对拍只收 rank0）
    _cr = (a.collect_ranks or "all").strip().lower()
    _collect_here = (_cr in ("", "all")
                     or _RANK in {int(x) for x in _cr.split(",") if x.strip()})

    # ── 常驻服务：初始化只做一次，之后进命令循环（不建 profiler / 不预造输入）──
    if a.serve:
        return serve_loop(cfg, source, a, dev)

    # ── 多点位预热跑（--points）：一个进程内逐点位重建几何/池/输入，各自预热+采集 ──
    #   ★ 用户口径（2026-10-20）："不同点位不需要重复重启" —— 权重/进程组/算子注册只做这一次。
    if a.points:
        import points_runner
        # ★ 必须传**正在运行的**模块对象（`__main__`）：points_runner 若自己 `import runner`，
        #   会以脚本身份再加载一份 runner 模块 → `_RANK` 恒 0（8 rank 全开 profiler）、
        #   图/池缓存也分成两份（实测点 2 崩）。见 NOTES/2026-10-20-points-warm-run.md。
        sys.exit(points_runner.run_points(cfg, source, a, dev,
                                          runner_mod=sys.modules[__name__]) or 0)

    print(f"== 真跑：L={a.seq} layers={a.layers or '全部'} iters={iters} 卡={cfg['run'].get('device', 'npu:0')} "
          f"megamoe={cfg['backends']['megamoe']} prof={bool(a.prof)}"
          f"{'(ALOP)' if a.prof else ''}"
          + (f"（profile step = {max(1, a.layers or n_stack) * iters}）" if a.prof else ""))
    inputs = None
    if a.ids_file:                                # 与官方 bench 读**同一份** input_ids（对拍前提）
        if a.ids_file.endswith(".json"):
            import json as _json
            ids = _json.load(open(a.ids_file))
            ids = ids if isinstance(ids, list) else ids.get("input_ids", ids)
        else:
            ids = torch.load(a.ids_file)
            ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
        ids = [int(i) for i in ids]
        _off = getattr(a, "ids_offset", None)
        _need = int(a.seq) * (_nseq if _packed else 1)
        if _packed or _off is not None:
            _off = int(_off or 0)
            _before = len(ids)
            ids = ids[_off:_off + _need]
            print(f"  ids 切片 [{_off}:{_off + len(ids)}]（原 {_before}，本次 {len(ids)}）", flush=True)
        print(f"  ids 文件 {a.ids_file} → {len(ids)} 个 token: {ids[:8]}…")
        inputs = embed_ids(cfg, ids, source, dev)
        cfg["run"]["seq_len"] = len(ids)
        print(f"  嵌入 {tuple(inputs.shape)} {inputs.dtype}")
    elif a.prompt:
        tok = _tokenizer(cfg)
        if a.chat:
            ids = _chat_ids(cfg, tok, a.prompt, a.thinking)
            print(f"  [chat] thinking_mode={a.thinking} → {len(ids)} 个 token")
        else:
            ids = tok.encode(a.prompt, add_special_tokens=True)
        if a.seq > 0:
            ids = ids[: a.seq]
        else:
            a.seq = len(ids)          # --seq 0：按 prompt 全量 prefill（自定义 prompt 常用）
        print(f"  prompt {a.prompt!r} → {len(ids)} 个 token: {ids[:12]}{'…' if len(ids) > 12 else ''}")
        inputs = embed_ids(cfg, ids, source, dev)
        cfg["run"]["seq_len"] = len(ids)          # ★ 层栈的 T 要跟提示长度一致
        print(f"  嵌入 {tuple(inputs.shape)} {inputs.dtype}")
    # ── decode 阶段（贪心续写）：给了 --decode N 就在**单跑**路径上逐 token 续写 ──
    #   全量重算版：每步重跑整段 prefix（数值 = prefill 等价，已验证与官方逐位一致）。
    if a.decode:
        if not (a.prompt or a.ids_file):
            raise SystemExit("--decode 需要 --prompt 或 --ids-file 提供提示")
        _tok = _tokenizer(cfg)
        _eos = getattr(_tok, "eos_token_id", None)
        _stop = {int(_eos)} if _eos is not None else ()
        if a.decode_mode == "inc":
            decode_incremental(cfg, source, ids, int(a.decode), _tok, stop_ids=_stop,
                               profiler=profiler, prof_dir=prof_dir, layers=a.layers or None)
        else:
            decode_greedy(cfg, source, ids, int(a.decode), _tok, stop_ids=_stop)
        return
    # ── 命中：真前缀（hit.source=real）→ 先跑 stage A 把前缀真算一遍，再进测量循环 ──
    # 合成权重（weights.mode=synthetic）用 fake：前缀 = 零值池、各层共用，不跑 stage A
    # （合成权重下 real 没有数值意义，跑 40 层前缀纯属浪费）。
    hit_cfg = cfg.get("hit") or {}
    # DS41M_HIT_REAL_FORCE=1：允许 `--layers N` 也走真前缀（stage A 真算 → stage B 读池）。
    # ★ 只为**快速调试**：病 C（层 14 门控 NaN 崩）只在 40 层全量时咬人，`--layers 2`
    #   能秒级复现命中路径的数值问题（层 0/1 的 KV 链本身就够定位窗口/池口径）。
    _force_hit = os.environ.get("DS41M_HIT_REAL_FORCE", "").strip().lower() in ("1", "on", "true", "yes")
    real_prefix = (bool(hit_cfg.get("ratio", 0))
                   and str(hit_cfg.get("source", "fake")).lower() == "real"
                   and (not a.layers or _force_hit))
    if real_prefix and (cfg.get("weights") or {}).get("mode", "synthetic") == "synthetic":
        print("  [hit-real] ⚠️ 合成权重 + hit.source=real：合成权重各层同源，真前缀没有数值意义"
              "（前缀池本就是共享的零值），建议用 hit.source=fake")
    _shared = {"_win_cache_A": None, "_win_cache_B": None, "_cmp_cache": None} if real_prefix else None
    if real_prefix:
        _stack0, _meta0 = assemble.build_stack(cfg)
        _L, _rep = int(_meta0["L"]), assemble.replay_tail_of(cfg)
        _M = max(1, int(_meta0["T_main"]) - _rep)            # 真正新增的 token 数
        H = max(0, _L - _M)                                  # 前缀长度（stage A 真算的）
        if inputs is None:                                   # 同一串 token 给两段：
            inputs = make_input(cfg, _L)                     # stage B 的 replay 必须与 stage A 尾部同源
        print(f"  [hit-real] stage A：真算前缀 H={H}（占位 T_main={_meta0['T_main']}, replay={_rep}）")
        t_a = time.time()
        # ★ 生产口径：stage A 必须跑**完整层数**（n_layers=None）——前缀池要覆盖所有层。
        #   只有 `DS41M_HIT_REAL_FORCE=1`（纯调试：--layers N 快速迭代）时才跟着截断，
        #   否则 stage A 照样跑满 40 层 → 照样撞上病 C 的崩溃层，调试白费。
        run_forward(cfg, source, H, (a.layers or None) if _force_hit else None, None,
                    allow_placeholder=a.allow_placeholder, prog=prog, inputs=inputs,
                    input_ids=(ids if (a.prompt or a.ids_file) else None),
                    shared=_shared, hit_stage="A")
        print(f"  [hit-real] stage A 完成 {time.time() - t_a:.1f}s"
              f"（前缀 KV/压缩 KV/索引缓存/压缩器 state 已入共享池，不计入 profile）")
        if _RANK == 0:
            # DeepInfer `stats_logger` 同式：cached / (cached + batched)。
            # batched = non_cached token 数（M）；replay tail 是重建窗口的额外行，不计入
            # “本步 batched tokens”口径。
            print(f"  [hit] prefill_caching_rate (GPU {100.0 * H / max(1, H + _M):.2f}%)"
                  f"  cached={H} batched={_M} replay_tail={_rep}", flush=True)
    # ★ 输入只造一次（以前每轮 run_forward 里 make_input(cfg, T) 现造 [65536,5120] 的
    #   随机输入：fp32 randn 1.3GB + 转 bf16 + H2D ≈ **3.8 s/轮**，占稳态 pass 的 92%！
    #   命中模式只用尾部 2048 行，更是白造。hoist 到循环外 → 各轮同输入（也和官方 bench
    #   "同一份 packed 输入重复跑"的口径一致）。
    # ★ 按**实际用量**造：命中 fake 前缀时 run_forward 只吃尾部 T_main 行（`x[-T_main:]`），
    #   L=1M 那些点上按 L 造就是 1M×5120 的 fp32 随机数（20 GB + 转 bf16 + H2D，几十秒）纯浪费。
    _T_buf = min(a.seq, 1 << 20)
    if hit_cfg.get("ratio", 0) and not real_prefix:
        _T_buf = assemble.hit_m(cfg, int(a.seq)) + assemble.replay_tail_of(cfg)
    # ★ CP × DP（2026-10-25）：world>cp 时每个 CP 组是一个 DP 组。
    #   合成输入按组给不同 seed，否则各 DP 组会算同一份数据（假 DP）；
    #   组内所有 rank 必须同 seed（同一条序列被拆到组内各 rank）。
    #   `DS41M_CP_DP_INPUT=1` 显式开启；固定 ids/prompt 时不改 seed（同源口径）。
    _cp_size_dp = int(os.environ.get("DS41M_CP_SIZE", "1") or 1)
    _dp_input = os.environ.get("DS41M_CP_DP_INPUT", "").strip().lower() in (
        "1", "on", "true", "yes")
    _seed_base = int(getattr(a, "input_seed", 0) or 0)
    if _dp_input and _cp_size_dp > 1 and int(a.world) > _cp_size_dp \
            and not (a.prompt or a.ids_file):
        _dp_group = _RANK // _cp_size_dp
        _seed0 = _seed_base + _dp_group * 1000003
        print(f"  [cp-dp] world={a.world} cp={_cp_size_dp} → "
              f"DP{int(a.world) // _cp_size_dp}；本组 rank={_RANK} seed={_seed0}",
              flush=True)
    else:
        _seed0 = _seed_base
    if inputs is None and not a.layers:
        inputs = make_input(cfg, _T_buf, seed=_seed0)
        print(f"  输入缓冲已建：{tuple(inputs.shape)} {inputs.dtype}（各轮复用）", flush=True)
    # ── DP 双 batch（--batches N）────────────────────────────────────────────
    #   每个 iteration 顺序跑 N 个 batch，各带一份**独立**输入（不同 seed）。
    #   一个 trace 里是 N 段完整 pass —— 官方 DP 双 batch 的"两个 batch 先后执行"口径。
    #   ★ 不是一次前向 batch=N：mini 的 attention/indexer/池都是单序列，真 batch 要改算子契约。
    _nb = max(1, int(getattr(a, "batches", 1) or 1))     # 用户口径的 batch 数
    _npass = 1 if _packed else _nb                       # 每次 run_forward 的 pass 数
    # ★ 逐序列 seed 可显式指定（序列隔离对拍用：packed [A,B] vs [A,C] 里 A 的 seed 必须相同）
    _seed_env = [int(v) for v in os.environ.get("DS41M_PACKED_SEEDS", "").replace(",", " ").split()]
    def _seq_seed(b):
        return _seed_env[b] if b < len(_seed_env) else _seed0 + 7919 * b
    if inputs is not None and not a.layers:
        if a.prompt or a.ids_file or real_prefix:
            # ★ 真前缀（stage A）与固定 ids：stage B 必须与 stage A **同源**，
            #   各 batch 只能用同一份输入（换 seed 会破坏命中数值口径）。
            # ★ packed + ids：inputs 已按 [offset: offset+nseq*seq] 切好（包含所有序列），
            #   不能再复制 _nb 份，否则拼接后变成 2×T。
            _seqs_in = [inputs] * (1 if _packed else _nb)
            print(f"  [dp-batch] batches={_nb}（固定 ids/真前缀：各 batch 输入相同）", flush=True)
        else:
            _seqs_in = [make_input(cfg, _T_buf, device=inputs.device,
                                   seed=_seq_seed(b)) for b in range(_nb)]
            print(f"  [dp-batch] batches={_nb}：每个 {_T_buf} 行"
                  f"（独立随机输入，seed={[_seq_seed(b) for b in range(_nb)]}）", flush=True)
    else:
        if _packed:
            # packed + 无 ids：逐 batch 造**确定性**合成输入再拼接（--input-seed 固定）。
            # 不能走 run_forward 内建：那一路单序列建 [T,H]，拼不起来。
            _seqs_in = [make_input(cfg, _T_buf, seed=_seq_seed(b))
                        for b in range(_nb)]
            print(f"  [packed] 合成输入：{_nb} × {_T_buf} 行"
                  f"（seed={[_seq_seed(b) for b in range(_nb)]}）", flush=True)
        else:
            _seqs_in = [None] * _nb
            print(f"  [dp-batch] batches={_nb}（输入由 run_forward 内建）", flush=True)
    if _packed:
        _batch_inputs = [torch.cat(_seqs_in, dim=0)]
        print(f"  [packed] 输入拼接：{tuple(_batch_inputs[0].shape)}"
              f"（{_nb} × {_T_buf}）", flush=True)
    else:
        _batch_inputs = _seqs_in
    _T_run = int(_T_buf) * (_nb if _packed else 1)   # packed：每序列 R 行 × N（命中 R<L）
    # 本次口径一行打全（采完看日志就能确认当时是什么设置，不用回头猜）
    _moe_bar = bool(os.environ.get("DS41M_MOE_BARRIER"))
    _pass_bar = os.environ.get("DS41M_PASS_BARRIER", "").strip().lower() in ("1", "on", "true", "yes")
    # ★ 强均路由的真实判据在 exec_ops.py:452：`ctx["input_ids"] is None` 才启 —— 给了真 ids
    #   （--prompt/--ids-file）就是**自然路由**。只看环境变量会在真权重跑时误报 ON。
    _ids_given = bool(a.prompt or a.ids_file)
    if _ids_given:
        _elb = "OFF(给了输入 ids → 自然路由)"
    elif os.environ.get("DS41M_NO_BALROUTE"):
        _elb = "OFF(DS41M_NO_BALROUTE)"
    else:
        _elb = "ON(无 ids → 强均路由，等价官方 force_eplb)"
    print(f"  口径：elb/强均路由={_elb}"
          f"；跨 rank barrier={'每层 mega_moe 前(DS41M_MOE_BARRIER 调试)' if _moe_bar else ('每 pass 前(DS41M_PASS_BARRIER)' if _pass_bar else '仅起跑线一次')}"
          f"；weights={ (cfg.get('weights') or {}).get('mode', 'synthetic')}"
          f"；hit={hit_cfg.get('ratio', 0)}/{hit_cfg.get('source', '-')}"
          f"；prof={'on' if a.prof else 'off'}", flush=True)
    # ★ 预热轮（--warmup-iters K）：真权重首 pass 含权重装载/H2D，**别写进 trace**。
    #   跑在起跑线 barrier 与 profiler.start() **之前** → step() 根本不计入，不需要 skip_first
    #   （两者叠加会多扔一轮实测 pass：实测 iters=3 只录到 2 轮）。
    for _w in range(max(0, a.warmup_iters)):
        print(f"  [warmup {_w + 1}/{a.warmup_iters}] 预跑一轮（不计入 trace）", flush=True)
        for _b in range(_npass):
            run_prefill_auto(cfg, source, _T_run, a.layers or None, None,
                             allow_placeholder=a.allow_placeholder, prog=prog,
                        inputs=_batch_inputs[_b],
                        input_ids=(ids if (a.prompt or a.ids_file) else None),
                        shared=_shared, hit_stage=("B" if real_prefix else None))
    # ★ 2026-10-21 用户口径：预热先**排空设备**，再开采；
    #   否则预热尾部还在设备队列里，trace 的第一个 pass 会混入前一轮的尾巴。
    torch.npu.synchronize()
    print(f"  [warmup] {max(0, a.warmup_iters)} 轮预热完成 → "
          f"torch.npu.synchronize() 已排空，开始采集", flush=True)
    # ★ 起跑线（2026-10-15 用户口径）：**只在最最开始**同步这一次，**然后**才开采。
    #   采集区内（含 mega_moe 前后）不插任何 barrier；--prof 本身也不插（旧行为已删）。
    if start_line_sync():
        print("  [起跑线] 跨 rank barrier ×1 完成 → 现在开 profiler 开采", flush=True)
    else:
        print("  [起跑线] 未做跨 rank barrier（单卡 / DS41M_START_NOBARRIER=1）", flush=True)
    if profiler is not None:
        # ALOP start() 需要 device ID（int）
        _dev_id = 0
        _dev_str = cfg["run"].get("device", "npu:0")
        if ":" in _dev_str:
            _dev_id = int(_dev_str.split(":")[-1])
        profiler.start(device=_dev_id)
    r = None
    it_secs = []
    _pipe = bool(os.environ.get("DS41M_PIPELINE")) and iters > 1
    if _pipe:
        print("  [计时口径] 流水模式：forward 之间不排空（对齐官方 hit-inner-reps 的取数方式）", flush=True)
    if _pass_bar:
        print("  [计时口径] 每 forward 前 barrier+排空（DS41M_PASS_BARRIER）：等待不计入 forward 时长", flush=True)
        if _pipe:
            print("  ⚠️ DS41M_PASS_BARRIER 与 DS41M_PIPELINE 同开：barrier 会把流水排空（口径互斥，建议只留其一）", flush=True)
    _t_pipe = time.time()
    for it in range(iters):
        if it > 0 and _pass_bar:            # ★ 边界 barrier：在计时起点之前 → 等待不进 pass 的账
            pass_barrier_sync(it)
        t_it = time.time()
        for _b in range(_npass):            # ★ packed=1 个 packed pass；sequential=N 个独立 pass
            _t_b = time.time()
            r = run_prefill_auto(cfg, source, _T_run, a.layers or None, profiler,
                                 allow_placeholder=a.allow_placeholder, prog=prog,
                            inputs=_batch_inputs[_b],
                            collect_dir=(a.collect_layers if _collect_here else ""),
                            input_ids=(ids if (a.prompt or a.ids_file) else None),
                            stop_profiler=(it == iters - 1 and _b == _nb - 1),
                            shared=_shared, hit_stage=("B" if real_prefix else None),
                            no_sync=(_pipe and (it < iters - 1 or _b < _nb - 1)))
            # ★ 2026-10-21 pure：测量区间里**不打印**（每 batch/每轮一次 f-string + flush
            #   是无必要的 host 操作；trace 轮里还会在 pass 边界制造假空泡）。
            if _nb > 1 and not ds41_log.pure():
                print(f"  [dp-batch] L={_T_run} "
                      f"batch={_b + 1}/{_nb} {time.time() - _t_b:.3f}s", flush=True)
        it_secs.append(time.time() - t_it)
        if iters > 1 and not ds41_log.pure():
            if _nb > 1:
                _per = it_secs[-1] / _nb
                print(f"  [iter {it + 1}/{iters}] {it_secs[-1]:.3f}s"
                      f"（{_per:.3f}s/batch，{_per * 1000 / max(1, a.layers or n_stack):.2f} ms/层）",
                      flush=True)
            else:
                print(f"  [iter {it + 1}/{iters}] {it_secs[-1]:.3f}s"
                      f"（{it_secs[-1] * 1000 / max(1, a.layers or n_stack):.2f} ms/层）", flush=True)
    if _pipe:
        torch.npu.synchronize()
        tot = time.time() - _t_pipe
        print(f"  [计时口径] 流水：{iters} 轮合计 {tot:.3f}s → 稳态均 {tot / iters * 1000:.0f} ms/forward"
              f"（首轮含权重装载，官方口径会把首轮排除）", flush=True)
    if _pass_bar:
        # ★ 收尾 barrier（2026-10-16）：最后一轮之后也对齐一次，杜绝"快的 rank 先退进程 →
        #   落后 rank 的 MoE/集合通信找不到邻居 → **硬死锁**"（1M 因 MoE 未分块 OOM 拖慢
        #   4 个 rank，先跑完的 4 个已退出 → 卡在 pass4 层 21-25 实测踩过 ✗）。
        #   位置在 profiler.stop() 之后、且不参与任何计时 → 不进 trace、不改任何 pass 的账 ✓。
        pass_barrier_sync(iters)
    if iters > 1:                       # 性能采集：只看稳态（PERF_RULES §0 第三条）
        # 稳态口径：剔除冷启首轮（权重装载/编译）。稳态样本 ≥3 取**中位数**。
        _excl = {0}
        warm = [s for i, s in enumerate(it_secs) if i not in _excl]
        if not warm:
            warm = it_secs[1:] or it_secs
        if len(warm) >= 3:
            import statistics as _stat
            _val = _stat.median(warm)
            _kind = "中位"
        else:
            _val = sum(warm) / len(warm)
            _kind = "均值"
        print(f"  多轮耗时：{['%.3f' % x for x in it_secs]}s  均(除首轮)={_val:.3f}s"
              f"  最快={min(it_secs):.3f}s"
              f"  [稳态 {len(warm)} 轮{_kind}；剔除轮={sorted(_excl)}]")
    meas = prog.end_stage()
    try:
        peak = torch.npu.max_memory_allocated() / 2 ** 30
        resv = torch.npu.memory_reserved() / 2 ** 30
        tot = torch.npu.get_device_properties(0).total_memory / 2 ** 30
        print(f"  显存：峰值 {peak:.2f} GiB / 保留 {resv:.2f} / 总 {tot:.1f} GiB"
              f"（峰值/T = {peak * 1024 / max(1, a.seq):.3f} MiB/token）")
    except Exception as exc:                                # noqa: BLE001
        print(f"  （显存读数失败：{exc}）")
    print(f"\n完成：{r['seconds']:.1f}s；执行 {sum(r['exec'].values())} 个算子调用 "
          f"（{len(r['exec'])} 种），跳过 {sum(r['skip'].values())} 个")
    print("  执行：", dict(r["exec"]))
    print("  跳过：", dict(r["skip"]))
    if r.get("implicit"):
        print("  隐式（由量化算子内部产生，不单独发）：", dict(r["implicit"]))
    if r["placeholder"]:
        print(f"  ⚠️ 用了占位输入的算子：{r['placeholder']}（数值无意义，仅用于打通算子图/出 trace）")
    if r["errors"]:
        print(f"  ⚠️ 出错 {len(r['errors'])} 处（前 5）：")
        for li, k, e in r["errors"][:5]:
            print(f"     层{li} {k}: {e[:150]}")
    if a.prof:
        if profiler is not None:
            profiler.finalize()
        print(f"  trace → {prof_dir}")
    if r["errors"]:
        # 有错就非零退出：让脚本/CI 不会把"跑完了"当成"跑对了"（EP 自洽那次就是被吞掉的）
        import collections as _c
        by_key = _c.Counter(k for _l, k, _e in r["errors"])
        if by_key and max(by_key.values()) >= 3:
            sys.exit(3)
    if (a.prompt or a.ids_file) and not a.layers:
        logits = head_logits(cfg, r["x"], r.get("pre_mix"), source)
        top = torch.topk(logits, 8)
        tok = _tokenizer(cfg)
        print("\n== 贪心下一个 token ==")
        for sc, tid in zip(top.values.tolist(), top.indices.tolist()):
            piece = tok.decode([tid])
            print(f"   {sc:9.3f}  id={tid:<7} {piece!r}")
        nxt = int(top.indices[0])
        import math
        p_ = torch.softmax(logits, dim=-1)
        ent = float(-(p_ * (p_ + 1e-12).log()).sum())
        print(f"  → 取 top1: {tok.decode([nxt])!r}（整句续写不做，只做单步贪心）")
        print(f"  分布诊断：熵 {ent:.2f} nats / 均匀分布熵 {math.log(logits.numel()):.2f} nats"
              f" → 归一化熵 {ent / math.log(logits.numel()):.3f}"
              f"（1.0 = 完全随机）；top1 概率 {float(p_.max()):.2e}")
        if a.dump_hidden:
            # pos_offset：这批 hidden 的**绝对起始位置**（命中 stage B = L-T_main；stage A = 0）
            # → tools/compare_tail.py 靠它把两次跑的同一段行对齐
            torch.save({"hidden": r["x"].cpu(), "logits": logits.cpu(),
                        "pos_offset": int(r.get("pos_offset", 0)),
                        "L": int(r["meta"].get("L", 0)),
                        "hit": float(r["meta"].get("hit", 0) or 0)}, a.dump_hidden)
            print(f"  末层 hidden + logits 已存 {a.dump_hidden}")
    if r.get("snap"):
        print("  ── 逐算子张量统计（layer 0/1，rank0；看哪一步 max 炸）──")
        for li, k, sh, mn, mx in r["snap"][:40]:
            flag = "  ← 爆" if mx > 50 else ""
            print(f"    L{li} {k:<20} {str(sh):<20} mean={mn:+.4e} absmax={mx:.4e}{flag}")
    if a.collect_layers:
        print(f"  [collect] 目标目录 {a.collect_layers}；录到 {len(r.get('collected') or {})} 层")
    if a.collect_layers and r.get("collected"):
        import atexit
        def _save():
            os.makedirs(a.collect_layers, exist_ok=True)
            p = os.path.join(a.collect_layers, f"collect_rank{_RANK}.pt")
            torch.save({"layers": r["collected"], "rank": _RANK, "n": len(r["collected"])}, p)
            print(f"  逐层 dump → {p}（{len(r['collected'])} 层）")
        d = r["collected"]
        os.makedirs(a.collect_layers, exist_ok=True)
        torch.save({"layers": d, "rank": _RANK, "n": len(d)},
                   os.path.join(a.collect_layers, f"collect_rank{_RANK}.pt"))
        print(f"  逐层 dump → {a.collect_layers}/collect_rank{_RANK}.pt（{len(d)} 层）")
    if r["errors"]:
        import collections as _c
        by_key = _c.Counter(k for _l, k, _e in r["errors"])
        import collections as _cc
        layer_counts = _c.Counter(k for _l, k, _e in r["errors"])
        n_layers_run = len(as_done := {li for li, _k, _e in r["errors"]}) or 1
        sys_fail = {k: v for k, v in by_key.items() if v >= max(3, 0.5 * (a.layers or model_spec.get().n_layers))}
        if sys_fail:
            print(f"  ⛔ 系统性失败（在多数层上失败，多半是权重口径/上下文错，不是数值问题）："
                  f"{sys_fail}")
            print(f"     首条报错：{r['errors'][0][2][:200]}")
    print("  阶段实测（用于校准 progress.py 的权重）：")
    for k, v in meas.items():
        print(f"    {k:<22} {v:6.1f}s")


if __name__ == "__main__":
    main()

# ── 点位边界（core/memlife.py）──────────────────────────────────────────────
# 进程级几何缓存：ctx（T/win_abs/池地址）/ 算子序列 / 模块计划。
# CP 的 plan 固定缓冲（`plan._rt` 里的 gather full/g）**按 L 分配**，是天量级
# （1M/CP8 单块 ≈1 GiB，4 个 plan 槽就是十几 GiB）⇒ 点位边界必须连缓冲一起放。
POINT_CACHES = (_CTX_CACHE, _SEQ_CACHE, _MODULE_PLAN_CACHE)


def POINT_RELEASE():                                     # noqa: N802 —— memlife 约定名
    try:
        import cp_runtime as _cpr
        while _CP_STATE_CACHE:
            _old_key, _old = _CP_STATE_CACHE.popitem(last=False)
            try:
                _cpr.release_plan_buffers(_old.plan)
            except Exception:                            # noqa: BLE001
                pass
    except Exception:                                    # noqa: BLE001
        pass

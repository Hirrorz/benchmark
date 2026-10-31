"""idx_fastcall.py —— 索引器 4 个厂商算子的「薄下发」通道（A6，默认关）。

对齐 `ops/mq_fastcall.py`（A1）的三级结构，对象换成索引器三族厂商算子：

    quant_lightning_indexer            (QLI)
    quant_sparse_lightning_indexer     (QSLI)
    indexer_prologue_qw / _k           (prologue)

# 0. 口径（先说清楚哪些**省不了**）

这 4 个算子都是 **cannbotdsl JIT 算子**：`torch.ops.cann_ops_transformer.ds41.<op>` 的
`PrivateUse1` impl 只是薄壳，真正的工作在 `ds41_runtime/python/ops/<op>.py`（vendor 尾部）
+ `cannbotdsl` 的 provider 调用桥。cProfile（4 forward，
`tmp/t042/t042_prof/ds41m_host_r0.prof`）里每层栈的**单位成本**：

    indexer_prologue_qw      49.4 ms / 32 次  = 1.54 ms/次（cum，含子帧）
    quant_sparse_lightning   34.5 ms / 16 次  = 2.16 ms/次（cum，含子帧）
    quant_lightning_indexer  32.9 ms / 16 次  = 2.06 ms/次（cum，含子帧）
    indexer_prologue_k       22.4 ms / 20 次  = 1.12 ms/次（cum，含子帧）
    invoke_provider_call    163.3 ms / 244 次 = 0.67 ms/invoke（tt 仅 3.4 µs）
    resolve_provider_call   121.4 ms / 244 次 = 0.50 ms/invoke
    export_tensor          3376 次 / 32.6 ms  = 9.7 µs/张量
    arg_check.wrapper       828 次 / 42.5 ms

⇒ **我们自己的 Python 胶水**（组 kwargs、`d = torch.ops...` 属性链、`.contiguous()`）
只有几 µs/次，**不是**大头；大头在 provider 桥里（每 invoke ~0.8 ms，其中 Python 侧
~0.65 ms + 原生 `_host_runtime.invoke` ~0.17 ms）。因此：
- 「原生 invoke 那 0.17 ms」**本文件不碰**（要在 C++ 里改，且不是 Python 侧可省项）；
- `inspect_storage`（每张量一次 `torch_npu.get_npu_format`，C++）**不做 memo**：
  `npu_format_cast_` 是**原地**改格式，任何「格式不变」的假设都不可证 ⇒ 保留 vendor 实现；
- `resolve_stream`（`torch.npu.device()` + `current_stream`）**不做 memo**：流会在
  图捕获/侧流之间切换，缓存即错。这两项合计 ~0.2 ms/invoke 留在原地。

# 1. 三级通道（默认 L0 = 与本文件引入前逐字节相同）

  DS41M_IDX_FAST=0（默认）  L0  原样 `torch.ops.cann_ops_transformer.ds41.<op>(...)`
  DS41M_IDX_FAST=1          L1  预绑定 impl 函数（跳过 torch 派发器 + 命名空间属性链）。
                                下行代码与 L0 完全相同，**零数值风险**
                                （同一个 impl 函数、同一组实参）。
  DS41M_IDX_FAST=2          L2  L1 + provider 桥的「静态编组」缓存：
                                a) 跳过 `ParameterLowering.bind_physical` 的整条循环
                                   （n≈35 个槽位逐个 `abi_arguments[i]` + 契约相等比较 +
                                   标量校验 ⇒ ~0.11 ms/invoke）；
                                b) `provider_tensor_items` / `provider_tensor_contract_items`
                                   结构常量缓存（名字表 + 契约元组，逐次对比实参身份）；
                                c) `TorchNPUIntegration.export_tensor` 的**指纹**缓存
                                   （布局/dtype/flags 不变就直接复用 layout + dlpack mode）。

# 2. 为什么 L2 的每一环都能证明「逐位不变」

`bind_physical`（`parameter_lowering.py:177`）在
`layout.positional_identity and not kwargs and len(args)==n` 时**直接 return args**
（`parameter_lowering.py:180-187,246-249`）。这 4 个 kernel 的 vendor 尾部都是**全位置实参**
调用 `compiled(...)`（`ops/indexer_prologue_qw.py:2604` 15 个、
`ops/quant_lightning_indexer_dsl.py:4650` 35 个），所以快路成立。

但我们**不假设**它成立：L2 第一次碰到某个 `ProviderCallable` 时（`_build`）会
**真跑一遍 vendor 的 `bind_physical` / `provider_tensor_items` /
`provider_tensor_contract_items`**，要求 `bind_physical(args, {}) is args` **且**
我复刻的 items/contracts 与之**逐项同身份**；任一条不满足就**永久放弃**这个 callable
（`_ST[id] = (self, None)`，只花一次）。
=> 「没验证」在这里变成「**挂载时已与 vendor 逐位对齐**」。

`export_tensor`（`torch_npu_integration.py:327`）的输出是
`_TensorExport(original, exporter, device, layout, dlpack_call_mode)`，
其中 `layout=(data_ptr, shape, stride, dtype三元组)`，`exporter` 只由
`requires_grad` 与 dtype 决定（fp8 家族才 `.view(int8)`），另有
`_validate_tensor_semantics`（`is_neg`/`is_conj`/`layout is strided`）。
指纹 `_fp()` 把 **data_ptr/shape/stride/dtype/storage_offset/requires_grad/
is_neg/is_conj** 全取进来 ⇒ 指纹相同 ⇒ 字段必然相同。
缓存里**不放任何张量**（只放 layout/mode/是否要 int8 视图），每次调用重建
`_TensorExport` ⇒ **不钉住任何显存**（每层新分配的输出也不会泄漏）。

# 3. 失败了怎么办

`_build` 抛 `_Skip` ⇒ 该 callable 永远走 vendor 原路（不打印、不重试）。
调用点 `_go()` 的 L2→L1→L0 回退严格照抄 `mq_fastcall.mqsmla()`（打印一行 `[idx-fast]`）。
桥内快路只在 `_ARM_DEPTH > 0` 且已挂载时生效 ⇒ **没被 L2 调过的算子（mqsmla 等）
一个都不受影响**。

# 4. 统计

`DS41M_IDX_FAST_STATS=1` 退出时打印：各级次数、回退次数、桥挂载成功/放弃、
export 指纹命中率。

# 4.1 「复用 kwargs dict」经实测是**净亏**，故不采用

CPU 微基准（`/tmp/a6_bench_kw.py`，11 个关键字，30 万次，本机无卡）：

    字面量 kwargs 直调  f(a=1, ..., k=11)          0.435 µs/次   ← 现状
    复用 dict + 11 次赋值  d["a"]=1; ...; f(**d)  0.908 µs/次   ← 更慢
    复用 dict 零赋值      f(**D)                   0.636 µs/次   ← 仍更慢
    现场 dict(...) + f(**d)                        1.027 µs/次

原因：`f(*a, **d)` 在 C 层**必然**为目标函数重建一个 dict，复用只能在「不再展开」
时才有意义。所以 L1 的收益**只**来自跳过 torch 派发器与命名空间属性链。
代价：走本模块的 L0 路径比原始直调多 ~0.6 µs/次（≈12 µs/forward），可忽略。

# 5. 预期（host，不含原生 invoke）—— 全部来自本机 CPU 微基准 `tmp/a6_bench/bench_bridge.py`

同一份**真实** `ParameterLoweringPlan`（35 槽 = 20 张量 + 15 标量，与 QLI 的
`compiled(...)` 同形），纯 Python 编组耗时：

    vendor: bind_physical + items + contracts                     330.4 µs/次
    vendor + fast_abi（main 默认开）                               93.5 µs/次   ← 现状
    快路（本文件）：20 次类型守卫 + items 重建                        1.9 µs/次

⇒ **L2 在 fast_abi 之上再省 ≈ 92 µs/invoke**（没有 fast_abi 时是 328 µs）。
省下的量与槽数近似线性（同一台机、同样本，`/tmp/a6_slot_scale.py` 实测）：

    槽位  35（QLI）  22（QSLI）  15（prologue_qw）  11（prologue_k）
    省 µs      93.8       61.6            43.3            30.7

按 cProfile 的 21 次/invoke-per-forward（qw 8 + k 5 + qli 4 + qsli 4，见 §0）
折算 ⇒ 桥编组部分 **≈1.12 ms/forward**；`export_tensor`（CPU 侧 Python 部分
bf16 3.30 → 1.90 µs、fp8 4.96 → 2.54 µs，约 14 张量/invoke）再 **0.4~0.7 ms/forward**
⇒ **合计 ≈1.1~1.9 ms/forward**。L1 另加 ≈0.3 ms/forward
（15 µs/次，沿用 A1 `tmp/a1_bench/bench_dispatch.py` 的同级 schema 实测）。

**这 21 次里没有一次是 C++ 内部**：`cannbotdsl._host_runtime.invoke` 那 ~0.17 ms/invoke
本文件一分都省不到（在 C++ 里）。
只有 **host-bound 的回放尾段**能把这部分兑现成 wall（见
`NOTES/2026-10-26-a4-deepinfer-host.md` §3）；头段 host 被设备完全盖住。
"""
from __future__ import annotations

import atexit
import os
import sys

import torch

# ─────────────────────────── 开关 ───────────────────────────
_LVL = None


def level() -> int:
    global _LVL
    if _LVL is None:
        # ★ 2026-10-26（T-055，本机 8 卡）：默认 **2**（L2）。
        #   交错 6 臂（各 8 轮）：L0 108.22/108.05 vs L2 **106.28/107.47** ⇒ **−1.26 ms**，
        #   L1 108.88（无收益）；六臂 `[packed-digest]` 全同 `08443723284dd9ed`。
        #   快路统计：`l2=189 回退=0 | 桥: 挂上=6 快路=183 原路=6 | export 命中=1612`。
        #   `DS41M_IDX_FAST=0` 可回退到厂商原路。
        v = os.environ.get("DS41M_IDX_FAST", "2").strip()
        _LVL = int(v) if v in ("0", "1", "2") else 0
    return _LVL


_STATS_ON = os.environ.get("DS41M_IDX_FAST_STATS", "").strip().lower() in (
    "1", "on", "true", "yes")

_OPS = ("qli", "qsli", "prolog_qw", "prolog_k")

_FULL = {
    "qli": "quant_lightning_indexer",
    "qsli": "quant_sparse_lightning_indexer",
    "prolog_qw": "indexer_prologue_qw",
    "prolog_k": "indexer_prologue_k",
}

# impl 文件在 sys.modules 里的唯一后缀（不能按点号 import：vendor 包没有稳定
# __init__ 链，与 A1 `mq_fastcall._scan` 同一做法）。注意取的是 **内层 .py**
# 里的 `@impl(...,"PrivateUse1")` 函数，不是 `_dsl/__init__.py` 里的转发壳。
_IMPL_SUFFIX = {
    "qli": "/quant_lightning_indexer_dsl/quant_lightning_indexer.py",
    "qsli": "/quant_sparse_lightning_indexer_dsl/quant_sparse_lightning_indexer.py",
    "prolog_qw": "/indexer_prologue_qw_dsl/indexer_prologue_qw.py",
    "prolog_k": "/indexer_prologue_k_dsl/indexer_prologue_k.py",
}

_STATS = {
    "l0": 0, "l1": 0, "l2": 0, "fallback": 0, "impl_miss": 0,
    "arm_ok": 0, "arm_skip": 0, "fast": 0, "vendor": 0,
    "exp_hit": 0, "exp_miss": 0,
}
for _o in _OPS:
    _STATS["n_" + _o] = 0


def stats() -> dict:
    return dict(_STATS, level=level())


# ─────────────────────────── vendor 入口定位 ───────────────────────────
_PKT: dict[str, object] = {}
_IMPL: dict[str, object] = {}
_NONE = object()


def _scan(suffix):
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None) or ""
        if f.endswith(suffix):
            return m
    return None


def _packet(short):
    """预绑定的 `torch.ops.cann_ops_transformer.ds41.<op>.default`。"""
    fn = _PKT.get(short)
    if fn is None:
        pk = getattr(torch.ops.cann_ops_transformer.ds41, _FULL[short])
        fn = getattr(pk, "default", pk)
        _PKT[short] = fn
    return fn


def _impl(short):
    """vendor 的 `@impl(..., "PrivateUse1")` 函数本身（派发器最终调的就是它）。"""
    fn = _IMPL.get(short)
    if fn is None:
        m = _scan(_IMPL_SUFFIX[short])
        if m is None:                      # 模块还没进 sys.modules：下次再找，不钉死
            return None
        fn = _NONE
        cand = getattr(m, _FULL[short], None)
        if callable(cand):
            fn = cand
        _IMPL[short] = fn
    return None if fn is _NONE else fn


# ─────────────────────────── L0 ───────────────────────────
def _l0(short, args, kwd):
    d = torch.ops.cann_ops_transformer.ds41
    return getattr(d, _FULL[short])(*args, **kwd)


# ─────────────────────────── L1 ───────────────────────────
def _l1(short, args, kwd):
    return _packet(short)(*args, **kwd)


# ─────────────────────────── L2：provider 桥 ───────────────────────────
class _Skip(Exception):
    """快路不可用（不是算子本身的错）→ 静默回退。"""


class _St:
    __slots__ = ("n", "pairs", "contracts")

    def __init__(self, n, pairs, contracts):
        self.n = n
        self.pairs = pairs            # tuple[(slot_name, arg_index)]
        self.contracts = contracts    # tuple[Contract]，与 tensor_items 一一对应


_TT = torch.Tensor
_INT8_DTYPES = (10, 12, 14)       # vendor export_tensor 里要走 int8 视图的 dlpack 码

_ARM_DEPTH = 0
_ACTIVE = False

_ORIG_CALL = None
_ORIG_EXPORT = None
_TENSOR_EXPORT = None
_BRIDGE_TRIED = False
_BRIDGE_OK = False
_RB = None
_ST: dict[int, tuple] = {}        # id(callable) -> (强引用, _St | None)


def _build(self, args, kwargs):
    """与 vendor 逐位对齐地重建静态编组；任何一条不成立就永久放弃。"""
    if kwargs:
        raise _Skip("kwargs")
    if getattr(self, "_debug_plans", ()):
        raise _Skip("debug_plans")
    if getattr(self, "_noop", False):
        raise _Skip("noop")
    L = self._parameter_lowering
    lay = L.layout
    if not lay.positional_identity:
        raise _Skip("positional_identity=False")
    if lay.struct_ref_indices:
        raise _Skip("struct_ref")
    n = lay.public_slot_count
    if len(args) != n:
        raise _Skip("argc")
    # ★ 真跑一遍 vendor 的物理绑定，要求原样返回（含标量校验）
    phys = L.bind_physical(args, {})
    if phys is not args:
        raise _Skip("bind_physical not identity")
    abi = L.abi_arguments
    names = lay.physical_slot_names
    pairs = []
    contracts = []
    for i in lay.provider_slot_indices:
        pairs.append((names[i], i))
        contracts.append(abi[i])
    items_v = L.provider_tensor_items(phys)
    contracts_v = L.provider_tensor_contract_items(phys)
    # TensorList 会被 vendor 摊平 ⇒ 长度/身份都对不上，这里一并挡掉
    if len(items_v) != len(pairs) or len(contracts_v) != len(contracts):
        raise _Skip("tensor_list")
    for k, (name, i) in enumerate(pairs):
        vn, vv = items_v[k]
        if vn != name or vv is not args[i]:
            raise _Skip("items mismatch")
        if contracts_v[k] is not contracts[k]:
            raise _Skip("contracts mismatch")
        if type(args[i]) is not _TT:
            raise _Skip("not exact torch.Tensor")
    return _St(n, tuple(pairs), tuple(contracts))


def _install_bridge():
    """把 ProviderCallable.__call__ / export_tensor 换成带缓存的版本。幂等、失败静默。"""
    global _BRIDGE_TRIED, _BRIDGE_OK, _ORIG_CALL, _ORIG_EXPORT
    global _TENSOR_EXPORT, _RB
    if _BRIDGE_TRIED:
        return _BRIDGE_OK
    _BRIDGE_TRIED = True
    try:
        from cannbotdsl.core.compiler import function as _fn
        from cannbotdsl.core.compiler import runtime_bridge as _rb
        from cannbotdsl.core.compiler import torch_npu_integration as _ti

        PC = _fn.ProviderCallable
        cur = PC.__call__
        if getattr(cur, "_a6_idx", False):                 # 已装过（重复 import 情形）
            _ORIG_CALL = cur
            _RB = _rb
            _BRIDGE_OK = True
            return True
        _ORIG_CALL = cur
        _ORIG_EXPORT = _ti.TorchNPUIntegration.export_tensor
        _TENSOR_EXPORT = _ti._TensorExport
        _call._a6_idx = True
        _export_tensor._a6_idx = True
        PC.__call__ = _call
        _ti.TorchNPUIntegration.export_tensor = _export_tensor
        _RB = _rb
        _BRIDGE_OK = True
    except Exception:                                      # noqa: BLE001  vendor 改名/缺失 → 不动
        _BRIDGE_OK = False
    return _BRIDGE_OK


def _call(self, *args, **kwargs):
    """ProviderCallable.__call__ 的快路版：非本通道的调用一字不改。"""
    global _ACTIVE
    if _ACTIVE or _ARM_DEPTH <= 0:
        return _ORIG_CALL(self, *args, **kwargs)
    key = id(self)
    ent = _ST.get(key)
    if ent is None:
        out = _ORIG_CALL(self, *args, **kwargs)            # 首次：原路 + 建状态
        _STATS["vendor"] += 1
        try:
            st = _build(self, args, kwargs)
        except _Skip:
            st = None
            _STATS["arm_skip"] += 1
        except Exception:                                  # noqa: BLE001 绝不影响已完成的调用
            st = None
            _STATS["arm_skip"] += 1
        else:
            _STATS["arm_ok"] += 1
        _ST[key] = (self, st)
        return out
    if ent[0] is not self or ent[1] is None:               # id 复用 / 已放弃
        return _ORIG_CALL(self, *args, **kwargs)
    st = ent[1]
    # ── 每次调用的实参守卫（全在 invoke 之前，失败不产生任何副作用）──
    if kwargs or len(args) != st.n:
        return _ORIG_CALL(self, *args, **kwargs)
    items = []
    for name, i in st.pairs:
        v = args[i]
        if type(v) is not _TT:
            return _ORIG_CALL(self, *args, **kwargs)
        items.append((name, v))
    plan = self._acquire_plan()
    _ACTIVE = True
    try:
        _RB.invoke_provider_call(
            plan, args, tuple(items), self._debug_plans,
            struct_ref_items=(), expected_contracts=st.contracts)
    finally:
        _ACTIVE = False
    _STATS["fast"] += 1
    return None


# ── export_tensor 指纹缓存（缓存里不放张量）────────────────────────────
_FP_MEMO: dict[tuple, tuple] = {}
_FP_CAP = 4096


def _fp(tensor):
    return (int(tensor.data_ptr()), tensor.shape, tensor.stride(),
            tensor.dtype, tensor.storage_offset(), tensor.requires_grad,
            tensor.is_neg(), tensor.is_conj())


def _export_tensor(self, tensor, device):
    if not _ACTIVE or type(tensor) is not self._torch.Tensor:
        return _ORIG_EXPORT(self, tensor, device)
    key = _fp(tensor)
    ent = _FP_MEMO.get(key)
    if ent is not None and ent[0] == device:
        _STATS["exp_hit"] += 1
        layout, mode = ent[1], ent[2]
        exporter = tensor.detach() if key[5] else tensor
        if layout[3][0] in _INT8_DTYPES:
            exporter = exporter.view(self._torch.int8)
        return _TENSOR_EXPORT(tensor, exporter, device, layout, mode)
    rec = _ORIG_EXPORT(self, tensor, device)               # 负位/共轭/非 strided 由 vendor 抛
    _STATS["exp_miss"] += 1
    if len(_FP_MEMO) < _FP_CAP:
        _FP_MEMO[key] = (device, rec.layout, rec.dlpack_call_mode)
    return rec


# ─────────────────────────── 公共入口 ───────────────────────────
def _l2(short, args, kwd):
    global _ARM_DEPTH
    fn = _impl(short)
    if fn is None:
        raise _Skip("impl 未加载")
    if not _BRIDGE_OK:
        _install_bridge()
    _ARM_DEPTH += 1
    try:
        return fn(*args, **kwd)
    finally:
        _ARM_DEPTH -= 1


def _go(short, args, kwd):
    _STATS["n_" + short] += 1
    lv = level()
    if lv >= 2:
        try:
            r = _l2(short, args, kwd)
            _STATS["l2"] += 1
            return r
        except _Skip:
            _STATS["impl_miss"] += 1
            lv = 1
        except Exception as e:                             # noqa: BLE001
            _STATS["fallback"] += 1
            print(f"  [idx-fast] L2 失败回退 L1 ({short}): {type(e).__name__}: {e}",
                  flush=True)
            lv = 1
    if lv >= 1:
        try:
            r = _l1(short, args, kwd)
            _STATS["l1"] += 1
            return r
        except Exception as e:                             # noqa: BLE001
            _STATS["fallback"] += 1
            print(f"  [idx-fast] L1 失败回退 L0 ({short}): {type(e).__name__}: {e}",
                  flush=True)
    _STATS["l0"] += 1
    return _l0(short, args, kwd)


def qli(args, kwd):
    """quant_lightning_indexer。`args` 位置实参元组，`kwd` 关键字 dict（可复用）。"""
    return _go("qli", args, kwd)


def qsli(args, kwd):
    """quant_sparse_lightning_indexer。"""
    return _go("qsli", args, kwd)


def prolog_qw(args, kwd):
    """indexer_prologue_qw。"""
    return _go("prolog_qw", args, kwd)


def prolog_k(args, kwd):
    """indexer_prologue_k（原地写 k_cache / k_scale_cache）。"""
    return _go("prolog_k", args, kwd)


# ─────────────────────────── 统计 / 自检辅助 ───────────────────────────
def _dump():
    lv = level()
    if lv <= 0 or not _STATS_ON:
        return
    calls = {o: _STATS["n_" + o] for o in _OPS}
    print(f"  [idx-fast] L={lv} l0={_STATS['l0']} l1={_STATS['l1']} l2={_STATS['l2']}"
          f" 回退={_STATS['fallback']} impl缺={_STATS['impl_miss']}"
          f" | 桥: 挂上={_STATS['arm_ok']} 放弃={_STATS['arm_skip']}"
          f" 快路={_STATS['fast']} 原路={_STATS['vendor']}"
          f" | export 命中={_STATS['exp_hit']} 未命中={_STATS['exp_miss']}"
          f" | 调用 {calls}", flush=True)


atexit.register(_dump)


def reset_caches() -> None:
    """自检/换臂用：清掉入口定位、挂载状态与缓存的 level（**不卸载** patch）。"""
    global _LVL
    _LVL = None
    _PKT.clear()
    _IMPL.clear()
    _ST.clear()
    _FP_MEMO.clear()

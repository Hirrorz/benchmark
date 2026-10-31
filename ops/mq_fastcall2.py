"""mq_fastcall2.py —— mqsmla 的「薄下发 L2」（B3；默认 **关**）。

结构照 `ops/idx_fastcall.py`（A6，索引器三族，实测 −1.26 ms/forward、189 次全走 L2、
零回退），但对象换成 mqsmla 主调用（40 次/forward、0.59 ms/次，其中 565 µs 在
cannbotdsl provider 桥里）。

# 0. 先修 A1 那版为什么是 +98 ms（证据见 NOTES/2026-10-26-b3-mq-l2.md §1）

    日志 tmp/AB_MQFAST_20261026_1354/fast2/logs/rank0.log：
      `[mq-fast] L2 失败回退 L1: AttributeError: 'NoneType' object has no attribute
       '_addr_vec_eligible'` —— **280 条 = 7 forward × 40 层，一次不漏**
      （同文件 `L1 失败回退 L0` **0 条** ⇒ L1 是好的，坏的只有 L2）。

    A1 的 L2 **一次都没跑起来**：它自己复刻了 vendor 尾部
    （`mq_fastcall.py:180-214`），于是要靠 `_tail()`（`mq_fastcall.py:119-131`）按
    `__file__` 后缀 `"/python/ops/mixed_quant_sparse_flash_mla.py"` 反查 vendor 模块 ——
    注意这个后缀要求路径里**有一层字面叫 `python` 的目录**。实测真实运行进程里这个模块
    是
        /mnt/.../ds41_runtime/pysite_full/usr/local/python3.11.16/lib/python3.11/
        site-packages/ops/mixed_quant_sparse_flash_mla.py
    （`endswith(...) == False`，见 bench/test_mq_fastcall2.py T8a 的字符串判据），
    路径里是 `python3.11` 而不是 `python` ⇒ 后缀永不命中。
    而 `_tail()` **不缓存否定结果** ⇒ 每次调用都重来一遍：
      (i) 全表扫 `sys.modules` —— 实测 492 µs（真实 PYTHONPATH + harness 载入后 2183 个
          模块）/ 523 µs（只 import 时 2296 个模块），折 ~0.23 µs/模块；
      (ii) **在失败之前**已经分配了 `out = torch.empty(q.shape, bf16)`（6.3 MB）和 `lse`
           （`mq_fastcall.py:191-192`）—— 每一个 forward 40 层 = 40 次白分配/白释放；
      (iii) `print(..., flush=True)`（`mq_fastcall.py:229-231`），8 rank 汇聚日志。
    观测口径：+97.6 ms/forward ÷ 40 = **+2.45 ms/次**；其中 (i) 占 ~0.5 ms/次
    （= ~20 ms/forward），其余 ~1.8 ms/次是 (ii)+(iii) —— 本机已知这个分配器在压力下
    会进 flush-retry 慢路径（NOTES 2026-10-26-hit95-large-l-moe-stall），但**没有卡就
    无法把 (ii) 与 (iii) 的分摊坐实**，§5 里如实标为推断。

    **结论：A1 的慢不是"缓存键不稳定"，是「定位器写死了源码树布局 + 否定结果不缓存 +
    失败点之前还白分配了两个设备张量 + 每条调用一次 flush 打印」。**
    本版的对应修法：**不在 vendor 调用之前分配任何张量**（分配留在 vendor impl 里，
    见 §2(1)），因此 (ii) 这一类成本结构上不存在。

# 1. 本版的三条修法（对应上面三处）

  a) **不再复刻 vendor 尾部**。L2 直接把 vendor 的 PrivateUse1 impl 当黑盒调用
     （与 A1 的 L1 同一条路，实测 digest 与基线全同），所以**根本不需要 `_tail()`**，
     也就不存在"按树路径找模块"这个坑。唯一要定位的是 impl 函数本身，用的是
     ``/mixed_quant_sparse_flash_mla_dsl/mixed_quant_sparse_flash_mla.py`` 后缀 —— 这个
     后缀在**源码树和 site-packages 两种布局下都成立**（A6 的四个索引器算子同款后缀，
     实测 189/189 命中）。
  b) **定位失败不逐调用重扫**：`_IMPL_STATE="dead"` 后 `level()` 直接返回 0 ⇒ 整条通道
     退化成"直调 vendor"，之后**零扫描、零异常、零打印**。
     但注意 vendor 的 DSL 模块是**首次下发时**才被 import 的（实测：`import exec_ops`
     之后 `sys.modules` 里一个 mqsmla 模块都没有、连 `torch.ops...ds41` 属性都不存在），
     所以「第一次没扫到」**不能**判死 —— 那会把整条通道变成静默 no-op。折中是
     **总扫描预算** `_SCAN_BUDGET=12`：未命中就保持 unset、下次再扫，预算用完才永久
     dead。最坏一次性成本 12×0.5 ms ≈ 6 ms（进程级，不是 per-call），一旦 vendor 模块
     被 import 就在下一次调用自愈命中。
  c) **打印封顶**：进程内最多 `_MAX_NOTES=3` 行（+1 行"后续不再打印"），其余只计数，
     `atexit` 汇总一行。

# 2. 逐位不变的三段论证

  (1) **L1（跳过 torch 派发器）**：调用的就是 `@impl(..., "PrivateUse1")` 注册的那个
      函数对象本身，实参 `(q, **kwargs)` 与 `torch.ops...` 的 schema 一一对应。
      经验证据：A1 的 `DS41M_MQ_FAST=1`（同一条路）在 64k@0.97 上 digest
      `08443723284dd9ed` 与基线逐位相同（NOTES 2026-10-26-mq-fastcall-and-envhoist.md）。
      本模块另加两道静态判据：`__name__` 与 `__module__` 都要对得上（防"扫到同名别的函数"）。

  (2) **L2 的 provider 桥快路**：只在**挂载时逐项与 vendor 对齐**之后才启用
      （`_build`，判据与 A6 完全一致）：
        * `L.bind_physical(args, {}) is args`（vendor 的物理绑定原样返回调用方 tuple；
          `parameter_lowering.py:177,246-249` 的 positional-identity 快路）
        * 我复刻的 `tensor_items` / `expected_contracts` 与 vendor 的
          `provider_tensor_items` / `provider_tensor_contract_items` **逐项同身份**
        * 槽位是 Tensor/Scalar 且**没有** StructContract(BUFFER) 槽、没有 TensorList 槽、
          没有 struct_ref 槽、`_debug_plans` 为空、无 kwargs、argc 相等
      每次调用仍逐个槽做 `type(v) is torch.Tensor` 守卫，并对每个标量槽**原样调用
      vendor 的 `validate_host_scalar_value`**（`contracts/elements.py:668`）——
      也就是说 vendor 在 `bind_physical` 循环里做的**唯一有副作用的检查一条不少**，
      省掉的只是"重建 tuple / 属性链 / 重复比较"。
      最终喂给 `runtime_bridge.invoke_provider_call` 的
      `(plan, public_values, tensor_items, struct_ref_items, expected_contracts)`
      与 vendor 原路**逐项同身份** ⇒ 下游（inspect_storage / validate_storage /
      resolve_stream / prepare_submit / native resolve_first / native invoke）
      **一行都没被绕过**。三条我们明确**不碰**的项（理由同 A6）：
        * `inspect_storage`：`npu_format_cast_` 原地改格式，"格式不变"不可证；
        * `resolve_stream`：流会在图捕获/侧流之间切换，缓存即错；
        * `prepare_submit`（`record_stream`×N，实测 50 µs/次，是 NOPREP 那条 −2.0 ms 的来源）：
          它是**生命周期契约**，不是纯计算，跳过不安全 ⇒ 放弃。

  (3) **`export_tensor` 指纹缓存**：指纹 = `data_ptr/shape/stride/dtype/storage_offset/
      requires_grad/is_neg/is_conj`，命中即复用 `layout` 与 `dlpack_call_mode`；缓存里
      **不存张量**（每次重建 `_TensorExport`，不钉显存）。
      本版比 A6 多一道：**每个指纹键的第一次命中会与 vendor 双跑对账**
      （`_ORIG_EXPORT` 也跑一遍，比对 layout/mode/device），一致才继续用；不一致 ⇒
      `_FP_DEAD=True` 永久关掉 memo 并返回 vendor 的记录。`DS41M_MQ_L2_AUDIT=0` 可关掉对账，
      默认 **1**（每个键只多花一次，进程内几十次量级）。

# 3. 预期（本机 CPU 实测，`tmp/b3_tree/bench/bench_parts.py`，17 槽 = 15 张量 + 2 标量）

    vendor bind_physical + items + contracts（fast_abi 已开）   29.7 µs/invoke
    本模块快路                                                 1.4 µs/invoke
    export_tensor（真实 vendor 实现）                           3.8 µs/张量
    ⇒ 合计省 ≈ 28 + 40（15 张量 × 2.6）≈ **65~70 µs/invoke × 40 次/forward ≈ 2.6~2.8 ms/forward host**

  wall 只在 host-bound 的回放尾段兑现（见 NOTES/2026-10-26-a4-deepinfer-host.md §3）。

# 4. 开关 / 统计

    DS41M_MQ_L2=0（默认）   关：exec_ops 走 `mq_fastcall.mqsmla()`（A1 的通道，含 DS41M_MQ_FAST）
    DS41M_MQ_L2=1           开：L2（L1 + 桥快路 + export 指纹）
    DS41M_MQ_L2_LEVEL=0|1|2 自检/分臂用，**覆盖**上面的开关：0=L0 1=L1 2=L2
    DS41M_MQ_L2_STATS=1     退出时打印一行汇总
    DS41M_MQ_L2_AUDIT=0     关掉 export 首命中双跑对账（默认开）
    DS41M_MQ_L2_DEAD_AFTER=2  L2 连续异常多少次后**永久**退到 L1（默认 2）
    （定位器另有**总扫描预算** 12 次：vendor 的 DSL 模块要到首次下发才 import，
      所以「第一次没扫到」不等于失败；用完预算才判死）

纪律：只读 vendor 已导出的名字；只 patch `cannbotdsl` 的两个方法（与 A6 同一处），
不改 vendor 源码、不改 `ds41_runtime/`。
"""
from __future__ import annotations

import atexit
import os
import sys

import torch

# ─────────────────────────── 开关 ───────────────────────────
_LVL = None          # None = 还没解析
# ★ 2026-10-26（T-057，两机交错 A/B，各 8 轮）：**默认改成开**。
#   本机 L0 107.49/107.23 vs **L2 104.55/103.67**；邻机 L0 107.30/106.54 vs **L2 104.87/103.00**
#   ⇒ **−3.0 ~ −3.3 ms**，四臂 digest 全同 `08443723284dd9ed`；
#   快路统计：`impl=found 调用=360 l2=360 回退=0 放弃=0`（360 = 40 层 × 9 轮）。
#   `DS41M_MQ_L2=0` 可回退到厂商原路。
_ON = os.environ.get("DS41M_MQ_L2", "1").strip().lower() in ("1", "on", "true", "yes", "2")
_RAW_LVL = os.environ.get("DS41M_MQ_L2_LEVEL", "").strip()
_STATS_ON = os.environ.get("DS41M_MQ_L2_STATS", "").strip().lower() in ("1", "on", "true", "yes")
_AUDIT = os.environ.get("DS41M_MQ_L2_AUDIT", "1").strip().lower() not in ("0", "off", "false", "no")
try:
    _DEAD_AFTER = max(1, int(os.environ.get("DS41M_MQ_L2_DEAD_AFTER", "2")))
except ValueError:
    _DEAD_AFTER = 2
_MAX_NOTES = 3       # ★ 进程内最多打印几行（A1 的 +98 ms 有一半是逐调用 flush 打印）

_STATS = {
    "n": 0, "l0": 0, "l1": 0, "l2": 0, "fallback": 0, "impl_miss": 0,
    "arm_ok": 0, "arm_skip": 0, "fast": 0, "vendor": 0,
    "exp_hit": 0, "exp_miss": 0, "aud_ok": 0, "aud_bad": 0,
    "scans": 0, "notes": 0, "dead_at": -1,
}


def level() -> int:
    """0=L0（厂商原路）/ 1=L1（直接调 impl）/ 2=L2（L1+桥快路+export 指纹）。"""
    global _LVL
    if _LVL is None:
        if _RAW_LVL in ("0", "1", "2"):
            _LVL = int(_RAW_LVL)
        else:
            _LVL = 2 if _ON else 0
    if _IMPL_STATE == "dead" and _LVL > 0:
        return 0                       # ★ 定位失败 ⇒ 整条通道归零（零扫描/零异常/零打印）
    return _LVL


def enabled() -> bool:
    """exec_ops 在 **import 期**读一次（与其它 env 快照同一纪律）。"""
    return level() > 0


def stats() -> dict:
    return dict(_STATS, level=level(), impl=_IMPL_STATE)


def _note(msg: str) -> None:
    """封顶打印：进程内最多 _MAX_NOTES 行（storm guard）。"""
    _STATS["notes"] += 1
    if _STATS["notes"] <= _MAX_NOTES:
        print(f"  [mq-l2] {msg}", flush=True)
    elif _STATS["notes"] == _MAX_NOTES + 1:
        print(f"  [mq-l2] （后续同类信息不再打印，用 DS41M_MQ_L2_STATS=1 看汇总）",
              flush=True)


# ─────────────────────────── vendor impl 定位 ───────────────────────────
# 不能用点号 import：vendor 包在两种布局下的顶层包名不同（源码树
# `cann_ops_transformer_experimental` / wheel `cann_ops_transformer`），
# 而且 DSL 模块是 `torch.library` 注册期被 import 的，名字不唯一。
# 后缀用**包内相对路径**（两种布局都成立），与 A6 的 _IMPL_SUFFIX 同款做法。
_IMPL_SUFFIX = "/mixed_quant_sparse_flash_mla_dsl/mixed_quant_sparse_flash_mla.py"
_FN_NAME = "mixed_quant_sparse_flash_mla"
_META_NAME = "mixed_quant_sparse_flash_mla_metadata"

_IMPL_STATE = "unset"        # unset / found / dead
_IMPL = None                 # (attn_fn, meta_fn)
_IMPL_TRIES = 0
# ★ 有界 + 自愈：vendor 的 DSL 模块是**首次下发时**才被 import 的（实测 `import exec_ops`
#    之后 sys.modules 里一个 mqsmla 模块都没有），所以「第一次没扫到」不能判死——那时它
#    只是还没被 import。策略：未命中就保持 unset，下次再扫；**总扫描次数**封顶
#    `_SCAN_BUDGET`，用完才永久 dead。最坏一次性成本 12×0.5 ms ≈ 6 ms，之后零成本。
_SCAN_BUDGET = 12


def _accept(m):
    if m is None:
        return None
    attn = getattr(m, _FN_NAME, None)
    meta = getattr(m, _META_NAME, None)
    if not callable(attn) or not callable(meta):
        return None
    # 静态判据：名字与模块都要对得上（防扫到同名不同物的函数）
    if getattr(attn, "__name__", "") != _FN_NAME:
        return None
    modname = getattr(attn, "__module__", "") or ""
    if not modname.endswith("mixed_quant_sparse_flash_mla_dsl.mixed_quant_sparse_flash_mla"):
        return None
    return (attn, meta)


def _impl():
    """返回 (attn_fn, meta_fn) 或 None。**失败永久缓存**，绝不逐调用重扫。"""
    global _IMPL, _IMPL_STATE, _IMPL_TRIES
    if _IMPL_STATE == "found":
        return _IMPL
    if _IMPL_STATE == "dead":
        return None
    if _IMPL_TRIES >= _SCAN_BUDGET:
        _IMPL_STATE = "dead"
        _STATS["dead_at"] = _STATS["n"]
        _note(f"vendor impl 未找到（已扫表 {_IMPL_TRIES} 次）⇒ 本进程永久走厂商原路")
        return None
    _IMPL_TRIES += 1
    _STATS["scans"] += 1
    found = None
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None) or ""
        if f.endswith(_IMPL_SUFFIX):
            found = _accept(m)
            if found is not None:
                break
    if found is not None:
        _IMPL, _IMPL_STATE = found, "found"
    return _IMPL


# ─────────────────────────── L0 / L1 ───────────────────────────
def _l0(q, kw):
    d = torch.ops.cann_ops_transformer.ds41
    return d.mixed_quant_sparse_flash_mla(q, **kw)


def _l1(q, kw):
    impl = _impl()
    if impl is None:
        raise _Skip("impl 未找到")
    return impl[0](q, **kw)


# ─────────────────────────── L2：provider 桥快路 ───────────────────────────
class _Skip(Exception):
    """快路不可用（不是算子本身的错）→ 静默回退。"""


class _St:
    __slots__ = ("n", "pairs", "scalars", "contracts")

    def __init__(self, n, pairs, scalars, contracts):
        self.n = n
        self.pairs = pairs            # tuple[(slot_name, arg_index)]  ← 只存结构，不存张量
        self.scalars = scalars        # tuple[(arg_index, ElementInfo, slot_name)]
        self.contracts = contracts    # tuple[Contract]，与 pairs 一一对应


_TT = torch.Tensor
_INT8_DTYPES = (10, 12, 14)   # vendor export_tensor 里要走 int8 视图的 dlpack 码

_ARM_DEPTH = 0
_ACTIVE = False

_ORIG_CALL = None
_ORIG_EXPORT = None
_TENSOR_EXPORT = None
_BRIDGE_TRIED = False
_BRIDGE_OK = False
_RB = None
_ST: dict[int, tuple] = {}    # id(callable) -> (强引用, _St | None)


def _build(self, args, kwargs):
    """与 vendor 逐位对齐地重建静态编组；任何一条不成立就永久放弃这个 callable。"""
    if kwargs:
        raise _Skip("kwargs")
    if getattr(self, "_debug_plans", ()):
        raise _Skip("debug_plans")
    if getattr(self, "_noop", False):
        raise _Skip("noop")
    L = getattr(self, "_parameter_lowering", None)
    if L is None:
        raise _Skip("no lowering")
    lay = L.layout
    if not lay.positional_identity:
        raise _Skip("positional_identity=False")
    if lay.struct_ref_indices:
        raise _Skip("struct_ref")
    if any(k == "buffer" for k in lay.physical_slot_kinds):
        raise _Skip("struct buffer slot")          # vendor 会 marshal 成 memoryview
    n = lay.public_slot_count
    if len(args) != n:
        raise _Skip("argc")
    # ★ 真跑一遍 vendor 的物理绑定，要求原样返回（含标量校验）
    phys = L.bind_physical(args, {})
    if phys is not args:
        raise _Skip("bind_physical not identity")
    abi = L.abi_arguments
    names = lay.physical_slot_names
    kinds = lay.physical_slot_kinds
    pairs = []
    contracts = []
    scalars = []
    for i in lay.provider_slot_indices:
        pairs.append((names[i], i))
        contracts.append(abi[i])
    if len(kinds) != n:
        raise _Skip("slot kind count")
    for i, kind in enumerate(kinds):
        if kind == "scalar":
            from cannbotdsl.core.contracts.elements import element_for_tag
            scalars.append((i, element_for_tag(abi[i].element), names[i]))
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
    return _St(n, tuple(pairs), tuple(scalars), tuple(contracts))


def _install_bridge():
    """把 ProviderCallable.__call__ / export_tensor 换成带缓存的版本。幂等、失败静默。

    与 `idx_fastcall._install_bridge`（A6）**可叠加**：两边都各自抓 `cur` 作为 _ORIG_*，
    且都只在自己的 `_ACTIVE`/`_ARM_DEPTH` 窗口内走快路 ⇒ 安装顺序无关、互不吞掉。
    """
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
        if getattr(cur, "_b3_mq", False):                  # 已装过（重复 import 情形）
            _ORIG_CALL = cur
            _RB = _rb
            _BRIDGE_OK = True
            return True
        _ORIG_CALL = cur
        _ORIG_EXPORT = _ti.TorchNPUIntegration.export_tensor
        _TENSOR_EXPORT = _ti._TensorExport
        _call._b3_mq = True
        _export_tensor._b3_mq = True
        PC.__call__ = _call
        _ti.TorchNPUIntegration.export_tensor = _export_tensor
        _RB = _rb
        _BRIDGE_OK = True
    except Exception:                                      # noqa: BLE001  vendor 改名/缺失 → 不动
        _BRIDGE_OK = False
    return _BRIDGE_OK


def _call(self, *args, **kwargs):
    """ProviderCallable.__call__ 的快路版：非本窗口的调用一字不改。"""
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
        except Exception:                                  # noqa: BLE001  绝不影响已完成的调用
            st = None
            _STATS["arm_skip"] += 1
        else:
            _STATS["arm_ok"] += 1
        _ST[key] = (self, st)
        return out
    if ent[0] is not self or ent[1] is None:               # id 复用 / 已放弃
        return _ORIG_CALL(self, *args, **kwargs)
    st = ent[1]
    # ── 每次调用的实参守卫（全在 invoke 之前；失败不产生任何副作用）──
    if kwargs or len(args) != st.n:
        return _ORIG_CALL(self, *args, **kwargs)
    items = []
    for name, i in st.pairs:
        v = args[i]
        if type(v) is not _TT:
            return _ORIG_CALL(self, *args, **kwargs)
        items.append((name, v))
    for i, info, name in st.scalars:
        # vendor 在 bind_physical 循环里对每个标量做的那次校验，一条不少地照做
        try:
            _validate_scalar(info, args[i], name)
        except BaseException:                              # noqa: BLE001  交给 vendor 去报原错
            return _ORIG_CALL(self, *args, **kwargs)
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


def _validate_scalar(info, value, name):
    from cannbotdsl.core.contracts.elements import validate_host_scalar_value
    return validate_host_scalar_value(info, value, path=name)


# ── export_tensor 指纹缓存（缓存里不放张量）+ 首命中双跑对账 ──────────────
_FP_MEMO: dict[tuple, tuple] = {}
_FP_CAP = 4096
_FP_DEAD = False


def _fp(tensor):
    return (int(tensor.data_ptr()), tensor.shape, tensor.stride(),
            tensor.dtype, tensor.storage_offset(), tensor.requires_grad,
            tensor.is_neg(), tensor.is_conj())


def _export_tensor(self, tensor, device):
    global _FP_DEAD
    if (not _ACTIVE or _FP_DEAD
            or type(tensor) is not self._torch.Tensor):
        return _ORIG_EXPORT(self, tensor, device)
    key = _fp(tensor)
    ent = _FP_MEMO.get(key)
    if ent is not None and ent[0] == device:
        layout, mode, audited = ent[1], ent[2], ent[3]
        exporter = tensor.detach() if key[5] else tensor
        if layout[3][0] in _INT8_DTYPES:
            exporter = exporter.view(self._torch.int8)
        if audited:
            _STATS["exp_hit"] += 1
            return _TENSOR_EXPORT(tensor, exporter, device, layout, mode)
        # ★ 每个键的**第一次命中**与 vendor 双跑对账；不一致 → 永久关掉 memo
        rec_v = _ORIG_EXPORT(self, tensor, device)
        if (type(rec_v) is _TENSOR_EXPORT and rec_v.layout == layout
                and rec_v.dlpack_call_mode == mode and rec_v.device == device):
            _FP_MEMO[key] = (device, layout, mode, True)
            _STATS["aud_ok"] += 1
            _STATS["exp_hit"] += 1
            return _TENSOR_EXPORT(tensor, exporter, device, layout, mode)
        _FP_DEAD = True
        _FP_MEMO.clear()
        _STATS["aud_bad"] += 1
        _note("export 指纹对账不一致 ⇒ 永久停用 export 缓存（其余快路不受影响）")
        return rec_v
    rec = _ORIG_EXPORT(self, tensor, device)               # 负位/共轭/非 strided 由 vendor 抛
    _STATS["exp_miss"] += 1
    if len(_FP_MEMO) < _FP_CAP:
        _FP_MEMO[key] = (device, rec.layout, rec.dlpack_call_mode, not _AUDIT)
    return rec


# ─────────────────────────── 公共入口 ───────────────────────────
_L2_FAILS = 0


def _l2(q, kw):
    global _ARM_DEPTH, _L2_FAILS
    impl = _impl()
    if impl is None:
        raise _Skip("impl 未找到")
    if not _BRIDGE_OK:
        _install_bridge()
    if not _BRIDGE_OK:
        raise _Skip("桥安装失败")
    _ARM_DEPTH += 1
    try:
        out = impl[0](q, **kw)
    except BaseException as e:                             # noqa: BLE001
        _L2_FAILS += 1
        if _L2_FAILS >= _DEAD_AFTER:
            _mark_dead(f"L2 连续异常 {_L2_FAILS} 次 ⇒ 本进程永久退 L1: "
                       f"{type(e).__name__}: {e}")
        raise
    finally:
        _ARM_DEPTH -= 1
    _L2_FAILS = 0
    return out


def _mark_dead(msg):
    """把整条通道永久关掉 ⇒ `level()` 之后一律返回 0（直调厂商原路）。"""
    global _IMPL_STATE
    if _IMPL_STATE != "dead":
        _STATS["dead_at"] = _STATS["n"]
    _IMPL_STATE = "dead"
    _note(msg)


def mqsmla(q, kw):
    """mqsmla 主调用。`kw` 是 exec_ops 组好的 kwargs dict（与 torch.ops 同形）。"""
    _STATS["n"] += 1
    lv = level()
    if lv >= 2:
        try:
            r = _l2(q, kw)
            _STATS["l2"] += 1
            return r
        except _Skip:
            _STATS["impl_miss"] += 1                 # 定位失败不打印（有 budget 兜底）
            lv = 1 if _IMPL_STATE == "found" else 0  # impl 没找到就没必要再试 L1
        except BaseException as e:                   # noqa: BLE001
            _STATS["fallback"] += 1
            _note(f"L2 失败回退 L1: {type(e).__name__}: {e}")
            lv = 1
    if lv == 1:
        try:
            r = _l1(q, kw)
            _STATS["l1"] += 1
            return r
        except _Skip:
            _STATS["impl_miss"] += 1
        except BaseException as e:                   # noqa: BLE001
            _STATS["fallback"] += 1
            _note(f"L1 失败回退 L0: {type(e).__name__}: {e}")
    _STATS["l0"] += 1
    return _l0(q, kw)


def mqsmla_metadata(*args, **kw):
    """metadata 生成：**永远走厂商原路**。

    exec_ops 已按 ctx 做值缓存（稳态 ~0 次/forward），省这里没有收益；而它一旦慢/错
    就是"改数值"级别的事故 ⇒ 按「宁可放弃」直接不碰。`_impl()` 里仍然要求 metadata
    入口存在，只是把它当作"找对了模块"的静态判据。
    """
    d = torch.ops.cann_ops_transformer.ds41
    return d.mixed_quant_sparse_flash_mla_metadata(*args, **kw)


# ─────────────────────────── 统计 / 自检辅助 ───────────────────────────
def _dump():
    lv = level()
    if not _STATS_ON:
        return
    print(f"  [mq-l2] L={lv} impl={_IMPL_STATE} 调用={_STATS['n']}"
          f" l0={_STATS['l0']} l1={_STATS['l1']} l2={_STATS['l2']}"
          f" 回退={_STATS['fallback']} impl缺={_STATS['impl_miss']}"
          f" dead_at={_STATS['dead_at']} 扫表={_STATS['scans']} 打印={_STATS['notes']}"
          f" | 桥: 挂上={_STATS['arm_ok']} 放弃={_STATS['arm_skip']}"
          f" 快路={_STATS['fast']} 原路={_STATS['vendor']}"
          f" | export 命中={_STATS['exp_hit']} 未命中={_STATS['exp_miss']}"
          f" 对账好={_STATS['aud_ok']} 坏={_STATS['aud_bad']}", flush=True)


atexit.register(_dump)


def reset_caches() -> None:
    """自检/换臂用：清掉 level、定位状态、挂载缓存与 export 缓存（**不卸载** patch）。"""
    global _LVL, _IMPL_STATE, _IMPL, _IMPL_TRIES, _FP_DEAD, _L2_FAILS
    _LVL = None
    _IMPL = None
    _IMPL_STATE = "unset"
    _IMPL_TRIES = 0
    _FP_DEAD = False
    _L2_FAILS = 0
    _ST.clear()
    _FP_MEMO.clear()

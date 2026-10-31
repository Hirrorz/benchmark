"""算子注册表 —— mini 的**唯一**算子定义处（§7 纪律 1：层里不准出现算子名）。

一个算子 = 一条 spec：
    kernel   : trace 里能看到的算子/内核名（align_ops 用它对齐；dry-run 直方图也按它汇总）
    shapes   : 输入形状（符号式，用 ctx 求值；M2 真跑时用来造张量）
    count    : 本次调用次数（函数，吃 ctx）—— MoE 类按 chunk 数放大
    affects  : 哪些 backend 开关会替换/跳过它（换算子只需要改这里）
    trace    : 校准证据：参考 trace（kernel_details.csv）里该算子 Input Shapes 的原文
    note     : 归因/置信度说明

ctx 里有什么（由 run.py 组装）：
    T        本层实际 token 数（受 hit/CP/replay 影响）
    chunk    MoE 分块大小（moe_chunk_max_len）
    nchunks  ceil(T/chunk)
    H, I     hidden / moe_intermediate
    ratio    本层压缩比（0/1/2）
    rtopk    路由 topk
    e_local  本 rank 的专家数（EP 切分后）
    S        窗口大小(128)
    idx      层号
    is_kv_src / is_idx_src / is_candidate / engram   层角色（0/1）—— **数量**增量用

纪律：**哪些层有这个算子 = recipes.yaml 说了算；每层跑几次 = 这里的 count 说了算**
（例如 kv_src 层多 1 次 rope 是数量 → 用 count；索引器只有 8 层有是存在 → 写进 recipes）
"""

def _chunks(ctx):
    return max(1, -(-ctx["T"] // ctx["chunk"]))


OPS = {
    # ── 通用（40 层都有）───────────────────────────────────────────
    # 校准口径与推导方法见 CALIBRATION.md；注释里是参考 trace 的单层次数
    "rmsnorm":      dict(kernel="RmsNorm",                 shapes=[["T", "H"]],                    count=lambda c: 1),   # attn_norm / ffn_norm（各 1 次）
    "q_norm":       dict(kernel="RmsNorm",                 shapes=[["T", "1280"]],                 count=lambda c: 1),   # q_lora 上
    "kv_norm":      dict(kernel="RmsNorm",                 shapes=[["T", "512"]],                  count=lambda c: 1),   # 压缩 KV 上（4 个 norm 合计 4/层 ✓）
    "hc_pre":       dict(kernel="HcPre",                    shapes=[["T", "H"]],                    count=lambda c: 1),   # 配方里调用 2 次（attn 段/ffn 段）
    "hc_post":      dict(kernel="mhc_post_fwd_asc_kernel",  shapes=[["T", "H"]],                    count=lambda c: 1),   # 配方里调用 2 次
    "mul":          dict(kernel="Mul",                      shapes=[["T"]],                         count=lambda c: 1),   # 1（其余残差在 extras）
    "add":          dict(kernel="Add",                      shapes=[["T"]],                         count=lambda c: 1),   # 1（其余残差在 extras）
    "cast":         dict(kernel="Cast",                     shapes=[["T", "160"], ["T", "80"], ["T", "6"]], count=lambda c: 3),   # 3（其余残差在 extras）
    "tmove":        dict(kernel="TensorMove",               shapes=[["T", "H"], ["T", "1", "512"]], count=lambda c: 2),   # 2（参考 trace 85 = 2×40+5）
                    # 注：执行侧 .contiguous() 不产内核，所以在我们的 trace 里看不到（已知差）
    "dyn_mx_quant": dict(kernel="DynamicMxQuant",           shapes=[["T", "H"]],                    count=lambda c: 6 + c["is_idx_src"] + c["engram"]),
                    # 6 = 3×[T,5120] + [T,1280] + [T,8,4096] + [T,8192]
                    # idx_src 层多 1 个 [T,1280]，engram 层多 1 个 [T,6144]（250 = 6×40 + 8 + 2 ✓）

    # ── 注意力（每层都有）—— 4 个量化矩阵乘已按**真权重形状**归因 ──
    # 归因证据：WEIGHTS_INVENTORY.md 的权重形状 × trace 的 QMM 输入形状，两者逐一对上；
    # 权重展平后 = groups×n×16×32（如 40×320×16×32 = 1280×5120 = wq_a）。
    "wq_a":         dict(kernel="QuantBatchMatmulV3",       shapes=[["T", "H"], ["1280", "5120"]],  count=lambda c: 1,
                         trace="2048,5120;40,320,16,32;80,1280,2;2048,80,2", note="wq_a [1280,5120]（MLA q 低秩投影）"),
    "wq_b":         dict(kernel="QuantBatchMatmulV3",       shapes=[["T", "1280"], ["32768", "1280"]], count=lambda c: 1,
                         trace="2048,1280;1024,80,16,32;20,32768,2;2048,20,2", note="wq_b [32768,1280]（64 头 × 512）"),
    "wkv":          dict(kernel="QuantBatchMatmulV3",       shapes=[["T", "H"], ["512", "5120"]],   count=lambda c: 1,
                         trace="2048,5120;16,320,16,32;80,512,2;2048,80,2", note="wkv [512,5120]（压缩 KV 投影）"),
    "wo_b":         dict(kernel="QuantBatchMatmulV3",       shapes=[["T", "8192"], ["5120", "8192"]], count=lambda c: 1,
                         trace="2048,8192;160,512,16,32;128,5120,2;2048,128,2", note="wo_b [5120,8192]（输出投影）"),
    "engram_proj":  dict(kernel="QuantBatchMatmulV3",       shapes=[["T", "6144"], ["25600", "6144"]], count=lambda c: 1,
                         trace="2048,6144;800,384,16,32;96,25600,2;2048,96,2", note="仅 engram 层（1,14），2/pass ✓"),
    "rope":         dict(kernel="InplacePartialRotaryMul",  shapes=[["T", "64", "1", "512"]],       count=lambda c: 3 + c["is_kv_src"]),
                    # 3 = 2×[T,64,1,512] + 1×[T,1,1,512]；kv_src 层多 1 个大块 [1025,1,1,512]（124 = 3×40 + 4 ✓）
    "kv_epilog":    dict(kernel="KvCompressEpilogV2",       shapes=[["384", "544"], ["2176", "544"]], count=lambda c: 2, affects=["megamoe"],
                         trace="384,544;2048,512;2048 / 2176,544;2048,512;2048", note="每层 A/B 两块各 1 次（时序 ABABAB…）；38 层无 compressor 权重 → 这是共享 cache 的 epilog，不是压缩写入"),
    "kv_compress":  dict(kernel="KvCompressEpilogV2",       shapes=[["1025", "512"]],               count=lambda c: 1, affects=["megamoe"],
                         trace="524544,320;1025,512;1025（×3）/ 1048832,320;2048,512;2048（×1）", note="真压缩写入：只有 kv_src 4 层有 attn.compressor.{wkv,wgate,norm} 权重（84 = 2×40 + 4 ✓）"),
    "mqsmla":       dict(kernel="MqsmlaKernel",             shapes=[["T", "H"]],                    count=lambda c: 1),   # 1
    "wo_a":         dict(kernel="TransposeQuantBatchMatMul", shapes=[["T", "8", "4096"]],          count=lambda c: 1,
                         trace="2048,8,4096;8,32,256,16,32;2048,8,64,2;8,64,1024,2", note="wo_a [8192,4096]（转置量化 MM；4096 = 64 头 × 64）"),
    # 索引器（只长在 idx_src / kv_src 层上；affects = 换算子要改的开关）
    "idx_prolog":   dict(kernel="indexer_prologue_qw_kernel", shapes=[["T", "H"]],                  count=lambda c: 1, affects=["indexer_prolog_qw"]),
    "qli":          dict(kernel="QuantLightningIndexer",    shapes=[["T", "H"]],                    count=lambda c: 1, affects=["indexer_prolog_qw"]),
    "qsli":         dict(kernel="QuantSparseLightningIndexer", shapes=[["T", "H"]],                 count=lambda c: 1, affects=["indexer_prolog_qw"]),

    # ── MoE（未融合链：megamoe=none 时的形态）──
    "gating":       dict(kernel="MoeInitRoutingV3",         shapes=[["T", "H"]],                    count=lambda c: _chunks(c), affects=["gate_topk"]),
    "gating_sort":  dict(kernel="Sort",                     shapes=[["T", "rtopk"]],                count=lambda c: _chunks(c)),
    "gate_topk":    dict(kernel="MoeGatingTopKHash",        shapes=[["T", "H"]],                    count=lambda c: _chunks(c), affects=["gate_topk"]),
    "reroute":      dict(kernel="MoeReRouting",             shapes=[["T", "rtopk"]],                count=lambda c: _chunks(c)),
    "dispatch":     dict(kernel="hcom_alltoallv_",          shapes=[["T", "H"]],                    count=lambda c: 2 * _chunks(c), affects=["megamoe"]),
    "gmm":          dict(kernel="GroupedMatmul",            shapes=[["T", "I"]],                    count=lambda c: 2 * _chunks(c), affects=["megamoe"]),
    "swiglu_quant": dict(kernel="SwigluGroupQuant",         shapes=[["T", "2I"]],                   count=lambda c: _chunks(c), affects=["swiglu", "megamoe"]),
    "combine":      dict(kernel="hcom_alltoall_",           shapes=[["T", "H"]],                    count=lambda c: _chunks(c), affects=["megamoe"]),
    "finalize":     dict(kernel="MoeFinalizeRoutingV2",     shapes=[["T", "rtopk"]],                count=lambda c: _chunks(c)),
    # ── 融合路径（megamoe=ascendc|deepgemm）：gating 仍在外面，其余被一条融合算子取代 ──
    "gate":         dict(kernel="FusedMatMul",              shapes=[["T", "H"], ["384", "5120"]],   count=lambda c: 1,
                         trace="2048,5120;384,5120", note="ffn.gate.weight [384,5120]（BF16）→ 路由打分，1/层"),
    "topk_gate":    dict(kernel="moe_topk_gate_forward_kernel", shapes=[["T", "H"]],                count=lambda c: 1, affects=["gate_topk"],
                         trace="moe_topk_gate_forward_kernel_asc_kernel", note="1/层"),
    "mega_moe":     dict(kernel="mega_moe_impl",            shapes=[["T", "H"]],                    count=lambda c: 1, affects=["megamoe"]),   # 1/层
    # ── engram 层（1,14）额外插在层首（fetch 可与主干重叠）──
    "engram_fetch": dict(kernel="EngramFetch",              shapes=[["T", "1280"]],                 count=lambda c: 1),
    "engram_wait":  dict(kernel="EngramFetchWait",          shapes=[["T", "1280"]],                 count=lambda c: 1),
    # ── 命中/replay 的形状变换（§8/§10；不是真正的算子，但会出现在时间线上）──
    "gather_tail":  dict(kernel="GatherV2",                 shapes=[["T", "H"]],                    count=lambda c: 1),
    "scatter_tail": dict(kernel="IndexCopy",                shapes=[["w", "H"]],                    count=lambda c: 1),
}

# 未显式校准的 op（通常是从 modules.d/*.yaml 新加的模块）自动补默认条目：
#   kernel = iface.kernel or op，shapes = []（不参与形状对齐），count = 1。
# 目的：新模块进 recipes 时门禁不再因为"忘了改 opgraph.py"直接变红（DX P0-3）；
# 精确直方图/形状对齐仍要求在 OPS 里写显式条目（显式条目优先，不会被覆盖）。
_AUTO_SKIP_FLAGS = {"no_impl", "declared_only", "implicit", "placeholder",
                    "runner_hook", "boundary_only"}


def refresh_defaults() -> list[str]:
    """给 OP_IFACE 中尚无 OPS 条目的真实算子补默认条目（幂等）。"""
    from modules.op_iface import OP_IFACE          # noqa: PLC0415 —— 避免顶部循环导入

    added = []
    for op, iface in OP_IFACE.items():
        if op in OPS:
            continue
        if iface.flags & _AUTO_SKIP_FLAGS:
            continue
        if not iface.impl and not iface.kernel:
            continue
        OPS[op] = dict(kernel=iface.kernel or op, shapes=[], count=lambda c: 1,
                       affects=[], auto=True,
                       note="auto：未校准（需要精确直方图时在 opgraph.py 补 shapes/count/trace）")
        added.append(op)
    return added


refresh_defaults()

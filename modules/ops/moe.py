"""FFN / MoE 算子（融合 + 未融合链）。"""
from modules.registry import register_group

CLASSES = register_group(["gate", "topk_gate", "mega_moe",
                          "gating", "gating_sort", "reroute", "dispatch", "gmm",
                          "swiglu_quant", "combine", "finalize",
                          "gate_topk", "gather_tail", "scatter_tail"])

"""通用/残差流算子（每个 op 一个 Module 类）。"""
from modules.registry import register_group

CLASSES = register_group(["rmsnorm", "hc_pre", "hc_post",
                          "mul", "add", "cast", "tmove", "dyn_mx_quant"])

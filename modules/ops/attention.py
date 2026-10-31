"""注意力算子。"""
from modules.registry import register_group

CLASSES = register_group(["wq_a", "q_norm", "wq_b", "wkv", "kv_norm", "rope",
                          "kv_epilog", "mqsmla", "wo_a", "wo_b"])

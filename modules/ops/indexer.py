"""索引器算子。"""
from modules.registry import register_group

CLASSES = register_group(["idx_prolog", "idx_k_write", "qli", "qsli"])

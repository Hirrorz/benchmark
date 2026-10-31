"""边界层算子（runner 钩子）。"""
from modules.registry import register_group

CLASSES = register_group(["boundary_cut"])

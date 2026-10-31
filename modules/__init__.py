"""模型模块规格（Phase 1：纯声明 + 静态解析；运行时不导入）。

设计见 NOTES/2026-10-23-module-spec-viewer.md：
  · 每个 op 一个 Module 类（`modules/ops/*`，由 OP_IFACE 生成，接口唯一事实源）；
  · 上下文分层（作用域链 + provides 上浮 + 最近祖先解析）；
  · 导出器 `export_model_html` 生成单文件离线查看器 `tools/model_viewer.html`。

本包**顶层不导入 torch / exec_***，保证纯 CPU 可跑。
"""
__all__ = ["spec", "names", "registry", "op_iface", "recipe_map", "compose",
           "shapes", "providers", "weights", "tree", "runner_iface", "context"]

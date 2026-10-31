#!/usr/bin/env python3
"""探明 ds41 算子组的注册条件（谁先谁后、要不要 shim）。

背景：cann_ops_transformer 内部有两条路径会加载 cann_ops_transformer_experimental，
撞上就"重复注册"→ ds41 组注册失败（indexer/mqsmla 全丢）。
用法：python3 tools/probe_ds41_ns.py <变体号 1..4>
"""
import sys

V = sys.argv[1] if len(sys.argv) > 1 else "1"
import torch          # noqa: E402
import torch_npu      # noqa: E402

import custom_ops     # noqa: E402,F401

if V == "3":
    sys.path.insert(0, "shim")
if V == "4":
    sys.path.insert(0, "shim")
    import cann_ops_transformer_experimental  # noqa: F401

try:
    import cann_ops_transformer  # noqa: F401
    if V == "1":
        import cann_ops_transformer_experimental  # noqa: F401
    if V == "2":
        pass                       # 只 import cann_ops_transformer
except Exception as exc:           # noqa: BLE001
    print(f"  变体 {V}: ✗ import 失败 {type(exc).__name__}: {str(exc)[:70]}")
    raise SystemExit(0)

d = getattr(torch.ops.cann_ops_transformer, "ds41", None)
names = [x for x in dir(d) if not x.startswith("_")] if d else []
print(f"  变体 {V}: ds41 = {len(names)} 个；有 mqsmla={'mixed_quant_sparse_flash_mla' in names}")

# modules.d — 小模块配置（可选）

把一个模块单独放一个 `*.yaml`，`modules/spec_loader.py` 会按文件名排序合并到主配置
`../model_modules.yaml` 之后：

```yaml
# modules.d/my_op.yaml（单模块可省略 schema）
op: my_op
category: attention
container: attn.core
impl: exec_ops:_my_op
requires:
- {key: hidden.x, type: Tensor, shape: "[T,H]"}
provides:
- {key: attn.my_out, type: Tensor, shape: "[T,H]"}
params:
  my_scale: 1.0
```

- 同名 op 默认报错；确实要覆盖主配置时写 `override: true`。
- 以 `_` 开头的文件会被跳过（草稿/备份）。
- 目录可用 `DS41M_MODULE_SPEC_DIR=<dir>` 覆盖。
- 改完跑 `python3.11 tools/test_module_config.py` 与 `tools/check_module_spec.py`。

# 模块开发指南（插件式）

目标：**谁都参与开发**——写一个模块 = 写一个声明了上下文接口的 Module 类；
把模块加进配方，模型就多一块积木；重新导出后，HTML 会**按代码**展示模型结构与
每个模块的上下文接口，而不是靠人脑想象。

```
插件类（接口声明） ──注册──▶ OP_IFACE / MODULES ──配方引用──▶ 模块树 ──导出──▶ model_viewer.html
                                      │                                   ├─ 结构（40 层 → op）
                                      └── 检查器/构建器/运行时探针          ├─ 积木库（全部插件）
                                                                          ├─ 组装蓝图（段/模板）
                                                                          └─ 模块构建器（可视化拼装）
```

## 0. SSH 下怎么查看（不必下载 HTML）

**方式 A（推荐）：端口转发 + 自动刷新**

```bash
# 远端
source env_local.sh
python3.11 tools/serve_viewer.py --config configs/mini_ds41_flash.yaml --port 8765

# 本地另开终端（或 VS Code Remote-SSH 的「端口」面板转发 8765）
ssh -L 8765:127.0.0.1:8765 <user>@<host>
# 本地浏览器打开 http://localhost:8765/
```

改完 `modules/`（`.py` / `.yaml` / `.yml`，含 `modules/specs/`）/ `recipes.yaml` /
模板后，浏览器 **1.5s 内自动刷新**（服务端按源文件 mtime 重建，约 150ms）。
`/api/data` 是完整 JSON；`/api/version` 是源文件指纹。
默认只绑 `127.0.0.1`，只经 SSH 隧道访问；可信内网才用 `--bind 0.0.0.0`。

**方式 B：终端 CLI**（数据与 HTML 同源）

```bash
python3.11 tools/view_model.py layers
python3.11 tools/view_model.py tree 20 --view split --keys
python3.11 tools/view_model.py op 2 attn.compressor.kv_compress
python3.11 tools/view_model.py plugin kv_compress
python3.11 tools/view_model.py catalog --category attention
python3.11 tools/view_model.py blueprint
python3.11 tools/view_model.py search kv_compress
```

**方式 C：全屏 TUI**（纯 curses，无第三方依赖）

```bash
python3.11 tools/view_model.py tui
# ↑↓ 移动 · Tab 切栏 · Enter 展开/选择 · 1/2/3 视图 · p 预设 · / 搜索 · d 详情 · q 退出
```

## 1. 新增/修改一个模块（配置驱动，不改 Python 声明）

**模块定义、参数、容器/路径都在配置里**，唯一事实源：

| 文件 | 内容 |
|---|---|
| `modules/specs/model_modules.yaml` | 每个模块的 `requires/provides/weights/params/flags/impl/kernel/category/container` |
| `modules/specs/modules.d/*.yaml` | 单模块小文件（可选）；同名 op 需 `override: true` 才覆盖主配置 |
| `modules/specs/recipe_map.yaml` | 容器展示信息 + `(配方段, op, 出现序号) → 层内路径` |
| `modules/specs/key_names.yaml` | 键别名（`H→model.hidden`）、多义说明、命名空间默认 tier |
| `modules/specs/model_abstract.yaml` | **自动导出**的整模型抽象：model 常量 / 40 层 / 7 模板（steps+模块树）/ 用到的模块接口与参数（32 KiB） |
| `recipes.yaml` | 层内跑哪些算子（组合来源） |
| `configs/*.yaml` | 模型/运行参数（优先级高于模块 `params` 默认值） |

```bash
cd /mnt/xds/sfs/l00972766/ds41_mini && source env_local.sh

# ① 在 modules/specs/model_modules.yaml 的 modules: 下加/改一条（或点构建器「复制 YAML 配置」）
# ② 把 "my_attn_gate" 加进 recipes.yaml 的对应段
# ③ 门禁 + 重新导出（HTML 自动出现该模块）
python3.11 tools/test_module_config.py
python3.11 tools/check_module_spec.py
python3.11 -m modules.export_model_html --config configs/mini_ds41_flash.yaml
python3.11 tools/export_model_yaml.py --config configs/mini_ds41_flash.yaml \
    --out modules/specs/model_abstract.yaml            # 可选：整模型抽象 YAML
```

新模块**不需要改 Python 声明**：`modules/ops/__init__.py` 会为 OP_IFACE 里未分组的
op 自动生成 Module 类；`opgraph.py` 也会自动补默认条目（`kernel=op, shapes=[], count=1`，
标记 `auto`），门禁不会因为没同步这两个文件而变红。要精确的直方图/形状对齐时，
再到 `opgraph.py` 的 `OPS` 写显式条目（显式条目优先）。

配置条目示例（**接口就是契约**）：

```yaml
- op: my_attn_gate
  category: attention
  container: attn.core          # 默认装配容器
  impl: exec_ops:_my_attn_gate  # 实现函数仍在代码里；配置只声明"调谁"
  kernel: MyGateKernel
  after: [mmsmla]               # 顺序约束（同层出现时）
  requires:
  - {key: hidden.x, type: Tensor, shape: "[T,H]"}
  - {key: model.hidden, type: int}
  provides:
  - {key: attn.my_out, type: Tensor, shape: "[T,H]"}
  weights:
  - {suffix: attn.my_attn_gate.weight}
  params:                        # 模块参数：字面量或 "${祖先键}" 引用
    gate_scale: 1.0
    hidden: "${model.hidden}"
  doc: 一句话说明
```

- **参数解析**：执行该模块前，`params` 中运行配置（ctx）没有的键会按
  `"${key}"` 沿祖先作用域求值后临时注入 ctx，执行完撤销；优先级
  **运行配置 > 模块配置**。缺引用会直接报错，不静默用默认值。
- **父节点不汇总子节点 I/O**：容器只声明自己的键；子节点缺的键由
  `modules.context.Scope` 沿 `leaf → container → layer → pass → process` 解析。
- 工具生成（写 `modules/specs/modules.d/<op>.yaml`，不改主文件）：
  ```bash
  python3.11 tools/new_module.py my_op --category attention --container attn.core \
      --requires 'hidden.x=[T,H]' --provides 'attn.my_out=[T,H]' \
      --param gate_scale=1.0 --impl exec_ops:_my_op
  ```
  默认写到 `modules/specs/modules.d/`，门禁自动可见。若用
  `--out-dir /tmp/x` 做沙盒，跑门禁必须带 `DS41M_MODULE_SPEC_DIR=/tmp/x`
  （该变量**替换**默认 modules.d，不是叠加；不设时模块根本没加载，门禁会“假绿”，
  显式指向不存在的目录现在会直接报错）。
- 动态 Python 插件仍支持（`@module` + `modules/ops/plugins/`，
  `tools/new_module.py --python-stub` 服务这条旧路径；单元素 requires/weights
  的 tuple 尾逗号已修）；默认工作流是改配置。
- 覆盖/回退：`DS41M_MODULE_SPEC=<yaml>`、`DS41M_RECIPE_MAP=<yaml>`、
  `DS41M_KEY_NAMES=<yaml>`；文件缺失才回退
  `op_iface_builtin.py` / `recipe_map_builtin.py` / `names_builtin.py`；
  `DS41M_SPEC_STRICT=0` 允许解析失败时回退。

## 2. 上下文契约（写模块时只关心这个）

- `requires`：模块需要的键。键名用语义命名空间（`hidden.x` / `model.hidden` /
  `position.offset` / `cache.cmp.pool` …），解析规则是**最近祖先 → 层 → 本次 forward →
  进程**；`cache.*` 按“执行序最近的前序写者”解析（跨层池）。
  **父容器不需要声明或转发子节点的键**——子节点自己声明，缺的沿祖先链取。
- `provides`：模块写进自己字典的键；模块完成后**上浮到层工作集**，后面的兄弟/其它容器可见。
- `weights`：从权重清单里读的键；`via="_packed"` 表示读加载期打包结果，
  `synthetic` 表示运行期合成键。支持 `ffn.experts.{E}.w1.weight` 专家模板。
- `flags`：`implicit`（trace 计数但不下发）、`packed_variant`、`boundary_only`、
  `runner_hook`、`no_impl`、`placeholder`、`trace_only` 等；没有实现就显式写 `no_impl`。
- `container` / `after`：装在哪个容器、顺序约束。装配顺序仍由 `recipes.yaml` 决定；
  这两个字段用于**新插件的默认路径**与**顺序校验**。

### 硬规则

1. **模块顶层不 import torch / exec_ops**（导出器/门禁必须纯 CPU 可跑）；
   `execute()` 里再惰性 import。
2. **每个 ctx/st 读写的键都要声明**。漏声明不会立刻崩，但 `DS41M_CTX_TRACE=1`
   的运行时探针会把它列为“读漏/写漏”。
3. 权重必须能对上 `weights_manifest`（或显式加入 `modules/weights.py` 的忽略清单）。
4. 新算子要能被 `recipes.yaml` 引用；存在性只由配方决定。

## 3. HTML 查看器（改完代码的“验收视角”）

重新导出后打开 `tools/model_viewer.html`：

- **积木库**：全部已注册模块（含未被配方使用的），按分类筛选；点卡片看
  requires/provides/weights/params、内核/实现位置、装在哪（模板/层/路径），
  并可复制 **YAML 配置**（Python 声明保留为旧入口）。
- **组装蓝图**：`recipes.yaml` 的每个段（common / attn_core / attention.* / moe.* / engram…）
  与 40 层实际复用的模板；点任意积木跳到结构视图对应位置。
- **模块构建器**：填接口/参数 + 选基线模板/插入点，静态验证“插入点之前的所有 provides +
  层/进程基线”是否满足 requires；显示插入预览（前后积木）；一键生成可**复制/下载**的
  **YAML 模块配置**，以及要加进 `recipes.yaml` 的配方片段（`- my_op`）。
- **结构**：40 层 → 容器 → op；每个节点看上下文接口、解析自哪个祖先、权重、形状、内核、源码。

## 4. 门禁与命令

```bash
python3.11 tools/test_module_config.py                              # 配置加载/覆盖/新增模块
python3.11 tools/test_dx_regressions.py                             # DX 回归（自动注册/opgraph/stub/viewer YAML）
python3.11 tools/check_module_spec.py --configs 'configs/*.yaml'   # 顺序/权重/接口/形状/确定性
python3.11 -m modules.export_model_html --config configs/mini_ds41_flash.yaml
python3.11 -m modules.export_model_html --config configs/mini_ds41_flash.yaml --check
JSDOM_PATH=/tmp/jsdom_env/node_modules/jsdom node tools/test_model_viewer.js
python3.11 tools/diff_digest.py base.log mod.log                    # Phase 1.5/2 A/B
```

`check_module_spec.py` 的关键断言：40 层 DFS 叶子序列 == `run.expand_layer`（6 种口径）；
recipes 算子都有插件；权重双向覆盖；每个 require 可解析；形状可求值；导出确定性。

## 5. 常见问题

- **构建器报“未解析”**：插入点之前没有任何积木/基线提供该键。换插入点、加一个上游积木，
  或把该键标成可选（行尾 `?`）。
- **`phantom` 诊断**：提供了但本层没人消费——可能是给后续层/调试用的，确认后在
  `tools/check_module_spec.py` 的白名单里登记。
- **新容器**：插件 `container="attn.my_stage"` 即可自动建容器；展示名可在
  `modules/specs/recipe_map.yaml` 的 `containers` 里补。
- **同一 op 在不同段有不同容器**（如 split 的两次 rope）：在
  `modules/specs/recipe_map.yaml` 按 `(段, op, 出现序号)` 显式登记，优先级高于插件的
  `container`。
- **门禁报 `recipes−opgraph`**：新 op 已自动登记；若仍报错，检查它是否 `no_impl` /
  `runner_hook`（这类不会自动补），或 `recipes.yaml` 的 op 名拼写。

## 6. 运行时阶段

- Phase 1（当前）：声明 + 导出 + 查看器；运行时不改。
- Phase 1.5：`DS41M_CTX_TRACE=1` 探针 + `DS41M_CTX_ALIASES` 别名视图（默认关）。
- Phase 2（已落地，**默认开**；`DS41M_MODULE_RUNTIME=0` 一键回退 legacy）：
  由模块树驱动层内执行——
  `modules/runtime.py` 用 `compose.compose_layer` 生成执行计划（叶子序 = 树 DFS 序），
  runner 首次使用会与 legacy `run.expand_layer` **严格对拍**（`ModuleRuntimeOrderError`），
  不一致默认直接报错；`DS41M_MODULE_RUNTIME_FALLBACK=1` 才回退 legacy。
  **放行**：plain prefill、hit（stage A / fake stage B）、packed、CP；
  **强制 legacy**：增量 decode（`pos_offset>0`）、serve decode、DSpark。
  `--decode-mode full` 每步等价 prefill，模块树合法介入（A/B token 逐位一致）。
  叶子仍调 `exec_ops.run_op`，packed 分派与 `boundary_cut` 钩子不变。
  CPU 门禁：`python3.11 tools/test_module_runtime.py`（默认/pad/skip_moe/skip_engram/
  unfused/split 六口径）与 `check_module_spec.py` 的 `check_runtime_plan`。
  NPU 侧已验证：plain/hit/packed/CP2/pad 全绿、decode 负向通过；2026-10-24
  用户拍板**默认开**。

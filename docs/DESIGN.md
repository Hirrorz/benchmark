# ds41_mini 设计文档

> 版本：2026-10-26
> 代码基线：分支 `feat/mini-packed-dual-batch`，HEAD `87a3ad3`，工作区含未提交改动
> 读者：维护者、采集/分析同学、要新增算子或模块的开发者
> 关联文档：
> - `docs/PLAN.md`：早期 M0–M5 方案与里程碑（历史设计，部分已被实现超越）
> - `docs/CODE_ANALYSIS_AND_USAGE.md`：代码级分析与使用手册（命令、配置、模块地图）
> - `MODULE_DEV.md`：模块插件开发指南（配置即事实源）
> - `PARITY_STATUS.md`：数值对拍现状
> - `PORTABLE.md` / `PORTABLE_TROUBLESHOOTING.md`：便携包使用与缺件排查

---

## 目录

- [0. 摘要](#0-摘要)
- [1. 背景与目标](#1-背景与目标)
- [2. 设计原则](#2-设计原则)
- [3. 总体架构](#3-总体架构)
- [4. 核心子系统设计](#4-核心子系统设计)
  - [4.1 模型结构与几何](#41-模型结构与几何)
  - [4.2 算子图与模块系统](#42-算子图与模块系统)
  - [4.3 执行引擎](#43-执行引擎)
  - [4.4 权重子系统](#44-权重子系统)
  - [4.5 命中、replay 与 packed](#45-命中replay-与-packed)
  - [4.6 多卡与通信](#46-多卡与通信)
  - [4.7 性能设计](#47-性能设计)
  - [4.8 观测、验收与对拍](#48-观测验收与对拍)
  - [4.9 可移植与分发](#49-可移植与分发)
- [5. 关键流程](#5-关键流程)
- [6. 接口契约](#6-接口契约)
- [7. 质量门禁](#7-质量门禁)
- [8. 设计决策与权衡](#8-设计决策与权衡)
- [9. 风险与已知缺口](#9-风险与已知缺口)
- [10. 演进路线](#10-演进路线)
- [11. 附录](#11-附录)

---

# 0. 摘要

ds41_mini 是一个**不依赖官方模型代码**的 DS4.1-Flash 最小算子图 benchmark。它只靠
配置 + 配方 + 算子声明自己装配 40 层前向，在 Ascend NPU 上真跑，产出与官方
`ds41_light` bench 同构的 MindStudio 流水图，并支持真权重逐层数值对拍、命中、
packed、decode、常驻 bench 与便携分发。

设计上分三层：

| 层 | 回答的问题 | 主要载体 |
|---|---|---|
| **声明层** | 模型长什么样、每层有哪些算子、各多少次、接口是什么 | `configs/*.yaml`、`model_spec.py`、`assemble.py`、`recipes.yaml`、`opgraph.py`、`modules/specs/*.yaml` |
| **执行层** | 怎么把算子真正下发到 NPU | `exec_ops.py` + 各 `exec_*` 插件、`modules/runtime.py`（执行序） |
| **支撑层** | 权重从哪来、池/命中/打包怎么做、怎么观测和分发 | `weights*.py`、`batch.py`、`replay.py`、`alop.py`、`env_pack.sh`、`tools/*` |

三条主线：

1. **图忠实**：算子集合、次数、形状、顺序与官方 trace 对齐；
2. **数值可证**：真权重 + 同输入 + EP8 下，固定口径 40 层逐位一致；
3. **工程可移植**：任意一台有 Ascend 卡 + CANN 的机器，`env_pack.sh` + `selfcheck` 就能跑。

2026-10-24 起，执行序新增**模块树驱动**（Phase 2，默认开）：`recipes` 先被装配成
带容器路径的模块树，再用树 DFS 序驱动层内执行；首次使用会与 legacy
`run.expand_layer` 做严格逐项对拍，不一致默认直接报错。

---

# 1. 背景与目标

## 1.1 问题

官方 `ds41_light` bench 能给出真实流水图，但依赖官方仓库代码、官方运行时和共享盘环境，
迁移成本高、调用链不透明，且很难在别人占用机器时做小规模实验。我们需要一个：

- **最小**：只保留“权重怎么放、用哪些算子、算子怎么排”三件事；
- **独立**：不 import 官方 `adaptlib` / `executor/` / `models/`；
- **同构**：产出的 trace 能与官方逐算子对齐；
- **可移植**：不依赖共享盘也能在新机器跑起来；
- **可演进**：新增算子/层类型/模块时不需要动主流程。

## 1.2 目标

| 目标 | 判据 | 现状 |
|---|---|---|
| **图结构一致** | 同 config/同 seq 下，与官方 trace 的算子集合 + 次数一致、形状一致、单算子时长 ±10% | 已达成（见 `CALIBRATION.md`） |
| **数值一致** | 固定口径（deepgemm + QSLI 注入）真权重 EP8，40 层 hidden 逐位一致 | 已达成（`PARITY_STATUS.md`） |
| **端到端精度** | 16k 真 ids 的 next-token 与官方参考一致（`id=396 ' that'`） | 已达成 |
| **性能可采集** | 合成/真权重两套采集入口，输出 MindStudio trace + 稳态耗时 | 已达成 |
| **可移植** | 新机器 `env_pack.sh` + `selfcheck` 全绿；便携包不依赖共享盘 | 已达成（`PORTABLE.md`） |
| **可扩展** | 新增模块不改 Python 主流程；配置 + 门禁即可 | Phase 2 已落地 |

## 1.3 非目标

- 训练/反向/优化器；
- 通用推理服务（常驻 bench 是固定 8 rank 的性能探针）；
- 命中率曲线扫描、调度策略、交互式 TUI；
- 官方模型代码的逐行等价复刻（只复刻算子图与数值链路）；
- 在所有 Ascend 型号上保证同一性能（当前验证机型 Ascend950DT）。

## 1.4 关键数字

| 项 | 值 |
|---|---|
| 模型 | DS4.1-Flash，40 层 |
| hidden / moe_intermediate | 5120 / 2304 |
| 路由专家 / topk | 384 / 6 |
| 层角色 | win 0-1；c2a 2-19；c1a 20-39 |
| kv 源层 | 2、8、14、20 |
| index 源层 | 2、8、14、20、24、28、32、36 |
| candidate 源层 | 20 |
| engram 层 | 1、14 |
| 合成权重 EP8 prefill | 与官方同量级（按各自口径对比） |
| 真权重 EP8 16k prefill | 约 2370 ms（官方 2394.7 ms） |
| 真权重 EP8 16k hit | 约 742 ms |

---

# 2. 设计原则

## P1 不依赖官方代码

“自研”指**装配层与驱动层**，不是算子层：

- 不 import 官方 `adaptlib` / `executor/` / `models/`；
- 厂商算子通过 `torch.ops.custom` / `cann_ops_transformer` / `tile_kernels` /
  `deep_gemm` 直接调用；
- 官方只作为**参考 trace / 参考 dump / 参考配置**使用。

## P2 声明式单一事实源

| 事实 | 唯一来源 | 消费方 |
|---|---|---|
| 模型结构/维度/层角色表 | `model_spec.py` + `configs/*.yaml` | `assemble`、`weights_manifest`、`exec_*`、门禁 |
| 本次前向跑哪些层、每层 T | `assemble.build_stack()` | `run.py`、`runner.py`、`points_runner` |
| 每层有哪些算子、顺序 | `recipes.yaml` | dry-run、模块装配、runner |
| 算子 kernel 名/次数/形状 | `opgraph.py` | dry-run 直方图、trace 对齐、selfcheck |
| 模块接口（requires/provides/weights/params） | `modules/specs/model_modules.yaml` | 模块树、查看器、门禁、运行时探针 |
| 运行时执行序 | `modules/compose.py` → DFS | `modules/runtime.py` |
| 权重清单 | `weights_manifest.py`（由真权重索引生成） | 合成/真权重共用 |

原则：**能配置的不要写死；写死的必须是格式/硬件契约**（NZ=29、MXFP8 block=32 等）。

## P3 声明与执行分离

- `opgraph.OPS`：声明式算子表（kernel/次数/形状/affects）；
- `exec_ops.OPS`：可执行函数表（`key -> fn(st, ctx, w)`）；
- `modules`：模块接口层（上下文契约 + 容器树）；
- 三者靠**同名约定**耦合，dry-run 与真跑走同一条展开路径。

## P4 合成/真权重共用一条执行路径

`SyntheticSource` 与 `RealSource` 产出**同名、同 dtype、同 shape** 的权重，共用：

- `weights_manifest.layer_manifest()` 的每层清单；
- `weights.shard()` 的 EP/TP 切分；
- `runner.pack_layer_weights()` 的后端布局打包。

合成模式只替换“张量从哪来”，不替换执行逻辑。

## P5 图与执行同源

`run.ctx_for / expand_layer` 同时服务 dry-run 与 runner；模块运行时又用
`compose.compose_layer` 生成同一序列，并与 `expand_layer` 严格对拍。避免
“dry-run 一套、真跑一套”。

## P6 可观测、可对拍、可恢复

- ALOP/msprof trace；
- `--collect-layers` 逐层 hidden；
- `DS41M_DUMP_*` 逐算子/逐行快照；
- `selfcheck` 逐算子调用/健康；
- `compare_layers` / `align_ops` / `diff_traces` / `check_next_token`；
- 任何优化都要能被 trace 或数值对拍证伪。

## P7 可移植

- `env_pack.sh` 按“显式环境变量 → 包内 runtime → 共享盘”顺序探测；
- 便携包把 runtime 与 code 分层打包（base 只打一次，code 秒级）；
- `selfcheck` 作为新机器的验收门；
- 不依赖 PyPI、不依赖共享盘（用包内 runtime 时）。

## P8 渐进式演进

- 模块系统分 Phase 1（声明/导出/查看器）→ Phase 1.5（运行时探针）→
  Phase 2（模块树驱动执行）；
- Phase 2 可用 `DS41M_MODULE_RUNTIME=0` 一键回退 legacy；
- 严格对拍 + 可回退，避免“大爆炸式替换”。

---

# 3. 总体架构

## 3.1 分层视图

```text
入口/编排层
├─ run.py                 dry-run：不碰卡，输出算子序列/直方图
├─ runner.py              单 rank 真跑：prefill / hit / decode / serve / points
├─ launch_mini.py         多 rank supervisor：每 rank 一张物理卡
├─ points_runner.py       一次进程多点位/多批/per-DP
├─ tools/bench_server.py  常驻 8 rank supervisor（--serve）
└─ tools/bench_cli.py     常驻客户端

声明/几何层
├─ configs/*.yaml         模型、并行、运行、权重、后端、命中、特性
├─ model_spec.py          结构/维度/层角色表的单一事实源 + 聚合校验
├─ assemble.py            40 层栈、层角色、每层 T、hit/replay 几何
├─ recipes.yaml           每层算子顺序与存在性
├─ opgraph.py             算子声明表：kernel/次数/形状/affects
└─ modules/specs/*.yaml   模块接口契约 + 容器路径映射 + 键别名

执行层
├─ modules/compose.py     recipes → 带容器路径的步骤序列 + 模块树
├─ modules/runtime.py     模块树 DFS → 可执行叶子序；与 legacy 严格对拍
├─ exec_ops.py            OPS 注册表 + run_op 分派 + 池/rope/QMM/元数据
├─ exec_indexer.py        idx_prolog / QLI / QSLI（+ packed 变体）
├─ exec_moe.py            MegaMoE ascendc/deepgemm、TP、EP-decode
├─ exec_engram.py         取表/侧流/ElasticBuffer
├─ exec_dspark.py         DSpark 草稿模型（serve 用）
└─ exec_ops_packed.py     packed compressor

支撑层
├─ weights.py / weights_manifest.py   合成/真权重、切分、加载、释放
├─ engram_layout.py                   压缩词表/素数桶/n-gram hash
├─ decode_layout.py                   P→D / D→P 专家布局转换
├─ batch.py                           packed varlen 布局算术
├─ replay.py                          decoder replay 尾段裁剪
├─ alop.py / profiling.py             profiling 后端
├─ ds41_log.py / progress*.py         日志与进度
└─ cp_layout.py / cp_state.py         CP 行布局与状态（按需）

工具/验收层
├─ tools/selfcheck.py                 算子自检（Tier0/Tier1）
├─ tools/compare_layers.py            collect 逐层对拍
├─ tools/align_ops.py / diff_traces.py 直方图/trace 对拍
├─ tools/op_time_table.py             8 rank 耗时汇总
├─ tools/run_hit.sh / run_real.sh     两套采集入口
├─ tools/model_viewer.html            模块/结构查看器（生成物）
├─ tools/serve_viewer.py              在线查看器（SSH 端口转发）
├─ tools/view_model.py                CLI/TUI 查看器
├─ tools/new_module.py                新模块 YAML 生成器
└─ tools/make_portable_kit.sh         便携包（runtime base + code）
```

## 3.2 模块职责

| 模块 | 职责 | 关键入口 |
|---|---|---|
| `model_spec.py` | 结构参数单一事实源、聚合校验、进程级单例 | `set_spec()` |
| `assemble.py` | 层栈/角色/每层 T/hit 与 replay 几何 | `build_stack()` |
| `recipes.yaml` | 每层算子顺序与存在性 | — |
| `opgraph.py` | 算子声明表（kernel/次数/形状/affects） | `OPS` |
| `modules/spec_loader.py` | 加载/校验模块 YAML、覆盖目录、参数求值 | `load_spec_file()` |
| `modules/compose.py` | recipes → 容器路径/步骤/树（DFS 序 = 执行序） | `compose_layer()` |
| `modules/runtime.py` | Phase 2 执行计划、严格对拍、参数注入 | `build_plan()` / `verify_order()` |
| `modules/context.py` | 作用域解析（leaf→container→layer→pass→process）、Ctx | `Scope` / `Ctx` |
| `modules/runtime_probe.py` | `DS41M_CTX_TRACE` 读漏/写漏探针 | `wrap_ctx()` / `summary()` |
| `exec_ops.py` | 执行表、dispatch、池/rope/QMM/元数据 | `run_op()` |
| `exec_indexer.py` | 索引器 QLI/QSLI、索引 K 缓存 | `_qli()` / `_qsli()` |
| `exec_moe.py` | MegaMoE 三路径、HCCL 组、后端适配 | `_mega_moe()` |
| `exec_engram.py` | 取表/侧流/ElasticBuffer | `_engram_fetch()` |
| `weights.py` | 合成/真权重 Source、切分、host cache | `make_source()` |
| `batch.py` | packed 布局算术（cu/位置/块表/裁尾） | `cu_of()` / `tail_slice()` |
| `replay.py` | decoder replay 策略与尾段裁剪 | `apply()` |
| `runner.py` | 单 rank 主控、pass 循环、图、serve/points | `run_forward()` |
| `launch_mini.py` | 多 rank 启动与日志聚合 | `main()` |

---

# 4. 核心子系统设计

## 4.1 模型结构与几何

### 4.1.1 ModelSpec

`ModelSpec` 是 dataclass，默认值逐字等于早期硬编码值。`build(cfg)` 从
`cfg["model"] / ["run"] / ["parallel_config"]` 覆盖；`validate()` 做聚合校验：

- 层表长度/取值/越界/重复；
- `kv_source_layers ⊆ index_source_layers`；
- `candidate_source_layer` 必须在 kv 源层且非 win 层；
- `topk ≤ n_routed_experts`，专家数能被 `moe_ep_size` 整除；
- `block_size == 128`、滑窗 128 对齐、量化块整除各维度；
- `engram_num_embeddings` 与 `engram_layers` 一一对应。

**设计意图**：结构参数以前散落在 `exec_*` 常量、`weights_manifest` 字面量、
`assemble` magic number 三处；现在集中到一处，换结构只改配置，写错在启动时
fail-fast。

### 4.1.2 层角色

`assemble.attention_kind()` 由五张表推出 6 种层类型：

```text
ratio == 0                         → win
idx in kv_source_layers:
    ratio == 1                     → c1a_source
    else                           → c2a_source
idx in index_source_layers         → c1a_indexsource
else:
    ratio == 1                     → c1a_consumer
    ratio == 2                     → c2a_consumer
```

`build_stack()` 为每层生成：

```python
{idx, ratio, attn, is_kv_src, is_idx_src, is_candidate, engram, T}
```

角色只影响两件事：

- **存在性**：recipes 按 `attn` 类型选段；
- **数量**：`opgraph.count(ctx)` 根据 `is_kv_src / is_idx_src / engram` 做增量，
  例如 `rope = 3 + is_kv_src`、`dyn_mx_quant = 6 + is_idx_src + engram`。

### 4.1.3 每层 T 与 hit/replay/packed

```python
# 命中：按 128 对齐，与官方 light bench 同口径
M = round((1 - h) * L / 128) * 128
R = M + replay_tail                 # replay_tail 默认 128
T_main = R if h > 0 else L

# packed：本 pass 行数 = ΣR_i（逐序列可不同 L/hit）
T_pass = sum(row_lens)

# decoder replay：cut 之后只跑最后 W 行（W = sliding_window）
T_layer = W * n_seqs if (cut and idx >= cut) else T_pass
```

`meta` 同时携带 `T_main / T_mains / T_pass / n / cut / w / L / hit / lens / pos_offs`，
供 runner 做位置、池子、压缩长度与逐序列偏移。

**设计意图**：所有“这次前向到底算多少 token”的公式只写一处，避免
assemble/run/runner 各写一遍导致漂移。

---

## 4.2 算子图与模块系统

### 4.2.1 三张表的分工

| 表 | 回答 | 约束 |
|---|---|---|
| `recipes.yaml` | **哪些层有它、按什么顺序** | 只写名字与 `_段` 引用；不写次数 |
| `opgraph.py` | **每层跑几次、trace 里叫什么、形状** | `count(ctx)`；0 次不进直方图 |
| `modules/specs/model_modules.yaml` | **它的接口契约**（requires/provides/weights/params/flags/container） | 每个模块只声明自己；父容器不汇总子节点 I/O |

三者的关系：

```text
recipes.yaml ──存在性/顺序──┐
                            ├─► dry-run：opgraph.count → 直方图
opgraph.py ──次数/kernel────┘

recipes.yaml ──compose──► 步骤序列 + 容器树 ──DFS──► 执行序
modules/specs ──接口/容器/顺序约束──┘
```

### 4.2.2 为什么需要模块层

早期只有 `recipes + opgraph + exec_ops`：能跑，但“层内每个算子的输入输出契约”
只存在于代码里，新增算子/调整顺序靠人脑。模块层把契约显式化：

- `requires`：模块需要的键（`hidden.x` / `model.hidden` / `cache.cmp.pool` …）；
- `provides`：模块写出的键，完成后上浮到层工作集；
- `weights`：从权重清单读的键，支持专家模板；
- `params`：模块参数，支持 `"${model.hidden}"` 引用；
- `flags`：`implicit`、`packed_variant`、`boundary_only`、`runner_hook`、
  `no_impl`、`placeholder`、`trace_only` 等；
- `container / after`：装在哪、顺序约束。

**不变量**：模块顶层不 import torch/exec_ops（导出器/门禁必须纯 CPU 可跑）；
每个 ctx/st 读写键都要声明（`DS41M_CTX_TRACE=1` 会抓读漏/写漏）。

### 4.2.3 模块装配与执行序

`modules/compose.py`：

1. `tracked_sequence()` 按 `recipes` 展开，记录最内层段名；与
   `run.expand_layer` 同序、同过滤（`DS41M_PAD_OPS`、`DS41M_SKIP_MOE`、
   `DS41M_SKIP_ENGRAM`）；
2. `recipe_map.path_of(seg, op, occurrence)` 把 `(段, op, 出现序号)` 映射到容器路径；
3. `build_tree()` 按执行序插入容器/叶子，**每次插入都追加到当前路径链尾部**，
   因此 DFS 序 == 执行序；split 视图“进入→离开→再进入”会开 `#k` 容器，避免顺序错乱；
4. `Step` 记录 `op/path/segment/occurrence/rows_phase/view`。

`modules/runtime.py`：

- `build_plan(cfg, recipes, layer, view)` → `LayerPlan`；
- `verify_order(plan, legacy_ops)` 与 legacy 序列**逐项**对拍，不一致抛
  `ModuleRuntimeOrderError`；
- runner 首次使用某层计划时对拍并缓存 `_MODULE_PLAN_CACHE`；
- `DS41M_MODULE_RUNTIME=0` 一键回退；`DS41M_MODULE_RUNTIME_FALLBACK=1`
  允许顺序不一致时软回退 legacy；
- 放行：plain prefill、hit（stage A / fake stage B）、packed、CP；
- 强制 legacy：增量 decode（`pos_offset>0`）、serve decode、DSpark；
- 叶子仍调用 `exec_ops.run_op`，packed 分派与 `boundary_cut` 钩子不变。

### 4.2.4 模块目录

`modules/specs/model_modules.yaml` 当前 41 个模块、18 个容器：

| 分类 | 数量 | 代表 |
|---|---:|---|
| moe | 12 | `gate`、`topk_gate`、`mega_moe`、`dispatch`、`gmm`、`combine` |
| attention | 10 | `wq_a`、`wq_b`、`wkv`、`rope`、`mqsmla`、`wo_a`、`wo_b`、`kv_epilog` |
| indexer | 4 | `idx_prolog`、`qli`、`qsli`、`idx_k_write` |
| engram | 3 | `engram_fetch`、`engram_wait`、`engram_proj` |
| boundary | 3 | `boundary_cut`、`gather_tail`、`scatter_tail` |
| common | 3 | `hc_pre`、`hc_post`、`rmsnorm` |
| placeholder | 4 | `mul`、`add`、`cast`、`tmove` |
| compressor | 1 | `kv_compress` |
| implicit | 1 | `dyn_mx_quant` |

容器：`attn`、`attn.compressor`、`attn.core`、`attn.hc`、`attn.indexer`、
`attn.indexer.q_proj`、`attn.indexer.search`、`attn.kv_indexer`、`attn.kv_proj`、
`attn.o_proj`、`attn.q_proj`、`engram`、`ffn`、`ffn.hc`、`ffn.moe`、
`ffn.moe.experts`、`ffn.moe.routing`、`layer.epilogue`。

### 4.2.5 Phase 1 / 1.5 / 2

| Phase | 内容 | 状态 |
|---|---|---|
| 1 | 声明 + 导出 + 查看器（运行时不改） | 完成 |
| 1.5 | `DS41M_CTX_TRACE=1` 探针 + 别名视图 | 完成（默认关） |
| 2 | 模块树驱动层内执行；与 legacy 严格对拍 | **默认开（2026-10-24）** |

Phase 2 的边界是刻意的：decode/DSpark 有额外状态机，先强制 legacy；等模块运行时
在这些路径上完成 A/B 后，再逐步放开。

---

## 4.3 执行引擎

### 4.3.1 两套注册表

| | `opgraph.OPS` | `exec_ops.OPS` |
|---|---|---|
| 性质 | 声明式：kernel/次数/形状/affects | 可执行：`key -> fn(st, ctx, w)` |
| 用途 | dry-run 直方图、trace 对齐、selfcheck 覆盖表 | 真跑 dispatch |
| 耦合 | 同名约定 | 同名 key 查表 |

几个刻意保留的“非算子”：

- `dyn_mx_quant`：隐式，由量化算子内部产生，dry-run 计数、执行侧跳过；
- `mul/add/cast/tmove`：占位语义（数值恒等），默认不下发，`DS41M_PAD_OPS=1` 才进 trace；
- `gather_tail/scatter_tail`：声明但无实现，属命中路径；
- 未融合 MoE 链（`gating`…`finalize`）：当前**没有执行实现**，只有 ascendc/deepgemm
  融合路径可跑。

### 4.3.2 dispatch

`run_op(key, st, ctx, w)`：

1. 查 `OPS[key]`；
2. 若 `ctx` 有 `cu_seqlens`（`batch.is_packed`），优先 `OPS[key + "@packed"]`，
   没有则回落普通实现；
3. 没有实现返回 False，runner 记为 skipped；
4. 调用 `fn(st, ctx, w)`；
5. 可选 `DS41M_MEMDBG` / `_dump_tensors`。

`@op_packed` 只覆盖契约差异最大的三个算子：

- `kv_compress@packed`：一次 batched `ds41.compressor`；
- `qli@packed` / `qsli@packed`：一次 batched 索引器。

其余算子在普通实现内部按 `cu` 处理（`kv_epilog`、`mqsmla`、`rope`、`wo_a`、
`idx_prolog`）。

### 4.3.3 插件加载

`exec_ops.ensure_ops(cfg)`：

1. import `custom_ops`、`cann_ops_transformer`、`cann_ops_nn`（顺序敏感；
   只 import `cann_ops_transformer` 即可，额外的 experimental/shim 会让 `ds41`
   子组注册 0 个算子）；
2. `_apply_spec(cfg)` 用 `model_spec` 刷新模块级常量；
3. 按固定顺序导入 `exec_indexer → exec_moe → exec_engram`，各自
   `register(OPS, SKIP, core, cfg)`；
4. 若存在 `exec_ops_packed.py`，再 `register(OPS, core)`。

### 4.3.4 错误处理

- 逐算子异常被捕获，记录 `(layer, key, msg@file:line)`；
- 系统性错误（同一条 ≥3 次）→ `sys.exit(3)`；
- `DS41M_MOE_STRICT` / `DS41M_DECODE_STRICT` 控制是否对 MoE 失败硬退出；
- 算子缺少实现时显式 skipped，不静默当成功。

---

## 4.4 权重子系统

### 4.4.1 每层清单

`weights_manifest.py` 由真权重 `model.safetensors.index.json` + 48 个分片头生成，
三张表：

- `COMMON`：40 层都有的 attention/norm/gate/共享专家/hc；
- `COND`：只在部分层（compressor/indexer/engram）；
- `EXPERTS`：每层 × 每专家的 `w1/w2/w3`（mxfp4，I8）。

`cond_holders(cfg)` 按 config 推导 COND 持有层，不再依赖字面层号列表。

### 4.4.2 两条 Source

| | `SyntheticSource` | `RealSource` |
|---|---|---|
| 构造 | 按 manifest 形状/ dtype 随机生成 | safetensors mmap 读取 |
| dtype 分布 | F8_E4M3 0..126、E8M0 120..135、I8 -120..120、BF16/F32 randn | 真值 |
| engram 表 | `HOST_RESIDENT` 强制 CPU；行数可缩小 | 读真表，host 分片 |
| 按层释放 | `lazy_experts`（单卡默认 true） | `resident` / `lazy_experts` |
| 跨层共享 | `share_across_layers` | 不适用 |
| host cache | 无 | pinned host / memfd 跨进程 |

统一接口：`kind / expert_range / get / iter_layer / invalidate_experts / release /
summary`。

### 4.4.3 切分

- **EP**：连续切专家，rank r 拿 `[r*E/ep, (r+1)*E/ep)`；384/8 = 48/rank；
- **MoE TP**：本 rank 拥有全部 384 专家 × `I/tp`；非均匀 128 对齐切分，
  I=2304、tp=8 → `[384,384,256,256,256,256,256,256]`；
- **Attention TP**：按头维切，当前 `attn_tp_size=1` 不启用；
- **engram TP**：`model.engram_tp_size`，真正分片在 `exec_engram._vendor_buffer`
  （每卡 `ceil(rows/tp)`，不足补零）。

### 4.4.4 打包

`runner.pack_layer_weights()` 统一完成：

- mxfp8/mxfp4 的 transpose / NZ(29) / fp4 打包；
- gate 转置、bias fp32、k_norm/compressor norm fp32；
- wq_b scale 配对；
- 合成模式按“去层号”缓存，真权重按层缓存；
- 真权重打包后释放 6 个 raw ND fp8 权重。

### 4.4.5 真权重 host cache / resident

- `host_cache: true`：首次读入 pinned 主机内存，之后 H2D；40 层首遍 52s → 再遍 1.2s；
- `resident: true`：40 层全常驻（EP8 可行）；
- `DS41M_WEIGHT_HOST_CACHE=0` 关闭；
- `DS41M_MOE_HOST_CACHE=<socket>`：跨进程 memfd 共享专家权重（常驻 bench 用）。

---

## 4.5 命中、replay 与 packed

### 4.5.1 命中几何

```python
M = round((1 - h) * L / 128) * 128
R = M + replay_tail
```

- 只算尾部 `R` 个 token；
- 位置是**绝对位置** `L-R .. L-1`；
- 池子/块表/槽位按 `L` 开；
- 索引器与压缩注意力的可见长度 `seqused_k = L//ratio`；
- 窗口分支不回看前缀；
- replay 段的压缩 KV 写入抑制（`slot=-1`）。

### 4.5.2 fake / real

| | `fake` | `real` |
|---|---|---|
| stage A | 不跑 | 先真算前缀 `[0,H)`，写 shared 池 |
| 前缀内容 | 池初值（零） | 真实 KV/压缩 KV/索引缓存/压缩器 state |
| 数值含义 | 只对形状/次数/耗时 | 与 prefill 数学等价 |
| 适用 | 合成权重、性能矩阵 | 真权重、数值对拍 |
| packed | 支持 | 不支持（runner 直接拒绝） |

`real` 要求 `H % ratio == 0` 且 `off % ratio == 0`；否则前缀末组 carry state
无法精确复原（会打印提示）。

### 4.5.3 replay

- `decoder_replay` 是**有界近似**：cut 之后只跑最后 W 行，W = sliding_window；
- 不是逐位等价（W ≥ 序列长时才是）；
- `replay.apply()` 按 packed/single 两种尾段裁剪，逐序列前移 `pos_offs`；
- 与 graph tail 配合：仅当 replay 真正裁了、非 prof/collect/dump 时入图。

### 4.5.4 packed

- 多请求 token 维拼接 + `cu_seqlens`；`T_pass = ΣR_i`；
- 逐序列可用不同 L/hit（v2 点位）；
- b=1 也默认 packed（2026-10-21 统一），只有 packed 不支持的组合
  （`--decode / --layers / --prompt / real 前缀命中`）自动回退 sequential；
- `split` 视图：层 20 的 `c1a_source_split`，`boundary_cut` 后 q/indexer/attn/MoE
  只跑 compact 行；CP 下走 `boundary_indices/compact_boundary_index` 三步行重分布。

**关键不变量**：

- `seqused_k / cmp_len` 必须是**绝对可见压缩长度** `(off+T)//ratio`；
- replay 段压缩行必须 `slot=-1`；
- 每 pass 按 `row_lens` 重建 `cu`，b=1 也要重建；
- packed indexer 断言 `cu[-1] == T`。

---

## 4.6 多卡与通信

### 4.6.1 启动隔离

`launch_mini.py`：

- 每 rank 一个 `runner.py` 子进程；
- `ASCEND_RT_VISIBLE_DEVICES=<cards[r]>`，子进程逻辑卡恒 `npu:0`；
- `MASTER_ADDR/PORT` 自动选空闲端口；
- `DS41M_NO_VISIBLE=1` 关闭卡隔离（排查 topo/rank 映射）。

### 4.6.2 通信域

| 通信域 | 用途 | 创建 |
|---|---|---|
| HCCL WORLD | 默认集合通信、MoE 对称缓冲 | `dist.init_process_group("hccl")` |
| MegaMoE 组 | 对称缓冲 | 复用 WORLD；单卡自建 1-rank 组 |
| engram 专用组 | 每层一个（层 1/14） | `dist.new_group(ranks=all)` |
| MC2 域 | decode dispatch/combine | `exec_moe._mc2_group()` |

### 4.6.3 同步点

| 同步 | 开关 |
|---|---|
| 起跑线 barrier | 默认 1 次（world≥2）；`DS41M_START_NOBARRIER=1` 关 |
| 每 pass barrier | `DS41M_PASS_BARRIER=1`（等待不计入 pass） |
| decode 每步 barrier | `DS41M_DECODE_BARRIER=1` |
| 常驻 token 广播 / cancel | 自动 |

### 4.6.4 并行形态

- EP8（默认）：每 rank 48 专家；
- TP8（实验）：全部 384 专家 × I/8；
- CP（按需）：`cp_layout.py` / `cp_state.py`，行重分布与 boundary compact；
- DP：mini 每 rank 一条序列；per-DP 仅在 points 的 `dps` 实验形态中支持。

---

## 4.7 性能设计

性能优化遵循一条纪律：**先证明热点，再改；改完必须能被 trace/对拍证伪**。

| 设计 | 作用 | 备注 |
|---|---|---|
| MoE 分块 `moe_chunk_max_len` | 控制单次 MoE workspace，避免 16k 顶穿 HBM | `chunks = ceil(T/chunk)` |
| packed 统一 | b=1 也走 varlen，减少分支/特殊路径 | sequential 仅 A/B |
| `_PACK_CACHE_L` / `_PACKED_SUB_CACHE` | 权重打包/子字典跨层复用 | 合成/真权重分别缓存 |
| `_CTX_CACHE` / `_SEQ_CACHE` | 层 ctx / 算子序列复用 | 几何变化时清 |
| META 按几何清 | 避免每 pass 阻塞 H2D | `_META_GEO` |
| engram 预取 | 侧流提前取表；host 提前 2 层 | `DS41M_NO_ENGRAM_PREFETCH` |
| graph tail | prefill 尾段固定 W 行入图 | 仅 replay 裁尾时 |
| decode graph | T=1 全 40 层 NPUGraph | `DS41M_DECODE_GRAPH=1` |
| boundary split | 层 20 KV 全长 → compact 512 行 | `DS41M_BOUNDARY_SPLIT=1` |
| pure mode | 热循环不打印/不取墙钟 | `--log-level pure` |
| host 侧优化 | 元数据每 pass 一次、池 slot 缓存等 | 见 WORKPLAN |

性能采集口径：

- profile step = **每层一次**；step 数 = `iters × 层数`；
- 第 N 轮包含 trace 导出，看稳态用中间轮；
- `--warmup-iters K` 在 `profiler.start()` 前跑，不污染 trace；
- `--prof-ranks all|0|0,3` 控制哪些 rank 落 trace；
- 稳态统计剔除第 0 轮、图捕获轮及其后一轮；样本 ≥3 取中位数，否则取均值；
- 真权重首 pass 含权重装载/H2D，必须用 warmup 或剔除，否则 trace/耗时被污染。

---

## 4.8 观测、验收与对拍

### 4.8.1 四类独立证据

| 证据 | 回答 | 工具 |
|---|---|---|
| dry-run 直方图 | 图“应该”长什么样 | `run.py --dry-run` + `align_ops.py` |
| 真实 trace | 图“实际”长什么样、耗时多少 | ALOP/msprof + `diff_traces.py` |
| collect hidden | 数值逐层是否一致 | `compare_layers.py` |
| next-token | 端到端精度是否对齐 | `check_next_token.py` |

ALOP 有两种引擎：**aclprof C 级**（默认，轻量，落 `PROF_*` 原始目录）与
**torch_npu.profiler**（`python_profiling_config.enabled: true`，落
`ASCEND_PROFILER_OUTPUT/` + `trace_view.json`，MindStudio 直开）。
配置见 `configs/alop_mindstudio*.yaml`；ALOP 失败不会向被采集程序抛异常。

### 4.8.2 selfcheck

- Tier 0：python/torch/torch_npu/yaml、CANN 路径、厂商命名空间、23 个算子入口、NPU 可见性；
- Tier 1：合成权重真跑 0..24 层（覆盖全部层角色），在 `exec_ops.run_op` 上挂钩子，
  逐算子统计调用/失败/跳过/耗时，检查输出健康；
- 未覆盖算子按 `opgraph.OPS` 逐条列出并写原因；
- 退出码：0 全过 / 1 有算子失败 / 2 Tier0 挂。

### 4.8.3 数值对拍

- `collect_rank{R}.pt = {"layers": {层号: Tensor[T,H] fp32}, "rank", "n"}`；
- 判据：相对误差 `<1e-3` ✅、`<5e-2` ⚠️、`≥5e-2` ❌；
- 深层受官方 run-to-run 混沌放大影响，逐层表不能当回归门；端到端看 next-token；
- 固定口径下已做到 40 层逐位一致（deepgemm + QSLI 注入）。

### 4.8.4 模块门禁

| 门禁 | 断言 |
|---|---|
| `tools/test_module_config.py` | 配置加载/覆盖/新增模块 |
| `tools/check_module_spec.py` | 40 层 DFS 叶子序 == `run.expand_layer`（6 口径）；recipes 算子都有插件；权重双向覆盖；requires 可解析；形状可求值；确定性 |
| `tools/test_module_runtime.py` | 默认/pad/skip_moe/skip_engram/unfused/split 六口径 |
| `tools/export_model_html.py --check` | 导出确定性 |
| `tools/test_model_viewer.js` | 查看器结构/交互（jsdom） |
| `tools/diff_digest.py` | A/B 日志摘要差异 |

---

## 4.9 可移植与分发

### 4.9.1 env_pack

`env_pack.sh` 的探测顺序（先命中先用）：

1. 显式环境变量（`DS41_PACK_RUNTIME` / `DS41_CANN` / `DS41_DEPS` / `TILELANG_RT` /
   `DS41_PYTHON` / `DS41_TORCH_SITE`）；
2. 包内 `runtime/`；
3. 共享盘老路径。

关键点：

- CANN **不校验版本**，只要能跑通；
- 厂商算子包按目录扫描拼 `ASCEND_CUSTOM_OPP_PATH` / `LD_LIBRARY_PATH`；
- `PYTHONPATH` 顺序照抄验证过的老配方（顺序反了 `ds41` 命名空间注册 0 个算子）；
- `unset TMPDIR`（SFS 上 TMPDIR 会让 CANN JIT 的 rmtree 失败）；
- 末尾生成 `env_local.sh`，之后新 shell 只需 source 它。

### 4.9.2 便携包

`tools/make_portable_kit.sh` 分层打包：

```text
ds41_mini_runtime_base_<tag>.tar.gz   # runtime ~十几 GB，基本不变
ds41_mini_code_<时间>.tar.gz          # 代码，秒级
```

- 默认带 torch/CANN，可用 `--no-torch` / `--no-cann` 裁剪；
- `--rebuild-base` 只在 runtime 变更时用；
- 目标机要求：NPU 驱动 + python3.11；镜像自带 CANN/torch 时可不带；
- 解包后 `source env_pack.sh` + `selfcheck` 验收。

### 4.9.3 在线查看器

- `tools/model_viewer.html`：离线单文件（生成物）；
- `tools/serve_viewer.py`：SSH 端口转发 + 1.5s 自动刷新 + `/api/data`；
- `tools/view_model.py`：终端 CLI/TUI；
- `modules/export_model_html.py`：从模块规格 + 配方 + 层栈导出数据。

---

# 5. 关键流程

## 5.1 dry-run

```text
run.py:main
  ├─ assemble.load_yaml
  ├─ assemble.build_stack → 40 层 + 每层 T
  ├─ for layer:
  │    ├─ ctx_for(layer) → T/H/I/ratio/roles/suffix
  │    ├─ expand_layer(recipes, layer) → 算子名序列
  │    └─ apply_ops → opgraph.count → kernel 直方图
  └─ extras（会话级残差）→ hist
```

## 5.2 单 pass prefill（模块运行时开）

```text
run_forward
  ├─ PASS_NO += 1
  ├─ packed cu 重建
  ├─ assemble.build_stack → 层栈
  ├─ engram 预取排程
  ├─ 图判定（tail/decode）
  ├─ shared 池初始化
  ├─ for layer:
  │    ├─ ctx 补齐（含 cu/lens/pos_offsets/engram_*）
  │    ├─ 权重装载 → strip_prefix → pack_layer_weights
  │    ├─ legacy_seq = expand_layer(...)
  │    ├─ build_plan(compose) → verify_order(plan, legacy_seq)
  │    ├─ for op in plan.ops:
  │    │    └─ exec_ops.run_op(op, st, ctx, w)
  │    ├─ profiler.step()
  │    └─ post_layer_cb（serve W1 建 AC/NZ）
  └─ 图捕获/重放、profiler.stop、池跨 pass 保留
```

## 5.3 命中（real）

```text
main
  ├─ 判定 hit.source=real
  ├─ _shared = {}
  ├─ run_forward(hit_stage="A")   # 前缀 [0,H) 真算，写入 shared；不进 profile
  └─ 测量循环:
       run_forward(hit_stage="B") # 只算尾部 R；复用 shared
```

## 5.4 packed pass

```text
cfg["_packed"] = {n, lens, row_lens, cu}
  ├─ T_pass = Σ row_lens
  ├─ run_forward 每 pass 按 row_lens 重建 cu
  ├─ exec_ops.run_op: 有 cu → 优先 @packed 实现
  └─ assemble/build_stack: cut 后 T = w * n
```

## 5.5 decode

- full：每步全量重算，金标准，O(N²)；
- inc：prefill 一次 + 每步 T=1，读持久池；强制 `DS41M_PERLAYER_CACHE=1`；
- graph：`torch.npu.NPUGraph` 捕获 T=1 尾段/全 40 层；首两个 forward eager；
- 多步 decode 建议 `DS41M_DECODE_BARRIER=1` 或 `DS41M_ENGRAM_HOST=1`。

## 5.6 常驻 serve

- `bench_server.py` 起 8 个 `runner.py --serve`；
- W1 干净 prefill → W2 decode 布局/1 步 decode → W3 长度阶梯；
- 请求间 `reset_request_state`：清 ctx/seq/ring/engram 预取/META，保留 AC/NZ/packed/图；
- 布局模式：`ep8` / `ep8-fidelity` / `tp8-fidelity`（实验）/ `tp8-dual`（未实现）。

---

# 6. 接口契约

## 6.1 CLI 入口

| 入口 | 用途 |
|---|---|
| `run.py <cfg> --dry-run` | 不碰卡，输出算子序列/直方图 |
| `runner.py <cfg> --seq ...` | 单 rank 真跑 |
| `launch_mini.py --config ... --world N --cards ...` | 多 rank |
| `runner.py --points <json>` | 多点位 |
| `runner.py --serve --control ...` | 常驻（由 bench_server 调） |
| `tools/run_hit.sh` / `run_real.sh` | 两套采集入口 |
| `tools/selfcheck.py` | 验收门 |

## 6.2 YAML schema

7 段：`model / parallel_config / run / weights / backends / hit / features`。
字段含义与默认值见 `docs/CODE_ANALYSIS_AND_USAGE.md` 第 4 章。

## 6.3 模块接口

```yaml
- op: my_op
  category: attention
  container: attn.core
  impl: exec_ops:_my_op
  kernel: MyKernel
  after: [mqsmla]
  requires:
  - {key: hidden.x, type: Tensor, shape: "[T,H]"}
  provides:
  - {key: attn.my_out, type: Tensor, shape: "[T,H]"}
  weights:
  - {suffix: attn.my_op.weight}
  params:
    scale: "${model.hidden}"
  flags: []
```

硬规则：

1. 模块顶层不 import torch/exec_ops；
2. 每个 ctx/st 读写键都要声明；
3. 权重必须能对上 `weights_manifest`（或显式加入忽略清单）；
4. 新算子要能被 `recipes.yaml` 引用。

## 6.4 执行函数

```python
def _my_op(st: dict, ctx: dict, w: dict) -> None:
    ...
```

- `st`：激活/中间张量表；
- `ctx`：层上下文（T/H/I/ratio/roles/pos_offset/packed/…）；
- `w`：本层权重（含 `_packed` 布局缓存）。

## 6.5 数据格式

| 数据 | 格式 |
|---|---|
| collect | `{"layers": {i: Tensor[T,H] fp32}, "rank", "n"}` |
| trace | `<out>/<host>_<pid>_<ts>_ascend_pt/ASCEND_PROFILER_OUTPUT/*.csv` |
| points v2 | `[{"tag", "batches":[{L,hit}]}]`；per-DP 用 `dps` |
| 常驻状态 | 进程内；跨请求保留 AC/NZ/packed/图，清 ctx/ring/池 |

## 6.6 关键环境变量

| 变量 | 作用 |
|---|---|
| `DS41M_MOEMODE` | `auto/ascendc/deepgemm/decode` |
| `DS41M_MOE_CHUNK` | MoE 分块（`auto`/数字） |
| `DS41M_DECODER_REPLAY` | `1` 或 JSON `{on,w,cut,force}` |
| `DS41M_GRAPH_TAIL` | prefill 尾段入图 |
| `DS41M_DECODE_GRAPH` | decode 图（`1/check/tail/eager`） |
| `DS41M_MODULE_RUNTIME` | Phase 2 模块执行（默认 1；0 回退 legacy） |
| `DS41M_MODULE_RUNTIME_FALLBACK` | 顺序不一致时软回退 |
| `DS41M_BOUNDARY_SPLIT` | 层 20 边界拆分（默认 1） |
| `DS41M_PERLAYER_CACHE` | 池/缓存按拥有者层隔离 |
| `DS41M_PAD_OPS` | 恢复占位算子 |
| `DS41M_CTX_TRACE` | 模块 ctx 读写探针 |
| `DS41M_DUMP_*` | 逐算子/逐行诊断 |
| `DS41M_LOG_LEVEL` | `pure/error/warn/info/debug` |
| `DS41M_UNIFY_PACKED` | points 默认 packed（默认 1） |

---

# 7. 质量门禁

## 7.1 提交前

```bash
python3.11 run.py configs/mini_single.yaml --dry-run
python3.11 tools/selfcheck.py --card N
python3.11 tools/test_module_config.py
python3.11 tools/check_module_spec.py --configs 'configs/*.yaml'
python3.11 tools/test_module_runtime.py
python3.11 -m modules.export_model_html --config configs/mini_ds41_flash.yaml --check
```

## 7.2 采集前

- `npu-smi info` 确认卡空；
- `source env_pack.sh` / `env_local.sh`；
- trace 盘空间 ≥ 2–4 GB/rank；
- 真权重采集确认 ids/参考 dump 在位；
- 采集脚本自带四条验收（自然路由、engram 就绪、trace 次数、next-token）。

## 7.3 变更分类

| 变更 | 必跑 |
|---|---|
| 新增算子/模块 | dry-run + selfcheck + check_module_spec + 对拍 |
| 改 recipes/opgraph | dry-run 直方图 + trace 对齐 |
| 改 runner/exec | selfcheck + 单卡冒烟 + 逐层对拍 |
| 改权重/切分 | 合成/真权重各一遍 + PARITY |
| 改 portable/env | 新机器 selfcheck + 便携包解包测试 |

---

# 8. 设计决策与权衡

| # | 决策 | 理由 | 代价 |
|---|---|---|---|
| D1 | 不 import 官方代码，只复刻图/数值链路 | 可移植、可读、可控 | 需要自己维护算子清单与形状 |
| D2 | `recipes + opgraph` 两套表靠同名耦合 | 声明与执行分离，dry-run 快 | 名字写错时真跑会静默 skipped（靠 selfcheck 兜底） |
| D3 | 合成/真权重共用 manifest + shard + pack | 执行路径唯一，合成结果可迁移 | 合成权重需模拟 dtype/分布 |
| D4 | hit 用绝对位置 + 池按 L 开 | 与官方 hit 口径同构 | 池内存按 L 开，长序列占用大 |
| D5 | replay 有界近似（只跑最后 W 行） | 大幅省算力，命中/解码可用 | 非逐位等价；W 小于序列长时有近似 |
| D6 | packed 统一 b=1 | 去掉特殊路径，b=1/b>1 同一套代码 | 与 sequential 的数值不能直接逐位比 |
| D7 | 模块接口配置化（Phase 1/2） | 新增模块不改主流程；可视化/门禁 | YAML 与代码可能漂移，需要门禁 |
| D8 | Phase 2 用 legacy 序列做 oracle | 渐进替换，风险可控 | 两套序必须同步维护；decode/DSpark 暂不能进 |
| D9 | 便携包 runtime/code 分层 | runtime 大且稳定，code 频繁变 | 需要 base 版本管理；目标机仍需驱动 |
| D10 | 合成权重 `share_across_layers` | 生成一次、换层不释放，性能干净 | 只适用于合成；真权重不能开 |
| D11 | MoE 分块（官方同口径） | 控制 workspace，避免 OOM | trace 里每层多次 MegaMoe，需按块数断言 |
| D12 | ALOP 作为唯一 profiling 后端 | 两种引擎统一接口，失败不抛 | 纯 C 级模式没有 `op_statistic.csv` |

---

# 9. 风险与已知缺口

## 9.1 架构风险

1. **三套表漂移**：`recipes` / `opgraph` / `modules/specs` 任一不同步都会导致
   门禁红或运行时 skipped。对策：`check_module_spec` 的六口径对拍 + `selfcheck`。
2. **Phase 2 边界**：decode/DSpark 仍走 legacy；模块运行时在这些路径上还没有
   A/B。对策：`_module_runtime_here` 显式判定 + `DS41M_MODULE_RUNTIME=0` 回退。
3. **模块 YAML 假绿**：`DS41M_MODULE_SPEC_DIR` 指向不存在目录时会报错；
   但 `DS41M_SPEC_STRICT=0` 允许回退，需确认门禁环境。
4. **CP/boundary split**：层 20 的 compact 行重分布与图捕获耦合，CP 下只支持单序列；
   packed 多请求边界拆分待接。

## 9.2 数值/性能风险

1. **官方自身混沌**：run-to-run 1e-3 噪声会被网络放大；逐层对拍不能当回归门；
2. **MoE 后端不等价**：ascendc 与 deepgemm 数值不等价；对官方参考用 deepgemm；
3. **TP decode**：第 2 步起 MoE 入口 NaN（历史遗留）；低时延路径优先 EP；
4. **DSpark ep8**：草稿 EP 已跑通首步，但 ep8 inc T=1 gate NaN / full aicore 507015；
5. **per-DP**：只覆盖各 DP span 数相同的场景；大 L/多批需 dummy span 对齐；
6. **未融合 MoE**：`megamoe: none` 的链没有执行实现，只能用于 dry-run/声明；
7. **`run.batch` 占位**：真正 batch 入口是 `--batches` + `--batch-mode`；
8. **packed real-hit**：不支持，runner 直接拒绝。

## 9.3 工程风险

1. **共享卡/兄弟容器**：跑前 `npu-smi info`，等 HBM 回基线；
2. **背靠背 HCCL 未清场**：A→B 间隔太短会卡在起跑线 barrier；
3. **TMPDIR**：共享盘 TMPDIR 会让 CANN JIT 卡死；`env_pack.sh` 已 unset；
4. **逻辑卡号 ≥1**：aicore fault；永远物理卡可见 + 逻辑 npu:0；
5. **便携包 base/code 版本不匹配**：code 包需与 base tag 对齐；
6. **文档漂移**：README/COLLECT 文档与配置默认值可能不一致（如 `moe_chunk_max_len`），
   以代码/配置为准。

---

# 10. 演进路线

| 阶段 | 目标 | 关键工作 |
|---|---|---|
| **近期** | 模块系统收口 | decode/DSpark 接入模块运行时；`modules/specs` 与 `opgraph` 自动同步；门禁进 CI |
| **近期** | packed/per-DP 完整 | 任意 span 的 dummy 对齐；packed+CP 边界拆分 |
| **中期** | 采集自动化 | run_hit/run_real 结果入库；trace 断言/精度 gate 自动判定；一键 SUMMARY |
| **中期** | 可移植性 | 便携包在无共享盘机器全绿；CANN/torch 版本矩阵；selfcheck 的 Tier 0 覆盖更多缺件 |
| **长期** | 性能模型 | 用 trace + op_time_table 做逐算子耗时预测；boundary split/graph 的自动调参 |
| **长期** | 更多模型/后端 | 新模型配置化；ascendc/deepgemm/TP 的数值-性能对照矩阵 |

---

# 11. 附录

## 11.1 文件地图

| 路径 | 作用 |
|---|---|
| `README.md` | 上手、最小条件、自检、命令、坑 |
| `docs/DESIGN.md` | 本文：设计目标、架构、决策、权衡 |
| `docs/CODE_ANALYSIS_AND_USAGE.md` | 代码级分析与使用手册 |
| `docs/PLAN.md` | 早期 M0–M5 方案（历史） |
| `MODULE_DEV.md` | 模块插件开发指南 |
| `PORTABLE.md` / `PORTABLE_TROUBLESHOOTING.md` | 便携包与缺件排查 |
| `PARITY_STATUS.md` | 数值对拍现状 |
| `WORKPLAN.md` | 当前会话工作与风险 |
| `configs/` | 运行配置 |
| `recipes.yaml` | 每层算子配方 |
| `opgraph.py` | 算子声明表 |
| `modules/` | 模块接口、装配、运行时、查看器导出 |
| `modules/specs/` | 模块规格/容器映射/键别名/模型抽象 |
| `exec_*.py` | 执行插件 |
| `weights*.py` | 权重清单与 Source |
| `batch.py` / `replay.py` | packed / replay |
| `alop.py` / `profiling.py` | profiling |
| `tools/` | 自检、采集、对拍、查看器、便携包 |

## 11.2 模块系统文件

| 文件 | 作用 |
|---|---|
| `modules/specs/model_modules.yaml` | 41 个模块的接口/权重/参数/flags |
| `modules/specs/modules.d/*.yaml` | 单模块覆盖（可选，`override: true`） |
| `modules/specs/recipe_map.yaml` | 容器展示信息 + `(段,op,序号) → 路径` |
| `modules/specs/key_names.yaml` | 键别名/多义说明/命名空间 tier |
| `modules/specs/model_abstract.yaml` | 自动导出的整模型抽象（33 KiB） |
| `modules/compose.py` | recipes → 步骤/容器树（DFS=执行序） |
| `modules/runtime.py` | Phase 2 执行计划 + 严格对拍 |
| `modules/context.py` | 作用域解析/Ctx |
| `modules/runtime_probe.py` | 读写漏探针 |
| `modules/tree.py` | 模型树 JSON/HTML 数据 |
| `modules/export_model_html.py` | 导出 HTML |
| `tools/model_viewer.html` | 离线查看器（生成物，~874 KiB） |
| `tools/serve_viewer.py` | 在线查看器（SSH 端口转发） |
| `tools/view_model.py` | CLI/TUI 查看器 |
| `tools/new_module.py` | 新模块 YAML 生成器 |
| `tools/check_module_spec.py` | 模块/顺序/权重/形状/确定性门禁 |
| `tools/test_module_runtime.py` | 六口径运行时计划门禁 |

## 11.3 术语

| 术语 | 含义 |
|---|---|
| **pass** | 一次完整前向（一次 `run_forward`） |
| **step** | profiler 打点，每层一次；`step 数 = iters × 层数` |
| **iteration** | main 外层一轮，含 1 个（packed）或 N 个（sequential）pass |
| **layer role** | win / c2a_source / c2a_consumer / c1a_source / c1a_indexsource / c1a_consumer |
| **hit** | 前缀命中：只重算尾部 R 个 token |
| **stage A/B** | 真前缀先算一遍 / 被测量的命中尾段 |
| **replay** | decoder 尾段有界重放（cut 后只跑最后 W 行） |
| **packed** | 多请求 token 维拼接 + `cu_seqlens` |
| **boundary split** | 层 20 KV 全长算完 → compact 512 行（`DS41M_BOUNDARY_PAD`）继续算后续层 |
| **AC** | ascendc MegaMoE 的预转换权重 |
| **NZ** | FRACTAL_NZ 格式（format 29） |
| **P→D / D→P** | prefill 布局 → decode 布局 / 反向 |
| **QLI / QSLI** | QuantLightningIndexer / QuantSparseLightningIndexer |
| **MQSMLA** | mixed quant sparse flash MLA |
| **hc_pre / hc_post** | HyperConnection 前后处理 |
| **Phase 2** | 模块树驱动层内执行（默认开） |

## 11.4 推荐阅读顺序

1. `README.md` —— 先跑起来；
2. 本文第 1–4 章 —— 设计目标与架构；
3. `docs/CODE_ANALYSIS_AND_USAGE.md` —— 代码级细节与命令；
4. `MODULE_DEV.md` —— 新增模块；
5. `docs/CALIBRATION.md` / `docs/M2_REPORT.md` —— 校准方法与历史坑；
6. `docs/COLLECT_HIT.md` / `docs/COLLECT_REAL.md` —— 两套采集；
7. `docs/DECODE.md` / `docs/BENCH_RESIDENT.md` —— decode 与常驻；
8. `PARITY_STATUS.md` —— 数值现状与根因；
9. `PORTABLE.md` / `PORTABLE_TROUBLESHOOTING.md` —— 迁移到新机器。

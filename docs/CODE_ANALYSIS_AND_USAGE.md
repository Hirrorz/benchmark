# ds41_mini 代码分析与使用指南

> 分析对象：`/mnt/xds/sfs/l00972766/ds41_mini`
> 基线：2026-10-22，分支 `feat/mini-packed-dual-batch`（HEAD `87a3ad3`，与 `master` 同提交；
> packed 统一 / 多批 / per-DP / builder 重设计等改动仍在工作区，尚未提交）
> 文中路径均相对仓库根；`file:line` 对应当前工作区。

本文回答三个问题：

1. **这个 benchmark 是什么、有哪些能力**（第 1～2 章）
2. **怎么用**（第 3～6 章：环境、命令、配置、按任务的操作手册）
3. **代码怎么组织、一次运行发生了什么**（第 7～10 章：架构、执行流、关键不变量、扩展方法）

---

## 目录

- [1. 项目定位](#1-项目定位)
- [2. 功能特性](#2-功能特性)
- [3. 环境与快速开始](#3-环境与快速开始)
- [4. 配置系统](#4-配置系统)
- [5. 使用手册（按任务）](#5-使用手册按任务)
- [6. 采集与验收](#6-采集与验收)
- [7. 代码架构](#7-代码架构)
- [8. 核心执行流](#8-核心执行流)
- [9. 关键数据结构与不变量](#9-关键数据结构与不变量)
- [10. 扩展开发与排障](#10-扩展开发与排障)
- [11. 附录](#11-附录)

---

# 1. 项目定位

## 1.1 一句话

**ds41_mini 是一个不依赖官方模型代码的 DS4.1-Flash 最小算子图 benchmark**：只靠
`configs/*.yaml` + `recipes.yaml` + `opgraph.py` 自己"装配"出 40 层前向，在 Ascend NPU 上真跑，
产出与官方 `ds41_light` bench **同构的 MindStudio 流水图**，并支持与官方逐层数值对拍。

它只关心三件事：

| 关注点 | 载体 | 回答的问题 |
|---|---|---|
| **权重怎么放** | `parallel_config` + `weights.py` | 只由并行策略决定（EP/TP/CP 怎么切） |
| **用哪些算子** | `recipes.yaml` + `opgraph.py` | 每层有哪些算子、各多少次 |
| **算子怎么排** | `recipes.yaml` + `assemble.py` | 层内顺序、层间角色、每层 token 数 |

不 import 官方仓库的 `adaptlib` / `executor/` / `models/`；厂商算子通过
`torch.ops.custom` / `cann_ops_transformer` / `tile_kernels` / `deep_gemm` 直接调用。

## 1.2 设计原则

1. **声明与执行分离**：`opgraph.py` 只声明"算子叫什么、每层几次"，`exec_ops.py` 只负责
   "怎么真调"；`recipes.yaml` 只写"哪些层有、按什么顺序"。
2. **结构变化只改配置**：层数、压缩比、源层、engram 层、并行策略、seq 长度都在 YAML；
   `model_spec.py` 是结构参数的单一事实源，启动时做 fail-fast 校验。
3. **合成与真权重共用一套切分/布局/打包**：`SyntheticSource` 与 `RealSource` 产出同名、
   同 dtype、同 shape 的权重，执行路径完全一致；合成模式只是"生成器不同"。
4. **图与执行是一条路**：dry-run 与真跑共用 `run.ctx_for / expand_layer`，保证
   "dry-run 打印的算子序列"就是"真跑下发的算子序列"。

## 1.3 验收口径

| 层面 | 判据 | 工具 |
|---|---|---|
| **图结构** | 同 config/同 seq 下，与官方 trace 的**算子集合 + 次数一致、形状一致、时长 ±10%** | `run.py --dry-run` + `tools/align_ops.py` |
| **数值** | 真权重 + 同输入 + EP8，40 层 hidden 逐位一致（deepgemm 口径） | `--collect-layers` + `tools/compare_layers.py` |
| **端到端精度** | 末位 token 与官方参考一致（`id=396 ' that'`） | `tools/check_next_token.py` |
| **可移植性** | 新机器 `selfcheck` 全绿（Tier0 + Tier1） | `tools/selfcheck.py` |

当前状态：真权重 EP8 16k prefill 约 **2370 ms**（官方 2394.7 ms），hit 口径约 **742 ms**；
深层逐层数值受官方自身 run-to-run 混沌放大影响，但预填充/命中的**末位 token 均对齐**；
在固定口径（deepgemm + QSLI 注入）下已做到 **40 层逐位一致**（见 `PARITY_STATUS.md`）。
合成权重的绝对耗时只用于自身口径对比，不与官方数字逐项比较。

## 1.4 不做的事

- 命中率曲线扫描 / 调度策略 / 交互式 TUI；
- 训练、反向、优化器；
- 官方模型代码的等价复刻（只复刻"算子图 + 数值链路"）；
- 通用推理服务（常驻 bench 是固定 8 rank 的性能探针，不是产品 server）。

---

# 2. 功能特性

## 2.1 能力矩阵

| 能力 | 状态 | 入口 | 说明 |
|---|---|---|---|
| **dry-run 算子图** | ✅ | `run.py` | 毫秒级、不碰卡，输出逐层算子序列 + 全会话直方图 |
| **单卡合成权重前向** | ✅ | `runner.py` | 日常冒烟 / selfcheck / trace 校准 |
| **多卡 EP8 前向** | ✅ | `launch_mini.py` | 每 rank 只见一张物理卡；DP8/EP8/CP1 |
| **真权重 EP8** | ✅ | `configs/mini_real.yaml` / `run_real.sh` | 476 GB safetensors，流式/常驻/host cache |
| **前缀命中（hit）** | ✅ | `hit:` 配置 / `run_hit.sh` | fake 零值前缀（性能）与 real stage A/B（数值） |
| **官方 hit97 8 点矩阵** | ✅ | `run_hit.sh` | 16k…1m，逐点 h/M/R 可覆盖 |
| **多点位一次进程** | ✅ | `runner.py --points` / `points_runner.py` | 权重/进程组只加载一次，逐点预热+采集 |
| **packed 多 batch** | ✅ | `--batches N --batch-mode auto` | varlen `cu_seqlens`，b=1 也默认 packed |
| **per-DP 不同批次** | ⚠️ 实验 | points JSON 的 `dps` 字段 | 各 DP 的 MoE span 必须一致（或 dummy 对齐） |
| **decode（全量/增量）** | ✅ | `--decode N --decode-mode full\|inc` | full 为金标准；inc 持久池 + 1 token/步 |
| **NPUGraph decode** | ✅ | `DS41M_DECODE_GRAPH=1` | 全 40 层图捕获，稳态 ~25 ms/token（tp8） |
| **P→D / D→P 布局转换** | ✅ | 常驻 serve / `decode_layout.py` | EP-NZ / TP all2all / host cache 回载 |
| **常驻 benchmark** | ✅ | `bench_server.py` + `bench_cli.py` | W1/W2/W3 预热，prefill/P→D/decode/D→P 分阶段计时 |
| **DSpark 投机解码** | ⚠️ 实验 | `DS41M_DSPARK=B`（serve） | ep8 仍有 NaN/aicore 问题，见 WORKPLAN |
| **ALOP profiling** | ✅ | `--prof --alop-config` | 两种引擎：aclprof C 级 / torch_npu Python 级 |
| **算子自检** | ✅ | `tools/selfcheck.py` | Tier0 环境 + Tier1 逐算子（0..24 层） |
| **逐层/逐算子对拍** | ✅ | `compare_layers.py` / `cmp_*.py` | hidden、logits、逐算子快照、逐行 |
| **trace 对拍/分析** | ✅ | `align_ops.py` / `diff_traces.py` / `op_time_table.py` | 直方图、trace diff、耗时表 |
| **打包分发** | ✅ | `tools/build_pack.sh` + `env_pack.sh` | 代码 + 轻量 runtime（≈380 MB） |

## 2.2 两条并行的采集入口

| | `tools/run_hit.sh` | `tools/run_real.sh` |
|---|---|---|
| 权重 | 合成（假生成） | 真权重 476 GB |
| 路由 | elb 强均（可控） | 自然路由（不做负载均衡） |
| 前缀 | fake（池子零值） | stage A 真算前缀 |
| engram | 缩小表 / 侧流 | 真表 384M 行、`engram_tp_size=8` |
| 目的 | 形状 / 次数 / 耗时矩阵 | 数值 + 真实算子流水 + 精度 gate |
| 默认后端 | auto（先 ascendc 后 deepgemm） | deepgemm |
| 验收 | trace 份数、elb/起跑线/侧流行 | 自然路由、engram 就绪次数、trace 次数、next-token |

两份脚本**互不影响**，各自独立演进出口径。

## 2.3 模型结构速览

40 层 DS4.1-Flash（`model_spec.py` 默认值 / 所有 mini 配置一致）：

| 项 | 值 |
|---|---|
| hidden / moe_intermediate | 5120 / 2304 |
| 层数 / 路由专家 / topk | 40 / 384 / 6 |
| 共享专家 | 1 |
| 滑窗 | 128 |
| 注意力头 / head_dim | 64 / 512 |
| q_lora / o_lora / o_groups | 1280 / 1024 / 8 |
| index_n_heads / index_head_dim / index_topk | 32 / 128 / 512 |
| engram 层 | 1、14；表行数 384,006,168 / 384,016,682 |

层角色（由配置四张表推出）：

```
层 0,1              ratio=0  win          纯滑窗
层 2..19            ratio=2  c2a          source: 2,8,14
层 20..39           ratio=1  c1a          source: 20；indexsource: 24,28,32,36
candidate_source_layer = 20               产候选块 (2048×8)，供 24/28/32/36 的 QSLI
engram_layers = 1, 14                     层首插 engram_fetch/wait/proj
kv_source_layers   = 2,8,14,20            写压缩 KV
index_source_layers= 2,8,14,20,24,28,32,36 跑索引器（QLI 或 QSLI）
```

---

# 3. 环境与快速开始

## 3.1 最小条件

| 需要 | 说明 |
|---|---|
| 1 张 Ascend NPU + 驱动 | 本机验证机型 Ascend950DT；EP8 需 8 张 |
| CANN | **版本不限**，只要求算子能跑通；`env_pack.sh` 自动找可用的一套 |
| python3.11 | 这套 torch 只有 cp311 包（容器默认 python3.12 装不了） |
| 厂商算子包 | `compressor / matmul / megamoe / vector` 四个 vendor |
| 包内 `runtime/` | 厂商算子 Python 包 + `custom_ops` .so + py311 依赖 + tilelang/TileKernels + 预编译扩展（≈380 MB） |
| 可选 | 真权重目录（476 GB）；官方参考 dump（对拍用） |

**不需要**：官方仓库代码、共享盘（用包内 runtime 时）、PyPI 网络。

## 3.2 上手三步（新机器）

```bash
tar -xzf ds41_mini_pack_<日期>.tar.gz && cd <解出的目录>
source env_pack.sh                          # ① 探测环境，生成 env_local.sh
python3.11 tools/selfcheck.py --card 5      # ② 验收（5 换成空卡号）
python3.11 run.py configs/mini_single.yaml --dry-run   # ③ 不碰卡，出算子图
```

第一次 `selfcheck` 全绿（`✅ 算子自检通过`）即表示这台机器可跑。
之后每个新 shell 只需 `source env_local.sh`（`env_pack.sh` 的生成物，不必重复探测）。

## 3.3 常用命令速查

```bash
# ── 环境 ──
source env_pack.sh
python3.11 tools/selfcheck.py --card 5          # 0..24 层逐算子
python3.11 tools/selfcheck.py --full            # 40 层
python3.11 tools/selfcheck.py --tier0-only      # 只查环境

# ── dry-run（不碰卡）──
python3.11 run.py configs/mini_single.yaml --dry-run
python3.11 run.py configs/mini_ds41_flash.yaml --dry-run --dump-json /tmp/dry.json

# ── 单卡合成冒烟 ──
python3.11 runner.py configs/mini_single.yaml --seq 1024 --device npu:0 --layers 4
python3.11 runner.py configs/mini_single.yaml --seq 1024 --device npu:0 --prof --out ./prof/m2

# ── 8 卡 EP8 合成 ──
python3.11 launch_mini.py --config configs/mini_ds41_flash.yaml --world 8 \
    --cards 0,1,2,3,4,5,6,7 --progress off

# ── 8 卡 EP8 采集（3 轮 trace）──
python3.11 launch_mini.py --config configs/mini_ds41_flash.yaml --world 8 \
    --cards 0,1,2,3,4,5,6,7 --logdir <日志目录> --timeout 1800 \
    --seq 1024 --iters 3 --prof --out <trace目录> --alop-config configs/alop_mindstudio.yaml

# ── 单卡真权重冒烟 ──
python3.11 runner.py configs/mini_real.yaml --seq 1024 --device npu:0 \
    --weights-path /path/to/DeepSeek-V4.1-Flash-... --layers 2

# ── 采集入口 ──
bash tools/run_hit.sh -L 16k 32k 64k 128k 256k     # 合成 hit 矩阵
bash tools/run_real.sh -L 16k                       # 真权重 prefill + hit（带精度）
bash tools/run_real.sh -L 16k --mode prefill --parity
```

> `runner.py` **没有** `--parity` 参数；`--parity` 属于 `tools/selfcheck.py` 与
> `tools/run_real.sh`。runner 自身用 `--collect-layers` / `--dump-hidden` 做取证。

## 3.4 输出物

| 入口 | 主要产物 |
|---|---|
| `runner.py --prof` | `<out>/<host>_<pid>_<ts>_ascend_pt/ASCEND_PROFILER_OUTPUT/{op_statistic,kernel_details,...}.csv` + `trace_view.json` |
| `launch_mini.py` | 每 rank 一份 trace；`<logdir>/rankN.log`、`all.log` |
| `run_hit.sh` | `<out>/SUMMARY.md`、`rounds.tsv`、`PARAMS`、每点 `L<L>/prefill/` |
| `run_real.sh` | `<out>/SUMMARY.md`、`rounds.tsv`、`config_<mode>.yaml`、`logs_<mode>/`、`prefill_<mode>/` |
| `--collect-layers` | `collect_rank{R}.pt`（`{"layers": {层号: Tensor[T,H] fp32}, "rank", "n"}`） |
| `--points` | 每点 `<tag>_r{0|1}/prefill` + `results.tsv` |

---

# 4. 配置系统

## 4.1 YAML 顶层 schema

所有 `configs/mini_*.yaml` / `packed_*.yaml` / `point_*.yaml` 都是同一套 7 段结构：

```yaml
model:            # 结构、维度、层角色表、engram 布局
parallel_config:  # world/cp/moe_ep/attn_tp/dense_tp/lmhead_tp
run:              # seq_len / batch(占位) / iters / moe_chunk_max_len
weights:          # mode / path / engram_table_rows / lazy_experts /
                  # share_across_layers / host_cache / resident
backends:         # megamoe / indexer_prolog_qw / gate_topk / engram / swiglu / hc_post
hit:              # ratio / source(fake|real) / replay_tail
features:         # decoder_replay
```

### model 段

| Key | 含义 | 默认/典型值 |
|---|---|---|
| `name` | 模型名 | `deepseek_v4_1_flash` |
| `hidden` / `moe_intermediate` | 隐藏维 / 专家中间维 | 5120 / 2304 |
| `n_layers` | 层数 | 40 |
| `n_routed_experts` / `n_shared_experts` / `topk` | 路由专家 / 共享专家 / topk | 384 / 1 / 6 |
| `sliding_window` | 滑窗大小，必须 128 对齐 | 128 |
| `compress_ratios` | 每层压缩比，只能是 0/1/2 | `[0,0]+[2]*18+[1]*20` |
| `kv_source_layers` | 写压缩 KV 的层 | `[2,8,14,20]` |
| `index_source_layers` | 跑索引器的层（须包含 kv 源层） | `[2,8,14,20,24,28,32,36]` |
| `candidate_source_layer` | 产候选块的层（须在 kv 源层内） | 20 |
| `engram_layers` | engram 层 | `[1,14]` |
| `engram_offload` / `engram_tp_size` | 表放主机 / 按卡分片数 | true / 真权重必须 8 |
| `engram_num_embeddings` | 每层表行数（与 engram_layers 一一对应） | `(384006168, 384016682)` |
| `head_dim` / `n_attention_heads` / `qk_rope_head_dim` | MLA 维度 | 512 / 64 / 64 |
| `q_lora_rank` / `o_lora_rank` / `o_groups` | 低秩投影 | 1280 / 1024 / 8 |
| `index_n_heads` / `index_head_dim` / `index_topk` | 索引器 | 32 / 128 / 512 |
| `rope_theta` / `compress_rope_theta` | win 层 / 压缩层 rope base | 10000 / 160000 |
| `rope_scaling` | YaRN 参数（真权重） | `{rope_type: yarn, factor: 16, ...}` |
| `hc_mult` / `hc_sinkhorn_iters` / `hc_eps` | HC 超参 | 4 / 20 / 1e-6 |
| `rms_norm_eps` / `routed_scaling_factor` / `swiglu_limit` | 数值 | 1e-20 / 1.5 / 10.0 |

### run 段

| Key | 含义 | 备注 |
|---|---|---|
| `seq_len` | 逻辑序列长度 L | `--seq` 覆盖；`--prompt --seq 0` = 按 prompt 全量 |
| `batch` | **占位，代码不读取** | 真正入口是 `--batches` |
| `iters` | 前向轮数 | `--iters` 覆盖；profile step = `iters × 层数` |
| `moe_chunk_max_len` | MoE 分块大小 | `chunks = ceil(T/chunk)`；官方真权重配置常用 4096，当前 `mini_real.yaml` 设 8192（16k → 2 块） |

### weights 段

| Key | 含义 | 备注 |
|---|---|---|
| `mode` | `synthetic` / `real` | `--weights-path` 只覆盖路径 |
| `path` | 真权重根目录 | 合成模式也留作默认 |
| `engram_table_rows` | 合成 engram 表行数 | `0` = 用真实规模；真权重忽略 |
| `lazy_experts` | 按层生成/释放专家 | 未设时：单卡 true，多卡 false |
| `share_across_layers` | 40 层共用一份合成权重 | **只对合成模式有意义** |
| `host_cache` | 真权重首次读入 pinned 主机内存 | 多次前向才有收益 |
| `resident` | 真权重全部常驻 HBM | EP8 ~51 GiB 可装下 |

### backends 段

| Key | 可选值 | 说明 |
|---|---|---|
| `megamoe` | `ascendc` / `deepgemm` / `none` | `none` 的未融合链**当前没有执行实现** |
| `indexer_prolog_qw` | `ascendc` | 实际只走 ascendc 融合算子 |
| `gate_topk` | `tilelang` | tile_kernels `moe_topk_gate` |
| `engram` | `tilelang` | tile_kernels engram |
| `swiglu` | `fused` / `fallback` | 影响未融合链/swiglu 路径 |
| `hc_post` | `ascendc` / `tilelang` | 真权重参考配置用 tilelang |

> **重要**：`megamoe: none` 会选到 `recipes.yaml` 的 `moe.unfused` 段（gating/sort/reroute/
> dispatch/gmm/swiglu/combine/finalize），但这些 key **没有 `exec_ops` 实现**，真跑会全部
> 记为 skipped。当前可跑的 MoE 只有 `ascendc|deepgemm` 的融合路径。

### hit 段

| Key | 含义 |
|---|---|
| `ratio` | 命中率 h；`0` = plain prefill |
| `source` | `fake`（前缀=池初值零，不跑 stage A）/ `real`（先真算 stage A） |
| `replay_tail` | 重算尾段长度，默认 128 |

### features 段

| Key | 含义 |
|---|---|
| `decoder_replay` | 只跑 `cut` 之后最后 W 行（SWA 有界近似，非逐位等价） |

## 4.2 层角色如何推导

`assemble.attention_kind()`（`assemble.py:28`）的规则：

```
ratio == 0                        → win
idx in kv_source_layers:
    ratio == 1                    → c1a_source
    else                          → c2a_source
idx in index_source_layers        → c1a_indexsource
else:
    ratio == 1                    → c1a_consumer
    ratio == 2                    → c2a_consumer
```

对应 `recipes.yaml` 的 `attention:` 段：

| 类型 | 层 | 额外算子 |
|---|---|---|
| `win` | 0,1 | 无 |
| `c2a_source` | 2,8,14 | `kv_compress + idx_prolog + qli` |
| `c2a_consumer` | 3-7,9-13,15-19 | 无 |
| `c1a_source` | 20 | `kv_compress + idx_prolog + qli`（并产候选块） |
| `c1a_indexsource` | 24,28,32,36 | `idx_prolog + qsli` |
| `c1a_consumer` | 21-23,25-27,… | 无 |

## 4.3 配置校验

`model_spec.set_spec(cfg, strict=True)` 在 runner 启动时做聚合校验（`model_spec.py:202`），
不通过直接 `SystemExit`。校验包括：

- 列表长度、取值范围（`compress_ratios ∈ {0,1,2}`）、层号越界/重复；
- `kv_source_layers ⊆ index_source_layers`；
- `candidate_source_layer` 必须在 `kv_source_layers` 且不能是 win 层；
- `topk ≤ n_routed_experts`，专家数能被 `moe_ep_size` 整除；
- `block_size == 128`、`sliding_window` 128 对齐、`quant_block` 整除各维度；
- `engram_num_embeddings` 与 `engram_layers` 一一对应。

离线批量检查：`python3.11 tools/check_specs.py --strict`。

---

# 5. 使用手册（按任务）

## 5.1 dry-run：不碰卡看算子图

```bash
python3.11 run.py configs/mini_single.yaml --dry-run
python3.11 run.py configs/mini_single.yaml --dry-run --layers 0,20,21
python3.11 run.py configs/mini_single.yaml --dry-run --dump-json /tmp/dry.json
```

输出：

- 每层：`层号 | T | attn 类型 | 算子(次数)`；
- 全会话直方图（按 trace kernel 名聚合，含 `extras` 会话级残差）；
- `--dump-json` 的 `{meta, layers, hist, extras}` 给 `tools/align_ops.py` 用。

注意：`mul/add/cast/tmove` 是**纯占位算子**（`x*1.0` / `x+0.0` / dtype 往返 / `.contiguous()`），
默认不下发；只有 `DS41M_PAD_OPS=1` 才恢复到执行序列（dry-run 计数默认仍包含它们）。

## 5.2 单卡合成权重

```bash
# 冒烟：前 4 层
python3.11 runner.py configs/mini_single.yaml --seq 1024 --device npu:0 --layers 4

# 出 trace
python3.11 runner.py configs/mini_single.yaml --seq 1024 --device npu:0 \
    --prof --out ./prof/m2 --alop-config configs/alop_mindstudio.yaml

# 指定物理卡（逻辑恒 npu:0）
ASCEND_RT_VISIBLE_DEVICES=5 python3.11 runner.py configs/mini_single.yaml \
    --seq 1024 --device npu:0 --layers 4
```

- mini 的合成配置通常显式设 `share_across_layers: true`（代码默认 false）：40 层共用
  一份假权重、换层不释放，性能采集时每轮耗时才干净。
- 未给 `--ids-file/--prompt` 时用随机激活，只能看 trace/算子计数。
- `--iters N` 控制轮数；`--warmup-iters K` 在 `profiler.start()` 前跑 K 轮（不计入 trace）。

## 5.3 多卡 EP8

```bash
python3.11 launch_mini.py --config configs/mini_ds41_flash.yaml --world 8 \
    --cards 0,1,2,3,4,5,6,7 --logdir /tmp/logs --seq 8192 \
    --iters 1 --progress off
```

`launch_mini.py` 的行为：

- 每个 rank 一个 `runner.py` 子进程；
- `ASCEND_RT_VISIBLE_DEVICES=<该 rank 的物理卡>`，子进程内逻辑恒 `npu:0`；
- `runner` 侧 `parallel_config.moe_ep_size = world_size = --world`（EP=world）；
- `MASTER_ADDR/PORT` 自动选空闲端口；所有日志合并到 `all.log`（`[rN]` 前缀）。

常用参数：

| 参数 | 含义 |
|---|---|
| `--world` | rank 数（= 卡数） |
| `--cards` | 物理卡号，数量必须等于 world |
| `--console all\|rank0\|off` | 控制台回显策略 |
| `--timeout` | 整组超时（秒） |
| `--progress off` | 关闭阶段进度（stderr；透传给 runner） |
| `--bar` / `--bar-fd` / `--bar-every` | launch 侧进度条 |

## 5.4 真权重

```bash
# 单卡冒烟（需权重目录）
python3.11 runner.py configs/mini_real.yaml --seq 1024 --device npu:0 \
    --weights-path /path/to/DeepSeek-V4.1-Flash-cann-mxfp8-mxfp4 --layers 2

# EP8 + 真 ids + 精度 gate（推荐用脚本）
bash tools/run_real.sh -L 16k
bash tools/run_real.sh -L 16k --mode hit --hit 0.96875
bash tools/run_real.sh -L 16k --parity
```

> `run_real.sh --parity` 只负责跑 `--collect-layers` 收数据，不会自动调用
> `compare_layers.py`；对拍要另跑（见 5.10）。

真权重关键配置：

```yaml
model:
  engram_tp_size: 8          # 必须；=1 会让每卡拷整张表（≈1.6 TB pinned）→ OOM
  compress_rope_theta: 160000
  rope_scaling: {...}
weights:
  mode: real
  host_cache: true           # 首次读入 pinned 主机内存
  resident: false            # true = 40 层全常驻（EP8 可装下）
```

- `run_real.sh` 默认 `deepgemm`、`iters=6`、prof 轮 `PROF_ITERS=2`、`hit=0.96875`；
- 只支持 `L=16k`（除非 `--layers N` 冒烟）；
- `--layers` 冒烟必须 ≥15（engram 层是 1 和 14，截断到 14 以下会"假绿"）；
- 四条验收：自然路由、ElasticBuffer 就绪、trace 次数、next-token。

## 5.5 命中（prefix cache hit）口径

### 几何

```python
M = round((1 - h) * L / 128) * 128     # 与官方 light bench 同口径
R = M + replay_tail                     # 默认 replay_tail = 128
T_main = R
```

示例（L=65536, h=0.9707）：`M=1920`、`R=2048`。

### 两种 source

| | `fake` | `real` |
|---|---|---|
| stage A | 不跑 | 先真算前缀 `[0, H)`，写入 shared 池 |
| 前缀内容 | 池初值（零） | 真实 KV / 压缩 KV / 索引缓存 / 压缩器 state |
| 数值含义 | 只对形状/次数/耗时 | 与 prefill 数学等价 |
| 适用 | 合成权重、性能矩阵 | 真权重、数值对拍 |
| packed | 支持 | **不支持**（runner 直接拒绝） |

`source: real` 还要求 `H % ratio == 0` 且 `off % ratio == 0`（官方命中点天然满足）；
否则前缀末组的进位 state 无法精确复原，会打印一次提示。

### 执行侧口径（两种 source 共用）

- 只算尾部 `R` 个 token；
- 位置是**绝对位置** `L-R .. L-1`；
- 池子/块表/槽位按 `L` 开；
- 索引器与压缩注意力的可见长度 `seqused_k = L//ratio`；
- 窗口分支不回看前缀；
- replay 段的压缩 KV 写入被抑制（`slot=-1`），避免覆盖 stage A。

### 使用

```yaml
hit:
  ratio: 0.9707
  source: fake
  replay_tail: 128
```

```bash
# 合成权重：fake 前缀 + 强均路由
bash tools/run_hit.sh -L 64k
bash tools/run_hit.sh -L 64k --hit 0.99
bash tools/run_hit.sh -L 16k@0.99 128k:3968

# 真权重：stage A + stage B
bash tools/run_real.sh -L 16k --mode hit --hit 0.96875
```

`run_hit.sh` 官方矩阵（`MATRIX`）：

| L | M | R=M+KEEP(140) | h | 官方稳态 ms |
|---|---|---|---|---|
| 16k | 512 | 640 | .9688 | 131.0 |
| 32k | 1024 | 1152 | .9688 | 137.3 |
| 64k | 1920 | 2048 | .9707 | 192.7 |
| 80k | 2432 | 2560 | .9703 | 238.9 |
| 128k | 3968 | 4096 | .9697 | 370.9 |
| 256k | 7808 | 7936 | .9702 | 746.0 |
| 512k | 15744 | 15872 | .9700 | 1937 |
| 1m | 31488 | 31616 | .9700 | 4330 |

## 5.6 多点位、多 batch、per-DP

### 多点位（一次进程跑多个 L/h）

```bash
bash tools/run_hit.sh --points /tmp/points.json -o /data/exp1
bash tools/run_hit.sh --points-json '[{"L":81920,"hit":0.96,"batches":1,"replay":true}]'
bash tools/run_hit.sh --points-spec '80k@0.96/b1/r1 80k@0.96/b2/r1 120k@0.9/b1/r0'
```

点位 JSON 三种格式：

```jsonc
// 旧格式
{"L": 81920, "hit": 0.96, "batches": 1, "replay": true, "replay_tail": 128}

// v2：一个 point 内多个 batch（同一次 packed pass）
{"tag": "mix", "batches": [{"L": 8192, "hit": 0.96}, {"L": 16384, "hit": 0.90}]}

// per-DP：每个 rank 自己一条（实验性）
{"tag": "dp8mix", "dps": [
  {"rank": 0, "batches": [{"L": 8192, "hit": 0.96}]},
  {"rank": 1, "batches": [{"L": 8192, "hit": 0.96}, {"L": 4096, "hit": 0}]}
]}
```

- `points_runner` 一次进程内逐点重建几何/池/输入，权重打包缓存跨点复用；
- 点与点之间清图缓存、池、ctx、META，不清权重；
- 产物命名：`(L1_h1&L2_h2)_r{0|1}/prefill`；per-DP 非全同名为 `<tag>_r{0|1}/` + `dps.json`。

### 多 batch（packed）

```bash
python3.11 runner.py configs/mini_ds41_flash.yaml --seq 8192 \
    --batches 4 --batch-mode auto --world 8 --rank 0 ...
```

- `auto`：`batches>1` → packed；**b=1 的普通 prefill 也走 packed**；
- packed **不支持** `--decode / --layers / --prompt`、以及 `hit.source=real`，自动回退 sequential；
- `--batch-mode sequential` 仅用于显式 A/B，不得用于报数；
- packed 把 N 条序列拼成一次前向，`T_pass = ΣR_i`，用 `cu_seqlens` 恢复边界；
- 逐序列可以有不同 L / hit（v2 点位），`pos_offsets` 逐序列前移。

## 5.7 decode

```bash
# 全量重算（金标准，慢）
python3.11 launch_mini.py --config configs/mini_real.yaml --world 8 \
    --cards 0,1,2,3,4,5,6,7 --seq 16384 --ids-file ids_16k.pt --decode 8

# 增量（prefill 一次 + 每步 1 token）
... --decode 8 --decode-mode inc

# 稳定性：每步 barrier
DS41M_DECODE_BARRIER=1 ...
```

- `--decode N` 只产生 **N-1 次** decode forward（最后一个 token 不需要前向）；
- 增量模式强制 `DS41M_PERLAYER_CACHE=1`，池按 `L_total = prompt + n_new` 一次开好；
- 多步 decode 建议 `DS41M_DECODE_BARRIER=1`（各 rank 步序对齐）或 `DS41M_ENGRAM_HOST=1`；
- `DS41M_DECODE_GRAPH=1` 用 `torch.npu.NPUGraph` 捕获 T=1 增量 decode，默认 `cut=21`，
  `DS41M_DECODE_GRAPH_CUT=0` 为全 40 层；首两个 forward 必须 eager。

## 5.8 常驻 benchmark

```bash
# 终端 A：起 8 rank 常驻服务
python3.11 tools/bench_server.py start \
    --config configs/mini_real.yaml --world 8 --cards 0,1,2,3,4,5,6,7 \
    --logdir /mnt/xds/sfs/l00972766/tmp/bench_resident \
    --job resident --mode auto --serve-l-max 4096

# 终端 B：预热 + 跑请求
python3.11 tools/bench_cli.py warmup --job resident \
    --lengths 16,32,64,128,256,512,1024 --decode 1
python3.11 tools/bench_cli.py run --job resident \
    --prompt "请用一句话介绍杭州。" --decode 32 --repeat 3
python3.11 tools/bench_cli.py status --job resident
python3.11 tools/bench_cli.py stop   --job resident
```

- 预热三段：**W1** 干净 prefill；**W2** decode 布局 / 1 步 decode；**W3** 长度阶梯；
- 布局模式：`ep8`（默认）、`ep8-fidelity`、`tp8-fidelity`（实验）、`tp8-dual`（未实现）；
- 目标：预热后每次 prompt prefill ~1 s（L ≤ 预热阶梯时硬上限 3 s），decode 单独计时；
- 无 NPU 自测：`bench_server.py start --fixture` + `bench_fake_rank.py`。

## 5.9 profiling 采集

### ALOP 两种引擎

| 引擎 | 条件 | 产物 |
|---|---|---|
| aclprof C API | 默认 / `python_profiling.enabled=false` | `PROF_*` 原始目录 |
| torch_npu.profiler | `python_profiling_config.enabled: true` | `ASCEND_PROFILER_OUTPUT/` + `trace_view.json`（MindStudio 直开） |

配置示例：`configs/alop_mindstudio.yaml`（MindStudio trace）、
`configs/alop_mindstudio_stack.yaml`（带 Python 调用栈）。

### 关键口径

```bash
python3.11 launch_mini.py --config ... --world 8 --cards ... \
    --seq 1024 --iters 3 --prof --out <trace目录> \
    --alop-config configs/alop_mindstudio.yaml --warmup-iters 1
```

- **profile step = 每层一次**，所以 `step 数 = iters × 层数`；
- 第 N 轮的耗时包含 trace 导出（3～4 GB 要几十秒），看稳态用中间轮；
- `--warmup-iters K`：`profiler.start()` 前跑 K 轮，不污染 trace；
- `--prof-ranks all|0|0,3`：只让部分 rank 开 profiler；
- 每个 rank 一份 trace 目录：`<out>/<host>_<pid>_<ts>_ascend_pt/`。

### trace 里看什么

| 文件 | 用途 |
|---|---|
| `op_statistic.csv` | 按 `OP Type` 汇总次数/耗时 |
| `kernel_details.csv` | 逐 kernel，`Step Id` = 0..iters×层数-1，`Stream ID` 看侧流 |
| `operator_details.csv` | 算子级明细 |
| `trace_view.json` | 导入 MindStudio 看流水图 |

engram 三件套应落在**非主流**（非 `Stream ID=61`）的侧流上；主干层算子都在主流。

## 5.10 对拍与精度工具

```bash
# 逐层 hidden 对拍（两边都是 collect_rank{R}.pt）
python3.11 tools/compare_layers.py --ref <官方collect> --ours <mini collect> --rank 0
python3.11 tools/compare_layers.py ... --logits        # 额外比 top1/top5 重合

# 精度 gate（mini 日志 vs 官方参考 dump）
python3.11 tools/check_next_token.py --log <rank0.log> \
    --ref-dump /mnt/xds/sfs/l00972766/parity/ref/dump_rank0_call1.pt

# dry-run 直方图 × 官方 trace
python3.11 tools/align_ops.py --ours /tmp/dry.json \
    --ref <trace>/ASCEND_PROFILER_OUTPUT/kernel_details.csv --ref-passes 4

# 两个 trace 差在哪
python3.11 tools/diff_traces.py <traceA> <traceB> --passes-b 4

# 8 rank 算子耗时汇总（对齐官方 op_summary）
python3.11 tools/op_time_table.py --trace-dir <trace目录> --top 25
```

判据：`compare_layers.py` 相对误差 `<1e-3` 为 ✅，`<5e-2` 为 ⚠️，`≥5e-2` 为 ❌；
层集合/形状不一致会直接报错，不做截断/广播。

## 5.11 打包与分发

```bash
bash tools/build_pack.sh          # → dist/ds41_mini_pack_<日期>.tar.gz + SHA256SUMS
DIST_SHARE=<dir> bash tools/build_pack.sh   # 额外拷一份到共享盘
```

包 = 代码（`git archive`，只含跟踪文件）+ `runtime/`（≈380 MB）。
**注意**：`git archive` 不包含未提交改动；当前分支的 packed/多批改动还在工作区，
打包前需先提交或手工确认。

---

# 6. 采集与验收

## 6.1 `tools/run_hit.sh`（合成权重）

固定口径：dp8/ep8/cp1 + elb 强均路由 + 合成假权重 + fake 前缀 + 起跑线 barrier ×1 + engram 侧流。

```bash
bash tools/run_hit.sh                          # 官方 8 点全跑
bash tools/run_hit.sh -L 16k 32k 64k 128k 256k # 指定点
bash tools/run_hit.sh -L 64k --hit 0.99        # 全局命中率
bash tools/run_hit.sh -L 64k@0.99 128k:3968    # 逐点 h / 直接给 M
bash tools/run_hit.sh --points <json>          # 多点位一次进程
bash tools/run_hit.sh --dry-run / --list       # 只解析 / 看矩阵
```

每点跑两轮：

- **prof 轮**：`--prof` + 8 份 trace；
- **noprof 轮**：干净稳态耗时。

默认 `iters`：普通 4；`--graph-tail` 默认开且未显式给 `--iters` 时改为 6。
验收：每轮 `rc=0`、trace 份数 = 8（或 `--prof-ranks` 数量）、
`elb/balroute`、`[起跑线]`、`[engram] 侧流已启用` 三行自检。

产物：`SUMMARY.md`、`rounds.tsv`、每点 `PARAMS`（口径指纹，参数变了会重采）、
`L<L>/prefill/` trace。

## 6.2 `tools/run_real.sh`（真权重）

```bash
bash tools/run_real.sh -L 16k                    # prefill + hit
bash tools/run_real.sh -L 16k --mode prefill
bash tools/run_real.sh -L 16k --mode hit --hit 0.96875
bash tools/run_real.sh -L 16k --parity           # + 逐层 collect（rank0）
bash tools/run_real.sh -L 16k --layers 15 --no-precision  # 冒烟
bash tools/run_real.sh -L 16k --engram-host      # 逃生：主机取表
```

默认：`deepgemm`、`iters=6`、prof 轮 2 轮、`resident=true`、`host_cache=false`。
MoE 块长从 `configs/mini_real.yaml` 的 `run.moe_chunk_max_len` 读取：**当前工作区是 8192**
（16k 每层 2 块）；README / `docs/COLLECT_REAL.md` 里写的 4096（4 块）是旧口径，以配置为准。

四条自动验收：

1. `[elb/balroute]` **缺席**（自然路由）；
2. rank0 日志里 `厂商 ElasticBuffer 就绪` 恰好 = engram 层数（2）；
3. trace 断言：`EngramFetch = EngramFetchWait = engram层数 × iters × 卡数`，
   `MegaMoe = 层数 × iters × 卡数 × 块数`；
4. 精度 gate：`NEXT_TOKEN mini=396 ref=396 match=YES`。

`--parity` 只负责 collect，不自动跑 `compare_layers.py`；需另跑对拍。

## 6.3 `tools/selfcheck.py`（验收门）

```bash
python3.11 tools/selfcheck.py                 # Tier0 + Tier1（0..24 层）
python3.11 tools/selfcheck.py --card 5        # 指定物理卡
python3.11 tools/selfcheck.py --tier0-only    # 只查环境
python3.11 tools/selfcheck.py --full          # 0..39 层
python3.11 tools/selfcheck.py --parity <官方collect目录>
python3.11 tools/selfcheck.py --json out.json
```

- **Tier 0**：python/torch/torch_npu/yaml、CANN 路径、厂商命名空间、23 个算子入口、NPU 可见性；
- **Tier 1**：合成权重真跑 0..24 层（覆盖全部层角色），在 `exec_ops.run_op` 上挂钩子，
  逐算子统计调用/失败/跳过/耗时，检查输出 NaN/Inf（前 64 行）；
- 未覆盖算子按 `opgraph.OPS` 逐条列出并写原因；
- 退出码：0 全过 / 1 有算子失败 / 2 Tier0 挂。

---

# 7. 代码架构

## 7.1 分层图

```text
入口层
├─ run.py             dry-run：不碰卡，输出算子图
├─ runner.py          单 rank 真跑：prefill / hit / decode / serve / points
├─ launch_mini.py     多 rank supervisor：每 rank 一张物理卡
├─ points_runner.py   一次进程多点位 / 多批 / per-DP
├─ bench_server.py    常驻 8 rank supervisor（--serve）
└─ bench_cli.py       常驻客户端（warmup / run / status / stop）
          │ config + CLI
          ▼
声明 / 几何层
├─ configs/*.yaml
├─ model_spec.py      结构参数单一事实源 + 聚合校验
├─ assemble.py        40 层栈、层角色、每层 T、hit / replay 几何
├─ recipes.yaml       每层算子顺序与存在性
└─ opgraph.py         算子声明表：kernel 名 / 次数 / 形状 / affects
          │ layer stack + op sequence
          ▼
执行层
├─ exec_ops.py        OPS 注册表 + run_op 分派 + 池 / rope / QMM / 元数据
│  ├─ exec_indexer.py (+_packed)   idx_prolog / QLI / QSLI
│  ├─ exec_moe.py                  MegaMoE / TP / EP-decode
│  ├─ exec_engram.py               取表 / 侧流 / ElasticBuffer
│  └─ exec_dspark.py               DSpark 草稿模型（serve 用）
          │ weights + ctx
          ▼
支撑层
├─ weights.py + weights_manifest.py   合成 / 真权重统一切分与加载
├─ engram_layout.py                   压缩词表 / 素数桶 / n-gram hash
├─ decode_layout.py                   P→D / D→P 专家布局转换
├─ batch.py                           packed varlen 布局算术
├─ replay.py                          decoder replay 尾段裁剪
├─ alop.py + profiling.py             profiling 后端
└─ ds41_log.py / progress.py / progress_bar.py   日志与进度
          │
          ▼
工具层
└─ tools/*  自检 / 采集 / 对拍 / trace 分析 / 探针 / 打包 / builder
```

## 7.2 模块职责

| 模块 | 行数量级 | 职责 | 关键入口 |
|---|---|---|---|
| `runner.py` | 3849 | 单 rank 真跑主控；pass 实现；serve/points/decode | `run_forward()` `runner.py:525` |
| `alop.py` | 2165 | NPU profiling 工具（aclprof / torch_npu） | `ALOPProfiler` `alop.py:1286` |
| `exec_ops.py` | 1791 | 算子执行表、dispatch、池/rope/QMM/元数据 | `run_op()` `exec_ops.py:1755` |
| `exec_moe.py` | 1080 | MegaMoE 三路径（ascendc/deepgemm/TP/decode） | `_mega_moe()` `exec_moe.py:915` |
| `exec_indexer.py` | 872 | 索引器 QLI/QSLI、索引 K 缓存 | `_qli()` `exec_indexer.py:452` |
| `decode_layout.py` | 805 | P→D/D→P 布局转换、decode GMM 缓存 | `prepare_decode()` `decode_layout.py:539` |
| `exec_engram.py` | 799 | engram 取表/侧流/ElasticBuffer | `_engram_fetch()` `exec_engram.py:598` |
| `exec_dspark.py` | 743 | DSpark 草稿模型（3 阶段 MTP） | `DSparkDraft` `exec_dspark.py:59` |
| `weights.py` | 714 | 合成/真权重 Source、切分、host cache | `make_source()` `weights.py:701` |
| `progress_bar.py` | 570 | launch 侧进度条 | `Bar` `progress_bar.py:263` |
| `points_runner.py` | 443 | 多点位批量跑 | `run_points()` `points_runner.py:67` |
| `exec_indexer_packed.py` | 334 | packed QLI/QSLI | `register()` `exec_indexer_packed.py:31` |
| `model_spec.py` | 330 | 结构单一事实源 + 校验 | `set_spec()` `model_spec.py:300` |
| `launch_mini.py` | 325 | 多 rank 启动器 | `main()` `launch_mini.py:179` |
| `engram_layout.py` | 264 | engram 哈希布局 | `ngram_hash_ids()` `engram_layout.py:191` |
| `batch.py` | 213 | packed 布局算术 | `cu_of()` `batch.py:54` |
| `exec_ops_packed.py` | 209 | packed compressor | `register()` `exec_ops_packed.py:30` |
| `replay.py` | 204 | decoder replay 策略/裁剪 | `apply()` `replay.py:120` |
| `run.py` | 188 | dry-run + 共用展开函数 | `dry_run()` `run.py:126` |
| `weights_manifest.py` | 181 | 每层权重清单 | `layer_manifest()` `weights_manifest.py:163` |
| `assemble.py` | 143 | 层栈/角色/T | `build_stack()` `assemble.py:65` |
| `ds41_log.py` | 139 | 日志等级 | `install()` `ds41_log.py:90` |
| `opgraph.py` | 97 | 算子声明表 | `OPS` `opgraph.py:31` |
| `progress.py` | 95 | runner 阶段进度 | `Progress` `progress.py:40` |
| `profiling.py` | 33 | runner ↔ ALOP 适配 | `make_profiler()` `profiling.py:14` |

## 7.3 三个"单一事实源"

| 事实源 | 回答的问题 | 消费方 |
|---|---|---|
| `model_spec.py` | 模型结构/维度/层角色表/并行参数 | `assemble` / `weights_manifest` / `exec_*` / 校验 |
| `assemble.build_stack(cfg)` | 这次前向跑哪些层、每层角色、每层 T | `run.py` / `runner.py` / `points_runner` |
| `recipes.yaml + opgraph.py` | 每层有哪些算子、顺序、各多少次 | dry-run；runner 用 `exec_ops` 同名 key 执行 |

**纪律**（`opgraph.py:23`）：

- **哪些层有这个算子** → 写 `recipes.yaml`（存在性）；
- **每层跑几次** → 写 `opgraph.count(ctx)`（数量，如 kv_src 层多 1 次 rope）。

## 7.4 两套注册表

| | `opgraph.OPS` | `exec_ops.OPS` |
|---|---|---|
| 性质 | **声明式**：kernel 名、形状、次数、affects | **可执行**：`key -> fn(st, ctx, w)` |
| 用途 | dry-run 直方图、trace 对齐、selfcheck 覆盖表 | 真跑 dispatch |
| 数量 | 39 个名字 | 20 个核心 + 7 个插件 + 3 个 packed 别名 |
| 耦合 | 靠**同名约定**；`recipes.yaml` 写名字 | 同名 key 查表 |

几个重要事实：

- `opgraph` 的 `shapes / affects / trace / note` 目前**没有代码消费**，只服务于校准与人工维护；
- `dyn_mx_quant` 是**隐式算子**：dry-run 计数有，执行侧由量化算子内部产生，runner 跳过；
- `gather_tail / scatter_tail` 只声明，无实现、不在 recipes；
- 未融合 MoE 链（`gating`…`finalize`）**没有执行实现**，`megamoe: none` 不可真跑；
- 插件（indexer/moe/engram）在 `exec_ops.ensure_ops(cfg)` 时注册，并从 `SKIP` 中移除自己的 key；
- packed 分派规则：**ctx 有 `cu_seqlens` 就优先 `OPS[key+"@packed"]`**，没有则回落普通实现。

## 7.5 插件加载

`exec_ops.ensure_ops(cfg)`（`exec_ops.py:89`）：

1. `import custom_ops, cann_ops_transformer, cann_ops_nn`（顺序敏感，见 10.1 坑）；
2. `_apply_spec(cfg)` 用 `model_spec` 刷新模块级常量；
3. `_load_plugins(cfg)` 按固定顺序导入：
   `exec_indexer` → `exec_moe` → `exec_engram`，各自 `register(OPS, SKIP, core, cfg)`；
4. 若存在 `exec_ops_packed.py`，再 `register(OPS, core)` 注册 packed 覆盖。

## 7.6 一次 dry-run 的调用链

```
run.py:main
  ├─ assemble.load_yaml(config)
  ├─ assemble.build_stack(cfg)          # 40 层 + 每层 T
  ├─ for layer in stack:
  │    ├─ run.ctx_for(cfg, layer)       # T/H/I/ratio/角色/后缀
  │    ├─ run.expand_layer(cfg, recipes, layer)
  │    │    ├─ engram 层先插 engram 段
  │    │    ├─ 展开 common / _attention / _moe
  │    │    └─ 过滤 _PAD_OPS（默认不下发）
  │    └─ run.apply_ops(counts, seq, ctx)
  │         └─ opgraph.OPS[name]["count"](ctx) → kernel 直方图
  └─ extras（会话级残差）→ hist
```

## 7.7 一次真跑的调用链

```
runner.py:main (runner.py:3300)
  ├─ 预扫 --log-level → ds41_log.install()
  ├─ argparse → assemble.load_yaml → packed 几何 → model_spec.set_spec (fail-fast)
  ├─ exec_ops.ensure_ops(cfg)          # vendor import + 插件注册
  ├─ torch.npu.set_device(dev)
  ├─ world>1: dist.init_process_group("hccl") + get_hccl_comm_name(init_comm=True)
  ├─ weights.make_source(cfg)          # SyntheticSource / RealSource
  ├─ --serve  → serve_loop()           # 常驻分支，不再往下
  ├─ --points → points_runner.run_points()
  ├─ --decode → decode_greedy() / decode_incremental()
  ├─ hit.source=real: 先跑 stage A（不进 profile）
  ├─ 输入准备（ids/prompt/随机）+ packed 拼接
  ├─ --warmup-iters: 预热 pass（不进 trace）
  ├─ start_line_sync() 一次 barrier
  ├─ profiler.start()
  ├─ for it in range(iters):
  │    for batch in range(_npass):
  │      run_prefill_auto(...)          # 奇数 T 可能拆成 prefill + decode
  │        └─ run_forward(...) (runner.py:525)
  │             ├─ exec_ops.PASS_NO += 1
  │             ├─ assemble.build_stack(cfg) → 层栈
  │             ├─ 每 pass 重建 packed cu / 清 META
  │             ├─ hit/replay/geometry 处理
  │             ├─ for layer in stack:
  │             │    ├─ ctx = _CTX_CACHE or run.ctx_for(...)
  │             │    ├─ weights: source.iter_layer → strip_prefix → pack_layer_weights
  │             │    ├─ engram 预取排程（提前 2 层 / 层 0 头）
  │             │    ├─ seq = _SEQ_CACHE or run.expand_layer(...)
  │             │    ├─ for key in seq:
  │             │    │    └─ exec_ops.run_op(key, st, ctx, w)
  │             │    ├─ profiler.step()        # 每层一次
  │             │    └─ post_layer_cb（serve W1 建 AC/NZ）
  │             └─ 图捕获/重放、profiler.stop、池跨 pass 保留
  ├─ 稳态统计（剔除首轮/捕获轮/其后一轮，≥3 样本取中位）
  ├─ profiler.finalize()
  ├─ head_logits / --dump-hidden / --collect-layers
  └─ 打印耗时表、算子错误统计
```

---

# 8. 核心执行流

## 8.1 pass / iteration / step

| 概念 | 定义 | 代码 |
|---|---|---|
| **pass** | 一次完整前向 = 一次 `run_forward()` | `runner.py:525`，进入先 `PASS_NO += 1` |
| **iteration** | main 外层一轮 `for it in range(iters)` | `runner.py:3695` |
| **step** | profiler 打点单位 = **每层一次** | `runner.py:1546` |
| **warmup** | `profiler.start()` 前的 pass，不产生 step | `runner.py:3655` |

一个 iteration 包含的 pass 数：

- packed：`_npass = 1`（N 条序列拼成一次前向）；
- sequential：`_npass = --batches N`（每 batch 一个独立 pass）；
- `--decode`：每个 token 一个 T=1 pass（增量）或一次全量 pass（full）；
- hit real：stage A 是独立一个 pass（不计入采集），stage B 是测量 pass；
- 奇数 T：在单序列、无 hit、无 profiler/collect 的路径上，`run_prefill_auto` 会把
  `T` 拆成 `T-1` prefill + `1` decode 两个 pass。

> 因此 **trace step ≠ pass**：一个 40 层 pass 产生 40 个 step。`tools/diff_traces.py`
> 里也专门提醒这一点。

## 8.2 run_forward 内部阶段

`run_forward()` 的骨架（`runner.py:525-1709`）：

1. **pass 计数与日志模式**（`:542`）；
2. **packed cu 重建**：每 pass 按 `row_lens` 重建，b=1 也重建（`:554-579`）；
3. **层栈**：`assemble.build_stack(cfg)`；`--layers N` 截断（`:586-604`）；
4. **engram 预取排程**：厂商侧流在稳态把全部 engram 层提到层 0 头；首 pass 在层 1/2 头；
   host 路径提前 2 层（`:617-655`）；
5. **图判定**：prefill 尾段图 / decode 图（`:658-718`）；
6. **META 几何变化清理**：`(T, pos_offset, n, cu, lens, L_full, engram_ctx)` 变化时
   `exec_ops.meta_clear()`（`:719-732`）；
7. **shared 池初始化**：跨层/跨 pass 的 `_shared`（`:738-744`）；
8. **输入与命中**：`x`、`pos_offset`、hit fake/B 切片（`:783-876`）；
9. **replay**：策略解析 + 裁尾（`:877-984`）；
10. **prefill 尾段图 / decode 图**：捕获或重放（`:985-1153`）；
11. **层循环**（`:1154-1551`）：
    - ctx 补齐（`:1166-1208`）；
    - collect 采样（`:1220-1226`）；
    - 权重装载与打包（`:1238-1258`）；
    - engram 预取发起（`:1270-1304`）；
    - 算子序列（`:1305-1306`）；
    - 逐算子 `exec_ops.run_op`（`:1336-1537`）；
    - `profiler.step()` + `post_layer_cb`（`:1546-1551`）；
12. **图捕获收尾并立即 replay 一次**（`:1557-1635`）；
13. **profiler.stop / 同步**（`:1636-1651`）；
14. **池跨 pass 保留**（`:1662-1672`）；
15. 返回 `x / logits / collected / timings`（`:1705-1709`）。

## 8.3 ctx：层上下文

`run.ctx_for()`（`run.py:27`）+ runner 补齐（`runner.py:1166-1208`）后，`ctx` 里有：

| 类别 | 字段 |
|---|---|
| 几何 | `T`、`chunk`、`H`、`I`、`ratio`、`rtopk`、`e_local`、`S`、`idx` |
| 角色 | `is_kv_src`、`is_idx_src`、`is_candidate`、`engram` |
| 维度 | `head_dim`、`index_topk`、`rope_theta`、`compress_rope_theta`、`rope_scaling` |
| 命中/位置 | `pos_offset`、`L_full`、`hit_replay_tail`、`pos_offsets`、`row_lens` |
| packed | `n_seqs`、`cu_seqlens`、`seq_lens` |
| 解码 | `is_decode_step`、`split_tail` |
| engram | `engram_ctx`、`input_ids`、`engram_*` |
| 运行时 | `device`、`hc_mult`、`rope_dim`、`topk`、`moe_nspan` |
| 缓存后缀 | `suffix_win`、`suffix_cmp`、`suffix_idx`、`win_abs` |

## 8.4 池与缓存的数据流

### 窗口池（win）

- 行宽 `KV_ROW_W = head_dim + head_dim/16 = 544`（512 数据 + 32 scale）；
- 槽位数 `WIN_SLOTS = 3 × sliding_window = 384`；
- **block 0 恒空**（算子契约）；
- 持久 A 池：按 `(pos) % WIN_SLOTS` 写最后 `min(S,T)` 行（`exec_ops._kv_epilog` `exec_ops.py:711`）；
- prefill 全长池 `_full_pool`：`nblk=ceil(T/BS)`，槽位 `(1+pos//BS)*BS + pos%BS`；
  packed 时每序列占独立物理块段；
- 命中时 `win_abs` 用绝对位置索引（`DS41M_PERLAYER_CACHE=1`）。

### 压缩池（cmp）

- 行宽 `CMP_ROW_W = index_head_dim*2 + index_head_dim/2 = 320`（mxfp4 256 + scale 64）；
- 按 ratio 分池：`_cmp_pool_r1`、`_cmp_pool_r2`；
- 行数含 carry：`n_full = max(n, L_full//ratio + 1)`；
- 槽位是**绝对压缩行号**；
- 压缩器 ring state 每层一份：`_cmp_state_L{idx}`；
- 命中 replay 段写入抑制：`loc < (off+rep)//ratio` → `slot=-1`。

### 索引 K 缓存

- `_idx_k_r{ratio}` / `_idx_ks_r{ratio}`；
- 每块 128 个压缩行；block 0 恒空；
- 旋转块表 `[last, 1, 2, ...]`（官方实测口径）；
- 候选源层额外写组合缓存 `_idx_qsli_r{ratio}`。

### 命中串层问题与 `DS41M_PERLAYER_CACHE`

默认所有池只按 ratio 分，一个 pass 内多个写者共用一块，靠"读紧跟自己的写"侥幸成立
（prefill 与官方逐位一致）。命中 stage B 读前缀时会读到**最后一个写者**，因此提供：

```bash
DS41M_PERLAYER_CACHE=1
```

给窗口池按本层、压缩池按 kv 源层、索引缓存按 idx 源层加后缀；`off==0`（prefill）时与旧行为逐位相同。

## 8.5 packed 的实现边界

packed 不是"整套并行实现"，而是：

1. **布局算术统一在 `batch.py`**：
   - `cu_of / is_packed / seq_lens / pos_offsets / local_positions / local_rows /
     rot_block_table / tail_slice`；
2. **只对契约差异最大的 3 个算子做 `@packed` 覆盖**：
   - `kv_compress@packed`（`exec_ops_packed.py`）：一次 batched `ds41.compressor`；
   - `qli@packed`、`qsli@packed`（`exec_indexer_packed.py`）：一次 batched 索引器；
3. **其余算子在普通实现里处理 cu**：
   - `kv_epilog` 的 packed 窗口池分支；
   - `mqsmla` 的 packed 分支（块表/逐请求 seq/压缩长度）；
   - `rope` / `wo_a` 的绝对位置 gather；
   - `idx_prolog` 的 packed rope。

关键口径（历史上多次踩坑）：

- `seqused_k` / `cmp_len` 必须是**绝对可见压缩长度** `(off+T)//ratio`，
  不是本段行数，也不是 `L//ratio`；
- replay 段压缩行必须 `slot=-1`（packed 实现曾经漏掉，导致层 20 压缩池多写）；
- 每 pass 必须按 `row_lens` 重建 `cu`，b=1 也要重建；
- `cu[-1]` 必须等于输入行数 T（packed indexer 有防御断言）。

## 8.6 多 rank 协同

### 启动隔离

`launch_mini.py`：

- 每 rank 一个 `runner.py` 子进程；
- `ASCEND_RT_VISIBLE_DEVICES=<cards[r]>`，子进程逻辑卡恒 `npu:0`；
- `MASTER_ADDR/PORT` 自动选空闲端口；
- `DS41M_NO_VISIBLE=1` 可关闭卡隔离（排查 topo/rank 映射用）。

### 通信域

| 通信域 | 用途 | 创建 |
|---|---|---|
| HCCL WORLD | 默认集合通信；MoE 对称缓冲 | `dist.init_process_group("hccl")` + `get_hccl_comm_name(init_comm=True)` |
| MegaMoE 组 | 对称缓冲 | 复用 WORLD；单卡时自建 1-rank 组 |
| engram 专用组 | **每层一个**（层 1/14 各一个） | `dist.new_group(ranks=all)`，key=(layer,tp) |
| MC2 域 | decode dispatch/combine | `exec_moe._mc2_group()`，懒建 |

### 同步点

| 同步 | 开关 | 说明 |
|---|---|---|
| 起跑线 barrier | 默认 1 次（world≥2） | `DS41M_START_NOBARRIER=1` 关 |
| 每 pass barrier | `DS41M_PASS_BARRIER=1` | 等待不计入 pass 时长 |
| decode 每步 barrier | `DS41M_DECODE_BARRIER=1` | 厂商 engram 侧流对步序敏感 |
| 常驻 token 广播 / cancel | 自动 | 保证 8 rank 同步退出 |

## 8.7 权重子系统

### 两条 Source

| | `SyntheticSource` | `RealSource` |
|---|---|---|
| 构造 | 按 manifest 形状随机生成 | safetensors mmap 读取 |
| dtype 分布 | F8_E4M3 0..126、E8M0 120..135、I8 -120..120、BF16/F32 randn | 真值 |
| engram 表 | `HOST_RESIDENT` 强制 CPU；行数可缩小 | 读真表，host 分片 |
| 按层释放 | `lazy_experts`（单卡默认 true） | `resident` / `lazy_experts` |
| 跨层共享 | `share_across_layers` | 不适用 |
| host cache | 无 | pinned host / memfd 跨进程 |

### 切分

- **EP**：连续切专家，rank r 拿 `[r*E/ep, (r+1)*E/ep)`；384/8 = 48/rank；
- **MoE TP**（`DS41M_MOE_TP=1`）：本 rank 拥有全部 384 专家 × `I/tp`；
  非均匀 128 对齐切分，I=2304、tp=8 → `[384,384,256,256,256,256,256,256]`；
- **Attention TP**：按头维切，当前配置 `attn_tp_size=1`，不启用；
- **engram TP**：`model.engram_tp_size`，真正的分片在 `exec_engram._vendor_buffer`
  （每卡 `ceil(rows/tp)`，不足补零）；`weights.shard()` 里的 engram 分支是占位。

### 打包

`runner.pack_layer_weights(w, cfg, layer_idx)`（`runner.py:231`）统一完成：

- mxfp8/mxfp4 的 transpose / NZ(29) / fp4 打包；
- gate 转置、bias fp32、k_norm/compressor norm fp32；
- wq_b scale 配对；
- 合成模式按"去层号"缓存，真权重按层缓存；
- 真权重打包后 `_release_packed_raw` 释放 6 个 raw ND fp8 权重（省 ~5.4 GiB）。

### 真权重 host cache 与常驻

- `weights.host_cache: true`：首次读入 pinned 主机内存，之后 H2D；实测 40 层首遍 52.1s → 再遍 1.2s；
- `weights.resident: true`：40 层全常驻（EP8 可行）；
- `DS41M_WEIGHT_HOST_CACHE=0` 关闭；
- `DS41M_MOE_HOST_CACHE=<socket>`：跨进程 memfd 共享专家权重（常驻 bench 用）。

## 8.8 profiler 集成

- `profiling.make_profiler(prof_dir, alop_config)` → `ALOPProfiler`；
- runner 在起跑线 barrier 之后 `start(device=...)`；
- 每层 `profiler.step()`；
- 最后一个 pass `stop()`，退出前 `finalize()`；
- `--prof-ranks` 决定哪些 rank 真开 profiler（其余照跑但不落 trace）；
- `--points` 时每个点位单独建 profiler（`points_runner.py:273`）；
- ALOP 失败**不抛异常**给被采集程序（故障隔离）。

## 8.9 诊断与 dump

| 开关 | 产物 |
|---|---|
| `--collect-layers <dir>` | `collect_rank{R}.pt`，每层 `x.mean(dim=1)` fp32 CPU |
| `--dump-hidden <path>` | `{hidden, logits, pos_offset, L, hit}` |
| `DS41M_DUMP_OPS=<dir>` | 层 0/1 每算子后的 `x/residual/q/kv/attn_out/...` |
| `DS41M_DUMP_SLICE=<dir>` | 每算子/每层指定张量前 N 行（支持整池、TAIL 切片） |
| `DS41M_OP_DIGEST` / `DS41M_PACKED_DIGEST_LAYERS` | 算子/层入口 sha1 |
| `DS41M_STATS` / `DS41M_NAN_TRACE` | 统计 / NaN 追踪 |
| `DS41M_QS_DUMP=<dir>` | QSLI 入参快照 |
| `DS41M_SYNC_TRACE=1` | 慢同步栈 |
| `DS41M_MEMDBG=1` | 逐算子显存 |
| `tools/scan_first_nan.py` | 从 dump 目录找第一个 NaN 快照（元凶=前一个算子） |

---

# 9. 关键数据结构与不变量

## 9.1 池几何

| 常量 | 值 | 来源 |
|---|---|---|
| `BS` / `block_size` | 128 | 算子契约 |
| `KV_ROW_W` | 544 = 512 + 32 | `head_dim + head_dim/16` |
| `CMP_ROW_W` | 320 = 256 + 64 | `index_head_dim*2 + index_head_dim/2` |
| `WIN_SLOTS` | 384 = 3×128 | `3 × sliding_window` |
| `quant_block` | 32 | mxfp8 |
| `nz_format` | 29 | FRACTAL_NZ |
| `candidate_block_size` | 8 | 候选块 |
| `candidate_topk_blocks` | 2048 | 候选数 |
| `ROWS_PER_BLOCK`（indexer） | 128 | 索引 K 缓存 |

**不变量**：

- 所有池 **block 0 恒空**；
- 压缩池行数含 carry 行；
- 窗口池槽位按绝对/局部位置两套口径，`win_abs` 决定用哪套；
- replay 裁尾后池大小不缩（按逻辑长 L 开）。

## 9.2 collect 文件格式

```python
{
  "layers": {0: Tensor[T, H] fp32, 1: ..., ...},
  "rank": 0,
  "n": 40,
}
```

- 每层入口 `x`，三维时 `x.mean(dim=1)`；
- `DS41M_COLLECT_LAYERS` 可只落指定层；
- `--collect-ranks all|0|0,3` 控制哪些 rank 落盘；
- `compare_layers.py` 要求两边同格式、同层集合、同形状。

## 9.3 trace 语义

- 每个 rank 一份 trace；
- `Step Id` = 0..`iters×层数-1`（每层一个 step）；
- 同一层内多 stream 并行，`Start Time` 排序只是近似；
- 合成模式会多出预处理噪声（`StatelessRandom/Normal`、`Transpose/TransData`、
  `Range/FloorDiv`、`Cos/Sin/Pow`、`Fill/ZerosLike`），对拍时按已知差处理。

## 9.4 points 文件格式

```jsonc
// 旧
[{"L": 81920, "hit": 0.96, "batches": 1, "replay": true, "replay_tail": 128}]

// v2
[{"tag": "mix", "batches": [{"L": 8192, "hit": 0.96}, {"L": 16384, "hit": 0.90}]}]

// per-DP
[{"tag": "dp8mix", "dps": [{"rank": 0, "batches": [...]}, ...]}]
```

## 9.5 数值口径要点

- `M = hit.non_cached_tokens`（DeepInfer 口径，直接使用、无 128 对齐）；
  未显式给时按 `round((1-h)L)` 换算（仍不对齐）；
- `R = M + KEEP_TOKENS`（V4.1 默认 140）；
- `run_real.sh` / `run_hit.sh` 的 R 展示值同样用 `M + KEEP`；
- 命中位置是绝对位置 `L-R .. L-1`；
- `seqused_k = L//ratio`（索引器/压缩注意力的可见长度）；
- replay 段压缩 KV 不写（`slot=-1`）；
- 压缩池 `seqused_cmp_kv = (off+T)//ratio`（绝对可见压缩长度）。

---

# 10. 扩展开发与排障

## 10.1 加一个算子

```text
① opgraph.py：登记
   "my_op": dict(kernel="MyKernel", shapes=[["T","H"]], count=lambda c: 1,
                 affects=[...], note="...")
② recipes.yaml：决定哪些层有它、放在哪
   attention.c2a_source: [..., my_op, ...]
③ exec_ops.py：实现
   @op("my_op")
   def _my_op(st, ctx, w):
       ...
       st["x"] = ...
④ packed（可选）：@op_packed("my_op") 注册到 OPS["my_op@packed"]
⑤ 验证：
   run.py --dry-run                   # 计数/顺序
   tools/selfcheck.py --layers 25     # 真调用/健康
   align_ops.py                       # 与 trace 对齐
```

注意事项：

- `count(ctx)` 返回 0 时该层不进直方图；
- 未融合 MoE 类按 `_chunks(ctx) = ceil(T/chunk)` 放大；
- `affects` 目前只是人工元数据，换 backend 时记得同步；
- `dyn_mx_quant` 这类隐式算子应放进 `exec_ops.IMPLICIT`，不要真下发。

## 10.2 改模型结构

1. 改 `configs/*.yaml` 的 `model:`；
2. 若新增结构键，在 `model_spec.ModelSpec` 加字段 + `build()` 覆盖 + `validate()` 规则；
3. `exec_*._apply_spec(cfg)` 消费新常量；
4. `python3.11 tools/check_specs.py --strict` 校验全部配置；
5. `python3.11 run.py <cfg> --dry-run` 看层栈/算子图是否合理。

层角色变化（如新增 kv 源层）只需改四张表：
`compress_ratios / kv_source_layers / index_source_layers / candidate_source_layer / engram_layers`。
`weights_manifest.cond_holders(cfg)` 会自动推导每层该有哪些 COND 权重。

## 10.3 换 MoE 后端

- `backends.megamoe: ascendc | deepgemm`；
- `exec_moe._MoeBackend` 适配器：`inputs / prepare / call(单 span) / release_raw`；
- `_AscendcBackend` 用 `cann_ops_transformer.ops.mega_moe`；
- `_DeepGemmBackend` 用 `deep_gemm.fp8_fp4_mega_moe`；
- `auto` 先试 ascendc，失败**大声回退** deepgemm；
- `DS41M_MOEMODE=ascendc` 强制（失败会抛，别在正式采集里开）；
- 两后端**数值不等价**；对官方参考时用 deepgemm。

## 10.4 加采集点位

- 单点：`run_hit.sh -L <L>[@h|:M]`；
- 多点：`--points-json` / `--points-spec` / `--points <file>`；
- v2：一个 point 内多个 `{L,hit}` 批次，同一次 packed pass；
- per-DP：`dps` 字段（实验性，span 必须一致或 dummy 对齐）；
- 产物命名会自动跟随几何，不要手写目录。

## 10.5 调试流程（推荐顺序）

```text
1. run.py --dry-run                       # 先确认图对不对（不碰卡）
2. selfcheck --tier0-only                 # 环境/入口
3. runner --layers 2 ~ 4                  # 单卡最小真跑
4. selfcheck --layers 25                  # 全部算子种类
5. runner --prof --layers 4               # 看 trace
6. DS41M_DUMP_OPS / DUMP_SLICE            # 逐算子定位
7. compare_layers / cmp_ops_vs_official   # 数值对拍
8. scan_first_nan                         # NaN 定位
9. align_ops / diff_traces                # 图差异
```

## 10.6 常见坑

| # | 坑 | 处置 |
|---|---|---|
| 1 | `TMPDIR` 指共享盘会让 CANN JIT 卡死 | `env_pack.sh` 已 `unset TMPDIR`；别自己指回 `/tmp` 以外的奇怪路径 |
| 2 | 逻辑卡号 ≥1 会 aicore fault | 永远 `ASCEND_RT_VISIBLE_DEVICES=<物理卡>` + 逻辑 `npu:0`；别写 `--device npu:N`（N≥1） |
| 3 | 共享卡机器被兄弟容器占用 | 跑前 `npu-smi info`；只按 PID 杀；等 HBM 回基线 |
| 4 | `ds41` 命名空间注册 0 个算子 | `PYTHONPATH` 顺序：tilelang/deps/厂商包在 shim **之前**；用 `env_pack.sh` |
| 5 | QLI/QSLI 静默返回空候选 | 必须显式传 `candidate_topk_blocks / candidate_block_size` |
| 6 | mega_moe 建 HCCL 组端口冲突 | 端口 29517-29596 自动挑；`DS41M_MOE_PORT` 指定 |
| 7 | 真权重 engram 表占满主机内存 | `engram_tp_size=8`（每卡 ~12.9 GiB）；=1 会 1.6 TB pinned |
| 8 | 16k 真权重 OOM | `moe_chunk_max_len=4096`（官方同款 4 块）；`PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:4096` |
| 9 | 背靠背 8 卡 run 首次 collective 挂死 | 等 HBM 回基线（~5.1 GB/卡）再起下一跑 |
| 10 | `megamoe: none` 跑不起来 | 未融合链没有执行实现；用 ascendc/deepgemm |
| 11 | `run.batch` 改了没效果 | 它是占位配置；用 `--batches` |
| 12 | packed 与 sequential 数值不可直接比 | 跨不同批组成的逐位对拍不可作判据（MoE 累加顺序） |
| 13 | 命中 stage B 读到别层前缀 | `DS41M_PERLAYER_CACHE=1` |
| 14 | 多步 decode 第 3 步崩 | `DS41M_DECODE_BARRIER=1` 或 `DS41M_ENGRAM_HOST=1` |
| 15 | `run_hit.sh` 里 `DS41M_MOEMODE=...` 不生效 | 脚本会先清环境；用 `--moemode` 参数 |
| 16 | 文档说 `iters=4`，实际跑 6 | graph-tail 默认开且未显式 `--iters` 时为 6 |
| 17 | `run_real.sh` 文档说 ascendc/iters=3 | 实际默认 deepgemm / iters=6 / prof 2 轮 |
| 18 | `runner.py --parity` 报错 | runner 无此参数；用 `selfcheck --parity` 或 `run_real.sh --parity` |
| 19 | 打包漏掉新代码 | `build_pack.sh` 用 `git archive`，未提交改动不会进包 |
| 20 | 真权重 `mini_real_ascendc.yaml` 缺 `engram_tp_size` | 直接用有 1.6 TB pinned 风险；走 `run_real.sh` 会强制写 8 |

## 10.7 已知差（执行侧 vs 官方）

1. `rope` mini 2/层，模型 3/层（第 3 次在压缩 KV 上，属 compressor 路径）；
2. `tmove`/`mul` 在内核里可能被优化掉（`.contiguous()` / `×1.0`）；
3. `Cast` 比模型多（`logits.float()` 等）；
4. 合成模式预处理噪声（`StatelessRandom`、`Transpose`、窗口稀疏索引、rope 三角、池初始化）；
5. `MhcPost` vs `mhc_post_fwd_asc_kernel`：同语义不同实现；
6. 官方自身 run-to-run 1e-3 级噪声会被网络放大 → 对拍口径必须是同层同输入逐位。

---

# 11. 附录

## 11.1 文件地图

| 文件 | 职责 |
|---|---|
| `README.md` | 上手 / 最小条件 / 自检 / 命令 / 坑 |
| `PARITY_STATUS.md` | 精度对拍现状（40 层逐位一致 + 根因） |
| `WORKPLAN.md` | 当前会话工作与分支状态 |
| `env_pack.sh` | 环境探测 + 生成 `env_local.sh` |
| `configs/` | 全部运行配置 |
| `recipes.yaml` | 每层算子配方 |
| `opgraph.py` | 算子声明表 |
| `assemble.py` | 层栈/角色/T |
| `run.py` | dry-run |
| `runner.py` | 单 rank 真跑主控 |
| `launch_mini.py` | 多 rank 启动器 |
| `points_runner.py` | 多点位 |
| `exec_ops.py` | 算子执行表 + 基础设施 |
| `exec_indexer.py` / `_packed` | 索引器 |
| `exec_moe.py` | MegaMoE |
| `exec_engram.py` | engram |
| `exec_dspark.py` | DSpark |
| `weights.py` / `weights_manifest.py` | 权重 |
| `model_spec.py` | 结构单一事实源 |
| `engram_layout.py` | engram 布局 |
| `decode_layout.py` | P→D/D→P |
| `batch.py` | packed 布局 |
| `replay.py` | decoder replay |
| `alop.py` / `profiling.py` | profiling |
| `progress.py` / `progress_bar.py` / `ds41_log.py` | 进度/日志 |
| `tools/` | 自检、采集、对拍、分析、探针、打包、builder |
| `docs/` | 采集方法、decode、设计、历史报告 |

## 11.2 环境变量速查

### 环境引导（`env_pack.sh`）

| 变量 | 作用 |
|---|---|
| `DS41_SHARE` | 共享盘根 |
| `DS41_PACK_RUNTIME` | runtime 根 |
| `DS41_CANN` | 显式 CANN 路径 |
| `DS41_DEPS` | 纯 Python 依赖目录 |
| `DS41_PYTHON` | 指定 python3.11 |
| `DS41_TORCH_SITE` | torch/torch_npu site-packages |
| `TILELANG_RT` | tilelang/TileKernels 根 |
| `DS41_RT` / `DS41_PACK_ROOT` | 导出给运行时 |

### 执行/性能

| 变量 | 作用 |
|---|---|
| `DS41M_MOEMODE` | `auto` / `ascendc` / `deepgemm` / `decode` |
| `DS41M_MOE_CHUNK` | MoE 分块（`auto`/数字） |
| `DS41M_MOE_TP` | TP MoE（prefill+decode） |
| `DS41M_TP_DECODE_ONLY` | 只 decode 走 TP |
| `DS41M_MOE_DUMMY_SPAN` | per-DP span 对齐 |
| `DS41M_DECODER_REPLAY` | `1` 或 JSON `{on,w,cut,force}` |
| `DS41M_GRAPH_TAIL` | prefill 尾段入图 |
| `DS41M_DECODE_GRAPH` | decode 图（`1`/`check`/`tail`/`eager`） |
| `DS41M_DECODE_GRAPH_CUT` | 入图起始层（默认 21，0=全 40 层） |
| `DS41M_DECODE_BARRIER` | decode 每步 barrier |
| `DS41M_PASS_BARRIER` | 每 pass barrier |
| `DS41M_START_NOBARRIER` | 关起跑线 barrier |
| `DS41M_PERLAYER_CACHE` | 池/缓存按拥有者层隔离 |
| `DS41M_PAD_OPS` | 恢复占位算子 |
| `DS41M_UNIFY_PACKED` | points 默认 packed（默认 1） |
| `DS41M_PACKED_SEEDS` | 逐批 seed |
| `DS41M_PD_MERGE` / `DS41M_DECODE_LAYOUT` / `DS41M_PD_TP_XFER` | P→D/D→P 口径 |
| `DS41M_DECODE_HOST_NZ` | decode NZ 放主机 |
| `DS41M_SERVE_DECODE` | `full` / `inc` |
| `DS41M_DSPARK` | DSpark 草稿 token 数 |
| `DS41M_ENGRAM_HOST` | 主机取表（丢 EngramFetch） |
| `DS41M_NO_ENGRAM_PREFETCH` | 关 engram 预取 |
| `DS41M_ENGRAM_NOSIDE` | 关侧流 |
| `DS41M_WEIGHT_HOST_CACHE` | 关真权重 host cache |
| `DS41M_MOE_HOST_CACHE` | 跨进程专家权重 memfd |
| `DS41M_LOG_LEVEL` | `pure/error/warn/info/debug` |

### 诊断

| 变量 | 作用 |
|---|---|
| `DS41M_DUMP_OPS` | 层 0/1 逐算子快照 |
| `DS41M_DUMP_SLICE` / `_TAIL` / `_FROM` | 每算子/层张量切片 |
| `DS41M_OP_DIGEST` / `DS41M_PACKED_DIGEST_LAYERS` | 摘要 |
| `DS41M_STATS` / `DS41M_NAN_TRACE` | 统计 / NaN |
| `DS41M_QS_DUMP` | QSLI 入参 |
| `DS41M_SYNC_TRACE` | 慢同步栈 |
| `DS41M_MEMDBG` | 显存 |
| `DS41M_INJECT_QSLI` | 注入官方未初始化 16 行，复现 parity |
| `DS41M_NO_ENGRAM` / `DS41M_NO_CMP` / `DS41M_SKIP_KV` / `DS41M_FAKE_Q` | 消融 |

## 11.3 术语表

| 术语 | 含义 |
|---|---|
| **pass** | 一次完整前向（一次 `run_forward`） |
| **step** | profiler 打点，每层一次；`step 数 = iters × 层数` |
| **iteration** | main 外层一轮，含 1 个（packed）或 N 个（sequential）pass |
| **layer role** | win / c2a_source / c2a_consumer / c1a_source / c1a_indexsource / c1a_consumer |
| **kv_src** | 写压缩 KV 的层（2/8/14/20） |
| **idx_src** | 跑索引器的层（2/8/14/20/24/28/32/36） |
| **candidate** | 产候选块的层（20） |
| **engram** | 带外部哈希表的层（1/14） |
| **hit** | 前缀命中：只重算尾部 R 个 token |
| **stage A / B** | 真前缀先算一遍 / 被测量的命中尾段 |
| **replay** | decoder 尾段有界重放（cut 后只跑最后 W 行） |
| **packed** | 多请求 token 维拼接 + `cu_seqlens` 恢复边界 |
| **AC** | ascendc MegaMoE 的预转换权重 |
| **NZ** | FRACTAL_NZ 格式（format 29） |
| **P→D / D→P** | prefill 布局 → decode 布局 / 反向 |
| **QLI / QSLI** | QuantLightningIndexer / QuantSparseLightningIndexer |
| **MQSMLA** | mixed quant sparse flash MLA |
| **hc_pre / hc_post** | HyperConnection 前后处理 |

## 11.4 推荐阅读顺序

1. `README.md` —— 先跑起来；
2. 本文第 1～5 章 —— 建立全局图景；
3. `docs/PLAN.md` —— 设计动机与边界；
4. `opgraph.py` / `recipes.yaml` / `assemble.py` —— 算子图三件套；
5. `runner.py` 的 `run_forward` + `exec_ops.run_op` —— 执行主链；
6. `docs/CALIBRATION.md` / `docs/M2_REPORT.md` —— 校准方法与历史坑；
7. `docs/COLLECT_HIT.md` / `docs/COLLECT_REAL.md` —— 两种采集口径；
8. `docs/DECODE.md` / `docs/BENCH_RESIDENT.md` —— decode 与常驻；
9. `PARITY_STATUS.md` —— 数值对拍的现状与根因。

## 11.5 相关文档索引

| 文档 | 主题 |
|---|---|
| `docs/PLAN.md` | 总体设计、里程碑、边界 |
| `docs/CALIBRATION.md` | dry-run 直方图 × 真实 trace 校准 |
| `docs/M2_REPORT.md` | 单卡 40 层真跑 + trace |
| `docs/WEIGHTS_INVENTORY.md` | 真权重 96085 tensor / 48 分片清单 |
| `docs/COLLECT_HIT.md` | `run_hit.sh` 合成命中采集 |
| `docs/COLLECT_REAL.md` | `run_real.sh` 真权重采集 |
| `docs/DECODE.md` | full/inc decode、P→D/D→P、图捕获 |
| `docs/BENCH_RESIDENT.md` | 常驻 benchmark、W1/W2/W3、布局模式 |
| `docs/HANDOFF_MINI.md` | 10 分钟交接 |
| `PARITY_STATUS.md` | 精度现状与逐层对拍根因 |
| `WORKPLAN.md` | 当前分支/会话的工作与风险 |

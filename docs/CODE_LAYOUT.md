# ds41_mini 代码分类（2026-10-26）

> 与 `CODE_MAP.md` 的分工：CODE_MAP 讲**执行流**，本文讲**文件归属**。
> 分类按“改动时会一起动”的原则，不按字母序。

## A. 入口与编排（7 个）

| 文件 | 行 | 作用 | 关键符号 |
|---|---:|---|---|
| `run.py` | 202 | 最小驱动 + **dry-run**（不占卡，打印算子序列） | `dry_run/expand_layer` |
| `launch_mini.py` | 325 | 多 rank 启动器（每 rank 一进程、绑卡、日志/进度条） | `main` |
| `points_runner.py` | 494 | 多点位循环（warmup → profiler.start → iters → finalize） | `run_points` |
| `runner.py` | 4910 | 引擎主干（前向、层循环、replay、EC、抢跑、图、采集接入） | `run_forward/_run_forward_impl` |
| `assemble.py` | 201 | 按 config 堆 40 层、解析层角色 | `build_stack` |
| `metadata.py` | 161 | **一次 forward 的唯一元数据**（BatchLayout/Geometry/Policy） | `InputMetadata.attach/layout_of/set_layout` |
| `policy.py` | 105 | **规则/策略**（读 `configs/policy.yaml` + 环境变量覆盖 + 推导） | `resolve/rule/EnvBool` |

## B. 结构与布局（4 个）

| 文件 | 行 | 作用 |
|---|---:|---|
| `model_spec.py` | 330 | 模型结构规格（hidden/层数/专家/topk/比例…）+ 校验 |
| `batch.py` | 251 | batch 布局单一事实源（`cu_of/is_packed/seq_lens/pos_offsets`） |
| `replay.py` | 213 | decoder replay 裁剪策略（cut/W、packed/单序列统一入口） |
| `kv_manager.py` | 319 | KV 池/块表（照 DeepInfer cache_engine） |

## C. 算子执行（6 个 + modules/）

| 文件 | 行 | 作用 |
|---|---:|---|
| `exec_ops.py` | 2462 | 算子注册表与分发（`OPS/run_op/_qmm/_rope/hc_*`）+ varlen kv_compress |
| `exec_indexer.py` | 1564 | 索引器（idx_prolog/QLI/QSLI/K 缓存）+ varlen 实现 + CP 分块 |
| `exec_moe.py` | 1121 | MegaMoE 后端（ascendc/deepgemm、对称缓冲、路由） |
| `exec_engram.py` | 736 | engram（vendor fetch/wait + 融合 QMM） |
| `exec_dspark.py` | 743 | DSpark 草稿模型 |
| `opgraph.py` | 126 | 算子定义处（层里不准出现算子名） |
| `deepinfer_compress.py` | 55 | DeepInfer TileLang 压缩 kernel 封装（可选） |
| `modules/` | — | 模块运行时（spec yaml + compose + 校验工具） |

## D. 权重与编码（3 个）

| 文件 | 行 | 作用 |
|---|---:|---|
| `weights.py` | 751 | 权重来源（合成/真实）+ 切分/打包 |
| `weights_manifest.py` | 181 | 每层权重清单（自动生成） |
| `engram_layout.py` | 264 | engram 哈希布局/ngram id（逐字复刻官方） |
| `decode_layout.py` | 844 | P→D 权重排布转换（decode 形态） |

## E. 上下文并行（3 个）

`cp_layout.py`（纯布局计划）→ `cp_runtime.py`（组/通信）→ `cp_state.py`（设备张量 + ctx）。

## F. 采集与诊断（6 个）

| 文件 | 行 | 作用 |
|---|---:|---|
| `alop.py` | 2165 | ALOP 采集（ACL C 级 / torch_npu.profiler 二选一；2026-10-26 起只用前者） |
| `profiling.py` | 33 | 采集工厂（唯一后端 = ALOP） |
| `ds41_log.py` / `ds41_memlog.py` | 145/65 | 日志等级 / 显存日志 |
| `progress.py` / `progress_bar.py` | 95/570 | 阶段进度 / 采集进度条 |

## G. 配置与工具

`configs/`（模型 yaml + `policy.yaml` + ALOP 档位）、`recipes.yaml`、`tools/`（run.sh 唯一入口、
run_hit/run_real 后端、检查器/探针）、`thirdparty/`。

## H. 文档

`README.md`（总说明）、`WORKPLAN.md`（任务清单）、`CODE_MAP.md`（执行流）、`MODULE_DEV.md`（插件开发）、
`docs/`（本目录）、`PARITY_STATUS.md`、`PORTABLE*.md`。

## I. 数据/缓存

`cache_engram_layout.pt` / `cache_engram_token_map.pt`（engram 表缓存，可删可重生）。

---

## 物理目录（2026-10-26 已完成）

```
ds41_mini/
  core/     runner.py points_runner.py launch_mini.py run.py assemble.py
  meta/     metadata.py policy.py model_spec.py
  layout/   batch.py replay.py kv_manager.py engram_layout.py decode_layout.py
  ops/      exec_ops.py exec_indexer.py exec_moe.py exec_engram.py exec_dspark.py opgraph.py deepinfer_compress.py
  cp/       cp_layout.py cp_runtime.py cp_state.py
  trace/    alop.py profiling.py ds41_log.py ds41_memlog.py progress.py progress_bar.py
  weights/  weights.py weights_manifest.py
  configs/ tools/ modules/ docs/ thirdparty/ + recipes.yaml
```
- 根目录 `.py` = 0；入口（`core/*.py` 与 18 个 `tools/*.py`）顶部有统一路径引导，
  把 7 个包目录插入 `sys.path`（在 docstring/`from __future__` 之后）；
- 数据路径（recipes/configs/cache/thirdparty/alop_config）基准 = 树根；
- 入口命令：`bash tools/run.sh ...`（内部 `core/launch_mini.py`）。
- 验证见 `NOTES/2026-10-26-package-reorg.md`（dry-run / check_module_spec / 单点 104.0ms / ACL 8/8）。

# ds41_mini 代码地图（2026-10-26）

> 目的：一页看懂“一个 point 从命令到设备”的路径、每个文件管什么、packed/CP 两条口径的边界。

## 1. 执行流（一次 run_hit 的调用链）

```
tools/run_hit.sh                # CLI：点位/卡/轮次/prof/ALOP 配置；拼出 points 文件
  └─ python3.11 launch_mini.py  # 起 8 个 rank 进程；每 rank 只看见自己那张卡；超时/日志/进度条
       └─ runner.py             # 引擎：权重装载 → 逐层 forward →（可选）msprof
            └─ points_runner.py # 多点位循环：每点 warmup N 轮 → profiler.start → iters 轮 → finalize
                 └─ exec_ops.run_op(key, st, ctx, w)   # 唯一算子分派入口
                      ├─ OPS[key]            # 单序列实现（exec_ops / exec_indexer / exec_moe / exec_engram / exec_dspark）
                      ├─ OPS[key+"@packed"]  # packed 实现（exec_ops_packed / exec_indexer_packed）
                      └─ 图尾：runner 在层 cut 处 replay ACL 图（层 cut..39，W 行）
```

## 2. 模块表（顶层 .py）

| 文件 | 行数 | 管什么 |
|---|---:|---|
| `runner.py` | 4908 | 引擎主干：前向、层循环、graph-tail 捕获/回放、replay 裁尾、有界抢跑、EC 阀、profiler 接入 |
| `points_runner.py` | 489 | 多点位循环与采集边界（warmup → start → iters → finalize） |
| `launch_mini.py` | 325 | 8 rank 启动（绑卡/日志/超时/进度条） |
| `run.py` | 201 | 入口与参数 |
| `assemble.py` | 187 | 按 config 堆 40 层，解析层类型/角色 |
| `model_spec.py` | 330 | 模型结构参数单一事实源 |
| `weights.py` / `weights_manifest.py` | 753/181 | 权重来源（合成/真实）与每层清单 |
| `batch.py` | 251 | **batch 布局单一事实源**：`is_packed(ctx) = cu_of(ctx) is not None` |
| `exec_ops.py` | 2569 | 算子注册表与分发（`OPS`/`run_op`）、`_qmm`/`_rope`/hc_pre/post/…；packed 选路；**packed kv_compress 已并入（2026-10-26）** |
| `exec_indexer.py` | 1597 | 索引器（idx_prolog/QLI/QSLI/K 缓存/压缩）+ **packed 实现已并入（2026-10-26）** + CP 分块（`_qli_cp`/`_qsli_cp`） |
| `exec_moe.py` | 1121 | MegaMoE 后端（ascendc/deepgemm）、对称缓冲、路由/强均 |
| `exec_engram.py` | 736 | engram vendor fetch/wait + 融合 QMM（`npu_quant_matmul` 直喂 fp8+e8m0） |
| `exec_dspark.py` | 743 | DSpark 草稿模型（mtp.0/1/2） |
| `cp_layout.py` | 260 | CP 纯布局层（切分计划） |
| `cp_runtime.py` | 194 | CP 组建立 + rank-major all-gather 还原 |
| `cp_state.py` | 188 | CP 运行时状态：整数计划 → 设备张量 + 本地/全量 ctx |
| `kv_manager.py` | 319 | KV 池/块表（照 DeepInfer cache_engine） |
| `replay.py` | 213 | decoder replay：尾段裁剪（单序列/packed 统一入口） |
| `decode_layout.py` | 844 | P→D 权重排布切换 |
| `opgraph.py` / `modules/` | 126 | 模块运行时（插件式算子图） |
| `alop.py` / `profiling.py` | 2165/33 | 采集：ALOP 调度层（yaml→数据类型→起停），后端二选一（ACL C / torch_npu.profiler） |
| `tools/run_hit.sh` | — | CLI + ALOP 配置选择 + points 拼装 |
| `ds41_log.py` / `ds41_memlog.py` | 145/65 | 日志等级 / 显存日志 |
| `progress.py` / `progress_bar.py` | 95/570 | 阶段进度 / 采集进度条 |

## 3. batch 口径（varlen 是唯一语义，没有 packed 变体）

- **每个算子只有一个入口**（`OPS[key]`），内部按 ctx 选实现：
  `run_op` 里不再有 `@packed` 查找（2026-10-26 删除该概念）。
- 判据：`batch.is_packed(ctx)` ⇔ `cu_of(ctx) is not None`；**b=1 也是 varlen**。
- 入口分派（实现内部）：
  - `exec_ops.kv_compress` → `_kv_compress_pk`（varlen）/ `_kv_compress`（CP compact 段）
  - `exec_indexer._qli`/`_qsli`/`_idx_k_write` → `_qli_pk`/`_qsli_pk`/`_idx_k_write_pk`（varlen）
    / `_qli_cp`/`_qsli_cp`/`_idx_k_cache`（CP）
- 线上证据：`[packed] 真 batch：1 条 × L=65536（本 pass 26354×1 行）`（日志字样保留，语义即 varlen）。

## 4. CP 现状（要合并的对象）

| 位置 | 做法 |
|---|---|
| `exec_indexer.py:552 _qli_cp` / `:910 _qsli_cp` | 本地行 = head/tail 两块(+pad)，**按块各跑一次标准单序列 prefill**（块 = 某前缀的尾部） |
| `exec_ops.py` kv_compress | CP 时用全长 gather 的非 packed 实现 |
| `cp_state.py` | 把计划变成设备张量；提供本地 ctx（含 `cp_positions`/`cp_total_q`） |
| `exec_indexer.py:542 _cp_full_ctx` | CP→全长视图（`_cp_full=1` 让下游按 CP1 几何跑） |

## 5. CP → packed 合并方案（设计，未实施）

**目标**：删掉“CP 分块各跑一次单序列”，让 CP 也用 `OPS["qli@packed"]/["qsli@packed"]/["kv_compress@packed"]`。

**映射**：CP 本地两块 = packed 的 **2 个请求**：
```
cu_seqlens = [0, nh, nh+nt]          # head 块、tail 块
seqused_k  = [nh//ratio, L//ratio]   # 各自前缀长（官方语义 = 段尾对齐）
cmp_residual_k = [nh%ratio, L%ratio]
block_table = 每请求各自的块表（K 池是本 rank 全长一份 → 两请求可共用同一张表）
```
**步骤**：
1. `cp_state.py` 增加 `to_packed_ctx()`：产出上面 5 个张量 + `batch_size=2`；
2. `exec_ops.run_op` 去掉 `_cp_of(ctx) is None` 条件（改为“CP 且提供了 packed ctx 才走 packed”）；
3. 删 `_qli_cp`/`_qsli_cp` 与 CP 的 kv_compress 分块（保留 `DS41M_CP_PACKED=0` 回退一个版本）；
4. **逐位验证**：CP1/CP2/CP4/CP8 各 L=8192 与今日 digest 对齐（`tmp/CP_BUCKET_CHECK` 口径）。

**风险**：K 缓存写入几何（块起点 ≠ 0）、compact 段（层 20）的块表、`_cp_full` 预取路径；
CP 已有逐位结论，任何差异都必须先归因再改判定。

## 6. 常用入口

```bash
# 唯一对外入口（2026-10-26 合并）：三种模式
bash tools/run.sh --dry-run --config configs/mini_ds41_flash.yaml      # 只打印算子序列（不占卡）
bash tools/run.sh --weights synth --points-spec "64k@0.97/b1/r1" --noprof-only \
  --iters 4 --cards 0,1,2,3,4,5,6,7 --no-bar --moemode deepgemm -o <out>   # 合成权重
bash tools/run.sh --weights real -L 16k --mode hit --hit 0.96875       # 真权重 + 精度口径
```
（内部后端仍是 `tools/run_hit.sh` / `tools/run_real.sh`，参数见各自 `--help`。）

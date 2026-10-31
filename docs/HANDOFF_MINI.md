# HANDOFF_MINI —— 10 分钟上手

给第一次接手 `ds41_mini` 的人：这是什么、怎么跑、每个数字什么意思、现在卡在哪、下一步先做什么。
（方案细节看 `PLAN.md`；当前状态与已知差看 `README.md` + `M2_REPORT.md`；校准方法看 `CALIBRATION.md`。）

## 0. 30 秒定位

自研最小 benchmark：**不 import 官方模型代码**，用 `configs/*.yaml`（结构）+ `recipes.yaml`（每层算子配方）
+ `opgraph.py`（算子规格）装配出 DS4.1-Flash 的 40 层前向，跑一遍 prefill 出与官方 `ds41_light`
同构的 msprof trace。**验收 = 算子集合/次数/形状对齐（数值不要求）。**

## 1. 跑通（不碰卡，≈1 分钟）

```bash
source /mnt/l00972766/ds41_bench/env_image.sh
source /mnt/l00972766/setup_ds41_env.sh
unset TMPDIR                       # ★ 必须：TMPDIR 指共享盘会让 CANN JIT rmtree 报 Directory not empty
cd /mnt/xds/sfs/l00972766/ds41_mini

# ① 出算子图（毫秒级）：每层序列 + 全会话直方图
python3.11 run.py configs/mini_ds41_flash.yaml --dry-run

# ② 与官方参考 trace 对拍（应 20 行比值全 1.00）
python3.11 run.py configs/mini_ds41_flash.yaml --dry-run --dump-json /tmp/dry.json
REF=/mnt/xds/sfs/l00972766/ds41_light/res/20261013/light_8card_ep8_official_all_hit1m_cp1_hitfake/prof/prefill/7f59c04b04ec_741667_20261013131619591_ascend_pt
python3.11 tools/align_ops.py --ours /tmp/dry.json \
    --ref $REF/ASCEND_PROFILER_OUTPUT/kernel_details.csv --ref-passes 4

# ③ 看两个 trace 差在哪（拿现成的 mini trace 自 diff 应全 0）
python3.11 tools/diff_traces.py prof/m2_single_v4 prof/m2_single_v4
```

真跑才需要卡（**先 `npu-smi info` 看卡空不空，用 `--device npu:N` 指定**；别人在跑就别抢）：

```bash
python3.11 runner.py configs/mini_single.yaml --layers 4 --seq 1024              # 冒烟 ≈14s
python3.11 runner.py configs/mini_single.yaml --seq 1024 --allow-placeholder \
    --prof --out /mnt/xds/sfs/l00972766/ds41_mini/prof/m2_single                  # 整跑+trace ≈34s
```

## 2. 每个数字什么意思

| 数字 | 口径 | 期望值 / 含义 |
|---|---|---|
| `执行 884 个算子调用（20 种）` | **执行侧**：runner 汇总我们发出的算子调用数 | 40 层全跑；对应参考 trace 的 20 行对拍（CALIBRATION.md 历史口径 21 行含 DynamicMxQuant，它不在 dry-run 直方图） |
| `跳过 62 个` | Tier B：`mega_moe`(40)/`idx_prolog`(8)/`qli`(4)/`qsli`(4)/`engram_fetch`+`wait`+`proj`(各 2) | 跑不了就如实跳过，不造假（原因在 `exec_ops.py: SKIP`；`engram_proj` 是尚未登记的算子，同样计入跳过） |
| trace 里 `5093 行内核 / 38 种` | **内核侧**：`kernel_details.csv` 的行数 | 一个算子内部会发多个内核 + 合成模式的预处理噪声（见 README"已知差"第 6 条） |
| `比值 1.00` | `align_ops.py`：我们直方图 ÷ 参考 trace（按 pass 归一） | 1.00 = 次数对上了；偏离 >15% 会标"偏多/偏少" |
| `34.0s / 4 层 14.1s` | 整跑（40 层+prof）/ 冒烟（含首层 JIT ≈10s） | 阶段分解 S0 4s / S1 权重装载 16s / S2 前向 14s / S3 trace 8s（`progress.py`） |
| `--ref-passes 4` | 参考 trace 里有 4 个完整前向（4 个 Step），归一到单 pass | **Step≠pass**：mini 的 runner 每层调一次 `profiler.step()`，40 个 Step 也还是 **1 个前向**（`diff_traces.py` 不需要 --passes） |

## 3. 现在卡在哪（Tier B，按优先级）

1. **indexer（最优先，性价比最高）**：`idx_prolog` / `qli` / `qsli` 缺"indexer 上下文"
   （`indexer.wq_b/wk/weights_proj` 权重 + 候选块）。recipes 的 `c2a_source` / `c1a_source` /
   `c1a_indexsource` 段落**已经写好**，差的是执行式（`exec_ops.py` 里在 SKIP 名单）。
   接上后 ratio>0 层的压缩 KV 注意力分支就有真实 top-k（现在是"空索引"）。
   扩展点：`exec_ops.py` 的 `_load_plugins()` —— 写 `exec_indexer.py` 提供
   `register(OPS, SKIP, core)` 即可，不用改 core。
2. **mega_moe**：需要对称缓冲 + HCCL 组（1 rank 也行）。建议**跟 M3 EP8 一起做**（`exec_moe.py` 插件位已留）。
3. **engram**：`ElasticBuffer` + host 表（~98 GB/层，只有真权重才真需要）。M4。
4. **真权重 M2.5**：`RealSource` 读路径已实测打通（`configs/mini_real.yaml`，层 0 1.0 GiB / 层 14 含表 95.6 GiB）。
   下一步：`runner.py configs/mini_real.yaml --layers 1 --seq 1024` 冒烟 → 整跑 → 与官方做数值对拍。

## 4. 改东西的三条纪律（改错地方两分钟就能炸图，务必遵守）

1. **算子定义只在 `opgraph.py`**（kernel 名/形状/count），层里不许出现算子名；
2. **层的算子顺序只在 `recipes.yaml`**（哪层有哪个算子 = 存在性；每层几次 = opgraph 的 count）；
3. **真实调用式只在 `exec_ops.py`**（官方调用点行号都抄在注释里）。

标准工作流：**改 → `--dry-run`（毫秒级）→ `align_ops.py` 对拍 → 有卡才 `runner.py` 出 trace →
`diff_traces.py` 看两个 run 差在哪**。动权重/结构改 `configs/*.yaml`，切权重来源改 `weights.mode`。

## 5. 坑（都踩过）

- `TMPDIR` 指共享盘 → CANN JIT 编译收尾 rmtree 报 `Directory not empty`（保持默认 /tmp）。
- vendor 算子只 `import cann_ops_transformer`（**别显式 import `cann_ops_transformer_experimental`**，
  会把 ds41 组注册搞挂、indexer/mqsmla 全丢）；见 `tools/probe_ds41_ns.py`。
- mqsmla 的稀疏索引里无效列（-1）**必须靠尾部**，放前面内核会读负位置 → aicore 异常。
- 8 卡机器可能被别人共用：跑前 `npu-smi info`；异常崩溃后内存回不落时等一会儿再跑。
- `.tmp/` 是历次调试输出（可删）；`prof/`、`.tmp/` 都在 .gitignore 里，不会进 git。
- 合成权重模式每层现生成专家权重（进 trace 是 `StatelessRandom/Normal` 噪声）；看 trace 时别当成模型算子。

## 6. 延伸阅读顺序

`README.md`（现状/命令/目录/已知差）→ `M2_REPORT.md`（每个算子的真实调用式 + Tier B 清单）→
`CALIBRATION.md`（"每层几次"的证据链与再校准流程）→ `PLAN.md`（设计与后续里程碑 M3/M4）→
`WEIGHTS_INVENTORY.md`（真权重 114 种命名模式）。

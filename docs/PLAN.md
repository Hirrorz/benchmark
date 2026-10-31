# 方案：自研最小算子图 benchmark（ds41_mini）

## 0. 目标与非目标

**目标**：一个**独立实现**的 DS4.1-Flash 前向，跑一遍 prefill 就能产出与
`ds41_light` benchmark **同形的流水图**（MindStudio 直开）。判据 = 同 yaml/同 seq 下
`op_statistic.csv` 的**算子集合与次数一致、单算子时长差 ≤10%**。

**非目标**：数值正确性（权重是合成的，值无意义）· 命中率/调度/decode/常驻 · 结果统计。

**关键取舍**：算子与形状必须忠实（那是流水图的本体），其余一切能省就省。

## 1. 设计：三层 + 一张层清单

```
config(yaml) ──► assemble: 40 层数组（每层一个类型 + 参数）
                     │
   weights.py ◄──────┤ 按并行策略分配/填充/打包权重（EP 分片、量化格式、后端布局）
                     │
   layers.py  ───────┘ 每种层自己的 forward：发哪些算子、什么形状/后端
                     │
   run.py ──► 1 个 batch × N 次前向；--list-ops 打算子序列；msprof 出 trace
                     │
   tools/align_ops.py ──► 与 ds41_light 的 op_statistic 对拍（差异表）
```

### 1.1 层类型清单（"定义每种层"——本 session 已把官方结构查清）

| 类型 | 出现层 | 该层要发的算子（要点） |
|---|---|---|
| `Embed` | 入口 | embedding gather（+ multimodal 可选） |
| `EmbAct`(engram) | 1, 14 | 多阶 hash → host-offload 查表(fetch) → gather → 投影 → 与主干融合 |
| `WinAttn` | 0,1（ratio 0） | wq/wkv 投影、rope、**Mqsmla 稀疏注意**（仅窗口）、o_proj |
| `C2AAttn` | 2…19 | 上面 + **compressor**(2:1, wgate) + indexer；source 层(2,8,14) 写压缩 KV/index cache，consumer 只读 |
| `C1AAttn` | 20…39 | 1:1 compressor、QLI/QSLI 索引；**layer20 = 候选块来源**，24/28/32/36 走 QSLI，21~23 复用 shared_idx |
| `MoE` | 除前 3 层外全部 | gating/topk → **dispatch(EP alltoall) → grouped MM → swiglu+quant → combine**（或 MegaMoE 融合算子一条龙） |
| `Dense` | 0..2 | MLP 三段（gate/up/down）+ rmsnorm |
| `MHC` | 每层前后 | hc_pre / hc_post（mhc 混合），gate_topk |
| `Norm/LMHead` | 末尾 | final norm + lm_head（可选，流水图尾部） |

> 层清单来源：本 session 对官方 40 层的调查（kv_source=[2,8,14,20]、index_source=[2,8,14,20,24,28,32,36]、
> candidate=20、sliding_window=128、compress_ratios=2×19+1×20）。数值语义不追求，**算子集合/形状/顺序**要一致。

### 1.2 权重（只和并行策略有关）
- 分配：按 `parallel_config`（world/ep/cp/tp）切分；EP 分片决定每 rank 的专家数（如 384/8=48）。
- 填充：合成权重——**NPU 侧生成 + 分块 D2H**（≈24 GB/s，已在 ds41_light 验证；比 CPU `uniform_` 快 30×）。
- 打包：按后端——`ascendc`（raw 直接喂 op）/ `deepgemm`（transform 成 kernel 布局，含 scale 重排）。
- `--dump-weights`：打印每层权重 shape/dtype/format 表，用于核对"权重布局只由并行策略决定"。

### 1.3 算子与后端
- 与 benchmark 保持**同一批后端**：`torch_npu` 融合算子、custom vendor（compressor/megamoe/matmul）、
  tilelang/TileKernels（gate_topk/hc_post）、DeepGEMM-Ascend（可选 megamoe）。
- 并行通信：EP 用 alltoall(v)/MegaMoE 算子；CP 用 zigzag 段交换（M2/M3 再加）。

## 2. CLI
```
bash mini.sh <yaml> --seq 8192 [--batch 1] [--iters 3] [--prof] [--list-ops] [--dump-weights] [--dry-run]
```
`--dry-run`：不建模型、只按层清单**打印算子序列**（毫秒级，用来先对齐"有哪些算子、什么顺序"）。

## 3. 里程碑（每步都有可验收产物）

> 状态更新（2026-10-14）：**M0/M1/M2/M5 ✅；M3/M4 ⬜**。数字出处：`M2_REPORT.md` / `CALIBRATION.md`。

| 里程碑 | 内容 | 验收 | 状态 |
|---|---|---|---|
| **M0** | 建仓（git init + PLAN.md + README + .gitignore） | 仓可提交；方案落盘 | ✅ |
| **M1** | 层清单 + `--dry-run` 算子序列；`layers.py` 每类的骨架 | `--dry-run` 打出的序列与官方同 config 的 kernel 名对齐（人工核对） | ✅ dry-run 直方图 × 参考 trace：20 行比值全 1.00 |
| **M2** | 单卡可跑：1×T=1024 forward + msprof trace | MindStudio 能开；算子集合 = 官方单卡 trace 的子集/一致 | ✅ 40 层：884 次调用 / 20 种 / 34.0s / 零错误（Tier B 62 次跳过） |
| **M3** | 8 卡 EP8（MoE dispatch/combine + 权重分片） | 与 `ds41_light` cp1/ep8 同 seq 的 op_statistic 对齐（±10%） | ⬜ |
| **M4** | CP2/4/8 + engram 层 + 后端切换（ascendc/deepgemm） | 各 config 都能出 trace；对齐脚本通过 | ⬜ |
| **M5** | `tools/align_ops.py`（两份 op_statistic → 差异表）自动化对拍 | 一条命令给出"对上了/差在哪" | ✅（另补 `tools/diff_traces.py`：trace × trace 差异表） |

## 4. 风险与对策
1. **忠实度**：自己写的层容易漏算子/错形状 → 对策：M0/M1 用 `--dry-run` 先把算子序列对齐，
   再进 M2；每个层类型都有"官方对应物"的 kernel 形状作参照（本 session 的 trace 可查）。
2. **并行通信最难（EP alltoall / CP 交换）**：先单卡（M2）把图弄对，再上 EP（M3），CP 最后（M4）。
3. **显存**：与 ds41_light 同量级；先小 seq 再加。
4. **环境**：完全复用 `ds41_light` 的 env 三连（CANN overlay / ds41 算子命名空间 / python3.11），
   但**代码零依赖**（不 import bench.py / adapt.py）。
5. **卡与他人共用 / engram barrier**：沿用"清干净 + 等 40~60 s 重试"。

## 5. 工作量
M1 ≈ 半天 · M2 ≈ 1 天 · M3 ≈ 1~2 天 · M4 ≈ 1~2 天 · M5 ≈ 半天。
（M1/M2 单卡即可，不占整机，适合穿插在别人跑大活时做。）

## 6. 边界：什么叫"不用官方代码"（重要）

"自研"指**装配层与驱动层**，不含算子层。三层拆开看：

| 层 | 本方案（mini） | 说明 |
|---|---|---|
| ① 模型装配 / 谁发算子 | **自研**（`layers.py` 每种层自己组织算子顺序） | 这是本方案要替换掉的"官方模型代码" |
| ② 引擎 / 权重装载 / 并行切分 | **自研**（`weights.py` + `run.py`） | 只要并行策略与官方一致，权重布局就一致 |
| ③ **算子本身 + 通信库** | **原样复用**：CANN custom vendor（QLI/QSLI/compressor/engram/MegaMoE/HcPre/HcPost/gate_topk…）+ torch_npu 融合算子 + HCCL | 这些**不是**官方仓库里的 Python 代码，而是 CANN 侧 vendor 包；官方仓库只是调用者。替换它们 = 流水图不再可比 ✗ |

推论：
- 验收判据（同 yaml/同 seq 下 `op_statistic` 对齐）**决定了算子不能换**，只能换装配。
- 因此 mini 仍需同一套容器环境（CANN + vendor 包 + python3.11），但 **Python 代码零依赖**：
  不 import `executor/`、不 import `models/deepseek_v4_1/`、不 import `bench.py`/`adaptlib/`。
- 唯一从 ds41_light"继承"的东西是**知识**（40 层结构、算子清单、形状清单、踩坑记录），不是代码。

## 7. 灵活性第一：任何流程/算子改动都要"改一处、立刻可验"

### 7.1 架构（为可改而设计）

```
config.yaml        结构：40 层的类型/参数/kv_source/engram…（结构变了只改这里）
recipes/*.yaml     每种层类型的**算子配方**：按顺序列出调哪些算子、依赖谁、什么形状
ops.py             ★ 算子注册表：一个算子 = 一个 spec（名字 / 来源(vendor|torch|deepgemm) / 形状推导 / 后端选项 / prof 名）
layers.py          薄封装：按 recipe 依次调 ops（不含任何硬编码算子细节）
assemble.py        按 config 把层堆起来
weights.py         按并行策略分配/填充/打包
run.py             驱动 + msprof；`--dry-run` 用同一份 recipe 打印算子序列与形状（不碰卡、毫秒级）
tools/align_ops.py 对拍：我们的序列 vs 官方/benchmark trace 的 kernel 名+形状 → 输出"差在哪一层哪个算子"
overrides/         我们**有意偏离**官方的点（登记 + 开关），一眼看清差异面
```

三条纪律：
1. **layers.py 里不准直接出现 `torch_npu.xxx` / vendor 算子名** —— 只能写 `OPS["rope"](x, ...)`；
   算子的真实调用、形状推导、后端选择全在 `ops.py` 一处。
2. **层的算子顺序不准写在 Python 里** —— 只在 `recipes/*.yaml`。
3. 每次改动后**先 `--dry-run`**（毫秒级出 op 序列）→ 再 `align_ops.py` 对拍 → 最后才上卡跑 trace。

### 7.2 "改哪儿"速查表

| 发生了什么 | 改哪里 | 验证 |
|---|---|---|
| 某算子换实现（如 indexer 从 ascendc 换 fused） | `ops.py` 里该算子加一个 backend | `--dry-run` + `align_ops` |
| 某算子调用点/参数变了（如加了 activation_clamp） | `ops.py` 该算子的 spec | 同上 |
| 某层的流程变了（如 source 层要额外写一份 cache） | 该层 `recipes/<type>.yaml` 加一行 | 同上（顺带看形状） |
| 层数/类型/结构变了（kv_source 层号、engram 层） | `configs/*.yaml` | `assemble.py --print-stack` |
| 并行策略变了（world/ep/cp） | `configs/*.yaml` 的 `parallel_config` | `--dump-weights` 核权重布局 |
| 官方/benchmark 也改了，要跟着对齐 | 先 `--dry-run` 看差异 → 改 `ops.py`/recipe → `align_ops` | 对拍通过即完成 |

### 7.3 为什么这样最省改动

- **形状推导集中在 ops.py**：算子改了只要重写它的 `infer_shapes()`，所有调用点自动跟上（不会漏改某层）。
- **dry-run 与真跑共用同一份 recipe**：不会出现"图描写的"和"实际跑的"两套真值（这是最容易腐化的地方）。
- **align_ops 是闭环**：不靠人眼比对 MindStudio，改成"命令输出差异表"，改完立刻知道对不对。
- **overrides/ 记录偏离**：任何"我们故意的不同"都有名字有开关，避免以后有人以为是 bug 或漏了。

## 8. decoder128（decoder-side SWA bounded replay）怎么进 mini

**它是什么**：最后一个 KV source 层（=20）之后的 21~39 层只持有滑窗 KV，prefill 时只需对
**每条 prompt 的最后 W(=128) 个 token** 计算 → 把这一段裁掉，是该特性唯一的图变化。
（vLLM #58132 已合入；`ds41_light` 的 `adaptlib/replay.py` 已实现并验证：
等价回归输出逐位一致；cp1/8192 557→349 ms；hit97 扫描 −34%~−49%。）

**在 mini 的架构里它是什么**：一个**层间形状变换**，天然落在装配层，不需要动任何算子：

```
config.yaml
  replay: {enable: true, cut: 21, w: 128}      # cut=最后一个 kv_source+1
assemble.py
  ... layers[0..20]  (T = L)
  ┄┄ GatherTail(rows = 每请求最后 w 行)  ← 新增的一个"行选取"stage（index_select/gather）
  ... layers[21..39] (T = w)
  ┄┄ ScatterTail（写回全长；末步只取最后一行）
```

要点与约束（照抄已验证实现的口径，保证 trace 可比）：
1. **形状规则**：`T(layer) = L (l=0..cut-1) | w (l=cut..39)`，`cut = max(kv_source)+1 = 21`；
   裁剪后 `R = min(L, w)` 才是尾段的真实 token 数（`L<=w` 时等价不裁）。
2. **额外算子**：行 gather（`index_select`/`copy_`）、写回 scatter；host 侧还有一次 metadata 重组
   （不是 device 算子，trace 里看不到——与 `ds41_light` 一致）。
3. **语义细节（对 trace 无影响，但别改坏）**：窗侧临时 KV cache 按**绝对位置**布块，使压缩 KV 的
   index 偏移量不变（`ds41_light` 的经验）；mini 只需保证**算子与形状**一致，不必复刻该语义。
4. **限制**：仅 cp_size==1 + eager（CP 会把"最后 128 token"切到不同 rank）；与命中场景天然契合
   （`R = (1-h)·L + 128`，尾段正好是那 128 行）。
5. **验收**：`--dry-run` 应打印 `layers 0..20 @T=L` → `GatherTail(128/请求)` → `layers 21..39 @T=128`；
   与 `ds41_light` **replay-on** 的 trace 对拍：尾段算子的 shape 由 L 变 128、QSLI/Mqsmla 次数不变但形状缩小，
   总量应显著下降（8192 时约 −37%）。
6. **落地位置**：`replay` 不进 `ops.py`（不是算子），进 `assemble.py` 的 stage 列表 + `recipes/` 的模型级
   变换声明；开关只在 config，默认关（与 `ds41_light` 的 `DS41L_DECODER_REPLAY` 同语义）。

## 9. 双权重模式：合成 / 真实**对等一等公民**

```yaml
weights:
  mode: synthetic            # synthetic | real —— 两种模式都必须能跑，且共用同一套切分/布局代码
  path: /mnt/xds/sfs/DeepSeek-V4.1-Flash-cann-mxfp8-mxfp4   # mode=real 时必填
```

### 9.0 统一接口（这是"对等"的落地方式）

```python
class WeightSource(Protocol):
    def get(self, name, shape_hint, **fmt) -> WeightBlob   # 本 rank 该 tensor 的切片（+scale/打包后）
    def iter_layer(self, layer_idx) -> Iterator[...]       # 按层流式（省内存）
    def summary(self) -> list[str]                          # --dump-weights 用

SyntheticSource   # NPU 生成 + 分块 D2H（≈24GB/s）
RealSource        # index 解析 + lazy 读 + 后端布局转换
```

**两条硬纪律**（否则两种模式会漂）：
1. **合成模式也走同一套 EP/TP 切分**——不是"随便造一份"，而是"造出**本 rank 该有的那份**"（形状/dtype/scale 分布与真实一致）。
2. **切分、布局转换、后端打包的代码只有一份**，`SyntheticSource`/`RealSource` 只负责"tensor 从哪来"。
   → 于是"合成能跑通"就几乎等价于"真实能跑通"，反之亦然；两条路互相当测试。

### 9.1 共用管线（两种模式都走）
| 步 | 做什么 | 谁做 |
|---|---|---|
| 取 | 合成：按切片生成；真实：读分片 | 各自的 Source（唯一差异点） |
| 切 | 专家 EP 连续切分 / head TP / engram 按 `engram_tp_size` | **共用** `weights.py` |
| 包 | 量化+scale 原样；DeepGEMM 走库函数 `transform_weights_for_mega_moe()`；AscendC 原样 | **共用** `weights.py` |

### 9.2 真权重实况（已核对）
`/mnt/xds/sfs/DeepSeek-V4.1-Flash-cann-mxfp8-mxfp4`：**476 GB / 48 分片 / 96085 tensor**（94722 专家、47675 带 scale），
**权重本身已量化**（mxfp8/mxfp4）→ mini 不做量化，只做读/切/包。
`engram` 两张表 ≈49 GB 必须走 host-offload（CANN `ElasticBuffer`；⚠️ `engram_write` 实测 36.4 s 且概率性
`HcclBarrier failed`，要做自动重试 + 特征串识别）。

### 9.3 验收矩阵（两种模式在每个里程碑都要出结果）

| 里程碑 | 合成 | 真实 |
|---|---|---|
| M2 单卡 | ✅ trace，算子集合对齐 | —（先不来） |
| **M2.5** | ✅ 回归（每次改动都跑，秒级/分钟级） | ✅ 同 config 同 seq 的 `op_statistic` 对齐 + **logits 数值对拍** |
| M3 EP8 | ✅ | ✅（EP 切分一致性在真权重下才真正验证） |
| M4 CP+engram | ✅ | ✅（engram offload 只有真权重才需要） |

**一句话**：合成模式负责"天天能跑、改完立刻验图"，真实模式负责"验真形状/真布局/真数值"，两者共用一套切分与布局代码，互为回归。

### 9.4 风险与对策
| 风险 | 对策 |
|---|---|
| 布局转错 → 静默错 | 逐层输出统计冒烟（min/mean/absmax 与 ds41_light 同层比对），不等跑完 |
| 476 GB 读取慢 / NFS 抖动 | lazy + 只读本 rank 切片；`--dump-weights` 核名字/shape/dtype/来源分片 |
| engram offload 概率性失败 | 自动重试（清干净 + 等 40~60 s）+ 特征串匹配 |
| 真权重显存顶墙 | 小 seq 起步 + `--seq` 上限预检；必要时复刻 memo/release 思路 |

## 10. 命中率支持（两种 KV 来源：真实前缀 / 构造随机）

```yaml
hit:
  ratio: 0.97          # h
  source: fake         # fake(构造随机 KV，只算后缀) | real(先真跑前缀再命中)
  replay_tail: 128     # 命中步末尾必须重算的 token 数（SWA 窗口），与 decoder128 天然契合
```

### 10.1 口径（照抄 ds41_light 已验证的定义）
- 命中后实际计算的 token 数 **R = (1−h)·L + replay_tail**，缓存前缀 **H = L − R**。
  例：h=0.97 → L=1m 时 R≈31616（这条曲线我们已实测 8 个 L 点，可当验收参照）。
- 两种来源**在图上是同一件事**（同 R 下算子集合与形状完全一致），差别只在于：

| source | 做法 | 代价 | 用途 |
|---|---|---|---|
| `fake` | KV 池里"标记/填充" H 个 token 已存在（**随机内容**），只算 R 个 token | 便宜（≈一次 R 的前向） | 天天跑、扫曲线、换算子 |
| `real` | **先真跑一遍 H token 的前缀 prefill** 写进真 KV，再跑 R 的命中步 | 贵（多一次全量 prefill） | 验证命中路径语义、与官方 offline 对拍 |

### 10.2 实现要点
1. 命中**不新增算子**，只改**形状与元数据**（`seq_used_q=R`、slot/block table 指向已存在的池 ✓）——这正是 §7 的"改一处"：只需在 `assemble.py` 决定本次前向的 `L→R` 映射 + 在 metadata 组装处标"H 已存在"。
2. **最后 `replay_tail`(=128) 个 token 必须重算**（它们的窗/index 依赖自身 KV），所以 `R = (1−h)L + 128` 而不是 `(1−h)L` —— 这一条与 §8 的 decoder128 是同一件事的两面。
3. `real` 模式复用同一份 weights/KV 池，只是"先跑一遍前缀"；两个 step 都要能单独 profiling（`--prof-step prefix|hit|both`）。

### 10.3 验收
与 `ds41_light` 的 **hit97 扫描 trace** 对齐（已有 8 个 L 点 × on/off 全量 trace，含 `op_statistic.csv` 与形状）：
同 L、同 h、同 source 下算子集合/次数一致、时长 ±10%。

---

## 11. 并行策略支持（cp / ep / tp / dp 可组合）

```yaml
parallel_config: {world_size: 8, cp_size: 1, moe_ep_size: 8, attn_tp_size: 1, dense_tp_size: 1, lmhead_tp_size: 1}
```

### 11.1 每种策略"改什么"（对图与权重的影响，集中登记）

| 策略 | 权重变化（`weights.py`） | 图变化（`ops.py`/recipe/assemble） | 已验证基线 |
|---|---|---|---|
| **EP** | 专家按 EP 连续切分（384/8=48 每 rank） | MoE 里多 **dispatch/combine 通信**（或走 MegaMoE 融合算子一条龙） | ep8 主线 ✓ |
| **CP** | 不变（序列切分不切权重） | 序列按 **zigzag 段**切；每 rank 处理 2 段；win KV 需**跨段回放**、压缩 KV 段合并；attention 段内做 | cp1 547.6 / cp2 345.4 / cp4 224.7 ms ✓ |
| **TP** | head / 中间维切分 | 投影后 **allreduce** | attn_tp=1 为主 |
| **DP** | 不变 | 请求分组（单请求 mini 影响小） | dp8 主线 ✓ |

### 11.2 可组合性约束（提前钉死，免得组合爆炸）
- **decoder128（§8）目前仅 cp_size==1**（CP 会把"最后 128 token"切到不同 rank）→ mini 里同样限制，违反时明确报错而非静默降级。
- `hit.real + CP`：前缀 prefill 也走 CP，需保证两段的 KV 都写对（M4 再加）。
- 组合矩阵维护在 `docs/matrix.md`（哪些组合已验证 ✅/未验证 ⬜/不支持 ✗），一眼看清。

### 11.3 验收
cp1/cp2/cp4/cp8 × hit{fake,real} 各出一份 trace，与 `ds41_light` 同 yaml 对齐（cp1/cp2/cp4 已有实测数字可对照）。

## 12. 运行进度：百分比 + 当前阶段（可读、可机读、失败可定位）

### 12.1 阶段划分与权重标定（**用实测值标定，不是拍脑袋**）

**mini 现状（runner 的 4 阶段，即 `progress.py: STAGES_MINI`）**——2026-10-14 实测标定：

| 阶段 | 权重（秒） | 说明 |
|---|---|---|
| S0 环境与算子注册 | 4 | import torch_npu + 注册 vendor 命名空间 |
| S1 逐层：权重装载 | 16 | 合成权重按层生成 + 打包（每次跑完 runner 打印实测值，换机器/换算子后照它更新） |
| S2 逐层：前向执行 | 14 | 40 层发算子 |
| S3 trace 汇总 | 8 | 含 profiler 解析（不 prof 时 ≈0） |

整跑实测：**4 层冒烟 14.1 s（含首层 JIT ≈10 s）；40 层 + trace 34.0 s**（884 次算子调用 / 20 种，
零错误；`M2_REPORT.md`）。

**M3 起接入完整引擎（多 rank / CP / engram）时的 8 阶段参照**——阶段表来自 2026-10-14 在本机实测的启动分解（cp4/noeng/合成权重，总 ≈120 s；真权重另标）：

| # | 阶段 | 典型耗时（合成/真权重） | 阶段内可数单位（用于推进百分比） |
|---|---|---|---|
| S0 | 环境与导入（torch_npu/CANN init） | 17 s | 无 → 只显示阶段与秒数 |
| S1 | 通信域初始化（HCCL groups） | 2 s | 无 |
| S2 | 模型组装（层堆叠 + 内核实例化） | 27 s | 已建层数/40 |
| S3 | **权重装载** | 45 s / 长（476 GB 读+切+布局） | 合成：层/40；**真实：已读字节/总需字节** |
| S4 | 池分配 + engram offload | 2 s / **36 s** | 表数/2 |
| S5 | 预热（首个 step） | 17 s | 无（首跑含编译，单独标"编译中"） |
| S6 | 测量前向 N 次 | 由 seq/策略决定 | iter/i（可细到 层/40） |
| S7 | 收尾（trace/汇总） | 秒级 | 无 |

> 阶段无细粒度时**退化为"阶段 + 已用时 + 基于标定的 ETA"**，绝不显示拍出来的假百分比（那比没有更糟）。

### 12.2 输出形态

```
[12:34:56]  45% │ S3 权重装载 12/40 层 │ 已用 82s · 预计再 1m02s
```
- **tty**：同行 `\r` 刷新（每秒 ≤1 次）；**非 tty**（nohup/重定向）：改成整行打印（不刷屏）。
- **8 rank**：只有 **rank0** 打进度（其余 rank 各自写文件），避免 8 份交错 ✗。
- **机读**：同时写 `res/<date>/<yaml>/progress.jsonl`（每行 `{ts, pct, stage, detail, elapsed}`）→
  以后任何一次挂住，直接看最后一行就知道停在哪一阶段（今天的 `ExecutionEngine(...)` 卡死就属于 S2/S3 ✗）。
- **失败时**：异常/超时退出前打印"最后阶段 + 最后一条进度行 + 各 rank 最后一条日志尾"，不用再去翻 8 个文件。
- 开关：`--progress auto|on|off`（默认 auto：tty 开、重定向也开但降频）、`--progress-every N`（默认 1 s）。

### 12.3 与 profiling 的关系
msprof 只是包在驱动外，stdout 进 trace 无冲突；S5/S6 的进度点与 trace 内的 step 边界对齐（`--iters 3` 时能看到 3 个前向段）→ 进度条与 MindStudio 时间线可以互相印证。

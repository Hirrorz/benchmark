# mini 的 decode 阶段（贪心续写）

> 2026-10-17 · 两版实现：全量重算（金标准）与增量（prefill 一次 + 每步 1 token）

## 1. 用法

```bash
cd /mnt/xds/sfs/l00972766/ds41_mini && source env_pack.sh
export DS41M_MOEMODE=ascendc          # 精度对官方时用 deepgemm（见下）
export DS41M_MOE_CHUNK=auto
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:4096

# ① 全量重算版（每步重跑整段 prefix；数值严格等价 prefill）
python3.11 launch_mini.py --config /mnt/xds/sfs/l00972766/winfix/config_pre.yaml \
  --world 8 --cards 0,1,2,3,4,5,6,7 --logdir <日志目录> --seq 16384 \
  --ids-file /mnt/xds/sfs/l00972766/parity/ids_16k.pt --decode 8

# ② 增量版（prefill 1 次 + 每步 1 token，读写 40 层各自的持久池）
... --decode 8 --decode-mode inc

# ③ 稳定性：每步边界 barrier（各 rank 步序钉齐；计时口径会含等待）
DS41M_DECODE_BARRIER=1 ...
```

短 prompt 用 `--prompt "..."`（该权重目录的 `tokenizer_config.json` **没有 chat_template**，
裸续写可能是"对不起…"这类文本，属正常；要聊天式输出需自己套模板）。

## 2. 增量 decode 的关键口径（都已实现并验证）

| 项 | 做法 |
|---|---|
| 层内池 | 强制 `DS41M_PERLAYER_CACHE=1`：40 层各自的窗口池/压缩池/索引缓存**同时存活**（prefill 里"各层共用一个池、读写紧邻"的侥幸不再成立） |
| 池尺寸 | 一律按 `L_total = len(prompt)+n_new` 一次开好（不中途重分配） |
| 窗口 | 新 token 写槽 `(1+t//128)*128 + t%128`；读侧窗口索引走**绝对位置**（`win_abs`） |
| 压缩 | 压缩器 ring state（`_cmp_state_L{i}`）留在 shared 跨步存活；压缩行/索引 K 行按 `off//ratio` 绝对行号，只写本步新增行 |
| engram | n-gram hash 需要上文：`input_ids` 末尾多带 3 个 id（`engram_ctx=3`），`_engram_fetch` 取 `h[-T:]`（len==T 时逐位不变） |
| MoE 缓冲 | ascendc 对称缓冲容量按**固定 chunk** 建、不随每步 T 变（否则各 rank 重建不同步 → aicore error） |

`run_forward` 新增 `pos_offset` / `l_full` / `engram_ctx` 三个参数，prefill 路径不传时行为与旧版逐位相同。

## 3. 已验证的数字

| 场景 | 结果 |
|---|---|
| 短 prompt「中国的首都是」增量 8 token | 每步 **0.69 s**；文本 `\n\n对不起，我有点激动。这句话`（连贯中文） |
| 同一 prompt **全量 vs 增量 第 1 步** | **完全一致**（id=271，top3 15.62/15.44/15.38） |
| 16k ids 增量 8 token | 第 1 步 `id=396 ' that'`（top1 33.75 vs 16.5）= 与官方参考同 id；第 2 步 `id=1225 ' she'`（46.5） |
| 16k 全量 vs 增量 第 1 步 | **完全一致**（同上 top3） |
| **16k 增量 8 token（完整）** | `" that she could not go on living with"`，**0 设备异常**，均 **0.74 s/token**（`DS41M_ENGRAM_HOST=1`） |
| **16k 增量 + 每步 barrier（厂商 engram）** | 同上 8 个 token **逐位同 id/同分数**，0 异常 ⇒ barrier 即可稳住厂商路径 |
| **chat 模板演示（zh）** | 「中国的首都是哪里？请只回答城市名。」→ **`北京`** + EOS ✓ |
| **chat 模板演示（en）** | 「What is the capital of France? Answer with one word.」→ **`Paris`** + EOS ✓ |

> 稳定性结论：多步 decode 必须 `DS41M_DECODE_BARRIER=1`（各 rank 步序对齐；厂商 engram
> 侧流对步序敏感，实测无 barrier 时 16k 第 3 步会 rank1 vector core exception）；
> 或 `DS41M_ENGRAM_HOST=1`（主机取表，无侧流 HCCL）。两者输出逐位相同。

## 4. 已知问题（已结 / 遗留）

- **已修**：ascendc 对称缓冲随每步 T 增大而重建 → 各 rank 不同步 → aicore/HCCL error。
  修法：缓冲容量固定（ascendc 按 chunk、deepgemm 按 L_full），见 [[ds41-moe-backend-parity]]。
- **已定位**：16k 增量无 barrier 时第 3 步 rank1 `vector core exception` = 各 rank 步序漂移
  撞厂商 engram 侧流 HCCL。用 `DS41M_DECODE_BARRIER=1` 或 `DS41M_ENGRAM_HOST=1` 即稳。
- **遗留**：全量重算版仍会偶发设备级 segfault/HCCL watchdog（多步、每步 T 变化）；
  **只用于单步金标准**，长续写请用增量版。
- **遗留**：增量版第 2 步起 logits 与全量重算**不同**（同 id、分数差 ~4）：T=1 与 T=N 在
  MoE/压缩器上的分批差异；待定位（不影响文本连贯性）。

## 5. 与官方的数值关系

对官方参考比精度时必须 `DS41M_MOEMODE=deepgemm`（官方参考也是 deepgemm）；`ascendc` 与
`deepgemm` 两个 MegaMoE 内核**数值不等价**（逐算子证据见 `PARITY_STATUS.md` 与本轮日志：
层 0 除 `moe_out` 外全部逐位相同）。decode 的池/索引实现与 MoE 后端无关，两版共用。

## 6. decode 入图（ALOP 采集）

```bash
# prefill 只当 warmup（不进图）；profiler.start() 在 prefill 之后，逐 decode 步采集
DS41M_DECODE_BARRIER=1 python3.11 launch_mini.py --config .../config_pre.yaml \
  --world 8 --cards 0,1,2,3,4,5,6,7 --seq 16384 --ids-file parity/ids_16k.pt \
  --decode 2 --decode-mode inc --prof --alop-config configs/alop_mindstudio.yaml \
  --out <trace目录> --timeout 600

# 汇总 8 rank 的 op_statistic.csv
python3.11 tools/op_time_table.py --trace-dir <trace目录> --top 25
```

- **`--decode N` 只产生 N-1 次 decode forward**（最后一个 token 不需要再前向）⇒ 要采 K 个
  decode 步就给 `--decode K+1`。
- 实测（16k prefill + 1 步 decode、8 rank）：`MegaMoe` 320 次（40 层 × 8 rank）、
  `MqsmlaKernel` 320、`HcPre` 640、`TransData` 3440（真权重逐层重打包，decode 最大头，
  770 µs/次）；trace 共 129,656 次算子调用。
- **稳定性**：厂商 engram 侧流在 decode 下偶发 `moe_topk_gate ... nan or inf`（尤其在
  `off % ratio != 0` 的近似 carry 步）。要**完整图**（含 `EngramFetch`）就保留厂商路径、
  并把采集步控制在偶数位起步的 1 步；只求稳定可加 `DS41M_NO_ENGRAM_PREFETCH=1`（但图里没有 EngramFetch）。

## 7. decode 提速：官方口径调研（2026-10-17）

**官方公布数据**（本机 pristine 仓 `docs/models/deepseek_v4_1/`）：
- 低时延 GBS=1、8 卡、8192：**TPOT 2.48 ms**（DSpark-7，平均步长 5.7）⇒ **原始单步 ≈14 ms**；
- 32 卡高吞吐：4.40 / 9.41 / 13.79 ms TPOT（DSpark-5，平均接受 4.1）⇒ 原始单步 ≈18–56 ms。

**mini 一步 0.75 s 的构成**（ALOP 实测）：TransData 336 ms（MoE 权重 transform 每层每步重做）+
MegaMoe 230 ms（**decode 用错算子**，官方只在 prefill 用）+ 胶水 ~150 ms + 真算子 ~20 ms。

**已落地的优化/结论**
1. `exec_moe._PACK_CACHE_L`：注意力/gate 权重按层 prepack 缓存 —— **无收益**（TransData 不是它）。
2. `exec_moe._PREP_L`：MoE transform 按层缓存 —— **会 OOM**（34 GB/卡，81.84 GiB 已分配 + 5.12 GiB 申请失败）
   ⇒ 默认关，只在 `DS41M_PREP_CACHE_N>0` 时缓存前 N 层。
3. **官方 decode MoE 已实现**（`exec_moe._moe_decode`，`DS41M_MOEMODE=decode`）：
   `npu_moe_distribute_dispatch_v2 → npu_grouped_matmul×2 + swiglu_group_quant → combine_v2` + 共享专家。
   - 已修：`expert_ids`/`expert_scales` 必须 **[T,K] 二维**；dispatch 的 per-token scale 要 `reshape_mx_scale`；
     GMM 权重必须 **NZ 格式**（`format_cast(29, customize_dtype=fp8, input_dtype=fp4)`）。
   - 仍卡：`aclnnGroupedMatmulWeightNz` 报 **161002**，且 `expand_x` 48 行（T=1 时应 ~1 行）说明
     dispatch 的 EP 分片没生效 —— 需要接官方 `CommManager` 的 **MC2 通信域**（`moe_ep_group_mc2`）才能继续。
4. decode MoE 一旦打通，可**同时**消掉 TransData（用原始 mxfp4 权重，无需 transform）与 MegaMoe，
   且内存不增 —— 这是通向 ~50 ms/步（再叠 DSpark+入图可到 10–15 ms/token）的主路径。

### 7.1 为什么 decode 不能直接用 MegaMoE（而 transform 不是"换权重"）

- 权重只有一份：checkpoint 里就是原始 mxfp4（`[E,2I,H/2]` uint8 + e8m0 块缩放）。
- `transform_weights_for_mega_moe` 是**纯布局重排**（MegaMoE 融合核的 ABI），不改变数值；
  官方在**加载时做一次**并常驻。mini 的真权重没有 `share_across_layers` 标记 → 退化成
  "每层每步重做"（0.86 GB/层 ≈ 8 ms）→ decode 一步白烧 336 ms。想按层缓存 = +34 GB/卡 → OOM
  （已实测，`DS41M_PREP_CACHE_N=0` 默认关）。
- MegaMoE 的成本由**固定容量的对称缓冲**决定，T=1 也按 chunk 跑满 → 5.8 ms/层；
  prefill 的大 T 摊薄后免费，decode 就是纯亏。官方 decode 用另一套**按真实 token 数计费**的
  融合算子（dispatch_v2 + GMM + combine_v2），所以 8 卡 GBS=1 能到 TPOT 2.48 ms。

### 7.2 官方 decode MoE 的三个 ABI 坑（2026-10-17 逐个实测）

| # | 现象 | 真因 / 修法 |
|---|---|---|
| 1 | `aclnnGroupedMatmulWeightNz failed 161002`，`x k=5120 vs weight k=9216` | NZ cast 必须对**未转置**的 `[E,2I,H/2]` 做（官方 `process_weights_after_loading`），调用处再 `.transpose(1,2)` 视图；我多转置了一次 |
| 2 | `161002` + `EZ1001: Only int64 is supported for groupList` | `group_list` 直接传 dispatch 返回的 **int64** `expert_tokens`，不能 `.to(int32)` |
| 3 | dispatch 回来 `expand_x` 48 行看着像没分片 | **正常**：48 = `global_bs(8) × min(e_local(48), topk(6))` 是**容量**（meta 公式），不是实际行数 |

### 7.3 平台口径：本机是 Ascend 950DT（`Ascend950DT_9582`）

官方在 950 上对 MC2 通信域有**专门配置**（`moe_ep_group_mc2`）：
- `group_type=3`（`hccl_op_expansion_mode=aiv`），`is_full_mesh_v2=False`；
- **不设** `comm_alg`（只有非 950 才用 `"fullmesh_v2"`）；buffer = 200 MB（默认值即够）；
- 用 **world 组的 comm name** 直接跑 dispatch 在 950 上会"收到 0 行"（`expert_tokens` 全 0）。
- mini 实现：`exec_moe._mc2_group()` —— `dist.new_group(全部 ranks, pg_options=Options(hccl_config=...))`，
  懒建（首个 decode 步、各 rank 同步进入）。

### 7.4 共享专家的官方口径（`forward_a8float8`）

`gate_up_proj → swiglu_group_quant(clamp_limit=10, round_scale=True, quant_mode=1) → down_proj`。
**clamp 在 silu 之前**（A=Min(A,limit)、B=Clamp(B,±limit)），不是 silu 之后。

### 7.5 ✅ 打通结果（2026-10-17 20:00）

| 项 | 结果 |
|---|---|
| 16k 首 token | **id=396 ' that'**（top1 32.5）—— 与官方参考一致 |
| 续写 | ' she' / ' could' / ' not'（连贯且与 MegaMoE 路径同 id） |
| 设备异常 | 0 |
| 步时 | 第 1 步 2.27 s（建 40 层 NZ 缓存）→ 稳态 **0.46 s/步** |
| 对比 | 旧路径（MegaMoE + 每步 transform）0.75 s/步；官方 MoE 无缓存时 0.82 s/步 |

**第 4 个 ABI 坑（新）**：`swiglu_group_quant` 必须用 `torch.ops.custom.npu_swiglu_group_quant`
（官方 low_latency_tp 走的就是它），**不能**用 `torch.ops.cann_ops_nn.swiglu_group_quant`：
本机 CANN 9.2 的后者 Python schema 是 `int dst_type`，而其 aclnn 内核把 attr 收成 **0**
→ `attr dst_type only support (...), got 0`（传 torch.dtype / 292 / 24 都一样，wrapper↔kernel 版本不匹配）。

**NZ 缓存（`exec_moe._DEC_W`）**：按层缓存 decode-GMM 布局的权重（EP8 下 ~850 MB/层），
建完立即 `RealSource.drop_layer_experts()` 释放原始专家权重 —— **同尺寸、显存净增 0**；
`iter_layer` 不再向盘上重取（`weights.RealSource.dropped`）。`DS41M_DEC_RELEASE_RAW=0` 可关。
首个 decode 步还会放掉 prefill 最后一层的 MegaMoE transform 缓存。

### 7.6 稳态 0.39 s/步的归因（ALOP 时间轴 + 微基准，2026-10-17 20:20）

**时间轴分析**（`dec_prof1` trace，主流单流串行）：kernel 合计 **288 ms/步**（利用率 75%），
kernel 之间的空档 **~74 ms/步**（155 个 >0.5 ms 的 gap，最大 5 个 6.5–11.4 ms，卡在
`RmsNorm→InplacePartialRotaryMul` 一类依赖点）。

| 项 | 每步 | 说明 |
|---|---:|---|
| `MoeDistributeDispatchV3` | 143 ms | 40 层 × **3.57 ms**（T=1！）|
| `MoeDistributeCombineV3` | 44 ms | 40 层 × 1.09 ms |
| 其余所有设备算子 | ~100 ms | TransData ~4.8 ms/步 + QMM/QBM/attention… |
| kernel 间空档 | ~74 ms | 主机/依赖导致的流停顿 |

**判据性微基准**（单卡）：
- 普通算子下发 **12–16 µs/op**（不是主机瓶颈；`empty` 2.2 µs、`copy_` 5.6 µs）。
- `npu_moe_init_routing_v2`（384 专家，T=1）**24 µs** —— TP 路径的本地路由几乎免费。
- dispatch/combine 在 `ep_world_size=1` 下**跑不了**（算子要求 EP≥2），无法单独测固有成本。
- **world 组 vs 专属 MC2 域 vs AIV/全网格**：4 层小模型稳态步时**全部 0.06 s**，输出一致
  ⇒ MC2 域配置不影响速度/数值；`DS41M_MC2_USE_WORLD=1` 可省掉建域的 ~1 s。

**低时延 TP 路径（`DS41M_MOE_TP=1`，官方 `moe_infer_tp`）现状**：
- 已实现：`init_routing_v2 → GMM×2(+swiglu) → finalize_routing → all_reduce`；权重按
  **intermediate 维切 1/8**（每 rank 全 384 专家 × 288），显存与 EP8 相同（~34 GB）。
- **TP 16k/40 层 prefill 卡死已定位并修复**：`swiglu_group_quant` 要求输入末维能被 **256** 整除，
  2304/8=288 → 576 ✗。官方用**非均匀 128 对齐切分**（`split_intermediate_aligned`：
  rank0–1 各 384、rank2–7 各 256），已照抄到 `weights._tp_slice_expert`。修后 prefill 正常
  （`[moe/tp] 第 1 次调用 OK，T=16384`）。
- **TP decode 仍不可用**：第 1 个 decode 步正常出 token，**第 2 步 MoE 入口 NaN**
  （`moe_topk_gate_forward input contains nan or inf` → AIV 断言 → HCCL watchdog）。
  **与 engram 无关**（`DS41M_NO_ENGRAM=1` 复现）；同一 2 层模型 EP 路径 4 token 全通、0 异常
  （稳态 **0.03 s/步**）。⇒ 限定在 TP 的跨步状态（all_reduce / 分片 GMM / 池写入）交互，
  需要设备级逐层 bisect。
- 稀疏 grouplist（`expert_tokens_num_type=2` + `group_list_type=2`）已接，
  但官方 8 卡低时延 YAML 里 `enable_grouplist_type2: False` —— 不是它慢的原因。

**结论**：EP 路径每步的硬成本里，dispatch+combine（187 ms）只能靠 TP 消掉；其余
~100 ms 设备时间 + 74 ms 空档要靠 **图捕获（npugraph）+ 多流** 才能压下去。
10 ms/token 需要「TP decode + 图捕获 + DSpark」三者叠加（官方 2.48 ms 是 DSpark-7 摊薄后的）。

### 7.7 ✅ 全 TP 打通并快于 EP（2026-10-17 23:20）

| 路径 | prefill 40L/16k | decode 稳态 | 首 token |
|---|---:|---:|---|
| EP/MegaMoE + EP decode | 224 s | 0.39 s/步 | 396 ✓ |
| **全 TP（TP8，prefill+decode 同一套）** | **119 s** | **0.23 s/步** | **396 ✓**（35.5，bf16 all_reduce 舍入） |
| EP prefill + PD 换形态 | 224 s | 0.20 s/步（v2 假阳性；v3 待验） | 396 ✓ |

关键修正：
- **"TP prefill 27 s/层"是假象** —— 当时 prefill 还在逐层缓存 NZ 权重并逐层 `empty_cache()`
  （22:31 已改为 prefill 不缓存）。实测 TP prefill 约 **3 s/层**，比 MegaMoE（5.6 s/层）还快。
- 微基准佐证：`init_routing_v2`（384 专家，T=16384）**0.40 ms**、384 组密集 GMM（98304 行）**0.76 ms**/层。
- 稀疏 grouplist（`expert_tokens_num_type=2`）在本机 decode 第 2 步会 NaN → **默认改回密集 type=1**。
- 全 TP 的代价：**首个 decode 步 26 s**（384 专家 × 40 层建 NZ 缓存，一次性）；
  之后稳态 0.23 s/步。若要更快，可在 prefill 期间预热 NZ 缓存。
- 用法：`DS41M_MOE_TP=1`（prefill/decode 都 TP）；`DS41M_TP_GL` 默认 1。

### 7.8 ✅ decode 全 40 层入图（2026-10-20）

`DS41M_DECODE_GRAPH=1` 用 `torch.npu.NPUGraph` **eager capture**（不做 torch.compile /
npugraph_ex；mini 是函数式执行器，不需要 dynamo/Meta）把 T=1 增量 decode 的整段
层 0..39 录成一张图，跨 token 复用。关键点：

1. **pos 元数据固定缓冲**（`exec_ops.dg_*`）：窗口/压缩 slot、rope cos/sin、QLI/QSLI 的
   `seqused_k`/residual/cache_index/qsli slot/`candidate_block_length` 等全部预分配，
   replay 前 `dg_update(pos)` 原地刷新——图里录的是地址，不是值。
2. **QSLI 几何冻结**：`nblk2`/`bt_w` 按 `L_full` 一次开满，图里形状不变。
3. **engram 图外预计算**：层 1/14 的 `engram_kv`（host 取表 + H2D + 解量化）在图前算好，
   图里只跑 wkv/gate；否则捕获期 H2D → `aclrtMemcpy 107030`。
4. **首两个 forward 必须 eager**：尾段首次出现的常量张量（如 QSLI `cu_seqlens`）
   要在捕获前建好；`_DG_RUN["steps"] >= 2` 才允许捕获。
5. `_qli_metadata` 在 DG 下不跨步缓存（metadata 编码 seq_k）。

真卡（tp8-fidelity，8×950DT，真权重，prompt 17 token，decode 20）：

| cut（入图层） | 稳态 | 前缀 | replay |
|---|---:|---:|---:|
| 21 | 0.114 s/tok | — | — |
| 15 | 0.089 s/tok | 73 ms | 9.8 ms |
| 2（含 engram 预计算） | 0.060–0.136 s/tok | 49–124 ms | 12.1 ms |
| **0（全 40 层）** | **0.024–0.026 s/tok** | **5.0–5.7 ms** | **12.6–13.7 ms** |

`DS41M_DECODE_GRAPH=check`（只建缓冲、不捕获）与图模式 20 token 逐 token 相同
（含 pos 29 的 id=71485）⇒ replay 逐位确定。同路径 eager 207.6 ms/tok →
图 **25 ms** = **8.3×**（早前 tp8 eager 250 ms 口径约 10×）。
剩余 replay 13 ms 主要是 TP MoE/attention 设备时间；10 ms/token 还需 DSpark-3
（权重已有 `mtp.0/1/2`，mini 尚无运行时）。详见
`NOTES/2026-10-20-decode-graph.md`（共享盘）。

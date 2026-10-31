# DeepSeek V4.1 Flash：Ascend 950 Prefill 三仓源码对照

分析日期：2026-10-31，Asia/Shanghai。

**结论：Hirrorz/benchmark 是一个混合实现。当前默认 indexer 已换成 DeepInfer 风格的 DeepGEMM + DeepSelect，主 attention 仍走 CANN 的量化 paged MQSMLA；CP、裁尾、metadata 生命周期也没有与任一对比仓完全一致。因此它适合研究某个固定配置的计算路径和性能变化，不能仅凭“算子名/次数接近”就认定能代表另外两仓的 Prefill 时延或真实缓存收益。**

本文读取了三个仓库的真实执行代码，区分当前分支、可选分支、旧注释。没有运行 NPU、加载权重或采集 msprof，文中的性能影响是由代码和数据规模推导；仓库注释中的历史实测数据不作为本次实测。源码版本和文件 SHA256 见同目录 `source-manifest.json`。

## 0. 基线与共同模型结构

| 仓库 | 固定 commit | 本次主要入口 |
|---|---|---|
| Hirrorz/benchmark | `eac678f460fe1e23c9fd12e397e93cdbc759be09` | `core/points_runner.py` → `core/runner.py` → `recipes.yaml` → `ops/exec_*.py` |
| deepseek-ai/DeepInfer | `553741557a57df44a46fb3ab72b6aafeaf945337` | `worker/model_runner.py` → `models/deepseek.py` → `decoder_layer.py` → `v4_modules/` |
| cann/cann-recipes-infer | `33b6f65b8ec6f0e092ce0f4fb3b514d7ca04a424` | `models/deepseek_v4_1/`；采用其中 950 EP8 / EP8+CP8 配置对照 |

DeepInfer 已有本地 checkout，本次只读；另两仓新建只读研究副本。DeepInfer 指向 `deepseek-ai/DeepInfer`，已核对远端 HEAD 与本地提交一致。

以下层号均从 **0** 开始：

| 层号 | 类型 | 全局 main KV / index-K | query 对应的 index/top-k |
|---|---|---|---|
| 0–1 | SWA | 没有全局压缩分支 | 无 indexer |
| 2–7 | SWA + C2 | 第 2 层生成，组内共享 | 第 2 层生成后共享 |
| 8–13 | SWA + C2 | 第 8 层生成后共享 | 第 8 层生成后共享 |
| 14–19 | SWA + C2 | 第 14 层生成后共享 | 第 14 层生成后共享 |
| 20–23 | SWA + C1 | 第 20 层生成，后 20 层共享 | 第 20 层生成 |
| 24–27 / 28–31 / 32–35 / 36–39 | SWA + C1 | 继续读第 20 层 | 各组首层重新评分，消费第 20 层的候选块 |

共同关键规模是 hidden=5120、主 attention 64×512、indexer 32×128、SWA=128、main top-k=512、候选块最多 2048×8、384 routed experts / top-6 / 1 shared expert；Engram 在第 1、14 层。C2 是 2:1 压缩，C1 不缩短长度。

DeepInfer 的参考模型是已有配置 `DeepSeek-V4.1-Flash-1017`；CANN 使用转换后的 checkpoint 路径，benchmark 可使用合成权重。模型结构接近，不代表权重 revision、量化产物和依赖版本已经相同。

固定主仓 commit 也没有自动固定机器上安装的二进制算子包。benchmark 使用 `deep_gemm_di`、CANN/custom operator 等外部运行时；本次未核验实际 950 环境里的 `.so`、编译参数和运行开关。下文“采用 DeepInfer 风格内核”描述的是接口和调用链，不能据此认定机器上装载的二进制与本次 DeepInfer commit 完全相同。CP 部分讨论公开代码支持的方案；DeepInfer 部署 Q&A 另称当前线上部署暂不使用 TP/CP，不能混为默认线上配置。

来源：[benchmark 配置](https://github.com/Hirrorz/benchmark/blob/eac678f460fe1e23c9fd12e397e93cdbc759be09/configs/mini_ds41_flash.yaml#L1)、[DeepInfer HF 配置适配](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/hugging_face/config.py)、[CANN EP8+CP8 配置](https://gitcode.com/cann/cann-recipes-infer/blob/33b6f65b8ec6f0e092ce0f4fb3b514d7ca04a424/models/deepseek_v4_1/config/deepseek_v4_1_flash_rank_8_8ep_8cp.yaml)。

## 1. Prefill 使用的算子

### 1.1 主 attention 与 indexer 并非同一套实现

| 部位 | benchmark 当前默认 | DeepInfer / 950 | CANN recipes / 950 |
|---|---|---|---|
| Q_A 与 SWA KV | 配方分别调用 `wkv`、`wq_a`，各自 quant matmul | 同输入时合并 `Q_A + SWA_KV` 投影 | `wq_a`、`wkv` 分开 |
| Q_B 与 index Q | 主 Q_B 与 `idx_prolog` 分开 | 有本地 indexer 时合并 Q_B / index Q_B 输出，再拆分 | 主 Q_B + indexer QW prologue |
| norm / RoPE / 量化 / store | CANN/custom 算子链，按配方执行 | `fused_norm_rope_hadamard_cast_store_kvcache` 等融合路径 | CANN RMSNorm、rotary、compress epilogue、index prologue |
| main attention | `mixed_quant_sparse_flash_mla`，paged 原始 KV / 压缩 KV 分别传入 | 将历史 SWA、当前 SWA、main KV 组装为连续 BF16，再调 `flash_mla_sparse_fwd` | `mixed_quant_sparse_flash_mla`，量化 paged KV |
| 源层 indexer | 默认 `DS41M_IDX_KERNEL=dg`：DeepGEMM dense MQA logits → DeepSelect top-k | 同类 DeepGEMM logits → DeepSelect | `quant_lightning_indexer`，融合评分、top-k、候选输出 |
| 候选消费层 indexer | 默认 `DS41M_QSLI_KERNEL=dg`：sparse schedule → sparse MQA → DeepSelect → index restore | 同类稀疏打分链 | `quant_sparse_lightning_indexer`，读取组合 K+scale cache |
| O 的分组投影 | 默认尝试 `deep_gemm.fp8_einsum`，打包/调用失败回退 transpose quant batchmatmul | `deep_gemm.fp8_einsum` | `npu_transpose_quant_batchmatmul` 等 CANN 路径 |
| MoE | 主配置默认 `deepgemm` MegaMoE；也可选 AscendC / unfused | 满足 FP8×FP4 与 EP 域条件时走 MegaMoE | 提供 AscendC / DeepGEMM 后端，所读 EP8 YAML 默认 AscendC |

**关键事实：benchmark 源码中“默认 vendor”的部分注释已过期。** `_qli()` 实际环境变量默认值是 `dg`，`_qsli_pk()` 也是 `dg`。若要复现 CANN 融合 indexer，需显式设置相应 vendor 开关，同时确认部署的算子包版本。

DeepInfer 的 Prefill 读取 FP4 存储的 main KV 时会解量化成 BF16 工作区。CANN/benchmark 的 MQSMLA 直接接收打包的量化 paged cache。两者不仅是 kernel 名字不同，**临时工作集、额外 load/dequant、索引拼接、HBM 访问路径都不同**；仅比较 FlashMLA/MQSMLA 本体时间会漏账。

950 平台还要避免误套 NVIDIA 优化结论：DeepInfer 的 `use_mega_mhc`、mega gate，以及将 norm/RoPE/attention/输出 cast 整体融合的条件含 `PLATFORM.is_gpu`，950 不走这些 GPU 专用分支。950 的 MegaMoE、部分 norm/RoPE/store 融合仍可用。

来源：[benchmark 配方](https://github.com/Hirrorz/benchmark/blob/eac678f460fe1e23c9fd12e397e93cdbc759be09/recipes.yaml)、[benchmark indexer 分派](https://github.com/Hirrorz/benchmark/blob/eac678f460fe1e23c9fd12e397e93cdbc759be09/ops/exec_indexer.py#L666)、[DeepInfer attention 后端](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/attention_backend.py#L26)、[DeepInfer 投影/缓存链](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/attn.py#L386)、[CANN indexer](https://gitcode.com/cann/cann-recipes-infer/blob/33b6f65b8ec6f0e092ce0f4fb3b514d7ca04a424/models/deepseek_v4_1/models/modules/indexer.py#L338)。

### 1.2 即便换成 DeepInfer indexer，外层图仍有差异

- benchmark 默认 QSLI 的 dense K 优先来自和 QLI 相同的 split index-K 池；CANN QSLI 则读专用组合缓存。benchmark 还保留对组合池的要求和准备代码，不能假设换了打分 kernel 后这些开销都已消失。
- benchmark `_qsli_dg_chunk()` 明确拒绝 `nseq>1`。默认 DG 并不覆盖所有 packed 多请求点位；这些点位必须显式选 vendor，或者扩展实现后再测。
- DeepInfer 在一个循环中按 query chunk 完成 logits → top-k，并立即释放 `p_sum`。benchmark 的 dense DG 分支在 `T>chunk` 时仍先分配整块 `[T,maxK]`，再把各 chunk 的 logits 填进去。QSLI 分支还在分块判断前计算整段 sparse logits，大 T 时随后又按 chunk 计算。**分块调用不等于与 DeepInfer 相同的流式工作集和调用次数。** compact 小 T 不一定触发此问题，长 Prefill / replay 关闭时尤其需要检查。

来源：[benchmark QSLI 限制和 split K](https://github.com/Hirrorz/benchmark/blob/eac678f460fe1e23c9fd12e397e93cdbc759be09/ops/exec_indexer.py#L1605)、[benchmark dense chunk](https://github.com/Hirrorz/benchmark/blob/eac678f460fe1e23c9fd12e397e93cdbc759be09/ops/exec_indexer.py#L2781)、[benchmark sparse chunk](https://github.com/Hirrorz/benchmark/blob/eac678f460fe1e23c9fd12e397e93cdbc759be09/ops/exec_indexer.py#L1722)、[DeepInfer chunk 生命周期](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/index_attn.py#L327)。

## 2. Prefill metadata 如何处理

需要把 metadata 分成三类：请求布局和 cache 地址（cu、positions、slots、block table）；kernel 的 tiling/schedule；模型内容产生的 top-k/candidates。三类不能用同样的缓存失效规则。

| 实现 | 构造与消费 | 生命周期 |
|---|---|---|
| DeepInfer | C++ `prepare_prompt_v4` 生成 pinned metadata、load plan 和 SWA pivots；device 展开索引；Ascend `prepare_cp_mirror` 联合生成 mirror 与 ks/ke；consumer view 切换 compact 几何 | 每次实际请求/调度步生成；展开结果在本步对象内复用；候选 schedule 在 `CrossLayerRuntime` 候选组内共享 |
| CANN | `AttnMetaData.build_attn_metadata` 构造 tensor 字典；`generate_kernel_metadata` 为 win/C2/C1 以及 QLI/C1候选/QSLI 生成 kernel metadata | forward 入口每次生成，同类层共享；CP encoder 的 prev/next 两段分别生成；CED tail 另有一套 |
| benchmark | dataclass metadata 管逻辑布局，`_meta` 缓存设备张量和 MQSMLA/vendor metadata；DG 候选组另走 `_meta_fwd` | **同时存在跨 forward 几何缓存和单 forward 缓存**，并非全部每 forward 重建 |

**CANN 的特点是提前生成并用 side stream 隐藏开销。** 它分别记录输入就绪、attention metadata 就绪、indexer metadata 就绪事件；第 0 层 attention 不必等完整 indexer metadata，首个 indexer 再等对应事件。encoder 与 tail 事件分开，减少不必要的等待。若关闭多流，仍有提前生成，但少了这部分重叠。

**DeepInfer 的特点是把请求 metadata 和内容相关 schedule 放进明确的 step / runtime 对象。** dense indexer 直接消费 ks/ke；候选 indexer 按 chunk 为候选组生成 sparse schedule，多个消费层复用；换 candidate source 或 compact 布局时清理派生结果。它不需要照搬 CANN 的六类 metadata 算子。

**benchmark 的缓存更激进。** `runner.py:1078–1086` 只在几何变化时调用 `meta_clear()`；`exec_ops.py:3061` 的 MQSMLA metadata 走 `_meta`，相同点位预热后可跨多次 forward 复用。另一方面，DG 的候选清洗、有效长度、sparse schedule 走 `_meta_fwd`，以 `PASS_NO` 为 epoch，每次新 forward 失效。

因此不能把提交说明里的“metadata 每 forward”理解为全部元数据都每次构造。**对固定几何的性能 microbenchmark，这种缓存可以成立；与每次处理新布局的真实引擎比较时，必须单独报告 metadata 成本。** 只依赖长度/位置的结果可以按完整几何复用；依赖候选内容的 schedule 不能只按 shape 或地址跨步复用，地址相同不代表 tensor 内容未变。

来源：[DeepInfer C++ metadata](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/csrc/engine/input_metadata/prepare_tensor_input_v4.cpp)、[DeepInfer sparse schedule](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/index_attn.py#L245)、[CANN metadata stream](https://gitcode.com/cann/cann-recipes-infer/blob/33b6f65b8ec6f0e092ce0f4fb3b514d7ca04a424/models/deepseek_v4_1/models/modeling_deepseek.py#L3155)、[benchmark 几何缓存失效](https://github.com/Hirrorz/benchmark/blob/eac678f460fe1e23c9fd12e397e93cdbc759be09/core/runner.py#L1078)、[benchmark 两种缓存](https://github.com/Hirrorz/benchmark/blob/eac678f460fe1e23c9fd12e397e93cdbc759be09/ops/exec_ops.py#L3269)。

## 3. CP 方案的实现

### 3.1 共性：mirror / zigzag 的 query 并行

三者都是把请求切成 `2C` 段，rank r 取第 r 段与第 `2C−1−r` 段，平衡因果位置。它们不是把 attention heads 当作 CP 拆分，也不是跨 rank 合并 ring attention 的 softmax/LSE：**每个 rank 拿本地 Q，补齐所需 K 后独立完成 attention。** MoE 继续走 EP。

例如 L=16、CP4：rank0 拿 `[0,1,14,15]`，rank1 拿 `[2,3,12,13]`，其余相同配对。对 C1 dense causal 扫描，四 rank 的 key 可见数量总和相同；实际稀疏 kernel 时延还受 padding、候选和 tiling 影响。

### 3.2 真正的差别在通信对象、重复计算与边界布局

| 维度 | DeepInfer | CANN | benchmark |
|---|---|---|---|
| segment padding | 每请求 `ceil(Q/(2C))` | 除 ceil 外，V4.1 设置 segment 最小 128 | 基本沿用 DeepInfer 的 ceil 布局 |
| SWA 交换 | all-gather 当前 SWA KV，再 reverse gather | 交换每个 segment 的量化尾窗；本地段主体不必全部传输 | 当前路径 all-gather 本地 SWA KV，构造全量窗口池，并可 side stream 预发 |
| main KV source | 本地计算 compressor projected input → AG → 压缩/写池；与 Q 投影重叠 | 各段本地压缩，处理跨段 remainder；量化后 AG main KV 行（512维时 320B/行） | **plain 默认**仍可走 full hidden AG → 每 rank 全量 compressor；**命中且 offset>0 默认**走 segment compression + 量化行 AG |
| index-K | 和 main pre-RoPE latent 共享计算来源，按 source segment load 复用 | 本地 quantize K+scale，合成约 68B/行载荷 AG，scatter 后 repack QSLI cache | 已有 68B 行 AG 路径，但依赖 segment latent、且 ratio>1；仍有 fallback / 全量 latent 过渡分支 |
| CP 通信后端 | DeepEP-Ascend BucketBuffer / userbuffer wait handle | `torch.distributed` CP group / HCCL | `torch.distributed` / HCCL，固定缓冲、chunk AG、异步 side stream |
| indexer 调用 | 一组 mirror 本地 Q，以逐行 ks/ke 表示非连续因果域 | prev/next 分开调 QLI/QSLI | DG 路径也仍逐 head/tail 块调用 |
| main attention 调用 | 本地 mirror Q 一次或 kernel 内部切分 | encoder prev/next 各一次 MQSMLA | 本地 mirror Q 配逐行 window index，通常一次 MQSMLA |
| decoder tail | compact 后重新 mirror 分配，继续 CP | tail gather 到每个 rank，**冗余 replay** | 重新 compact-mirror 分配，继续 CP |

benchmark 的 `_kv_compress` 分派要特别注意：`DS41M_CP_CMP_FLOW` 默认 `seg`，但 `_seg_ok` 默认要求 `pos_offset>0`。plain prefill 未满足时，不会自动使用 segment 分支；可选 `DS41M_CP_CMP_KERNEL=deepinfer` 才进入本地投影后 gather 的 DeepInfer compressor 分支。不能根据函数里的“DeepInfer 同款”注释认定默认走的就是该路径。

命中 segment 分支目前还可先 gather BF16 latent，再在后续 index-K 路径中做量化行 gather；需要按 trace 检查实际冗余载荷。整个 CP+hit 路径要求显式 `DS41M_CP_HIT=1`，并限制单请求；CP 裁尾也明确限制单序列。

单请求、最终 tail=140 时，DeepInfer/benchmark 的尾层每 rank 分配行数是 `2ceil(140/(2C))`（CP1 直接 140），例如 CP8 为 18，含 padding；CANN 对同一请求则每 rank replay 128。**CANN 此处承担重复工作，但每个小矩阵的 M 更大；另外两者总工作更少，却更容易暴露小矩阵和 host launch 瓶颈。** 不能据重复计算量直接预测哪一个更快。

来源：[DeepInfer CP attention](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/attn.py#L782)、[CANN main KV 量化后通信](https://gitcode.com/cann/cann-recipes-infer/blob/33b6f65b8ec6f0e092ce0f4fb3b514d7ca04a424/models/deepseek_v4_1/models/modules/compressor.py#L191)、[CANN index-K 通信](https://gitcode.com/cann/cann-recipes-infer/blob/33b6f65b8ec6f0e092ce0f4fb3b514d7ca04a424/models/deepseek_v4_1/models/modules/indexer.py#L141)、[CANN 冗余 tail 的明确语义](https://gitcode.com/cann/cann-recipes-infer/blob/33b6f65b8ec6f0e092ce0f4fb3b514d7ca04a424/models/deepseek_v4_1/models/modules/attention_data.py#L714)、[benchmark compressor 分派](https://github.com/Hirrorz/benchmark/blob/eac678f460fe1e23c9fd12e397e93cdbc759be09/ops/exec_ops.py#L2037)、[benchmark index-K CP](https://github.com/Hirrorz/benchmark/blob/eac678f460fe1e23c9fd12e397e93cdbc759be09/ops/exec_indexer.py#L916)。

## 4. SWA bounded replay 的实现

### 4.1 两种 replay 必须分开

1. **Encoder prefix replay**：main/index KV 已命中，但 encoder SWA state 没命中时，重算命中前缀末端的一小段以恢复 SWA；重算行不能覆盖已命中的 main/index KV。
2. **Decoder tail replay**：利用 CED / YOCO 共享 global KV 的结构，后半层只处理最后一段 Q，恢复 decode 所需的 decoder SWA state；不需要把全 prompt 的后半层都跑完。

140 是 DeepInfer 当前推荐的 decoder KEEP 配置，约束是 `KEEP >= window + max_backtracking`。它**不是“每次 prefix 命中都必须在所有层额外输入 140 token”的通用规则**。公开部署说明中，encoder SWA miss 用的是一个 128-token block 的回退。

### 4.2 三条 decoder 路径

| 实现 | 边界 | full 分支 | compact 分支 |
|---|---|---|---|
| DeepInfer，显式开启 compact | 第 20 层入口 | 保留 full hidden / metadata，为 source 层生成整个待更新区间的 main KV 和 index-K | 第 20 层的 Q、indexer、attention、MoE 即可使用 compact 行；之后各层继续 compact |
| CANN，所读 YAML 开启 CED tail | 第 21 层入口 | 第 20 层整体正常执行，包括 full Q/attention/MoE | 第 21–39 层每请求只 replay 最多 128 行；CP 时在每 rank 复制 |
| benchmark，默认 boundary split | 第 20 层内部，KV / index-K 写完后 | full KV / index-K 路径保留 | 第 20 层余下路径保留最多 **512** 行；第 21 层再次裁到 **140**；CP 时发生两次布局重分配 |

benchmark 的 512 是为当前 QMM 小 M 数值/分块对齐设置的额外形状要求，不是模型要求的 SWA window，也不是 DeepInfer 的 KEEP。对大于 512 的单请求，它比 DeepInfer compact 多算第 20 层的一部分 Q/index/attention/MoE，并可能多一次 512→140 的搬运；但比 CANN 的整层 full 第 20 层又少算一些。多请求时 benchmark 的 boundary split 判定不成立，应另外看第 21 层的 packed 裁尾。

DeepInfer 的 compact 还支持 chunked prefill：保留的是最终 prompt 尾窗与当前 chunk 的交集；完全位于尾窗之前的 chunk 仍留一个占位 query。不能把每个中间 chunk 都建模为固定 140 个 Q。benchmark 当前固定点位的 full/suffix pass 没有复现同样的调度过程。

### 4.3 数值含义

bounded replay 的 SWA 可见域收窄为 `max(replay_start, position−127)` 到当前 position；尾段前部缺少段前 SWA 历史，其误差可向后传播。因此这是有界近似，**不能证明与不开 replay 的完整 40 层计算逐位等价**。

DeepInfer 的 mask↔compact 等价验证，是两种相同截断语义的执行布局比较。benchmark 的旧 full-path / 真权重 parity，也不能直接为当前 DG indexer、CP、hit 和 boundary split 的所有组合背书。CANN 固定 128 与 DeepInfer 140 的回退覆盖也不同，应针对 DSpark 接受/拒绝序列单独验收，不能只看首 token。

来源：[DeepInfer 边界拆分](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/decoder_layer.py#L134)、[DeepInfer compact CP view](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/input_metadata/yoco.py#L94)、[CANN 第 21 层裁尾](https://gitcode.com/cann/cann-recipes-infer/blob/33b6f65b8ec6f0e092ce0f4fb3b514d7ca04a424/models/deepseek_v4_1/models/modeling_deepseek.py#L2467)、[benchmark 512 边界](https://github.com/Hirrorz/benchmark/blob/eac678f460fe1e23c9fd12e397e93cdbc759be09/core/runner.py#L1483)、[DeepInfer replay 近似与缓存策略说明](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md#L181)。

## 5. 后半层 token 少，host bound 如何解决

**当前三个主路径都仍然是 eager Prefill，不能说“通过 ACL/NPU Graph replay 解决了”。**

- DeepInfer：`run_cuda_graph = not is_prompt ...`；prompt 不走 graph。
- CANN：`model_worker.py` 仅在 `not is_prefill` 时调用 compiled model。YAML 写 `npugraph_ex` 不代表 Prefill 已入图；所读配置 `enable_superkernel=False`。
- benchmark：`Policy.graph` 恒 False，当前 runner 注释明确 compact 段和第 21–39 层全程 eager。README / CODE_MAP 中旧 graph-tail 说明不能作为当前行为证据。

尾层每层工作量小，但 kernel 数和 Python/C++ 调用数没有同比下降。直观模型是每层时间受 `max(host 提交时间, device 执行时间)` 加必要依赖等待限制；算术只是瓶颈判断思路，不是可直接代入的实测公式。

| 实现 | 已有的缓解机制 | 仍然存在的限制 |
|---|---|---|
| DeepInfer | C++ 请求 metadata；融合 Q/KV 和 Q/index-Q 投影；norm/RoPE/store 融合；MegaMoE；同候选组 schedule 复用；`forbid_cuda_sync()` 约束模型 launch；异步调度把 CPU 调度/后处理与设备执行重叠 | 异步调度减少跨步空泡，不能消除单次 forward 中的小 kernel 下发成本；950 不能计入 GPU-only MegaMHC/mega gate 收益 |
| CANN | metadata 提前在 side stream 构造，分阶段 event；indexer prologue、QLI/QSLI/MQSMLA、MegaMoE 等融合；Engram 预取/多流 | Prefill Python/eager 调用仍在；tail 128 仍有小 M；CP tail 冗余计算是一项取舍，不能称为已经证明最优的 host-bound 方案 |
| benchmark | 几何张量/ctx/配方/权重打包缓存；移走重复 env/import；去掉多处 `.item()` 和阻塞 H2D；cannbotdsl ABI 契约缓存；默认尝试 fp8 einsum；DG 候选组 schedule 复用；AG 预发；连续提交多次 pass | 仍有 Python 算子分派、厂商 wrapper 和小 kernel；CP 切尾代码仍会创建索引，部分多请求 replay 有 D2H；有 runahead event 等待；并非所有薄调用原型默认启用 |

benchmark 的 `mq_fastcall`、L2、绑定缓冲等是可选原型，不能看见文件就算作默认优化。`DS41M_DRAIN_PER_FORWARD` 的可执行默认值为 0，旧注释中的“默认开”也不准确。默认 CP 有 K=2 的 runahead event 环；非 CP 会结合 token 阈值选择是否启用。这些 event 等待会影响真实时间线。

多点位默认 pure-pass 模式最终用 **总墙钟 / forward 数**，测量末尾同步也包含在墙钟区间内，因此不是简单“只统计 kernel 时间”。但这仍是预热后固定输入、固定布局、连续提交的稳态计算吞吐；不能直接当作包含 tokenization、缓存拉取、排队、调度和采样的单请求 TTFT。

要确认是否已经缓解 host bound，需要在同一个点位报告：后半层 CPU launch span、设备 kernel 时间/idle gap、是否出现同步、端到端 forward 墙钟；不能仅用 kernel duration 缩短或总算子数减少判断。当前源码只能说明做了哪些缓解，不能证明三者的尾层已完全摆脱 host bound。

来源：[DeepInfer launch guard](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/worker/model_runner.py#L312)、[CANN Prefill eager 判定](https://gitcode.com/cann/cann-recipes-infer/blob/33b6f65b8ec6f0e092ce0f4fb3b514d7ca04a424/executor/core/model_worker/model_worker.py#L445)、[benchmark graph 关闭](https://github.com/Hirrorz/benchmark/blob/eac678f460fe1e23c9fd12e397e93cdbc759be09/meta/policy.py#L55)、[benchmark ABI 缓存](https://github.com/Hirrorz/benchmark/blob/eac678f460fe1e23c9fd12e397e93cdbc759be09/ops/fast_abi.py)、[benchmark pure-pass 计时](https://github.com/Hirrorz/benchmark/blob/eac678f460fe1e23c9fd12e397e93cdbc759be09/core/points_runner.py#L551)。

## 6. 缓存命中率构造是否合理

**作为“给定长度和后缀行数的合成计算测试”有价值；作为真实命中率基准，目前不充分。** 主要问题有五类。

### 6.1 逻辑命中率与实际计算行数要同时报告

benchmark 的构造是：

```text
M = round((1−h) × L)       # 或显式 non_cached_tokens
W = 140                   # 当前 replay_tail 最小值
R = M + W                 # 所有前半层实际输入行数
H = L − M                 # real stage A 的命中前缀长度
stage B 起点 = L − R = H − W
```

`H` 和 `H−W` 不是同一概念：前者是已缓存的逻辑前缀，后者是额外 replay 后的计算起点。不能把额外重算行也统计成业务新增 token。对合法构造，前半层计算量相对只算 M 的额外比例是 `W/M`，而计算被跳过的比例是 `1−R/L`，并不等于名义 h。

| 点位 | 业务新增 M | benchmark 实算 R | 额外 replay / M |
|---|---:|---:|---:|
| 1K，99% | 10 | 150 | 1400% |
| 64K，97% | 1966 | 2106 | 7.12% |
| 1M，99% | 10486 | 10626 | 1.34% |

因此短序列高命中时，测到的主要可能是 replay 和固定 launch 成本，而不是 M 个新增 token 的成本。还要区分 tokens/s 分母是 M、R，还是整个 L；三种数不能混用。

### 6.2 无条件 +140 与 DeepInfer benchmark / 生产缓存策略不同

DeepInfer `examples/benchmark.py` 的 `cached_tokens = context_length−non_cached_tokens`，先真实计算缓存前缀，再 fork 复用 KV/state，scheduler 处理剩余后缀；其计数并没有通用的 `M+140` 规则。

生产部署文档另行描述：block KV 按 128-token 粒度命中；SWA state 命中可直接复用，没命中才回退一个 block 做 encoder replay；YOCO 还要求给尾部留出窗口与回退余量。这个系统的完整跨请求缓存检索/存储并未全部开源，不能把离线 fork benchmark 等同于线上 cache-store。

benchmark 的固定 `+140` 可以描述一个特定“需要重建 prefix SWA”的实验设定，但应明确它在模拟什么，并同时设置 `swa_state_hit=true/false`、实际 `prefix_hit_tokens`、`replay_tokens`。**decoder KEEP 与 encoder prefix replay 应拆成独立参数。** 如果 SWA 已恢复，继续在前半层固定重放 140 行通常就不再是同一个测试问题。

CANN 这条离线 Prefill 入口还把 `start_pos` 初始化为 0；没有发现与 benchmark 的 hit-ratio sweep 相同的完整恢复流程。不能因为它有 block table 和 paged cache，就认为已覆盖任意前缀命中场景。

### 6.3 短序列低命中率存在明确边界问题

`t_main()` / 多点位 `rows` 当前没有统一限制 `R <= L`。例如：

```text
L=1024，h=0.10
M=round(921.6)=922
R=922+140=1062 > L
```

这不是原 prompt 的合法后缀长度。标量 offset 的 `max(0,L−R)` 只把起点截成 0，不能把 1062 行变回长度 1024 的请求。应让 encoder replay 不超过可重放的缓存前缀，例如 `E <= H`、`R=M+E<=L`，或者明确拒绝不满足所选缓存政策的点位；还要处理 0% 命中和 100% 附近的定义。

### 6.4 fake 零值前缀会改变实际 workload

多点位 `points_runner.py:399` 明确设置 `source="fake"`，默认不是先跑 stage A。main/index 前缀来自零值池，输入可为随机 hidden，合成权重可跨层共用；没有 ids 时还默认强均 MoE 路由。

这些设定有助于固定形状、减少初始化和隔离计算，但不能保证真实性能分布不变：

- 零值 index-K 会产生大量并列分数，可能让 top-k/candidate 集中，改变主 attention 的访存局部性；影响方向要实测，不能预先说一定更快。
- 强均 MoE 路由消除了真实专家倾斜；与真实 gate 或 EPLB 的开销与瓶颈不相同。
- 跨层共享权重、缩小 Engram 表、不同 Engram 存储位置改变工作集和数据搬运；DeepInfer 在 Ascend 默认强制设备存储 Engram，而 CANN/benchmark 配置可 offload。
- 连续重复一个固定点位会让 metadata 和内存驻留充分预热，不代表长短请求混合、cache eviction 或跨机器恢复的状态。

不能把 HBM/L2 的硬件命中率与这里的前缀 token cache hit rate 混为一谈；该项目控制的是后者的逻辑几何，未证明前者接近业务流量。

### 6.5 real stage A/B 是更好的正确性基线，但仍需独立验证

benchmark 有 real prefix 路径：stage A 真算到 `H=L−M`，stage B 使用同源输入从 `H−W` 重放到 L；重放区的 main/index KV slot 设为 −1，避免覆盖 stage A。这个设计可以用于验证“命中后缀是否正确读取历史”，但并不等于测到了真实缓存发现、异步 fetch、swap-in 或命中率分布。

应至少分开测四类：无缓存；main/index KV 与 encoder SWA 都命中；只有 main/index 命中、SWA miss；跨机/host KV 恢复。对每类分别记账 source-layer 的 main/index KV、encoder SWA state、decoder replay state，避免只用一个 hit 百分比覆盖所有状态。

来源：[benchmark 行数公式](https://github.com/Hirrorz/benchmark/blob/eac678f460fe1e23c9fd12e397e93cdbc759be09/core/assemble.py#L41)、[多点位强制 fake](https://github.com/Hirrorz/benchmark/blob/eac678f460fe1e23c9fd12e397e93cdbc759be09/core/points_runner.py#L339)、[real prefix 构造](https://github.com/Hirrorz/benchmark/blob/eac678f460fe1e23c9fd12e397e93cdbc759be09/core/runner.py#L4091)、[DeepInfer 真实 prefix + fork](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/examples/benchmark.py#L331)、[CANN Prefill 起点](https://gitcode.com/cann/cann-recipes-infer/blob/33b6f65b8ec6f0e092ce0f4fb3b514d7ca04a424/models/deepseek_v4_1/models/modules/attention_data.py#L558)。

## 7. 建议的公平比较方式与修改优先级

| 优先级 | 应先完成的工作 | 目的 |
|---|---|---|
| P0 | 拆开 `prefix_hit_tokens / encoder_replay_tokens / decoder_keep_tokens`；限制 `R<=L` | 修复命中定义混用和非法行数 |
| P0 | 固定并打印主 attention、QLI、QSLI、MoE、O projection、CP compression 的实际后端；默认路径不支持的 batch 直接拒绝 | 避免一张结果表混入不同算子图或未记录 fallback |
| P0 | 给当前 DG + CP + hit + split 建立真权重/真缓存的功能验收 | 旧 parity 不能覆盖新组合；含首 token、若干 decode、DSpark 拒绝回退 |
| P1 | 各测一次 metadata 预热复用与真实布局每步构造；记录 metadata/kernel/host idle 的分项 | 让性能对比的工作边界一致 |
| P1 | 统一第 20 层边界、KEEP、是否继续 CP、每 rank 有效/pad 行数 | 明确比较算法取舍还是比较实现效率 |
| P1 | 优先移除 plain CP 的 full hidden gather + 全量 compressor 复算，减少 latent 过渡通信 | 缩小与 DeepInfer/CANN CP 通信图的差异 |
| P1 | 将 DG indexer 的 logits→top-k 真正逐 chunk 流式执行，移除大 T sparse 预计算的重复调用 | 避免工作集或额外算子把比较放大 |
| P2 | 对尾层 CPU launch 做定点 profile，再决定融合、薄 ABI、固定 tail 图或更低 CP 度的收益 | 优化应依据尾层时间线；任何新图方案都属于后续改造 |

推荐结果表每行至少保留：`L、M、prefix_hit_tokens、encoder_replay_tokens、Q0_19、Q20、Q21_39、CP、DP、EP、每rank有效/pad行数、metadata模式、权重/路由/Engram模式、forward墙钟、尾层idle`。同时提供 40 层计算区间与完整 engine-step 区间；不把 forward throughput 标成 TTFT。

最终判断是：**benchmark 已有相当多针对 host 和 CP 的优化，但仍是 CANN attention + DeepInfer indexer/部分布局策略的混合实验图。最大的比较风险在实际工作量与测量边界不一致；最大的缓存基准风险在固定额外 140 行、fake 前缀和真实缓存状态未分离。**

本文没有修改三个实现仓库，也没有尝试执行 NPU 侧的代码。后续若依据报告调整 benchmark，应先锁定一个比较目标和一组真缓存验证点，再进行性能改造。

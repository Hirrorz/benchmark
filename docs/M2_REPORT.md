# M2 报告：单卡真跑 + msprof trace（40 层，T=1024）

命令（**TMPDIR 保持默认 /tmp**：指到 SFS 会让 CANN JIT 的 rmtree 报 `Directory not empty` ✗；
产物一律 `--out` 到共享盘）：

```bash
source /mnt/l00972766/ds41_bench/env_image.sh && source /mnt/l00972766/setup_ds41_env.sh
cd /mnt/xds/sfs/l00972766/ds41_mini
python3.11 runner.py configs/mini_single.yaml --seq 1024 --allow-placeholder \
    --prof --out /mnt/xds/sfs/l00972766/ds41_mini/prof/m2_single
```

产出 trace：`prof/m2_single/<host>_<pid>_<ts>_ascend_pt/`（含 `ASCEND_PROFILER_OUTPUT/kernel_details.csv`、
`analysis.db`、`ascend_pytorch_profiler.db` → **MindStudio 可直接打开**）。
最近一次（v4，含 mqsmla）：40 层 / **884 次算子调用 / 20 种算子 / 34 s**，**零错误、零占位输入**。

## 1. 真跑起来的算子（16 种，全部是真实 CANN 算子）

| 我们的 op | 内核 | 每层 | 调用的官方入口 |
|---|---|---|---|
| `rmsnorm` ×2 段 | RmsNorm | 2 | `torch_npu.npu_rms_norm`（attn_norm / ffn_norm） |
| `hc_pre` ×2 段 | HcPre | 2 | `torch.ops.custom.npu_hc_pre_v2` |
| `hc_post` ×2 段 | mhc_post_fwd_asc_kernel | 2 | `torch.ops.cann_ops_transformer.mhc_post` |
| `wq_a` | QuantBatchMatmulV3(+DynamicMxQuant) | 1 | `npu_dynamic_mx_quant` + `npu_quant_matmul` |
| `q_norm` | RmsNorm | 1 | `npu_rms_norm`（q_lora 1280） |
| `wq_b` | QuantBatchMatmulV3 | 1 | 同上 |
| `wkv` | QuantBatchMatmulV3 | 1 | 同上 |
| `kv_norm` | RmsNorm | 1 | `npu_rms_norm`（512） |
| `rope` | InplacePartialRotaryMul | 2 | `torch.ops.cann_ops_transformer.inplace_partial_rotary_mul` |
| `wo_a` | TransposeQuantBatchMatMul | 1 | `npu_transpose_quant_batchmatmul`（官方 permute 打包） |
| `wo_b` | QuantBatchMatmulV3 | 1 | `npu_quant_matmul` |
| `gate` | FusedMatMul | 1 | `npu_fused_matmul(fused_op_type="16cast32")` |
| `topk_gate` | moe_topk_gate_forward_kernel | 1 | `tile_kernels.moe.moe_topk_gate` |
| `mul`/`add`/`cast`/`tmove` | 元素级 | 1/1/3/2 | torch 原生 |
| `kv_epilog` | KvCompressEpilogV2 | 2（A/B 两块） | `torch.ops.custom.kv_compress_epilog_v2`（win 池 [blocks,128,1,544] fp8，block0 留空） |
| `kv_compress` | KvCompressEpilogV2 | 1（仅 kv_src 4 层） | 同上（cmp 池 [blocks,128,1,320] uint8，mxfp4） |
| `mqsmla` | MqsmlaKernel | 1 | `torch.ops.cann_ops_transformer.ds41.mixed_quant_sparse_flash_mla` + `..._metadata` |

`mqsmla` 的输入（按 ds41 版 schema，详见 `tools/probe_mqsmla.py`）：`q` [T,64,512] bf16 TND；
`ori_kv` = 全长临时 PA 池的 uint8 视图；`ori_sparse_indices` [T,1,128] int32 给**因果窗口内
128 个逻辑 token 号**（无效列置 -1 且**必须靠尾部** —— 放前面会让内核读到负位置直接 aicore 异常）；
`ori_topk_length = min(p+1,128)`；`metadata` [1024] int32 由 metadata 算子生成；ratio>0 层
按 ds41 硬校验必须给 cmp 三件套（本程序还没接 indexer，给"空索引"= 全 -1 + len 0，官方明确支持）。

**没跑起来的（Tier B，共 62 次跳过）**：`mega_moe`(40)、`idx_prolog`(8)、`qli`(4)、`qsli`(4)、
`engram_fetch/wait/proj`(各 2)。`mqsmla`/`kv_epilog`/`kv_compress` **已接上** ✓
算子本身在本机都在（`ds41.mixed_quant_sparse_flash_mla` 等已确认可加载），差的是**上下文**：
MQSMLA 要 sparse_indices/block_table/cu_seqlens/metadata，MegaMoE 要对称缓冲 + HCCL 组，
engram 要 ElasticBuffer + host 表。

## 2. 自恰性检查（本次最有价值的一步）

`tools/align_ops.py` 拿**我们的 trace** 当参考、**我们的模型**当预测，直接抓出三个真 bug：

1. `hc_pre`/`hc_post` 在配方重排后变成 4 次/层（配方 2 次 × count 2）→ 已改 count=1；
2. 重排 `common:` 时**漏了 `wo_b`**（QMM 变 3/层）→ 已补回；
3. `tmove` 模型 2/层 vs trace 0：`.contiguous()` 不产内核 —— **模型对（参考 trace 85=2×40+5），
   执行侧是已知差**，已写进 ops.py 注释。

修完再对参考 trace：**21 个建模算子比值全部 1.00** ✓

## 3. 已知差（执行侧 vs 模型，都记了原因）

| 现象 | 原因 |
|---|---|
| `rope` 我们 2/层、模型 3/层 | 第 3 次在压缩 KV 上，属 Tier B 的 compressor 路径 |
| `tmove`/`mul` 在内核里看不到 | `.contiguous()` / `×1.0` 被优化掉；模型按参考 trace 记 |
| `Cast` 我们 trace 313 > 模型 168 | `logits.float()` 等我们自己加的 dtype 转换也产 Cast |
| ~~`wo_a` 用占位输入~~ | ✅ 已消除：mqsmla 接上后 attention 输出是真的（数值自检 out_absmax≈10） |
| cmp 侧空索引 | indexer 未接 → ratio>0 层的压缩 KV 注意力分支 w=0（官方支持的配置；接了 indexer 才有真实 top-k） |

## 4. 下一步（按性价比排序）

1. ~~mqsmla 接上~~ ✅、~~kv_epilog/kv_compress~~ ✅；
2. `idx_prolog` / `qli` / `qsli`：indexer 上下文（indexer.wq_b/wk/weights_proj + candidate block），
   接上后 cmp 侧就有真实 top-k；
4. `mega_moe`：需 1-rank HCCL 组 + 对称缓冲（M3 的 EP8 里一并做）；
5. engram：M4（要 ElasticBuffer + host 表）。

## 5. M2.5：真权重（2026-10-14 打通）

**踩坑与根因**（定位链：逐算子同步 → 逐张量加载 318 个 → 打包 6 步二分）：
真权重的 E8M0 scale 从 safetensors 读出来是 `torch.float8_e8m0fnu`（合成权重是 uint8），
对该 dtype 做 transpose/`contiguous()` 会走 **AICPU fallback → 507018 aicpu exception**。
修法：`RealSource` 加载时统一 `.view(torch.uint8)`，喂算子时再 `.view(float8_e8m0fnu)`。

**实测**（`configs/mini_real.yaml`，EP8 取 rank0 那份，T=1024）：

| 规模 | 结果 |
|---|---|
| 3 层 | 23 种算子 / 72 次调用 / 24.9 s / 零错误 |
| **40 层全量（最终）** | **938 次调用 / 24 种算子 / 43.4 s / 零错误**（热态） |
| 40 层（含 engram 插件首次建缓冲） | 75.1 s / 23 种 |

执行覆盖（真权重下实测，24 种）：`hc_pre hc_post rmsnorm(×4) wq_a q_norm wq_b wkv kv_norm rope
kv_epilog mqsmla wo_a wo_b gate topk_gate mega_moe kv_compress(4) idx_prolog(8) qli(4)
engram_proj(2) + 元素级`。
**索引器 → 注意力链条已通**：`qli` 产出的真实 top-k（1023/1024 行有效）经
`st["cmp_sparse_indices"]/["cmp_topk_length"]` 喂给 mqsmla 的压缩 KV 侧 ✓。

未跑（各有硬约束，都写明原因）：
- `engram_fetch/wait`：ElasticBuffer 内部 HCCL 要求 **rankSize ≥ 2** → 单卡跑不了（M3/M4）
- `qsli`：需 candidate 块（candidate_topk_blocks 路径，下一轮）
- `engram_proj` 目前用零 kv **占位**（因 fetch 不可用），报告里显式标出

数值自检：KV 池写入 → 注意力读取链路通（`out_absmax≈10`）。
`dyn_mx_quant` 为隐式算子（由 `_qmm`/`wo_a` 内部产生）→ 模型照记、执行侧不计 skip。

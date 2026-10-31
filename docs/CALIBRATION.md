# 校准：让 dry-run 的算子直方图对上真实 trace（M1 验收）

> 参考 trace：`/mnt/xds/sfs/l00972766/ds41_light/res/20261013/light_8card_ep8_official_all_hit1m_cp1_hitfake/prof/prefill/<host>_<pid>_20261013131619591_ascend_pt/`
> 配置：`light_8card_ep8_official_all.yaml` + `--hit-ratio 0.97 --hit-fake-prefix`，**megamoe=deepgemm（融合路径）**
> 口径：L=64k、hit=0.97 → 单 pass `T_main = 0.03×65536 + 128 = 2048`；trace 里 4 个 Step（=4 个 pass），
> 每 pass 1800 个算子；`align_ops.py --ref-passes 4` 归一到单 pass。

## 1. 三条证据链（怎么把"次数"变成"结构"）

1. **按类型计数** → 每 pass 每个算子多少（`align_ops.py`）。
2. **按 Input Shapes 分组** → 每种形状各多少次（`tools/shape_seq.py --group`）。
   例：`QuantBatchMatmulV3` 162/pass = 4 个形状 ×40 + 1 个形状 ×2（engram）。
3. **按 Start Time 的形状序列解码** → 层的顺序结构（`tools/shape_seq.py`）。
   例：`KvCompressEpilogV2` 的形状序列是 `ABABAB…` ⇒ **每层恰好 2 次**，
   且 `C`（长块）只出现 4 次 ⇒ 4 个 kv_source 层各多 1 次。

三条合计把"每算子 × 每层"的矩阵解成 **基础次数 + 按层类型的增量**，全部可复现、可复核。

## 2. 校准结果（单 pass，T=2048）

| 算子 | 每层 | 按层增量 | 合计 | 证据 |
|---|---|---|---|---|
| RmsNorm | 4 | +5 会话级 | 165 | 660/4 ✓ |
| HcPre | 2 | — | 80 | ✓ |
| mhc_post_fwd_asc_kernel | 2 | — | 80 | ✓ |
| QuantBatchMatmulV3 | 4 | +2 engram 层 | 162 | 5 种形状：40×4 + engram 2 ✓ |
| DynamicMxQuant | 6 | +1×8 idx_src，+1×2 engram | 250 | 单元 `AABACD`，B 多 8、E 多 2 ✓ |
| InplacePartialRotaryMul | 3 | +1×4 kv_src | 124 | 形状 C=`1025,1,1,512` 恰 4 个 ✓ |
| KvCompressEpilogV2 | 2 | +1×4 kv_src | 84 | 序列 `ABABAB…`，C/D 共 4 ✓ |
| MqsmlaKernel | 1 | — | 40 | ✓ |
| TransposeQuantBatchMatMul | 1 | — | 40 | ✓ |
| FusedMatMul | 1 | — | 40 | 384 = 专家数 → 路由打分 ✓ |
| moe_topk_gate_forward_kernel | 1 | — | 40 | ✓ |
| mega_moe_impl | 1 | — | 40 | ✓ |
| Cast | 3 | +48 会话级 | 168 | ✓ |
| TensorMove | 2 | +5 会话级 | 85 | ✓ |
| Mul | 1 | +15 会话级 | 55 | ✓ |
| Add | 1 | +29 会话级 | 69 | ✓ |
| indexer_prologue_qw_kernel | 8 个 idx_src 层各 1 | — | 8 | ✓ |
| QuantLightningIndexer | 4 个 kv_src 层各 1 | — | 4 | ✓ |
| QuantSparseLightningIndexer | 4 层各 1 | — | 4 | ✓ |
| EngramFetch / EngramFetchWait | 2 个 engram 层各 1 | — | 2 / 2 | ✓ |

### 2.1 算子归因：trace 形状 × 真权重形状（互相印证）

4 个量化矩阵乘不是"泛泛的 attention 投影"，而是**逐一对上真权重**（`WEIGHTS_INVENTORY.md`）：

| 我们的名字 | trace 里的 QMM 形状 | 权重展平 | 真权重 tensor |
|---|---|---|---|
| `wq_a` | `2048,5120;40,320,16,32;80,1280` | 40×320×16×32 = 1280×5120 | `attn.wq_a.weight [1280,5120]` |
| `wq_b` | `2048,1280;1024,80,16,32;20,32768` | 1024×80×16×32 = 32768×1280 | `attn.wq_b.weight [32768,1280]` |
| `wkv` | `2048,5120;16,320,16,32;80,512` | 16×320×16×32 = 512×5120 | `attn.wkv.weight [512,5120]` |
| `wo_b` | `2048,8192;160,512,16,32;128,5120` | 160×512×16×32 = 5120×8192 | `attn.wo_b.weight [5120,8192]` |
| `wo_a` | `2048,8,4096;8,32,256,16,32;…` | 8×32×256×16×32 = 8192×4096 | `attn.wo_a.weight [8192,4096]` |
| `gate`（FusedMatMul） | `2048,5120;384,5120` | — | `ffn.gate.weight [384,5120]`（BF16） |

**权重清单反过来印证了按层次数**（这是最硬的一重交叉验证）：

| 权重 | 层数 | 与 trace 的一致性 |
|---|---|---|
| `attn.compressor.wkv` | 4 | KvCompressEpilogV2 的长块 4 个 = `kv_source_layers` ✓ |
| `attn.compressor.wgate` | **3** | 只 ratio=2 的 kv_src（2,8,14）——ratio=1 无需选择 ✓ |
| `attn.indexer.wq_b` / `weights_proj` | 8 | `index_source_layers` 8 层 ✓，且 dyn_mx_quant 在其上多 1 个 [T,1280] ✓ |
| `attn.indexer.wk` | 4 | QuantLightningIndexer 4 个 = `kv_source_layers` ✓ |
| `q_norm/kv_norm` | 40 | RmsNorm 4/层 = attn_norm+q_norm+kv_norm+ffn_norm ✓ |
| `ffn.shared_experts.{w1,w3,w2}` | 40 | 融进 `mega_moe_impl`（融合路径不单独出算子）✓ |

`dyn_mx_quant` 的 6 个形状也一一对上：`[T,5120]×3`（wq_a/wkv/ffn 输入）、`[T,1280]`（wq_b 输入）、
`[T,8,4096]`（wo_a 输入）、`[T,8192]`（wo_b 输入）；`[T,6144]` 只 engram 层（engram 表投影 ✓）。

分形状次数都记在 `ops.py` 的 `trace=` 字段里。

**当前状态：`tools/align_ops.py` 全部 21 行比值 = 1.00。**

## 3. 已知缺口（下一步）

参考 trace 里还有 **≈268 个/pass（14.9%）** 的小算子没建模，共 51 种，每一种都 <25/pass：

```
FloorDiv 24  Range 16  ReduceSum 15  ZerosLike 14  FloorMod 14  Fill 13  Sub 12
Index 8  GatherV2 8  ClipByValueV2 8  GatherElements 7  Less 6  _FusedPostScatterKernel 5
PadV3 5  GreaterEqual 5  Cumsum 5  BroadcastTo 5  ReduceMax 9  MaskedFill 9  RepeatInterleave 9 …
```

它们看着是**索引/元数据族**（稀疏索引器的块选择、路由表准备），多在 idx_src/kv_src 层附近。
价值判断：对"流水图完整性"有用，对"算子耗时"贡献小（都是小 shape）。
计划放进 M1.5：在 `recipes.yaml` 加一组 `idx_meta:`（挂到 idx_src 层）+ 用
`shape_seq.py` 确认它们到底挂在哪几层、是不是随 T 缩放。

## 4. 复现/再校准流程（换配置、换 kernel 后照做）

```bash
# ① 出我们的算子直方图（毫秒级，不碰卡）
python3 run.py configs/<config>.yaml --dry-run --dump-json /tmp/dry.json

# ② 对齐（--ref-passes = trace 里有几个前向）
python3 tools/align_ops.py --ours /tmp/dry.json \
    --ref <trace>/ASCEND_PROFILER_OUTPUT/kernel_details.csv --ref-passes 4

# ③ 某行对不上 → 看它的形状结构
python3 tools/shape_seq.py <kernel_details.csv> --op <算子名>            # 层内序列
python3 tools/shape_seq.py <kernel_details.csv> --op <算子名> --key Type  # mangled 名用 Type
python3 tools/shape_seq.py <kernel_details.csv> --op <算子名> --group     # 只按形状计数

# ④ 改 ops.py（count / shapes）或 recipes.yaml（存在性）后回到 ①
```

**换算子**（如 megamoe: deepgemm→ascendc、开/关 indexer_prolog_qw）：`ops.py` 里每条
spec 的 `affects=[…]` 已经标好它被哪个开关替换；`recipes.yaml` 的 `moe:` 段落按
`backends.megamoe` 二选一。换算子后重跑 ①② 即可看到哪一行的比值变了。

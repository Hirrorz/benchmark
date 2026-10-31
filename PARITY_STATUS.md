# 精度对拍现状（mini ↔ 官方 bench，真权重 16k）

> 2026-10-14 · 目标：同权重、同输入、同 EP，逐层 hidden 一致

## 1. 工具链（已通）

| 环节 | 做法 |
|---|---|
| 输入共享 | 官方与 mini 读**同一份** `parity/ids_16k.pt`（16384 token，sha256 记在 meta）；官方侧经 `lightctl.py --input_ids_file`（本会话给它补的参数） |
| 逐层 dump | 两侧都用**同一个表达式** `hidden_states.mean(dim=1)`（官方 `collect_layer_ids` 同式）：官方 `DS41L_COLLECT_LAYERS`、mini `--collect-layers` → `collect_rank{R}.pt`（40 层 × [16384,5120] fp32 = 13.4 GB/rank） |
| 对拍 | `tools/compare_layers.py --ref <A> --ours <B>` → 逐层表（✅<1e-3 / ⚠️<5e-2 / ❌）+ **第一处分歧层** |
| 官方参考 | 8 卡 EP8 真权重：`parity/ref/collect_rank0..7.pt` + `dump_rank0_call*.pt`；跑法见 `parity/run_ref_8card.sh` |
| mini EP | `launch_mini.py --world N --cards ...`（每进程 `ASCEND_RT_VISIBLE_DEVICES=<物理卡>` → 逻辑 0，绕开"逻辑卡号≥1 aicore fault"） |
| 权重加速 | `weights.host_cache`（真权重首读进主机内存并 pin）：40 层取权重 **52.1 s → 1.2 s**（实测，再遍）；只对**多次前向**有意义；`weights.resident` 则整模型常驻（EP8 = 49 GB 显存） |

## 2. 硬约束（实测，别再试）

- 官方真权重 **必须 EP8**：专家 ~300 GB，EP2 = 162 GB/rank、EP4 ≈ 87 GB/rank（都会 OOM），EP8 = 49 GB/rank 才装得下
- 官方**没有**逐层加载；等价省法是 `DS41L_RELEASE_MOE=1`（打包后放掉 raw 专家，省 33.6 GB/rank，**仅 prefill 安全**）
- mini 反而可以任意 EP（按层流式，一层 = 0.86 GB@EP8）

## 3. 本轮找到并修掉的 bug（都是"量级看不出来、数值全废"型）

| # | bug | 现象 / 定位方式 |
|---|---|---|
| 1 | **MoE 的 EP≥2 口径**：`e_local`（本地专家数）被当全局传给对称缓冲 → 内核按 `192/num_ranks` 校验 → 权重行数全错 | EP2 上 MoE **40 层一次都没跑成**，表现成"激活塌到 4.6"的假数值问题；被 per-op 静默吞掉 |
| 2 | **`hc_post` tilelang 契约**：x 必须 `[1,T,H]`、residual `[1,T,M,H]` | 我把 x 也扩成 4D → AssertionError |
| 3 | **rope 的 cos/sin 排列**：官方 `repeat_interleave(2)`（交错），我用 `cat(freqs,freqs)`（半半） | 旋错维度对 |
| 4 | **rope 的 base 分层**：win=10000、comp=`compress_rope_theta`=160000 | 38 个 comp 层基频错 |
| 5 | **rope 的 YaRN**：comp 层要 YaRN（factor=16 / beta 32,1 / orig 65536），win 层 original_max=0 退化为普通 RoPE | 修 3+5 后层 1 从 0.512 → 0.492 |
| 6 | **修复被自己的缓存吃掉**：`_rope_cos_sin` 的缓存键没带 base | "改了 rope 但对比表逐行完全相同"才被发现 |
| — | runner：per-op 异常**静默吞掉** | 现在"多数层失败"→ 打 ⛔ + **非零退出**（这个机制当场抓住了我自己 rope 补丁的 UnboundLocalError） |

## 4. 当前数字

| 对比 | 结果 |
|---|---|
| 层 0（嵌入） | **0.000e+00 ✅ 逐位一致**（两侧都是同一份 ids + embed.weight） |
| **层 0 输出（同 T=16384 逐算子对照）** | **全部 0.000e+00 ✅ 逐位一致**（q / attn_out / attn_post / ffn_norm_in / moe_in / moe_out / layer_out / topk_ids / topk_weight —— 与官方 `official_ops` 那次跑逐位相同） |
| 层 1（collect 口径，vs 08:54 那次官方） | 8.602e-03 ⚠️ —— **不是 mini 的错**：官方自身两次跑就差 1.409e-03（同 yaml 同 ids） |
| **官方(08:54) vs 官方(10:12)** | **1.409e-03** → **官方推理 run-to-run 非确定**（mini 与其中一次逐位相同） |
| **EP1 ↔ EP2 ↔ EP4（mini 自比，三份两两）** | **40 层全部 0.000e+00 ✅**（EP 切分数值无害已证；mega_moe 每 rank 40 次真执行、零错误） |
| 40 层曲线（mini 40L vs 08:54 官方） | 层 0 ✅ → 层 1 8.6e-3 ⚠️ → 层 2 起 0.38→1.0 ❌（**网络对 1e-3 级扰动混沌放大**，见下） |

### 4.1 两个根因（本轮找到并修掉，均已验证）

| # | 根因 | 证据 | 效果 |
|---|---|---|---|
| 7 | **rope 的 cos/sin 表被降成 bf16** —— 官方基类 `forward`→`_set_cos_sin_cache(..., dtype=torch.float32)`，**表是 fp32**；mini 的 `_rope_cos_sin` 里 `.to(torch.bfloat16)` | `q` 前 448 维逐位相同（100%）、只有 rope 的尾 64 维差 2.8e-3；同一份"逆 rope 前 q"配 **fp32 表**贴近官方(3.38e-3)、配 **bf16 表**贴近 mini(3.39e-3) —— 换表 dtype 匹配对象就换边 | 层 0 输出 **5.123e-01 → 8.602e-03**；同 T 逐算子后 **→ 0.000e+00** |
| 8 | **MoE 少传 `activation_clamp`**（SwiGLU 截断）—— 官方 `deepgemm_backend.py:212/263` 传 `moe.swiglu_limit`(=10.0)；mini 传 `None` | 官方参考实现 `modeling_deepseek.py:298-300`：`gate.clamp(max=10)`、`up.clamp(±10)`（非对称） | 层 0 实测 **no-op**（412 万个前激活无一 >10；max gate 3.6 / max|up| 4.97）—— 对深层量级大时必需 |

### 4.2 对拍方法学（踩过的坑）

- **必须同 T 对照**：官方 `official_ops/` 的逐算子 dump 是 **T=16384 跑的 512 行切片**；若拿 mini 的 **T=512 独立跑**去比，同算子在不同 T 上的分块/归约顺序不同 → 假差异（实测伪装成 `q` 3.6e-7、`attn_out` 8.3e-6、`attn_post` 1.4e-3）。同 T 后全部归零 ✅
- **collect 口径的极限**：官方 run-to-run 1e-3 噪声会被网络放大到 ~1.0（层 2-3 即饱和）→ **40 层"输出完全一致"在原理上不可达**；正确验收口径 = **同层给定同一输入 → 逐位一致**（层 0 已达成 ✅）
- runner 侧工具：`DS41M_DUMP_SLICE=<dir>`（可选 `N`，默认 512）逐算子存前 N 行切片

### 4.3 独立验证（真权重 CPU oracle，证明"孤立算子都对"）

| 环节 | 验证方式 | 结果 |
|---|---|---|
| gate / 路由 | 与 `x·Wᵀ` + `e_score_correction_bias` 口径手算 | corr **1.000000**、top-6 命中率 **1.000** |
| MoE | 真权重 mxfp4 解包手算 | 差 **2.5%**（= 激活量化噪声） |
| 注意力 | 用 dump 的 q/kv 手算滑窗 + sink | 差 **1.9%**；窗口错位诊断 **shift=0 最优 2.0%**（±1 → 6.5%/15%、±2 → 8%/31%） |
| wo_a 打包 / scale 布局 / 窗口索引 / `MXFP4_SCALE_GROUP_SIZE=32` / residual 刷新 | 逐字核对官方源码 | 全部一致 |

## 5. 已排除（层 0 内，逐式核对过）

窗口索引（与 `generate_win_topk_ids` 等价）· 全长池槽位（`block_ids[p//128]*128+p%128`，block 从 1 起、block 0 恒空）· 初始 `pre_mix`=identity · hc 展开 · rope 三处 · `hc_post` 形状 · MoE EP 口径 · 算子顺序

## 6. 靶子口径（无歧义）

`collect[0]` = 层 0 块的**输入** hidden（= 嵌入）→ 与官方**逐位一致 ✅**；
`collect[1]` = 层 0 块的**输出** → 相对误差 0.49 ❌ ← 卡点就在**层 0 块本身**。

## 7. 下一步

1. **层 0 差 0.49 的根因**（正在进行：子 agent + `DS41M_STATS` 逐算子统计 + 候选：`wo_a` 打包布局 / `mqsmla` 的 sinks 与 scale / `kv_epilog` 写池的量化参数 / kv 上 norm 与 rope 的先后）
2. 层 0 通 → 全量 40 层 EP8 对官方参考（`--world 8 --cards 0..7`）；EP 维度已证明无害，届时差异只剩"装配 vs 官方装配"
3. 再谈端到端"吐字"（当前 logits 分布归一化熵 0.60 = 有信号但没对齐）

---

## 7. 🏁 EP8 逐位对齐达成（2026-10-15 凌晨）

**结论：40/40 层、T=16384 全序列逐位一致（vs 官方 8 卡 EP8 真权重参考 `parity/ref/collect_rank0.pt`）**
`tools/compare_layers.py --ref ref --ours mini_c40c` → `✅ 40 层, ⚠️ 0, ❌ 0`，全部 0.000e+00。

### 7.1 这一轮修掉的三个根因

| # | 根因 | 修法 | 证据 |
|---|---|---|---|
| A | 索引器 **q** 的 rope 表自建了 `cat(freqs,freqs)`（半半）且**无 YaRN**，官方用的是与压缩器同源的 `cos_sin["comp"]`（**交错 repeat_interleave + YaRN 16/32,1/65536**） | `exec_indexer._rope_comp` 改为 `_CORE._comp_rope_tables(T, 1, ctx, device)` | 层 2 QLI 稀疏 511/512 → **512/512** |
| B | parity 钩子在**原地写入**的缓存上**调用前** dump → 存的是启动首跑遗留内容，导致"官方 K 与 mini 不一致"的整条误判 | 钩子改成调用后 dump（`out = _orig_wpk(...)` 之后） | 修正后按官方旋转块表对齐，K 缓存 **16384/16384 行逐字节相同** |
| C | QSLI 槽位分母用了 `candidate_block_size`(8)，官方是 `(block_size*ratio)//candidate_block_size` = **16**；同时拆分缓存也要用官方的**旋转块表** `[n,1,2,…]` | `_qsli_slots(n_rows, rows_blk, …)` + `_qslitab` 读写共用 | 组合缓存 96.8% → **100% 逐字节**；层 0–24 全绿 |

### 7.2 参考实现自身的一处"读未初始化显存"（已归因、可复现）

- 官方 `quant_sparse_lightning_indexer` 的 reader 对**候选块 0..15** 用 `block_table[0]` = **1025** 号块（`pa_block_size=128`、`page = block//16`），
  而写侧 `indexer_prologue_k`（storage_mode=1）把 p<16 的数据落在**行 2050**（实测 scale 126/126/125 = 真数据）。
- 于是官方 reader 读到的 16 行是**该算子从未写过的显存**：dump 出来 scale 字节 0..252 均匀分布、仅 2.3% 落在合法 E8M0 区间。
  后果：官方每行只选中前 128 个位置中的 ~80 个（垃圾分数挤掉 ~48 个），mini 零初始化 → 并列 → 全选中 → 集合差 9.4%。
- **注入实验**：`DS41M_INJECT_QSLI=<官方 L20_idx_qsli_cache.pt>` 把那 16 行字节灌进 mini 同位置 →
  层 24 QSLI **0/512 → 512/512 逐位一致**；带注入的 40 层 collect → **全 40 层逐位一致**。
- 这些字节在官方多次 run 间**完全稳定**（固定分配模式的确定性残留），所以它是可复现的参考状态，不是随机噪声。

### 7.3 复现命令

```bash
# mini（带参考缓存状态镜像；去掉 DS41M_INJECT_QSLI 则 4 个 QSLI 层会差 ~9% 集合）
source /mnt/l00972766/ds41_bench/env_image.sh; source /mnt/l00972766/setup_ds41_env.sh
unset TMPDIR; export ASCEND_RT_VISIBLE_DEVICES=5
cd /mnt/xds/sfs/l00972766/ds41_mini
DS41M_INJECT_QSLI=/mnt/xds/sfs/l00972766/parity/official_all15/ops/L20_idx_qsli_cache.pt \
  python3.11 runner.py configs/mini_real.yaml --seq 16384 --progress off \
  --ids-file /mnt/xds/sfs/l00972766/parity/ids_16k.pt --device npu:0 \
  --collect-layers /mnt/xds/sfs/l00972766/parity/mini_c40c  # （该目录已于 2026-10-30 清理删除；--collect-layers 是**输出**目录，重跑会自动重建）
python3.11 tools/compare_layers.py --ref /mnt/xds/sfs/l00972766/parity/ref \
  --ours /mnt/xds/sfs/l00972766/parity/mini_c40c
```

---

## 8. 可移植打包（2026-10-15）

与精度无关、但交付需要的部分：仓库已瘦身并做成**自包含分发包**（见 README §0/§8）：

- `env_pack.sh`（环境引导，**CANN 不锁版本**：找到哪套用哪套）+ `tools/selfcheck.py`（**验收门 = 算子自检**）
  + `tools/build_pack.sh`（`git archive` + 轻量 runtime → `dist/ds41_mini_pack_1015.tar.gz`，161 MB，
  sha256 `9b0d7fbac6e1855fe4f952ca2f8dd8a2d6204717275e0089fb98669e66e05972`）。
- 交付位置：`/mnt/xds/sfs/l00972766_vllm/stage_mini_1015/`（含 `README_DEPLOY_mini.md`）。
- 验收状态：**Tier 0 全绿**（干净环境下、用容器自带 CANN 而非覆盖层 → "不锁版本"成立）；
  **Tier 1（逐算子真调用）待跑** —— 当时 8 张卡被兄弟容器占满（每张 86/98 GB），
  卡空后：`python3.11 tools/selfcheck.py --card <空卡>`。
- 瘦身：`extra-info/`(1.6 G 崩溃 dump)、`prof/`、`.tmp*/`、`__pycache__/` → `attic/ds41_mini_1015/`（可逆）；
  跟踪文件 64 → 43。

---

## 9. 🏁 40/40 逐位复现（2026-10-17，deepgemm 口径）

今天从一个 9e-3 的差异追到**三颗根因**，最终在 **EP8 + deepgemm + QSLI 注入** 下与 `parity/ref`
**40/40 层逐位一致**（`统计: ✅ 40 层, 最差 0.000e+00`）。

1. **MegaMoE 后端**：`ascendc` 与 `deepgemm` 两个内核**数值不等价**。逐算子证据：层 0 除 `moe_out`
   外全部逐位相同（`moe_in`/`topk_ids`/`topk_weight`/`q`/`attn_out` 等 0 差），`moe_out` 2,449,929/2,621,440
   元素不同（max|Δ|=0.156）。精度对拍必须 `DS41M_MOEMODE=deepgemm`（官方参考也是 deepgemm）。
2. **压缩 rope 表越界**（`exec_ops._comp_rope_tables`）：cos/sin 表按 `n` 行建，但位置索引是 `p×ratio`
   （最大 = (n-1)×ratio）→ `index_select` **越界读**（NPU 静默不报错）⇒ 第 4097 个压缩行（token ≥8195）
   起 rope 全错。行级证据：层 3/4/5 **行 0..8194 逐位一致、行 ≥8195 全部不同**（8192 = T/ratio）。
   修：表按 `(n-1+row_off)*ratio+1` 行建（ratio=1 时与旧值相同 → c1a 层零回归）。
3. **QSLI 参考侧读未初始化 16 行**（§7.2 已归因）：修 1+2 后第一分歧变为**层 25 = 第一个 QSLI 层（24）
   的输出**，用 `DS41M_INJECT_QSLI=parity/official_all15/ops/L20_idx_qsli_cache.pt` 注入后 40/40 全绿。

### 复现命令

```bash
export DS41M_MOEMODE=deepgemm DS41M_MOE_CHUNK=auto DS41M_RELEASE_MOE=1
export DS41M_INJECT_QSLI=/mnt/xds/sfs/l00972766/parity/official_all15/ops/L20_idx_qsli_cache.pt
python3.11 launch_mini.py --config /mnt/xds/sfs/l00972766/winfix/config_pre.yaml \
  --world 8 --cards 0,1,2,3,4,5,6,7 --seq 16384 --iters 1 \
  --ids-file /mnt/xds/sfs/l00972766/parity/ids_16k.pt \
  --collect-layers <out> --collect-ranks 0 --progress off
python3.11 tools/compare_layers.py --ref /mnt/xds/sfs/l00972766/parity/ref --ours <out>
# → ✅ 40 层, ⚠️ 0, ❌ 0；全部 0.000e+00
```

### 参考侧确定性（同日实测）

`light` 两次独立跑（`parity/ref` vs `parity/official_ops2`）× rank0/rank7 × 40 层 **全部 0.000e+00**
⇒ 参考逐位确定；§4 关于"官方 run-to-run 1.4e-3 噪声"的说法作废（那两次不是同一口径）。

### 附：逐算子对拍工具

`tools/cmp_ops_vs_official.py`（mini 的 `DS41M_DUMP_TENSORS` 快照 ↔ 官方 `DS41L_PARITY_OPS` 同名张量，
自动标第一分歧算子；池按块主序比、官方 float8→fp32 口径已处理）与 `tools/cmp_rowwise.py`（按 token 行定位）。

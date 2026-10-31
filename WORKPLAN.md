# WORKPLAN（2026-10-21 · 当前会话）

> 机器：liteserver-hps-tokens-001-00006（8×Ascend950DT）
> 执行器：本机 `exec_agent_v3.sh` pid **8780**（14:33 起，环境自检通过），通道 `/mnt/xds/sfs/l00972766/cmds/`
> **14:32 本机 8 卡回到空闲基线**（兄弟作业退场）→ 原计划转投隔壁 -00003 的 A/B 已撤回，
> 本机就地跑 `cmd_20261021T1355Z_l22_tail_ab.sh`（撤回信 `agent_mail/old_20261021T1433Z.md`）。
> 上次会话在 09:28 单卡冒烟后被 kill；**13:49 容器 fd80073 被重启**（12:53→13:49 停机，
> 停时正在跑 `cmd_...T1345Z_l22_off2.sh` 的 B 段），代码/日志都在共享盘，无损失。
> **13:55 现状：8 卡被兄弟容器作业占用**（host pid 209051..209058 / 各 34.5GB / util 100%，
> 不在本容器 PID 命名空间）⇒ 不抢卡，命令里前置 `wait_npu_idle` 等它退场。
> **规则：每完成/变更一步就回来更新本文件。**

## A. MegaMoE 封装：日志系统 + deepgemm 切 chunk + 两后端同一套规则

- [x] `_BACKENDS` 适配器（`_AscendcBackend` / `_DeepGemmBackend`）：`inputs / prepare / call(单 span) / release_raw`
- [x] 调度器只留一份：`_mega_moe` 只做 专家号 → 缓冲复用 → prepare → release_raw → inputs → span 循环 → 写回
- [x] `_moe_cap` + 统一 span 循环：**deepgemm 也按 chunk 切**（官方 `get_moe_num_chunks` 口径；
      `ctx["chunk"]` = config `run.moe_chunk_max_len`，`DS41M_MOE_CHUNK=auto` 生效）
- [x] `_prepared_dg_cached` 不再顺手算 ascendc 的 ac（旧代码每层白拷贝 ~0.85 GB + format_cast）
- [x] `ds41_log.py` 日志分级（pure/error/warn/info/debug），`launch_mini` / `runner` 已接入
- [x] 单卡冒烟（deepgemm，T=1024 × 4 层）：**本机 09:37 通过**（82 调用 / 0 跳过 / 22 种）
- [x] EP8 ascendc 冒烟（`DS41M_MOE_CHUNK=1024`, T=4096, 2 层）：**09:45 通过**
      （8 rank / 35.4s / 41 调用 / 0 跳过；缓冲 `T=1024` ⇒ 每层 4 span；**ret:9 未复现** ⇒ 旧机为环境问题）
- [x] 单卡 deepgemm 多 span 冒烟（chunk=1024, T=4096, 2 层）：**09:46 通过**
      （缓冲 `T=1024` ⇒ 每层 4 span，12.5s / 41 调用 / 0 跳过；ascendc 预期性回退 deepgemm）
- [x] EP8 ascendc 默认口径（chunk=8192, T=16384）验证：**rc=0，786 调用/0 跳过，47.6s，峰值 18.68 GiB**
- [x] （回归）真权重 16k hit：**✅ id=396/396（重合 3/8）、稳态 123 ms、trace 断言全过、rc=0**
      （2026-10-21 18:14，`cmd_...T1802Z_tasklist12.sh`；比 10-18 的 181–188ms 更快）

## B. 统一 packed（b=1 也走 packed 路径，勿忘）

- [x] `batch.py`：n_seqs==1 且带 cu 也按 packed（2026-10-21 统一实现）
- [x] `exec_indexer.py`：同口径（n_seqs==1 走 packed）
- [x] `exec_ops.py`：分派口径 = 有没有 `cu_seqlens`（b=1 也走 packed）
- [~] 验证：b=1 packed vs 旧 sequential 逐位/算子数对拍（8 卡、seq=2048、40 层）
      - A 段（sequential）✅ 86s / 8 rank 全 0
      - B 段（packed）❌ **卡死在起跑线 `dist.barrier()`**（py-spy 8 rank 同一栈；
        plog `SynchronizeExecutedTask: report three minutes timeout! stream=63 task=3`）。
        A→B 间隔仅 3 s、卡上残留幽灵内存（5.8GB vs 基线 ~5.1GB）⇒ 判定为**背靠背 HCCL 未清场**，
        非 packed 代码问题（待冷却后单独重跑 B 复验）
- [x] 重跑 packed b=1（清场后单跑、HCCL_IF_BASE_PORT=65440）：**起跑线通过、40 层全跑完**，
      但收尾 `torch.npu.synchronize()` 报 **aicore 异常 507015**（rank1/2/4/5/6/7 六张卡；
      异步，主机侧层行已全部打印）→ **packed b=1 有真 bug**（sequential 同点无错）
- [x] 定位（SYNC_LAYERS=1）：层 **24（c1a_indexsource）的 `qsli`** 首次报错
      `507018 AclQueryEventRecordedStatus`（异步设备错在 qsli 的 host 同步处暴露）
- [x] 逐算子同步复验（L25 短配置，`cmd_20261021T1105Z_packed_opsync2.sh`）：**首个 ✗ 就是 qsli**
      - 报错码 **507018 = `ACL_ERROR_RT_AICPU_EXCEPTION`**（不是 aicore trap）→ QSLI 里唯一 AICPU
        部件是 **`quant_sparse_lightning_indexer_metadata` 调度 kernel**（`qsli_metadata_kernel.so`）
      - QSLI 实际几何是**回放裁尾**：T=128、off=1920、lens=[2048]、bt=(1,130)
        （L25 里回放从层 21 起裁到 W=128；40 层原跑同层也是这个运行时几何）
- [x] **根因已定位（dump 对拍）**：`cl / cand_i / cand_l` 逐位相同，唯一差异是
      **`cu_seqlens_q`：sequential=[0,128]，packed=[0,2048]**（2048=逻辑长 L，不是行数）
      - 机制：`replay.apply()` 只在 `pn>1` 的 packed_tail 分支改写 `packed_ctx["cu"]`；
        **b=1 走 single_tail 分支 → cu 不更新**（仍是 [0,L]），而 x 已裁到 W 行
      - runner 每 pass 开头重建 cu 的语句也只在 `n>1` 时执行 → b=1 永远修不回来
      - QSLI metadata AICPU 按 `cu[-1]=2048` 行调度，但 q/cand 只有 128 行 → AICPU 越界 507018
- [x] **修复**（2026-10-21）：
      - `replay.py` single_tail 分支：`packed_ctx["cu"] = [0, W]`
      - `runner.py`：每 pass 重建 cu 去掉 `n>1` 限制（n==1 也按 row_lens 重建）
      - `exec_indexer_packed.py`：qli/qsli 增加 `cu[-1] == T` 防御断言
      - 诊断钩子保留：`DS41M_QS_DUMP=<dir>`（层 24 落 qsli 入参快照）
- [x] 修复后验证：
      - `cmd_20261021T1125Z_packed_fix.sh`：packed b=1 跑通（rc=0, 76s），QSLI 入参 cu=[0,128] ✓
      - `cmd_20261021T1130Z_packed_ab_fix.sh`：**逐层对拍 25/25 层逐位相同**（max|Δ|=0）
- [x] 40 层全量 A/B（sequential vs packed b=1）：**40/40 层逐位相同**（stats: ✅40 ⚠️0 ❌0，max|Δ|=0）
- [x] packed-hit b=1 冒烟（h=0.97, L=16384, R=640）：40 层 / 786 调用 / 0 跳过 / rc=0
- [x] 结论归档：`NOTES/2026-10-21-packed-b1-cu-tail.md` + INDEX
- [x] `--batch-mode auto` 的 b=1 已切 packed（decode/--layers/--prompt/real 前缀自动回退）
- [x] 验证：`--points` 多点冒烟（`DS41M_UNIFY_PACKED=1`，3 点：4k plain b1 / 4k hit.96 b1 /
      4k hit.96 b2）：**rc=0 全绿**（`cmd_20261021T1140Z_points_unify.sh`）
- [x] hit b=1 数值 A/B：**40/40 层逐位一致**（三根因修完后，`cmd_...T1511Z_ab40_hit_fix3.sh`）
- [x] 定位 hit b=1 分歧（`DS41M_DUMP_SLICE` 逐算子快照 A/B，层 0–3）：
      - 层 2（c2a_source）的 **`_idx_k_r2`（索引 K 缓存）** 与 `idx_sparse` 先分叉；
        `idx_q/idx_w/idx_descale_q/cmp_latent/kv/q/residual` 全部逐位相同
      - **K 内容逐位相同**（seq 逻辑行 0..127 的 8192 字节 == packed 的），只是**落位不同**：
        - seq：压缩行 0..127 → slots **960..1023**（块 7 行 64..127 + 块 8 行 0..63；`block_table=None`）
        - packed：同样 128 行 → slots **1920..2047**（块 15 行 0..127；`bt=[17,1..16]`）
        - 差因：seq `_idx_k_cache` 的 `_ci_build(p)` 里 `r0 = p // ratio`，而 `p=row0=off//ratio=1920`
          ⇒ **二次除 ratio** = 960；packed 用 `loc_abs = loc + off//ratio` = 1920
      - ⇒ 这是 points_runner 注释里"comp/indexer 仍有分歧"的实体；两边各自与自己的 reader
        自洽，但 QLI 输出不同（127669/131072 个索引不同）→ 需按官方 `position_ids_c` 口径定谁对
- [x] 槽位口径实验 1（`DS41M_PK_IDXOFF=seq`，槽位+rope 一起减半）：层 0–20 的 collect 对拍转绿，
      但逐算子 dump 显示 `_idx_k_r2` 仍有 **内容**差（231 字节）——因为 rope 也被减半了（错）
- [x] 修正：**槽位基址与 rope 位置分离**（`_off_slot = off_c // ratio` 仅用于 cache_index；
      rope 仍用 `off_c`，官方 (row_off+i)*ratio 口径）
- [x] 复验（`cmd_20261021T1245Z_hit_l22_pk2.sh`，L22 短配置、逐算子 dump）：
      - **层 0–19 逐位一致**（`_idx_k_r2` 差异消失）
      - 层 20（c1a_source）：`q`/`kv`/`_full_pool`/`idx_sparse` 全逐位一致，但 **`attn_out` rel=1.25**
        → **mqsmla 输出分歧**；层 21 起 `_full_pool`（窗口池）跟着偏
      - 嫌疑：mqsmla 的**压缩 KV 块表 / cmp_topk_length / metadata**（未在 dump 键里）
- [x] **层 20 根因（调试打印对比）**：`seqused_cmp_kv` 口径错
      - seq：L2..L19 = 2048（ratio2）、L20 = **4096**（= off+T）；packed：**128 / 256（= 本段行数）**
      - 来源：`exec_ops_packed.kv_compress` 把 `_cmp_len_r{ratio}` 存成 `l_real`
        （本段写入行数）；顺序实现（exec_ops.py:1075）存 `(off+T)//ratio`
        —— 官方 `compressed_seq_lens = actual_seq_k // ratio` 与顺序版一致
      - 只有 ratio=1 的层（20+）显形（ratio>1 有 cmp_residual_k，seqused 不直接决定可见性）
- [x] **修复**：packed `_cmp_len_r` 改为 `off_c + l_real[i]`（绝对可见压缩长度）；
      plain prefill（off=0）逐位不变
- [x] 复验：L22 hit b=1 packed(IDXOFF=seq) vs seq 逐层 A/B（`...T1310Z_hit_l22_fix2.sh` +
      `...T1315Z_hit_l22_fix2_seq.sh`）：**层 0–20 全绿，只剩层 21 rel=3.08e-2**
- [x] 层 21 调试打印（`...T1320Z_l21_dbg.sh`）：mqsmla 入参（seqused=4096/cl/索引/块表）
      **与顺序版完全一致** ⇒ 差异在未 dump 的张量（窗口池内容 / 压缩池）
- [x] 加 dump 键：`_win_idx/_win_len/_bt/_cu/_cmp_sparse_in/_cmp_len_in/_cmp_pool_r*/_cmp_nblk_r*/
      _cmp_pc_r*/_cmp_len_r*/_full_pool_nblk`；池类张量按"从头看"
- [x] **TAIL 切片**（`DS41M_DUMP_SLICE_TAIL=1`）：被回放裁尾的层（L21 只有 128 行）原先被
      固定 `OFF=128` 切空 → 假"无差异"；现在 token 级张量取本层最后 N 行。另加
      `DS41M_DUMP_FROM` 对 slice dump 生效（只落层 20/21）
- [x] 卡空（14:32）后本机跑 `cmd_...T1355Z_l22_tail_ab.sh`：A/B 都 rc=0，但 **dump 0 文件**
      → 根因：TAIL 分支对不存在的键（`v=None`）取 `.shape` 抛异常，被外层 `except: pass`
      吞掉 → 整个算子快照不落盘。修：off 计算挪进 `isinstance(v, Tensor)` 之后；
      并把该 `except` 改成 `DS41M_DUMP_DEBUG=1` 时打印（不再全静默）
- [x] 重跑（`...T1441Z_l22_tail_ab2.sh`）：dump 正常（41 文件/臂），层 20/21 全键有差
- [x] **层入口 digest 定位**（`DS41M_PACKED_DIGEST_LAYERS=19,20,21`，`...T1450Z_l22_digest.sh`）：
      L19/L20 入口 sha1 **完全一致**，**L21 入口不同** ⇒ 分歧发生在**层 20 内部**（不是回放裁尾）
- [x] **层 20 根因**：压缩池多写。`...T1457Z_l22_pool.sh`（整池 dump）实证：
      - seq `_cmp_pool_r1` 非零行 **4096..4223**（128 行）；3968..4095 = 0
      - packed 非零行 **3968..4223**（256 行）；4096..4223 与 seq **逐位一致**
      - 机制：官方 hit.py「replay 行的压缩 KV 写入抑制（slot=-1），[0,H) 复用 stage A」，
        顺序实现 `exec_ops.py:_kv_compress` 有这段，**packed 实现漏了**
- [x] **修复**：`exec_ops_packed.kv_compress` 补同一段抑制
      （`loc < (off+rep)//ratio` → slot=-1；rep=ctx.hit_replay_tail）
- [x] 复验 1（L22 短配置，`cmd_20261021T1503Z_l22_fix3.sh`）：L19/L20/L21 入口 sha1 两边相同；
      层 20/21 逐算子**所有数值键 0 差**（只剩 packed 独有的 `_idx_bt_*` 块表键）
- [x] **复验 2（40 层真实配置，`cmd_20261021T1511Z_ab40_hit_fix3.sh`）：✅ 40 层全逐位一致**
      （compare_layers：✅40 ⚠️0 ❌0，最差相对误差 0.000e+00）
      ⇒ packed hit b=1 数值对齐**完成**（三根因：槽位口径 / seqused_cmp_kv / replay 段写入抑制）
- [x] **默认转正**（2026-10-21 15:15）：
      - `runner.py --batch-mode auto`：b=1 也走 packed；仅 packed 不支持的组合
        （`--decode/--layers/--prompt`、real 前缀命中）自动回退 sequential（不报错）
      - `points_runner.py`：`DS41M_UNIFY_PACKED` 默认改 1（=0 可退回旧 b1 路径）
      - 显式 `--batch-mode sequential` 仍可用于 A/B
- [x] 冒烟回归 ①：`cmd_...T1517Z_unify_smoke.sh` — auto+b=1 打印 `[packed] 真 batch：1 条`，
      22 层 436 调用 rc=0
- [x] 冒烟回归 ②：`cmd_...T1522Z_points_smoke.sh` — 3 点（4k plain b1 / 4k hit.96 b1 / hit.96 b2）
      全走新默认，rc=0（193.6 / 111.6 ms/pass）
- [x] EP8 ascendc 默认口径（chunk=8192, T=16384）：**rc=0，786 调用/23 种/0 跳过，
      47.6s，峰值 18.68 GiB**；后端 ascendc 融合算子（T=8192 缓冲 → 2 span/层）
- [x] 结论归档：`NOTES/2026-10-21-packed-hit-b1-indexer-slots.md` + INDEX
- [x] 清理临时大 dump（释放 ~124G：hit_l22_dump/pk2、hit_b1_dump×2）

## C. 收尾

- [ ] `NOTES/2026-10-21-megamoe-backend-unify.md` + `NOTES/INDEX.md` 加行
- [ ] git commit：packed 统一 + MoE 封装 + 日志系统（分两次提交更好审）

## 已知风险 / 待观察

- 03:34 的 `80k_hit0.97` L=16384 跑测：`ascendc 不可用（Acquire HCCL channel failed, ret: 9）`
  → 回退 deepgemm，层 1 卡 6 分钟。疑旧机抢卡/资源耗尽，本机复验。
- 本机根盘 97% 满（17G 可用）：TMPDIR 一律指 `/mnt/xds/sfs/l00972766/tmp`。
- **背靠背 8 卡 run 必须清场**：上一跑退出后等 HBM 回基线（~5.1GB/卡）再起下一跑，
  否则首次 collective（起跑线 barrier）会在设备侧挂死（2026-10-21 实测，A→B 只隔 3s 就中）。
- **本机 8 卡也有兄弟容器**：2026-10-21 10:07 起 npu-smi 出现宿主 PID 1698723…（本容器 ps 不可见）
  ×8、各 33.7GB、util 100% → 别人的作业。判据同 [[shared-npu-sibling-container]]：进程号量级
  与 /proc 不匹配即兄弟作业；不抢卡，等它退场。

## D. 构建器重设计：点位 × 批次 × 采集口径（2026-10-21 讨论定稿）

**已定口径（用户拍板）**
- 删除旧概念：`packed / 双 batch / batch-mode` 从 **UI 和生成的命令**里都去掉（packed 已是唯一实现）。
- **点位只是一个分组标签**；**L 全放在批次上**：`points[] → batches[{L, hit}]`，`hit=0` 即 plain prefill。
- 一个点的多个批次 **打包进同一次 pass**（不是顺序跑）；每批可不同 `hit`、可不同 `L`。
- **采集口径整次 run 统一**（不进点位 JSON）：prof 腿（仅 prof / 双腿 / 仅无 prof）、replay(+W)、
  轮数、warmup、图尾。
- **权重模块分开**（实现层）：
  - 合成模块：支持多批（逐批 L/hit），入口 `run_hit.sh`；
  - 真权重模块：**先不做多批**（单批 L/hit），parity **自动开**，入口 `run_real.sh`。
  - 共享：点位/批次模型、几何推导、采集口径、产物命名。
- **命令层不合并**：builder 按「权重 + 采集」自动挑正确入口。
- **产物命名**：按一个 pass 内各批规格拼接 `(L1_hit1&L2_hit2&…)/prefill`。

**引擎（已完成 2026-10-21）**——packed 支持逐序列 L/hit：
- [x] `cfg["_packed"]` 逐序列 `lens/row_lens/cu`（points_runner v2 生成）
- [x] `assemble.build_stack`：`T_pass=ΣR_i`，meta 增加 `T_mains/lens/pos_offs`
- [x] `runner` hit 分支逐序列 `pos_off`；`ctx` 增加 `pos_offsets/row_lens`；replay 逐序列前移
- [x] `batch.local_positions` 支持逐序列偏移（`pos_offsets`）
- [x] `exec_ops_packed.kv_compress`：start_pos/rope/池槽位/seqused/抑制上界 全部逐序列
- [x] `_mqsmla` packed：`cmp_topk_length=(pos_i+1)//ratio` 逐序列
- [x] `exec_indexer_packed`：K 缓存槽位/rope 逐序列
- [x] `points_runner`：v2 点位格式 `batches:[{L,hit}]`、逐批 seed（`DS41M_PACKED_SEEDS`）、
      产物命名 `(L1_h1&L2_h2)_r{0|1}/prefill`
- [x] **验证**：
      · 对称性（两批同 L/hit/seed、`DS41M_NO_ENGRAM=1`）→ **L0..L21 每层 seq0==seq1 逐位**
      · 跨轮 A/B（mixed vs uni 同规格批）：L0 逐位一致；L1 起 2e-6（MoE 按 T 打包的累加顺序），
        之后按混沌放大 ⇒ **跨不同批组成的逐位对拍不可作为判据**
      · 回归：L22 hit b=1 packed vs sequential —— 公共数值键**全 0 差**（只剩 packed 独有的块表键）
- [ ] 已知：合成 engram id 按行随机 → 跨批组成的数值对拍要 `DS41M_NO_ENGRAM=1` 或用真 ids

**builder（2026-10-21 已完成第一轮）**
- [x] 点位表 v2：**一行 = 一个批次**，新增「点位」分组列（同名 = 同一 pass）；
      JSON v2 `[{tag, batches:[{L,hit}]}]`；旧格式 `{L,hit,batches:N}` 导入时自动展开
- [x] 删除旧口径：`packed / 双 batch / batch-mode` 从 UI、预设、配置名、命令参数、文案**全部清零**
      （`grep -c packed bench_command_builder.html` = 0）；launch 卡只留「batch 数（同规格）」
- [x] 采集口径 run 级：replay 开关（→ `--no-replay`）、`--replay-tail`、`--log-level`
      （info/warn/error/pure/debug）+ 原有轮次/预热/prof 腿
- [x] 权重模块显式分开：多点卡 = **合成权重模块（性能 + 快速启动）**；
      real 卡 = **真权重模块（精度）**，`--parity` 默认开
- [x] 产物命名随引擎：`(L1_h1&L2_h2)_r{0|1}/prefill`
- [x] 测试：builder **392 项**（含新增的 replay/log-level/批次导入用例）；
      两个布局变体重新生成并各 392 项全过；布局结构测试 58 项全过
- [ ] 真权重模块多批（按用户决定：暂缓）
- [x] **导航收编为两个模块**：合成权重·性能（多点/批次 + 调试单跑）/ 真权重·精度（单跑 + 常驻）；
      hit 不再单列（= 多点里 hit>0 的批）；tests 401×3 + 布局 58 全过
- [ ] hit 卡片从 DOM 移除（导航已隐藏，`setEntry('hit')` 仍可用）
- [x] **日志等级四处打通**：builder 四张卡都有下拉（info/warn/error/pure/debug）；
      `runner.py` 新增 `--log-level`（argparse 前先扫 argv）；`run_real.sh` 新增
      `--log-level/--pure-log`；`run_hit.sh` 原本就有；launch_mini 靠 parse_known_args 透传
- [x] **不同 L 混批验证**：`[8192@0.96, 16384@0.90]` → `T_pass=2304（各行 512+1792）`，
      回放逻辑位置**逐序列** `8064..8191 / 16256..16383`，无崩溃 ⇒ 逐序列 L 生效
- [x] **per-DP（per-rank）不同 batch 组合 · 第一版（1-span 区间）**（2026-10-21 18:34）
      - 点位文件新形式：`[{"tag":"dp8mix","dps":[{"rank":0,"batches":[{L,hit}]}, …]}]`，每 rank 取自己那条
      - **前置 span 校验**：各 DP 的 `ceil(ΣR/chunk)`（含裁尾后的 `ceil(W×nseq/chunk)`）必须相同，
        不一致直接报错（实测 dp0=(1,) vs dp1=(4,) 被拦下，rc=1，不跑）
      - 逐批 seed 在 per-DP 下按 rank 偏移（各 DP 数据不同）
      - 产物：`<out>/<tag>_r{0|1}/` + `dps.json`（记录全部 DP 规格）
      - 冒烟：8 DP 各不同组合（1–2 批、混合命中率）→ rc=0；rank0/1/7 日志各行数/规格各自正确
      - 限制：只覆盖"各 DP 的 span 数相同"的场景（小 L/少批天然 1 span；`L≤128k` 且批数≤2–4）
- [x] per-DP 第二阶段：**dummy span 对齐**（2026-10-21 18:39–18:42，rc=0）：
      各 rank 跑满 `max(ceil(T_i/chunk))` 轮，少 token 的 rank 补 dummy span；
      `DS41M_MOE_DUMMY_SPAN=0` 时前置报错被拦下，默认开则每层补 `[mega_moe] dummy span ×1`
- [x] builder UI：per-DP 编辑器（2026-10-21 开放，见 E2「per-DP（已开放）」）

## E. 最终验收：builder 功能复测清单（2026-10-21 用户要求）

> 目的：本轮改了点位模型 v2、删了 packed/双batch、加了日志等级/权重模块/导航两分组，
> 交付前必须**整套复测**并留证据。规则：每项都要贴实际输出/数字，不能只打勾。

**E1 自动化（每次改动后都跑，全绿才算完成）** — 2026-10-21 复测通过
- [x] `node tools/test_bench_builder.js`（BENCH_HTML=base）→ **515 passed, 0 failed**（2026-10-22 复跑）
- [x] 两个布局变体各跑同一套 → **515 / 515 passed, 0 failed**
- [x] `node tools/test_layout_variants.js` → **90 passed, 0 failed**
      （id 全集/无重复、顶部脚本通栏、10+3 卡结构、场景导航 4 项、两个模块分组、主题初值、折叠/主题切换）
- [x] `python3 tools/gen_layout_variants.py --check` → 两个变体都是「最新」（无漂移）

本轮新增用例：per-DP **入口未开放**（模式块隐藏 / 编辑器不展示 / 导入 dps 明确拒绝且不污染多点预览）、
DSpark 开关与联动（需要 tp8 / verify 图需 DECODE_GRAPH / 关闭告警）、顶部通栏结构断言。

**E2 功能点（jsdom 已覆盖，但要在**浏览器里**再手点一遍）**
- [x] 点位表 v2：加行 / 上移 / 下移 / 删除；同「点位」名的行合成一个 point
- [x] 点位表校验：L 非 128 倍数、hit=150 → 行内红框 + 顶部状态定位到行，复制/下载被禁用
- [x] 导入 points.json：v2（tag+batches[]）正常；旧格式（L/hit/batches:N）自动展开成 N 行；
      坏 JSON / 未知键 → 明确提示
- [x] JSON 预览与「下载 points.json」内容一致（v2 结构、无空格）
- [x] 采集口径：replay 关 → 命令带 `--no-replay`；`--log-level pure` 能生成；改回 info 不再出现
- [x] 四张卡（多点/launch/real/hit 内部）都有「日志等级」下拉且能进命令
- [x] 真权重卡：`--parity` 默认勾选；合成卡标题/提示为「性能 + 快速启动」
- [x] 导航：两个分组「合成权重·性能 / 真权重·精度」；hit 不再单列；点每个入口卡片正确切换
- [x] 复制 / 下载 / 分享 URL 往返（含点位名、备注、log-level、replay）
- [x] 主题切换（◐）在 base 与两个布局变体里都正常；侧栏折叠正常
- [x] 打印预览：新壳层（左导航/底部条）隐藏，内容完整
- [x] **悬停说明**：字段标题有虚线下划线，悬停显示「含义 + 格式 + 示例/范围」；
      关键下拉（moemode/resMode/console/delivery/…）带每个选项的解释；点位表 L/hit 有格式提示
- [x] **新手引导**：多点卡顶部「第一次用？按这 3 步」→ 一键「冒烟 8k（1 点）」
      （1 轮 + 只跑稳态）；按钮用完可收起并记住；「使用说明」首节也是冒烟流程
- [x] **底部折叠区**：页面最下方通栏两栏「校验 / 摘要」，默认收起；校验标题右侧徽标
      （✗ N 错误 / ⚠ N 提示 / ✓ 无问题）实时更新，点标题展开正文
- [x] **生成的脚本**：置顶通栏（`#outBand`），宽屏下代码行铺满整行；右侧栏只留 chips/校验/摘要
- [x] **per-DP（已开放）**：点位卡顶部「模式」二选一 → per-DP 8 行 DP0..DP7；
      每行 `L@hit`（多批逗号）；「用第 1 行填充所有卡」；切回多点位表内容保留；
      `--points-spec` 置灰；导入 dps JSON 自动切模式
- [x] **MoE 后端默认**：三点位/launch/real 三处下拉默认 deepgemm，命令里显式写
      `--moemode deepgemm` / `export DS41M_MOEMODE=deepgemm`；切 ascendc 有「不可比」提示
- [x] **DSpark**：常驻卡可见「DSpark 投机解码」下拉（B=1..5）；选 B=5 且模式非 tp8 →
      红色错误提示；切到 tp8-fidelity 后服务端脚本出现 `export DS41M_DSPARK=5`；
      摘要/芯片出现 DSpark 行/标签

**E3 命令层复测（生成的命令要能真跑）**
- [x] 多点命令：`run_hit.sh --points <v2>` 跑 2 点（含一个多批点位）**rc=0**（2026-10-21 18:51–18:54）
      · 修前先暴露真 bug：`points_runner` 的 v2 分支用了未定义的 `_bs`（NameError，8/8 rank）
        → 补 `_bs = pt.get("batches")` 后重跑
      · 结果：`results.tsv` 两行 —
        `(8192_0.96&16384_0.96)_r1` 稳态 **0.140 s**（2 批同 pass；T=1280 行打包）
        `(32768_0.72)_r1` 稳态 **0.381 s**；8 rank 均「全部无算子错误」
      · 产物目录 `(L_h&L_h)_r1/prefill` 仅在 prof 轮生成；noprof 看 `results.tsv`/`SUMMARY`
- [x] per-DP（`dps`）命令：8 DP 各不同组合 rc=0（2026-10-21 perdp_smoke，18:36）
- [x] span 对齐（2026-10-21 dummy_span 18:39–18:42，rc=0）：
      · 混用 `dps`+多点位 → `✗ per-DP 模式只能有 1 个点位`（8/8 rank）
      · `DS41M_MOE_DUMMY_SPAN=0` + pre=[1,2,1…] → 前置报错 `✗ per-DP 的 MoE span 数不一致`
      · 默认（dummy 开）→ `[per-DP] span 不一致 → dummy 对齐：pre=[1,2,…] → 2 轮`，
        每层补 `[mega_moe] dummy span ×1`，全程 rc=0、无算子错误
- [x] 默认后端冒烟（2026-10-21 19:09–19:13，rc=0）：不传 `--moemode` 跑
      `run_hit.sh --points` 单点 8192@0.96 → `config_points.yaml` 为 `megamoe: deepgemm`，
      8 rank「全部无算子错误」，稳态 **0.119 s**（1 pass，12.68 GiB）；
      `check_perf_defaults.py` → ✅ PASS（light 规则未受影响）
- [x] 真权重命令：`run_real.sh -L 16k --mode hit`（deepgemm 默认，2026-10-21 19:41，rc=0）
      → `id=396 ' that'`、精度 gate 全过；修复 DG 缓存后稳态 **145 ms**
      （初测 526 ms；差距来自每 pass 重做 transform。DG 版 AC/raw 释放 + 常驻缓存已对齐，
      ascendc 基线 123 ms，约 +18%）

- [x] **serve packed 几何修复**（2026-10-22）：根因是 b=1 packed 后 serve 仍用启动
      `--seq` 的 `_packed`（W1 实际跑 T=1024）→ 层 2 mxfp8 QMM aicore / gate NaN。
      修复后真权重 ep8+inc 预热全过、decode 正常（~2.6s/token）。
- [x] **DSpark ep8**（2026-10-22 02:16，rc=0，无 P→D）：serve packed 几何修复 +
      草稿 EP 走 `_moe_decode`（融合 op 在 K=3 下 OOB）+ verify 切融合 MegaMoE
      （2.4s → 176ms）+ 对称缓冲多槽。实测 ~0.196 s/token（ep8 inc 基线 ~2.6 s/token）。
      遗留：accept 0/5（短 prompt + 空格 padding，属草稿质量调优）：草稿 MoE 按 EP 切 16 专家、`_moe_layout`
      强制融合 MegaMoE；runner 允许 ep8（inc/full）；builder 已放开并提示。
      阻塞点：真权重 ep8 inc 的 T=1 decode 在 `moe_topk_gate` 报 NaN（不设 DSpark 同样），
      ep8 full 在一次启动中报 aicore 507015；设备已回收（4.75GB 基线）。
      详见 `NOTES/2026-10-21-dspark-ep8-support.md` 的排查顺序。

**E4 交付物**
- [x] `tools/bench_command_builder.html`（主文件）
- [x] `tools/bench_command_builder.layout_studio.html` / `layout_bento.html` / `layouts.html`
- [x] 复测结果写回本清单（贴数字），并同步 NOTES/INDEX
      → **完整复测报告**：`NOTES/2026-10-21-builder-retest-report.md`
      （469×3 + 90；生成脚本 `bash -n` 全过；命令层 rc=0）

**E5 构建器 / pure / P→D 复验（2026-10-22）**
- [x] 构建器：每个点位 **Replay 打钩**（不是下拉；勾选写 `/r0|/r1`，全关才加 `--no-replay`）、
      per-DP 也有 Replay 勾选；环境自检改**独立陈列**（不进生成的脚本，cd/source 仍在 sh 里）；
      「跳 MoE / 跳 engram」开关 → `DS41M_SKIP_MOE/ENGRAM=1`（真权重口径给告警）。
      自动化 **515 passed × base/studio/bento** + layout 90 passed，`gen_layout_variants.py --check` 无漂移。
- [x] pure 模式点位级输出：`points_runner` 打印
      `[point i/N] 开始 HH:MM:SS tag specs replay` 与
      `[point i/N] 完成 HH:MM:SS tag 用时 Xs 每pass=Yms（iters×npass，除预热）`；
      `ds41_log._PROGRESS_RE` 放行这两行，`run_hit.sh` 的 pure 终端过滤器兼容 `[r0] ` 前缀。
      单点冒烟 `--points-spec '1k@0.5' --noprof-only --iters 2 --log-level pure --no-bar`：
      rc=0；驱动日志含 `[point 1/1] 开始 03:44:12 …` / `完成 … 用时 143.5s 每pass=106.8ms`。
      坑：launch 回显行有前导空格，终端过滤器已放宽 `^[[:space:]]*(\[r0\] )?…`（用该次
      driver 日志验证命中）。
- [x] **单卡性能预设**（用户口径）：构建器 launch 预设「单卡 · 跳 MoE+engram」
      → world=1 / cards=0 / `DS41M_SKIP_MOE=1` / `DS41M_SKIP_ENGRAM=1` /
      seq 2048 / iters 2 / warmup 1。用生成的脚本真跑（2026-10-22 04:01–04:03）：
      rc=0，iter1 **0.173 s**、iter2 **0.071 s**（40 层，只剩 attention/indexer/compress）。
      新手引导里有一键入口；`run_hit/run_real` 的「固定 8 卡」报错会指向该预设。
      自动化 **561×3 + 90** 全绿。
- [x] **P→D ep + MOEMODE=deepgemm 崩溃**（2026-10-22）：构建器脚本导出 deepgemm，`prepare_decode` ep 分支的 setdefault 不覆盖 → decode 走 prefill MegaMoE 读已释放 raw → `KeyError ffn.experts.0.w1.weight`。改为强制 `DS41M_MOEMODE=decode`；合成 40 层复现修后 rc=0。详见 `NOTES/2026-10-22-pd-ep-moemode-keyerror.md`。
- [x] **便携包 + 缺件文档**（2026-10-22）：`tools/make_portable_kit.sh`（默认 torch+CANN overlay，排除 logs/dump/备份/测试）；`PORTABLE.md` + `PORTABLE_TROUBLESHOOTING.md`（含 MegaMoE 安装步骤）；env_pack 增加 deepgemm-pkg / runtime/bin/ninja；`DS41_SHARE=/nonexistent` 下 import 全通。
      **selfcheck 三个 ❌ 的真根因 = 包里漏了 tilelang 模板 `src/`**（共享盘 `3rdparty` 符号链接被 `cp -aL` 实体化 → `TILELANG_TEMPLATE_PATH=None` → Ascend 后端 `"-I"+None`）：`make_portable_kit.sh` 补拷 `3rdparty/../src`（8.8 MB）后，`--layers 25` **24 ✅ / 0 ❌**，与正常环境逐项一致。完整包（含 12G CANN overlay）压缩后 **5.2 GB**。
- [x] **构建器/脚本环境显式化**（2026-10-22，用户口径）：交互 shell 残留的 `DS41M_PD_MERGE=1` 会让「沿用 prefill 权重」的 launch 偷偷走 P→D（清 raw 后重读 34 GB/rank，实测 28 s）。构建器生成的 launch/hit/real/points/resident/一键脚本现在**先 `unset` 本次未显式 export 的托管变量**（P→D、MoE 缓存/释放、replay/图、skip、serve 开关等 37 个），再由 export/extraEnv 覆盖；`run_hit.sh`/`run_real.sh` 的清理循环也补了 6 个 P→D 变量。jsdom 回归 **567 全绿**。
- [x] **ArgSort AiCPU 告警**（2026-10-22）：`_mqsmla` 小 T（DSpark verify T=6）的索引稳定重排
      用 int64 `argsort` → torch_npu 回退 AiCPU；key 改 float32（0..2K 精确）走 AiCore，
      CPU 三组形状对拍逐位相同；NPU 隔离复现 int64 有告警、float32 无。
      详见 `NOTES/2026-10-22-argsort-aicpu.md`。
- [x] **构建器 2026-10-22 增补**：设置面板（CANN/runtime/TMPDIR/输出根/trace 根/真权重，持久化+导入导出）；
      prompt 模板 raw/chat/chat+thinking（默认 chat，生成 `--chat`/`--thinking thinking`）；
      点位传递方式默认改为内联简写（`--points-spec`，多批自动回退内联 JSON）；
      真权重目录可从设置面板统一指定（launch `--weights-path` + real `--weights-path`）。
      自动化 **561×3 + 90** 全绿。
- [x] **多点单卡**（用户口径）：`run_hit.sh` 的 `--cards` 个数 → `--world`（1..8），
      `--points` / 逐 L 都支持单卡；trace 核对 `want=world`（单卡 1 份）。
      实测 2 点 + `DS41M_SKIP_MOE/ENGRAM=1`：rc=0，每 pass 70.1 / 69.2 ms；
      prof 轮单点单卡：trace 1/1 ✓、rc=0；
      构建器多点页加「单卡 · 跳 MoE+engram」预设（卡号=0 + 双开关，点位表不变），
      非 8 卡给未验证提醒、单卡未跳 engram 直接报错。
- [x] **P→D deepgemm 复验**（用户问「两套 MoE 权重不同，转换有重新校验吗」）：
      P→D 只读 EP raw（两个后端的融合权重都是旁路副本，不改 raw）；新增
      `DS41M_PD_VERIFY=1`（逐层 all2all 发送/接收字节和校验），并把 DG 的 raw 保留开关
      写进报错。真权重 15 层 A/B：all2all **1.3s** vs 读盘 TP **15.0s**，4 步 decode
      id/top3 **逐位一致**，15/15 层校验通过。详见 `NOTES/2026-10-22-pd-deepgemm-verify.md`。
      坑：合成权重不能做此 A/B（EP 48 专家 vs TP 384 专家，RNG 消耗不同 → 同专家值不同）。
- [~] **全按 DeepInfer 口径**（2026-10-22，不改结构）：
      - `KEEP_TOKENS` = **固定 140**（V4.1 统一形状，不随草稿长度动态）；`swa_block` =
        **固定 134**（window+6）；**旧 128/`ALLOW_SHORT` 已删除**；
      - 命中输入改 **`hit.non_cached_tokens` 直接给 M（无 128 对齐）**，`hit.ratio` 仅换算
        `round((1-h)L)`；命中率 = `cached/(cached+M)`（DeepInfer `prefill_caching_rate`）；
      - `t_main`/`replay_cut_w`/`build_stack`/runner/points_runner 全部走
        `hit_m` + `replay_tail_of`；`_kv_epilog` 窗口池写数 = `win_keep=window+回退`；
      - `run_hit.sh`/`run_real.sh` 写 `non_cached_tokens` 进配置，R 显示 = M+KEEP(140)。
      验证：单测 **23 项 PASS**；合成 hit 冒烟 `T_pass=1164`（M 1024+140）、W=140、rc=0
      （`/mnt/xds/sfs/l00972766/tmp/replay_keep140_hit_smoke/stdout.log`）；
      真权重 16k keep=140 ✅：8 rank id=396、R=652、稳态 99 ms、
      `prefill_caching_rate 96.88% (cached=15872 batched=512)`、rc=0
      （`/mnt/xds/sfs/l00972766/tmp/replay_keep140_realhit_smoke/`）。
      待办：旧 hit 矩阵按新口径重采；packed 多请求 real-prefix（fork）暂不做。
- [~] **多 pass trace + graph 前后空泡（2026-10-22 用户发现）**：
      ① trace 里只有 1 个 pass（`--iters 1` 采的）；要 MindStudio 看到**多 pass** →
         用 `--iters 3/4` 重采（graph 开，prof 轮现默认入图）。
      ② `traces4_20261022/(262144_0.97)_r1` 里有 **220.8ms**
         `aclrtSynchronizeDeviceWithTimeout`（另有 24.2ms 一条），尾图与前面计算之间有空泡。
         初判：`_GT_SYNC` 默认 1 → replay 前后各一次 `torch.npu.synchronize()`：
         前一次等 eager 前缀排空（220ms 是前缀执行时间，host 阻塞），后一次等图执行完；
         host 在 sync 返回后才 enqueue 图/下一 pass → 边界出现设备空泡
         （与 points_runner "pass 之间不排空"的流水口径相悖）。
      ✅ 结论（2026-10-22）：torch_npu 源码 `NPUGraph::replay()` =
         `AclmdlRIExecuteAsync(model_ri_, c10_npu::getCurrentNPUStream())` → **replay 与前后
         拷贝/算子同流有序**，两段 sync 都不是正确性必需。
         **noprof 默认 PRE=0/POST=0**（稳态口径）：64K@97% `replay#1/2/3 max|Δ|=0`，
         `iters=6` 稳态 **186.1（仅 POST）→ 142.3 ms/pass**。
         **profiler 轮按 T 自适应**（2026-10-22 用户追问"为什么还在"后改）：
         `DS41M_GRAPH_TAIL_SYNC_T` 默认 **8192**，只有 `T > 8192` 才默认 PRE=1/POST=1。
         干净卡实测：64K@90/95/96/97（T=6694/3418/2762/2106）关 sync 全部稳定，
         64K@97% trace 只剩 1 条 27.77ms profiler 收尾（Computing 107ms/pass）；
         64K@80%（T=13247）关 sync 报 507011/超时、64K@60% 直接 barrier+segfault。
         `SYNC_T=0` 回"prof 全开"旧行为；`DS41M_GRAPH_TAIL_SYNC=1` 全开；
         `_PRE`/`_POST` 可单独覆盖。
         **三档无空泡/稳定实现（2026-10-22）**：
         · `DS41M_GRAPH_TAIL_REPLAY_MODE=stream|event`：图 replay 放专用侧流，
           `side.wait_stream(main) → replay → main.wait_stream(side)`，host 不阻塞，
           默认不再加 PRE/POST host sync（`SYNC=1` 可叠加）；
         · profiler 流分离即上面这条（图固定侧流，与 profiler 采集流用 event 排序）；
         · `DS41M_GRAPH_TAIL_PROF_EAGER_T=N`：profiler 轮 `T>N` 直接退 eager trace
           （无图 replay、无 sync）。
         默认仍是自适应 host sync；三条 opt-in，待 64K@80% 验证。
      ✅ 多 pass trace：
         · 64K@97% 图 3-pass（sync 关）：`MODEL_WAIT_COMPLETE`=3、无 220ms 长 sync，
           仅 1 条 43.97ms profiler 收尾：
           `tmp/boundary_split/traces_mp_64k97/(65536_0.97)_r1/prefill/.../trace_view.json`；
         · 256K@97% **eager** 3-pass（`--no-graph-tail`）：`engram_fetch`=12（3 pass）：
           `tmp/boundary_split/traces_eager_256k97/(262144_0.97)_r1/prefill/.../trace_view.json`。
      ⚠️ profiler+图在 **256K@97%（R=8004）**（1/3 pass、sync 开/关）都 device error/挂起，
         **64K@60%（R=26354）** OOM/aicore 异常（eager 3-pass 也异常）；
         同点 noprof 图跑正常 → 稳态报数用图口径（noprof），trace 大 R 点用 1-pass/eager。
         prof+图的大形状稳定性列为已知限制（NOTES 已归档）。
      ★ 多 pass 时延口径（2026-10-22 用户要求）：**N 个 pass 的总时间 ÷ N**
         （不是单 pass、不是逐 pass 中位数）。`tools/trace_multipass_time.py <trace>` 直接算：
         · 64K@97% 图 3-pass：Computing 542.52ms/3=**180.84**、HcPre 492.29/3=164.10
           （含 profiler 开销；稳态报数用 noprof：6 pass 853.8ms → **142.3 ms/pass**）；
         · 256K@97% eager 3-pass：Computing 1084.17/3=**361.39**、HcPre 1076.27/3=358.76。
- [x] **图跨点位缓存/预捕获（2026-10-22 尝试，失败，已回退）**：
      现状（回退后）：**每个 pass 不重捕**（同点位 warmup pass1 eager 建 meta、
      pass2 捕获，之后稳态只 replay）；**每个点位重捕一次**（28 点=28 次），
      捕获在 warmup 内，稳态报数不受影响；`points_runner._reset_between_points`
      恢复清 `_GRAPH_TAIL`/`_GT_POOL_CACHE`。
      尝试过：同 L 跨 hit 保留图/池复用（cut 后 `pos_offset=L-512` 与 hit 无关，
      key 相同）。结果：第一点捕获/重放正常（`max|Δ|=0`），**第二点 replay 稳定报
      `device error 507011`，rank0 HCCL watchdog 终止**；split 图和 tail 图
      (`DS41M_BOUNDARY_GRAPH=tail`) 都复现。结论：prefill 图带 HCCL 通信和
      有状态压缩池，跨点位复用不安全；要做需改造成 dg_meta 式动态元数据 +
      显式状态重置，收益（省 warmup 捕获）不足以抵消风险。
      保留的低风险修补：`_full_pool` 允许跨点形状变化（更大/更小都拷前 N 行）、
      replay 失败后不再回退捕获第二张图（避免 10GiB OOM）。
- [x] **QSLI 组合缓存 16k 定长环（2026-10-22，用户口径）**：`DS41M_QSLI_CAP`
      **默认 0 = 按 L 开精确镜像**（不做跨 L 复用后改回）；`=16384` 才开环。
      写侧池固定
      `nblk2=1+ceil(cap/16)+1=1026`（1026×16×544≈8.9MB），绝对行先取模再查旋转块表；
      packed 每请求 `Qb=1026`。读侧 `bt_w` 固定=`nblk2`、候选 `cand_i` 同样取模
      （`page=cand//16` 保证不越界）。L≤16k 恒等（16k 官方逐位基线不受影响）；
      64K@97% 单卡 A/B：池 `(4099,16,1,544)→(1027,16,1,544)`，`cand_l/cl` 逐位相同，
      但 QSLI `sparse` 每行 ~15–22/512 个选择变化（前缀候选被环覆盖）⇒ **L>16k
      是性能口径的有界近似，不做数值对拍**。`DS41M_QS_DUMP` 增加候选范围与
      QSLI 输出快照（`qsli_out_*.pt`）。8 卡 A/B 因另一容器作业抢占在层 21
      MegaMoE barrier 死锁（已 kill，卡已清空；单卡 rc=0）。
      详见 `NOTES/2026-10-22-qsli-16k-ring.md`。待办（用户提议，精确方案）：
      候选块号**不能**改紧凑 rank（DSL `count_visible` 用绝对块号×8 做因果），
      应保留绝对块号、把**绝对 page（b//16）并集** gather 到固定紧凑池。
      实测（T=140 行 × 2048 候选，单卡、跳 MoE/engram）：并集 page 数
      64K@97%=145、64K@60%=334、128K@60%=536、256K@97%=191、256K@60%=**947**；
      **物理池 P 是并集 page 数（不是 max_cand_i//16）**，固定 **P=1024 页**
      （≈8.9MB）覆盖本矩阵，保守可 2048 页；**page table 宽度**才是
      max_page+1=L/128，固定 **2048** 宽（DSL 快路径上限，覆盖 256K）。
      层 20 在 tail 图模式是 eager，gather 在图外；图里只读固定池/表 ⇒
      精确且几何固定。低命中并集远大于单行 16k（256K@60%=110k token），
      严格 16k 共享池只能近似。验证：64K@97% 与 cap=0 逐位对拍。
      **16K 位置窗口不可取**：实测 2048 个候选里只有 0–264 个落在最后 16k
      （64K@97% 中位 250、128K@60% 中位 24），丢太多。
      **1M 方案**：按 2048 page = 256K token 分块（1M=4 块），每块只保留本块
      候选并重基、page table 恒 2048 宽走快路径，整块可见用 mask_mode=0、
      含 query 的最后一块用 mask_mode=3，输出加块偏移，`return_value=True`
      做块间 top-k 合并；所有候选保留且 1M 也快路径。单次 8192 宽表 + scalar
      路径是精确但慢的备选。尚未实现。
- [ ] **MoE 后端接线（2026-10-22 发现并修复）**：config `backend.megamoe` 以前
      没接到 `DS41M_MOEMODE`（exec_moe 只看 env，`auto` 在 EP≥2 优先 ascendc）→
      run_hit 多点位 banner 写 deepgemm、实际跑 ascendc。已在 runner.main 对
      **prefill-only** 用 `cfg.backends.megamoe` setdefault；`--moemode` 仍可覆盖。
      decode 路径不受影响（decode_layout 自己设 `DS41M_MOEMODE=decode`）。
      另在 `env_pack.sh` 默认加 `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:4096`。
- [~] **DeepInfer 口径矩阵（层 20 入图后开采）**：L=64K/80K/128K/256K ×
      hit=60/70/80/90/95/96/97%，共 28 点；**DP8 / 每 DP 1 batch**（`batches=1`,
      `batch_mode=auto`）；每点 `M=round((1-h)L)`（**无 128 对齐**）、`KEEP=140`、
      `R=M+140`，命中率按 `cached/(cached+M)` 记录。先冒烟 64K@0.90（split graph
      已入图：replay vs eager max|Δ|=0），再 `--points` 多点扩面；
      steady ≥3 样本取中位、prof 按单 pass。旧矩阵数据不可混比。
      **2026-10-22 试跑（QSLI 16k 环 + 图 + deepgemm）**：8 卡 64K@97% 图冒烟
      成功（125.1 ms/pass，1 pass，noprof）；28 点 `--noprof-only --iters 3`
      在 point 1（64K@60%，T=26354）失败：rank3 `engram_fetch_wait` URMA
      batch drain ret=255 + 全 rank MegaMoE barrier timeout。npu-smi 同时看到
      另一容器 8 进程（~24–29 GB/card）⇒ 又是跨容器干扰；已停，等卡空再重跑。
      日志：`tmp/matrix28_qsliring_1022/driver_console.log`。
- [x] **层 20 边界拆分 + 入图**：✅ 入图完成、**默认开**（`DS41M_BOUNDARY_SPLIT=1`，`=0` A/B）：
      `c1a_source_split` = `wkv/kv_norm/rope/kv_epilog/kv_compress/idx_k_write`（KV 全长）
      → `boundary_cut`（截到 **512**：`npu_quant_matmul` 的 M 分块安全值 +
      win scratch rebase）→ `wq_a/q_norm/wq_b/rope/idx_prolog/qli`（compact，
      K 池只读）→ `_attn_tail` + MoE/MLP（512 行）；层 21 入口 replay 再裁到 140。
      `npu_hc_pre_v2` 的 post/comb 按 M 分块 → `_hc_pre` 零补到 8192 再取尾。
      实测（L=16384、单卡）：
        · 跳 MoE/engram `--layers 21`：层 20 全部 digest + id=87166 逐位相同；
        · MoE(deepgemm)、engram 关 `--layers 21`：含 `mega_moe` 全部 digest 逐位相同；
        · 40 层 MoE+engram 全开：L39 最终 `hc_post` digest `fe930b9793cc`/`315ceef164e5`
          与未拆分基线逐位相同，末 token id=109526、top3 相同；
        · hit stage A/B（L=16384/M=512/R=652、layers 21）：stage B 层 20 **全部
          digest 逐位相同（含 `idx_w`）**；stage A 仅中间量 `idx_w` ~3.6e-6 不同，
          `idx_sparse`/最终输出不变（见 `NOTES/2026-10-22-qmm-shape-sensitivity.md`）。
      日志：`tmp/boundary_split/{pad_split,moe_base,moe_split,L40_base_d,L40_official,
      hitB_base,hitB_split}/all.log`。
      ✅ **入图（2026-10-22）**：`DS41M_GRAPH_TAIL` 开时图从层 20 `boundary_cut` 起，
      覆盖 compact 段 + 21..39（key=`split20`；输入 x/residual/hc_pre_t/hc_post_t，
      池走 `_GT_POOL_CACHE`，hc 补零缓冲模块级常驻）。图 replay vs eager `max|Δ|=0`
      （25/40 层、2 点位多点路径）；16k hit60 单卡 40 层 0.342→0.334 s/pass。
      默认开，PERF_RULES §3 例外已移除、§2 已登记。
      ⚠️ **数值口径边界**：自然路由（给 ids）下与未拆分基线逐位一致；合成假前缀
      benchmark（无 ids → `force_eplb` 强均路由）的 compact MoE 在 512 token 上重新
      均衡，`topk_idx`/`moe_out` 与未拆分的 R-token 路由不同（确定性差异，DeepInfer
      同口径）→ 性能矩阵不与旧未拆分数据做数值对拍。
      待办：① 28 点矩阵（用户已定：先入图再跑）；② EP8 真权重对拍（单卡 deepgemm 已验）。
- [x] **构建器迁移路径适配**（2026-10-22）：HTML 从自身 URL 自动检测 `…/ds41_mini`；
      旧共享盘默认 workdir/TMPDIR/outRoot 在迁移后自动跟随（`<新包>`、`<新包>/.tmp`、
      `<新包>/_out`）；TMPDIR 先 `mkdir -p`；自检命令改 `source env_pack.sh`；
      设置面板加「用当前目录」。jsdom **567 全绿**；迁移模拟输出
      `workdir=/data/kit/ds41_mini`、`outRoot=/data/kit/ds41_mini/_out`。
      详见 `NOTES/2026-10-22-builder-portable-paths.md`。
- [ ] **28 点矩阵（2026-10-23 进行中）**：64K 7 点已出（图开，Computing/4 pass：
      60% 1007.30 / 70% 745.70 / 80% 502.08 / 90% 266.69 / 95% 140.42 / 96% 121.01 /
      97% 102.04 ms）。80K@60%（T=32908）图开时 graph capture 完成后 steady pass 触发
      MegaMoE rank barrier timeout（signal=7/8）→ 507015 → 被包成
      `c10::OutOfMemoryError`；noprof 首 pass 40 层完整、capture 后 rank0 allocator
      `NPUEvent::query()` segfault。判定为图/allocator 大 T 交互问题，不是纯 HBM 不足。
      按用户指示“入图延后”，改 eager tail（`--no-graph-tail`）+ 任务级 ALOP：
      80K 7 点（60..97）全部通过（8 rank rc=0，trace 有效；Computing/N：60% 1501.37、
      70% 943.96、80% 647.78、90% 335.98、95% 172.08、96% 154.58、97% 154.03 ms）。
      128K@60/70 与 256K@80 均失败；**定量实测根因（2026-10-23 02:40）**：
      不是活张量 OOM（活峰值 19.94 GiB），是 caching allocator 把每层
      q/attn_out（[T,64,512] bf16，T=39462 时 4.82 GiB/层）的 freed blocks
      留成 reserved，逐层 +4.82 GiB；free 到 0.059 GiB 后 `process_events →
      NPUEvent::query()` segfault → MoE barrier timeout → 507015 →
      被包成 c10::OutOfMemoryError。80K@60 每层 4.02 GiB，pass2 L20 也到
      free 4.87 GiB 但塞得下。`DS41M_EMPTY_CACHE_EVERY=1` 实测 reserved
      5.09/free 82.9、rc=0，证明是缓存段。
      ✅ **2026-10-23 04:15：28/28 点位完成**。最后 5 点用
      `EMPTY_CACHE_EVERY=1` + 全 rank profiler 跑通（8 rank rc=0、trace 有效）：
      128K@60 2312.46、128K@70 1824.21、256K@80 2466.36、256K@70 3645.05、
      256K@60 4756.02 ms/forward；64K/80K/128K/256K 各 7/7。
      修复口径含 empty_cache 层间开销，不与标准点严格混比；汇总
      `tmp/MATRIX_28_20261023.tsv`，说明 `NOTES/2026-10-23-28point-matrix.md`。
- [x] **显存优化（2026-10-23）**：clean A/B（256K@60，noprof，empty_cache=1）：
      `DS41M_RELEASE_Q_ATTN` 峰值 48.45→**42.73 GiB**（-5.72，L10 alloc 35.664→22.843，
      free 40.177→52.998）；`NO_RESIDUAL_CLONE` 单独不降峰值但 L20 reserved
      10.068→8.465；`MOE_PREALLOC` 不降峰值（峰值在 mqsmla），暂 opt-in。
      64K@97 三开关全开与基线 packed-digest 逐位一致。
      torchair 无 mqsmla/mega_moe/engram GE converter，compile 整层会 graph break；
      `NPUGraph.update` 是 beta 路径，未验证。详见 `NOTES/2026-10-23-memory-opts.md`。
      注意口径：64K 是图开，80K+ 是 eager tail，不能直接混比。
      详见 `NOTES/2026-10-23-largeT-graph-barrier.md`。

## F. CP 接入（DeepInfer 风格，2026-10-23）

> 目标：**不改 cp1 原有框架/数值**，逐步把 CP 做进来；默认 `DS41M_CP_SIZE=1` 走旧路径。
> 记录：`NOTES/2026-10-23-cp-deepinfer-implementation.md`。

- [x] **Step 0/1 布局层**：`cp_layout.py`（head-tail/mirror、local/reverse index、local_blocks）、
      `cp_runtime.py`（CP 组 + rank-major all-gather）、`cp_state.py`（split/merge/ctx）；
      纯 CPU 13 lens×cp1/2/4/8 + 8 进程 gloo + state 测试全 PASS。
- [x] **Step 2 SWA**（`DS41M_CP_SIZE>1` 才进）：q 本地 + SWA KV all-gather 写 full 池；
      `cp_positions` 逐行全局位置（rope/窗口/槽位）；`_cp_ref_pad`（qmm 小 M 补到参考 M）。
      实测：L=1024 CP1/2/4/8 全 `25787a1402c9726b`；L=8192 CP1 vs CP8 全 `817edfc459a2305f`；
      逐算子 L0..L3 digest 对齐。非整除 L 的 digest 口径用 `CpState.last_local_rows`。
- [x] **Step 3 压缩/indexer**（2026-10-23 上午完成；验证见下方 553 起）：源层
      `kv_compress` gather hidden → 全长复算；`qli`/`qsli` 本地 q 查 full K；
      `mqsmla` 开 cmp（逐行全局 `cl`）。
- [x] **Step 4 层 20 边界**（F2，2026-10-23 下午）：CP compact-mirror 跨 rank 重分布；
      L=8192 compact CP1/2/4/8 全 `64e4c52c…`、L=1024 `2afe4dc2…`。
- [ ] **Step 5 MoE/EP、输出、packed 多请求**；之后按 16k/64k → 1M@0（CP4/CP8）采数。
- [x] **Step 3 压缩/indexer ✅（2026-10-23 上午）**：源层 `kv_compress` gather hidden → 全长复算；
      `qli`/`qsli` gather Q 侧全长 → 标准路径 → 切回本地；`mqsmla` 开 cmp（逐行全局 `cl`）；
      `run_op` 在 CP 下不取 @packed（否则 cmp 越界 507015）；`_idx_prolog` 参考 M 零补（w 分块）。
      对拍：L=1024 cmp 开 CP1=CP8 `02487e95…`；L=8192 CP1=CP8 `e09aa8d8…`。

## F2. CP 对齐 DeepInfer + KV 管理（2026-10-23 下午）

- [x] **索引器对齐**：`_qli/_qsli` CP 改为**本地 q 查 full K**（按 head/tail 块），
      不再 gather Q；`residual_k` 处理非对齐块（L=1000 验证）。
- [x] **层 20 边界 + 后 20 层截尾（YOCO compaction）**：`cp_layout.boundary_indices`
      （DeepInfer ref 逐字移植）+ `_boundary_cut_state_cp`（512 cut + 跨 rank 重分布）+
      `_cp_replay_trim`（512→140）+ `CpState.pos_offset/seq_len/abs_positions`。
      验证：L=8192 compact CP1/2/4/8 全 `64e4c52c…`；L=1024 compact CP1=CP8 `2afe4dc2…`。
- [x] **踩坑**：mqsmla 的 DSpark 小 T（≤32）"未来行"重排把 `ci<cl` 当上界，
      compact-mirror 后每 rank 18 行误触发 → CP 下跳过；QLI 的 `_per_row` 同样在
      CP 分块禁用。
- [x] **KV 管理系统**：`kv_manager.py`（KVGeometry/KVLayout/KVCacheManager/BlockManager，
      照 DeepInfer `cache_engine.py` 结构）+ `tools/test_kv_manager.py`（槽位/块表与
      inline 逐位一致）+ 执行路径接入（`DS41M_KV_MANAGER`，默认开；A/B 逐位相同）。
- [ ] 压缩输入仍 gather hidden（DeepInfer 是 gather projected KV，需另一套压缩 kernel）；
      1M 前评估是否值得换（通信 2.5–10× vs 数值不再逐位）。
- [x] **KV 管理接入 A/B**：`DS41M_KV_MANAGER=1/0` digest 逐位一致（8192 compact
      `64e4c52c…`、1024 full `02487e95…`）；`tools/test_kv_manager.py` 增加池形状自证
      （idx_k 必须 4D/5D——踩过 2D 分配导致 K 缓存整段错）。
- [x] **compact+pad 收尾**：L=1000（奇数块 + pad + 截尾）CP1=CP8 `5d7e6ce7…`。
- [x] **压缩 gather 占比实测（64K CP8）**：steady pass 合计 62.3ms（SWA 37.4 + hidden
      13.8 + 小张量 11.0），占层耗时 17.3s 的 **0.36%** ⇒ projected-KV 替换收益可忽略，
      不默认切；1M 外推 hidden gather ~0.2s 量级。
- [x] **DeepInfer 压缩 kernel 备选**：`thirdparty/` vendor（Apache-2.0）+ 加载器 +
      `DS41M_CP_CMP_KERNEL=deepinfer`（本地投影→gather projected→kernel→复用池写）。
      全 prefill/compact 两种形态都跑通；默认仍厂商 op（逐位）。
- [x] 256K/1M CP8 规模验证（方法/证据见
      `NOTES/2026-10-24-cp-scale-validation.md`）：
      · CP1@256K full 单卡 OOM（88.12/91.8，L0 mqsmla 32GiB）⇒ 不作参考；
      · 跨 cp 尺寸用 **skip MoE/engram + 最终全局末行** 口径；
        8192 `e09aa8d88940`、256K full `d38cdf064b7a`、256K compact
        `d605d9cf2019`，CP4w8=CP8w8 均 **max|Δ|=0**；
      · 全模型 256K CP8 full 32.28 GiB / 58.7s、compact rc=0；
      · **1M CP8**：skip full 81.08 GiB / 35.0s、全模型 compact 79.02 GiB /
        73.4s，均 rc=0；需 `DS41M_CP_AG_CHUNK=2048` +
        RELEASE_Q_ATTN/NO_RESIDUAL_CLONE/EMPTY_CACHE_EVERY；
      · 1M 全模型 full 未跑（预计顶墙；生产 replay 默认开）。
- [x] **并行口径修正（2026-10-25，用户指出）**：固定 8 卡 EP8，CP 尺寸决定
      DP=world/cp（CP8=dp1、CP4=dp2、CP2=dp4）。此前 CP4 用 world4（EP4）
      的数字作废。`DS41M_CP_DP_INPUT=1` 给每个 DP 组不同合成 seed。
      · CP4 模块 A/B（world8/cp4）：1024 full/compact、4096 full digest
        101/103/101 全一致，mod 40 行 order_ok；
      · CP2 256K 全模型 full/compact：52.03 GiB，58.8/53.4s，rc=0；
      · CP4 256K 全模型 full/compact：**31.43 GiB**，55.8/51.8s，rc=0；
      · **CP4 1M 全模型 compact：不可行**（87.37 GiB 仍 OOM：层1 wq_b 16GiB、
        engram 侧流 12.5GiB）⇒ 1M 全模型用 CP8（79.02 GiB 已通过）。

## G. 模块运行时 Phase 1.5 A/B（2026-10-24）

> 探针 `DS41M_CTX_TRACE=1` 只包 ctx/st（dict 子类），不改任何算子；闸门 = 逐算子 digest 与基线逐位一致。
> 日志/产物：`/mnt/xds/sfs/l00972766/tmp/module_ab_20261024/`（SUMMARY.txt 汇总）。
> **2026-10-24 06:58 起跑前空气检查：8 卡被兄弟容器 `VLLM::Worker_DP` 占满**
> （各 ~87 GB / HBM 92 GB·98 GB，另一 PID namespace）⇒ 按纪律**未抢卡、未启动**；
> 驱动脚本 `run_ab.sh` 已就绪（`bash -n` 通过）；`wait_and_run.sh` 后台守候
> （复用 `tools/wait_npu_idle.sh`：连续 3 次 ≤50 MB 抖动、最高卡 <8000 MB，24h 上限），
> 卡回基线后自动起 A/B。CPU 侧门禁已先跑：`check_module_spec` PASS、viewer jsdom **105 passed**。

- [x] full（`DS41M_DECODER_REPLAY=0 DS41M_BOUNDARY_SPLIT=0`）× base/probe @seq1024
- [x] compact（两者=1）× base/probe @seq1024
- [x] full × base/probe @seq8192
- [x] compact × base/probe @seq8192
- [x] `tools/diff_digest.py` 全绿（4 组：101/103/101/103 条逐字节一致，8 run 全 rc=0）
- [x] `python3.11 -m modules.runtime_probe ctx_trace_*.json` 得首版清单
      —— **但首版归因被污染**（诊断代码的读被算到算子头上，见
      `NOTES/2026-10-24-module-runtime-ab.md`）；已修 runner 的 set_current 作用域
- [x] 第二轮（07:29–07:35）：归因修复后重跑，digest 仍 4 组全绿
      （101/103/101/103），报告干净到只剩两类噪声：
      ① `batch.cu/n_seqs` = run_op 的 packed 分派读；② `.get(miss)` / `in` 被记成真实读
- [x] 探针 v2（CPU 门禁已过）：
      · `exec_ops.run_op` 内注入 `PROBE_SET`，作用域只包实现体（分派读不再算给 op）；
      · `runtime_probe.get` 未命中记 `probe`（optional），`in` 也记 probe，
        compare 新增 `probe_not_declared`，不再把 `st.get("k")` 当依赖；
      · `names.LEGACY` 补 `idx_sparse/suffix_*/idx_k_ready/moe_nspan/candidate_*/engram_*`；
      · 重新导出 viewer（757 KiB，`--check` OK，jsdom 105 全过）
- [x] 第三轮 probe（13:42–13:46）：v2 下 digest 仍 **4 组全绿**（101/103/101/103）；
      报告剩 15 个 op 的真读漏（batch.* 分派读与 get-miss 已消失）
- [x] 按干净清单修 `modules/op_iface.py`（15 个接口）+ `runner_iface` 常量
      （candidate/engram 几何）+ `boundary_cut` 的 in_place provides；门禁全过：
      `check_module_spec` PASS、viewer 858 KiB `--check OK`、jsdom 105、
      `test_module_runtime` 5901
- [x] Phase 2 NPU A/B（base vs `DS41M_MODULE_RUNTIME=1`）：已完成，见 H 段
      （plain 4/4、hit/packed 4/4、CP2 3/3、pad 3/3；默认开已落地）

## H. 模块运行时 Phase 2 骨架（2026-10-24，CPU-only）

> 默认关；只换"执行序来源"，叶子仍 `exec_ops.run_op`；DSpark/decode 强制 legacy。

- [x] `modules/runtime.py`：`build_plan / verify_order / ops_of / walk / signature`
      （`walk` 预留容器 enter/exit 钩子；树 DFS 序 == 执行序）
- [x] runner 接入：`DS41M_MODULE_RUNTIME=1`；首次与 legacy `expand_layer` 严格对拍，
      不一致默认抛错，`DS41M_MODULE_RUNTIME_FALLBACK=1` 回退；`_MODULE_PLAN_CACHE`
      随 `_SEQ_CACHE` 一起清
- [x] `tools/test_module_runtime.py`：六口径（default/pad/skip_moe/skip_engram/unfused/split）
      → **5901 passed / 0 failed**
- [x] `check_module_spec.py` 新增 `check_runtime_plan`（default + split），全配置 PASS
- [x] `MODULE_DEV.md` §6 补充 Phase 2 用法与门禁
- [x] NPU digest A/B：base vs `DS41M_MODULE_RUNTIME=1`（full/compact、1024/8192）
      **4/4 组逐字节一致**（101/103/101/103）；mod 组每次 40 行 `order=legacy-ok`、
      0 回退；8 run rc=0（2026-10-24 13:55–14:01）
      驱动 `/mnt/xds/sfs/l00972766/tmp/module_phase2_ab_20261024/run_phase2_ab.sh`
- [x] 闸门修正：`is_decode_step` 在 hit stage B 也因 `pos_offset>0` 为真 →
      改为 `hit_stage 非空 或 meta["hit"]>0` 放行；decode/DSpark 仍 legacy
- [x] hit/packed 口径 A/B（round 2，2026-10-24 14:26–14:32）：**4/4 组全绿**
      （hit_b1_full 101、hit_b1_compact 103、hit_b2_full 101、packed_b2_full 101）；
      每次 40 行 `order=legacy-ok`、0 回退；8 run rc=0
      驱动 `/mnt/xds/sfs/l00972766/tmp/module_phase2_ab2/run_phase2_ab2.sh`
      （单卡 hit 配置用临时副本：engram_tp=1/offload=false，不改仓库配置）
- [x] CP 口径 A/B（round 3，2026-10-24 14:32–14:38，2 rank/2 卡）：**3/3 组全绿**
      （CP2 seq1024 full 101、compact 103、seq4096 full 101）；每次 40 行
      `order=legacy-ok`、0 回退；6 run rc=0
      驱动 `/mnt/xds/sfs/l00972766/tmp/module_phase2_cp_ab/run_cp_ab.sh`
- [x] pad-ops 口径 A/B（round 4，2026-10-24 14:39–14:43）：**3/3 组全绿**
      （pad_1024_full 121、compact 123、pad_b2_full 121 行）；40 行
      `order=legacy-ok`、0 回退；6 run rc=0
      驱动 `module_phase2_pad_ab/run_pad_ab.sh`
- [x] decode 负向检查（14:44–14:50）：
      · `--decode-mode full`：每步就是 prefill（`pos_offset=None`），模块树**合法介入**
        （计划已缓存，不再打印）；token/top3 与 base 逐位一致；
      · `--decode-mode inc`：初始 prompt prefill 走模块树（40 行），
        **增量步（`pos_offset>0`）之后 0 行** ⇒ legacy 生效；token 与 base 逐位一致
- [x] 计划构建成本：40 层 `build_plan` 合计 **3.24 ms（0.081 ms/层）**，每层只建一次
- [x] 性能口径（8192，noprof）：full 659/658 ms、compact 447/446 ms（base/mod，
      稳态 3 轮中位）⇒ 无性能回归
- [x] 闸门逻辑单测：plain prefill/hit → True；inc decode/DSpark → False
- [x] 用户拍板（2026-10-24）：**默认开**；`DS41M_MODULE_RUNTIME=0` 一键回退；
      PERF_RULES §0/§2/§8 已登记；`check_perf_defaults.py` 增加静态+运行时校验
      （`=0` 无 `DS41_ALLOW_SUBOPTIMAL` → rc=2，已实测）
- [x] 入口/构建器显式化：`run_hit.sh`/`run_real.sh` 导出
      `DS41M_MODULE_RUNTIME=${...:-1}`；builder `STRICT_ENV` 纳入
      `DS41M_MODULE_RUNTIME(_FALLBACK)` + 重新生成布局变体；
      jsdom **567×3 + 90 全绿**
- [x] “默认 vs `=0`”回归（prefill8192）：full 101 行一致；compact 的 off 臂
      出现 `cmp_kv` 异常（13 处，均同源）→ 查实为**兄弟作业争用期环境非确定**
      （`NOTES/2026-10-24-compact-cmpkv-contention.md`）：空闲下 legacy×3、
      default×3、换分配器×1 共 7 次 digest 全同（`7b1cf254b900`/`0782ef71e03a`）
- [ ] 剩余默认 vs `=0` 回归（hit64k compact / packed b2 / CP2）：
      确定性对照已覆盖 compact；这三组待卡空重跑（驱动已加空气检查/rc=3 重试）

## I. 模块/参数/装配配置化（2026-10-25 用户要求）

> 目标：开发者改配置文件，不改 Python 声明；父节点不汇总子节点 I/O，
> 子节点缺键沿祖先作用域解析。记录：`NOTES/2026-10-25-module-config-driven.md`。

- [x] `modules/specs/model_modules.yaml`：41 个模块接口 + 16 个模块的 `params`
      （字面量 / `"${祖先键}"` 引用）
- [x] `modules/specs/recipe_map.yaml`：18 容器 + 72 条路径映射
- [x] 加载器：`spec_loader.py`（严格校验）、`op_iface.py`/`recipe_map.py`
      改为 YAML-first，缺失才回退 `*_builtin.py`；`DS41M_MODULE_SPEC` /
      `DS41M_RECIPE_MAP` 可覆盖
- [x] 运行时参数：`runtime.inject_params` 执行前注入、执行后撤销；
      运行配置优先；缺引用报错
- [x] 祖先解析：`modules/context.Scope` 链（容器不需要汇总子节点键）；
      `test_module_config` 覆盖
- [x] 构建器/积木库改为生成 YAML 配置（Python 声明保留为旧入口）
- [x] `modules/specs/modules.d/*.yaml` 小文件（单模块可省略 schema；
      同名 op 需 `override: true`；`DS41M_MODULE_SPEC_DIR` 可覆盖）
- [x] `check_module_spec` 增加 `check_config` 不变量：配置来源存在、impl/标志、
      容器登记、recipe_map 引用、params 真实 ctx 可解析
- [x] `tools/new_module.py` 默认生成 `modules.d/<op>.yaml`（`--python-stub` 保留旧路径）
- [x] `modules/specs/key_names.yaml`：94 别名 + 12 多义 + 17 分级；
      `names.py` 改为 YAML-first（`DS41M_KEY_NAMES` 覆盖，缺失回退 `names_builtin.py`）
- [x] 门禁：`test_module_config` 40 项（env 替换/无硬编码/生成器/键名配置）、
      全配置 × 全层 **8608 次参数引用 0 unresolved**、`check_module_spec` PASS、
      viewer 861 KiB `--check OK`、jsdom 105
- [x] NPU 跨口径回归（与配置化前基线逐字节一致）：8192 full 101 行（复测两次）、
      hit97 64k compact 103 行、packed b2 101 行、CP2 1024 full 101 行
- [ ] 可选下一步：把 `context.Ctx` 接入模块执行（当前是扁平 ctx + 参数注入）；
      `names.LEGACY` 配置化

## J. CP2/4/8 prof 空泡检查（2026-10-25 用户要求）

> 目标：CP2/CP4/CP8 各开 prof（rank0），查 trace 是否有大空泡；解释原因并尝试消除。
> 口径：world8/EP8，CP 决定 DP；L=65536 plain prefill 全模型；replay=0/boundary=0
> （eager 全序列，CP 结构最完整）；`--prof-ranks 0 --warmup-iters 2 --iters 3`
> （稳态取最后一个 forward）。驱动 `tmp/cp_bubble/run_cp_bubble.sh`，
> 汇总 `tmp/cp_bubble/SUMMARY.txt`，分析器 `tools/trace_gaps.py`。

- [x] 分析器 `tools/trace_gaps.py`：HcPre=80/forward 切窗；设备忙/空泡、
      通信暴露（只有通信在跑）、compute 空闲三分解、每层分摊、host 热点
      （旧 64K@97% trace 冒烟通过）
- [ ] CP2/CP4/CP8 prof 采集（09:51 起）
- [ ] 空泡归因（代码位置 + trace 证据）与消除 A/B
- [x] 首轮 prof（64K，CP2/4/8）：空泡定量（forward 1 稳态）
      CP2 2.62s / idle 101ms / comm 暴露 33.8ms；CP4 1.36s / 75.9ms / 39.5ms；
      CP8 0.748s / 76.5ms / 33.3ms。最大空泡 = forward 边界 18–21ms + 4 个压缩层
      各 ~2.1ms（gather 前，host 造索引）+ 层 1 的 50ms all-gather（CP2）
- [x] 归因 A/B：
      · 层 1 大 gather 由 **engram 取行/解量化**引起（NO_ENGRAM：all-gather 合计
        85.7→35.1ms，最大 50.1→5.2ms；vendor0 主机取表路径同样有 53.4ms ⇒ 非
        ElasticBuffer 特有）
      · 每 forward 39 次 ≥10ms `aten::to`（20–115ms）＝ `torch.tensor(inverse_reverse)`
        的阻塞 H2D（cp_runtime.py 旧实现每次 all-gather 都建），是层内空泡主因
      · forward 边界 14ms 无 host aten = `build_cp_plan` 纯 Python（实测 15.5ms/次）
- [x] 修复：`_cp_state_cached`（按几何缓存 CpState）、`plan._rt`（索引张量缓存）、
      无 pad 一律走 scatter 路径（省 cat+index_select 与 reverse 索引 H2D）；
      另修 `inverse_reverse` pad 应为 -1（旧实现 0 → `_inv_ok` 失效）
- [x] **发现并修复新空泡/回归**：去掉逐层阻塞 H2D 后 host 无限抢跑 →
      MegaMoE 对称缓冲 barrier 跨 rank epoch 错位（CP2 8 forward 内可崩：
      barrier timeout epoch=960 signal=6 target=8 → HCCL watchdog；不崩时 2.60→3.11s）。
      修法 `DS41M_RUNAHEAD_LAYERS=2`（事件环，host 只领先设备 2 层）→
      2.542s（比旧 2.604 快 2.4%），CP2 final-row 与旧码**逐位一致**（452f1d2201bb）
- [ ] fixed 三点（CP2/4/8）无 prof + prof 复测 + NOTES 归档
- [x] 第二轮（含 `split_local` 的 `bool(valid.all())` 同步修复）：
      no-prof CP2/CP4/CP8 = 2.538 / 1.280 / 0.664s（基线 2.604/1.349/0.744；
      -2.5% / -5.1% / -10.8%），final-row 与基线三组全部逐位一致
      （452f1d2201bb / bd157f58cf4f / f888bc1cf141）
- [x] 修复后 prof（forward 1）：空泡 101.1/75.9/76.5ms → **12.6/12.4/12.3ms**
      （0.5%/1.0%/1.8%）；>5ms 空泡 7/2/2 → 1/2/2；最大 21.1/19.8/18.3 → 6.1/5.8/5.7ms
- [x] engram 归因（NO_ENGRAM / VENDOR=0 A/B）：layer 1 的 50ms all-gather 由 engram
      取行+解量化引起（非 ElasticBuffer 特有）；未消除，属 engram 调度问题
- [x] 归档：`NOTES/2026-10-25-cp-bubble-prof.md`
- [x] 第三轮（`split_local` 修复后）最终 prof：空泡 **6.6 / 6.1 / 5.8ms**
      （0.3%/0.5%/0.9%），唯一 >1ms 空泡 = forward 末尾 `torch.empty(671MB)`+Python
      ~6ms；`aten::to` 2047/845/263 → 7/4/3ms（0 次 ≥10ms）
- [x] PERF_RULES 登记 `DS41M_RUNAHEAD_LAYERS=2`（§2 表 + §8 决策日志）+
      `check_perf_defaults.py` 静态/运行时门禁
- [x] 稳定性：CP8 20 forward（0.663–0.670s/中位 0.666）、CP2 12 forward
      （2.520–2.554s/中位 2.534），final-row 与基线逐位一致；PERF_RULES 门禁 rc=0
- [x] 抢跑门禁改成运行时 `CP_STATE is not None`（只有 CP 预填受影响）

## J2. CP 压缩层 hidden gather 侧流预取（2026-10-25 下午）

- [x] 实现：`runner._cp_maybe_prefetch_cmp` —— attn rmsnorm 之后在侧流发起
      `kv_compress` 需要的全序列 hidden all-gather（同一份 gather，提前）；
      `exec_ops._kv_compress` 消费时 `wait_event` + `record_stream`；x 身份/CP
      状态不一致自动回退正常路径。显存上限 `DS41M_CP_CMP_PREFETCH_MAX_GB=3`
      （256K 可、1M 自动关）；`DS41M_CP_CMP_PREFETCH=0` 关闭。
- [x] 无 prof A/B（iters=6 中位）：CP2 2.538→**2.532**、CP4 1.280→**1.276**、
      CP8 0.664→**0.660**；final-row 三组与基线逐位一致。
- [x] CP8 prof：通信暴露 30.9→**21.1ms**（压缩层 2.13→0.63ms/层），forward 664→657ms
- [ ] CP2/CP4 prof 复测 + 稳定性（CP8 20 / CP2 12）+ compact/replay A/B + 归档
- [x] CP2 复测发现：预取把 4 个压缩层的暴露（~2.8ms/层）消掉了，但 **L3 的 kv
      gather 出现 14ms 停顿**（跨 rank 等最慢者；单块 gather 仍复现）→ CP2 墙钟仍
      略好（2.508/2.506 vs 2.526 prof），但 collective 级不等长有风险。
      决策：auto 只在 **cp_size≥4** 开（CP4/CP8 实测无停顿）；CP2 保持原路径；
      `DS41M_CP_CMP_PREFETCH=1` 可强制开做实验。
- [ ] 最终三点（单块预取）+ CP8 20/CP2 12 稳定性 + compact/replay A/B
- [x] compact/replay（生产口径）连带修复：CP 预填的 split/tail 图捕获在**旧码**也崩
      （`_cp_replay_trim` 同步 H2D 在 GLOBAL 捕获里非法；此前 iters=1 未暴露）。
      `_gt_base` 增加 `exec_ops.CP_STATE is None` ⇒ CP compact 走 eager replay。
      CP8 64K compact **0.433s**（on）/0.438（off）；CP2 **1.388/1.395s**；
      on/off final-row 逐位一致（2809fd69b300 / d74f674313bf）。
- [x] J2 完成；PERF_RULES 登记 `DS41M_CP_CMP_PREFETCH=auto`；归档包含 §6/§7

## K. 28 点统一口径重测（2026-10-25 16:35 起）

- 口径：8卡 world8/dp8/ep8/cp1、合成、elb、b1/r1；replay+boundary 开；**eager tail**
  （图全关，大 T 必需）；**EMPTY_CACHE_EVERY=1 全点**；ALOP task-only + **全 rank**；
  warmup=2 + iters=4；按 L 分 4 进程。
- 驱动：`tmp/run_matrix28_unified.sh`；输出 `tmp/MATRIX_28_UNIFIED_20261025/`
  （`SUMMARY.txt`、`run_L*.log`、每点 8 rank trace）；汇总脚本
  `tmp/build_matrix28_unified.py` → `MATRIX_28_UNIFIED.tsv`。
- 注意：与旧 28 点（混口径：64K 图开、6 点修复口径）**不可逐点混比**。
- [ ] 跑完 → 汇总 TSV + 28/28 校验（每点 8 rank × 4 forward）→ NOTES/INDEX
- [x] 统一口径 28 点重测完成：4 个 L 进程全部 rc=0（333/252/260/309 s）；
      28 点 × 8 rank = 224 份 trace（16GB）；每点 4 forward（HcPre=320）。
      矩阵 `tmp/MATRIX_28_UNIFIED_20261025/MATRIX_28_UNIFIED.tsv`；
      归档 `NOTES/2026-10-25-matrix28-unified.md` + INDEX。

## L. 借鉴 DeepInfer 内存策略（2026-10-25 18:00 起）

- [x] 已搬：**有界抢跑扩到非 CP 高压前向**（`DS41M_RUNAHEAD_MIN_T=40000`，K=2）；
      **GC**（points_runner `gc.disable()` + 点位边界 `gc.collect()`）；
      **阶段边界 empty_cache**（`_free_device`，逐层默认 0）；
      **CP 固定缓冲池**（`cp_runtime._cached_buf`，kv_compress/kv_epilog，DeepInfer
      userbuffer 式；> `DS41M_CP_BUCKET_GB=4` 不缓存；plan 淘汰时释放）；
      **图捕获 thread_local**（DeepInfer 同款）。
- [x] 逐位校验：CP4@1024 `96e4dca5a15d`、CP8@64K `f888bc1cf141`、CP4@64K
      `bd157f58cf4f` —— 与基线完全一致。
- [x] expandable_segments 实测（DeepInfer 原设置）：256K@80（25.8GiB）+ 无逐层 EC
      **稳定 2197ms**（比 max_split+EC 2499 快 12%）；但 256K@60（48.4GiB）+
      **EC=1 反而 20.3s/forward**（max_split+EC 4.78s）⇒ 会在 MegMoE 重分配下退化。
      **已回退默认到 max_split**，expandable 仅作 opt-in。
- [ ] max_split + 无 EC + 有界抢跑 的承压上限（256K@90/80/60 测试中）→ 决定 EC 政策
- [ ] 未搬：activation/userbuffer 级 workspace 预分配（DeepInfer 能开 expandable 的
      前提）；当前 48GiB 点仍需 EC 阀
- [x] 有界抢跑承压（max_split，无逐层 EC）：256K@90 1071ms、@80 2149ms、@70 3188ms 全稳；
      @60 5876ms 不崩（EC 开 4784）⇒ 设界后 barrier timeout 消失
- [x] expandable 否决：@60 即使 EC=1 也 20.3s/forward；默认保持 max_split（opt-in）
- [x] Phase0 早释放（RELEASE_Q_ATTN/NO_RESIDUAL_CLONE）在 hit/replay L20 崩，未采用
- [x] EC 政策定稿：auto + 阈值 90000；默认验证 @60 4802ms（开）、@70 3191ms（关）
- [x] 非 CP 图（thread_local）冒烟 rc=0（0.330s）
- [x] 团队模式落地：`TEAM/`（16 角色/流水线/笔记区/BOARD/版本管理）；
      首轮：architect 设计、junior DX 报告（5 个 P0）、tooling EC trace 证据

## 2026-10-26 代码收敛（用户指令）— 待办与已完成

- [x] **packed 概念删除**：`exec_ops_packed.py` / `exec_indexer_packed.py` 并入主模块并删除；
      `run_op` 不再查 `@packed`，每个算子只剩一个入口
      （`OPS`: `kv_compress` / `qli` / `qsli` / `idx_k_write`）。
- [x] **默认不入图**：`DS41M_GRAPH_TAIL` / `run_hit.sh GRAPH` 默认 0（显式 `--graph-tail` 才开）；
      PERF_RULES §0 已改写，代价 64K@97 104.1→~158ms 记在规则里。
- [x] **禁用 python_profiling**：四处配置 `enabled: false` + PERF_RULES 规则；
      需要 CSV/trace_view 改用 msprof 导出 `PROF_*`。
- [x] **启动入口收敛为一**：新增 `tools/run.sh` 作为**唯一对外入口**，三种模式：
      `--weights synth`（→ `run_hit.sh`）/ `--weights real`（→ `run_real.sh`）/ `--dry-run --config`
      （→ `run.py`，不占卡）；参数原样透传。
      两个后端脚本与 `launch_mini.py` 保留为内部实现（同一启动链：
      `run.sh → run_hit/run_real → launch_mini.py → runner.py`）。
      `run_hit97*.sh`（指向 run_hit.sh 的符号链接）已删除。
- [x] **`_packed` 读取收敛**：`core/runner.py` 的 12 处 `cfg.get("_packed")` 全部改走
      `metadata.packed_view(cfg)`（唯一读取入口）；写入仍由 `metadata.set_layout/ensure` 同步镜像。
      验证：dry-run + 单点 64k@0.97 noprof 冒烟 rc=0（`tmp/PACKEDVIEW_SMOKE2_0920`）。
      后续可把 packed_view 改为从 `InputMetadata` 现算，然后删掉 `_packed` 这个 key。
- [ ] **CP 合入 varlen（待验证）**：`exec_indexer.py` 的 `_qli_cp` / `_qsli_cp`（本地 head/tail
      两块各跑一次单序列）仍是独立实现，未并入 `_qli_pk` / `_qsli_pk`。
      合并方案见 `CODE_MAP.md §5`：`cu_seqlens=[0,nh,nh+nt]` + 每请求 `seqused_k/residual` + 共享块表。
      **验收**：CP1/CP2/CP4/CP8 在 L=8192 上与原 digest 逐位一致（`tmp/CP_BUCKET_CHECK` 口径），
      再跑一个 CP>1 的 hit 点确认无数值漂移；未验前 `_cp_of` 例外**保留**。

## 2026-10-26 删图尝试（已回滚）与三个真 bug

- [x] 删图代码执行：`core/runner.py` 4919→4132（−787 行）、`points_runner` −38；dry-run/冒烟通过。
- [x] **回滚**：决定性 A/B 显示入图仍值 14%（图 OFF 155.5 / 图 ON 133.7 ms/forward），
      且此前“102.3ms”是缺 `lens` key 导致的假快（L0 mqsmla 失败、部分算子未执行）。
      `core/runner.py` 已从 `tmp/runner_pre_delgraph_0931.py` 恢复；`points_runner` 的点位间图清理已补回。
- [x] 三个真 bug 修复（129 算子错误 → 0）：`metadata.to_packed()` 补 `lens`；
      `ops/exec_indexer.py` 补 `import batch as batch_layout`；`idx_k_write` → `_idx_k_write_pk`。
- [ ] **T-041**：图路径 104→134ms 回归待查（eager 未变，问题在图专属路径）。
- 结论：图代码保留、默认关（`DS41M_GRAPH_TAIL` 默认 0，显式 =1 才开）。

## 2026-10-26 冲 100ms（用户口径：先不管精度）

- 目标：64k@0.97 每 forward 157.2 → **~100ms**（DeepInfer 同口径）。
- 数学：设备忙 88–106ms/forward；host 现 3.5ms/层（140ms）> 设备 ~2.5ms/层 ⇒ 只需把 host
  降到 **≤2.5ms/层（≤100ms）**，抢跑环 K=2 即可藏住，forward≈设备≈100ms。
- 执行：T-042（A，层内零分配，目标 host −30ms）、T-043（B，metadata 去重 −40ms + transposeQMM −9ms
  + 合并 QMM，允许近似）、T-044（融合 kernel，T0 −2.4ms，排后）。

## 2026-10-26 双机并行（用户口径）

- **以后能并行就双机并行**，不考虑跨机测量差异（用户明确：不在乎跨机影响）。
- 默认分工：本机 = 需要独占卡的实现/验证（当前 T-042/D）；邻机 = 执行器队列（矩阵/复现/长跑）。
- 邻机入口：`tmp/npu_exec/queue/` 丢作业，执行器常驻（`nohup bash tmp/npu_exec_executor.sh &`）；
  **不要再放 STOP 文件**（2026-10-26 12:06 误放导致执行器退出一次）。

## 2026-10-28 尾段空泡消除（P0 host 优化，进行中）

> 根因：`NOTES/2026-10-28-tail-bubble-host-bound-opt.md`；
> 方案：`NOTES/2026-10-28-tail-bubble-host-opt-plan.md`。
> 本轮口径：本机（208）有同事实验，**一律投 099 的 `tmp/npu_exec/` 队列**，本地只做代码/分析。

- [x] 根因定位：tail L21–39 span 恒定 62–64 ms、idle 40–46 ms（63–73%）、EVENT WAIT 仅 0.4% ⇒ host 下发
- [x] P0-1 `_mqsmla` cmp 分支热路径 `.item()` 删除（`DS41M_HOST_OPT=0` 回退，A/B 用）
- [x] P0-2 6 处零张量改 `_meta` 缓存（`exec_indexer` sp/tl 带显式复位；`exec_ops` ci/cl 常量免复位）
- [x] **job 059**（055/058 因 099 外部占用被跳过；用户 19:40 关闭后开跑，19:48 完成）：
      8k@0.90 CP1/CP2 **逐位一致**（e8f8ea353cf3 / d1ea9c782847）；
      256k@0.97 CP2 steady **249.1→235.6 ms（−13.5 ms）**，mqsmla gap 55.0→46.1 ms，
      metadata_kernel 53→7（QSLI metadata 复用生效）。详见 plan 笔记「验证结果」。
- [x] P0-3 `_qsli_cp`：idx_w 每层预转 + `st.copy()` + QSLI metadata 进程内复用
      （`DS41M_QS_MD_CACHE`，默认 1）；门禁已加进 `tools/check_perf_defaults.py`。
- [x] **job 060**（19:49–20:00）：h1/h0/h1 交错复测 **235.5/235.5 vs 249.2 ms（Δ=0.0 重复，
      P0 合计 −13.7 ms）**；HOSTTIME 预算：**rope 35.1 ms/pass（80×439 µs）**、
      mqsmla 26.1（40×652）、hc_pre 11.9、wo_a 11.3、mega_moe 9.5、qli 8.9（4×2.2 ms）、
      idx_prolog 6.8（8×0.85）、qsli 6.1（4×1.5）；算子循环 0.17 s/pass（4.25 ms/层）。
- [x] **P0-4**（`DS41M_CP_POS_CACHE`，默认 1）：`CpState.abs_positions` 缓存 +
      CP rope 的 cos/sin gather 跨层复用；顺带让 `cmp_cl_cp`/`win_cp` 的 data_ptr key 稳定命中。
- [x] **job 061**（19:59–20:07）：P0-4 验证 —— 逐位一致（CP1/CP2 与基线同 hash）；
      256k@0.97 CP2 **p1a 232.6 / p1b 233.2 / p0 234.7**（重复 Δ=0.6，**−2.0 ms**）。
      累计 249.2→232.6 ms（−16.6 ms，−6.7%）。tail idle 跨跑抖动大（56/65/75%），不作判据。
      门禁已加 `DS41M_CP_POS_CACHE` 检查。
- [x] **job 062**（20:08–20:17）：HOSTTIME 复测 + NOPREP 探针 —— **NOPREP 无余量**
      （l2off 27.9 vs noprep 28.4 ms，8k 三支逐位一致）；rope 仍 33–36 ms/pass（厂商绑定地板）。
- [x] **job 063**（20:19–20:24，099）：跨点位 sanity —— 64k@0.97 **157→154 ms（−3）**、
      256k@0.90 **538→524 ms（−14，−2.6%）**，收益非单点特例。
- [x] **P0-5 否定**：CP QSLI 双块合并（共享块表 repeat + batch=2）在 8k@0.90 CP2
      **设备挂死**（rc=124；4 个残留 rank 已清、HBM 回落 4750）。代码已删，勿复用。
- [x] 256k@0.97 final-row 逐位（HOST_OPT 0 vs 1）：**sha1 `65caafbbbd02` 相同**；
      跑后无残留、8 卡回 4751 MB 基线（`tmp/P0_256K_PARITY_1028_2033/`）。
- [x] 文档收口：plan 笔记「收口状态」；INDEX/PERF_RULES（HOST_OPT/CP_POS_CACHE 标已验证）；
      check_perf_defaults 已含 3 个新开关门禁。
- 结论方向：剩余 host 成本 = 厂商绑定地板（mqsmla 638 µs、rope 415 µs、qli 2.2 ms、
  idx_prolog 850 µs 每次）；Python 层可动项已全部落地/否决。

## 2026-10-29 桥直发（两条路线，用户要求都试）

> cProfile（`DS41M_CPROFILE=1`，8k CP2，189 次 provider 调用）：
> `resolve_provider_call` ≈460 µs/次（export_records 143 / inspect_storage 137 /
> declared_devices 53 / resolve_stream 58 / preflight 10 / native resolve_first 32）+
> native `invoke` 175 µs/次。⇒ 能省的只有 resolve 的 Python 部分，天花板 ≈56 次/pass ×~250 µs。

- [x] **路线 A：桥 patch（L3，`DS41M_BRIDGE_L3`，默认关）** —— `ops/bridge_l3.py`：
      缓存 `resolve_provider_call` 的 layout/device 解析（`RuntimePhysicalLayout` 不含地址，
      key = name/shape/stride/dtype/npu_format/device，**不含 data_ptr**）；命中跳过
      preflight/declared_devices/inspect_storage/aligned_layouts；流与 export 记录仍逐调用。
- [x] L3 v2 验证：8k@0.90 CP2 逐位一致（`d1ea9c782847`），但 **256k@0.97 第一 pass
      即 aicore exception**（rank4 `ERR00100` / aicore abnormal）⇒ **否决**；
      `ops/bridge_l3.py` 已删除、`exec_ops.py` 只留"已否决"注释。现场与清理见
      `NOTES/2026-10-29-bridge-l3-rejected.md`。带 data_ptr 的第一版 key 只赚 0.6 ms。
- [x] **路线 B：DeepInfer 直发栈评估（只读）** —— 他们的热算子是自家 C++ 模板 kernel
      `sparse_attn_fwd_kernel<<<num_ctas,0,stream>>>(params, aux)`（FlashMLA
      `ascend_kernels/prefill/sparse/kernel_body.h:1690`），host 侧只填 params 结构体。
      我们的 mqsmla/rope/indexer 是 cannbotdsl DSL kernel，接口与数值口径不同 ⇒
      "换自有 kernel 栈" = 重写 4 类热算子 + pybind，属周级工程，非补丁级；本轮不做。
- 教训：P0-5（QSLI 双块合并）与 L3 两次设备级失败都来自"Python 层猜缓存键"；
  涉及厂商内部 layout/元数据的快路必须由厂商 C++ 给语义保证。

## 2026-10-29 绑定直发（vendor bound call）——用户要求落地

- [x] 上限实测（`ops/bind_probe.py`）：`bound.invoke()` **21–23 µs/次** vs 正常调用
      **361–363 µs/次**（同 kernel/张量；resolve 390 µs、bind 85 µs 一次性）。
- [x] 三个默认开（门禁已加）：
      `DS41M_MQ_OUTBUF`（out/lse/workspace 常驻，惰性替换 vendor DSL 入口）、
      `DS41M_MQ_QBUF`（尾段 q 常驻拷贝，T≥32、≤64MB）、
      `DS41M_BIND_FAST`（完整实参指纹 → `bound.invoke()`；≤512MB/调、pinned≤2GB、
      `numel<262144` 不走、同 key 第 2 次出现才建绑定）。
- [x] 已验证：16k@0.97 hit=108、**hash 一致**、0.144 vs 0.188 s；
      256k@0.97 交错 A/B **−7 ms ×2**（hit=75）；256k combo+dump hash 一致。
- [x] 几何边界矩阵（10 点 base/combo 逐点 final-row）：**10/10 逐位一致、0 不一致**
      （`/workspace/tmp/MATRIX_BIND2B_*`；矩阵 iters=3 只作对拍，不作性能口径）。
- [ ] 待验证：decode（T=1，已用 numel 门槛排除绑定）、PD 合并、CP1 大规模点位
- 为什么不能照抄 DeepInfer 的"无键直发"：他们的 kernel 是自家 C++ 模板、params 按值传；
  要同样驱动 vendor DSL kernel 必须改 cannbotdsl host runtime（C++），但共享盘上的
  `cannbot-dsl` 仓**没有 `_host_runtime` C++ 源**（只有预编译 .so）⇒ 需厂商源码/新 wheel。
- 日志：`/mnt/xds/sfs/l00972766/tmp/P0_VALID_1028_1941/`（059）、`tmp/P0_REPEAT_<ts>/`（060）

## 2026-10-29 DP/CP 全量矩阵采集（CP2+CP4，用户任务）

- [x] 采集脚本 `tools/collect_dp_cp_matrix.sh`：CP2/CP4 × 20 长度（1k…1m）× 11 命中率
      （0/10/60/70/80/90/95/96/97/98/99%）；默认采集口径（prof 轮、`--prof-ranks 0`、`--pure-log`）；
      每块 `.done`/`.failed` 可续跑；几何闸跳过 `1k@10%`（M+KEEP140 > L）；磁盘闸 ≥10G。
- [x] 与 CP1 口径对齐（`tmp/SWEEP205_1027_0852`：205 点、noprof、一次进程、18m53s）：
      点集 = 220 去掉 `T_pass=M+140 > 104998` 的 14 点 + `1k@10%` = **205 点/CP**；
      已内置 `MATRIX_TPASS_CAP`（默认 104998，0=不设限）与**分 CP 覆盖** `MATRIX_TPASS_CAP_<CP>`
      （如 `MATRIX_TPASS_CAP_4=60000`）；点数运行期现算并打印 `[点集] CP=… 上限=…`，
      `.done` 记录 `points=N cap=…`。切 CP 后按各 CP 重新计算，不保证恒为 205。
- [x] 点集/进程分组开关 `MATRIX_L_GROUP`：1=每长度一进程；N=N 档合并；**0=一个 CP 全部点一个进程**
      （= SWEEP205 形式）。
- [x] 单进程跑久：`MATRIX_RUN_TIMEOUT`（默认 14400s，传给 `run_hit --timeout`；驱动默认 3600s 会中途杀）。
- [x] 进度速查 `tools/matrix_status.sh`（块状态 + 点计数 + 最后时延 + 争用标记）。
- [x] CP2/L1k 块第 2 点事故定位（详见 `NOTES/2026-10-29-l1k-hang-hccl-contention.md`）：
      host `aclrtSynchronizeEvent` 60.15s，设备侧 4 forward 仅 0.518s，尾部 HcclLaunchAicpuKernel
      ⇒ HCCL 等对端超时；同块 10 点在邻居任务退出后两次跑通 ⇒ 非形状 bug。
      本机 8 卡被另一容器共用（19.5–19.7GB/卡的批次周期性起停），判为争用挂死。
- [x] 加固：每块 3 次重试（重试轮 `DS41M_KEEP_LOG_FILES=1`）、失败现场 `.attemptN.<ts>`、
      开跑前等空闲窗口最长 30 分钟（`MATRIX_WAIT_FREE_SEC`）+ `.contended` 标记。
- [ ] **用户定稿跑法（2026-10-29）**：先 CP2 一个进程跑完全部点，再 CP4 一个进程跑完全部点
      （`MATRIX_L_GROUP=0`，即 SWEEP205 那种"一次进程、预热一次"的形式），**默认采集策略**
      （prof rank0 + `--pure-log`）。用户在 099 自行执行，本机不再跑。
- [x] ★**单进程连跑 200+ 点会累积显存**（2026-10-29 实测）：099 的 CP2 在 point 193
      （`524288@0.8`，L=512k、T=104998）8 rank OOM（84.9/91.75 GiB，reserved 87 GiB），
      而**全新进程**同量级点只需：512k@0.99=15.8GB、512k@0.9=31.6GB、1m@0.95=33.7GB
      （本机实测）⇒ 不是点不可行，是形状相关常驻缓存/图池跨点累积；点间
      `_free_device()`（gc+sync+empty_cache，`points_runner.py:268-269`）只能还空闲块。
      ~~⇒ 大批量必须分块重启进程~~（2026-10-29 二改：**分块重启是回避，已废弃**）。
- [ ] ★**跨点位累积的真修法**（2026-10-29，用户口径「不要用分块重启回避」）：
      `core/memlife.py` —— ①**登记表**：各模块把几何相关容器写进 `POINT_CACHES`
      （exec_ops 12 项 / exec_indexer 4 项 / mq_outbuf 的 out+lse 与 workspace / bind_fast
      的 vendor 绑定 / runner 的 ctx+seq+module-plan / layout batch+replay / engram id 表），
      需要额外动作的定义 `POINT_RELEASE()`（runner 的 CP plan LRU + gather 固定缓冲、
      exec_ops 的 META）；点位边界 `sweep()` 一次清完，不再手写清表。**权重/布局类缓存
      一律不登记**（跨点位复用是"一次预热跑多点位"的收益）。②**审计**：`DS41M_MEM_AUDIT=1`
      每点位边界打印 allocated/留存/Δ，留存超 `DS41M_MEM_GUARD_GB`（默认 4）自动按
      「模块容器增长量 / 张量形状」归因 ⇒ 以后再有人加没登记的缓存，日志直接点名。
      ③ **兜底**（`DS41M_MEM_SWEEP_UNKNOWN`，默认 2 GiB）：未登记但本点位期间长超过阈值的
      模块级容器**自动清掉并打印容器名**（`RESIDENT` 白名单与已登记的不动）⇒ "忘了登记"
      不再等于"静默累积 200 点后 OOM"，而是"慢一点 + 日志点名"。
      ④ `tools/points_mem_soak.sh`：8 点（1k→512k→回小点）一个进程跑完，末点留存须 < 阈值。
      ⑤ `tools/finish_cp2cp4.sh` 两段都改回**一个进程跑完**（`MATRIX_CHUNK_POINTS=0` 默认）。
      ⑥ 配套（都不碰卡）：`tools/selfcheck_memlife.py`（4 条机制自检，全过 ✅）、
      `tools/check_point_caches.py --strict`（静态扫"装设备张量却没登记"的模块级容器，0 个）。
      踩过的坑：`memlife` 一开始按**模块名**过滤 `sys.modules`，而本树是裸名导入
      （`exec_ops`/`runner`），结果一个都扫不到、sweep 静默变空操作 ⇒ 改按 `__file__` 目录判归属。
      **待验**：本机 8 卡要等同事用完；跑法见 ④，通过标准 = 末点留存 < 4 GiB 且无归因行。
- [ ] ★**点位失败隔离**（2026-10-29，用户口径「中间出问题不能全部挂死」）：`core/points_runner.py`
      ①点体整体 `try/except Exception` → traceback + 清显存 + **跳过继续**（`SystemExit`/
      `KeyboardInterrupt` 照旧上抛）；②`pass_barrier_sync` 挪到点体**最前面**——原来在
      `make_input/build_stack` 之后，抛异常的 rank 少调一次 barrier ⇒ 各 rank 错一拍 ⇒
      后面每次集合通信都错位 ⇒ **整轮挂死**（这就是"一个点挂全部挂"的机制）；
      ③watchdog 线程：单点超 `DS41M_POINT_TIMEOUT`（默认 1800s）→ `os._exit(9)`（设备侧
      卡死 Python 打断不了）；④续跑：`results.tsv` 已有 tag 跳过（`DS41M_POINT_REDO=1` 强制重采）、
      `points_failed.tsv` 记失败次数（rank0 原子写，达 `DS41M_POINT_MAX_ATTEMPTS`=3 跳过，
      否则确定性坏点永远堵在最前）、连续失败 `MAX_FAILS`=2 熔断、重跑前清该点位 `PROF_*` 半成品；
      有失败时 `run_points` 返回 **9**（驱动据此重试，重试只补没跑完的点）。
      自检（不碰卡）：`tools/selfcheck_points_loop.py`（假 runner **真跑** `run_points`：
      失败不中断 / 重跑补齐 / 达上限跳过，全过 ✅）、`tools/selfcheck_point_isolation.py`
      （结构 / 续跑 / watchdog rc=9 / 驱动传参，全过 ✅）。真卡失败注入待跑。
- [x] T_pass 上限实测（本机 8 卡，逐点独立进程 + HBM 峰值采样 `tools/calib_tpass.sh`）：
      CP2 = **262,144**（256k@0 通过 98.2GB；512k@0 失败）；CP4 = **524,288**
      （512k@0 通过 98.2GB；1m@0 失败）。236k/262k 档已贴顶（98.2–98.3GB），余量为 0。
- [x] L 维补测（CP2，全新进程）：512k@0.99（T=5383）**15.8GB**、512k@0.9（T=52429）**31.6GB**、
      1m@0.95（T=52429）**33.7GB**、1m@0.9（T=104998）**52.2GB** ⇒ **L 项影响有限，显存主要看
      T_pass**；099 的 OOM 是跨点累积（见上一条），单点全新进程在 CP1 口径（T≤104998）内最长也只要 52GB。
      ⇒ **维持 MATRIX_TPASS_CAP=104998（205 点）**，不提高到 262k/524k（那几档贴顶 98GB 无余量）。
- [x] `tools/finish_cp2cp4.sh`（2026-10-29 二改）：CP2 补跑（剩余 13 点，**一个进程跑完**，
      512k@0.8 最后）+ CP4 全量（预检 512k@0.8/1m@0.9 后，其余 **一个进程跑完**；
      `MATRIX_CHUNK_POINTS=0` 默认不分块，只留作诊断）；两段都带 `DS41M_MEM_AUDIT=1`。
- [x] 旁证数据集：`/workspace/tmp/COLLECT_DPCP_1029_0344/`（按长度分块，CP2 已完成 L1k–L80k 共 14 块；
      L≤96k 的点集与 205 版完全一致，可作交叉核对）。
- [ ] 采完：按默认口径解析（`tools/trace_report.py` 导出 + `analyze_tail.py`）并打包；
      筛掉 `.contended` 块后复跑，再出 forward 时延表。
- [x] ★**流水线固化**（2026-10-29 用户口径「以后按这个一步一步来」）：共享盘 **`PIPELINE.md`**
      —— 采集（一进程 205 点）→ 解析（`msprof --export` + forward 切分）→ 汇总（`TRACE_MATRIX`
      / `TRACE_MODEL` / `COMPARE_*`）→ 打包（`<扁平名>/` + 七件套）。含前置检查、验收清单、
      10 条已知坑（墙钟虚高 / 跨点累积 / 单点挂死 / pure-log 吞 `[mem]` / export 非幂等 /
      重试 mv 掉 results.tsv / env 写在 timeout 后 / 环境争用块 / memlife 扫不到裸名模块 /
      解析抢 I/O）、口径速查、工具索引。一键：`tools/run_cp1_matrix_pipeline.sh <OUT_ROOT> --cp N`。
      `PERF_RULES` §5 与 `TRACE_GUIDE_SIMPLE`/`TRACE_GUIDE_PARSE` 头部均已指向它。
- [ ] ★**缺陷：命中 + CP 的索引器没有按 CP 切分**（2026-10-30 用户指出）：`_qli()` 在
      `pos_offset>0` 时默认 `_qli_cp_full` —— **每个 rank 把整段 T_pass 的 Q gather 成全长、
      跑一遍全长 QLI、再切回本地行**。DeepInfer 的口径（`attn.py:550-580`，我们 `_qli_cp` 的
      docstring 原文）是「本地 q 直接对 full K 查」= **切**。现在等于用性能换了一个绕行：
      同一 case 三套 CP 的 QLI 时长几乎逐位相同（128k@0.8：CP1 236.3 / CP2 235.8 / CP4 235.8 ms），
      而 hit=0 走分块版时 QLI 随 CP 切（32k@0：60.6 / 27.1 / 14.0）。代价：CP2 128k@0.8
      每 forward QLI ≈ 59 ms（占 15%），CP 越大浪费越多。
      当时不切的理由：2026-10-28 实测分块版与全长版 **top-k 选择不等价**（层 2 末行 idx_sparse
      不同，max|Δ|=3.4e14），为了「CP 命中与 CP1 逐位一致」直接切成了全长。
      **根因未定位**。已知三处口径差异（都可能改变近似并列的 top-k 取舍）：
      ①分块版每块重建 `_qslitab(nblk_chunk+1)[:nblk_chunk]`（首块=池尾、且不含块 0），
      全长版用 K 缓存写侧的 bt（按全局 nblk）；②seqused_k：块前缀 (gs+nv)//ratio vs L//ratio；
      ③batch_size：1 vs nseq。
      修法：小 case（32k@0.9）跑 `DS41M_CP_QLI_FULL=0` 逐层 dump `cmp_sparse_indices` 与全长版
      diff → 定位第一处差异行 → 统一 block_table/seq_k 口径 → 逐位通过后把分块设为默认 → 复测收益。
      记录：NOTES/2026-10-30-cp-hit-indexer-full-recompute.md

- [ ] ★**开关审计（2026-10-30，244 个 / 默认关 70 个）** — 一项一项处置，详见
      `NOTES/2026-10-30-switch-audit.md`。**每条都必须给出 A/B 数字或明确的"不修理由"**。
  - [ ] **C1 `DS41M_MOE_CAST_INTO_SB`（默认 0）**：MoE 输入 `per_token_cast(out=对称缓冲)`
        直写，省每层 ~18 MB 读+写。当前理由只有一句"sf 取整口径未对拍"。
        **动作**：①读厂商 `per_token_cast` 的 sf 取整语义（cp.async/四舍六入？）；
        ②与官方参考 trace 的 MoE 输入 sf 逐位对拍（`op_summary` 里 mega_moe 输入）；
        ③通过 ⇒ 默认开 + 记录收益；不通过 ⇒ 在 §3 例外表写明"差在哪、影响多少"。
  - [ ] **C2 `DS41M_ENGRAM_HUGE1G`（默认 0）**：1G 大页因 torch 缓存分配器碎片拿不到，
        退回落 HUGE_FIRST。**动作**：量一次 engram 层取表时延（大页 vs 4K 页），
        给出毫秒差；若差额 < 5% ⇒ 保持关闭但登记；若大（>10%）⇒ 改成"启动时尝试一次、
        失败再回落"而不是永久关。
  - [ ] **C3 `DS41M_MQ_FAST`（默认 0）**：mqsmla 薄下发通道，注释里有逐位 digest 证据
        （64k@0.97 `08443723284dd9ed`）。**违反 PERF_RULES §0**。
        **动作**：①查 git/笔记为什么没默认开（是否踩过稳定性/超时）；
        ②按 §0 做一次全点 digest A/B（≥3 个 case）并默认打开；不能开就登记 §3。
  - [ ] **C4 `DS41M_RELEASE_Q_ATTN`（默认 0）**：mqsmla 后释放 q、wo_a 后释放 attn_out
        （显存），默认关着。**违反 §0「避免 OOM 也算性能」**。
        **动作**：量 128k/256k case 的峰值显存差与 forward 时间差；
        显存降且时间不涨 ⇒ 默认开；有副作用 ⇒ 登记理由。
  - [ ] **C5 `DS41M_DSPARK_DRAFT_GRAPH`（默认 0）**：收益 0.7 ms/step，属"允许例外"，
        但**没登记 §3 例外表**。**动作**：补登记（一行事），指向本条。
  - [x] **C6 ALOP graph-safe 遗留物（2026-10-30 已清）**：入图 2026-10-27 永久删除后，
        `configs/alop_mindstudio_graph.yaml` 的前提（graph replay）已不存在，且其注释声称的
        "run_hit.sh 自动选本文件"分支随 `$GRAPH` 一起被删（`ALOP_CFG_SET` 只剩赋值、无人读）。
        **已删**：该 yaml、`ALOP_CFG_SET`（run_hit.sh 2 处）、`meta/policy.py:GRAPH`
        （`DS41M_GRAPH_TAIL`，无消费者）；**已改**：PERF_RULES 2026-10-22 行加删除注记、
        `trace/profiling.py` / `run_real.sh` / `docs/COLLECT_REAL.md` 的"python_profiling 开"旧描述、
        `alop_shapes_aclc.yaml` 的 `GRAPH_TAIL_SYNC` 指引、`alop_mindstudio_stack.yaml` 用法行。
        验证：`py_compile` + `bash -n` + yaml 解析全过；全树 `grep GRAPH_TAIL/ALOP_CFG_SET` 仅余说明性注释。
  - [ ] **D1 `DS41M_PERLAYER_CACHE` 守卫（正确性，会静默算错）**：
        现状：decode/serve 强制 =1（`core/runner.py:2286`、`:3452`），**real-hit prefill 没有**。
        `core/run.py:63-72` 自己写了根因（stage B 读"别的层的前缀"）。
        **动作**：`hit.source == "real"` 且 prefill 且该开关未开 ⇒ **启动即拒绝**（errExit），
        提示"要么开 PERLAYER_CACHE，要么用 fake hit"；fake hit 保持现状（零影响）。
        验收：故意不设时启动失败并给出可操作提示；fake-hit 全量 205 case 逐位不变。
  - [ ] **D2 `DS41M_CP_HIT` 守卫（口径错，会静默跑成全量）**：
        CP>1 且点位是命中（hit>0）且未显式设 `DS41M_CP_HIT` ⇒ 打印**红色大警告**
        （或直接拒绝），避免"以为是命中数据、其实是全量"。
        验收：不设时日志出现醒目告警；显式 =1/0 时无告警。
  - [ ] **E1（缺陷）命中 + CP 的索引器：每 rank 全长复算 → 应改成按 CP 切**
        （`_qli_cp_full`，偏离 DeepInfer「本地 q 对 full K」；代价 CP2 128k@0.8 每 forward 15%）。
        记录：`NOTES/2026-10-30-cp-hit-indexer-full-recompute.md`。
        **已做**：开关修好（显式 0 能切走）；三轮 A/B 定位到根因在**索引 K 池内容不同**
        （同形状、sha 不同）；否定了"ctx 少 T"的假设。
        **下一步**：①三个 K 写者（`_idx_k_write` / `_idx_k_cache` / `idx_k_cache`）入口统一打点
        ②A/B 看各自用哪个写者、几何差在哪 ③统一写者后验 top-512 逐位
        ④通过后分块设默认 + 复测 QLI 时间。
  - [ ] ★**E2（缺陷，用户 2026-10-30 定调）：CP 压缩/索引缓存改成 DeepInfer 官方流程**
        **进展（2026-10-30）**：决策 A 已拍板——以官方 CP 流程为准（边界组不追 CP1 逐位）。
        ① 分段压缩已实现（`DS41M_CP_CMP_FLOW` 默认 `seg`，仅命中段），SKIP_MOE 下 L8 与 CP1
        仅 3 个边界行不同；② 算子 state 块语义已用哨兵钉死（每请求一块，`state_block_table` 生效）；
        ③ 负结果：overlap 方案否决；④ 待办：索引 K 行 gather（68 B/行）、QLI 切分块、性能验收、
        plain 段 507015/507035 定位。记录：`NOTES/2026-10-30-cp-official-flow.md`。
        **要求：性能与数值都等价，不是"换个写法数一样"**。官方口径（`CP8_COMM.md` §1，
        代码 **`/mnt/xds/sfs/DeepInfer/deepinfer/model_executor/models/v4_modules/`**
        ——2026-10-30 用户指令：cann-recipes 仓已删除、**永远不再参照**（PERF_RULES 铁律 2）；
        逐条对照见 `NOTES/2026-10-30-cp-official-flow.md`【DeepInfer 源码对照】）：
        ① `compressor.forward_cp`（compressor.py:191）：**按本 rank 两个 zigzag 段分别压缩**
        （各自 start_pos/state_block/remainder state）→ 本地行量化进 scratch（恒等槽位）→
        `gather_cp_segments`（每 rank 送两段、reverse_index 还原序列序）→
        `scatter_cache_rows` 按 `slot_mapping_cmp` 散写持久 cmp_cache。
        ② `sync_cp_state`（:266）：state 整块 all_gather，每 state block 取"该请求最后一段所在卡"的副本。
        ③ `gather_remainders`（:234）：每段末 hidden all_gather，补 ratio 余数缺口。
        ④ `indexer.update_cp_indexer_cache`（indexer.py:140）：**每段** write_prologue_k 量化进
        scratch → K+scale 行拼成一行 all_gather → 散写 indexer_cache / indexer_cache_scale →
        `repack_qsli_cache`。
        ⑤ QLI 用**本地 q 对 full K**（分块，不再 gather q / 不再每 rank 全长复算）。
        ⚠️ **2026-10-30 校准（以 DeepInfer 源码为准）**：上面 ②③ 是已删仓的机制名，
        DeepInfer 里**没有** `sync_cp_state`/`gather_remainders`——段间 carry 直接走
        `state_cache`（`layers/tilelang_ops/compress.py:186-190` 读、`:196-206` 写，
        每 seq 一个 `state_block_idx`）；压缩只算本地段（同文件 `:181-196`，
        `blk_id ∈ [qstart//r, klen//r)`）。**通信口径与 DeepInfer 有一处不同**：DeepInfer
        all_gather 压缩器**输入** `projected_kv`（`models/v4_modules/attn.py:785`，与 Q 投影
        overlap），我们 gather **输出压缩行 320B/行**——计算量相同、数值等价、通信更少，
        按铁律 1 保留我们的（已登记，不再视为偏离）。索引打分内核仍与 DeepInfer 不同
        （融合 `quant_lightning_indexer` vs `deep_gemm mqa_logits`+`DeepSelect`），
        **属未对齐缺陷，待用户拍板是否换**。
        现状偏离：mini 默认是 **gather hidden(10KB/token) → 每 rank 全量复算压缩**
        （`exec_ops.py:1579`），索引 K 也由本地 latent 直接写（无行 gather）。
        **实测后果**（2026-10-30，8k@0.9，T=959）：CP2 的 latent sha 与 CP1 不同
        （L8 `717e555c` vs `0250b8f1`）；sp 逐位：head 240/240 同、tail 101/239 与 199/239 不同。
        **验收**：① CP2/CP4 的 `cmp_latent`/index K/sp 与 CP1 **逐位相同**；
        ② QLI 时间 ≈ 1/CP × CP1（不再与 CP 无关）；③ gather 载荷按官方口径
        （压缩行 320B/行、K 68B/行），不再搬 hidden。
        记录：`NOTES/2026-10-30-cp-official-flow.md`。
        **E2 下一子任务（K 行 gather，已备好接口）**：`_kv_compress_cp_segments` 现在会把
        **逐段本地 latent** 存进 `st["_cp_seg_latents"]` / `st["_cp_seg_blocks"]`。实现步骤：
        ① 在 `ops/exec_indexer.py` 新增 `_idx_k_rows_cp(st, ctx, cp, w)`：每段用
        `_CORE._ifc.prolog_k(..., k_cache=scratch, k_scale_cache=scratch_s, cache_index=identity)`
        算本地 K（64B）+scale（4B），拼成 68B 行、按 R 行 pad；
        ② 用 `exec_ops._cp_gather_2seg` 收行（2 段/rank，全局段序）；
        ③ 按 `(1+p//BS)*BS + p%BS` 散写 `_idx_k_r{ratio}` / `_idx_ks_r{ratio}`（K 与 scale 分开）；
        ④ `_qli_pk`/`_qli_chunk` 传 `idx_k_ready=True` 只读池；
        ⑤ 验收：K 池 sha 与「全量 latent 写池」逐位相同（同源），再测 QLI 时间与 gather 载荷
        （应从 256 B/行 降到 68 B/行）。
        **探针结果（2026-10-30 修正后）**：**2/241 行**不同（段 0 与段 3 各 1 行，均为
        段边界重叠组）⇒ scratch 逐段算 K + 旋转块表寻址**已可用**。
        **已接线并做 A/B（2026-10-30）**：`_idx_k_rows_cp`（`DS41M_CP_K_GATHER`，默认关）
        可跑通，但与「全量 latent 写池」**不等价**（L8 sp 51 行、L14 61 行不同，K sha 异）。
        **已收敛（2026-10-30）**：scratch 改用**旋转块表**后 L8 sp 只差 **1 行**（段边界
        carry 组，决策 A 接受）⇒ `DS41M_CP_K_GATHER` **默认转正**（`=0` 可回退）。
        效果：gather 载荷 256 B/行 → 68 B/行；K prologue 计算 ×cp → ×1。
  - [ ] ★**E3（缺陷清单，铁律 1）：数值等价但性能拉垮的地方，全部揪出来改掉**
        规则已写进 `PERF_RULES.md` §0 铁律 1（2026-10-30 用户口径）：
        **借鉴实现不能只追数值等价，性能必须等价或更好**。
        | # | 位置 | 现在怎么"偷懒" | 官方/应有做法 | 代价（待量化） |
        |---|---|---|---|---|
        | E3.1 | `ops/exec_ops.py:1580` | CP 下 `gather hidden`（bf16，约 10 KB/token）→ **每 rank 全量复算压缩** | gather **压缩行** 320 B/行（`compressor.forward_cp`） | 通信 ~30–60×、算力 ×cp |
        | E3.2 | `ops/exec_indexer.py:782` `_qli_cp_full` | 命中 + CP：gather q → **每 rank 全长复算 QLI** | 本地 q 对 full K（K 行 gather，68 B/行） | CP2 128k@0.8 每 forward 15%，CP 越大越差 |
        | E3.3 | `ops/exec_indexer.py:796` | 每层 gather `idx_q/idx_w/descale`（约 2 KB/token） | 官方无此通信（q 本来就在本地） | 每层一条 all_gather |
        | E3.4 | `ops/exec_ops.py`（state） | 无 `sync_cp_state`：每 rank 用本地 state 复算 | state 整块 all_gather（官方 #4） | 逼出全量复算，且数值仍不等价 |
        | E3.5 | `ops/exec_ops.py`（remainder） | 无 `gather_remainders`：余数缺口靠全量复算兜 | 每段末 hidden all_gather（≤3 次） | 同上 |
        | E3.6 | `ops/exec_ops.py:1507` | `merge_global(proj)` 4 KB/token（实验路径） | 压缩行 320 B/行 | 12× |
        | E3.7 | `ops/exec_indexer.py:1200` | 每层 gather 候选索引/长度（int32） | 待查官方是否必要 | 待量化 |
        | E3.8 | `core/runner.py:234` | 每 forward `merge_global(x)`（边界拆分用） | 官方 `gather_cp_tail_hidden` 只收尾窗 | 待查 |
        **动作**：①逐项用 trace 量出通信/算子耗时与占比；②按 E2 的顺序改成官方式；
        ③每改一项做数值 + 性能双验收（逐位 + 同口径耗时）；④把每项的量化数字补进本表。



- [ ] ★**E5（2026-10-31 用户定方向）：候选池 `cl` 口径与 DeepInfer 对齐**
      —— 现状：L20 产的候选块 `sorted=True` ✓，但 `cl`（每行候选长度）我们厂商算子给的是
      **候选块个数**（实测 max 1024 = 8192/8），DeepInfer `get_candidate_compact_lens`
      给的是**紧凑 token 数**（Σ clamp(visible − blk*bs, 0, bs)）。
      厂商路径自洽（产出/消费同一个算子），但**跨实现口径分歧**：一旦切到
      `fp8_fp4_sparse_mqa_logits`，`end=` 必须按紧凑 token 数，否则候选裁剪会错。
      ⇒ **当前结果跑完后，性能优化的下一个方向就是这条**（同时含 dg deep-jit 工具链）。
      证据：`tmp/CANDPROBE3b_*` 的 `[cand-dbg] L20 … cl_max=1024 ref_max=512`。

- [ ] ★**E4（2026-10-30 用户拍板「要换内核」）：索引打分换成 DeepInfer 那套**
      —— 融合 `quant_lightning_indexer`（vendor aclnn）→ `deep_gemm.fp8_fp4_mqa_logits`
      + `deep_select.interface.topk`。**任务书：`TASK_indexer_kernel_deepgemm_1030.md`**。
      已核实（2026-10-30）：①dense `fp8_fp4_mqa_logits` 两边签名**逐字相同**
      （`rt/deepgemm-pkg/deep_gemm/_C.pyi:299` vs DeepInfer `_C.pyi:363`）；
      ②**缺 topk**：DeepSelect 源码在 `DeepInfer/vendor/DeepSelect`，预编译只有 **cp312**，
      我们跑 cp311 ⇒ 必须重编；③sparse/candidate 变体我们的 deep_gemm **没有**（源码也没有
      `sparse_mqa_logits.hpp`），DeepInfer 的有；④两版 `mqa_logits.hpp` 不同
      （我们 2026-10-12、DeepInfer 2026-10-20 `3a35326`）⇒ 需裁决是否用其源码重编独立包。
      验收：`dg` vs `vendor` sp 逐位；CP1/2/4 逐位（仅段边界 carry 组）；trace 口径不劣于 vendor；
      切默认后**删旧路径**（铁律 0）。cp311 编译与 GPU 任务都走 099 执行器。
      **进展（2026-10-30）**：阶段 0 完成——DeepSelect cp311 编好并通过 5/5 自检
      （复用 `s00576609/deepinfer-image-20261020` 的 bisheng wrapper 配方）；
      两版 `mqa_logits.hpp` 内核不同（新增 `kNumQueries`/`kSplitKV`）⇒ **现有 deep_gemm ≠ DeepInfer**。
      裁决：阶段 1 先用现有包打通 dense（只算集成验收，**不算 DeepInfer 对齐**），
      同时后台起 `deep_gemm-di` 编译，换包后重跑同一套 A/B 再进性能验收。
      **进展 2（2026-10-30）阶段 1a 达成**：plain 8192@0（命中仿真的前缀是假数据，不能用来判内核）
      L2/L8 两臂 **sp 每行集合逐位 + tl 逐位相同**（非 candidate 层）；
      差异只剩 top-k 并列的**排列顺序**（前 1023 行相同，之后不同）——
      dg 走 DeepInfer 的 `sorted_index=True`（升序），vendor 非升序。
      **裁决：顺序不判**，dg 是今后的顺序基准；下游 L14+ 的 4e-7 漂移属预期。
      ⚠️ **风险条目（铁律 0）**：历史全部数值基线（CP1 逐位口径、97% 命中矩阵、真权重 `id=396`）
      都是在 **vendor 顺序**下建立的，切 dg 后全链路约 1e-7 漂移，**真权重贪心 id 可能翻**。
      ⇒ dg 切默认前**必须**重跑 `tools/run_real.sh`+`mini_real.yaml`，id 变化要单独上报
      （哪里/为什么/是否纯排序漂移/影响多大），不许当"正常波动"。
      ⚠️ **基准真实性**：命中仿真（`8192@0.9`）的 dense K 中 **3617/4096 行全零**（前缀未算），
      命中场景的 sp 是在零值上做的 top-k ⇒ **任何涉及命中 sp 数值的结论都不可信**；
      性能数据不受影响（数据无关）。内核 A/B 必须用 plain。
      **进展 3（2026-10-30）阶段 2 达成（plain/b1）**：dg 内部 CP1 vs CP2/CP4 按全局行逐行比对
      **diff=0（含段边界 carry 组：CP2 `[2047,4095,6143,8191]`、CP4 `[1023,2047,…,8191]`）**，
      **原始张量级 `torch.equal` 成立（diff_elems=0，非集合级）**，`tl` 全等；
      dump 命名改用 `CP_STATE.cp_rank`（原 `os.getpid()` 跨臂串号已修）。
      覆盖限制：hit0、batch=1、`SKIP_MOE/ENGRAM`、仅 8192。证据 `tmp/ev2.md`。
      下一步：命中 CP → 真权重对拍 → 性能记账。
      **进展 4（2026-10-31，主控接手）命中 + CP2 打通**：三处根因全部修复并验证，
      记录 `NOTES/2026-10-31-hit-cp-nan-rootcause.md`：
      ① **`pos_offsets` 跨块陈旧**（CP 逐块调用时列表继承上一块）⇒ 压缩池按头块几何只开 2688 行、
      尾块 K/V 写错行 ⇒ L2 越界 33 万次 → `moe_topk_gate` NaN。修：单序列以标量 `pos_offset` 为准
      （`exec_ops._kv_compress_packed` + `exec_indexer.idx_k_cache` 两处）。
      ② **压缩池只覆盖本 rank 块末端** ⇒ 层 20 边界截断后 compact 跨 rank 重分布，rank1 索引到 8058
      而池只到 7295 ⇒ aicore trap 507015。修：池块数取 `max(本 rank 末端, L_full//ratio+1)`。
      ③ **engram 分片数 8 > world 2** ⇒ 只注册 1/4 表行、其余取回垃圾 → L1 全 NaN。
      修：`tp = min(tp, world_size)`（日志由 `rank 0/8` 变 `rank 1/2`，全表覆盖）。
      ④ 附带去掉容器本地依赖：dg 内核的 libstdc++ 改从**共享盘** `rt/ds-env-lib`（6.0.34，含
      CXXABI_1.3.15）注入（原硬编码 `/opt/ds-env` 导致 099 执行器 exit 2）。
      **验证（099 执行器）**：hit 0.3/0.5/0.7/0.9 + CP2 全部 `✅ 全部完成`；
      其中 0.5 另跑 **engram 开** 也通过；`oob=0`（L2 池 4352 行、L20 池 8448 行）。
      待办：L8 单行 cmp_kv NaN（512/542720，查是否会被选中）、plain CP2 dg 回归、CP4、L20 候选池核对。
      **进展 5（2026-10-31）**：
      ① **plain CP1 vs CP2 逐位通过（vendor 内核）**：L2/L8/L14 各 8191/8192 `diff=0`、
      L20 512/512 `diff=0`，`tl` 全等，**段边界 carry 行也 0 差异**（干净数据、
      无算子错误）。证据 `tmp/PLCP1b_*` / `tmp/PLCP2b_*` + `tmp/cmp_cp_vs_cp1.py`。
      ② **测试框架"盖草"已修**：`points_runner` 原来只打 `⚠️ N 处` 仍 `✅ 全部完成`/rc=0 ⇒
      实测掩盖了两起真错误（dg deep-jit 8 处、CP1 packed 断言 NameError 3 处）。
      现在**有算子错误 → rc=8 → ❌**。
      ③ 顺带修正 208 agent 的 `idx_k_cache` 越界断言（把合法 pad `-1` 判成越界、
      且消息引用不存在的 `nblk`）——它会**打断 qli 算子**并抛 NameError。
      ④ 遗留：**dg 内核路径在 099 执行器环境 deep-jit 编译失败**（L2/8/14 每层每轮），
      需把工具链注入 run 脚本（同 libstdc++ 类问题）；在修好前 dg 的数字不可用。
      **进展 6（2026-10-31）真权重 16k 验收通过**：`REALPAR208c_1031_033149`（208，8 卡 EP8，
      真表 engram 8 分片、自然路由、ninja 就绪）：
      **prefill 与 hit(0.96875) 两口径 `id=396/396`、重合 7/8、精度 gate ✅、
      口径自检 + trace 断言全过**；官方基准 id=396 ' that' ✓（官方 prefill 2394.7 ms 作参考，
      mini 稳态 703/710 ms，口径不同不作直接比较）。总用时 25m33s。
      ⑤ 注意：`118` 的 perf trace 采集未落盘（`out/L<L>/prefill` 无 `PROF_*`，`trace 0/1 ✗`），
      采集链路单独立项，不影响上面的正确性结论。
      **进展 7（2026-10-31，主控）dg 内核切默认 + CP 逐位 + deep-jit 根因**
      记录 `NOTES/2026-10-31-dg-kernel-default-and-cp-parity.md`：
      ① **`DS41M_IDX_KERNEL=dg` 已成为默认**（`env_pack.sh` 统一注入 ⇒ run_hit/run_real 同口径；
      `_qli` 兜底默认也改 dg）。未设变量实测日志 `索引器内核 = dg`、rc=0。
      ② **plain CP1 vs CP2 逐位（dg）**：L2/L8/L14 8191/8192 **diff=0**、L20 512/512 **diff=0**、
      `tl` 全等、**段边界 carry 行 0 差异**（`tmp/DGCPAB_1031_044202/`）。
      ⇒「本地 q 对 full K、每块只算一次」在 dg 内核下成立。
      ③ **trace 里是 DeepInfer 的 kernel**：`deep_gemm::mqa_logits_impl`、
      `topk_select_ascend::topk_select_kernel`、`deep_gemm::sparse_mqa_logits_impl/_metadata`
      （`tmp/DGPROF_1031_044007/.../PARSED/op_top.csv`）。
      ④ **deep-jit 跑不起来的根因**：`deep_gemm_di/_C.so` 硬编码
      `-nostdinc++ -isystem/opt/ds-env/.../c++`，而 208/099 的 `/opt/ds-env` 都被清掉 ⇒
      缓存命中能跑、一要重编就 `command.hpp:35`。修：`rt/setup_ds_env_headers.sh`
      （本机 gcc 头建符号链接树 + bisheng 自检），两台都 ✅。
      ⑤ 命中 + CP：hit0.5 **CP2 rc=0/0 错**、CP4 见 `tmp/DGHITCP_*`。
      ⑥ **候选层 QSLI 已写 dg 版**（`_qsli_dg_chunk`：组合缓存→dense K + 官方稀疏打分 +
      DeepSelect topk），分派在 `_qsli_pk`（b=1 预填实际路径）与 `_qsli_chunk`（CP 逐块），
      开关 `DS41M_QSLI_KERNEL`（**默认 dg**，见下）；**与入图无关**（PERF_RULES §0：2026-10-27 入图已永久删除，`dg_on()` 恒 False），
      **多请求打包（nseq>1）显式报错**（未实现，需显式回退 vendor）。
      第一版把组合缓存行的 scale 取成 `[64:68]`（实际布局是「8 个键 512 B + 8 个 scale 32 B」
      分段）⇒ 与 vendor top-512 零交集；已修。**A/B 必须用 plain 数据**（hit 的前缀是零值，
      规则 §E4 已写明），第一次用 hit0.5 做的 A/B 结论作废。
      ★ **已解决（同日）**：改用**共享 split index-K 池**（与 `_qli_dg` 同一份 K，
      = DeepInfer `use_shared_index_k`）后，与 vendor 的对比从 38/512 升到 **493/512**；
      自洽验证 `DS41M_QS_SPCHK`：同一进程里 dg「候选内稀疏 top-k」vs「全量 dense top-k」
      **逐行完全相同（L24 T=140：140/140 行、row0 512/512）** ⇒ 稀疏路径正确，
      剩余 ~4% 是 vendor 吃组合缓存 repack 的口径差。**`DS41M_QSLI_KERNEL` 默认已切 dg**。
      ⑦ 顺带修源层口径：DeepInfer 源层 topk 走**全量** logits 且不 restore
      （`index_attn.py:339-365`），原来的「候选块内 sparse + restore」是错的（L 大时）。
      默认对齐，`DS41M_DG_SRC_SPARSE=1` 复现旧口径。
      ⑧ **风险事件**：208（1 vCPU）上 hit0.5 + **CP4** 崩 `aicore timeout` → HCCL watchdog；
      同时段该机还并跑着别的 run，疑似主机饿死。099 复测 job `131_hit_cp4_dg`，结果未回。

---

## E2/E3 收尾记录（2026-10-30，QLI 分块任务）

### E3.2 `_qli_cp_full` → 本地 q 对 full K（**已完成，默认分块**）

| 项 | 内容 |
|---|---|
| 根因（已修） | `_idx_k_cache._ci_build` 入参语义是 **token 位置**，但两个非 dg-replay 调用点传 `row0 = pos_offset // ratio`（已是压缩行号）⇒ **二次除 ratio**，写区从 3616..4095 前移到 1808..2287。修法：调用点改传 `_pos0 = pos_offset`，函数体不动；`row0` 只留给 `_comp_rope_tables(row_off=)`。 |
| 数值验收 | 分块 vs 全长：sp 差异 40512 行 → 7978 行（-81%）；L20（ratio=1）6120 → **0**（逐位）；残余差异 jaccard 中位 0.996、`其它=0`、latent 只差 `{0,120,240,360}` 段边界行。判据已固化进 `PERF_RULES.md §6.1`。 |
| 性能验收（trace 口径） | QLI 总时长：CP1 1189.6 us / CP2-全长 1200.7 us / **CP2-分块 1011.8 us**（0.85× CP1，达标）。QLI 单次：74.4 / 75.0 / **31.6 us**；count 16 / 16 / 32（zigzag 两块）。 |
| 整 forward（非 prof rank） | CP2-分块 六个 rank 全 0.170 s（完全一致）。 |
| 遗留 | 分块不能达到 1/CP×CP1：单次减半（M 小）但调用次数 ×2（`cp_layout.local_blocks()` 每 rank 2 块），净收益 -15.7%。CP4 会更差。 |
| 开关 | `DS41M_CP_QLI_FULL=0` 为分块；默认命中段仍走全长（`pos_offset>0` 触发），**待下一步翻默认**。`DS41M_CP_K_GATHER` 默认开（nogath 为旧路径，仅 A/B）。 |

### C0：`_qli_cp_full` 在 CP>1 的候选源层有**既有缺陷**（全长臂专属）

| 项 | 内容 |
|---|---|
| 现象 | `_qli_cp_full` 把 Q gather 成全局 `total_q` 行后，QSLI（候选源层 L24/L28/L32）把全局行喂给只接受 `plan.local_rows` 的 `cp_state.all_gather_global` ⇒ `ValueError: x_local 行数 512 != plan.local_rows 70`（CP2/8k 实测 8 rank × 3 层 = 24 条），算子被吞后**静默降级**。 |
| 代码位置 | 报错点 `cp/cp_runtime.py:110-111`；调用者 `cp/cp_state.py:157`。日志里的 `exec_ops.py:3241` 是 patch 前的旧行号。 |
| 各臂对比 | CP1 **0 条**；CP2-全长 **24 条**；CP2-分块 **0 条**。 |
| 影响 | 全长臂的数值与耗时**都不能当基准**（r2–r7 稳态离散 0.164~0.187 s 就是各 rank 降级程度不同）。干净基线用 CP1 或分块臂。 |
| 处置（2026-10-30 裁决） | **不修**（收益是"得到一个马上要删的基线"，不值）。加运行时醒目警告；**C3 阶段删除 `_qli_cp_full`**。 |
| 顺带澄清 | "r1 慢 20×" 不是 bug、也不是 prof 标记错误：r1 标记确为 `prof=off`，但它与 r0 同属一个 CP 组（cp_size=2），r0 带 msprof 插桩后 **r1 在集合通信里必须等 r0**。⇒ 稳态口径必须排除 prof rank 所在 CP 组的**全部成员**，不能只排除 rank0。 |

---

## 2026-10-31 会话总计划（DeepInfer 内核对齐 / CP 缩放 / trace）

### 已完成（有证据，勿重做）
- [x] qli/qsli 换 DeepInfer 内核并**默认开**（`DS41M_IDX_KERNEL=dg`、`DS41M_QSLI_KERNEL=dg`；vendor 仅 A/B）。
      证据：CP1 vs CP2 逐位 `diff=0`（`tmp/DGCPAB_1031_044202`）、稀疏 vs 全量 dense top-k 140/140 行（`DS41M_QS_SPCHK`）。
- [x] 真权重 16k EP8 对拍**两台机器**均 `id=396/396`：099 `REALFINAL2_*`（741/743 ms）、208 `REAL208_*`（741/728 ms）。
- [x] 64k@0.97 **算子级 trace 包**（带 op_summary/算子名）：CP1 12 MB（8799/7784 有名）、CP4 28 MB（31639/18439）
      `tmp/PKG_64K_CP1B_1031_073644.tar.gz`、`tmp/PKG_64K_CP4STD_1031_073404.tar.gz`。
- [x] 工具链修复：deep-jit 头文件树 `rt/setup_ds_env_headers.sh`；`env_pack` 注入与内核解耦；
      `trace_report.ensure_export` 空目录误判（"空包"根因）；`run_hit` trace 核对扁平名兜底。
- [x] `_dg_paged_to_dense` 越界前置断言（trap→主机侧报错）；`_qli_dg_cp` 出口 sparse 索引有界断言。

### 进行中（代码树已冻结：exec_ops.py `b194091b…` / exec_indexer.py `5a554caf…`，08:20:52）
- [ ] **消 CP launch 空泡**（agent `a7f90db0`）：把每层重复的 CP 索引几何（`_dg_build_ks_ke`/`_qsli_slots`/
      `compact_lens`）按 DeepInfer `input_metadata.py:185 prepare_cp_mirror` 的结构上提为**每 forward 一次**，层内只引用。
      判据：64k@0.97 CP4 trace **任务数 7841 → <6000/fwd**，墙钟 211.3 → <190 ms；随后 8k CP2 冒烟 + CP1/CP2 逐位。
      结论落 `NOTES/2026-10-31-cp-bubble-host-indices.md`。
- [ ] **验证 + 归档**（agent `adac76fa`）：公式审计已完成（24/24 单序列几何与 DeepInfer `prepare_cp_mirror_ref` **逐位相同**）；
      待填 trace 任务数、逐位两项。结论落 `NOTES/2026-10-31-cp-index-hoist-from-deepinfer.md`。
- [ ] **DeepInfer 官方 prefill950 benchmark**（agent `ac0a1766`）：`DeepInfer/benchmarks/prefill_ascend950`，
      `deploy.sh` 建 venv + `collect.sh` 跑 smoke 配置 8 卡；README 自述「尚未在 950 上跑过、无实测数据」。

### 未解决缺陷（铁律 0：不许写成已修）
- [ ] **CP4 + hit0.5 挂死**：AIVEC 错误（cores 62–69，`507014 aicore execution times out`），rank0 静默。
      **与 K-gather R 修复无关**（该几何下旧代码四 rank 的 R 都是 265，已验证）。修 NaN 后由 `507015+NaN` 变为挂死，形态变了但没修好。
- [ ] **b>1 打包多序列 ks/ke**：`_qli_dg`（`exec_indexer.py:2651`）不拦 nseq>1，`_qsli_dg_chunk` 拦了 ⇒ 前者可达。
      我们的 K 池是**全局位置轴**、DeepInfer 的 `cu_k_cn` 是**按序列轴**，不能直接搬。两条路：
      (a) 池轴改成 cu_k_cn（DeepInfer 忠实，推荐）；(b) 保留全局轴 + 改 ks/ke 公式（仅当每段长度都是 ratio 倍数）。
      b=1 逐位中性。另建议给 `_cp_gather_2seg` 加跨 rank 形状断言。
- [ ] **1/CP 未达成**：8k CP2 0.90×/CP4 0.83×；64k CP4 0.93×。归因（trace）：算力降到 0.42×，但 **kernel 数 2168→7841（3.62×）**，
      每 kernel ~16.3 µs launch 空泡 ⇒ 空闲 27.2→127.6 ms（60.4% 墙钟）；**通信仅占 2.7%**（归零也只到 0.92×）。
- [ ] **099 无法采算子级 trace**：采集侧被降级（device_0/data 仅 1.7 MB、无 `msprof_analysis*.log`），导出只剩 README；
      208 同配置正常。⇒ **算子级 trace 一律在 208 采**，099 只用于性能口径。

### 流程纪律（本次踩过的坑，写死）
1. **同一代码目录，编辑 agent 与跑测必须串行**：本次两次拿到 `NameError: _meta` 的假数据（08:18、08:20），都是树在半写状态。
      跑测前先 `md5sum` 并记录。
2. 同机并行多卡作业：**必须不同卡 + 不同 `HCCL_IF_BASE_PORT`（合法 1024–65520）+ 不同 `HCCL_NPU_SOCKET_PORT_RANGE`**；
      本次 `66200` 越界直接 `parameter error`，两次撞卡造成 `0x4` 建组失败与假挂死。
3. **算子级 trace 只能用标准 `configs/alop_mindstudio.yaml`**；`alop_mindstudio_l0.yaml` 已删除，不得再用。
4. `msprof --export` 必须显式用 `$CANN/tools/profiler/bin/msprof`，且导出后**检查 `mindstudio_profiler_output/` 里有非 README 的 json/csv**，否则视为失败。

### 2026-10-31 追加：CP4+hit 二分结果 + 「完全对齐 DeepInfer」指令

**二分结果（099 干净机，冻结代码 `5a554caf`，点位 `8192@0.5 + CP4`）：**
```
qli=dg     + qsli=vendor : rc=0 nan=0 507014=0 507015=0 err=0 稳态 0.160s
qli=vendor + qsli=dg     : rc=0 nan=0 507014=0 507015=0 err=0 稳态 0.166s
```
⇒ **只要有一侧换回 vendor 就通过**。剩下唯一未验组合是 **dg + dg**。
判读规则（`144_maxk_cp4hit_099` 两臂对照）：
- dg+dg 旧口径（`DS41M_DG_MAXK_FULL=1`）也过 ⇒ 之前的「CP4+hit 挂死」是**环境/争卡**造成，撤销缺陷登记；
- dg+dg 旧口径挂、新口径过 ⇒ 确认 `max_seqlen_k` 全长是元凶。

**代码改动（已落盘，md5 `371370a626d76d7beb76b3ea666cf705`，编译通过）：**
`exec_indexer.py` 的 `fp8_fp4_mqa_logits(max_seqlen_k=...)`：从「全长 `L_full//ratio`」改为「本块 `ke.max()`」，
带 `DS41M_DG_MAXK_FULL=1` 回退。理由：CP4 下最前的块只看得到全长 1/4，内核却按全长铺迭代 ⇒ 单块工作量放 4×、8 块叠加 ⇒ 向量核（cores 62–69）超时。
⚠️ 这是**临时口径**，最终以 DeepInfer `max_kv_len`（InputMetadata 按 chunk 给）为准。

**用户新指令：完全对齐 DeepInfer 实现（不是近似优化）**，已下发执行 agent，四条硬要求：

| # | 对齐点 | DeepInfer 依据（唯一权威） | 我们的现状 |
|---|---|---|---|
| 1 | `max_seqlen_k` 来源 | `_compute_prefill_p_sum`：`max_seqlen_k=self.max_kv_len`，由 InputMetadata 按 chunk 给 | `L_full//ratio`，每块现算（临时改 `ke.max()`） |
| 2 | ks/ke 每 forward 一次 | `input_metadata.py:185 prepare_cp_mirror(...)` → `ks_all/ke_all`，全层共享 | 每层每块现算 |
| 3 | q 分块 | `_topk_prefill`：`for start_q in range(0,total_q,compute_chunk_size)`，每 chunk 一次 p_sum + `topk(end=topk_lens[chunk], output_idx_offset=ks[chunk])` | 一次整段调用 |
| 4 | 候选层 | `use_shared_index_k` + `candidate_mqa_lens`（`_get_candidate_inputs`） | 已有对应物，待核对语义 |

每条都要在 NOTES 里写「DeepInfer 文件:行 → 我们文件:行」对照，回归基线用 `adac76fa` 的 CPU 参考实现（24/24 单序列几何逐位相同）。

**跑测纪律（本轮新增）**：208 已停用（他人占用）；一切跑测走 099 执行器队列；改代码前记录 `md5sum`，改完冻结再跑。

### ★ 2026-10-31 16:40 结案：CP4+hit 不是代码缺陷（撤销登记）

`145_maxk_cp4hit_fix`（099 干净机，冻结代码 `371370a6`，点位 `8192@0.5 + CP4`，dg+dg）：
```
A_maxk_block (max_seqlen_k = 本块 ke.max()) : rc=0 nan=0 507014=0 507015=0 err=0 稳态 0.162s
B_maxk_full  (DS41M_DG_MAXK_FULL=1 旧口径) : rc=0 nan=0 507014=0 507015=0 err=0 稳态 0.162s
```
⇒ **两种 maxk 口径都通过**。结合此前的二分（单侧换 vendor 两臂也都 rc=0），结论：
**「CP4+hit 挂死 (AIVEC 507014)」是 208 上的环境问题（1 vCPU 争抢 + 我并行作业造成的脏状态），不是代码缺陷。撤销该缺陷登记。**

保留的改动：`max_seqlen_k` 按本块 `ke.max()`（`exec_indexer.py`，`DS41M_DG_MAXK_FULL=1` 可回退）。
在本点位它性能中性（两臂都 0.162s），保留是因为它把 CP4 每块的迭代空间从「全长」压到「可见上界」，
大 L 下应体现；**但必须在大点位（64k@0.97 CP4）上验证才可宣称收益**，否则按「无收益」记。

**教训（已写进流程纪律）**：1 vCPU 容器（208）不能跑多卡 + 多作业并行，会把「慢/争抢」伪装成「设备挂死」；
这类结论必须在 099 复核后才能定性。

### ★ 原则（2026-10-31 用户明确）：对齐优先于收益

> 用户原话：「不是有收益才留，必须对齐尽管没有收益」。

- **判据只有一条：与 `/mnt/xds/sfs/DeepInfer` 一致**。性能有没有提升**不是**保留/回退某处改动的依据。
- 反例（本次差点做错）：`max_seqlen_k` 临时改成 `ke.max()`，在 8k 点位两臂都 0.162s「没收益」，我曾打算按「无收益就回退」处理 —— 错。正确做法：按 DeepInfer 契约（`index_attn.py:95/:305` 的 `max_kv_len = runtime_max_model_len // kv_compress_ratio`，每 forward 算一次）实现，**不管是否更快**。
- 同理保留：q 分块（`index_attn.py:117/347-372`）、ks/ke 每 forward 一次、候选层 `get_candidate_compact_lens_ref`/`restore_candidate_indices_inplace_ref` 语义 —— 即使在我们几何下 chunk=8192 只有 2 次调用、量不到收益，也照 DeepInfer 写。
- 收益只在**对齐之后**单独记账（性能不达标属另一件事，不得用「没收益」为由偏离参考实现）。

### ★ 2026-10-31 17:30 CP forward 级 metadata 判据点结案（aa2dde8b，结论已归档）

- **冻结树**：`exec_indexer 023a7658 / exec_ops 92695fbc / dg_candidate c03f3280`；本轮**未再改设备侧代码**。
- **实测**（job 161，099，标准 `configs/alop_mindstudio.yaml`，显式 `msprof --export=on`，4 forward，rank0 prof）：
  CP4 `cp4dp1ep4_L65536_hit0.97_batch1_replayon_prefill` = **24254 行 → 6063.5 task/fwd**，
  全 trace span 766.85 ms ⇒ **191.71 ms/fwd**（forward 间在主流上无空隙，逐 forward 切分随切法变）。
  相对 6177（bb2aadb3）**−113.5（−1.84%）**；相对会话起点 7841 **−22.7%**。
- **`<2000/fwd` 结构性不可达**：只算有名 kernel（剔 3200/fwd 的 SQE/NOTIFY/UBDMA/MEMCPY/EVENT）
  仍有 **2863.5**（1.43×）；主计算流 stream 61 单独 **3194.2**（1.60×）；
  同点位 CP1 单卡（零 CP/EP）**2031.75/fwd > 2000**（job 155）；
  CP 专属开销 2869.3（EP4 1876.5 + CP gather 923.3 + AICPU 69.5）全归零地板仍 3194.2。
  ⇒ 需改判据（建议 ≤CP1 的 1.7×）或减模型每层算子数。
- **已证伪的线索**：stream 58（279.8/fwd）**不是** hidden 预取——`op_summary` 显示它是
  CP 压缩 latent 的 gather（`GatherV2 "2112,512;2106"` 84 + `"144,512;140"` 76 + 对应 `ConcatD`，
  21 层全长 + 19 层 replay，每层 1 次），105.37 ms 是 EVENT WAIT 阻塞计时，kernel 实占 0.41 ms/fwd。
  真正的 hidden 预取在 stream 54，仅 **3 次/fwd（~76 µs、6.6 MB/fwd）**，不是优化项；
  顺带发现 `core/runner.py:201-203` 的 `ctx["idx"]` 守卫是死代码（ctx 无 idx 键），
  代价 <0.1 ms/fwd，本轮不动。
  ⇒ **本节点没有可再消的「空转」通信**，再降任务数只能动模型每层算子数（需逐位证明）。
- 冻结树的复测 + 正确性门禁由同树作业 `162_cp_verify_final.sh`（脚本硬校验
  `023a7658/92695fbc`，不符 exit 9；含 8192 CP1 vs CP2 逐位 / 64k@0.97 CP4 trace /
  8192@0.5 CP4 冒烟）覆盖；结论文件 `NOTES/2026-10-31-cp-fwd-metadata.md`（INDEX 已加行）。
- **job 162 结果（09:37–09:44Z，rc=0）+ ③ 补跑（job 163，17:45 local）**：
  ① CP1 vs CP2 L2/L8/L14 **diff=0**（8191 same+1 empty）、L20 512/512 **diff=0** ✅（20 个 dump 的
  ks/ke 与官方 `prepare_cp_mirror_ref` 逐行一致）；①b dg vs vendor：L2/L8 仅排列差
  （`order_diff=7169/8192`、`k_sha` 同），L14 3365 / L20 512 行集合差 = **topk 顺序口径**级联
  （dg 升序 = `sorted_index=True` = 官方，vendor 不排序；同树逐位复现 ⇒ 非树差）；
  ② 冻结树 trace **24226 行 → 6056.50 task/fwd**（对 job 161 的 6063.5 差 −7 属复现噪声；
  对 `<6000` 判据仍高 56.5/fwd）、逐 forward 墙钟 **176.0/172.4/173.2 ms**（末轮 343.2 含
  profiler stop 尾巴，剔除）——比 job 161 快 18 ms，但同 trace 的
  `AscendCL@aclrtQueryEventStatus` 主机轮询 1387 次 vs job 161 的 16015 次（10×）
  ⇒ **主机状态差、不是代码收益**（不写成改进；无回归证据）；
  ③ 原脚本 `--prof` 参数错误（rc=2 未跑），去掉后重跑 **rc=0、四 rank 无算子错误、
  507014/507015=0、NaN=0**（单次通过，507014 仍按 UNRESOLVED 记；产物
  `tmp/CPVERIFY_HIT05_1031_094529/`，job `163_cp4_hit05_smoke.sh`）。
  ⚠️ `mega_moe` 的 task_time 两采不可比（0.85 vs 2.54 ms/发），引用只用 count。
- **计数口径定案（V1）**：唯一口径 = `AscendTask 全表行数 ÷ nf`（`--iters`，当前 4），
  不剔首轮、不取中位；CSV 与 sqlite 等价（join_miss=0）。「099 不产 HcPre」是误判——
  就地导出降级（kernel_name=N/A、stream_id=65535），**本地完整 msprof 导出**后 HcPre=320 正常。

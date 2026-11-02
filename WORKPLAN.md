### ★ 2026-11-01 09:1x 缺陷登记（P0 验收副产物）：`DS41M_IDX_KERNEL=vendor` + EP=1 单卡 → 507015

来源：P0 实跑 D 臂（job 207）一次同时改了两个变量（卡数 `0` vs `0,1`、内核 `dg` vs `vendor`），
单次结果不可判 ⇒ 补跑 job 208（新树 2×2）、job 209（**改前树** + 就地换文件 + trap 还原）消歧。
证据：`tmp/P0DISAMB_1031_212859/`、`tmp/P0PRE_1031_213130/`，
复盘：`NOTES/2026-11-01-p0-hit-geometry.md` §2.5。

| 臂 | 树 | 内核 | cards | rc | 507015 | 稳态 |
|---|---|---|---|---|---|---|
| A（207） | 新 | dg | 0,1 | 0 | 0 | 0.2555 s |
| E（208） | 新 | dg | 0 | 0 | 0 | — |
| F（208） | 新 | vendor | 0,1 | 0 | 0 | — |
| **I（209）** | **改前** | **vendor** | **0** | **1** | **5** | — |
| G（209） | 改前 | dg | 0,1 | 0 | 0 | 0.2555 s |
| J（209） | 改前 | vendor | 0,1 | 0 | 0 | 0.2557 s |

**结论**：失败组合 = `vendor` 内核 + `EP=1`（单卡），点位 8192@0.5 命中、b=1。
**改前树（`assemble 35d51ed6 / points_runner 1c0ed0f1 / runner 20db7483`，I 臂已 md5 核对）同样失败**
⇒ 既有缺陷，**与 P0 三个文件的改动无关**。
`dg` 单卡正常（E）、`vendor` 双卡正常（F/J）⇒ 既不是"单卡不可用"，也不是"vendor 全废"。
报错形态：`RuntimeError: npuSynchronizeDevice: ... error code is 507015`
（`[geo]` / `[backend]` 已打印、`results.tsv` 未写）。

| ID | 缺陷 | 位置 | 代价 | 处置 |
|---|---|---|---|---|
| **G8** | `DS41M_IDX_KERNEL=vendor` + EP=1 单卡 → aicore 507015 | `ops/exec_indexer.py` 的 vendor 分支（**行号未定位**，需 `ASCEND_LAUNCH_BLOCKING=1` 抓到具体 aclnn 调用） | 单卡无法用 vendor 内核做 A/B 对照；`dg` 单卡不受影响，故不影响 P0 与其后按 `dg` 的验收 | **未修**。既有问题，优先级低于 P1；需要单卡 vendor 对照时再定位 |

### ★ 2026-11-01 18:0x GPT 交接包吸收：P0（口径与守卫）

来源：`https://github.com/Hirrorz/benchmark` commit `67acf8e` 的
`DSV4.1-Flash-Benchmark-Implementation-Handoff.zip`（5 文件）+ 仓库 md。
只读验收见 `GPT设计验收与吸收计划.md`（绿区工作区）；**本轮只做 P0**，不碰数值路径、不上卡。

改动前冻结（md5，**基线 = 工作树，不是 GitHub commit**）：

| 文件 | 改前 | 改后 |
|---|---|---|
| `core/assemble.py` | `35d51ed6946bfd38eb411c46fa1a034a` | `2e5bf14f911858e2d62250f08a3c0cfa` |
| `core/points_runner.py` | `1c0ed0f1e9cb8f6e0d325c0be9e3fcbc` | `b48bde03ecd671264b9f9515beaf183d` |
| `core/runner.py` | `20db7483e07ec42a1b48ca8a37903f6b` | `3e26a52f3747969e81b8f3d4a429ef9f` |
| `tools/test_hit_geometry.py` | （新增） | `e9bba77380d7dd18977959bd8e8d1a3c` |

回滚：`tmp/BAK_P0_20261101_051148/{assemble,points_runner,runner}.py`（改前原件）。

#### P0-1 命中几何收口（缺陷 G1，已修）

- 新增 `assemble.hit_is_on()` / `assemble.check_hit_geometry(cfg, L, where=)`：
  命中开启时要求 **`M = hit_m > 0`** 且 **`R = M + replay_tail <= L`**，否则 `raise`，
  报错里带 `L / M / replay_tail / R / hit`。命中关闭时返回全长 `L`（旧行为）。
- 收口点（4 处，全部走同一个函数，不存在第二套公式）：
  1. `assemble.t_main(strict=True)` —— 供显式构造命中几何的入口用；
  2. `assemble.build_stack` 的命中分支 —— 经 `_run_forward_impl` 的
     `elif meta.get("hit")` 兑现，校验对象是 `meta["L"]`（CP 命中段是**全局 L**，
     段内 `run.seq_len` 只是局部行数，拿它校验会误报；stage A 是前缀段，不套 R 公式）；
  3. `core/runner.py` `run_forward` 的 CP 命中分支 —— 校验**全局** `L_glob`；
  4. `core/runner.py` packed 命中分支 —— 校验 `--seq` 的 `L`。
  另：`core/points_runner.py` 在生成点位几何后**逐批预检**，非法点直接
  `SystemExit("✗ 点位 pi 第 bi 批：...")`，不再静默。
- **为什么这么放**：`run.seq_len` 在单步 decode / verify 里是"本次行数"而不是序列长，
  在 `build_stack` 里无条件 strict 会把这些路径全打红（实测 `L=1024,h=0.1` 已复现
  `R=1062>L`；但 decode 的 `seq_len=1` 与命中口径无关）。所以默认 `t_main(strict=False)`，
  只在**真正用命中几何**的分支 strict。
- 验收：`tools/test_hit_geometry.py` → **PASS**（53 项）。关键两条：
  - 历史矩阵 `20 L × 11 hit = 220 点`：新校验**拒绝集合与旧脚本
    `finish_cp2cp4.sh:164` 的 `continue` 集合完全一致（1 点：`1k@0.1`）**，
    放行 219 点的 `R` 与旧 `T_pass` **逐点相同** ⇒ 收紧校验不改任何历史点位。
  - 非法点报错可定位（`L=1/127/128/1024/8192 × h∈{0.01,0.1,0.5,0.97,0.99}` 全量扫过）。

#### P0-2/P0-3 结果表带几何与指纹（缺陷 G6，已修）

- `results.tsv` **追加** 9 列（旧 5 列不动、旧表照样读得通，读取方都是按列名取）：
  `L / M / R / H / S / keep / actual_hit / plan_hash / backends`。
- 口径：`L`=点位最长序列；`M`=Σ non_cached；`R`=Σ 本次行数；`H = ΣL - ΣM`（前缀）；
  `S = L - R_首段`（query 起点，与 runner 的 `pos_off` 同口径）；
  `actual_hit = 1 - ΣM/ΣL`（**与 target_hit 分开报**，取整差异可见，不改历史 M）；
  `keep`=replay_tail。
- `plan_hash = sha1(L;M;R;rep;cp;world;backends)[:12]` 实现在 `assemble.plan_hash`（纯 CPU 可测）。
- `backends` 由 `assemble.backends_of(cp)` 生成，含
  `idx / qsli / moe / cp_cmp / cp_hit / cp / k_gather / qli` 八项，
  默认值与各模块读 env 的地方同口径（`dg/dg/auto/seg/0/on/chunk`，`auto`=运行期选）。
- 控制台每个点位新增 `[geo] ... target_hit=... actual_hit=... plan_hash=...` 与
  `[backend] ...` 两行（与 `results.tsv` 同源）。
- 验收：`report 里同一点位两次跑新列一致`（同点位几何确定性由 `_geo` 纯算术保证，已在
  测试里覆盖 H+M=L、S+R=L、actual_hit 与 M 自洽）；改 `DS41M_CP_QLI_FULL` /
  `DS41M_IDX_KERNEL` → `backends` 变 → `plan_hash` 变（测试覆盖）。

#### 必须在 WORKPLAN 挂的缺陷（铁律 0：显式化，不写"已修"）

| ID | 缺陷 | 位置 | 代价 | 处置 |
|---|---|---|---|---|
| **G1** | ~~命中几何无 `R<=L` / `M>0` 校验~~ | `core/assemble.py` | — | ✅ 已修（P0-1，见上） |
| **G2** | 第 20 层 compact 只在单序列成立 | `core/runner.py:1490-1493`（`n_seqs == 1`） | b>1 时层 20 多算 `(T-512)` 行的 q/index/attn/MoE | **未修**，待 P1-1。当前 b>1 走非 compact 路径，数值正确但多算 |
| **G3** | `qsli(dg)` 显式拒绝 `nseq>1` | `ops/exec_indexer.py:1619-1626` | b>1 必须回落 vendor 内核 ⇒ 与单序列不同算子图 | **未修**（合规的显式拒绝，不是静默错），待 P1-2 |
| **G4** | MoE 空 origin 用 `min(1,T)`，`T==0` 时 dummy 行数为 0 | `ops/exec_moe.py:1171-1196` | CP 收缩/空 DP 组下可能 barrier 超时（**未实测到，属潜在**） | **未修**，待 P2-1 |
| **G6** | ~~`results.tsv` 无指纹/无几何~~ | `core/points_runner.py:650-660` | — | ✅ 已修（P0-2/P0-3，见上） |
| **G7** | 命中 + CP 只支持单序列 | `core/runner.py:2095-2098`（`len(lens) != 1` → raise） | b>1 命中+CP 不可达 | **未修**（显式报错，不静默），与 P1 合并处理 |

#### 本轮**没做**（明确说明）

- `tools/finish_cp2cp4.sh:164` / `tools/collect_dp_cp_matrix.sh:99` 的 `continue` **保留**：
  它们是**生成点位**的闸，现在与 `points_runner` 的预检同口径（测试已证拒绝集合一致）；
  两者的区别是脚本侧**打印 `[skip]` 到 stderr**（可见），而 `points_runner` 侧改成硬失败。
  不改脚本是为了不打断正在跑的 CP4 全量点位（改了要重算点位表）。
- P1/P2/P3/P4 一律未开工（需另行确认验收口径，见 `GPT设计验收与吸收计划.md` §9）。
- CANN recipes 相关内容**零采纳**（`PERF_RULES.md` 铁律 2）；入图路径**零改动**（铁律）。
# WORKPLAN（2026-10-21 · 当前会话）

## ★ 2026-11-01 交接接手状态（R67 → 本 agent）

- **交接 §6.2① 命中+CP2/CP4 的 507035：已复验关闭**（矩阵 CP2 hit0.3/0.5/0.7/0.9 +
  CP4 hit0.5/0.7 全 rc=0、5070=0；seg 流默认生效且与 full 同速）。
  证据 `NOTES/2026-10-31-handoff-item1-hitcp-fixed.md`、`tmp/HITCP_MATRIX_1031_180652/`。
- **§6.2② 真权重对拍：✅ 通过**（job 203，`run_real.sh -L 16k` 默认 both）：
  prefill id=396/396、717ms；hit0.96875 id=396/396、712ms；官方 next-token id=396 未变。
  证据 `NOTES/2026-11-01-handoff-item2-real-parity-pass.md`、`tmp/REAL_PARITY_1031_181318/`。
- **§6.2④ 诊断清理：已完成**（`shape-iso` / `cp-seg-ab` 块与 `_rope_dump` 全部删除；
  `idx_k slot OOB` 守卫改为默认关、`DS41M_IDX_OOB_CHECK=1` 开——原实现每次 qli 一次设备
  max + 主机同步）。`exec_ops 91ac16e9` / `exec_indexer 262e3415`。
- **§6.2③ 性能记账：✅ 完成**（三拓扑 64k@0.97：noprof 稳态 CP1 210 / CP2 176 / CP4 164 ms；
  trace 结构 idle 3.0%/21.2%/59.8%。`NOTES/2026-11-01-handoff-item3-perf-accounting.md`，
  产物 `tmp/ACCOUNT_1031_182827/`、`tmp/ACCOUNT_TRACE_1031_183132/`）。



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

### ★ 2026-11-01 本机（a5/00006）复核：修掉三个真缺陷

环境侧先补了两处缺件（都不是代码问题）：覆盖层缺 `sympy`（torch 2.13 的
`torch.distributed.tensor` 需要它，缺了 `import torch_npu` 直接失败）——从
`l00972766_vllm` 容器 `docker cp` 出 `sympy`/`mpmath` 到 `/tmp/deps311` 并前置
`PYTHONPATH`；`deep-jit` 头文件树用 `rt/setup_ds_env_headers.sh` 重建。

- [x] **自检 Tier0 假红**：`tools/selfcheck.py` 的 `sys.path.insert(0, ROOT)` 让 repo 根的
      `ops/` 包遮蔽厂商顶层 `ops` ⇒ ds41 索引器 6 个算子 schema 重复注册 ⇒
      `import 厂商算子包 FAIL` + `算子入口 15/23`。改 `append` 后 **Tier0/Tier0.5/Tier1 全绿**
      （25 算子真跑、0 失败）。见 `NOTES/2026-11-01-selfcheck-ops-shadowing.md`。
- [x] **`align_ops` 静默全 0**：只认旧版 `kernel_details.csv` 的 `Type` 列，喂现行
      `op_statistic_*.csv`（`OP Type`+`Count`）时每行取空串 ⇒ 参考侧全 0。已兼容两种格式 +
      认不出列名直接报错；另修 `MhcPost` 别名（正则 `mhc_post` 匹配不到大写形式）。
- [x] **`rope` 直方图翻倍**：`recipes` 把 rope 拆成两处（CP 通信掩盖），而 `opgraph` 的
      `count=3+is_kv_src` 是整层总量 ⇒ 248 vs 真实 124。加 `per_layer=True` 显式标记，
      `run.apply_ops` + `export_model_html` 同语义。修后 **124/124 = 1.00**。
      注意**不能全局去重**：`hc_pre/hc_post/rmsnorm` 的 count 是"每次出现"语义。
      见 `NOTES/2026-11-01-align-ops-and-rope-count.md`。
- [ ] **待重校（本次未改，属既有差）**：`DynamicMxQuant` 250 vs 211、`Cast` 48 vs 150、
      `Add/Mul` 29/15 vs 81/38（`extras` 是按旧 trace 定的）、
      `TransposeQuantBatchMatMul` 40 vs 0（T-054 把 `wo_a` 换成 `deep_gemm.fp8_einsum` 后
      trace 里实际是 `Transpose`，opgraph 仍记旧 kernel 名）。
- [ ] **同类风险（暂未改）**：其余 `tools/*.py` 的 `sys.path.insert(0, ROOT)` 目前不 import
      厂商算子所以无害，但任何人加一句 `ensure_ops()` 就会复现同一假红；
      `tools/probe_megamoe.py:55` 读已删除的 `exec_moe.HIDDEN/INTER`（探针已过时）。

### ★ 2026-10-31 18:15 交接（R68 → 下一任）

- **交接文档**：`/workspace/HANDOFF_mini优化_R68.md` = `/mnt/xds/sfs/l00972766/HANDOFF_mini优化_R68.md`
  （取代 `HANDOFF_mini优化.md` R67 版 §6；当前树 md5、已完成项、缺陷 D1–D8、下一步优先级、运行/采集规则都在里面）。
- **当前树（18:06 冻结）**：`ops/exec_indexer.py=100422b4…`、`ops/exec_ops.py=6bf37add…`、
  `cp/cp_runtime.py=428e296f…`、`core/runner.py=20db7483…`；11 个文件未提交（勿 revert）。
- **最后一跑**：`201_handoff_hitcp2`（099，8192@0.5 CP2，A=seg/B=full）**两臂 rc=0、5070=0、
  211.0/210.1 ms/fwd**；R67 §6.2① 的复现未重现（单次通过，不等于对齐，按交接文档 §5-D2 复验）。
- **接手顺序**：命中+CP 矩阵 → b>1 packed+dg（D1）→ host-bound 第 2 刀（rope event/stream、qli 包装）。
n### ★ 2026-11-01 10:1x P1 执行登记（packed 多请求 × 层 20 compact × qsli(dg)）

依据：`GPT设计验收与吸收计划.md` §5 P1（G2/G3/G7）。详细记录见
`NOTES_2026-11-01-p1-packed-compact.md`（绿区 `agent/` 目录，同标题）。

**代码（md5）**

| 文件 | 改前 → 改后 |
|---|---|
| `core/runner.py` | `3e26a52f` → **`c7bfe26f`** |
| `ops/exec_indexer.py` | `262e3415` → **`c27f50e1`** |
| `ops/exec_ops.py` | `91ac16e9` → **`5febfd98`** |

备份：`tmp/BAK_P1_20261101_0920/`（原始 3e26a52f）、`_0940/`、`_0947/`、`_1000/`、`_1015/`。

**已通过**

1. **G3 关闭**：`_qsli_dg_chunk` 支持 packed 多请求（拼接 K 轴 + 逐请求局部化 + 绝对块号上界）。
   作业 217（`tmp/P1QSLI_20261101_100422`）：`split=0 / b2 / CP1` 下 dg `rc=0`、五层摘要齐全、零算子错误；
   vendor 同几何 `rc=1`（L24 后崩），但 L21/L24 与 dg **逐位相同**（`ab25e9f3…`、`d77c3c33…`、`0df124e1…`）。
   改前症状：L24/L28/L32 各报 `qsli(dg) 暂不支持 nseq=2`（作业 211）。
2. **默认路径零回归**：作业 218（`tmp/P1FINAL_20261101_100838`）默认臂 b=2 的 L20/L21/L39 摘要
   与「不 compact」参照臂**逐位一致**；b=1 仍照常 `4236 → 512（rebase 1 段）`。
3. **G9 已修**：CP 压缩全量复算的 ctx 原写死 `cu=[0,tot]`、`n_seqs=1`，多请求时把 2R 行当一条长 L 序列压缩
   → `aclnnMatmul 161002`（作业 210 复现）。改为按请求给 `cu/row_lens/n_seqs` 后消失（作业 215 起）。

**未通过 / 新登记**

- **G10（新）**：packed 多请求 compact 仍有**逐序列残差**。作业 216（`tmp/P1BIS_20261101_100052`）：
  L20 入口两臂逐位相同（`8cbe73b2`/`bbc09533`）⇒ 残差由层 20 的 cut 引入；L21 起 **seq1** 先分叉
  （sum/absmax 前 7 位一致、sha1 不同），L22 seq1 相对差 `5.5e-4`，L24 seq0 `2.9e-3` / seq1 `2.6e-2`，
  L39 seq0 `1.7e-2` / seq1 `8.8e-1`（只剩参照 12%）。⇒ 不满足语义容差，**不验收**。
  已修的一半：`rebase_window_pool` 的落位原用旧几何 `P_old`，层 21 起按 `P_new` 寻址，
  `sid=0` 与 P 无关 ⇒ 只有 seq≥1 错位（作业 215 把 seq1 的 L21 sum/absmax 拉回与全长一致）。
- **G7 仍未验收**：几何与 kv_compress 已就绪，但数值依赖 G10；默认**仍显式拒绝**多请求命中 + CP。

**默认行为（铁律：没验收的不许当默认）**

| 开关 | 默认 | 含义 |
|---|---|---|
| `DS41M_BOUNDARY_SPLIT` | 1 | 层 20 compact 总开关 |
| `DS41M_PACKED_BOUNDARY` | **0** | 多请求 compact + 命中+CP 多请求（G10 未验收，仅开发复现） |

**下一步**

1. G10 定位：在层 21 的 `_kv_epilog`/`mqsmla` 前打印 seq1 的 `slot` 与窗口池/压缩池内容 sha，与全长臂同位置比对。
2. G10 修好后跑验收矩阵 `CP1/2/4 × b∈{1,2} × hit∈{0.5,0.97}`（rc=0、无 5070x、逐层摘要一致），再把
   `DS41M_PACKED_BOUNDARY` 默认翻 1。
3. 附注：`tools/plot_matrix.py` 旧 tag 解析 bug 仍未修（预存在，与本批无关）。n### ★ 2026-11-01 14:0x P2/A5 进度（MoE 空 origin）+ 两条工具缺陷

**代码（md5 改前 → 改后）**

| 文件 | 改前 → 改后 | 内容 |
|---|---|---|
| `ops/exec_moe.py` | `450c2fc7` → **`1bd74bb9`** | `T==0` 时喂 1 行合法 dummy（零 hidden、id=0 的 topk、均分权重）参与集合通信，算完丢弃；对外仍返回 0 行 |
| `core/points_runner.py` | `b48bde03` → **`9463090a`** | ①`DS41M_ALLOW_EMPTY_DP=1` 允许某 DP 0 批（T=0 脚手架）；②空 DP 不参与裁尾裁决、按多数派补齐；③0 批时不 `torch.cat([])`；④`_S_geo` 空行守卫 |
| `tools/run_hit.sh` | `1408764e` → **`35a6f33e`** | 修 `sys.exit(f"✗ … batches:[{L,hit}] …")` —— f-string 里把 `{L,hit}` 当表达式求值，per-DP 点位走旧格式分支时直接 `NameError: name 'L' is not defined`（作业 220 复现） |
| `core/points_runner.py`（B1/P3-1，前一步） | `b48bde03` → `4adacfcd` | `[segs]` 三段命名打印 + `results.tsv` 新增 5 列（prefix_hit/enc_replay/dec_keep/rows_q0_19/rows_q21_39） |

备份：`tmp/BAK_P2_20261101_1340/`（points_runner 4adacfcd 之前）、`tmp/BAK_P2_20261101_1355/`（exec_moe 与 points_runner 原点）。
`run_hit.sh` 未先备份就覆盖（只改了一个 f-string，回退 = 把 `{{L,hit}}` 改回 `{L,hit}`）。

**已验收**

- B1 + P3-1（作业 219）：`[segs] prefix_hit=4096 enc_replay=0 dec_keep=140 | rows Q0-19=4236 Q20=4236 Q21-39=140`，
  `results.tsv` 19 列，`rc=0`。
- A5 脚手架（作业 221/222）：两 rank 都打出 `[empty-dp] ... 0 token`；`[per-DP] span 不一致 → dummy 对齐：pre=[1, 0] → 1 轮`。
  221 死因已量化：rank1 `torch.cat([])` → `ValueError`，rank0 在 `HcclChannelAcquire` 上等成 timeout 9（`hccl.hpp:134`）。

**未完成（A5 主体）**

作业 222（`tmp/A5EMPTY_20261101_135751`）：`rc=1`，rank1 出现 **507015 ×4 / 507057 ×206**。
逐算子首个失败点（rank1，T=0）：

```
hc_pre      → "Input tensor x's shape should be positive, but x.shape[0] is :0" (exec_ops.py:443)
rope        → "max() arg is an empty sequence" (exec_ops.py:737)
kv_epilog   → "x dimensions must be positive" (exec_ops.py:1523)
mqsmla      → "max(): Expected reduction dim … numel()==0" (exec_ops.py:2573)
wkv / wq_a  → aclnnQuantMatmulWeightNz 161002
gate        → aclnnFusedMatmul 561103
engram_wait → reshape 0 元素到 [0,-1] 歧义
kv_compress → IndexError: list index out of range
```

**结论（设计判定）**：逐算子补 0 行分支的代价 = 把几乎整条算子链都改成 0 行安全，风险高且与"空 origin"语义无关。
正确做法是**编排层短路**：`T==0` 的 rank 跳过所有非集合通信算子，只对**必须参与集合通信**的算子
（MegaMoE、CP 的 all-gather）喂 1 行 dummy，其余直接产出 0 行；并断言 dummy 不进入 `y`/gate/cache。
下一步按此实现，`_mega_moe` 的 dummy 路径要改成**不依赖 `st["topk_idx"]`**（gate 被跳过后没有它）。

**验收标准（不变）**：EP8 下 7 rank 有 token、1 rank T=0 → `rc=0`、无 5070x、无 timeout；
"全员非空"参考的真实行逐位相同。n### ★ 2026-11-01 14:1x A5 追加：短路方案的失败点与正解（设计定稿）

**代码**：`core/runner.py` `c7bfe26f` → **`51bce876`**（`_COLLECTIVE_OPS` + `_empty_T` 短路）；
`ops/exec_moe.py` `1bd74bb9` → **`29fb10de`**（dummy 不再依赖 `st["topk_idx"]`，自造 1 行 ids/weights）。

**作业 223（`tmp/A5EMPTY_20261101_140333`）**：`rc=1`，507057 从 206 降为 **0**（0 行算子链已不再被触发），
但出现 **507015 ×239**，内核侧证据：

```
[r1] MegaMoE rank barrier timeout: rank=1 barrier=0 epoch=1 signal=1 target=2
[r1] ✗ mega_moe: ... LocalScalarDenseNpu.cpp:24 ... error code is 507015
[r0] ([L0] mega_moe 成功：T=2188 ... y_absmax=251658240.000)
[r0] ([L2] mega_moe 起失败 —— 设备状态已被 rank1 的超时破坏)
```

两端对称缓冲**同尺寸**（`T=8192 / 978.0 MB / num_experts=384 × 2 rank`），所以不是缓冲几何不一致；
`target=2 signal=1` = 有一个 rank 没到这一道 barrier。原因：短路把 `engram_fetch/engram_wait/engram_proj`
也一起跳过了，而它们跑在**逐层独立的 2-rank HCCL 通信域**上（日志：`engram_tp_group_1（2 rank）`）
⇒ rank0 在 engram 域上等 rank1，rank1 早已去下一层 ⇒ MegaMoE 的 barrier 顺序错位。

**结论**：`T==0` 不能用"跳过算子"实现 —— 必须让该 rank 以 **1 行虚拟请求**走**完整算子栈**
（这样 engram/MoE 的集合通信轮次与形状都和正常 rank 对齐），出口再把结果截断成 0 行。
需要的几何一致性（这是未完成的部分）：

1. `T = 1`（dummy 行），`inputs = zeros(1, H)`；
2. ctx/metadata 同步成"1 行虚拟请求"：`cu_seqlens=[0,1]`、`row_lens=[1]`、`n_seqs=1`、
   `seq_lens=[L_full]`、`pos_offset` 保持不变（命中段语义仍是尾段）；
3. 池/缓存的写入只落在本 rank 自己的行 0（CP=1 时池是 rank 本地，不跨 rank 污染）；
4. 出口 `x = x[:0]`，并断言 `y` 为 0 行、无 nan/inf（不污染 gate/cache）；
5. 验收：EP8 下 7 rank 有 token、1 rank T=0 → `rc=0`、无 5070x、无 timeout；
   "全员非空"参考的真实行逐位相同。

实现位置：`core/runner.py::_run_forward_impl` 入口（T 归一 + 几何覆盖）+ `core/points_runner.py`
（空 DP 的 `_packed` 用 1 行虚拟几何建、pass 结束后丢弃）。**当前短路代码保留在树里但只在
`T==0` 时触发，生产路径不受影响**（`T==0` 只有 `DS41M_ALLOW_EMPTY_DP=1` 的测试点位才会出现）。n### ★ 2026-11-01 14:1x A5 结论：空 origin「不挂死」通过，「不污染」未通过（G12 登记）

**代码**（md5 改前 → 改后）

| 文件 | 改前 → 改后 | 内容 |
|---|---|---|
| `core/points_runner.py` | `9463090a` → **`20d6794f`** | 空 DP 改走 **1 行虚拟请求**（`row_lens=[1]`、`cu=[0,1]`、**关 hit**、`T_run=1`），跑完整栈；结果不写 `results.tsv`（`_SkipEmptyOrigin`） |
| `core/runner.py` | `51bce876` → **`187f332f`** | `T==0` 的兜底短路加**显式告警**（不再静默跳算子；正常路径已不走它） |

备份：`tmp/BAK_P2_20261101_1435_{points,runner}.py`。

**作业与判据**

| 作业 | 内容 | 结果 |
|---|---|---|
| 224（`tmp/A5EMPTY_20261101_141102`） | 2 卡 per-DP：rank0 有 1 批、rank1 **0 批** | **rc=0**，**零 5070x、零算子错误**；rank1 日志 `本轮 T=1` + `[empty-dp] 空 origin（1 行虚拟请求）跑完，不写 results.tsv`；两 rank 稳态 0.311 s |
| 225（`tmp/A5CLEAN_20261101_141421`） | G=两 rank 均有 token / F=rank1 空，同 `4096@0.5/b1` | 两臂都 rc=0；**rank0 摘要：L0 相同、L20/L39 不同** |

rank0 逐层摘要（污染判据）：

```
G: L0 cd8e956d | L20 660e0020 sum=-2.979941e17 | L39 65f6e4af sum=-4.151640e17
F: L0 cd8e956d | L20 d7e47c4f sum=-4.071081e17 | L39 babeb4cd sum=-8.114895e17
                    ↑ 相对差 3.7e-1        ↑ 相对差 9.5e-1
```

**判定**
- ✅ 空 origin 不再挂死/报错（A5 的主要工程价值：`HcclChannelAcquire timeout 9`、`rank barrier timeout target=2 signal=1` 都不再出现）。
- ❌ "dummy 不污染输出"**不成立**：L0 逐位相同（说明 rank0 的输入与种子口径没变），
  从 L1 起分叉。首个可疑环节是 **engram 的逐层 2-rank 主机窗口**（日志：`engram_tp_group_1`、
  "厂商主机窗口按通信域池化，每层一块"）—— rank1 的虚拟行会往**对端可读的共享表**里写一条，
  位置/投影与 rank0 的真实行不同 ⇒ 污染对端。

**下一步（G12）**
1. 先证伪/证实 engram 假设：同一几何下把 `engram` 关掉（或 `DS41M_ENGRAM=0` 若有）再跑 225 两臂，
   若 L20/L39 恢复逐位相同 ⇒ 就是 engram 共享窗口；否则继续二分（MoE all2all / 压缩池）。
2. 若确认是 engram：空 origin 的虚拟行必须写成"对共享表**幂等**的条目"（同 id、同位置、同投影），
   或让该 rank 的 engram 只**读**不写（需厂商算子支持"只参与不写入"的能力位）。
3. 在此之前，`DS41M_ALLOW_EMPTY_DP` 保持**仅测试用**；A5 的验收状态记为「机制可用、判据未过」。n### ★ 2026-11-01 14:2x P3-2 验收 + C6 总账

**P3-2（metadata 复用 vs 重建，分列）—— 通过**

| 文件 | 改前 → 改后 |
|---|---|
| `ops/exec_ops.py` | `5febfd98` → **`9a67228f`** |
| `core/runner.py` | `187f332f` → **`2b5b0aec`** |
| `core/points_runner.py` | `20d6794f` → **`d18d193f`** |

内容：`META_STATS` 计数（`geo_*` = 进程级 `_meta`，`fwd_*` = forward 级 `_meta_fwd`）+ `meta_reset/meta_snapshot`；
runner 每 pass 归零并在出口打印 `[meta-reuse]`；`results.tsv` 增 4 列（geo/fwd 的 hit/miss）。

作业 226（`tmp/P3META_20261101_142306`），`rc=0`、零算子错误：

```
[r0] [meta-reuse] pass=1 T=4236 geo=399/51(hit/miss) fwd=9/3(hit/miss) rebuild=54
[r0] [meta-reuse] pass=2..5               geo=448/0      fwd=9/3      rebuild=3
[r0] digest L20 3e64f004  L39 87009fea   ← 与同几何历史值一致（零回归）
results.tsv：23 列（tag…backends, prefix_hit…rows_q21_39, meta_geo_hit…meta_fwd_miss）
```

解读：稳态每 forward 只有 **3 次** metadata 重建（全是 forward 级 `_meta_fwd` 的 miss），
进程级几何缓存命中 448 次、0 重建 ⇒ 「ms/forward」里 metadata 重建的成本可忽略，
P3-2 要的"两口径分列"已可回答。

**C6 —— GPT §14 的 18 行逐文件对账已产出**

交付物：`agent/GPT-S14-文件级改造总账.md`（绿区，文档）。
口径：责任不省略、落点可改。统计：**已落 5 / 进行中 6 / 待做 5 / 待裁决 1**（改写 2 行均写明落点与理由）。
缺口按依赖排序：①`cp_state` 去标量 `pos_offset`（逐请求坐标，同时解 G10 的一半）；
②C5 不可变 stage view；③C4 新 plan 层（必须排在 ①②之后，否则并列第三套几何）；
④B5 稀疏 halo / 真流式 chunk / kv_manager 语义；⑤G12、空段；⑥C1 recipes 待裁决；⑦报表与测试。

**下一步**：C1 仍等用户裁决；代码线继续按缺口①（`cp_state` 逐请求坐标）→ G10 的路线推进。n### ★ 2026-11-01 14:3x G10 定位完成：分叉点 = 层 20 的 `mega_moe`（b>1 特有）

**诊断工具（已落树，默认关）**

| 开关 | 作用 | 落点 |
|---|---|---|
| `DS41M_CUT_DIGEST=1` | cut 出口打印每请求**首/末保留行**摘要（判 cut 是否选错行） | `core/runner.py::_boundary_cut_state_ms` |
| `DS41M_ROWDIG_AT=<层>` | 该层**每个算子之后**打印末行 sha1 + sum（末行 = 最后一个请求的最后一个 token，两臂是同一个逻辑 token） | `core/runner.py` 算子循环 |

`core/runner.py` md5：`2b5b0aec` → **`478accfe`**（cut-digest）→ **`f9a9fd55`**（rowdig）。

**作业 227（`tmp/G10CUT_20261101_142849`）**：cut 出口
`seq1[512:1024] first=1d2217d2 last=f2a4bae9 sum_last=-1.475683e+02`
—— 与层 20 后续所有"只读"算子的末行 sha 完全相同 ⇒ **cut 选行正确，没有选错行**。

**作业 228（`tmp/G10ROW_20261101_143249`）**：层 20 逐算子对拍（A=compact T=1024 / B=全长 T=8472）

| 算子 | A 末行 | B 末行 | 结论 |
|---|---|---|---|
| `rmsnorm` / `gate` / `topk_gate` | `e1c7def9` | `e1c7def9` | 一致 |
| `hc_pre`（两枚） | `436e7156` / `44a68d23` | 同 | 一致 |
| `kv_norm` / `kv_epilog` / `kv_compress` / `wkv` / `rope` / `wq_a` / `wq_b` / `q_norm` / `idx_prolog` / `qli` / `mqsmla` / `wo_a` | `f2a4bae9` | `f2a4bae9` | 一致 |
| **`mega_moe`** | `b37c376a`，`sum=+1.772381e+11` | `bfd89eee`，`sum=+1.772547e+11` | **首个分叉**，相对差 **9.4e-5** |
| `wo_b` / `hc_post`（层 20 尾） | `36d30555` / `44cbdfae` | 随之分叉 | 差异被放大 |

**关键对照**：单请求（b=1）的 compact 与全长段是**逐位一致**的（历史上四臂 `[packed-digest]`
sha1 完全相同，P0 也复核过）。同一批算子、同样的"行数从 4236 变 512"，b=1 逐位、b=2 不逐位
⇒ 根因不是"批大小影响 GEMM 分块"，而是 **b>1 时 MoE 的跨请求组合**（all2all 里两个请求的
token 一起排序/分发，compact 后少了一半行，专家分组与求和顺序变了）。

**判决（G10）**
- 机制正确：cut/几何/窗口池/索引行账全部对（12 个算子逐位一致）。
- 数值不等价：MoE 的 9.4e-5 相对扰动经 19 层 1e17 量级运算放大到 L39 的 **8.8e-1**。
- 这类放大在本仓是已知现象（`runner.py` 原注释："只在末层差异 = 数值混沌/求和顺序"）。
- 因此 **compact 的 b>1 不能声称"与全长段对齐"**。当前默认关（`DS41M_PACKED_BOUNDARY=0`）是正确姿态。

**两条可选出路（需决策）**

1. **工程化收口**：把判据改成"**同一进程内** compact 开/关的末行相对差 ≤ 容差"，并给出容差表；
   但实测 L39 已达 8.8e-1 ⇒ 不满足，等于放弃 b>1 compact。
2. **结构改造**：让 b>1 的 compact 段也保持 MoE 的**跨请求组合不变**（即 compact 只裁 KV/attn 的
   行，MoE 仍按全长批喂）—— 代价是 MoE 段收益归零（MoE 是层 20 的主要成本之一），需要重新评估
   这条优化的净收益。

在决策前不再改动 compact 的默认行为；G10 保留为**未验收缺陷**（默认 opt-in、有复现命令）。n### ★ 2026-11-01 14:3x B6 验收通过：`tools/sweep.py`（dry-run 矩阵枚举 + 定点复跑）

**新文件**：`tools/sweep.py`（md5 **`1f9fab00`**，6095 B）

口径（按 `GPT设计验收与吸收计划.md` §4.2-B6 的改写）：**不做黑盒 autotune 引擎**，
只做「笛卡尔积枚举 + 可解释点位 + 复用 run_hit 的定向复跑」：

- 轴：`L` / `hit` / `b`(batches) / `cp`(=`DS41M_CP_SIZE`) / `replay`；
- `cp` 是**进程级**环境变量 ⇒ 按 cp 分组，每组一次 `run_hit.sh`（权重/进程组只加载一次）；
- `--dry` 打印的命令与 `--run` 实际执行的 argv **同一份构造**（不会 dry/实际漂移）；
- 结果落在 `<out>/cp<N>/results.tsv`（run_hit 自己写，23 列含几何/指纹/后端/metadata 计数）。

**验收**

| 步骤 | 命令 | 结果 |
|---|---|---|
| 本地 dry | `sweep.py --matrix 'L=8192,16384;hit=0.5;b=1,2;cp=1,2' --dry` | 8 点 / 2 组，命令打印正确 |
| 蓝区 dry | 同上，1 点 | 打印 `bash …/tools/run_hit.sh --points-json '<points:1 点>' --cards 0,1 --iters 4 -o /tmp/sweep/cp1 --noprof-only` |
| 蓝区实跑（作业 229，`tmp/SWEEPB6_20261101_143821`） | `--matrix 'L=8192;hit=0.5;b=1;cp=1' --run` | `sweep rc=0`；`run/cp1/results.tsv` 有该点位一行；稳态 **0.2555 s**（与 P0 基线 0.2555 s 一致 ⇒ 零性能回归） |

`results.tsv` 行样例（节选）：`… batch1 … 8192 4096 4236 4096 3956 140 0.500000 965de9e6b0fb
idx=dg,qsli=dg,moe=deepgemm,cp_cmp=seg,… 4096 0 140 4236 140 448 0 9 3`

**用法（写进 tools 惯例）**

```bash
# 只看要跑什么（不占卡）
python3.11 tools/sweep.py --matrix 'L=8192,16384;hit=0.5,0.97;b=1,2;cp=1,2' --dry
# 真跑（按 cp 分组，各自一个 run_hit 进程）
python3.11 tools/sweep.py --matrix 'L=8192,16384;hit=0.5,0.97;b=1,2;cp=1,2' \
    --out /mnt/xds/sfs/l00972766/tmp/SW1 --cards 0,1 --iters 4 --run
```

**注意**：`--cp 2/4` 的臂会真的按 `DS41M_CP_SIZE` 起多卡进程组，占用更多卡；
`--dry` 不占卡，建议先 dry 再定点复跑（这也正是 GPT「自动执行」一节的可复现版本）。n### ★ 2026-11-01 14:5x B2 验收通过（`hit_alignment = token | block[:BS]`）+ 208 通道实测

**B2（GPT §12.2 `hit_alignment`）**：`core/assemble.py` `2e5bf14f` → **`709b65dc`**；
`tools/test_hit_geometry.py` `b862b843` → **`2ad27498`**。备份 `tmp/BAK_B2_20261101_1530/`。

- 新增 `assemble.hit_alignment(cfg)`：`cfg["hit"]["alignment"]` 优先、其次 `DS41M_HIT_ALIGN`，
  默认 **`token`（逐位不变）**；`block[:BS]`（默认 BS=128）把非缓存段**向上取整到块边界**。
- `hit_m` 换算完再对齐；几何收口 `check_hit_geometry` 仍**显式拒绝** `M==0` / `R>L`（不 clamp）。
- `backends_of()` 增加 `align=token|block128`，结果表可分辨口径。

**CPU 验收（蓝区，无卡）**：`rc=0`、**61 PASS / 0 FAIL**（原 53 + B2 新增 8 条）。
新增断言：默认 `M(8192,0.97)=246` 不变；`block:128` 下 `L=1024,h=0.53 → 481→512`；
`L=8192,h=0.97 → 246→256`；对齐把 `R` 顶过 `L` 时**显式拒绝**；非法 alignment **报错不静默**；
alignment 改变 ⇒ `plan_hash` 改变。

**实跑验收（作业 230，`tmp/B2RUN_20261101_144514`）**：两臂 `rc=0`、零 5070x

| 臂 | `[geo]` | `[backend]` | `plan_hash` | 稳态 |
|---|---|---|---|---|
| `token` | `L=8192 M=3850 R=3990 H=4342 S=4202` | `…,align=token` | `499aefa3db1a` | 0.2395 s |
| `block:128` | `L=8192 M=3968 R=4108 H=4224 S=4084` | `…,align=block128` | `f81bde888a1d` | 0.2472 s |

`M` 3850→3968（+118，128 的块取整）、`R` 3990→4108、`actual_hit` 0.5300→0.5156 —— 符合块对齐语义。

**208 通道实测（环境事实）**

- `python 工具/a5.py "…"` → `liteserver-hps-tokens-001-00006.novalocal`，8×Ascend950DT，**空闲**（util 0%）。
- 三台共享 `/mnt/xds/sfs`（同一份代码/产物），所以改 099 立刻在 208 可见（交接文档同一结论）。
- **但 208 现在跑不起来**：`ModuleNotFoundError: No module named 'sympy'` ⇒
  `Failed to load the backend extension: torch_npu`（作业 `tmp/B2HOST208`，rc=1，4 s 即失败）。
  099 的 `env_pack.sh` 自带这些纯 py 依赖，208 缺 `sympy/pyyaml/requests`。
- 结论：**要真正双机并行，需先在 208 装这三个纯 py 包**。建议装到**私有目录**
  `/mnt/xds/sfs/l00972766/deps208`（`pip install --target`）+ 运行时 `PYTHONPATH` 前置，
  不动系统 site-packages、不污染他人环境，删目录即可回退。**安装前需用户点头**（写共享存储）。n### ★ 2026-11-01 16:0x 208 打通：共享依赖 + env_pack 自动接入（双机可并行）

**背景**：`/mnt/xds/sfs` 是三台（099/208/246）共享的网盘式挂载（同一份代码/产物）。
208 = `liteserver-hps-tokens-001-00006.novalocal`，8×Ascend950DT，空闲；通道 `工具/a5.py`。
208 原缺 `sympy/pyyaml/requests`（`torch_npu` 因此在 `import` 阶段就失败：
`ModuleNotFoundError: No module named 'sympy'` → `Failed to load the backend extension: torch_npu`，4 秒即 rc=1）。

**修法（不动系统 site-packages）**

1. 从 099 的**主机本地**站点包把三个包连同依赖拷到共享盘：
   `/mnt/xds/sfs/l00972766/deps_py`（92 MB）——
   `sympy(+mpmath)`、`requests(+urllib3/certifi/idna/charset_normalizer)`、`pyyaml(+_yaml)`；
2. `env_pack.sh` md5 `93833966` → **`24834b7c`**：⑥ 段的依赖搜索链新增
   `elif [ -d "/mnt/xds/sfs/l00972766/deps_py/yaml" ]; then DEPS=/mnt/xds/sfs/l00972766/deps_py`，
   并把缺依赖的告警改成给出该路径。备份 `tmp/BAK_208_20261101_1600_env_pack.sh`。
   ⇒ 两台机器 `source env_pack.sh` 后都会打印 `[env_pack] deps: /mnt/xds/sfs/l00972766/deps_py`。

**跨机一致证据（同一臂，B2 token 口径 `8192@0.53/b1/r1`）**

| 项 | 099（作业 230） | 208（`tmp/B2HOST208b`） |
|---|---|---|
| `[geo]` | `L=8192 M=3850 R=3990 H=4342 S=4202` | **完全相同** |
| `plan_hash` | `499aefa3db1a` | **`499aefa3db1a`** |
| `backends` | `…,align=token` | 同 |
| 稳态 | 0.2395 s | 0.240 s |
| rc | 0 | **0** |

**以后怎么在 208 跑**（脚本放共享盘，208 直接执行；128 KB 级脚本用 `_push_b64.py` 推到
`/mnt/xds/sfs/l00972766/tmp/`，再 `python 工具/a5.py "bash /mnt/xds/sfs/.../xxx.sh"`）：

```bash
# 1) 写脚本到共享盘（用 099 通道落盘）
python 工具/_push_b64.py _p6/run208.sh /mnt/xds/sfs/l00972766/tmp/run208.sh
# 2) 208 上跑（长任务 nohup，避免 a5.py 的 socket 超时）
python 工具/a5.py "nohup bash /mnt/xds/sfs/l00972766/tmp/run208.sh > /tmp/run208.log 2>&1 & sleep 2; echo ok"
```

**注意（并行纪律）**：两台共享同一份**代码树**，所以
①「改代码」与「跑作业」仍要串行（改前 `md5sum` 冻结，跑完再改）；
② 真正可以并行的是**只读代码的验证作业**（不同产物目录 `-o`）；
③ 208 是共享机，有他人占用要让出（099 是用户独占）。n### ★ 2026-11-01 15:1x G12 分层定位 + 一次误覆盖事故（已恢复）

#### ① G12：污染始自**层 1 的 engram 算子组**（作业 231，`tmp/G12LAY_20261101_150904`）

同点位 `4096@0.5/b1/r1`，两臂 rank0 逐层末行摘要：

| 层入口 | G（两 rank 有 token） | F（rank1 空，1 行虚拟请求） | 一致 |
|---|---|---|---|
| L0 | `cd8e956d` | `cd8e956d` | ✓ |
| L1 | `d728c55c` | `d728c55c` | ✓ |
| **L2** | `24876de5` | `38d61ceb` | **✗ 首次分叉** |
| L3 | `73d6e322` | `2bee8e7b` | ✗ |
| L20 | `660e0020` | `d7e47c4f` | ✗ |

L0/L1 完全一致 ⇒ 分叉由**层 1 的算子**引入（L1 = 第一个 engram 层；MegaMoE 要到层 20 才出现，
所以**不是 MoE**）。与"engram 的逐层 2-rank 主机窗口按通信域池化、对端可读"的假设一致。
两臂 `rc=0`、零 5070x。

结论（G12）：空 origin 的虚拟行会在层 1 往**对端可读的 engram 表**写一条，位置/投影与对端真实行
不同 ⇒ 污染。要修需要厂商算子提供"参与但不写入"的能力位，或让虚拟行写成对共享表**幂等**的条目。
在拿到该能力前，`DS41M_ALLOW_EMPTY_DP` 保持**仅测试用**。

#### ② 误覆盖事故：`meta/policy.py`（已恢复，铁律 0 登记）

- 我准备新增"命中策略 façade"时，把新文件推到了 **`meta/policy.py`**，而该文件**仓库里本来就有**
  （"Policy —— 规则/策略的唯一入口"，加载 `configs/policy.yaml`，被 `core/runner.py`、
  `meta/metadata.py` 引用）。推送把原文件覆盖了（`5963821d…` → `9563263b…`）。
- 恢复：`git cat-file -p HEAD:meta/policy.py > meta/policy.py` → 回到 **`5963821d631700ec8811f4914f61969e`**，
  `py_compile` OK，17 个顶层成员（`CP`/`REPLAY`/`MOE_BACKEND`/`EnvBool`/`EnvInt`/`EnvStr`…）齐全。
- 复验：CPU 套件 **全绿** —— `test_hit_geometry` **61 PASS / 0 FAIL**、`test_cp_layout` PASS、
  `test_cp_state` PASS、`test_batch_replay` PASS（`rc=0`）。
- 教训（已改流程）：**推"新"文件前先 `ls` 目标目录并看输出**，不要在同一条命令里
  "ls + 推"。本次 `ls` 与 push 写在同一调用里，输出顺序上我看不到就推了。

#### ③ C6 第 2 行（`meta/policy.py`）的实际结论：**本仓已满足**

GPT §14 要求"`meta/policy.py`：归一化配置，requested/actual 策略；移除热路径 env 决策"。
本仓的 `meta/policy.py` **正是这个角色**（`configs/policy.yaml` 声明式规则 + env 覆盖 +
`dp = ep // cp` 推导 + `EnvBool/EnvInt/EnvStr` 类型化包装），runner/metadata 都在用。
⇒ 该项**不需要新建模块**，只需（可选重构）把 `assemble` 的命中口径（含 B2 的 `align`）搬进来，
避免"命中策略在 assemble、配置策略在 policy"两处。已记入 C6 总账的"改写"栏。n### ★ 2026-11-01 15:2x B4 验收通过（CP 块对齐 `cpblock[:BS]`）+ P5 真权重首跑失败待查

**B4（GPT §7/§14 `partition=mirror_aligned`）—— 通过**

落点决定：`cp_layout` 的 mirror 分块是"每 rank 两段、各 `ps=ceil(R/(2·cp))` 行"，
要让两段正好落在 KV 块边界 ⇔ **`R % (2·cp·BS) == 0`**。所以 B4 做在**几何层**
（`DS41M_HIT_ALIGN=cpblock[:BS]`），不改分块公式 —— 后者在 `R < 2·cp·ps` 时会让
head/tail 段重叠或丢尾（`inverse_reverse` 只能指一个全局行），等于引入第三套几何。

| 文件 | md5（改前 → 改后） |
|---|---|
| `core/assemble.py` | `709b65dc` → `e538d2f8` → **`1118b1f0`**（含两处修正，见下） |
| `tools/test_hit_geometry.py` | `2ad27498` → `2d648df4` → **`be6b22b2`** |

修正 1（真 bug）：`hit_m` 的 cpblock 分支要读 `replay_tail_of(cfg)`，而 `points_runner._m_of`
传的是**只有 `hit` 的最小 dict** ⇒ `KeyError: 'model'`（作业 232 的 cpblock 臂 rc=1 就是这个）。
修法：`swa_block_tokens` 改用 `(cfg.get("model") or {})`，容忍缺失。
修正 2（显示）：`backends_of` 只把 `block` 映射成 `blockN`，`cpblock` 落到 `token` ⇒ 结果表分不清口径；
改为三分支，并在 CPU 套件加断言 `align=cpblock128`。

CPU 验收（蓝区）：`rc=0`、**67 PASS / 0 FAIL**（66 + 1）。
新增断言：`cpblock cp=2 → R 对齐到 512 倍数（4468+140=4608）`、`每 rank 两段各 1152=9×128`、
`cp=1 → 256 倍数`、`2·cp·BS > L 时显式拒绝`、`三种口径 plan_hash 互不同`、`backends 含 cpblock128`。

实跑验收（作业 233，`tmp/B4RUN_*`，CP2 + `DS41M_CP_HIT=1`，点位 `8192@0.53/b1/r1`）：

| 臂 | `M` | `R` | `actual_hit` | `plan_hash` | `backends` | rc | 稳态 |
|---|---|---|---|---|---|---|---|
| `token` | 3850 | 3990 | 0.5300 | `6d47cebbae8b` | `…,align=token` | 0 | 0.2000 s |
| `cpblock:128` | 3956 | **4096** | 0.5171 | `e5868f4e2b04` | `…,align=cpblock128` | **0** | 0.2029 s |

`R=4096 = 8×512` ⇒ 每 rank 两段 `ps=1024=8×128`（块对齐成立）；两臂零 5070x、零算子错误。

**P5（真权重签字）—— 首跑失败，原因待查**

208 上 `bash tools/run_real.sh -L 16k --mode prefill --iters 3 --warmup-iters 1`：
驱动表头打印后 `rc=1`，**stdout 里没有错误正文**（自然路由检查是 OK 的）。下一轮：
去 `run_real.sh` 自己的产物目录找 rank 日志（`logs/rank*.log`），并核对
`parity/ids_16k.pt` 与权重目录是否在 208 可见（`PRE.log` 里 weights 探测为空）。

**已确认可用的探针（默认关，别忘）**

| 开关 | 作用 |
|---|---|
| `DS41M_HIT_ALIGN` | `token`（默认）/ `block[:BS]`（B2）/ `cpblock[:BS]`（B4） |
| `DS41M_CUT_DIGEST` / `DS41M_ROWDIG_AT` | 层 20 的 cut 行摘要 / 逐算子末行摘要（G10 定位用） |
| `DS41M_ALLOW_EMPTY_DP` | 空 DP 脚手架（A5/G12，仅测试） |
| `DS41M_PACKED_BOUNDARY` | packed 多请求 compact（G10，默认 0） |n### ★ 2026-11-01 15:3x P5 根因修复：`run_real.sh` 的 prefill 口径写错（已改）

**事故**：208 上 `bash tools/run_real.sh -L 16k --mode prefill` 首跑 `rc=1`，驱动表头打印后无错误正文。
真因在它自己生成的配置里（`tmp/P5REAL_*/run.log`）：

```
ValueError: packed 命中几何：命中几何非法（要算的行数超过序列长）
  —— L=16384、M=16384、replay_tail=140、R=16524
  （hit={'ratio': 0.0, 'source': 'real', 'replay_tail': 140, 'non_cached_tokens': 16384}）
```

`run_real.sh:395` 无条件写 `non_cached_tokens = round((1-h)*L)`；prefill 口径 `h=0` ⇒ `nc=L` ⇒
`hit_is_on()` 为真、`M=L`、`R=L+KEEP>L` ⇒ P0 的几何收口（A1/B3 吸收）按铁律 0 **显式拒绝**。

**裁决**：这是**调用方的配置 bug**，不是收口过严 —— 仓库既有断言（`test_hit_geometry`：
"非法点必须拒绝 L=8192 h=0.0 nc=8192；M=8192 R=8332"）就是这个契约。我先试过让
`M>=L` 直接返回 L，结果**打翻 6 条既有断言**（`放行 L=1 h=0.01`、`非法点必须拒绝` 等），
因此**回退该改动**，改调用方。

| 文件 | md5（改前 → 改后） |
|---|---|
| `tools/run_real.sh` | `8ffe3e30` → **`fd6c4e6e`** |
| `core/assemble.py` | 回退到 **`1118b1f0`**（等于 B4 修好后的版本，未新增 M>=L 特例） |
| `tools/test_hit_geometry.py` | `be6b22b2` → **`2653ae25`** |

`run_real.sh` 的改法（注释同步写清）：

```python
cfg['hit']['non_cached_tokens'] = (
    int(round((1 - float(h)) * int(L))) if float(h) > 0 else 0)
```

prefill ⇒ `ratio=0 + nc=0`（命中关闭），几何即全长 L。

**新增 CPU 断言（蓝区 70 PASS / 0 FAIL）**

- `nc==L 被拒绝且信息含 L/M/R` —— 把"严格契约"钉死；
- `ratio=0 + nc=0（命中关闭）→ 返回全长 L`；
- `该口径 hit_is_on=False（几何走全长，hit_m 不参与判定）`。

**P5 复跑状态**：208 上重跑已加载真权重并跑到 `L30/39`（约 73%，权重累计 ~105 s），
说明口径修复生效。最终签字（rc / 精度 gate / 与官方基准对比）下一轮记录。n### ★ 2026-11-01 15:4x P5 真权重签字：**精度 gate 通过**（prof 轮 rc=0）

208（`liteserver-hps-tokens-001-00006`）上 `bash tools/run_real.sh -L 16k --mode prefill`，
修好 prefill 口径后（`run_real.sh` `8ffe3e30` → **`fd6c4e6e`**）重跑：

| 轮次 | rc | 用时 | trace | 结论 |
|---|---|---|---|---|
| prof-prefill | **0** | 272 s（4m32s） | 8/8 份 → `/mnt/xds/sfs/l00972766/real_prof/L16384/prefill_prefill` | **精度 gate ✅ PASS（top-1 与官方一致）** |
| noprof-prefill | 进行中 | — | — | 稳态数字下一轮取 |

旁证（prof 轮 rank 侧）：`执行 827 个算子调用（24 种），跳过 0 个`；
`engram_fetch/wait/proj` 各 2 次（层 1、层 14）、`mega_moe` 40 次、`kv_compress` 4 次、
`idx_prolog` 8 / `qli` 4 / `qsli` 4 / `idx_k_write` 1；`隐式 dyn_mx_quant 39`；
显存峰值 57.39 GiB（3.587 MiB/token，L=16384）。
8 个 rank 都出现 `[engram] 层 1 专用进程组就绪（2 rank…）`→ 真权重路径的 engram 域也在。

**P5 结论（先记账）**：真权重口径（EP8、自然路由、`engram_tp_size=8`、resident 权重）
**能跑通且 top-1 与官方一致** —— 这是"改完东西没把模型跑坏"的签字。
稳态 ms/forward 与官方 2394.7 ms 的对比待 noprof 轮结束再填。

### ★ 2026-11-01 15:4x C3 核查：`local_quant_ag` **在参考实现里不存在**

- 本仓：`ops/exec_ops.py`、`core/runner.py` 里**没有** `local_quant_ag`（grep 无命中）。
- 参考：`/mnt/xds/sfs/DeepInfer` 存在，但 `grep -rln "local_quant_ag"` **无命中**。
⇒ GPT §B/C 里的 `local_quant_ag` 不是现成算子名，而是它自己提的**通信策略**（本地量化后
all-gather），要实现就得**新写一个分布式算子**（量化 + 集合通信 + 反量化），没有参考实现可对拍。

**C3 状态：未做，且不是"改默认值"就能完成的**。按铁律 1（性能等价或更好 + 数值等价才能改默认），
正确路径是：
1. 先写出该 op 的**单机等价实现**（本地量化 → AG → 反量化，可与"先 AG 再量化"逐位对拍）；
2. 用 §231 那类 A/B（同点位、开/关该 op）证明数值等价 + 墙钟更快；
3. 才有资格谈"作为默认通信"。
在此之前不改默认，也不新建半成品算子。需要用户确认是否投入这条线（工作量 ≈ 一个新 op + CP 验证）。n### ★ 2026-11-01 15:4x C4 第一步落地：`cp/cp_plan.py`（计划对象 + 六条路由声明）

**新文件**

| 文件 | md5 |
|---|---|
| `cp/cp_plan.py` | **`fa09a2a2`**（5,988 B） |
| `tools/test_cp_plan.py` | **`532046d6`**（5,276 B） |

**做法（为什么是 façade 而不是第三套几何）**

GPT §14 要 `cp/cp_plan.py` 承担 carry/halo/KV/boundary/terminal-state + MoE 计划；
同一条 §14 也写了「`不要建立并存的第三套几何`」。本仓几何真源是
`cp_layout.build_cp_plan`（纯 CPU，13 lens × cp1/2/4/8 覆盖）+ `cp_state` 绑定。
所以 `cp_plan` 先做**声明层**：

- `build_plan(seq_lens, cp_size, cp_rank, window, keep, replay, moe_span)`
  → `CpPlanView{layout, routes, seq_lens, cp_size, cp_rank}`；
- `layout` **就是** `cp_layout.build_cp_plan(...)` 的返回（逐字段相同，测试钉死）；
- `routes` 六条：`carry`（回退 1 行）、`halo`（`lookback=window-1` 且
  `pred_segments=2·cp−1`）、`kv_gather`（基线 `full_current_ag`）、`boundary`
  （`cut_rows=keep`，仅 `replay` 时开）、`terminal_state`（每段一次）、
  `moe`（`spans` 透传，跨 rank 必须一致）；
- `describe(view)` 打一行摘要，便于日志/断言。

**CPU 验收（蓝区）**：`rc=0`、**14 PASS / 0 FAIL**

1. **几何身份**：8 组 lens × cp1/2/4/8 × 每个 rank 全组合，`local_lens/local_cu/
   local_index/reverse_index/total_q/total_per_rank` 与 `cp_layout` **逐字段相同**
   ⇒ 证明没引入第二套几何；
2. `mirror_blocks == (rank, 2·cp−1−rank)`；
3. `local_rows == sum(local_lens)`，且 `cp=1` 等于全长（identity，无 pad）；
4. `cp=1` ⇒ carry/halo/kv_gather/terminal_state **全关**；`cp>1` 全开；
   `replay=False` ⇒ boundary 关；
5. `halo.pred_segments == 2·cp−1`（CP4→7、CP8→15，与实测口径一致）；
6. `moe.spans` 原样透传。

摘要样例：`cp2/r1 len=(8192,) local=(4096,) blocks=(1, 2) routes: carry=on halo=on
kv=full_current_ag boundary=on moe_spans=1`

**C4 剩余（下一步）**

- 把 `cp_state` 的"plan→设备索引/缓冲"绑定收进 `cp_plan`（现仍在 `cp_state.py`）；
- `halo` 从声明变成**执行**：按 `pred_segments` 生成反向 gather 清单（B5 的落点）；
- 这一切都在 `CpPlanView` 上加字段，**不改** `cp_layout` 公式 —— 保证 CP1/2/4/8 的
  既有 13×4 覆盖不回归。n### ★ 2026-11-01 15:5x C4 第二步：`halo` 改成**按几何计算**；撤回一条未核实的旧说法

**改了什么**

| 文件 | md5（改前 → 改后） |
|---|---|
| `cp/cp_plan.py` | `fa09a2a2` → **`bf5b0560`** |
| `tools/test_cp_plan.py` | `532046d6` → **`8c2d6edf`** |

`cp_plan._halo_blocks(layout, seq_lens, window, keep)`：尾部 `keep` 行 query 的 KV 可见范围
`[q-(window-1), q]` 在 mirror 分块（每序列 `ps=local_lens/2`、共 `2·cp` 块）上投影，
返回**相交块数与块号列表**；`routes["halo"].detail` 现在是
`{lookback, pred_segments, pred_blocks, why, legacy_claim_2cp_1}`。

**⚠️ 撤回（铁律 0：不许留未核实的数字）**

我上一轮在 WORKPLAN 里写过「halo 的 `pred_segments = 2·cp−1`（CP4→7、CP8→15），与实测口径一致」。
按本仓 mirror 分块**复现不出**该数字：

| 配置 | ps | 尾部窗口相交块数 |
|---|---|---|
| cp2 L=16384 | 4096 | **1** |
| cp4 L=16384 | 2048 | **1** |
| cp8 L=16384 | 1024 | **1** |
| cp8 L=8192 | 512 | 1 |
| cp8 L=4236 | 265 | 2 |
| cp8 L=1024 | 64 | 5 |
| cp4 L=1024 | 128 | 3 |
| cp2 L=1024 | 256 | 2 |

`ps ≥ window+keep` 时恒为 1 块；只有短序列才会多块（cp8/L=1024 → 5 块），
**任何 tiling 都得不到 15/7**。该说法已从代码里删掉，`cp_plan` 只保留可计算的定义，
并在字段里显式标注 `legacy_claim_2cp_1: "存疑并撤回"`。
（若将来要用回 15/7 这个口径，必须先给出它的**可计算定义**与出处，否则不许写进代码/文档。）

**CPU 验收（蓝区）**：`rc=0`、**20 PASS / 0 FAIL**（14 + 6 新增）

新增断言覆盖：`L=16384` 在 cp2/4/8 下前驱块数 == 1；短序列的**算出值**
（cp8/L=1024 → 5、cp4 → 3、cp2 → 2、cp8/L=4236 → 2、cp8/L=8192 → 1）；
多序列（`[1024,1024]` cp8）前驱块数为正。

摘要样例：`cp2/r1 len=(8192,) local=(4096,) blocks=(1, 2) routes: carry=on halo=on
kv=full_current_ag boundary=on moe_spans=1`

**C4 现状**：`cp_plan` 已能回答"这一轮 CP 要拿哪些块"（`pred_blocks`）。
下一步（B5 落点）：把这些块变成 `cp_runtime` 的**反向 gather 清单**（只取相交块，不再全量 AG），
并用"全量 AG 基线 vs 按清单取"的逐位对拍做验收。n### ★ 2026-11-01 15:5x P5 真权重签字 —— **全部通过**（gate + 口径自检 + trace 断言）

208（`liteserver-hps-tokens-001-00006`）`bash tools/run_real.sh -L 16k --mode prefill`，
修复 `run_real.sh` 的 prefill 口径后（`8ffe3e30` → **`fd6c4e6e`**）：

| 轮次 | rc | 用时 | trace | 结论 |
|---|---|---|---|---|
| prof-prefill | **0** | 272 s | 8/8 | 精度 gate ✅ / trace 断言 ✅ |
| noprof-prefill | **0** | 273 s | — | 口径自检 ✅ |

驱动结论行：`✅ 全部完成（口径自检 + trace 断言 + 精度 gate 全过）`
（`/mnt/xds/sfs/l00972766/real_prof/driver_20261101_072910.log`；
轮次明细 `real_prof/rounds.tsv` 两行：`prof rc=0/272s/8份`、`noprof rc=0/273s`）

官方基准（同权重/同 ids/同 EP8）：prefill **2394.7 ms**、6841.7 tok/s；next-token `id=396 ' that'`。
**mini 的 16k prefill 约 90 s/forward（273 s/轮）** —— 这是"逐算子 Python 驱动器"的固有开销，
本项目的口径是**逐位对齐**而非追平官方引擎吞吐（P5 的判据是"能跑 + top-1 一致"）。

算子计数（rank 侧，prof 轮）：827 次调用 / 24 种 / 跳过 0；`mega_moe` 40、`engram_*` 各 2
（层 1、14）、`kv_compress` 4、`idx_prolog` 8、`qli` 4、`qsli` 4、`idx_k_write` 1；
隐式 `dyn_mx_quant` 39；峰显 57.39 GiB（3.587 MiB/token @ L=16384）。

**P5 状态：完成。** 意义：在所有改动（A1–A7、B1–B6、C4、G9/G10 诊断、B2/B4 对齐、
P3-1/P3-2 计数、A5 空 origin 脚手架）之后，**真权重路径仍然跑通且 top-1 与官方一致** ——
"没有把模型改坏"的签字。

**附：P5 修的那条 bug 值得记住** —— `run_real.sh` 用 `non_cached_tokens=L` 表达普通 prefill，
在 P0 的几何收口下等价于"命中开启 + 非缓存段=全长"，`R=L+KEEP>L` 被显式拒绝。
正确写法是 `ratio=0 + nc=0`（命中关闭）。这类"配置层用命中字段表达非命中语义"的坑，
以后写入口脚本要按 `hit_is_on()` 的真值来选字段。n### ★ 2026-11-01 16:0x B5 第一步：`halo` 从"声明"变成**可执行清单**（块 → 持有 rank → 行区间）

| 文件 | md5（改前 → 改后） |
|---|---|
| `cp/cp_plan.py` | `bf5b0560` → **`0653882f`** |
| `tools/test_cp_plan.py` | `8c2d6edf` → **`24a9b2f5`** |

**新增 API（纯 CPU，无 torch/npu）**

- `HaloSpan{block, rank, pos_lo, pos_hi}`：一段必须可见的 KV；
- `_block_owner(b, cp)`：mirror 布局下块 b 归 `min(b, 2·cp−1−b)` 持有；
- `halo_manifest(view, seq_index, window, keep)`：尾部窗口 `[L-keep-(window-1), L)` 相交的
  全部块，逐块给出持有 rank 与位置区间；
- `halo_covered(manifest, lo, hi)`：**无洞覆盖**判定（运行时守卫与 CPU 测试共用同一份逻辑）。

**CPU 验收（蓝区）**：`rc=0`、**57 PASS / 0 FAIL**（20 + 37 新增）

新增断言：`cp∈{1,2,4,8} × L∈{128,1024,4236,8192,16384}` 全组合下
① 清单对 `[lo, L)` **无洞覆盖**；② 块↔rank 归属合法（`0≤rank<cp` 且等于 `_block_owner`）；
③ 区间合法（`pos_lo<pos_hi≤L`）；④ 多序列 `[1024,1024]` 时每个 `seq_index` 各自被覆盖。

**为什么这一步值得单独做**：B5 的实质要求是"尾部 query 的窗口必须可见**全部相交块**，
不是只取紧邻一块"。现在这句话从"注释"变成了**可计算、可断言**的东西：
`manifest` 给出该取哪些块，`halo_covered` 给出"取全了没有"。

**B5 剩余（下一步）**

1. 运行时守卫 `DS41M_CP_HALO_CHK=1`：在 CP 的 attention 前用 `halo_covered` 断言
   "已 gather 的 KV 覆盖尾部窗口"，不覆盖就**显式报错**（而不是静默算错）；
2. 选择性 gather `DS41M_CP_KV_GATHER=selective`：按 `manifest` 只取相交块，
   与基线 `full_current_ag` 做**逐位对拍**（同点位、CP2/CP4）；
3. 只有对拍通过才谈改默认（铁律 1）。

**注**：按当前几何，`L≥8192` 时尾部窗口通常只相交 **1 块**（`ps ≥ window+keep`），
所以选择性 gather 的收益主要在**短序列/多请求**场景；这条结论会影响是否值得做第 2 步，
留给数据说话（下一步会先跑一次"清单 vs 全量"的块数统计）。n### ★ 2026-11-01 16:1x B5 收益量化：replay 段省 **73.9%–99.6%** 的 KV 搬运（推翻了上一轮的直觉）

| 文件 | md5（改前 → 改后） |
|---|---|
| `cp/cp_plan.py` | `0653882f` → **`28341368`** |
| `tools/halo_report.py`（新） | **`5aa64498`** |

**新增**

- `cp_plan.gather_stats(view, window, keep, seq_index, mode)`：
  `rows_total = sum(local_lens) × cp`（★ 修正：全量 AG 搬的是**所有 cp 个 rank** 的行，
  上一版少乘 `cp`）、`rows_needed` = 尾部窗口相交块的块内行、`saved_pct`、`all_gathered`；
- `mode="replay"|"full"`：**决定收益是否存在** —— replay（层 21+ 只算 KEEP=140 行）时
  只需尾部窗口；full（全长段每行都要自己的窗口）时并集 = 全部行 ⇒ 收益恒为 0；
- `tools/halo_report.py`：矩阵报告 + 自带断言（`rows_needed<=rows_total`、
  `blocks_needed<=blocks_total`）。

**实测（蓝区，`python3.11 tools/halo_report.py`，全部组合通过断言）**

mode=replay（`rows_needed` 恒为 267 = keep + window−1）：

| cp | L | ps | 块(需/总) | rows_need / rows_tot | saved |
|---|---|---|---|---|---|
| 2 | 1024 | 256 | 2/4 | 267 / 1024 | 73.9% |
| 2 | 8192 | 2048 | 1/4 | 267 / 8192 | **96.7%** |
| 2 | 65536 | 16384 | 1/4 | 267 / 65536 | **99.6%** |
| 8 | 1024 | 64 | 5/16 | 267 / 1024 | 73.9% |
| 8 | 8192 | 512 | 1/16 | 267 / 8192 | **96.7%** |
| 8 | 65536 | 4096 | 1/16 | 267 / 65536 | **99.6%** |

mode=full：全部 0.0%（`all_gathered=是`）。

**结论（推翻我上一轮写的那句"大 L 收益很小"）**
那是把"相交块数少"错当成"搬运量小"。实际上：`ps` 大时虽然只相交 **1 块**，但那 1 块只需
**267 行**（窗口裁到 `[L-keep-(window-1), L)`），而全量 AG 要搬**全部 `L` 行量级**的数据
（`2·ps·cp ≈ L`）⇒ **块数少 ≠ 搬运少**。所以：

- **层 21+ 的 replay 段**：选择性 gather 值得做，收益 74%–99.6%（L 越大越划算）；
- **层 0–20 的全长段**：没有收益，必须保持全量 AG（否则每行的窗口都要重取）。

⇒ B5 的正确落点是**只替换 replay 段的 KV 获取**，不影响全长段。下一步：
`DS41M_CP_KV_GATHER=selective`（仅 replay 段生效，按 `halo_manifest` 取块），
与基线 `full_current_ag` 在 CP2/CP4 做**逐位对拍**；对拍通过前不改默认（铁律 1）。n### ★ 2026-11-01 16:1x B5 运行时守卫上线并验收（`DS41M_CP_HALO_CHK=1`）

| 文件 | md5（改前 → 改后） |
|---|---|
| `core/runner.py` | `f9a9fd55` → **`ec293a88`** |
| `cp/cp_plan.py` | （上一轮的 `28341368`，被 runner 调用） |

**守卫做什么**（只读诊断，不改执行路径）：CP>1 时在层 20/21/39 的入口，用
`cp_plan.halo_manifest` 算出"尾部窗口所需的 KV 块"，用 `halo_covered` 断言**无洞覆盖**，
并打印"选择性 gather 能省多少"；一旦出现 GAP 就 **显式 raise**（B5 要防的正是静默漏算）。

判据窗口：`[L-keep-(window-1), L)`。

**作业 234（`tmp/B5CHK_20261101_154525`）**：CP2 + `CP_HIT=1`，点位 `8192@0.5/b1/r1`

```
[r0] [halo-chk] L20 cp2 len=2118 覆盖[1851,2118): OK 块 1/4 行 267/2120 (选择性可省 87.41%)
[r0] [halo-chk] L21 cp2 len=2118 覆盖[1851,2118): OK 块 1/4 行 267/2120 (选择性可省 87.41%)
[r0] [halo-chk] L39 cp2 len=2118 覆盖[1851,2118): OK 块 1/4 行 267/2120 (选择性可省 87.41%)
```

- `rc=0`、**零 5070x**、稳态 0.206 s（与同几何历史值一致）
- 覆盖状态全 `OK`；`267 = keep(140) + window−1(127)`；全量 AG 搬 2120 行 ⇒ 省 **87.41%**
  （与 CPU 报告 cp2/R≈2118 的 87.4% 一致；L 越大越接近 99%）

**B5 剩余（结论已足够明确，但实现需要改寻址）**

选择性 gather 需要把"KV 池按全局位置寻址"改成"按清单块寻址"（`all_gather_global` 返回的
`total_q` 行被 mqsmla/压缩池直接按绝对位置索引）。这是一次**寻址层改造**，不是加个开关：
必须同时改 `cp_runtime.all_gather_global`、窗口/压缩池的 slot 计算、以及 `idx_k` 池；
验收要求"与全量 AG 逐位相同"（CP2/CP4 各一组）。在当前"一轮一改一验"的节奏下，
建议单独排一轮做（含回滚点），不要和其他改动混在一起。n### ★ 2026-11-01 16:2x C5 第一小步：布局覆盖**有作用域**（`stage_view`）+ runner 接入

| 文件 | md5（改前 → 改后） |
|---|---|
| `meta/metadata.py` | `748c488d` → **`325d2a28`** |
| `tools/test_metadata_layout.py`（新） | **`8cb42470`** |
| `core/runner.py` | `ec293a88` → **`7c20d7c7`** |

**新增 API（`metadata` = 布局唯一入口）**

- `snapshot_layout(cfg)`：存档 `cu/row_lens/seq_lens/n` 四个字段（无 metadata → `None`）；
- `restore_layout(cfg, snap)`：还原并**返回是否真的改过**（漏还原能被发现）；
- `@contextlib.contextmanager stage_view(cfg, **kw)`：进块改、出块**必定**还原（异常也还原）。

**runner 接入**：函数头的存档改成 `_md.snapshot_layout(cfg)`（原来手写只存 `cu/row_lens` 两个
字段的私有元组），出口改成 `_md.restore_layout(cfg, _lay_orig)`（四个字段一起还原，并在
`DS41M_META_TRACE=1` 时打印"已还原"）。语义等价，但格式与还原点收敛到 `metadata` 一处 ——
这正是 C5 说的"布局只在作用域内可见"。

**CPU 验收（蓝区）**：`rc=0`、**12 PASS / 0 FAIL**
① 存档/还原四字段逐字段相同；② 布局未变时 `restore` 返回 `False`（不产生无谓写）；
③ `stage_view` 作用域内生效、退出还原、**异常退出也还原**；④ 嵌套 `stage_view` 逐层还原。

**实跑回归（作业 235，`tmp/B5CHK_*`）**：CP2 + `CP_HIT=1`，`8192@0.5/b1/r1`
`rc=0`、**零 5070x**、`[halo-chk]` 三行与作业 234 **完全一致**、稳态 0.211 s（234 为 0.206 s，2% 级噪声）。

**C5 剩余**

- `_boundary_cut_state_ms` / `_cp_replay_trim` 里的 `_md.set_layout(...)` 改成 `stage_view` 作用域
  （需要把"层 20 之后所有层"包进同一个作用域 ⇒ 要在 `_run_forward_impl` 的层循环外开）；
- `Geometry`/`Policy` 也从 `cfg` 里搬进 `InputMetadata`（现在是 `cfg["hit"]`、`cfg["run"]` 各读一处）；
- 目标：`cfg` 只留一份 `_meta_obj`，`_packed` 镜像在测试里被断言"永不手改"。n### ★ 2026-11-01 16:3x C5 第二步：把"谁能写布局/`_packed`"钉成**可执行不变量**（并修掉抓到的第二写入点）

**新文件 + 修复**

| 文件 | md5（改前 → 改后） |
|---|---|
| `tools/test_meta_single_source.py`（新） | **`a1092155`** |
| `core/assemble.py` | `1118b1f0` → **`e95f7c68`** |

**为什么要这条测试**：本项目出过"compact 改了 layout 没还原 → 漏到下一个 pass → moe gate 吃 nan"
（P1-1c）。这类 bug 运行期极难定位，但**写入点是可以静态枚举的**。测试把"谁能写"钉成 allowlist：

1. `cfg["_packed"] = …` 只能出现在登记文件；
2. `.set_layout(` 调用只能出现在登记文件；
3. 每个登记项都必须在注释里写明原因；
4. 扫描结果**打印出来** —— 新增写入点直接 FAIL，逼着走 review（铁律 0）。

**扫描抓到的真实问题**：`core/assemble.py::ensure_packed` 直写 `cfg["_packed"]`
（第二写入点，dry-run/工具补默认单序列布局）。已改成经 `metadata.attach + set_layout`，
由 metadata 生成镜像。改后扫描结果：

```
packed      meta/metadata.py:155/163        （owner 写镜像）
packed      core/points_runner.py:506       （点位构建期）
packed      core/runner.py:3048/3503/4243   （decode/CP/命中 构建期）
set_layout  core/runner.py:838/928/1051     （CP/compact，C5 待改 stage_view）
set_layout  core/assemble.py（新的合法调用方）
```

**验收**

- `tools/test_meta_single_source.py`：`rc=0`、**7 PASS / 0 FAIL**；
- `tools/test_hit_geometry.py`：**70 PASS / 0 FAIL**（assemble 改动无回归）；
- 实跑回归（作业 236，`tmp/B5CHK_20261101_155650`）：CP2 + `CP_HIT=1`，
  `rc=0`、**零 5070x**、`[halo-chk]` 三行与前两次完全一致、稳态 0.211 s。

**C5 剩余**

- `core/runner.py` 的三处 `set_layout` 收成 `stage_view` 作用域（需要把"层 20 之后"
  这段跨层的布局变化表达成一次性作用域，或在循环内用 push/pop 栈）；
- `Geometry`/`Policy` 从 `cfg["hit"]`/`cfg["run"]` 搬进 `InputMetadata`；
- 目标：`cfg` 只剩 `_meta_obj`，`_packed` 镜像"永不手改"（本测试已守住）。n### ★ 2026-11-01 16:4x C5 第三步：runner 的三处布局改写收成**栈式 push/pop**

| 文件 | md5（改前 → 改后） |
|---|---|
| `meta/metadata.py` | `325d2a28` → **`af61932c`** |
| `core/runner.py` | `7c20d7c7` → **`20f8360c`** |
| `tools/test_metadata_layout.py` | `8cb42470` → **`bb35b027`** |

**新增（`metadata` = 布局唯一入口）**

- `push_layout(cfg, **kw)`：把当前四字段压栈后再应用（返回栈深）；
- `pop_layout(cfg)` / `pop_all_layouts(cfg)` / `layout_depth(cfg)`。

**runner 改动**：三处 `_md.set_layout(cfg, cu=…, row_lens=…)`（CP 边界拆分 / CP replay 裁剪 /
packed 层 20 截尾）全部改成 `_md.push_layout(...)`；pass 出口先 `pop_all_layouts(cfg)`
（把本 pass 压的栈全部还原），再用**入口快照** `restore_layout(cfg, _lay_orig)` 兜底。
⇒ 现在"漏还原"在结构上不可能：出口无条件弹栈，且栈里的每层都有快照。

**CPU 验收（蓝区）**

- `tools/test_metadata_layout.py`：`rc=0`、**19 PASS / 0 FAIL**（12 + 7 新增）
  新增断言：push 深度 +1、push 生效、二层 push、pop 回上一层、`pop_all` 回原值、
  空栈 `pop→False` / `pop_all→0`；
- `tools/test_meta_single_source.py`：`rc=0`、**7 PASS**（写入点仍只有登记的那几个）。

**实跑回归（作业 237，`tmp/B5CHK_20261101_160206`）**：CP2 + `CP_HIT=1`，`8192@0.5/b1/r1`
`rc=0`、**零 5070x**、`[halo-chk]` 三行与 234/235/236 完全一致、稳态 **0.212 s**
（此前 0.206/0.211/0.211 s，属噪声级）。

**C5 现状**：布局变化 = 显式 push/pop + 出口兜底；写入点 = 静态可枚举（有测试守）。
**剩余**：`Geometry`/`Policy` 从 `cfg["hit"]`/`cfg["run"]` 搬进 `InputMetadata`，
让 `cfg` 只剩 `_meta_obj` 一个几何来源。n### ★ 2026-11-01 16:5x C6 #7 第一步：`CpState` **逐请求坐标**（去掉"只能等长同起点"的结构限制）

| 文件 | md5（改前 → 改后） |
|---|---|
| `cp/cp_state.py` | `b615ecd2` → **`86a97cc5`** |
| `tools/test_cp_state_offsets.py`（新） | **`6eb68ad6`** |

**为什么做这个**：`CpState.pos_offset` 是**标量** ⇒ packed 多请求只能"等长 + 同起点"，
这正是 G10（b>1 compact 与全长段不逐位）与 A5（空 origin 虚拟行位置）的结构根因之一。
GPT §14 第 7 行明确要求"plan→设备索引/缓冲绑定，**不用单个 scalar offset**"。

**改动（向后兼容）**

- 新增字段 `pos_offsets: tuple | None`，`from_plan(..., pos_offsets=None)`：
  为空 ⇒ 旧标量路径（**逐位不变**）；非空 ⇒ 按 `plan.local_cu` 把每个请求的起点展开成
  **逐行偏移张量** `_off_rows`；
- `abs_positions`：优先逐请求（缓存键 `("per", tuple(pos_offsets))`），否则标量；
- `with_compact(..., pos_offsets=None)` 透传；
- 长度校验：`len(pos_offsets) != 请求数` ⇒ **显式 `ValueError`**（不静默截断）。

**CPU 验收（蓝区）**

- `tools/test_cp_state_offsets.py`：`rc=0`、**8 PASS / 0 FAIL**
  ① identity：`pos_offsets=[o,o]` 与标量 `pos_offset=o` **逐元素相同**（旧语义不变）；
  ② 不等起点：请求 0 用 +100、请求 1 用 +500，且两区间 **不重叠**（旧标量口径做不到）；
  ③ 有 pad（L 不整除 2·cp）时逐请求路径不抛错、行数一致；
  ④ 长度不符 ⇒ `ValueError`。
- `tools/test_cp_state.py`（既有）：`rc=0`、`cp_state: PASS (13 lens × cp 1/2/4/8)` —— 零回归。

**下一步（消费端接线）**

1. runner 构造 `CpState` 时把 `ctx["pos_offsets"]`（packed 多请求已有这个列表）传进去；
2. `with_compact` 的调用点同样传逐请求起点（compact 段起点 = 各请求 `L-W`）；
3. 接线后 **G10 的"等长"前提就不需要了**，可再跑一次 b=2 compact A/B 看残差是否消失；
4. 消费端若仍有 `ctx["pos_offset"]` 标量假设（mqsmla/idx_prolog/压缩池），逐个改成读 `abs_positions`。n### ★ 2026-11-01 17:0x 今日改动**整体回归**（CPU 8 套 + NPU 矩阵 CP×b×hit）

**A. CPU 套件（蓝区，`TORCH_DEVICE_BACKEND_AUTOLOAD=0`）—— 全绿**

| 套件 | rc | PASS | FAIL |
|---|---|---|---|
| `test_hit_geometry` | 0 | 70 | 0 |
| `test_cp_layout` | 0 | （13 lens × cp1/2/4/8 全 PASS） | 0 |
| `test_cp_state` | 0 | （8 lens × cp1/2/4/8 全 PASS） | 0 |
| `test_batch_replay` | 0 | 34 | 0 |
| `test_cp_plan` | 0 | 57 | 0 |
| `test_metadata_layout` | 0 | 19 | 0 |
| `test_meta_single_source` | 0 | 7 | 0 |
| `test_cp_state_offsets` | 0 | 8 | 0 |
| **合计** | — | — | **0** |

**B. NPU 矩阵（`tools/sweep.py`，作业 239→240，`tmp/REG_*`）**

矩阵：`L=8192 × hit∈{0.5,0.97} × b∈{1,2} × cp∈{1,2}`，`--cards 0,1 --iters 4`。

| 组 | 结果 |
|---|---|
| cp1（4 点） | **全部 `rc=0`、`errors=0`**：0.2555 / 0.5309 / 0.1132 / 0.1164 s |
| cp2（b1 两点） | **`rc=0`、`errors=0`**：0.2108（h=0.5）/ 0.1710（h=0.97） |
| cp2（b2 两点） | **被 G7 守卫显式拒绝**（`命中 + CP 只支持单序列`）—— 这是**有意**的：packed 多序列命中 + CP 尚未验收（G10/G12），opt-in 开关 `DS41M_PACKED_BOUNDARY=1` |
| 5070x | **0**（全组） |
| 算子错误 | **0**（全组） |

**C. 过程中修掉的两个工具缺口**

1. `tools/sweep.py` 没给 cp>1 的命中点位设 `DS41M_CP_HIT=1` ⇒ 作业 239 的 cp2 组整组被 D2 守卫拦住。
   已修：`build_cmd` 在 `cp>1 且任一点位 hit>0` 时自动 `env["DS41M_CP_HIT"]="1"`（`88386f92`）。
2. `run_hit.sh` 的 f-string bug（`{L,hit}` 当表达式）早前已修（`35a6f33e`）。

**结论**：今天所有改动（A1–A7、B1–B4/B6、C4 `cp_plan`、C5 三步、C6 #7 第一步、B5 守卫、
P3-1/P3-2）**没有引入回归**：CPU 8 套 0 FAIL；NPU 矩阵里所有"应当能跑"的点位都是
`rc=0 / errors=0 / 无 5070x`，而被拦下的两个点位是**设计上的显式拒绝**（G7），不是失败。
稳态量级与历史一致（cp1 b1 0.2555 s = P0 基线；cp2 b1 0.1710–0.2108 s）。n### ★ 2026-11-01 17:1x C6 #7 第二步：逐请求坐标**接线到 CP 构造点**

| 文件 | md5（改前 → 改后） |
|---|---|
| `core/runner.py` | `20f8360c` → **`b65add2a`** |

**接线点（三处）**

1. `run_forward` 里 CP 的 `CpState.from_plan(...)`：新增 `pos_offsets=_offs`，
   来源 = `cfg["_meta_override"]["pos_offs"/"pos_offsets"]`（CP 分支自己写的逐请求起点），
   长度必须等于请求数，否则退回 `None`（标量路径）；
2. `_boundary_cut_state_cp` 的 `cp.with_compact(keep, pos_offset=pos_off2, seq_len=…)`：
   多请求时传 `pos_offsets=[pos_off2] × n`（compact 段每请求起点；等长时与标量**同值**）；
3. `_cp_replay_trim` 的 `with_compact(..., pos_offset=new_ofs, ...)`：同上，传 `[new_ofs] × n`。

**为什么要接**：状态层（上轮 `86a97cc5`）已经支持逐请求；但没人喂它 ⇒ 能力是死的。
接上之后，"真正的逐请求起点"只差**上游给出不同的起点**（当前 shot 口径下各请求起点相同，
所以结果与标量路径**逐位相同** —— 这是本轮回归能保持零变化的根据）。

**回归（作业 241，`tmp/REG_20261101_162345`）**

| 组 | 结果 |
|---|---|
| cp1（4 点） | 全 `rc=0`、`errors=0`：0.2553 / 0.5294 / 0.1129 / 0.1166 s |
| cp2 b1（2 点） | `rc=0`、`errors=0`：0.2113 / 0.1769 s |
| cp2 b2（2 点） | 仍被 G7 守卫**显式拒绝**（设计如此，opt-in `DS41M_PACKED_BOUNDARY=1`） |
| 5070x / 算子错误 | **0 / 0** |

与上一轮（作业 240）逐点对比：`0.2555→0.2553`、`0.5309→0.5294`、`0.1132→0.1129`、
`0.1164→0.1166`、`0.2108→0.2113`、`0.1710→0.1769` —— 全部在噪声带内，**零回归**。

**还要做什么才能真正支持"不等长 packed"**（诚实标注）

状态层与接线已就绪，但下游仍假设等长：
1. `_boundary_cut_state_ms` 的 `pos_off2` 是标量（需改成逐请求：`pos_offsets[i] + (R_i − W)`）；
2. 窗口池 `rebase_window_pool` 按 `rows_blocks_per_seq`（取 max 行数）算 P，逐请求行数不同时
   需要逐请求 P（或按 max 行数统一开池、逐请求只填自己的行）；
3. 压缩池 `_cmp_pool_r{ratio}` / `idx_k` 的行账同理；
4. G7 的"命中+CP 只支持单序列"守卫要先解除（它现在连 opt-in 都要 `DS41M_PACKED_BOUNDARY=1`）。

⇒ 不等长 packed 是**一轮独立工作量**（上面 4 项 + 与全长段的逐位对拍），不能顺带做。n### ★ 2026-11-01 17:3x C5 收尾（第一半）：命中策略进 `InputMetadata`（单源 + 镜像同步）

| 文件 | md5（改前 → 改后） |
|---|---|
| `meta/metadata.py` | `af61932c` → **`0875c995`** |
| `tools/test_metadata_layout.py` | `bb35b027` → **`e8bc2237`** |

**新增**

- `@dataclass HitPolicy{ratio, non_cached_tokens, replay_tail, alignment, source}` + `is_on()`；
- `InputMetadata.hit` 字段；`InputMetadata.from_cfg` 从 `cfg["hit"]` 填充；
- `hit_of(cfg)`：命中策略的**唯一读取入口**。★ 兼容口径：`cfg` 可能只是 `{"hit": {...}}`
  的最小 dict（`points_runner._m_of`、`assemble.hit_m` 的调用方就这样传）—— 此时**现场构造**
  一个 `HitPolicy` 返回、**不写回**，避免污染最小 dict；
- `set_hit(cfg, **kw)`：改 metadata 并同步 `cfg["hit"]` **镜像**；未知字段 ⇒ `ValueError`。

**CPU 验收（蓝区，8 套全跑）**：`FAIL 总数 = 0`

| 套件 | PASS |
|---|---|
| `test_hit_geometry` | 70 |
| `test_batch_replay` | 34 |
| `test_cp_plan` | 57 |
| **`test_metadata_layout`** | **25**（19 + 6 新增） |
| `test_meta_single_source` | 7 |
| `test_cp_state_offsets` | 8 |
| `test_cp_layout` / `test_cp_state` | rc=0（13×4 / 8×4 全 PASS） |

新增断言：attach 后 `hit_of` 读到三项字段；`is_on`；`set_hit` 同步镜像；未知字段报错；
最小 dict 可读且**不改写** `cfg`。

**实跑回归（作业 242，`tmp/REG_20261101_163032`）**

| 组 | 结果 |
|---|---|
| cp1（4 点） | 全 `rc=0`、`errors=0`：0.2554 / 0.5308 / 0.1151 / 0.1206 s |
| cp2 b1（2 点） | `rc=0`、`errors=0`：0.2093 s 起（另一点同量级） |
| cp2 b2（2 点） | 仍由 G7 守卫**显式拒绝**（设计如此） |
| 5070x / 算子错误 | **0 / 0** |

与作业 241 对比：cp1 四点 `0.2553/0.5294/0.1129/0.1166` → `0.2554/0.5308/0.1151/0.1206`，
全部在噪声带内（h=0.97 两点略高 ~3%，与 metadata 多一次 `HitPolicy` 构造无关——同一 pass
内只构造一次；判定为机器噪声）。

**C5 剩余（第二半）**

- `Geometry`/`Policy` 里仍有从 `cfg["run"]`/`cfg["features"]` 现场读的字段（`pos_offset`、
  `decoder_replay` 等）：下一步把这些也改成"只在 `attach()` 读一次"；
- `assemble.hit_m/check_hit_geometry` 目前仍读 `cfg["hit"]`（走 `policy` façade）。
  下一步改成读 `metadata.hit_of(cfg)`，让命中口径真正只有一个入口；
- 目标不变：`cfg` 只剩 `_meta_obj`（+ 镜像），且镜像由 `set_*` 系列单向同步。n### ★ 2026-11-01 17:4x C5 收尾（第二半）：`assemble` 的命中口径统一走 `metadata.hit_of`

| 文件 | md5（改前 → 改后） |
|---|---|
| `core/assemble.py` | `e95f7c68` → **`b12dd741`** |
| `meta/metadata.py` | `0875c995` → **`2880cc89`** |

**改动**

- `assemble.hit_alignment / hit_is_on / hit_m / replay_tail_of` 全部改读 `metadata.hit_of(cfg)`
  （`assemble` 里已无 `cfg.get("hit")` 直读）；
- `HitPolicy.alignment` 默认改成 **空串**（= "未显式指定"），使优先级保持
  **cfg 显式 > `DS41M_HIT_ALIGN` > 默认 `token`** —— 这条是必须的：第一版把默认写成 `"token"`，
  导致 `DS41M_HIT_ALIGN` 被吞，`test_hit_geometry` 立刻抓到
  `FAIL · backends 串里出现 align=cpblock128`（69 PASS / 1 FAIL）。改成空串后恢复 70 PASS。

**正确的"新鲜度"设计（关键）**

`hit_of(cfg)` 在有 `_meta_obj` 时**以 `cfg["hit"]` 为准即时刷新** `m.hit` 再返回。
原因：仓库里仍有大量 `cfg["hit"] = dict(...)` 整块赋值（runner/points_runner），
若直接读缓存对象就会拿到**陈旧值**（那是静默的行为改变，铁律 0 不允许）。
等写入侧全部改成 `set_hit` 后，才可以翻转优先级、去掉刷新。

**验收**

- CPU 8 套：`rc=0`、**FAIL 总数 0**
  （`test_hit_geometry` 70、`test_batch_replay` 34、`test_cp_plan` 57、
  `test_metadata_layout` 25、`test_meta_single_source` 7、`test_cp_state_offsets` 8、
  `test_cp_layout`/`test_cp_state` rc=0）；
- 实跑回归（作业 243，`tmp/REG_20261101_163747`）：
  cp1 四点 `0.2553 / 0.5294 / 0.1136 / 0.1177`、cp2 b1 两点 `0.2103 / 0.1717`，
  全部 `errors=0`；cp2 b2 仍由 G7 守卫显式拒绝；**零 5070x、零算子错误**。

**C5 状态**：布局（push/pop + 单写入源测试）与命中策略（`HitPolicy` + `hit_of`）都已单源化。
**剩余**：`Geometry`/`Policy` 里仍现场读 `cfg["run"]`/`cfg["features"]` 的字段
（`pos_offset`、`decoder_replay`）也收进 `attach()`；以及把 `cfg["hit"]` 的写入侧换成 `set_hit`。n### ★ 2026-11-01 17:5x C5 收尾（第三半）：命中策略的**写入侧**也收敛到 metadata

| 文件 | md5（改前 → 改后） |
|---|---|
| `meta/metadata.py` | `2880cc89` → **`df0da23f`** |
| `core/points_runner.py` | `d18d193f` → **`3408d7ca`** |

**新增 `metadata.set_hit_dict(cfg, d)`**（写入侧唯一入口）

- 已知字段（`ratio`/`non_cached_tokens`/`replay_tail`/`alignment`/`source`）写进 `HitPolicy`，
  并同步 `cfg["hit"]` **镜像**；
- **未知字段原样保留在镜像里**（老代码可能读它们 —— 不能悄悄丢，这是"责任不省略"的底线）；
- `cfg` 没有 `_meta_obj` 时先建一个（`_packed` 一并生成）。

`core/points_runner.py` 里原来那句 `cfg["hit"] = dict(base_hit, ratio=…, replay_tail=…, source="fake")`
已改为 `metadata.set_hit_dict(cfg, dict(...))`。

**为什么这一步必要**：读取侧（`assemble` 的四个函数）已经统一走 `metadata.hit_of`；
但如果写入侧还有人整块覆盖 `cfg["hit"]`，`hit_of` 就得保留"即时刷新"补丁才能正确
（见上一轮）。写入侧收敛后，这个补丁迟早可以删掉、变成真正的单向单源。

**验收**

- CPU 8 套：`rc=0`、**FAIL 总数 0**；
- 实跑回归（作业 244，`tmp/REG_20261101_164316`）：
  cp1 四点 `0.2554 / 0.5285 / 0.1122 / 0.1181`、
  cp2 b1 两点 `0.2112 / 0.1699`，全部 `errors=0`；
  cp2 b2 仍由 G7 守卫显式拒绝；**零 5070x、零算子错误**。

与作业 243 对比：`0.2553/0.5294/0.1136/0.1177` → `0.2554/0.5285/0.1122/0.1181`；
`0.2103/0.1717` → `0.2112/0.1699` —— 全在噪声带内。

**C5 完成度**

| 子项 | 状态 |
|---|---|
| 布局：`stage_view` / 栈式 push-pop / 单写入源测试 | ✅ |
| 命中：`HitPolicy` + `hit_of` 读取侧单源 | ✅ |
| 命中：写入侧 `set_hit_dict` | ✅（本轮） |
| `Geometry`/`Policy` 的 `cfg["run"]`/`cfg["features"]` 现场读 | ⬜ 剩余 |
| 删掉 `hit_of` 的"即时刷新"补丁（真正单向） | ⬜ 下一步（需确认无整块赋值残留） |n### ★ 2026-11-01 18:0x B5 第三步：`halo_gather_plan` —— "谁发哪些行"的精确本地切片

| 文件 | md5（改前 → 改后） |
|---|---|
| `cp/cp_plan.py` | `28341368` → **`59a7797a`** |
| `tools/test_cp_plan.py` | `24a9b2f5` → **`93459d53`** |

**为什么需要它**：`all_gather_global` 是**全量**交换（每个 rank 发 `sum(local_lens)` 行）。
B5 只想让相关 rank 各发**自己块里的那小段**。但"发哪几行"必须能精确算出来，否则
collective 的收发形状对不齐（这是最容易写出静默错数的地方）。

**新增 `cp_plan.halo_gather_plan(view, seq_index, window, keep)`**

返回 `[{rank, block, local_row0, nrows, pos_lo, pos_hi}]`：
- `rank` = 该块持有者（mirror: `min(b, 2·cp−1−b)`）；
- `local_row0` = 该块在**持有 rank 本地行轴**上的起点（`pos_lo − b·ps`）；
- `nrows` = 必须发送的行数（已被窗口裁到 `[pos_lo, pos_hi)`）；
- 并做**边界断言**：段必须落在本块内（`0 ≤ local_row0`、`local_row0+nrows ≤ ps`），
  越界直接 `ValueError`（不许静默截断）。

**CPU 验收（蓝区）**：`rc=0`、**105 PASS / 0 FAIL**（57 + 48 新增，`cp∈{2,4,8} × L∈{1024,4236,8192,16384}`）

每组都验四条：
1. `halo_gather_plan` 段数 == `halo_manifest` 段数；
2. 每段落在持有 rank 的块内（`0 ≤ local_row0`、`+nrows ≤ ps`）；
3. 段区间**互不重叠**且并集 `== [max(0, L−keep−(window−1)), L)`；
4. `rank == _block_owner(block, cp)`。

**下一步（接到 cp_runtime）**

1. `cp_runtime.merge_global_rows(x_local, plan, gather_plan, group)`：
   每 rank 按自己的段做 `all_gather_into_tensor`（尺寸 = 各 rank 段数之和，需先 `all_gather` 计数），
   然后按 `pos_lo` 散写回目标池 —— **只写这几段**，池的整体布局不变（消费端零改动）；
2. `exec_ops` 的 CP KV 分支在 replay 段（`T == keep` 左右）走这条路径，全长段仍走 `merge_global`；
3. 验收：同点位 `DS41M_CP_KV_GATHER=full|selective` 两臂**逐位相同**（CP2/CP4 各一组），
   且 selective 臂的 gather 字节数按 `halo_report` 的预测下降（cp2/R≈2118 → 省 87%）。

**为什么"池布局不变"是关键**：`_kv_epilog` 已经把 gather 结果按全局位置写进池，
消费者（mqsmla/压缩池）按绝对位置读 —— 只要**写入的落点不变、只少写不需要的行**，
消费端就完全不用改，数值也逐位不变。这把 B5 从"改寻址"降级成"改一次 gather 的范围"。n### ★ 2026-11-01 18:1x B5 第四步：搬运三元组 `scatter_plan`（CPU 可验的"照单搬"）

| 文件 | md5（改前 → 改后） |
|---|---|
| `cp/cp_plan.py` | `59a7797a` → **`20a1c417`** |
| `tools/test_cp_plan.py` | `93459d53` → **`bf623444`** |

**新增 `cp_plan.scatter_plan(view, seq_index, window, keep)`**

返回 `[{dst_pos, src_rank, src_local_row}]`：目标池的全局位置 `dst_pos` 应写入"持有者 rank
本地第 `src_local_row` 行"的值。runtime 只要照单搬运 + 散写，**不需要理解 mirror 分块**。

半区规则（这是最容易写错的地方，已用测试钉死）：
- head 块（`block < cp`）在本 rank 的 `[0, ps)`；
- tail 块（`block ≥ cp`）在本 rank的 `[ps, 2·ps)`；
即 `src_local_row = (0 if block<cp else ps) + local_row0 + i`。

**CPU 验收（蓝区）**：`rc=0`、**153 PASS / 0 FAIL**（105 + 48 新增，`cp∈{2,4,8} × L∈{1024,4236,8192,16384}`）

每组四条：
1. `dst_pos` **恰好**覆盖 `[max(0, L−keep−(window−1)), L)` 且**无重复**（逐点比对 `range`）；
2. `src_local_row ∈ [0, 2·ps)`；
3. **半区与块类别一致**（head 块 → `<ps`，tail 块 → `≥ps`）；
4. `src_rank == _block_owner(dst_pos // ps, cp)`。

**runtime 侧的关键发现（这决定了 B5 的改法）**

读 `exec_ops._kv_epilog` 的 CP 分支后确认：**写侧本来就是部分的** —— 窗口池只写
`kv[-wa:]`（`wa = min(win_keep, T)`），压缩 epilog 也用 `kv[-wa:]`。也就是说
**只有 gather 是全量的**。所以 B5 的正确改法是：

> 只 gather `wa` 段所需的全局行（= `scatter_plan` 的 dst 集），gather 结果按 `dst_pos` 直接写池，
> 池布局与消费端**零改动**；全长段（`wa == T`）继续走 `merge_global`。

**下一步（runtime，单列一轮）**

1. `cp_runtime.merge_global_rows(x_local, plan, triples, group)`：按 `src_rank` 分组 →
   `all_gather_into_tensor`（先交换各 rank 段数）→ 按 `dst_pos` 排序返回**紧凑张量**；
2. `_kv_epilog`：`DS41M_CP_KV_GATHER=selective` 且 `wa < T` 时走它，槽位用 `dst_pos` 现算；
   其余情况**逐字不变**；
3. 验收：同点位 `full` vs `selective` 两臂**逐位相同**（CP2/CP4），selective 臂的 gather 字节
   按 `halo_report` 预测下降（cp2/R≈2118 → 省 87%）。n### ★ 2026-11-01 18:2x B5 数值前提**已被证实**：选择性 gather 与全量 gather 逐位相同

**闸门（只校验不改路径）**：`DS41M_CP_KV_GATHER=check`

作业 248（`tmp/B5CHK2_20261101_170810`，CP2 + `CP_HIT=1`，`8192@0.5/b1/r1`）：

```
[kv-gather-check] L0  T=4236 取[3975,4236) (261,512) vs 全量切片 (261,512) 逐位=相同 max|Δ|=0.000e+00
（40 层 × 2 rank × 5 pass = 400 次，全部"逐位=相同"）
same=400  diff=0      rc=0      零 5070x
```

`261 = win_keep(134) + BS(128) − 1`：覆盖尾部 query 的**全部**窗口读范围。

**过程中被这条闸门抓到的两个真 bug（都不是"顺手改好了"，是先错后证）**

1. `all_gather_into_tensor` 的入参形状：我最初把 `world*maxn` 的整块当成本地分片传进去 ⇒
   `output tensor size must be equal to world_size times input tensor size`（160 错/rank）。
   正解：`inp` 是**本地分片**（`maxn` 行），`out` 才是 `world*maxn`。
2. 校验把**合并后的全局张量**当成本地张量喂进 `merge_global_rows` ⇒ 全层不一致
   （`max|Δ|` 6–13）。正解：保存合并**前**的 `_kv_local_before_merge`。

两次都是判据当场暴露并**显式 raise**（`不许静默继续`）—— 这就是闸门存在的意义。

**新增 API**

| 文件 | md5（改前 → 改后） | 内容 |
|---|---|---|
| `cp/cp_runtime.py` | `428e296f` → **`8d9f5562`** | `_row_source(p, ps, cp_size)` + `merge_global_rows(x_local, plan, pos_lo, pos_hi, group)`：按 mirror 归属取行 → 交换段数 → `all_gather_into_tensor`（本地分片 maxn 行）→ 按 dst 顺序返回紧凑张量。**单序列限定**，多序列显式拒绝 |
| `cp/cp_state.py` | `86a97cc5` → **`76aa6211`** | `CpState.merge_global_rows(...)` 薄封装（`_kv_epilog` 拿到的是 CpState 不是模块） |
| `ops/exec_ops.py` | `9a67228f` → **`d77e697c`** | `=check` 闸门 + 保存 `_kv_local_before_merge` |

**下一步（接线，`=selective`）**

`_kv_epilog` 在 replay 段（`_lrow is None` 且 `T > wa`）用 `kv_sel = merge_global_rows(local, lo, T)`：
- 窗口池 A：`_contig(kv_sel[-wa:])`（值同 `kv[-wa:]`）；
- scratch 池 B：`kv_compact_epilog(pool, kv_sel, slot[lo:T])`。
两块池的落点不变（同一个 `slot` 公式，只少写 `[0, lo)` 那些行），消费端**零改动**；
全长段（`wa >= T`）逐字不变。验收：`full` vs `selective` 两臂逐位相同（CP2/CP4）+ gather 量降 ~87%。n### ★ 2026-11-01 18:3x B5 **结论：当前几何下没有可省的 gather 量**（撤回 87–99% 的口径）

**撤回的那条**：我此前用 `tools/halo_report.py` 报"replay 段选择性 gather 可省 73.9%–99.6%"。
作业 249/251 的实测证明了它是**口径错误**：

- 层 20（裁剪前）：本层**全部 T 行**都要参与 attention ⇒ 需要的 KV 就是全部 T 行，
  不可能只取尾部窗口；
- 层 21+（compact 段）：`T ≈ keep ≈ 140`，全量 gather 本来只有 140 行 —— 已经很小。

也就是说 `halo_report` 把**全序列 L** 当成了分母，而真实要 gather 的量是"本层 T"。
正确结论：**在现有几何下 B5 没有可省的搬运量**。

**实测证据（作业 251，`tmp/B5SEL_20261101_172053`，CP2 + `CP_HIT=1`）**

| 臂 | rc | L0 | L20 | L39 | 5070x | 稳态 |
|---|---|---|---|---|---|---|
| `full` | 0 | `8e536cd3` | `893fcdb7` (2.949e17) | `08c29528` | 0 | 0.215 s |
| `selective` | 0 | `8e536cd3` | **`c7a9b82f` (3.729e17)** | **`26cbb7fe`** | 0 | 0.279 s |

选择性臂虽然 `rc=0` 且打印"只取[3975,4236) 261 行（省 93.8%）"，但**数值不同**：
它在层 20 少取了 `[4096-127, 3975)` 那几行（query 的窗口下界），并且层 20 的读本来就
需要全部行 ⇒ 必然算错。这说明"省 93.8%"是**少算了不该省的行**换来的。

**处置（铁律 0）**：`DS41M_CP_KV_GATHER=selective` 改成**显式拒绝**
（`T > keep` 直接 raise 并指向本条结论），代码树里不留一条会算错的捷径；
`=check` 闸门保留（它证明的是"尾部切片取值正确"，这条**仍然成立**，作业 248 的
400/400 逐位相同有效）。

`ops/exec_ops.py` md5：`d77e697c` → **`713838cb`**

**B5 的最终状态（诚实）**

| 子项 | 状态 |
|---|---|
| `halo_manifest` / `halo_covered` / `scatter_plan` / `halo_gather_plan` | ✅ 已落，CPU 153 PASS（几何正确） |
| `merge_global_rows`（按 mirror 归属取部分行） | ✅ 已落，NPU 400/400 层逐位相同 |
| 运行时守卫 `DS41M_CP_HALO_CHK=1` | ✅ 已验收（作业 234–237） |
| **"选择性 gather 省通信"** | ❌ **前提不成立**（本层全行都要 attention）；已撤回并改成显式拒绝 |
| 真正可能有收益的场景 | 需要"query 只覆盖尾部窗口、而 T 远大于 keep"的形态（例如未来的长序列 decode/chunked prefill），当前 bench 点位不具备 |

⇒ B5 按 GPT 的**语义责任**（halo 可见性必须覆盖全部相交块、不许只取紧邻段）已落地并有测试；
按**性能收益**这条，在当前几何下**不成立**，已如实撤回。n### ★ 2026-11-01 18:4x 收口：`halo_report` 口径修正 + 回归确认（B5 守卫不影响默认路径）

| 文件 | md5（改前 → 改后） |
|---|---|
| `tools/halo_report.py` | `5aa64498` → **`1ab2e110`** |

**为什么改**：B5 的"省 87–99%"已被作业 251 否证（撤回见上一条），但**工具还在打印这个数** ——
工具里留一个误导口径，比代码里留一个错误分支更危险（下一个人会拿它当依据）。按铁律 0 修正：

- 报告现在**按 query 范围分两行**给出真实 need：
  - `全行`（层 20，裁剪前）⇒ `need = T` ⇒ **0.0%**；
  - `仅尾部`（层 21+，compact）⇒ `need = keep+window−1 = 267`，但那一层 `T≈140` ⇒ **0.0%**；
- 表尾明确写出结论：**当前 bench 点位没有可省的 gather 量**；只有"query 仅尾部 **且** T ≫ keep"
  （长序列 chunked prefill / decode）才有收益。
- 同时修掉一处语法错误（中文引号里嵌了 ASCII 双引号，`py_compile` 当场拦下；
  这是本轮唯一一次把坏文件推到树的短暂窗口，已立即修复并复验 `rc=0`）。

**修正后的输出（蓝区实跑，`rc=0`）**

```
 cp       T  rows_need       query      省%
  2    4236       4236        全行    0.0%
  2    4236        267       仅尾部   93.7%
  8     140        140       全行    0.0%
  8     140        140      仅尾部   0.0%     ← compact 段本来就只有 140 行
```

**回归确认（作业 252，`tmp/REG_20261101_172503`）**

B5 的 selective 守卫（默认 `full` 不走新分支）对默认路径**零影响**：

| 组 | 结果 |
|---|---|
| cp1（4 点） | 全 `rc=0`、`errors=0`：0.2553 / 0.5296 / 0.1115 / 0.1169 s |
| cp2 b1（2 点） | `rc=0`、`errors=0`：0.2100 / 0.1703 s |
| cp2 b2（2 点） | 仍由 G7 守卫显式拒绝（设计如此） |
| 5070x | **0** |

与作业 244（`0.2554/0.5285/0.1122/0.1181`、`0.2112/0.1699`）逐点差异均在噪声带内。

**B5 全链路最终状态**：语义责任（halo 可见性/清单/搬运几何）**已落并有 153 条 CPU 断言**；
`merge_global_rows` 在 NPU 上 400/400 层逐位正确；**性能收益不成立**（口径错误已撤回，
`=selective` 改为显式拒绝，工具口径同步修正）。n### ★ 2026-11-01 18:5x P4 第一步：host 侧归因**量化完成**（作业 253）

**方法**：`DS41M_HOSTTIME=1`（逐算子 host 墙钟，不采 prof），点位 `8192@0.5/b1/r1`，CP1。

**总量（关键）**：稳态一个 pass 的 host 下发 **合计 0.16 s / wall 0.14 s**，
而稳态 forward 是 **0.255 s** ⇒ **host 侧约占 forward 的 63%**，
每 pass **1091 次算子下发**，per-call 均值 **142.3 µs**。
（rank1 侧合计 0.30 s —— 比 rank0 更差，说明 host 成本对 rank 不均衡，值得单独看。）

**host 时间排行（rank0，稳态 pass，合计 0.16 s）**

| 算子 | 合计 | 次数 | µs/次 |
|---|---|---|---|
| `mqsmla` | 0.0296 s | 40 | **739.8** |
| `wq_b` | 0.0162 s | 40 | 405.5 |
| `qli` | 0.0148 s | 4 | **3707.3** |
| `wo_a` | 0.0104 s | 40 | 259.3 |
| `mega_moe` | 0.0098 s | 40 | 244.9 |
| `rope` | 0.0068 s | 80 | 85.2 |
| `idx_prolog` | 0.0068 s | 8 | 845.5 |
| `qli.topk` | 0.0066 s | 4 | 1637.8 |
| `hc_post` / `hc_pre` | 0.0047 / 0.0044 s | 80 / 80 | 58.5 / 55.3 |
| `qli.kpool` | 0.0044 s | 4 | 1089.4 |
| `kv_epilog` | 0.0040 s | 40 | 100.4 |
| `rope.k` | 0.0038 s | 80 | 47.5 |
| `topk_gate` | 0.0037 s | 40 | 91.5 |
| `wo_b` | 0.0034 s | 40 | 84.5 |

**归因结论（与 GPT 的 P4 建议对照）**

- GPT 提的是"rope 的 event/stream 精简 + `HcclLaunch` 批量"。数据说：**rope 家族只占 0.0106 s（6.6%）**，
  而 `qli` 一家（含 topk/kpool 子相）≈ **0.0258 s（16%）**、`mqsmla` ≈ **0.0296 s（19%）**。
  ⇒ 真要砍 host，**先砍 qli / mqsmla**，rope 的收益上限只有个位数百分比。
- `qli` **3.7 ms/次** 是异常高的（4 次就 15 ms）。它内部 topk 1.64 ms、kpool 1.09 ms，
  剩下 ~1 ms 在 dense 化/元数据；三者都值得单独看。
- `mqsmla` 740 µs × 40 次 = 30 ms：这是**每层固定成本**，优先看它的 per-layer 元数据/槽位计算能否复用。

**下一步（P4 第二步，单列）**

1. 先量化 `qli` 的 3.7 ms breakdown（它自己有 `_tq()` 计时点：`qli.topk` / `qli.kpool` 等）；
2. 挑**一个**点做优化 + A/B（稳态 ms/forward，同进程两臂，`DS41_ALLOW_SUBOPTIMAL` 放行）；
3. 铁律 1：性能要"等价或更好"且数值逐位不变，才允许改默认。

**注**：本轮不改代码（只归因）。P4 的价值在于"别再按 GPT 的猜测去砍 rope"——
数据已经指到 qli/mqsmla 上了。n### ★ 2026-11-01 19:0x P4 第二步结论：**分派层不是目标**，优化面在 qli/mqsmla 内部

**读了 `exec_ops.run_op` 的实现**：它只做 `OPS.get(key)`、几个关断开关的布尔判断、`PROBE_SET`
检查和一个 `try` —— 没有字符串拼接、没有 env 重读、没有逐次张量操作。**分派层已经很精简**。

所以 1091 次/pass × 142 µs 里，绝大部分是**算子实现体本身**（Python 侧），不是分派框架：

- `mqsmla` 739.8 µs/次 × 40 = **29.6 ms**（最大单项）
- `qli` 3707.3 µs/次 × 4 ≈ **14.8 ms**（其中 topk 6.6 ms、kpool 4.4 ms、其余 ~3.8 ms）
- `wq_b` 405.5 µs × 40 = 16.2 ms、`wo_a` 259.3 µs × 40 = 10.4 ms

**P4 判定（诚实）**

1. GPT 提的"rope event/stream 精简"：rope 家族合计 10.6 ms（6.6%），**上限就这么多**；
2. 真正的杠杆在 `qli` 与 `mqsmla` 的实现体内部，但那是**逐算子优化**，需要逐相位归因
   （qli 已有 `_tq` 相位点：kpool/kske/logits/cand/topk/wb）+ 每改一处都要 A/B 逐位对拍；
3. 本项目的口径是**逐位对齐**，任何 host 优化都必须"数值不变"才允许进默认（铁律 1）。

**本轮不改代码**：在没有逐相位数据前改 qli/mqsmla 属于猜测式优化，违反"改前给三元组"的规矩。
P4 的交付物到此是：**量化归因 + 目标清单 + 否定 GPT 的 rope 假设**。

**若要继续 P4（需要单独的较长时段）**

1. 给 qli 六个相位在一次 pass 内的占比打表（`_tq` 已有）；
2. 对占比最高的相位做单点优化（候选：kpool 的池构建、topk 前的 dense 化），
   同进程 A/B（`DS41M_*` 开关切换）+ 逐位对拍（`DS41M_PACKED_DIGEST_LAYERS=0,20,39`）；
3. 只有"稳态更快 且 摘要逐位相同"才改默认。n### ★ 2026-11-01 19:1x C5 **收尾完成**：命中策略写入侧也被不变量守住

| 文件 | md5（改前 → 改后） |
|---|---|
| `meta/metadata.py` | `df0da23f` → **`a9913157`** |
| `tools/test_meta_single_source.py` | `a1092155` → **`c8c90858`** |

**新增不变量**：`cfg["hit"] = ...`（整块覆盖）**只允许出现在 `meta/metadata.py`**。
扫描结果（蓝区实跑）：

```
hit  meta/metadata.py:253  cfg["hit"] = hp.to_dict()      ← set_hit 的镜像
hit  meta/metadata.py:286  cfg["hit"] = _mir              ← set_hit_dict 的镜像
（其余文件 0 处直写）
```

配合上一轮的 `push_layout`/`restore_layout` 与 `_packed` 单写入源，现在三条不变量都由
`tools/test_meta_single_source.py` 守着：**`_packed` 直写 / `set_layout` 调用 / `cfg["hit"]` 直写**。

**同时**在 `hit_of` 里加了**漂移告警**（`DS41M_META_TRACE=1` 时）：若检出有人绕过
`set_hit` 直接改 `cfg["hit"]`，打印 `[hit-drift] before → after`。
这条刷新目前仍是**兼容窗口**（不是设计目标），但现在已经**可观测**、且有测试守着，
所以下一步（删掉刷新、让 metadata 单向权威）随时可做而不会再静默回退。

**验收**

- CPU 8 套：`rc=0`、**FAIL 总数 0**
  （hit_geometry 70、batch_replay 34、**cp_plan 153**、metadata_layout 25、
  **meta_single_source 8**、cp_state_offsets 8、cp_layout/cp_state rc=0）；
- 实跑回归（作业 254，`tmp/REG_20261101_173803`）：
  cp1 四点 `0.2555 / 0.5308 / 0.1152 / 0.1202`、cp2 b1 两点 `0.2096 / 0.1711`，
  全部 `errors=0`；cp2 b2 仍由 G7 守卫显式拒绝；**零 5070x**。

**C5 最终完成度**

| 子项 | 状态 |
|---|---|
| 布局：`stage_view` / 栈式 push-pop（`pop_all_layouts` 出口兜底） | ✅ |
| 布局：`_packed` 直写 / `set_layout` 调用 单写入源 | ✅（测试守） |
| 命中：`HitPolicy` + `hit_of` 读取侧单源 | ✅ |
| 命中：`set_hit` / `set_hit_dict` 写入侧单源 | ✅（测试守） |
| 命中：漂移告警（`DS41M_META_TRACE=1`） | ✅ |
| `Geometry`/`Policy` 里 `cfg["run"]`/`cfg["features"]` 现场读 | ⬜ 官方口径已由 `from_cfg` 读一次，但仍有消费方直读；属"继续收敛"而非缺口 |
| 删掉 `hit_of` 刷新（单向权威） | ⬜ 可做（现在有测试+告警保护） |n### ★ 2026-11-01 19:5x 收工登记（本轮工作线到此为止）

**交付物（绿区 `agent/` 目录）**

| 文件 | 内容 |
|---|---|
| `今日收口_2026-11-01.md` | 作业一览（207–254 + verify_*）、17 个文件 md5、两条被修正的结论、事故与恢复、一键复核（含 `--npu` 实跑） |
| `交接_接着做什么_2026-11-01.md` | **下一步交接**：按依赖排序（5 条）+ 每条落点 `file:line` + 验收命令；三条被修正的说法；三项待裁决；开工纪律；env 速查 |
| `GPT-S14-文件级改造总账.md` | GPT §14 的 19 行逐文件对账（含"六、刷新"段，与今日实况对齐） |

**代码侧新增的可复跑入口**

- `tools/verify_all.sh`（md5 `3a82beef`）：一条命令跑 8 套 CPU 判据 + 打印关键不变量；
  `--npu` 额外提交矩阵到执行器。今日实跑：CPU `FAIL=0`、矩阵 6 点 `rc=0`、`5070x=0`、
  cp2 b2 由 G7 守卫显式拒绝（设计行为）。

**本轮线结束时的最终状态（如实）**

| 项 | 状态 |
|---|---|
| A1–A7 | ✅ 验收（207/208/209） |
| B1 / B2 / B3 / B4 / B6 | ✅（219/226、230/233、229） |
| **B5** | 语义责任 ✅（153 断言 + 400/400 层逐位）；**性能收益 ❌ 已撤回**（作业 251） |
| C4 / C5 / C6 | ✅（`cp_plan` 153 断言；metadata 三步收尾 + 三条写入不变量；总账） |
| P2 机制 / P3-1 / P3-2 / **P5** | ✅（P5 真权重 prof+noprof `rc=0`、精度 gate PASS） |
| **P4** | 归因 ✅；**代码未动**（GPT 的 rope 假设被否证） |
| G3 / G9 | ✅ 关闭 |
| G10 / G12 | 已定位（层 20 `mega_moe` / 层 1 engram），**未修** |
| **C1 / C3 / G10 出路** | ⏸ **等用户裁决**（三项，见交接文档 §四） |
| 不等长 packed / kv_manager / 真流式 chunk / 空段 / `plan_compiler` | ⬜ 未做（交接文档 §二 给了顺序与验收） |

**为什么没标"完成"**：目标要求"逐条采纳并验收"，而上表里仍有 ⏸（三项待裁决）与 ⬜（五项未做），
且 B5 的"性能收益"是被数据否证而非交付。按铁律 0，不把"部分完成"写成"完成"。

**下一轮线开工第一件事**（无论谁接手）：`bash tools/verify_all.sh` —— 不绿先别动代码。n### ★ 2026-11-01 20:2x C5 **真正收尾**：命中策略单向权威（漂移即报错）

| 文件 | md5（改前 → 改后） |
|---|---|
| `meta/metadata.py` | `a9913157` → **`d6aad1d8`** |
| `tools/test_metadata_layout.py` | `e8bc2237` → **`c2f99973`** |

**改了什么**：`hit_of` 里的"以 `cfg["hit"]` 为准**静默刷新**"补丁删掉了，改成：
若 `cfg["hit"]` 与 `metadata.hit` 不一致 ⇒ **显式 `RuntimeError`**（`[hit-drift] …`
写入应走 `metadata.set_hit/set_hit_dict`）；过渡期可 `DS41M_HIT_DRIFT_OK=1` 显式放行
（放行时才按 cfg 覆盖，行为等同旧路径）。

**为什么现在敢删**：写入侧已全部收敛（上一轮 + `tools/test_meta_single_source.py` 守着
"`cfg["hit"]` 直写只允许出现在 metadata"），且本轮 NPU 矩阵实测 **`hit-drift` 出现 0 次**
（cp1/cp2 两臂的 1091×N 次 `hit_of` 调用里，没有一次检出漂移）。

**验收**

- CPU（`tools/verify_all.sh`）：8 套 `rc=0`、**FAIL=0**，
  `test_metadata_layout` 由 25 → **27 PASS**（新增：raw 写 ⇒ 报错含 `hit-drift`；
  `DS41M_HIT_DRIFT_OK=1` ⇒ 放行并按 cfg 覆盖）；
- NPU 矩阵（作业 255，`tmp/REG_20261101_191558`）：

| 组 | steady | errors |
|---|---|---|
| cp1 b1 h=0.5 | 0.2555 s | 0 |
| cp1 b2 h=0.5 | 0.5341 s | 0 |
| cp1 b1 h=0.97 | 0.1131 s | 0 |
| cp1 b2 h=0.97 | 0.1185 s | 0 |
| cp2 b1 h=0.5 | 0.2095 s | 0 |
| cp2 b1 h=0.97 | 0.1702 s | 0 |
| `hit-drift` | **0 次** | — |
| 5070x | **0** | — |
| cp2 b2（两点） | — | 由 G7 守卫显式拒绝（设计如此） |

**C5 至此完整**：布局（`stage_view`/栈式 push-pop/`_packed` 单写入源）+ 命中
（`HitPolicy` 读取单源 + `set_hit(_dict)` 写入单源 + 漂移显式报错）——**两边都是单向权威、
都有测试守着**。交接文档 §二-2 的"删刷新补丁"一项已完成。n### ★ 2026-11-01 20:5x qli 相位表（P4 归因细化）：**rank 间差 3 倍**，疑似索引缓存未命中

**数据（同一份 `DS4عال1M_HOSTTIME=1` 记录，`tmp/P4HT_20261101_172830`）**

| rank | 整算子 `qli` | `qli.kpool` | `qli.logits` | `qli.topk` |
|---|---|---|---|---|
| rank0（稳态 pass，整 pass 0.16 s） | 3.71 ms/次 | **1.09 ms/次** | — | 1.64 ms/次 |
| rank1（整 pass 0.30 s） | **12.04 ms/次** | **4.76 ms/次** | 4.95 ms/次 | 1.85 ms/次 |

- 两 rank 的 `qli` 次数相同（4 次/pass）、代码路径相同，但 **rank1 是 rank0 的 3.2 倍**；
  分相位看，`qli.kpool` 差 **4.4 倍**（4.76 vs 1.09 ms），`topk` 只差 1.1 倍。
- `qli.kpool` 的实现是 `_dg_paged_to_dense(...)`（`ops/exec_indexer.py:2785`），
  它内部**已经**按 `(n, bt.data_ptr(), rpb, device, cu_k)` 缓存了 `(p, slot)` 索引
  （`_DG_SLOT`，见该函数注释："旧代码每次调用 Range/FloorDiv/… + `int(slot.max())` 同步"）。
  剩余成本 = 真正的 gather 派发 + 缓存**未命中**时的重建。

**假设（可一次实验判定）**：rank1 上 `bt` 的**对象身份每层都在变**（旋转块表），
于是 `_DG_SLOT` 每次都 miss ⇒ 每层重建 `p/slot`（含 `int(slot.max())` 同步）⇒ kpool 4.4 倍。

**实验（只加日志，不改逻辑）**：在 `_DG_SLOT` 命中/未命中两处计数，`DS41M_DG_SLOT_STAT=1` 时
按 `(rank, layer)` 打印；同一 pass 对比两 rank 的 miss 次数。
若 rank1 的 miss 数 ≈ 4（每次调用都 miss）、rank0 为 0 ⇒ 假设成立。

**若成立，修法（低风险）**：把缓存键从 `bt.data_ptr()` 改成**块表内容**的指纹
（例如 `bt` 的 `int16` 视图做 sha 或直接用 `(bt._version, bt.shape)` + 内容比较），
或让 `idx_k_write` 复用同一块 bt 缓冲（原地旋转）。前者不动 kernel，后者可能减少一次 H2D。

**注意**：这是**纯 host** 改动，数值应逐位不变 ⇒ 验收 = `tools/verify_all.sh` 全绿
+ `DS41M_PACKED_DIGEST_LAYERS=0,20,39` 两臂摘要相同 + 稳态 ms/forward 下降。

**顺带确认**：`qli.logits`（4.95 ms/次）在 rank1 上也偏高，值得同一次实验里一起打点。n### ★ 2026-11-01 21:0x P4 归因收敛：`qli.logits` 是**真内核**，没有 Python 侧可省的重复

**读代码结论（`ops/exec_indexer.py:2880-2919`）**

`qli.logits` 这一相位就是 `_dgd.fp8_fp4_mqa_logits(...)` 的调用点，三处分支：

| 分支 | 条件 | 调用 |
|---|---|---|
| b>1 打包 | `nseq > 1` | 按序列逐段调（段内 ks=0、ke=段内因果长），先 `p_sum` 填 `-inf` |
| 整段 | `T <= _qchunk` | 一次 `fp8_fp4_mqa_logits(q, k_dense, wq, ks, ke, max_seqlen_k=_maxk)` |
| 分段 | `T > _qchunk` | 按 q 切块多次调用（`_qchunk` 由 `maxk` 推出，上限 8192） |

- **没有** `get_sparse_mqa_logits_metadata` 的每层重建（那条只在 `DS41M_DG_SRC_SPARSE=1` 的
  源层自用路径上，默认不调，见 `2960-2967`）；qsli 才有 `qsli_sched` 的 `_meta_fwd` 缓存。
- 所以 5.25 ms/次 是**内核派发 + 大输出**的成本：单序列下 `p_sum` 形状 `[T, maxk]`
  （本次 T=4236、maxk≈4096 ⇒ bf16 ≈ 33 MB），4 次/pass。

**为什么 rank1 的 3.2 倍被证伪**（作业 257）：`_DG_SLOT` 两 rank 都是 `HIT=38/MISS=2`、
`bt_ptr` 相同，同 pass 内两 rank 的 qli 相位几乎完全一致（kpool 2.75/2.76 ms、logits
5.25/5.24 ms、topk 1.88/1.87 ms）。**之前那张"rank1 慢 3.2 倍"的表是跨 pass 比较**，
含首次装载/分配的一次性成本 —— 判定为**比较错误**，不是 rank 差异。

**qli 真实稳态成本（两 rank 一致）**

| 相位 | µs/次 | ×4 次/pass | 占整 pass host（0.16 s） |
|---|---|---|---|
| `qli.logits` | ~5250 | **21.0 ms** | ~13% |
| `qli.kpool` | ~2750 | **11.0 ms** | ~7% |
| `qli.topk` | ~1880 | **7.5 ms** | ~5% |
| 合计 | | **~39.5 ms** | **~25%** |

**还剩一个候选（有风险，需 A/B）**：把 `p_sum` 的输出缓冲改成**跨层复用**（`_cache_buf`）。
kernel 只写 `[ks, ke)` 区间（`[0, ks)` 不写），现在用 `torch.empty` ⇒ 那部分是**未初始化垃圾**；
复用后会变成"上一层的残留值"。若 topk 真的只看 `[ks,ke)`（`end=ke`），数值不变；
但**垃圾→残留**是"从不确定变成确定"，必须用 `DS41M_PACKED_DIGEST_LAYERS=0,20,24,39` 两臂逐位对拍，
不通过就回退。预计收益 ≤ 2 ms/pass，**性价比一般**。

**P4 判定（最终）**：host 侧的"免费午餐"到此为止 ——
`mqsmla`（29.6 ms）与 `qli`（39.5 ms）合计约 43% 的 host，但都在**实现体/内核派发**里，
继续压需要逐算子重写，且每步都得满足"更快且逐位不变"。
按投入产出，**建议 P4 到此收口**，把时间投到功能缺口（不等长 packed / plan_compiler）。n### ★ 2026-11-01 20:4x 128K 三组默认采集（作业 258）——trace 已出

**命令口径**：`tools/run_hit.sh --points-spec '128k@<h>/b1/r1' --cards 0..7 --iters 4`
（**默认采集策略**：ALOP prof 轮 + 无 prof 稳态轮；prof 默认只采 rank0，
`--alop-config configs/alop_mindstudio.yaml`），`cp>1` 显式 `DS41M_CP_HIT=1`。

| 组 | 并行 | 稳态（results.tsv） | L | M | R | H | S | hit | plan_hash | trace |
|---|---|---|---|---|---|---|---|---|---|---|
| ① 128K h90 | **dp8 / ep8 / cp1** | **0.6510 s** | 131072 | 13107 | 13247 | 117965 | 117825 | 0.900002 | `b72fa27d657a` | **8.0 MB** |
| ② 128K h80 | **dp4 / ep8 / cp2** | **3.8527 s** | 131072 | 26214 | 26354 | 104858 | 104718 | 0.800003 | `43c981a44fa4` | **14 MB** |
| ③ 128K h60 | **dp2 / ep8 / cp4** | **3.4507 s** | 131072 | 52429 | 52569 | 78643 | 78503 | 0.599998 | `703694aa6cef` | **15 MB** |

- 三臂 `rc=0`、**零 5070x、零 `⚠️`**；backend 串：`idx=dg,qsli=dg,moe=deepgemm,cp_cmp=seg,cp_hit=1,cp=N,align=token`
- ②③ 的 rank 侧稳态：cp2 `3.728 s`(r0) / `3.853 s`(r1)；cp4 `3.326 s`(r0) / `3.451 s`(r1)
- 峰值显存（rank1 侧）：cp2 **10.24 GiB**、cp4 **11.25 GiB**

**trace 目录（PROF_* 原始，ALOP）**

```
① /mnt/xds/sfs/l00972766/tmp/TRACE128K_20261101_203521/cp1_h0.90/cp1dp8ep8_L131072_hit0.9_batch1_replayon_prefill/prefill/PROF_000001_20261101203643260_03963267QQOEOEQJ
② /mnt/xds/sfs/l00972766/tmp/TRACE128K_20261101_203521/cp2_h0.80/cp2dp4ep8_L131072_hit0.8_batch1_replayon_prefill/prefill/PROF_000001_20261101203811573_03977256LAJLQPMN
③ /mnt/xds/sfs/l00972766/tmp/TRACE128K_20261101_203521/cp4_h0.60/cp4dp2ep8_L131072_hit0.6_batch1_replayon_prefill/prefill/PROF_000001_20261101203948515_03990302PMPIOKRN
```

**读法提醒（重要）**

- 三臂的 **R 不同**（13247 / 26354 / 52569）⇒ 稳态**不能横比**："谁更快"要按 `h` 对齐
  （同一 R 下比 cp），或按"每 token 成本 = steady / R"归一：
  ① 0.6510/13247 = **49.1 µs/行**；② 3.8527/26354 = **146.2 µs/行**；③ 3.4507/52569 = **65.6 µs/行**。
  这个归一值差异主要来自 **cp 的 KV all-gather 与压缩池**随 R 增长（cp2 尤其高，值得追）。
- trace 大小 8/14/15 MB 只是 `PROF_*` 原始目录；MindStudio 视图要事后
  `msprof --export` 转（见 `PIPELINE.md`）。
- 若要 A/B（例如 cp1 下 h90 vs 其它开关），**必须同 R**；跨 h 的比较请用上面的归一列。

**下一步（若要继续分析）**：从三份 trace 里各取一份 `op_summary`，看
`HcclLaunchAicpuKernel` / `Muls` / `Matmul` 在 cp1/2/4 下的占比 —— 这正是 cp2 每行成本偏高的可疑点。n### ★ 2026-11-01 21:0x 128K trace 导出：**原始目录已产出，CSV 导出在本 CANN 上没跑通**（如实登记）

**已交付（可用）**：三份 `PROF_*` 原始采集目录（ALOP，默认策略，只采 rank0），路径与大小见上一条。
MindStudio 可直接打开这些目录；本机命令行导出没成功。

**导出尝试记录（都在 `tmp/` 下的脚本里，可复跑）**

| 尝试 | 结果 |
|---|---|
| `msprof`（PATH 里） | `error while loading shared libraries: libascendalog.so` ⇒ 需显式 `LD_LIBRARY_PATH` 指向 `cann-9.2.0/aarch64-linux/lib64` 等 |
| `msprof --export --output=... -dir=<PROF>` | `Argument --export : expected one argument`（语法要求 `--export=on`） |
| `--export=on --output=<结果目录> -dir=<PROF>` | `The path ".../_export/cp1" does not have PROF dir` ⇒ **`--output` 是"包含 PROF_* 的输入目录"**，不是结果目录 |
| `--export=on --output=<PROF父目录>` | `Export all data in prefill done`，但产物只有 `mindstudio_profiler_output/README.txt`（无 csv） |
| 加 `--type=op_summary` | `Argument --type: invalid value. Please input 'text' or 'db'` |
| `--type=text`（三臂） | 均 `Process profiling data complete`，**仍然只有 README.txt**；伴随 `WARNING Cluster Tuning did not complete!` |

**结论**：本 CANN（9.2.0）的 `msprof --export` 需要额外解析资产（profiler plugin / cluster tuning），
在这台机器上没配齐 ⇒ **命令行导出这条路暂时走不通**。
替代路径（按成本排序）：
1. **MindStudio GUI 打开 `PROF_*` 目录**（数据完整，可直接看 timeline/op 视图）；
2. 用 `--type=db` 导出 db 再用 sqlite 查 `op_summary` 表（未试，值得一次）；
3. 只看 host 侧：`DS41M_HOSTTIME=1`（已有能力，不需要 trace）。

**不影响已交付的结论**：三臂的稳态与几何（`results.tsv`）已拿到，且**归一化每行成本**
（cp1 49.1 / cp2 **146.2** / cp4 65.6 µs/行）已经足以指出"cp2 异常"这一疑点，
只是**设备侧归因**要等 trace 能读出来。

**下一步（若要设备侧归因）**：优先试 `--type=db`；若也失败，改用 MindStudio GUI 或找 CANN 侧
profiler 资产（`tools/profiler/lib64` 是否缺 `libmsprof*.so`）。n### ★ 2026-11-01 21:0x 128K **纯性能重测**（作业 259，`--noprof-only --iters 8`）——推翻上一组数字

| 组 | R | 上组（默认：先 prof 轮再 noprof 轮） | **本次（纯 noprof，8 轮）** | 差异 |
|---|---|---|---|---|
| dp8/ep8/**cp1** h90 | 13247 | 0.6510 s | **0.5265 s**（526.55 ms/fwd） | −19% |
| dp4/ep8/**cp2** h80 | 26354 | 3.8527 s | **0.5561 s**（556.10 ms/fwd） | **−86%** |
| dp2/ep8/**cp4** h60 | 52569 | 3.4507 s | **0.5711 s**（571.07 ms/fwd） | **−83%** |

三臂 `rc=0`、`err=0`。

**为什么会差 6–7 倍**：默认采集策略是"先 prof 轮（只采 rank0）再 noprof 轮"。
`PERF_RULES` 早写明了这个坑：**CP 组里只要有被采集的 rank，所有集合通信都要等它**
⇒ cp2/cp4 时 rank0 就在 CP 组内，稳态被 msprof 拖慢 6–7 倍；cp1(dp8) 时 rank0 只在自己那个
DP 组，影响小但也偏了 19%。**上一组数字不能当性能用**。

**归一化每行成本（本次，符合预期）**

| 组 | ms/forward | R | µs/行 |
|---|---|---|---|
| cp1 | 526.55 | 13247 | **39.8** |
| cp2 | 556.10 | 26354 | **21.1** |
| cp4 | 571.07 | 52569 | **10.9** |

⇒ **CP 越大、每行越便宜**（并行度提升），与上一组"cp2 每行成本 3 倍"的异常结论**相反**。

**撤回**：上一条把"cp2 每行成本 146.2 µs（3 倍于 cp1）"当成疑点 —— 那是**采集污染**造成的假象，
不是 KV all-gather 或压缩池的问题。task 侧的 trace 归因（导出失败）也就不必再追了。

**结论**：128K 三组在当前实现下稳态 **0.53–0.57 s/forward**（R 从 13k 到 53k），
三组几乎相同 ⇒ 说明这条路径的瓶颈**不是行数**，而是层数 × 每层固定开销（与 P4 的
host 归因一致：每 pass 1091 次算子下发）。

**记法（写进规则）**：**任何 CP 相关的性能数，必须用 `--noprof-only`**；
要 trace 就单独跑一次 prof，绝不把同一进程的 prof 轮与 noprof 轮混作性能数。
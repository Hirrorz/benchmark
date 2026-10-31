# 采集方法：真权重 + EP8 的 profiling（**带精度**：能正常吐字 × 自然路由）

> 一句话：**用 `tools/run_real.sh` 一条命令**，在 8 卡上跑「真权重 + 真 ids + EP8 + engram 真表」的
> prefill / 命中两种口径，每个口径产出**与合成口径同构的 MindStudio 算子流水图**（8 rank 各一份）
> + 一个干净的单 pass 稳态耗时 + 一条**精度凭据**（末位 token 与官方参考逐位对齐）。
>
> 与合成入口 `tools/run_hit.sh` 的分工：那是**假的**（权重假生成、路由强均、前缀零值）——用来对
> 形状/次数/耗时；这是**真的**（真权重、自然路由 = 不做负载均衡、engram 真表按卡分片）——用来对
> **数值 + 真实算子流水**。两份脚本各自独立，互不影响。
>
> 面向执行者（不是设计文档）。定稿 2026-10-16，对应 `tools/run_real.sh`。

---

## 0. 三条命令跑完（TL;DR）

```bash
cd <ds41_mini 仓根>                       # 里面应有 tools/run_real.sh、configs/mini_real.yaml、launch_mini.py
source env_pack.sh                        # 只在**新 shell** 需要
npu-smi info | sed -n '1,30p'             # ① 空气检查：8 张卡 Util=0%（有别人在跑就别开，见 §7）

bash tools/run_real.sh -L 16k             # ② prefill + hit 两个口径各一轮（默认 --mode both）
# 想先看会跑什么而不真跑：bash tools/run_real.sh -L 16k --dry-run

less /mnt/xds/sfs/l00972766/real_16k/SUMMARY.md   # ③ 看汇总（或 <你的 -o 目录>/SUMMARY.md）
```

跑起来之后**每个 rank 的输出实时打到控制台**（带 `[r0]`…`[r7]` 前缀）且同时落盘，最底下一行是
进度条（终端不支持时降级成每 30s 一条日志快照）。结束时最后一张表就是交付物。

---

## 1. 前置条件（跑之前确认四件事）

| 检查 | 命令 | 合格标准 |
|---|---|---|
| **8 张卡空着** | `npu-smi info` | NPU Util 全 0%；HBM 只有基线（~5 GB/卡）。有人共用 → 先等（这台机器有过邻居容器） |
| 环境已引导 | `source env_pack.sh` | `python3.11 -c "import torch_npu"` 能过 |
| 输出盘有空间 | `df -h <输出目录>` | 一个口径一份 trace ≈ 2–4 GB/rank（`--collect-ranks 0` 时只 1 份） |
| 真 ids / 官方参考在位 | `ls /mnt/xds/sfs/l00972766/parity/{ids_16k.pt,ref/dump_rank0_call1.pt}` | 两个都在（精度 gate 要用；缺了只能用 `--no-precision`） |

> ⚠️ 根盘常年贴满：`/tmp` 满了进程会假死；`env_pack.sh` 已处理 `TMPDIR`，别自己指回 `/tmp`。
> ⚠️ 驱动会自己 `ulimit -l unlimited`（厂商主机窗口要 12.9 GiB/层的 pinned 内存）；**别在跑之前把它降回去**。

---

## 2. 标准流程：16k 真权重 EP8（prefill + hit）

```bash
bash tools/run_real.sh -L 16k
```

- 默认口径（**写进 `PARAMS_*` 指纹**，改任何一项都会触发重跑）：

  | 项 | 值 |
  |---|---|
  | 并行 | dp8 / ep8 / cp1（8 卡） |
  | 权重 | `DeepSeek-V4.1-Flash-cann-mxfp8-mxfp4`（476 GB，safetensors mmap） |
  | 路由 | **自然路由（不做负载均衡）** —— 给了真 ids 就自动关掉强均 |
  | engram | `offload=true` + **`engram_tp_size=8`**（每卡 ~12.9 GiB 分片）+ 真表（384M 行/层） |
  | 权重常驻 | `resident=true`、`host_cache=false` |
  | 其它 | `moemode=ascendc`、`iters=3`、`ulimit -l unlimited` |

- 每个口径跑两轮：**prof 轮**（出 trace，默认 `--iters 3`）+ **noprof 轮**（干净单 pass 稳态耗时，
  用来对官方 2394.7 ms / 6841.7 tok/s）。
- 耗时：16k 单口径 ≈ 3–5 min（首 pass 含权重装载）；两个口径 ≈ 8–12 min。

**跑完自动打四条验收**（任何一条不过会写进 SUMMARY 并让脚本 exit≠0）：

1. **自然路由**：`[elb/balroute]` 行**必须缺席**（出现 = 没给 ids 或被人为开了强均）；
2. **厂商路径**：`[engram] 厂商 ElasticBuffer 就绪` **恰好 2 × world_size 次**（层 1、层 14 各一次 **× 每卡一行** ⇒
   8 卡 = **16** ✓）。注：早期这里写的是"恰好两次"，那是按 **rank0 单条日志**估的 ✗ —— 全 rank 汇总应为 16
   （2026-10-17 实测：8 rank 时 `=16`、`不可用=0` ✓；按旧的"=2"判会误报 ✗）。
   出现 `厂商 ElasticBuffer 不可用` → 会静默退回主机取表（trace 缺 `EngramFetch`）→ FAIL；
3. **trace 断言**：`EngramFetch = EngramFetchWait = engram层数×iters×卡数`（16k 全量 = 2×iters×8）、
   `MegaMoe = 层数×iters×卡数`（= 40×iters×8）；
4. **精度 gate**：`NEXT_TOKEN mini=396 ref=396 match=YES`（官方参考 = id 396 `' that'`，score 37.750）。

---

## 3. 常用变体（挑一行抄）

| 想干什么 | 命令 |
|---|---|
| 只普通 prefill（全长 16k，"能吐字"那条） | `bash tools/run_real.sh -L 16k --mode prefill` |
| 只命中口径（stage A 真前缀 + stage B 重算尾段） | `bash tools/run_real.sh -L 16k --mode hit` |
| 换命中率 | `bash tools/run_real.sh -L 16k --mode hit --hit 0.96875` |
| 加一轮逐层对拍（rank0 落中间张量 + 对官方参考） | `bash tools/run_real.sh -L 16k --parity` |
| trace 里跳过含装载的首 pass | `bash tools/run_real.sh -L 16k --iters 3 --warmup-iters 1` |
| **冒烟**（只跑前 15 层，~2 min） | `bash tools/run_real.sh -L 16k --layers 15 --no-precision` |
| 只要 trace，不要稳态轮 | `bash tools/run_real.sh -L 16k --prof-only` |
| 只要稳态，不要 trace | `bash tools/run_real.sh -L 16k --noprof-only` |
| 换输出/profiling 目录 | `bash tools/run_real.sh -L 16k -o <输出> -P <trace目录>`（`-P` 支持 `{L}`/`{mode}` 占位） |
| 重采（覆盖已有结果） | `bash tools/run_real.sh -L 16k --force` |
| 逃生：厂商路径炸了 | `bash tools/run_real.sh -L 16k --engram-host`（退回主机取表；数值等价但 trace **缺** `EngramFetch/Wait`） |
| 看会跑什么 | `bash tools/run_real.sh -L 16k --dry-run` / `--help` |

> ⚠️ **`--layers` 冒烟必须 ≥15**：engram 层是 **1 和 14**，截到 14 层以下就一个 engram 层都没有，
> 冒烟会"假绿"（既没有厂商就绪、也没有 EngramFetch）。
> ⚠️ **MoE 分块默认开**（`moe_chunk_max_len: 4096`，官方 `get_moe_num_chunks` 同口径）：16k 时每层发
> **4 次** MoE —— 这是官方真权重跑法（cann-recipes config 全写 4096），也是唯一能把 16k×40 层塞进
> 91.8 GiB 的姿势（厂商融合算子内部 workspace 按每 call token 数走，T=16384 单次 ≈20 GiB 瞬时）。
> 关掉用 `--no-moe-chunk`（那时 trace 期望 = 每层 1 次 MegaMoe）。

---

## 4. 交付物：跑完你手上有什么

```
<输出>/
├─ SUMMARY.md                  ← 一张表：口径 / h / R / trace份数 / prof用时 / 稳态 / 精度gate
├─ rounds.tsv                  ← 每轮一行（供画图/回归对比）
├─ driver_<ts>.log             ← 全部日志（8 rank 输出 + 脚本自己的每一步）
├─ L16384/
│  ├─ config_prefill.yaml / config_hit.yaml   ← 本轮真实配置（含溯源）
│  ├─ PARAMS_prefill / PARAMS_hit             ← 口径指纹（--force 重采时的判据）
│  ├─ logs_prefill/rank{0..7}.log             ← 每 rank 原始日志
│  ├─ logs_prefill2/…                         ← noprof 轮（机器可读的一轮）
│  └─ prefill_prefill/<host>_<pid>_<ts>_ascend_pt/   ← 每 rank 一份 MindStudio trace
│     └─ ASCEND_PROFILER_OUTPUT/{op_statistic,kernel_details,operator_details}.csv + trace_view.json
└─ L16384/prefill_hit/…                        ← 命中口径的 trace（同上）
```

交付给谁看什么：

- **要算子流水图** → `<trace>/ASCEND_PROFILER_OUTPUT/trace_view.json`（导进 MindStudio）；
- **要对次数/时长** → `op_statistic.csv`（按 `OP Type` 汇总）、`kernel_details.csv`（按 `Stream ID` 看
  侧流：engram 三件套应落在**非 61** 的流上，主干层算子都在 61）；
- **要对数值** → SUMMARY 里的精度 gate 行 + `logs_prefill/rank0.log` 里 `== 贪心下一个 token ==` 段。

---

## 5. 这些参数什么意思（口径细节）

| 参数 | 含义 / 为什么这么定 |
|---|---|
| `--mode prefill\|hit\|both` | prefill = 全长 16k 一次算完（"能吐字"那条）；hit = 前缀命中口径（stage A 真算前缀、stage B 只重算尾段，读共享池）。两者**数学等价**，末位 token 都应打出 396 |
| `--hit h` | 命中率（hit 口径的 `hit.ratio`）；`R`（重算 token 数）由 `h` 和 L 推出 |
| `--iters N` | prof 轮跑几个 pass（trace 里的次数期望按它算）；默认 3 |
| `--warmup-iters K` | 起跑线 barrier 之后、`profiler.start()` 之前先跑 K 个**不进 trace** 的 pass（首 pass 有装载/H2D，会污染流水图） |
| `--collect-ranks 0` | 窗口落盘只收 rank0（省 8× 磁盘）；默认 all |
| `--resident / --lazy` | 权重常驻 HBM / 逐层换入换出。真权重建议 resident（HBM 峰值实测 ~51/91.8 GiB） |
| `--engram-host` | 退回"主机线程取表"：数值等价、但 trace 里**没有** `EngramFetch/EngramFetchWait` |
| `--moemode ascendc` | MegaMoe 后端。与合成口径一致；**对官方 trace 时注意官方是 deepgemm 系内核**，逐算子名会对不上（口径说明见 §8） |
| `--alop-config <yaml>` | **采集引擎是 ALOP**（2026-10-16 用户口径：以后都用 ALOP）。默认 `configs/alop_mindstudio.yaml` = 纯 aclprof C 级（`python_profiling` 永久禁用）→ 产出 `PROF_*` 原始目录；MindStudio trace 由 `msprof --export` 事后转出（见 `PIPELINE.md`） |
| `--no-alop-config` | 用 ALOP 内置默认（纯 aclprof C 级）：更轻，但只有 `PROF_*` 原始目录 → 次数断言自动跳过并提示 |
| `--no-moe-chunk` | 关掉官方同款 MoE 分块（默认开 4096 token/块，见 §3 警告） |

---

## 6. 中断 / 重跑：不会白跑

- 每轮跑完写 `PARAMS_<mode>`（口径指纹）+ `rounds.tsv` 一行；再次运行**同指纹**的点会**跳过**
  （打印"已有结果，跳过"）。
- 想强制重采：`--force`（旧 trace 会先改名成 `*.bak_<ts>` 再重跑，不直接删）。
- 中断后直接重跑同一条命令即可；只有没跑完的那一轮会重来。

---

## 7. 出问题怎么查（按症状查表）

| 症状 | 大概率原因 | 处置 |
|---|---|---|
| 脚本开头报 `⚠️ env_pack.sh 引导失败` | 新 shell 没 source 环境 | `source env_pack.sh` 后重跑 |
| `ModuleNotFoundError: No module named 'torch'` | 同上（或绕过了 env_pack） | 同上 |
| 汇总里 `[elb/balroute]` 行出现 | 没给 `--ids-file`（或 ids 文件丢了）→ 退回强均路由 | 补 ids；这条自检会 FAIL |
| `厂商 ElasticBuffer 不可用（…）` | ≥2 rank 不满足 / 建组顺序问题 / 厂商初始化失败 | 看括号里的异常；先试 `--layers 15` 冒烟；实在不行 `--engram-host`（放弃那两个算子） |
| trace 里 `EngramFetch` = 期望的**两倍** | 侧流半路失败 → 主干把整条链重跑了一遍 | 搜日志 `⚠️ [engram] 侧流发起失败`（现在**永远会打**）；按异常修（历史上是 `KeyError: '_packed'`） |
| 每个 pass 都打 `厂商 ElasticBuffer 就绪` | 单槽缓存被两层互相顶掉（已修：每层一个专用 HCCL 组） | 确认日志里有 `层 N 专用进程组就绪` |
| 进程假死 / 被 OOM killer 收尸 | `engram_tp_size` 写错（=1 → 每卡 183 GiB）/ `/tmp` 满 | 确认 config 里 `model.engram_tp_size: 8`；`TMPDIR` 别指 `/tmp` |
| **跑到层 33+ 全部 rank 静默不动**，或刷 `NPU out of memory. Tried to allocate 20.1 GiB` | 40 层常驻权重（~48 GiB）+ 厂商 MoE 算子每次 ≈20 GiB 的**瞬时 workspace**（CANN 内部，与分块/缓冲容量都无关，实测过）——总账 68 < 91.8 够，但**默认分配器把大块切碎**了 | 驱动已默认 `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:4096`（≤4 GiB 的块不再被拆 → 20 GiB 找得到整块；这就是 CANN 提示原文那句）。仍不行再退 `--lazy`（慢 30×，trace 多逐层搬运）或 `--iters 1 --warmup-iters 1`（profiler 缓冲 ∝ pass 数） |
| 每层 `MegaMoe` 次数 = 层数×4 | 正常：16k prefill 走官方同款 MoE 分块（`moe_chunk_max_len: 4096`） | 想回"每层 1 次"用 `--no-moe-chunk`（但 16k 全 40 层会顶显存） |
| 仓库目录突然涨到几百 GB（`extra-info/data-dump/`） | 设备异常时 CANN 往 **cwd** 写 dump（单次 20+ GB） | 驱动已把 `ASCEND_WORK_PATH` 指到 `<输出>/_ascend_work`；历史残留可整目录删（它们只是异常现场） |
| `Acquire HCCL channel failed` + 崩 | 厂商资源建在了 `mega_moe` 对称缓冲**之前** | 别把 engram 发起点提到层 0 之前（驱动已按 层1头/层2头 排程） |
| 8 卡被别的容器占了 | 邻居共用（这台机器常见） | `npu-smi info` 看 HBM/Util；等对方退出，或换卡表 `--cards` |

---

## 8. 与官方对拍时的三点口径说明

1. **MoE 后端**：mini 走 `ascendc`（MegaMoe），官方 trace 是另一套（deepgemm 系）内核算子名 →
   **逐算子名不等价**，能对的是"层结构 / 次数 / 相对时长"。
2. **MoE 分块**：16k/40 层跑的是**官方同款 4 块**（`moe_chunk_max_len: 4096`）→ 每层 4 次 MoE。
   hit 口径的每轮 token ≈ 640 < 4096 → **1 次**（与官方 hit 点 trace 同构）；16k prefill 的对照基准
   （官方 2394.7 ms）本身就是 cann-recipes 带分块跑出来的，所以这条是**同口径**，不是近似。
3. **真权重图比合成多出的东西**：真权重没有 `share_across_layers` → 每层每 pass 都要重做一次
   expert 权重打包，trace 里会多出逐层 `TransData`/`Copy`；这是**真实负载**的一部分，不是 bug。
3. **官方基准数字**（同权重 / 同 ids / 同 EP8，`parity/ref/result.json`）：
   prefill **2394.7 ms / 6841.7 tok/s**、next-token **id 396 `' that'`（37.750）**。
   mini 的耗时不该逐项对（层数/后端/装载口径不同），**逐位对齐的只有末位 token**。

---

## 9. 谁负责什么

| 事 | 谁 |
|---|---|
| 跑采集、贴 SUMMARY | 采集侧（照本文件 §0/§2） |
| 口径判据（什么叫"对得上"） | 设计侧（`PLAN.md` / `HIT97_MATRIX.md`） |
| 算子流水图解析、逐算子对拍 | 分析侧（MindStudio，吃 §4 的 trace） |

---

## 附录 A：2026-10-17 首次实测记录（真权重 EP8 16k，ALOP）

命令：`bash tools/run_real.sh -L 16k --parity` → 再 `--iters 3 --warmup-iters 1`（both 口径）。

| 口径 | prof 用时 | noprof 用时 | 稳态（除首轮） | trace | 精度 gate |
|---|---|---|---|---|---|
| prefill | 3m48s | 2m39s | **2370 ms** | 8 份 | ✅ `id=396/396`（top8 重合 6/8） |
| hit (h=0.96875) | 5m04s | 3m42s | **742 ms** | 8 份 | ✅ `id=396/396`（top8 重合 3/8） |

- 官方基准（同权重/ids/EP8）：prefill **2394.7 ms** ⇒ mini 快 ~1% ✓；
- trace 断言实测：`EngramFetch=48`（=2×3×8 ✓）、`MegaMoe=3840`（=40×3×8×4 ✓）；
- `next_token_prefill.json` = `{ok:true, mini_top1:396, ref_top1:396, top8_overlap:6}` ✓。

**逐层对拍（mini 真 prefill vs `parity/ref`，`tools/compare_layers.py`，rank0 与 rank7 结论逐字相同 ✓）**：

```
统计: ✅ 1 层, ⚠️ 2 层, ❌ 37 层    最差: 层 19 (相对误差 4.375e-01)
⚠️ 第一处分歧层: 层 1 相对误差 9.027e-03 （层 0 及以下一致）
❌ 首次严重分歧: 层 3 相对误差 7.598e-02
```

- **层 0 绝对误差 = `0.000e+00`（与官方逐位相同 ✅）** ✓ —— 这是逐层对拍能给的最强结论；
- 层 1 起 9.0e-3 → 层 3 7.6e-2 → 层 19 最差 4.4e-1：与仓库早先记录的 **官方自身 run-to-run 混沌**
  （1.4e-3 → 层 2-3 ~1.0）同量级 ⇒ **"✅40" 这种预期不现实** ✗；深层正确性只能靠
  **next-token（id=396）** 与稳态时延判 ✓，不要拿逐层表当回归门 ✗。
- `collect/` 8×13.4 GB ≈ **107 GB**（parity 轮产物，对拍做完后可删 ✓ —— 删前问一声 ✓）。

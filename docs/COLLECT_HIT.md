# 采集方法：前缀命中（hit）prefill profiling

> 一句话：**用 `tools/run_hit.sh` 一条命令，在 8 卡上跑若干「序列长度 × 命中率」的点，每个点产出一份
> MindStudio trace（8 个 rank 各一份）+ 一个干净的单 pass 稳态耗时**，用来和官方 `ds41_light` 同点对拍。
>
> 面向执行者（不是设计文档）。照抄命令即可；每步都写了「看到什么算成功」「出错怎么办」。
> 定稿 2026-10-15，对应 `tools/run_hit.sh`（旧名 `run_hit97.sh` / `run_hit97_8points.sh` 是它的软链）。

---

## 0. 三条命令跑完（TL;DR）

```bash
cd <ds41_mini 仓根>                       # 里面应有 tools/run_hit.sh、configs/、launch_mini.py
source env_pack.sh                        # 只在**新 shell** 需要；之后 source env_local.sh 即可
npu-smi info | sed -n '1,30p'             # ① 空气检查：8 张卡 Util=0%（有别人在跑就别开）

bash tools/run_hit.sh                     # ② 官方 hit97 矩阵 8 个点全跑（16k…1m）
# 想看落点/参数而不真跑：bash tools/run_hit.sh --dry-run

less /mnt/xds/sfs/l00972766/hit97_8points/SUMMARY.md   # ③ 看汇总（或 <你的 -o 目录>/SUMMARY.md）
```

跑起来之后**每个 rank 的输出会实时打到控制台**（带 `[r0]`…`[r7]` 前缀），同时全部落盘，
**最底下一行是进度条**（原地刷新；终端不支持时自动降级成每 30s 一条日志快照）。结束时最后一张表就是交付物。

---

## 1. 前置条件（跑之前确认三件事）

| 检查 | 命令 | 合格标准 |
|---|---|---|
| **8 张卡空着** | `npu-smi info` 看 `NPU Util(%)` 与 `HBM-Usage` | 全 0% 且显存只有基线占用；有人共用时先等 |
| 环境已引导 | `source env_pack.sh`（或已 `source env_local.sh`） | 不报错；`python3.11 -c "import torch_npu"` 能过 |
| 输出盘有空间 | `df -h <输出目录>` | ≥ 8 个点约 10 GB（大 L 单点 trace ~1–2 GB） |

> ⚠️ **根盘常年贴满**是这台机器的老毛病：如果 `/tmp` 满了，进程会假死。`env_pack.sh` 已经把
> `TMPDIR` 引导到共享盘；自己另开 shell 时别把 `TMPDIR` 指回 `/tmp`。

---

## 2. 标准流程：跑官方 hit97 矩阵 8 个点

```bash
bash tools/run_hit.sh
```

- 点位 = 官方 `HIT97_MATRIX.md` cp1 列那 8 个（**L 与 h 各自不同**，默认照抄官方值）：

  | L | 16k | 32k | 64k | 80k | 128k | 256k | 512k | 1m |
  |---|---|---|---|---|---|---|---|---|
  | M（新增 token） | 512 | 1024 | 1920 | 2432 | 3968 | 7808 | 15744 | 31488 |
  | h | .9688 | .9688 | .9707 | .9703 | .9697 | .9702 | .9700 | .9700 |
  | 官方稳态 ms | 131.0 | 137.3 | 192.7 | 238.9 | 370.9 | 746.0 | 1937 | 4330 |

- 每个点跑两轮：**prof 轮**（出 trace）+ **无 prof 轮**（干净单 pass 稳态耗时，用来对官方 ms）。
- 耗时：16k 约 2–4 min/点，256k 以上明显更久；**8 点全量约 30–60 min**。

---

## 3. 常用变体（挑一行抄）

| 想干什么 | 命令 |
|---|---|
| 只跑某几个序列 | `bash tools/run_hit.sh -L 64k 128k 256k` |
| 用统一命中率跑 | `bash tools/run_hit.sh -L 32k 64k 128k --hit 0.97`（也认 `97%` / `97`） |
| 逐点给命中率 | `bash tools/run_hit.sh -L 16k@0.99 64k@0.97` |
| 直接给 M（新增 token 数） | `bash tools/run_hit.sh -L 100k:3072` |
| 只要 trace，不要稳态轮 | `bash tools/run_hit.sh -L all --prof-only` |
| 重采（覆盖已有结果） | `bash tools/run_hit.sh -L 64k --force` |
| trace 落到别的盘/目录 | `bash tools/run_hit.sh -L 64k -P /your/big/disk/prof` → `<P>/L65536/prefill` |
| 每点一个独立目录 | `bash tools/run_hit.sh -L 16k 64k -P '/your/disk/prof_{L}'` → `/your/disk/prof_16384` |
| 换整个输出根（日志+汇总） | `bash tools/run_hit.sh -o /your/disk/exp1` |
| 先看再跑 | `bash tools/run_hit.sh -L 64k -P /data/prof --dry-run`（逐点打印 h/R/trace 落点，不碰卡） |
| 关进度条 | `bash tools/run_hit.sh -L 64k --no-bar` |
| 快照更密 | `bash tools/run_hit.sh -L 64k --bar-every 10` |
| 验 ETA 模型（不碰卡） | `bash tools/run_hit.sh -L all --eta-from <某个 rounds.tsv>` |
| 忘了参数怎么写 | `bash tools/run_hit.sh --help` / `bash tools/run_hit.sh --list` |

选点规则：`-L` 后面可以跟任意多个（空格或逗号分隔），顺序照给；**不给 `-L` 就是官方 8 点**。
矩阵外的 L（比如 `100k`）必须显式给命中率（`100k@0.97`）或 M（`100k:3072`），否则直接报错并提示。

---

## 4. 交付物：跑完你手上有什么

```
<输出根>/                                  默认 /mnt/xds/sfs/l00972766/hit97_8points（本机）
├── SUMMARY.md                     ← 先看这个：每点的 trace 份数、稳态 ms、口径自检
├── driver_<时间戳>.log            ← 全过程日志（driver_latest.log 是它的软链）
└── L<L>/
    ├── prefill/                   ← prof 轮产出：**8 份 MindStudio trace**（MindStudio 直接打开）
    ├── launch.log / launch_noprof.log        ← 两轮的完整控制台输出
    ├── logs/  、logs2/            ← rankN.log（每 rank 一份）+ all.log（8 rank 合并，带 [rN] 前缀）
    │                                + ascend_log/（CANN plog，排查设备报错用）
    └── PARAMS                     ← 本点口径指纹（L/M/h/replay_tail/iters）
```

**验收标准（每点）**

1. `SUMMARY.md` 里 `trace份 = 8`；
2. `[prof] rc=0` 且两个 rc 都是 0；
3. 口径自检三行都在 rank0.log 里（脚本会自动打印）：
   - `[elb/balroute] 强均路由 ON…`  → 路由口径对了（= 官方 `force_eplb`）
   - `[起跑线] 跨 rank barrier ×1 完成 → 现在开 profiler 开采` → 只在最开头同步一次
   - `[engram] 侧流已启用…` → engram 走侧流（`EngramFetch/Wait` 能与主干重叠）
4. `无prof稳态(均,除首轮)` 与官方同点 ms 同量级（mini vs 官方 ≈ 1.1–1.2×）。

**跑的时候怎么看进度**（终端的进度条 / 日志里的快照，二者同源）：
```
S2 前向 pass2/4 层18/40 │ ███████░░░░░ 62% │ 点3/8 L=65536 prof轮 │ 已用1m42s·再~52s（估） │ 总已用12m30s 剩余粗估~9m40s
```
- 左边是**当前轮**（pass 序号 + 层号），右边 `总已用/剩余` 是整个 run 的估计（括号里写明依据：按已完成轮次的中位数）；
- 终端宽度不够时**按优先级丢尾**：层号 > 百分比 > 哪个点 > 用时 > 明细；
- 日志里每 ~30s 一条 `[bar HH:MM:SS] …` 快照（**纯文本，没有 `\r`**），直接 `grep '^\[bar' driver_*.log` 就能复盘时间轴；
- 每点结束还会打一块 **总进度表**（每轮耗时 + 状态），SUMMARY.md 里也多了 `prof用时 / noprof用时` 两列。
- ⚠️ 全 run **第一轮含冷编译**，实测比后续慢 ~3×（例如 16k 的 prof 轮 246s vs 之后 87s），ETA 里按中位数剔除了它。

---

## 5. 这些参数什么意思（口径）

| 参数 | 含义 |
|---|---|
| `-L` | 序列长度（前缀总长）。写 `16k` / `1m` / `16384` 都认 |
| `--hit` / `@h` / `:M` | 命中率 h：只重算尾部 `R = M + KEEP` 个 token（`M = round((1-h)·L)` 或直接 non_cached），前缀走缓存 |
| `--replay-tail` | 默认 KEEP=140（DeepInfer V4.1）；R = M + 140 |
| `-o` / `-P` | 输出根 / trace 目录（见 §3） |
| `--iters` | trace 里跑几个 pass（默认 4，与官方该点 trace 口径一致） |
| `--console` | 控制台回显：`all`（默认，8 rank 都回显）/ `rank0` / `off`（只落盘） |
| `--cards` | 用哪 8 张卡（默认 `0,1,2,3,4,5,6,7`） |
| `--timeout` | 单轮超时（秒，默认 3600） |
| `--bar` / `--no-bar` | 进度条开关（默认开）。标签如 `点3/8 L=65536 prof轮`，尾注是全 run 级的 `总已用/剩余粗估` |
| `--bar-every` | 日志快照间隔（秒，默认 30）；`--bar-fd` 是真终端 fd（驱动自动 `exec 9>&1` 递进来） |

固定口径（脚本会先清掉这些开关的残留环境变量，避免"我明明设过"）：**dp8 / ep8 / cp1**、
**elb 强均路由**、**合成假权重**（`weights.mode=synthetic`）、**fake 前缀**（`hit.source=fake`，前缀用零值池）、
**跨 rank barrier 只在起跑线一次**、**engram 侧流**。

---

## 6. 中断 / 重跑：不会白跑

- **跑满了 8 份 trace 且口径一致** → 下次自动 `[skip]`（秒退）。
- **口径变了**（改了 `--hit` / `--replay-tail` / `--iters`）→ 不会误判成"采过"，会把旧目录挪成
  `prefill.bak_<时分秒>` 再重采（profiler 只会往目录里追加，混了就废）。
- **半途被杀**（没采满 8 份，或指纹已被摘掉）→ 下次同样挪旧目录重采。
- 想强制重采：`--force`。

## 7. 出问题怎么查

| 现象 | 先看哪 | 常见原因 / 处置 |
|---|---|---|
| 卡在 `set_device` 不动 / 507033 / `halAsyncDmaJettyCreate failed` | `npu-smi info` | **卡被别的容器占用**：等它跑完；起 run 前必须逐卡空气检查 |
| 某个 rank 静默 >300s | 控制台会打 `⏳ rankN 已静默 Ns` | 看该 rank 的 `logs/rankN.log` 尾部；设备报错再去 `logs/ascend_log/` 找 plog |
| `[prof] rc!=0` 或 `trace 份数 < 8` | `L<L>/launch.log` + `logs/rankN.log` | 该点会记进 `SUMMARY.md` 的失败清单；修好后单点重跑 `-L <L> --force` |
| 进程假死、日志不滚 | `df -h /` 与 `/tmp` | 根盘写满 → 清 /tmp 或把 `TMPDIR` 指到共享盘后重跑 |
| mega_moe 打了 `ascendc 不可用 → 回退 deepgemm` | `logs/rank0.log` | 不该出现：出现说明 HCCL 建组顺序被别的资源破坏了（例如把厂商 ElasticBuffer 建在了 mega_moe 之前），报给维护者 |
| 想确认 trace 里有什么 | `L<L>/prefill/*/ASCEND_PROFILER_OUTPUT/op_statistic.csv` | 找 `EngramFetch` / `EngramFetchWait` / `MegaMoe` 等；engram 两个算子应在**另一条 stream** 上（侧流；本次实测侧流 stream 58、主干 61，编号每次跑可能不同，看 `Stream ID` 列有两条即可） |

## 8. 谁负责什么

- **执行者**：按本文档跑、确认 §4 的验收四条、把 `SUMMARY.md` + 失败点日志回给维护者。
- **维护者**：`tools/run_hit.sh` 本身（口径变更、加开关）；对拍工具 `tools/op_time_table.py`
  （mini trace × 官方 `op_summary.csv` 的逐算子次数/耗时对齐表）。
- 官方基准（对拍用，不用重跑 light）：`ds41_light/res/20261010/hit97_prof_unified/L*/op_summary.csv`。

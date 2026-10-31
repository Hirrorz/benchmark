# ds41_mini —— DS4.1-Flash 最小算子图 benchmark（可移植版）

**一句话定位**：一个**自研的最小 benchmark** —— 不依赖官方模型代码（`adaptlib` / `executor/` /
`models/` 一律不 import），只靠 `configs/*.yaml` + `recipes.yaml` + `opgraph.py` 三件套自己"装配"，
跑一遍 prefill 就产出与官方 `ds41_light` bench **同构的流水图**（msprof trace，MindStudio 直开），
并且能在**任何一台装了 Ascend 卡的机器**上跑起来（本项目自带轻量运行时依赖，不依赖共享盘）。

- 只关心三件事：**权重怎么放**（只由并行策略决定）、**用哪些算子**、**算子怎么排**。
- 图层面验收：同 config/同 seq 下与官方 trace 的算子**集合+次数一致、形状一致、时长 ±10%**。
- **数值精度**：已与官方 8 卡 EP8 真权重做到 **40 层逐位一致**（见 `PARITY_STATUS.md`；用真权重模式复现）。
- 不做：命中率曲线扫描 / 调度策略 / 交互式 TUI。
- **常驻 benchmark（2026-10-19）**：8 rank 常驻 + 手动预热 + 简单进度条；预热后每次 prompt
  prefill 目标 ~1 s（L ≤ 预热阶梯时硬上限 3 s），decode/P→D/D→P 分开计时。入口/命令见
  `docs/BENCH_RESIDENT.md`，命令由 `tools/bench_command_builder.html` 生成。

## 0. 目录分类（先看这张表）

| 路径 | 放什么 | 迁移/打包 |
|---|---|---|
| `*.py`（根目录） | 引擎核心模块（runner/exec_ops/exec_moe/…），靠同目录 import | **不要移动/改名**；打包必带 |
| `configs/` | 运行配置 yaml | 必带 |
| `tools/` | 入口脚本（run_hit/run_real/launch_mini）、`selfcheck.py`、命令构建器 HTML | 必带；测试/生成器可不带 |
| `docs/` | 设计与使用文档 | 可选 |
| `parity/` | ids / 官方参考（ref 很大） | 只带 `ids_16k.pt` |
| `runtime/` | 便携包运行时（仅便携包里有）：厂商算子、torch、tilelang、deps、CANN overlay | `make_portable_kit.sh` 生成 |
| `logs/` `_out/` `PROF_*` `extra-info/` `dist/` | 运行产物 / dump / 旧构建 | **不进包**，可清理 |
| `cache_engram_*.pt` | engram 哈希/布局缓存（小） | 保留 |
| `PORTABLE.md` / `PORTABLE_TROUBLESHOOTING.md` | 便携包使用 / 缺件排查 | 必带 |

> 迁移到没有共享盘的机器：见 `PORTABLE.md`；缺 CANN/torch/厂商算子（含 MegaMoE）
> 的安装与排查见 `PORTABLE_TROUBLESHOOTING.md`。

## 0. 上手三步（新机器）

```bash
tar -xzf ds41_mini_pack_<日期>.tar.gz && cd <解出来的目录>
source env_pack.sh                    # ① 引导环境（探测本机 CANN/算子包/依赖，写 env_local.sh）
python3.11 tools/selfcheck.py --card 5   # ② 验收：算子自检（把 5 换成你的空卡号）
python3.11 run.py configs/mini_single.yaml --dry-run   # ③ 不碰卡，出算子图
```

第一次跑到 `selfcheck` 全绿（`✅ 算子自检通过`）就说明这台机器能跑这个 benchmark 了。
之后每个新 shell 只需要 `source env_local.sh`（env_pack.sh 生成的，不用重复探测）。

## 1. 最小条件（能跑起来需要什么）

| 需要 | 说明 | 从哪来 |
|---|---|---|
| **1 张 Ascend NPU + 驱动** | 本机验证机型 Ascend950DT；多卡可跑 EP8（8 张） | 机器/容器 |
| **CANN** | **版本不限**——本工具不校验版本，只要求算子能跑通；`env_pack.sh` 自动在 `$DS41_CANN` → 覆盖层镜像 CANN → `/usr/local/Ascend/ascend-toolkit/latest` → `/usr/local/Ascend/cann-*` 里找第一套能用的 | 镜像自带 或 覆盖层 |
| **python3.11** | 这套 torch 只有 cp311 包（容器默认 python3=3.12 装不了） | 镜像 |
| **厂商算子包** | `compressor / matmul / megamoe / vector` 四个 vendor（`/usr/local/Ascend/custom/*/vendors/*`） | 镜像自带（`env_pack.sh` 自动扫目录拼接；也可回退共享盘覆盖层） |
| **包内 `runtime/`** | 厂商算子的 python 包 + `custom_ops` .so + py311 纯 py 依赖 + tilelang/TileKernels + 预编译 torch 扩展（≈380 MB，**打在本包里**） | 本包 |
| 可选 | 真权重目录（476 GB，跑 `--parity` 或 `weights.mode=real` 才要）；官方参考 dump（对拍用） | 共享盘 |

**不需要**：官方仓库代码（mini 不 import 它）、共享盘（用包内 runtime 时）、PyPI 网络。

## 2. 环境引导 `env_pack.sh`

```bash
source env_pack.sh          # 必须 source（环境才留在当前 shell）；末尾会生成 env_local.sh
source env_local.sh         # 之后每个新 shell 用这条即可
```

它做的事：组件按 ① 显式环境变量 → ② 包内 `runtime/` → ③ 共享盘老路径 的优先序探测；
CANN 用「有 `aarch64-linux/` 或 `libascendcl.so`」判可用、**找到哪套用哪套**；厂商算子包按目录扫描
自动拼 `ASCEND_CUSTOM_OPP_PATH`/`LD_LIBRARY_PATH`；`PYTHONPATH` 顺序照抄验证过的老配方
（deps/tilelang/厂商包在 shim **之前**——顺序反了 `ds41` 命名空间会注册 0 个算子）；
`unset TMPDIR`（SFS 上的 TMPDIR 会让 CANN JIT 的 rmtree 失败）；缺的补件（预编译 torch 扩展、
python3.11 头文件）缺了才补。

覆盖开关：`DS41_PACK_RUNTIME` / `DS41_CANN` / `DS41_DEPS` / `TILELANG_RT` / `DS41_PYTHON` / `DS41_SHARE`。

## 3. 算子自检 `tools/selfcheck.py`（验收门）

口径：**不校验 CANN 版本**——只要这些算子在目标机上能跑通就算过。

```bash
python3.11 tools/selfcheck.py                 # Tier0 环境 + Tier1 逐算子（0..24 层，T=1024）
python3.11 tools/selfcheck.py --card 5        # 指定物理卡（内部转 ASCEND_RT_VISIBLE_DEVICES=5 + npu:0）
python3.11 tools/selfcheck.py --tier0-only    # 只查环境（不碰卡）
python3.11 tools/selfcheck.py --full          # 再跑满 40 层
python3.11 tools/selfcheck.py --json out.json # 机读结果
```

- **Tier 0（不碰卡）**：python/torch/torch_npu/yaml、CANN 路径与版本（**只打印不判定**）、厂商命名空间
  import、**23 个算子入口存在性**（ds41 组 9 个 + custom 2 + cann_ops_transformer 2 + torch_npu 5 +
  tile_kernels 5）、NPU 可见性。
- **Tier 1（需 1 张卡）**：用合成权重让 mini 真跑 0..24 层——这一段层角色覆盖了**全部**算子种类
  （普通层 / engram 层 1 / kv+idx 源层 2 / 候选源层 20 / QSLI 消费层 24）。自检在 `exec_ops.run_op`
  上挂钩子，**逐算子**统计调用次数/失败/跳过/耗时，并检查输出健康（NaN/Inf/爆量，看前 64 行）。
  跑完出「算子 × 状态」表；**没跑到的算子按 `opgraph.py` 注册表逐条列出**（不假装覆盖）。
- `--parity <官方collect目录>`：可选，跑真权重 + 与参考逐层对拍（要真权重目录 + 参考 dump）。
- 退出码：0 全过 / 1 有算子失败 / 2 Tier 0 就挂了（便于脚本化）。
- 默认开**阶段进度条**（`[02:50:19] 50.3% │ S2 逐层：前向执行 层 1/24 win · 已用 39s · 预计再 0.6m`）；
  `--quiet` 关掉。
- Tier 1 默认**不喂** input_ids（合成权重下 engram 走零占位路径；喂真 ids 会因合成表被缩小而"行号越界"，
  那是本就不成立的组合）。真链路验证走 `--parity`。
- 表里 ⚪（未覆盖）的行都写了原因：未融合 MoE 链在本配置被 `mega_moe` 取代、`gate_topk` 被 tilelang
  `topk_gate` 取代、`dyn_mx_quant` 在 QMM 内部产生、`gather/scatter_tail` 属命中路径、`transpose_qmm` 并入 `wo_a`。

**本机实测（2026-10-15，单卡合成权重，0..24 层）**：`✅ 28 个算子真跑过且输出健康；⚠️/❌ 0；
⚪ 13（均有明确原因）；完成 92.7s / 595 次算子调用（27 种）/ 跳过 0`。这一轮还顺带抓到并修掉了
合成模式跑不了第 2 层的真 bug（`SyntheticSource` 缺 `resident`）。

输出长这样（本机实测，Tier 0 部分）：

```
== Tier 0 · 环境（不碰卡）========================================
  ✅ python ≥3.11（torch 只有 cp311 包）                 3.11.6
  ✅ import torch                                   2.13.0+cpu
  ✅ import torch_npu                               2.13.0+gitfcc7937
  ✅ import yaml                                    6.0.3
  ✅ CANN 路径（版本不校验）    /mnt/.../ascend_full/usr/local/Ascend/cann-9.2.0 Version=9.2.0
  ✅ import 厂商算子包 (custom_ops/cann_ops_transformer)
  ✅ import tile_kernels (moe/mhc/engram)
  ✅ 算子入口存在性（23/23）                             全在
  ✅ NPU 可见性                                      device_count=8

  Tier 0：✅ PASS
```

Tier 1 的表（`算子 | 调用 | 失败 | 跳过 | 累计耗时 | 状态 | 层 | 备注`）里每个 ✅ 都代表
**这个算子的真调用点**在这台机器上成功跑过且输出健康。

## 4. 跑一跑

```bash
# ① 算子图（毫秒级，不碰卡）
python3.11 run.py configs/mini_single.yaml --dry-run
python3.11 run.py configs/mini_single.yaml --dry-run --dump-json /tmp/dry.json   # 给 align_ops 用

# ② 单卡真跑（合成权重；--layers 冒烟；--prof 出 msprof trace）
python3.11 runner.py configs/mini_single.yaml --seq 1024 --device npu:0 --layers 4
python3.11 runner.py configs/mini_single.yaml --seq 1024 --device npu:0 --prof --out ./prof/m2

# ③ 单卡真权重（需要权重目录；--weights-path 可覆盖 config 里的路径）
python3.11 runner.py configs/mini_real.yaml --seq 1024 --device npu:0 \
    --weights-path /path/to/DeepSeek-V4.1-Flash-... --layers 2

# ④ 多卡 EP8（launch_mini 给每个 rank 只暴露一张物理卡，逻辑恒为 npu:0）
python3.11 launch_mini.py --config configs/mini_ds41_flash.yaml --world 8 \
    --cards 0,1,2,3,4,5,6,7 --progress off
```

- `--ids-file <a.pt|a.json>` 传入 input_ids（仓库不自带；给了就走 embed→40 层→head。
  不给又不给 `--prompt` 时用随机激活，只适合看 trace/算子计数）。
- 多卡机先 `npu-smi info` 确认卡空；**只用自己的卡**，杀进程按 PID 杀。

### 4.1 性能采集（EP8 + msprof + 多轮）

```bash
# 8 卡 EP8，采集 3 轮（= 3×40 层 = 120 个 profile step，MindStudio 里按 iteration 逐段看）
python3.11 launch_mini.py --config configs/mini_ds41_flash.yaml --world 8 \
    --cards 0,1,2,3,4,5,6,7 --logdir <日志目录> --timeout 1800 \
    --seq 1024 --iters 3 --prof --out <trace目录>
```

- `--iters N`（也可写进 config 的 `run.iters`）：一个 trace 里跑 N 轮前向。profiler 的 step 边界
  = **每层一次**，所以 step 数 = N×层数；第 N 轮的耗时**包含 trace 导出**（3~4 GB 要几十秒），
  看稳态耗时请用中间几轮。
- 合成权重默认 `share_across_layers: true`（40 层共用同一份假权重、换层不释放）——
  合成权重没有"层"的含义，共用后每轮前向的耗时才是干净的（实测 3 层：1.50s → 0.23s/轮）。
- 每个 rank 自己一份 trace 目录（`<out>/<host>_<pid>_<ts>_ascend_pt/`），用 MindStudio 打开该目录；
  `ASCEND_PROFILER_OUTPUT/step_trace_time.csv` 每行一个 step（Computing / Communication / Free），
  `kernel_details.csv` 的 `Step Id` 列就是 0..N×层数-1。
- 实测（2026-10-15，T=1024，EP8 合成权重）：iter1 31.4s（含预热）/**iter2 0.713s = 17.8 ms/层**/
  iter3 65.1s（含导出）；8 rank 全部 exit 0、0 报错。
- `mega_moe` 后端：`auto` 会先试厂商 ascendc 对称缓冲，**缺字段（调用路径未接）就自动回退 deepgemm**；
  想看 ascenc 路径报什么用 `DS41M_MOEMODE=ascendc` 强制（会抛，别在正式采集里开）。

### 4.2 命中（prefix cache hit）—— 和官方 hit97 点同口径

口径来自官方 `adaptlib/hit.py`（产品侧"命中请求复用前缀的压缩 KV / 索引 KV / 压缩器 state，
只重算 `R=(1-h)L+replay` 个 token，但 KV 跨度仍是 L"）。

```yaml
hit:
  ratio: 0.9707      # → T_main = (1-h)·L + replay_tail（L=64k 时 T_main=2048，与官方该点一致）
  source: fake       # fake = 前缀用池子初值（零值）| real = 先真算一遍前缀（stage A）
  replay_tail: 128
```

- **执行侧（两种 source 共用）**：只算尾部 `R` 个 token；**位置是绝对位置** `H-replay..L-1`；
  池子/块表/槽位按 **L** 开；索引器与压缩注意力的可见长度 `seqused_k = L//ratio`；
  窗口分支**不回看前缀**（官方 hit stage B：窗口跑在本步 scratch、索引重定基、`seqused_ori_kv=R`）。
- **`source: fake`（合成权重用这个）**：前缀不真算，池子初值（零）就是"缓存内容"，
  并且**所有层共用这一份前缀池** —— 压缩 KV 池按 ratio 共享、索引器缓存（官方 `kv_cache.shared_idx`）
  共享、窗口池是每层自己重写的 scratch。→ 形状/算子次数/耗时与官方 stage B 同口径，
  **数值不是"真命中"**（前缀读出来是零）。
- **`source: real`（真权重/数值口径）**：先跑 **stage A** 把前缀 `[0,H)` 真算一遍，
  KV / 压缩 KV / 索引器缓存 / 压缩器 state 全部落进同一个 `shared` 池（跨 stage 存活），
  再跑被测量的 stage B 复用它们；stage A 不进 profile。要求 `H % ratio == 0` 且
  `off % ratio == 0`（官方命中点在 ratio∈{1,2} 下天然满足），否则前缀末组的进位 state
  无法精确复原（会打印一次提示）。
  replay 段的行会被重算并重写（值逐位相同：同权重同输入，且组边界对齐）——
  官方是"抑制 replay 行的写入"，两者等价。
- `run.batch`（DP 维度）**仍未实现**：mini 每 rank 一条序列，表示不了官方 DP8 的 per-rank 不同请求。

### 4.3 采集入口 `tools/run_hit.sh`（一次启动跑多个序列）

dp8/ep8/cp1 + elb 强均路由 + 合成假权重 + fake 前缀 + 起跑线 barrier×1 + engram 侧流，
每点跑 `--prof` 轮（trace，应 8 份/点）与无 prof 稳态轮：

```bash
bash tools/run_hit.sh -L 16k 32k 64k 128k 256k   # 一次跑这 5 个点（顺序照给）
bash tools/run_hit.sh                            # 不给 -L = 官方 8 点全跑
bash tools/run_hit.sh -L 64k --hit 0.99          # 命中率传参：M = round((1-h)·L)
bash tools/run_hit.sh -L 64k@0.99 128k:3968      # 逐点给 h / 直接给 M
bash tools/run_hit.sh -o /data/exp1              # 换输出根（日志/汇总/默认 trace）
bash tools/run_hit.sh -L 64k -P /data/traces     # 只换 trace 目录 → /data/traces/L65536/prefill
bash tools/run_hit.sh -L 64k -P '/data/prof_{L}' # {L} 占位 → /data/prof_65536
bash tools/run_hit.sh --list / --dry-run / --force / --prof-only
```
**要交给别人跑**：完整方法文档见 [`docs/COLLECT_HIT.md`](docs/COLLECT_HIT.md)（前置检查 → 命令 → 验收 → 排查）。

`--dry-run` 会逐点打印 `h / R / trace 落点`，不碰卡；trace 目录规则：
`-P` 未给 = `<out>/L<L>/prefill`，给了 = `<P>/L<L>/prefill`，`P` 里含 `{L}` 就直接替换、不再追加。
- **进度条**（`progress_bar.py`）：终端一行原地刷新（`层/pass · 百分比 · 哪个点 · 用时/ETA`），
  同时每 30s 往日志追加一条纯文本快照；终端那一路走驱动 `exec 9>&1` 递进来的 fd，**不进 tee**，
  所以日志里永远没有 `\r`。`--no-bar` 可关，`--bar-every` 调快照密度。
- 日志**全部上屏且同时落盘**：`$OUT/driver_<时间戳>.log`（+`driver_latest.log`）、每点
  `launch*.log`、`logs{,2}/rankN.log`、`logs{,2}/all.log`（8 rank 合并，带 `[rN]` 前缀）、`SUMMARY.md`。
- 每个点位目录带**口径指纹** `PARAMS`（L/M/h/replay_tail/iters）：参数变了不会误判成"已采过"，
  会把旧 trace 挪成 `prefill.bak_*` 再采；某 rank 静默 >300s 会打告警。

### 4.4 采集入口 `tools/run_real.sh`（**真权重 + EP8，带精度**）

与 4.3 **并列**的另一个入口（那份合成、这份真权重，互不影响）：真权重 `DeepSeek-V4.1-Flash-cann-mxfp8-mxfp4`
+ 真 ids（**自然路由 = 不做负载均衡**）+ `engram_tp_size=8`（真表按卡分片，每层一个专用 HCCL 组）
+ 精度 gate（末位 token 对齐官方 `id=396 ' that'`）：

```bash
bash tools/run_real.sh -L 16k                       # prefill + hit 两个口径各一轮（prof + noprof）
bash tools/run_real.sh -L 16k --mode prefill        # 只普通 prefill（"能吐字"那条）
bash tools/run_real.sh -L 16k --mode hit --hit .96875
bash tools/run_real.sh -L 16k --parity              # + 逐层对拍轮（rank0，对官方 parity/ref → ✅40）
bash tools/run_real.sh -L 16k --iters 3 --warmup-iters 1
bash tools/run_real.sh -L 16k --layers 15 --no-precision   # 冒烟（★ 必须 ≥15：engram 层是 1 和 14）
bash tools/run_real.sh -L 16k --engram-host         # 逃生：退回主机取表（丢 EngramFetch/Wait 两个算子）
bash tools/run_real.sh -L 16k --no-moe-chunk        # 关掉官方同款 MoE 分块（默认开，见下）
bash tools/run_real.sh -L 16k --no-alop-config      # 用 ALOP 内置默认（纯 C 级，只有 PROF_* 目录）
```

**采集引擎 = ALOP**（2026-10-16 口径：以后都用 ALOP）。`--prof` 即 ALOP；默认 `configs/alop_mindstudio.yaml`
（`python_profiling_config.enabled: true`）→ 出与合成口径同一套 MindStudio trace，trace 次数断言照常生效。

**MoE 分块默认开**（`moe_chunk_max_len: 4096`，官方 `get_moe_num_chunks` 同口径）：厂商融合算子的内部
workspace 按每 call 的 token 数走（T=16384 单次 ≈20 GiB 瞬时），40 层常驻权重下 16k 会顶穿 91.8 GiB →
按官方口径切成 4 块（每层发 4 次 MegaMoe，数值逐 token 等价）。hit 口径每轮 ≈640 token < 4096 → 仍是 1 次。

**要交给别人跑**：完整方法文档见 [`docs/COLLECT_REAL.md`](docs/COLLECT_REAL.md)。
跑完自动打四条验收：无 `[elb/balroute]` 行（自然路由）、`厂商 ElasticBuffer 就绪` 恰好 = engram 层数
（每层一个专用 HCCL 组，窗口按通信域池化）、trace 次数（`EngramFetch = engram层数×iters×卡数`、
`MegaMoe = 层数×iters×卡数×块数`）、精度 gate（`NEXT_TOKEN mini=396 ref=396 match=YES`）。

## 5. 对拍（可选，真权重）

```bash
# 逐层 hidden 对拍（mini vs 官方；两边都是 collect_rank{R}.pt）
python3.11 tools/compare_layers.py --ref <官方collect目录> --ours <mini collect目录> [--rank 0] [--logits]
# 模型直方图 × 官方 trace
python3.11 tools/align_ops.py --ours /tmp/dry.json --ref <trace>/ASCEND_PROFILER_OUTPUT/kernel_details.csv --ref-passes 4
# 两个 trace 差在哪
python3.11 tools/diff_traces.py <traceA> <traceB> [--passes-b 4]
```

当前精度状态、复现命令、根因分析见 **`PARITY_STATUS.md`**（40 层逐位一致；含官方侧 QSLI 读未初始化
显存的发现与 `DS41M_INJECT_QSLI` 复现开关）。

## 6. 目录地图

| 文件 | 职责 |
|---|---|
| `README.md` | 本文件：上手 / 最小条件 / 自检 / 命令 / 目录 / 坑 |
| `PARITY_STATUS.md` | 精度对拍现状（当前文档：40 层逐位一致 + 根因） |
| `env_pack.sh` | 环境引导（探测 + 拼环境 + 生成 `env_local.sh`） |
| `configs/mini_single.yaml` | 单卡合成权重（日常冒烟 / 自检用） |
| `configs/mini_ds41_flash.yaml` | 目标配置（EP8，与官方正式对照） |
| `configs/mini_real.yaml` / `mini_real_ascendc.yaml` | 真权重配置（`weights.mode=real`） |
| `configs/mini_indexer.yaml` / `mini_moe.yaml` | 索引器 / MegaMoE 单算子联调配置 |
| `recipes.yaml` | **每种层的算子配方**（存在性 + 顺序）+ `extras` 会话级残差 |
| `opgraph.py` | **算子注册表**：kernel 名 / 形状 / count / affects（唯一算子定义处） |
| `modules/` | 模块接口/装配/运行时/查看器（模块规格在 `modules/specs/`；Phase 2 默认开） |
| `assemble.py` | 按 config 堆 40 层；层角色（kv_src/idx_src/候选/engram）与每层 T |
| `run.py` | dry-run 驱动（不碰卡） |
| `runner.py` | 真跑驱动（NPU + msprof trace + 逐层 collect / dump） |
| `launch_mini.py` | 多 rank 启动器（每 rank 只见一张物理卡） |
| `exec_ops.py` | 算子执行表：op key → 真实 vendor/torch_npu 调用（含 SKIP 名单） |
| `exec_indexer.py` | 索引器插件（idx_prolog / QLI / QSLI + 缓存池几何） |
| `exec_moe.py` | MegaMoE 插件（ascendc/deepgemm 后端 + HCCL 组） |
| `exec_engram.py` + `engram_layout.py` | engram 插件（hash/取表/门控 + 官网哈希布局） |
| `weights.py` / `weights_manifest.py` | 权重来源（合成/真权重共用一套切分）+ 每层权重清单 |
| `progress.py` | 阶段进度（S0~S3 标定表 + ETA） |
| `tools/selfcheck.py` | **算子自检（验收门）** |
| `tools/run_hit.sh` / `tools/run_real.sh` | 两个采集入口（合成命中矩阵 / 真权重 EP8 带精度） |
| `tools/check_next_token.py` | 精度 gate（mini 末位 token × 官方参考 dump） |
| `tools/compare_layers.py` | 逐层精度对拍器 |
| `tools/align_ops.py` / `diff_traces.py` | 直方图 × trace / trace × trace |
| `tools/build_pack.sh` | 打包（代码 + 轻量 runtime → tar.gz + sha256） |
| `tools/probe_*.py` | 单算子最小调用探针（mqsmla / megamoe / indexer / vendor 命名空间） |
| `tools/gen_manifest.py` / `shape_seq.py` / `trace_coverage.py` | 清单生成 / 校准取证 |
| `docs/DESIGN.md` | **设计文档**：目标/架构/核心子系统/决策/权衡/演进 |
| `docs/CODE_ANALYSIS_AND_USAGE.md` | 代码级分析与使用手册 |
| `docs/` | **采集方法文档**（`COLLECT_HIT.md` 合成 / `COLLECT_REAL.md` 真权重）+ 历史设计（PLAN、M2 报告、M1 校准、权重清单、早期交接） |

## 7. 常见坑（都踩过）

1. **`TMPDIR` 指共享盘会让 CANN JIT 卡死**（rmtree 报 `Directory not empty`）→ `env_pack.sh` 已 `unset TMPDIR`。
2. **逻辑卡号 ≥1 会 aicore fault**（`hc_pre` 等）→ 永远用「`ASCEND_RT_VISIBLE_DEVICES=<物理卡>` + 逻辑 `npu:0`」；
   `selfcheck.py --card N` / `launch_mini.py --cards` 都按这个来。别写 `--device npu:N`（N≥1）。
3. **共享卡机器**：跑前 `npu-smi info`；兄弟容器占着卡时别抢；杀进程只按 **PID**；设备幽灵内存要等回落。
4. `ds41` 命名空间注册数变 0 = `cann_ops_transformer_experimental` 的 stub shim 抢在了真包前面
   （`PYTHONPATH` 顺序）→ 用 `env_pack.sh` 的固定顺序，别自己拼。
5. 索引器算子（QLI/QSLI）必须显式传 `candidate_topk_blocks`/`candidate_block_size`，否则**静默返回空候选**。
6. mega_moe 要建 HCCL 组（端口 29517-29596 自动挑，冲突时用 `DS41M_MOE_PORT` 指定）。
7. 真权重模式下 engram 表走 host-resident（层 1/14 各 ~96 GB 主机内存）→ 机器内存要够。

## 8. 打包与分发

```bash
bash tools/build_pack.sh          # → dist/ds41_mini_pack_<日期>.tar.gz + SHA256SUMS
```

包 = 代码（git archive，只有跟踪文件）+ `runtime/`（轻量依赖，≈380 MB 未压缩）。
校验：`sha256sum -c SHA256SUMS`。分发包同时会放一份到共享盘 handoff 目录。

## 9. 已知差（执行侧 vs 模型 / mini vs 官方）

1. `rope` 我们 2/层（模型 3/层）：第 3 次在压缩 KV 上，属 compressor 路径。
2. `tmove`/`mul` 在内核里看不到：`.contiguous()` / `×1.0` 被编译器优化掉；模型按参考 trace 记账。
3. `Cast` trace 比模型多：`logits.float()` 等我们自己加的 dtype 转换也产 Cast。
4. **合成模式的预处理噪声**（trace 里有、官方没有）：`StatelessRandom/Normal`（每层专家权重现生成）、
   `Transpose/TransData`、窗口稀疏索引计算（`Range/FloorDiv/…`）、rope 的 `Cos/Sin/Pow`、池初始化 `Fill/ZerosLike`。
5. **`MhcPost` vs `mhc_post_fwd_asc_kernel`**：同一语义、不同实现内核（我们走 aclnn）；diff 工具会并列显示。
6. 官方自身有 run-to-run 1e-3 级噪声（chaotic 放大）→ 对拍口径必须是**同层同输入**逐位，见 `PARITY_STATUS.md`。

延伸：设计见 `docs/PLAN.md`；M1 校准取证见 `docs/CALIBRATION.md`；历史状态见 `docs/M2_REPORT.md`、`docs/HANDOFF_MINI.md`。

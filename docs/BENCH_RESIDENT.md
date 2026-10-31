# 常驻 benchmark（无 TUI，简单进度条）

2026-10-19 用户口径：先不做交互式界面，终端里一行文本进度条即可；要执行的命令由
`tools/bench_command_builder.html` 的“常驻 benchmark”面板生成。

目标：8 个 rank 进程常驻占满 8 卡，权重布局/编译/图捕获/池都只做一次；预热后每次
prompt 只跑计算。

- prefill：目标 ~1 s，**L ≤ 预热阶梯最大值时硬上限 3 s**（取 8 rank 的 max）；
- decode：每 token 单独计时（首 token 不算 decode）；
- P→D / D→P：独立阶段、独立进度条；
- L 超过预热包络：显示实测值并标注“未承诺”，不虚报 3 s。

### P0 实测（真权重 EP8，AC 40 层 + release raw，graph tail 开，单跑冷启）

| L | 首 pass（含读权重/融合） | 稳态 pass2 |
|---|---|---|
| 128 | 95.2 s | **0.175 s** |
| 256 | 217.9 s | **0.271 s** |
| 512 | 169.9 s | **0.291 s** |
| 1024 | 290.3 s | **0.348 s** |
| 2048 | 281.7 s | **0.291 s** |
| 4096 | 96.9 s | **0.377 s** |

峰值 allocated ≈61.4 GiB（首 pass raw+AC 共存时；稳态更低）。这说明 512–2048 的
“预热后 ≤3 s”有充足余量；首 pass 的 1.5–5 min 正是常驻服务要摊掉的部分。

## 1. 进程与命令

```
HTML 命令生成器                    终端 A                          终端 B
bench_command_builder.html  →  bench_server.py start  ←→  bench_cli.py run/warmup/...
                                  ├─ 8 × runner.py --serve        └─ 一行进度条 + 结果
                                  ├─ boot / W1 / W2 / W3 进度条
                                  └─ tools/moe_host_cache.py（手动开/停）
```

### 1.1 生成命令（v2 双模式）

打开 `tools/bench_command_builder.html`：

- **简单模式（默认）**：页面已选中“常驻 benchmark / ep8”。在“请求队列”里填 1~N 条
  prompt（可分别设 chat / decode / repeat），然后：
  - 「全部」页一次看三块脚本（启动 / 请求 / 收尾）；
  - 也可以分别复制「启动」到终端 A、「请求」+「收尾」到终端 B；
  - 勾「dry-run」先只校验命令；勾「生成一键脚本」后多出「一键」页（后台起服务 +
    跑队列 + trap 自动收尾）。
- **高级模式**：暴露 `bench_server` / `bench_cli` / `moe_host_cache` / `launch_mini` /
  `run_hit` / `run_real` 的全部安全开关、serve 环境变量、客户端参数、自定义 env/args。
- 顶栏“导入 / 导出”：下载/上传 JSON，或复制带 `#cfg=` 的分享链接（打开即恢复）。
- 旧的单文件构建器备份在 `tools/bench_command_builder.legacy.html`。

**tp8-fidelity 目前标为实验性**：P→D / D→P 机制可用，但全量 TP decode 的数值尚未验证
（2026-10-20：100 token 从第二个 token 起乱码）；正确性优先一律用 ep8。

### 1.2 手动开 host cache（可选，tp8-fidelity 必需）

```bash
python3.11 tools/moe_host_cache.py start --config configs/mini_real.yaml \
  --layers 0-39 --experts 0-383 --progress-bar
python3.11 tools/moe_host_cache.py status
python3.11 tools/moe_host_cache.py stop
```

- 全量 268.9 GiB / 92,160 keys，约 3 min（≈1.5 GiB/s）；`--progress-bar` 在终端画加载条。
- daemon 与 rank 通过 memfd + SCM_RIGHTS 共享页；8 rank 不会各存一份。
- `stop` 后若 rank 仍持有 mmap，页不会释放；先 `bench_cli.py hostcache-detach`（idle 时）。

### 1.3 启动常驻服务（终端 A，前台阻塞）

```bash
python3.11 tools/bench_server.py start \
  --config configs/mini_real.yaml --world 8 --cards 0,1,2,3,4,5,6,7 \
  --logdir /mnt/xds/sfs/l00972766/tmp/bench_resident/resident \
  --job resident --mode auto --host-cache /tmp/ds41_moe_host_cache.sock \
  --serve-l-max 4096
```

- `--mode auto`：一律 `ep8`（全量 decode，已 5 连跑稳定）；`tp8-fidelity` 需显式指定
  （host cache 是它的前提）。
- `--serve-l-max`：常驻池容量（prompt+decode 的最大 token 数）；超长请求会被拒绝。
- `--warmup-on-start 128,256,512`：启动后自动预热；不勾则手动在终端 B 发 warmup。
- Ctrl-C / `bench_cli.py stop` 整组停止（HCCL 不能单 rank 重启）。

### 1.4 预热（终端 B，手动）

```bash
python3.11 tools/bench_cli.py warmup --job resident \
  --lengths 16,32,64,128,256,512,1024 --decode 1
```

- **W1**：干净 prefill（不缓存 AC），raw 暂留；
- **W2**：ep8 建 40 层常驻 AC（逐层建完即释放该层 raw；全量 decode 不再建 NZ），
  跑 1 步 T=1 编译 kernel；tp8-fidelity 跑一次完整 `P→D → 1 步 decode → D→P`；
- **W3**：长度阶梯各跑一次真实 prefill（形状/图捕获），显示每桶耗时。

### 1.5 跑任意 prompt

```bash
python3.11 tools/bench_cli.py run --job resident \
  --prompt "请用一句话介绍杭州，并说明它最有名的景点。" --decode 32 --repeat 3
```

输出：每次 run 的 prefill / P→D / decode ms/token / D→P / 首 token / 文本；
`--repeat N` 的第 2 次起是稳态，摘要取中位。最后附日志目录和 `tail -f` 命令。

### 1.6 其它命令

```bash
python3.11 tools/bench_cli.py status   --job resident
python3.11 tools/bench_cli.py cancel   --job resident   # 安全点取消，服务继续常驻
python3.11 tools/bench_cli.py stop     --job resident
python3.11 tools/bench_cli.py hostcache-apply  --job resident
python3.11 tools/bench_cli.py hostcache-detach --job resident
```

## 2. 进度条

一行文本，阶段 + 完成数 + 百分比 + 耗时 + ETA：

```
[hostcache ] [██████████░░░░] 168.3/268.9 GiB  62%  1.5 GiB/s  ETA 1m07s
[W1 AC     ] [██████░░░░░░░░] 12/40 层  8.4s  ETA 19.5s
[W2 decode ] [██████████████] 40/40 100% done
[W3 shapes ] [███████░░░░░░░] 3/5 桶  L=1024  1.02s
[prefill   ] [██████████████] 40/40 100% done  1.02s
[P→D       ] [███████████░░░] 33/40 层  2.7s  ETA 0.5s
[decode    ] [████████░░░░░░] 12/32 tok  8.3s  690 ms/tok
[D→P       ] [██████████░░░░] 28/40 层  2.2s  ETA 0.9s
```

识别的是 rank 原有的人类可读行（`[W1 12/40]`、`[ 12/40] layer ...`、`[inc 3/32]`），
不侵入 `run_forward` 热路径。`--no-bar` 关闭；非 tty 时每 ≥5 s 打一行快照。

## 3. 布局模式

| 模式 | 请求间常驻 | P→D | decode | D→P | 实测显存（rank0/1 最重） |
|---|---|---|---|---|---|
| `ep8` | EP AC（34.5 GiB）+ pinned host EP NZ（~36 GiB） | 0 | **增量（host NZ）2.15 s/tok**；full（基线）0.26 s/tok | 0 | 稳态 **48.7 GiB**；W2 峰值 **50.5 GiB** |
| `ep8-fidelity` | EP raw（P→D 后 drop）+ device EP NZ（~36 GiB）；AC 不常驻 | **4.8 s**（run；W2 冷 17.3 s） | **增量 EP（dispatch/combine）0.37 s/tok**（8 token，ids 与 host-NZ inc 逐位相同） | **4.6 s**（host cache 回 raw 33.6 GiB/rank） | P→D 55.3 GiB；D→P 54.5 GiB |
| `tp8-fidelity` | EP raw（AC 不常驻，prefill 按层现融合 ~0.2 s/40 层） | all2all，实测 **2.5 s** | **增量 TP（每步 T=1）0.21–0.32 s/tok**（10 token） | host cache 回载 EP raw，实测 **5.5 s** | P→D 峰值 59.8 GiB；D→P 后 44.3 GiB |
| `tp8-dual` | EP AC + TP NZ | 0 | TP | 0 | 不可行（AC 34.5 + TP NZ 44.8 + 其它 >91），未实现 |

- **ep8** 最稳：P→D/D→P 都是 0；prefill 直接命中常驻 AC。
  **decode 默认 `DS41M_SERVE_DECODE=inc`**（2026-10-20 用户口径：不要全量重算）。
  ep8 的 AC 与 device EP NZ 不能共存，NZ 只能放 pinned host，每 token 回拷 ~36 GiB →
  实测 **~2.2 s/token**；显式 `DS41M_SERVE_DECODE=full` 才走全量重算
  （0.26 s/token，数值基线，O(L)）：每一步把 `prompt+已生成` 整段重跑一遍 prefill
  （每步先清压缩器 ring state/池）。
  full 基线实测 **0.26 s/token**（偶数补长，每步单次 prefill），且逐位可复现
  （同 prompt 3/5 连跑 ids 完全一致）；prefill 稳态 **0.16 s**（首触 ~0.34 s）。
  增量 decode 的 2026-10-19 NaN 已在 2026-10-20 定位并修复：`hit_replay_tail` 原来在所有
  `pos_off>0` 的步都把压缩 KV 写成 `slot=-1`，第一组完成时读到 carry/部分行；现在只在
  hit stage B 抑制，且只写完整压缩组（carry 留给 state_cache）。ep8 inc 8 token 文本连贯、
  ids 与 full 同口径；代价是 host NZ 每 token 回拷 ~36 GiB（~2.2 s/tok）。
- **ep8-fidelity（2026-10-20 新增）**：把 P→D 搬到请求边界——prefill 用 EP raw，
  P→D 在设备上建 EP decode NZ（drop raw，不常驻 AC），decode 每 token 不再回拷；
  D→P 从 host cache 回 raw。8 token 真卡：prefill 1.684 s、P→D 4.82 s、
  decode **0.37 s/tok**、D→P 4.57 s，ids 与 ep8 host-NZ inc 逐位相同。
  EP 的 dispatch/combine 仍使其比 tp8 慢 ~1.4×；长 decode 比 host-NZ inc 快 ~5.8×。
- **tp8-fidelity（增量 TP 已验证）** 对应用户描述的 `prefill → P→D → decode → D→P → 下一 prompt`：
  P→D 逐层 all2all 并释放该层 EP raw；D→P 从 host cache 按层批量 H2D 回载本 rank 的
  48 个 EP 专家（33.6 GiB/rank）。**AC 与 TP NZ 不能同时常驻**（实测 AC=34.5 GiB，
  不是早期估的 18 GiB；34.5+44.8+raw/其它必然 >91.75），所以 P→D 前释放 AC，
  D→P 后下一次 prefill 现融合（0.5 s/40 层，仍远低于 3 s）。
  decode 走 **TP 增量（每步 T=1）**：`serve_loop` 现在会把 `DS41M_SERVE_DECODE=inc`
  **写回环境**（以前只放在局部变量，`serve_run_once` 实际按 `full` 跑 T>1 的 TP MoE →
  第 2 个 token 起乱码）；解码步只算 1 个 token，MoE 走 `moe_infer_tp`（TP all_reduce）。
  实测（launch，4 步与读盘 TP 逐位一致；**2026-10-20 常驻真卡复验 10 token**）：
  P→D 2.5 s、**TP 增量 decode 0.212–0.320 s/tok**（10 token，文本连贯）、
  D→P 5.47 s、**D→P 后下一次 prefill 0.86 s（首跑）/ 0.40 s（W3 稳态）**。
  carry 按官方口径只写完整压缩组（`n_valid=(off+T)//ratio-off//ratio`），部分组留给
  state_cache；普通 decode 不再打提示，只有 hit replay（stage B）会打
  `[hit] 压缩起点 …`（有界近似）。TP 与 EP 从第 2 个 token 起可不同
  （all_reduce vs dispatch/combine 的累加顺序），首 token 与 prefill 头一致。
- **decode 图（2026-10-20，`DS41M_DECODE_GRAPH=1`）**：T=1 增量 decode 的层
  **0..39 整段**用 `torch.npu.NPUGraph` 捕获（不是 torch.compile/npugraph_ex，效果等价）。
  依赖 pos 的元数据（窗口/压缩 slot、rope、QLI/QSLI 的 seqused/索引）全部住**固定地址
  缓冲**，replay 前 `dg_update(pos)` 原地刷新；QSLI 池几何按 `L_full` 冻结；engram 层在
  图外预计算 `engram_kv`，图里只跑 wkv/gate。tp8-fidelity 真卡 20 token：
  **稳态 0.024–0.026 s/token**（前缀 5 ms + graph replay 13 ms），同路径 eager
  207.6 ms/token → **8.3×**（早前 tp8 eager 250 ms 口径约 10×）；
  `DS41M_DECODE_GRAPH=check`（只建缓冲、不捕获）20 token
  与图模式**逐 token 相同**。`DS41M_DECODE_GRAPH_CUT=0`（默认 21；0=全 40 层）。
  注意：首两个 forward 必须 eager（尾段常量要在捕获前建好）；engram host gather 偶发
  25–64 ms 尖刺，会抬高 P99。
- **MoE 块长**：常驻池 ≤1024 时默认 `DS41M_MOE_CHUNK=1024`（vendor 对称缓冲
  10.05 GiB → 1.31 GiB）；同 prompt 的 ids 与 8192 逐位一致（已验证）。
  池 >1024 时回退 8192。
- 单卡 HBM 96 GiB，空载基线 4.75 GiB，进程可用 ~91 GiB。实测峰值用
  `torch.npu.max_memory_allocated()`；超 88 GiB 触发降级。

## 4. 预热/请求状态治理

预热后跨请求保留：`_PREP_AC_L`（仅 ep8：raw 已被 NZ 替换，prefill 必须吃常驻 AC）、
`DECODE_CACHE`（ep8 host-NZ 元数据 / tp8 TP NZ）、`_PACK_CACHE_L` / `_PACKED_SUB_CACHE`、
`_GRAPH_TAIL`、池缓冲。

每个请求前清（`reset_request_state(clear_req=True)`）：`_CTX_CACHE`、`_SEQ_CACHE`、
`_cmp_state_L*`、`_cmp_sparse*`、`_li_cand*`、`_cmp_len_in`、`_win_idx`、`_ffn_pre`；
**并且**（2026-10-19 新增，修"第 2 个请求 decode 第 2 步 layer2 NaN"）：
`exec_engram._PREFETCH`、`exec_moe._CACHE["src"/"w"]`、`exec_ops.META`、
`exec_indexer._MD_CACHE`/`_QSLI_TB`（graph tail 开启时不清 META，保持图地址稳定）。
池数据清零，避免短 prompt 读到上一轮的未来行。

**prompt 长度约定**：decode 请求会把 prompt 补到 **≥16**。
- **full decode（默认）：补到偶数**。单次 prefill（不拆分），且偶数 T 只可能是
  0/2 mod 4，天然避开 mqsmla 的 `T%4==1` aicore trap；全量重算每步清 state，
  不再依赖 ring 相位。
- 增量 decode（`DS41M_SERVE_DECODE=inc`，默认）：补到**奇数**（ratio=2 部分组边界）。
- chat prompt：在 **BOS 之后**插中性空格 token（保住聊天模板语义）；
- raw prompt（无 BOS）：在 **末尾**补（前置空格会让模型顺着空格继续输出，实测文本为空）。

## 5. 日志与排障

每次 run 结束输出：

```
日志：<logdir>/rank0.log … <logdir>/rank8.log  +  <logdir>/all.log
tail -f <logdir>/all.log
```

| 现象 | 处理 |
|---|---|
| `PERF_RULES 检查未通过` | 默认开关被降级；恢复默认，或 A/B 时设 `DS41_ALLOW_SUBOPTIMAL="原因"` |
| `tp8-fidelity 需要 host cache` | 先 `moe_host_cache.py start`；或 `--mode ep8` |
| `prompt N + decode M > 常驻池容量` | 加大 `--serve-l-max` 重启；池不中途重分配 |
| `rank 未 ACK` / 结果超时 | HCCL 已失步 → supervisor 整组杀；重启服务（不要单 rank 重启） |
| `load_state_dict` / 权重缺 | 检查 weights 路径与 host cache `--layers/--experts` 范围 |
| 进度条不动但日志在走 | rank 输出的人类行没匹配；看 `rank0.log` 最后 20 行 |
| host cache stop 后内存没降 | rank 仍持有 mmap → `hostcache-detach`（idle）或停服务 |
| 卡被别的容器占用 | 起 run 前 `npu-smi info` 看进程与 HBM；等幽灵内存回落 |

## 6. 自测（无 NPU）

```bash
python3.11 tools/bench_server.py start --fixture --world 8 --cards 0,1,2,3,4,5,6,7 \
  --logdir /tmp/bench_fixture --job fixture --mode ep8 --host-cache off
python3.11 tools/bench_cli.py status --job fixture
python3.11 tools/bench_cli.py warmup --job fixture --lengths 128,256,512
python3.11 tools/bench_cli.py run --job fixture --prompt hi --decode 8 --repeat 3
python3.11 tools/bench_cli.py stop --job fixture
```

`--fixture` 用 `tools/bench_fake_rank.py` 模拟 8 rank 的 ctl 协议/进度行/结果，
用于验证 supervisor 的 ACK、聚合、进度转发和客户端渲染，不占 NPU。

## 7. 已知限制

- **tp8-fidelity 增量已真卡验证（2026-10-20）**：10 token 文本连贯，
  **0.212–0.320 s/tok**；之前的乱码有两个根因：①`serve_loop` 的 `inc` 没写回环境 →
  实际跑 full（T>1 TP MoE 数值错误）；②`hit_replay_tail` 在所有 `pos_off>0` 的步
  抑制压缩 KV 写入（slot=-1）→ 第一组完成时读到 carry/部分行 → gate NaN。
  两处都已修；奇数绝对位置的 carry 仍是近似（会打提示，不崩），尚未做 full 逐位对拍。
- **ep8 增量是默认**（`DS41M_SERVE_DECODE=inc`）：AC 与 device EP NZ 不能共存，
  NZ 只能放 pinned host，每 token 回拷 ~36 GiB → **2.16–2.30 s/token**
  （2026-10-20 真卡 8 token 验证，文本连贯、0 崩）；能跑、但不是低时延路径。
  低时延增量请走 tp8（TP NZ 常驻设备，**0.212–0.320 s/tok**）。
- `tp8-dual` 不实现：AC 34.5 + TP NZ 44.8 + raw/其它放不进 91.75 GiB。
- tp8 的 **D→P 实测 5.5 s**（目标 ≤3.5 s）：瓶颈是 33.6 GiB/rank 从 memfd→pinned→H2D
  的串行拷贝。后续可做双缓冲/多线程或直接 DMA；不影响"下一 prompt prefill ≤3 s"。
- ep8 的 `DS41M_SERVE_DECODE=inc`（增量 decode，NZ 在 pinned host）**不推荐**：
  实测 2.18 s/token，且 T=1 边界会在部分请求的首步触发 gate NaN。
  NZ 常驻设备会 OOM（AC+NZ 构建峰值 81.5 GiB，见 3 节内存表）；不要设
  `DS41M_DECODE_HOST_NZ=0`。默认 `full` 无这些限制。
- 常驻模式不带 profiler；要 trace 仍用 `launch_mini.py --prof`。
- prefill 3 s 承诺只覆盖 `L ≤ 预热阶梯最大值`；超过显示实测，不承诺。
- host cache 未启动时 D→P 会退化为磁盘逐张量回载（数百秒）；tp8-fidelity 启动时直接拒绝。
- 短 prompt 会被自动补 token（≥16 且奇数）；首 token 与"不补"的裸 prompt 不逐位一致
  （裸短 prompt 本身在 step2 会 NaN，无法作为参考）。

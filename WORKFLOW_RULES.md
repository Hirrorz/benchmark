# 跑任务规则（v3，2026-11-03 增加「禁止前台等待」）

## 铁律 1：任何等待都必须在后台，前台查询禁止超过 30 秒

**背景**：多次因为在前台写 `for i in $(seq 1 30); do sleep 30; done` 轮询，
把整个会话挂死十几分钟。用户明确要求固化。

**规则**：

- ✅ 需要「等某个任务跑完」→ 用 `run_in_background: true` 起后台 job，
  然后**立刻返回**，用下次工具调用做一次快速状态查询（<30 s）。
- ✅ 前台查询只允许「一次性、无 sleep」或「单次 sleep ≤ 5 s」的快照，
  例如 `df -h`、`pgrep`、`tail -5 log`、`awk 'END{print NR-1}' results.tsv`。
- ❌ **禁止**前台出现这种模式：
  ```bash
  for i in $(seq 1 N); do sleep 30; done      # ← 严禁
  while ! grep -q DONE log; do sleep 60; done   # ← 严禁
  ```
- ❌ 禁止以「等 3 分钟看稳定性」为理由在前台 sleep 循环。
  要看稳定性 → 后台 job 里观察，或隔一段时间做两次独立快照对比。

**判断口径**：单次工具调用的**实际耗时**不应超过 30 秒。
如果发现某个操作天然要更久（du 全盘、扫 30T、等网格跑完），
一律改成后台 job。

## 铁律 2：多点位/网格必须验证「8 卡都活着」

**背景（2026-11-03 事故）**：网格 attempt 崩溃后，7 个 rank 变孤儿占着卡，
下一轮 attempt 只起了 `rank 6` 一个进程，产出 25 分钟的**单卡假数据**，
`npu-smi` 上只看到卡 6 占 98 GB、其余卡 4.7 GB 基线。

**规则**：

- 每轮 attempt **开始前**必须 `hard_clean`：3 轮杀死所有
  `launch_mini|core/runner.py`，再等 8 卡 HBM 全部回落到 < 8 GB 才启动。
- attempt **启动后 30 秒内**必须验证：
  ```bash
  pgrep -f 'core/runner\.py.*--rank' | wc -l     # 必须是 8
  ```
  不是 8 → 立刻停，清理重来，**不要让它跑下去**。
- 数据入库前抽查：各 rank 日志大小应相近（±10%），
  8 个 rank 的"完成"条数应一致。

## 铁律 3：点位几何必须先验合法性，非法会让整轮死

`hit > 0` 且 `M + replay_tail > L` 时，runner 抛
`命中几何非法（要算的行数超过序列长）` 并 `SystemExit` ——
**不是单点跳过，是整轮网格直接退出**。生成点位表时必须过滤。

另：`R = M + replay_tail` 超过 32628（实测）会让 DeepSelect 抢不到
workspace（8/16 GB）→ `Insufficient_Device_Memory`，也是整轮死。

## 铁律 4：跑任务不能长 sleep 才发现失败

- 启动后 **10–15 秒**做首查：日志大小 > 0、无 `Traceback`/`aicore`/`OOM`。
- 之后用**日志增长**判活，不要用 CPU 占用（程序在等集合通信时 CPU 可以是 0%）。
- 连续 3 次快照（间隔 20–30 s）日志不增长且进程还在 → 视为卡死，
  抓 `pgrep` + 最后一屏日志 + `npu-smi` 再决定杀不杀。

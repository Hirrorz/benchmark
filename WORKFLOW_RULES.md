# 跑任务规则（防「长 sleep 后才发现一开始就失败」）

> 2026-11-02 立规。起因：多次用 `Start-Sleep 90~120` 等 NPU 作业，等完才发现进程从第 1 秒就失败。
>
> **2026-11-02 修正 1（重要）**：`pcpu == 0.0` **不是**卡死判据。NPU 作业里 host 大多数时候
> 就在等 device，CPU 必然接近 0%。用它判卡死会**误杀正在正常跑的作业**（实测日志
> 3 秒增长 7 KB 的同时 CPU 恰为 0.0）。真正的判据见 §2.3。

## 一、禁止项

1. **禁止单次 `sleep` 超过 30 秒**。任何等待都必须能被下一轮检查打断。
2. **禁止「启动 → 长时间等待 → 一次性看结果」**这种黑盒式跑法。
3. **禁止只用一个信号判活/判死**（尤其 CPU 占用）。

## 二、必须项

### 2.1 启动后 10 秒内首查（最重要）

任何后台任务启动后，**10 秒内**必须确认它没有在启动阶段就失败：

```bash
ps -eo pid,etime,pcpu,cmd | grep <任务关键字> | grep -v grep   # 进程在不在
grep -iE 'Segfault|Traceback|Assertion|RuntimeError' <日志> | head -3
```

**10 秒内就出现 `Segfault`/`Traceback`/`Assertion` → 立即停，不要等。**

### 2.2 每 20 秒一轮短轮询（同一次工具调用内完成）

不要每轮单独发命令（SSH 建连开销比等待还大）。

### 2.3 健康判据（**按可靠性排序**，用前两个为主）

| 优先级 | 判据 | 阈值 | 说明 |
|---|---|---|---|
| **1（主）** | **日志字节数无增长** | **≥ 60s** | 最可靠。真卡死时日志必然停 |
| **1（主）** | 日志出现 `Segfault`/`Traceback`/`Assertion`/`RuntimeError` | 出现即判 | 已经失败 |
| **2（辅）** | `npu-smi info` 的 `NPU Util` 持续为 0 且日志无增长 | ≥ 60s | 两者同时成立才判死 |
| **2（辅）** | 进程消失但结果文件不存在 | 立即 | 异常退出 |
| 3 | 结果文件存在但 `err != 0` / `rc != 0` | 立即 | 跑完但失败 |
| ~~✗~~ | ~~`pcpu == 0.0`~~ | **不作判据** | **误报源**：NPU 作业 host 本来就空转 |

**采样方法**（3 秒两次采样，比单次 `ps` 可靠得多）：

```bash
s1=$(stat -c %s "$LOG"); sleep 3; s2=$(stat -c %s "$LOG")
[ "$s2" -gt "$s1" ] && echo "RUNNING（日志在增长）" || echo "SUSPECT（3s 无增长）"
```

### 2.4 失败处置顺序

1. **kill 残留进程**（避免占卡/占显存影响下一次）
2. **拉日志尾部 20 行**（去噪：`grep -vE 'destroy|^export '`）
3. **定位首错**（`grep -iE 'Segfault|Traceback|Assertion'`）
4. **报给用户：一句话结论 + 首错原文**
5. 同一原因不许连跑第三次，除非能说清这次哪里不同

## 三、标准模板

```bash
# 1) 启动
nohup bash /path/task.sh > /path/task.out 2>&1 & echo started

# 2) 首查（10~12s 后）
sleep 12
ps -eo pid,etime,pcpu,cmd | grep task | grep -v grep
grep -ciE 'Segfault|Traceback|Assertion|RuntimeError' /path/task.out

# 3) 短轮询（20s × N，同一次调用）
for i in $(seq 1 8); do
  sleep 20
  alive=$(pgrep -f task >/dev/null && echo 1 || echo 0)
  errs=$(grep -ciE 'Segfault|Traceback|Assertion|RuntimeError' /path/task.out || true)
  s1=$(stat -c %s /path/task.out); sleep 3; s2=$(stat -c %s /path/task.out)
  grew=$([ "$s2" -gt "$s1" ] && echo 1 || echo 0)
  echo "[$i] alive=$alive errs=${errs:-0} log_grew=$grew"
  [ "${errs:-0}" -gt 0 ] && break
  [ "$alive" = "0" ] && break
  # 只有「日志连续两轮不增长」才判死
done
```

## 四、自检三问（启动前）

1. **10 秒后我会看什么？**
2. **什么信号说明它一开始就失败了？**（必须是 `Segfault`/`Traceback`/日志停增，不能只看 CPU）
3. **失败了我先做什么？**（kill → 拉日志 → 报首错）

# 采集命令手册（可直接复制执行）

全部在 **208**（`liteserver-hps-tokens-001-00006`）上跑，工作目录
`/mnt/xds/sfs/l00972766/ds41_mini`。

---

## 0. 每次开新 shell 的前置

```bash
cd /mnt/xds/sfs/l00972766/ds41_mini
source env_pack.sh
export DS41M_CP_BUCKET=0          # ★ 必须 0：DeepEP 与 ALOP profiler 不兼容
export DS41M_CP_HIT=1             # 命中 + CP 路径
export DS41M_MOE_CAST_INTO_SB=1
unset LD_PRELOAD
```

> `DS41M_CP_BUCKET=1`（DeepEP URMA）只在 **noprof** 时可用；
> 开 ALOP 采集时必须 `=0`（否则 `halAsyncDmaJettyCreate drvRetCode=7` → 507015）。

---

## 1. 单点位采集（最常用）

```bash
# CP1，128k@0.8，采 rank0，warmup 2 + iters 8
DS41M_CP_SIZE=1 bash tools/run_hit.sh \
    --points-spec '128k@0.8/b1/r1' \
    --cards 0,1,2,3,4,5,6,7 \
    --prof-ranks 0 --prof-warmup 2 --iters 8 \
    -o /mnt/xds/sfs/l00972766/tmp/one_28k_cp1
```

`--prof` **不用写**（默认就开）。`results.tsv` 里同时给出：
- 稳态（取未被采集的 rank，如 `rank1`）→ 真实 forward 时间
- trace 在 `<out>/<tag>/prefill/PROF_*/`

产物：

```
<out>/results.tsv                      ← 稳态 + 几何 + err
<out>/logs/rank{0..7}.log              ← 每 rank
<out>/<tag>/prefill/PROF_*/            ← trace（需 msprof export）
```

### 导出成 MindStudio CSV

```bash
P=$(find <out> -maxdepth 6 -name 'PROF_*' -type d | head -1)
bash -lc "msprof --export=on --type=text --output='$P'"
# → $P/mindstudio_profiler_output/*.csv + msprof_*.json
```

> **必须用 `bash -lc`**。手动设 `LD_LIBRARY_PATH` 只会得到 `README.txt`。

### 打包

```bash
# 只要 CSV（~4 MB/点位）
tar czf /mnt/xds/sfs/l00972766/tmp/mso.tgz -C "$P" mindstudio_profiler_output

# 完整（~96 MB/点位）
tar czf /mnt/xds/sfs/l00972766/tmp/proffull.tgz -C "$(dirname "$P")" "$(basename "$P")"
```

---

## 2. 不采集、只要性能数（快 3–9 倍，无 trace）

```bash
DS41M_CP_SIZE=2 bash tools/run_hit.sh \
    --points-spec '128k@0.8/b1/r1' \
    --cards 0,1,2,3,4,5,6,7 \
    --noprof-only --prof-warmup 2 --iters 8 \
    -o /mnt/xds/sfs/l00972766/tmp/np_128k_cp2
```

这种模式可以安全用 `DS41M_CP_BUCKET=1`（DeepEP，更快）。

---

## 3. 多点位（一条命令跑一批）

点位内联写法（逗号或空格分隔）：

```
L@hit[/bN][/r0|1][/tTAIL]
  128k@0.8         → L=131072, hit=0.8
  128k@0.8/b1/r1   → batches=1, replay=on（默认）
  64k@0.97%        → 也认百分号
```

```bash
DS41M_CP_SIZE=2 bash tools/run_hit.sh \
    --points-spec '128k@0.8/b1/r1,128k@0.9/b1/r1,128k@0.95/b1/r1' \
    --cards 0,1,2,3,4,5,6,7 \
    --prof-ranks 0 --prof-warmup 2 --iters 8 \
    -o /mnt/xds/sfs/l00972766/tmp/multi_cp2
```

或写成 JSON 文件（更长的列表用这个）：

```bash
cat > /tmp/pts.json <<'JSON'
[{"L":131072,"hit":0.8,"batches":1,"replay":true},
 {"L":65536,"hit":0.9,"batches":1,"replay":true}]
JSON
DS41M_CP_SIZE=1 bash tools/run_hit.sh --points /tmp/pts.json ...
```

---

## 4. 网格生成器（三 CP × 20 L × hit）

```bash
cd /mnt/xds/sfs/l00972766/ds41_mini
for CP in 1 2 4; do
  python3.11 tools/gen_grid.py --out /tmp/points_cp$CP.json --cp $CP --report-skipped
done
# → 每 CP 357 个点位 + points_cp*.json.skipped.tsv（跳过的 43 个及原因）
```

内置三条实测硬约束（会自动过滤）：
- `R = M + replay_tail ≤ 32628`（DeepSelect workspace 上限，超了 `Insufficient_Device_Memory`）
- `hit=0` 时 `L ≤ 32768`（全算路径，更大必崩）
- `hit>0` 时 `R ≤ L`（否则 runner 抛「命中几何非法」**整轮退出**）

---

## 5. 出表格

```bash
python3.11 tools/gen_table.py \
    --root <网格根目录> \
    --out  <根>/grid_final.tsv \
    --md   <根>/grid_final.md
```

列：`seq_len / hit_rate / strategy / per_die_tokens / forward_40_layers(ms) / TPS/card / err`

只采信 `err=0` 的行（`err>0` 表示有算子被 DeepSelect 跳过）。

---

## 6. 判活 / 排错（重要）

### ① 必须确认 8 个 rank 都活着

```bash
pgrep -f 'core/runner\.py.*--rank' | wc -l      # 必须是 8
```

**不是 8 就是僵尸残留**：前一轮崩溃后 7 个 rank 变孤儿占着卡，
下一次只起 1 个 rank，会产出**单卡假数据**（`npu-smi` 上只有一张卡 90+ GB）。

### ② 清理干净再重启

```bash
for r in 1 2 3; do
  for p in $(pgrep -f 'launch_mini|core/runner\.py'); do kill -9 $p; done
  sleep 4
done
# 等 8 卡 HBM 全部回落到 < 8 GB
npu-smi info | grep -oE '[0-9]+ */ *98304'
```

### ③ 卡住 vs 在算

```bash
stat -c %s <out>/<日志>        # 记下；sleep 20 再取一次
```
20 秒内字节数增长 = 在跑；连续 3 次不增长且进程还在 = 卡死。

**不要用 CPU 占用判断**（等集合通信时 CPU 可以是 0%）。

### ④ 大 L 的每个点位很慢

`L=1M` 单点位约 **88 s**（10 个 forward × 8.8 s），不是卡住。

---

## 7. 已知坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `halAsyncDmaJettyCreate drvRetCode=7` / 507015 | `DS41M_CP_BUCKET=1` 与 ALOP 同开 | 采集时设 `=0` |
| `Insufficient_Device_Memory` 8/16 GB | DeepSelect workspace 随 local T 增长 | `R ≤ 32628` |
| `命中几何非法` + 整轮 `rc=1` | `hit>0` 且 `M+140 > L` | 用 `gen_grid.py` 过滤 |
| 只有 1 个 rank 在跑 | 崩溃后孤儿进程占卡 | 强杀 3 轮 + 等 HBM 回落 |
| `err=5`（CP2/CP4 小 L 高发） | DeepSelect 闭源内核 stream 泄漏 | 无法根治；只采信 `err=0` |
| msprof export 只出 README.txt | 没用登录 shell 的 CANN 栈 | `bash -lc "msprof ..."` |
| `du -sh /mnt/xds/sfs/*` 卡几分钟 | 遍历 30 T | 别全盘 du，直接查目标目录 |

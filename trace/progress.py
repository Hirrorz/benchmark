"""阶段进度（§12）：百分比按**实测标定的阶段表**给，不拍脑袋。

用法（M2 起真跑时）：
    p = Progress(total_hint_s=120)
    p.stage("S3 权重装载", detail_fn=lambda: f"{done}/{total} 层")
    p.tick(0.3)          # 该阶段内部进度 0~1
    p.done("S3 权重装载")
"""
from __future__ import annotations

import os
import sys
import time

# 阶段表：名称 → 权重（秒），来自 2026-10-14 实测的启动分解（cp4/noeng/合成权重 ≈120s）
STAGES = [
    ("S0 环境与导入", 17),
    ("S1 通信域初始化", 2),
    ("S2 模型组装", 27),
    ("S3 权重装载", 45),
    ("S4 池/engram offload", 2),
    ("S5 预热首跑", 17),
    ("S6 测量前向", 10),
    ("S7 收尾", 1),
]
_TOTAL = sum(w for _, w in STAGES)


# mini（单卡 runner）的阶段表 —— 2026-10-14 实测标定：
#   4 层跑：S0 4.2s / S1 1.6s / S2 11.1s（含首层 JIT ≈10s）；40 层整跑共 34s（S3 含 profiler 解析 8s）
# 每次跑完 runner 会打印实测值，换机器/换算子后照它更新这张表。
STAGES_MINI = [
    ("S0 环境与算子注册", 4),
    ("S1 逐层：权重装载", 16),
    ("S2 逐层：前向执行", 14),
    ("S3 trace 汇总", 8),
]


class Progress:
    def __init__(self, enabled=True, every=1.0, stream=None, stages=None):
        self.stages = list(stages or STAGES)
        self.on = enabled
        self.every = float(every)
        self.tty = (stream or sys.stderr).isatty()
        self._t0 = time.time()
        self._last = 0.0
        self._stage, self._base, self._frac = "", 0.0, 0.0
        self._done_s = 0.0
        self.measured = {}                      # 阶段名 → 实测秒数（跑完打印）

    def _emit(self, line, force=False):
        now = time.time()
        if not self.on or (not force and now - self._last < self.every):
            return
        self._last = now
        end = "\r" if self.tty else "\n"
        sys.stderr.write(line + end)
        sys.stderr.flush()

    def stage(self, name):
        """进入某阶段（上一个阶段按其耗时记账）。"""
        if self._stage:
            self.measured[self._stage] = self.measured.get(self._stage, 0.0) + \
                time.time() - self._stage_t0
        self._stage_t0 = time.time()
        self._stage, self._frac = name, 0.0

    def end_stage(self):
        if self._stage:
            self.measured[self._stage] = self.measured.get(self._stage, 0.0) + \
                time.time() - self._stage_t0
            self._stage = ""
        return dict(self.measured)

    def tick(self, frac, detail=""):
        # ★ 2026-10-21 host 优化：关闭时直接返回。旧版即使 `on=False` 也要算百分比/ETA/
        #   `strftime` 再进 `_emit` 才丢弃 —— 每层一次白算（40 层/pass 属于纯 host 开销）。
        if not self.on:
            return
        self._frac = max(0.0, min(1.0, frac))
        names = [n for n, _ in self.stages]
        idx = names.index(self._stage) if self._stage in names else 0
        tot = sum(w for _, w in self.stages)
        done = sum(w for _, w in self.stages[:idx]) + self.stages[idx][1] * self._frac
        pct = 100.0 * done / tot
        el = time.time() - self._t0
        eta = (el / done * (tot - done)) if done > 0.5 else 0.0
        self._emit(f"[{time.strftime('%H:%M:%S')}] {pct:5.1f}% │ {self._stage}"
                   + (f" {detail}" if detail else "")
                   + f" │ 已用 {el:.0f}s" + (f" · 预计再 {eta/60:.1f}m" if eta > 1 else ""))

    def done(self, name=""):
        self._emit(f"[{time.strftime('%H:%M:%S')}] 100.0% │ {name or self._stage} 完成"
                   f" │ 总 {time.time()-self._t0:.0f}s", force=True)

"""显存实测日志（默认关；`DS41M_MEMLOG=1` 开）。

只做查询，不改分配行为：
  · `torch.npu.memory_allocated()`：PyTorch caching allocator 当前占用；
  · `memory_reserved()`：allocator 已向驱动预留的段；
  · `max_memory_allocated()`：本进程 allocator 峰值（pass 开始时 reset）；
  · `mem_get_info()`：驱动侧 **整卡 free/total**（含厂商对称缓冲、ElasticBuffer
    等非 torch 分配）——判断“卡还剩多少”以这个为准。

环境变量：
  DS41M_MEMLOG=1             总开关
  DS41M_MEMLOG_OPS=1         逐算子打印（默认只逐层）
  DS41M_MEMLOG_FROM/TO       层号范围（默认 0..39）
  DS41M_MEMLOG_RANKS=0       打印的 rank（all 或逗号列表；默认只看 rank0）
"""
import os

import torch
import torch_npu

_ON = os.environ.get("DS41M_MEMLOG", "").strip().lower() in ("1", "on", "true", "yes")
ENABLED = _ON
OPS = os.environ.get("DS41M_MEMLOG_OPS", "").strip().lower() in ("1", "on", "true", "yes")
FROM = int(os.environ.get("DS41M_MEMLOG_FROM", "0") or 0)
TO = int(os.environ.get("DS41M_MEMLOG_TO", "39") or 39)


def _parse_ranks():
    s = (os.environ.get("DS41M_MEMLOG_RANKS", "0") or "0").strip().lower()
    if s in ("all", "*"):
        return None
    out = set()
    for t in s.replace(",", " ").split():
        if t.lstrip("-").isdigit():
            out.add(int(t))
    return out or {0}


_RANKS = _parse_ranks()


def enabled(rank=None):
    if not _ON:
        return False
    if _RANKS is None:
        return True
    if rank is None:
        rank = int(os.environ.get("RANK", "0") or 0)
    return int(rank) in _RANKS


def mem(tag, rank=None):
    if not enabled(rank):
        return
    try:
        free, total = torch_npu.npu.mem_get_info()
        alloc = torch_npu.npu.memory_allocated()
        res = torch_npu.npu.memory_reserved()
        peak = torch_npu.npu.max_memory_allocated()
        r = rank if rank is not None else os.environ.get("RANK", "0")
        print(f"[mem] r{r} {tag} alloc={alloc / 2**30:.3f} res={res / 2**30:.3f} "
              f"peak={peak / 2**30:.3f} free={free / 2**30:.3f}/{total / 2**30:.3f} GiB",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[mem] {tag} ERR {type(exc).__name__}: {exc}", flush=True)

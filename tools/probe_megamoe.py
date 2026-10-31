#!/usr/bin/env python3
import os as _os, sys as _sys
_ds41_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    _q = _os.path.join(_ds41_root, _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)



"""单独验证 mega_moe 的最小调用（不进 runner，方便定位卡在哪一步）。

用法：
  python3.11 -u tools/probe_megamoe.py                 # 默认 T=1024 / E=384 / npu:2
  python3.11 -u tools/probe_megamoe.py --T 2048 --experts 8 --device npu:2
  python3.11 -u tools/probe_megamoe.py --mode ascendc  # 只看 ascendc 单卡卡在哪（会失败）

自己造**与真权重同形状/dtype/分布**的权重（清单见 weights_manifest.py）与输入，
走的是 exec_moe.py 里 runner 会走的同一条代码路径（`_mega_moe(st, ctx, w)`）。

真权重目录 /mnt/xds/sfs/DeepSeek-V4.1-Flash-cann-mxfp8-mxfp4 的形状（逐字）：
    ffn.experts.{e}.w1.weight  I8      [2304, 2560]    ← 一字节两个 fp4（K 折半）
    ffn.experts.{e}.w1.scale   E8M0    [2304, 160]     ← 每 32 个 K 元素一个
    ffn.experts.{e}.w2.weight  I8      [5120, 1152]
    ffn.experts.{e}.w2.scale   E8M0    [5120, 72]
    ffn.shared_experts.w1/w3   F8_E4M3 [2304, 5120] / scale E8M0 [2304, 160]
    ffn.shared_experts.w2      F8_E4M3 [5120, 2304] / scale E8M0 [5120, 72]
"""
import argparse
import os
import sys
import time

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exec_ops                                             # noqa: E402
import exec_moe                                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, default=1024)
ap.add_argument("--experts", type=int, default=384)
ap.add_argument("--topk", type=int, default=6)
ap.add_argument("--device", default="npu:2")
ap.add_argument("--mode", default="auto", choices=["auto", "ascendc", "deepgemm"])
ap.add_argument("--skip-zero-check", action="store_true")
a = ap.parse_args()

os.environ["DS41M_MOEMODE"] = a.mode
exec_ops.ensure_ops()
exec_moe.register(exec_ops.OPS, exec_ops.SKIP, exec_ops)

T, E, K = a.T, a.experts, a.topk
H, I = exec_moe.HIDDEN, exec_moe.INTER
DEV = a.device
torch.npu.set_device(DEV)
print(f"== probe mega_moe: T={T} E={E} topk={K} H={H} I={I} dev={DEV} mode={a.mode}", flush=True)


# ── 造权重（形状/dtype 与真权重一致；数值随机，分布合理）──
def _e8m0(shape):
    """E8M0 = 2^(x-127)；取 124~132 附近，避开 0/255 这类退化位型。"""
    return torch.randint(124, 132, shape, device=DEV).to(torch.uint8)


def _fp8(shape):
    """fp8_e4m3（避开 0x7F=NaN 与负零以外的符号位随机，别造出"全正权重"这种离谱分布）。"""
    b = torch.randint(0, 127, shape, device=DEV).to(torch.uint8)
    sign = torch.randint(0, 2, shape, device=DEV).to(torch.uint8) << 7
    return (b | sign).view(torch.float8_e4m3fn)


def fake_layer_weights():
    w = {}
    t0 = time.time()
    for e in range(E):
        p = f"ffn.experts.{e}."
        w[p + "w1.weight"] = torch.randint(-128, 128, (I, H // 2), dtype=torch.int8, device=DEV)
        w[p + "w3.weight"] = torch.randint(-128, 128, (I, H // 2), dtype=torch.int8, device=DEV)
        w[p + "w1.scale"] = _e8m0((I, H // 32))
        w[p + "w3.scale"] = _e8m0((I, H // 32))
        w[p + "w2.weight"] = torch.randint(-128, 128, (H, I // 2), dtype=torch.int8, device=DEV)
        w[p + "w2.scale"] = _e8m0((H, I // 32))
    w["ffn.shared_experts.w1.weight"] = _fp8((I, H))
    w["ffn.shared_experts.w3.weight"] = _fp8((I, H))
    w["ffn.shared_experts.w1.scale"] = _e8m0((I, H // 32))
    w["ffn.shared_experts.w3.scale"] = _e8m0((I, H // 32))
    w["ffn.shared_experts.w2.weight"] = _fp8((H, I))
    w["ffn.shared_experts.w2.scale"] = _e8m0((H, I // 32))
    torch.npu.synchronize()
    print(f"造权重完成：{len(w)} 个 tensor，{time.time() - t0:.1f}s", flush=True)
    return w


w = fake_layer_weights()
ctx = dict(H=H, I=I, T=T, topk=K, e_local=E)
torch.manual_seed(0)

# 输入：x[T,H] bf16 + topk_idx/topk_weight（形状/dtype 与 runner 里 tilelang gate 的输出一致）
x = (torch.randn(T, H, dtype=torch.float32, device=DEV) * 0.1).to(torch.bfloat16)
logits = torch.randn(T, E, dtype=torch.float32, device=DEV)
tw, ti = torch.topk(torch.softmax(logits, dim=-1), K, dim=-1)
st = {"x": x, "topk_idx": ti.to(torch.int32), "topk_weight": tw.float()}
print(f"输入：x{tuple(x.shape)} {x.dtype}；topk_idx{tuple(st['topk_idx'].shape)} {st['topk_idx'].dtype}"
      f"；topk_weight{tuple(st['topk_weight'].shape)} {st['topk_weight'].dtype}", flush=True)

t0 = time.time()
try:
    exec_moe._mega_moe(st, ctx, w)
except Exception as exc:                                    # noqa: BLE001
    import traceback
    traceback.print_exc()
    print(f"\n✗ mega_moe 失败（{time.time() - t0:.1f}s）：{type(exc).__name__}: {exc}", flush=True)
    sys.exit(1)
torch.npu.synchronize()
y = st["moe_out"]
print(f"\n✓ mega_moe 首次调用 ok（含权重打包 + JIT）：{time.time() - t0:.1f}s", flush=True)
print(f"  y: {tuple(y.shape)} {y.dtype}  后端={exec_moe._CACHE['backend']}", flush=True)
_p = exec_moe._CACHE["w"]
print(f"  权重布局（deepgemm transform 后）："
      f"l1 {tuple(_p['l1'][0].shape)}{_p['l1'][0].dtype}/{tuple(_p['l1'][1].shape)}{_p['l1'][1].dtype}"
      f"  l2 {tuple(_p['l2'][0].shape)}{_p['l2'][0].dtype}/{tuple(_p['l2'][1].shape)}{_p['l2'][1].dtype}"
      f"  shared_l1 {tuple(_p['sh1'][0].shape)}", flush=True)
print(f"  统计：absmax={y.float().abs().max().item():.4e} mean={y.float().mean().item():.4e} "
      f"nan={int(torch.isnan(y).sum())} inf={int(torch.isinf(y).sum())}", flush=True)

t0 = time.time()
for _ in range(3):
    st["x"] = x
    exec_moe._mega_moe(st, ctx, w)
torch.npu.synchronize()
print(f"  热态 3 次：{(time.time() - t0) / 3 * 1000:.1f} ms/次（T={T}）", flush=True)

if not a.skip_zero_check:
    st0 = {"x": torch.zeros_like(x), "topk_idx": st["topk_idx"], "topk_weight": st["topk_weight"]}
    exec_moe._mega_moe(st0, ctx, w)
    torch.npu.synchronize()
    amax = st0["moe_out"].float().abs().max().item()
    print(f"  零输入自检：y_absmax={amax:.3e} → {'✓ 应为 0' if amax == 0 else '⚠️ 非 0（数值可疑）'}",
          flush=True)

free, total = torch.npu.mem_get_info()
print(f"  显存：free {free / 2**30:.2f}/{total / 2**30:.2f} GB", flush=True)

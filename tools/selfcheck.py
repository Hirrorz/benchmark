#!/usr/bin/env python3



"""算子自检 —— ds41_mini 的验收门（换机器后先跑这个）。

口径：**不校验 CANN 版本**——只要目标机上这些算子能跑通即可。
所以自检不比对任何数值基线，只回答三个问题：
  ① 环境能不能把 vendor 算子命名空间拉起来？（Tier 0，不碰卡）
  ② 每个算子能不能用**真调用点**（mini 自己的 exec_* 代码）成功跑一次？（Tier 1，需 1 张卡）
  ③ 输出是不是健康的（非 NaN/Inf、形状/次数量级对得上）？

Tier 1 的做法：用合成权重让 mini 真跑 0..24 层（这一段的层角色覆盖了**全部**算子种类：
普通层 / engram 层 1 / kv+idx 源层 2 / 候选源层 20 / QSLI 消费层 24），
在 `exec_ops.run_op` 上挂钩子逐算子计时/计次/查健康，跑完出表。

用法：
    source env_pack.sh
    python3.11 tools/selfcheck.py                # 环境 + 逐算子（0..24 层，T=1024）
    python3.11 tools/selfcheck.py --card 5       # 指定**物理**卡（内部转成 npu:0，绕开逻辑卡号坑）
    python3.11 tools/selfcheck.py --full         # 再跑满 40 层
    python3.11 tools/selfcheck.py --json out.json
    python3.11 tools/selfcheck.py --parity <官方collect目录>   # 可选：真权重逐层对拍
退出码：0 全过；1 有算子失败；2 Tier 0 就挂了。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ds41_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "meta", "layout", "ops", "cp", "trace", "weights"):
    _q = _os.path.join(_ds41_root, _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Tier 0 期望存在的算子（mini 真调用点用到的**全部** vendor 入口）
EXPECT = [
    ("cann_ops_transformer.ds41", "compressor"),
    ("cann_ops_transformer.ds41", "indexer_prologue_qw"),
    ("cann_ops_transformer.ds41", "indexer_prologue_k"),
    ("cann_ops_transformer.ds41", "mixed_quant_sparse_flash_mla"),
    ("cann_ops_transformer.ds41", "mixed_quant_sparse_flash_mla_metadata"),
    ("cann_ops_transformer.ds41", "quant_lightning_indexer"),
    ("cann_ops_transformer.ds41", "quant_lightning_indexer_metadata"),
    ("cann_ops_transformer.ds41", "quant_sparse_lightning_indexer"),
    ("cann_ops_transformer.ds41", "quant_sparse_lightning_indexer_metadata"),
    ("cann_ops_transformer", "mhc_post"),
    ("cann_ops_transformer", "inplace_partial_rotary_mul"),
    ("custom", "npu_hc_pre_v2"),
    ("custom", "kv_compress_epilog_v2"),
    ("torch_npu", "npu_rms_norm"),
    ("torch_npu", "npu_dynamic_mx_quant"),
    ("torch_npu", "npu_quant_matmul"),
    ("torch_npu", "npu_transpose_quant_batchmatmul"),
    ("torch_npu", "npu_fused_matmul"),
    ("tile_kernels.moe", "moe_topk_gate"),
    ("tile_kernels.mhc", "mhc_post_fwd"),
    ("tile_kernels.engram", "engram_hash"),
    ("tile_kernels.engram", "engram_gate_fwd"),
    ("tile_kernels.engram", "fused_weight"),
]
# 健康检查看的输出（算子可能写其中任意几个；只看前若干行，避免拖慢）
HEALTH_KEYS = ("x", "residual", "q", "kv", "qr", "attn_out", "o_lora", "logits",
               "topk_idx", "topk_weight", "moe_out", "cmp_kv", "cmp_latent", "engram_out")
HEALTH_ROWS = 64
# 某些 op key 在本配置下**故意**跑不到：写清原因，别让它看起来像缺口
UNCOVERED_WHY = {
    "gating": "未融合 MoE 链 —— 本配置 megamoe=ascendc，整链被 mega_moe 取代",
    "gating_sort": "同上（未融合链）",
    "reroute": "同上（未融合链）",
    "dispatch": "同上（未融合链，走 hcom_alltoallv）",
    "gmm": "同上（未融合链）",
    "swiglu_quant": "同上（未融合链）",
    "combine": "同上（未融合链，走 hcom_alltoall）",
    "finalize": "同上（未融合链）",
    "gate_topk": "被 tilelang 的 topk_gate 取代（本配置 gate_topk: tilelang）",
    "dyn_mx_quant": "在 QMM 算子内部产生（IMPLICIT），不单独发",
    "gather_tail": "命中/replay 的形状变换（hit.ratio=0 时不出现）",
    "scatter_tail": "同上（命中/replay）",
    "transpose_qmm": "已并入 wo_a 承担",
}


def _has_op(ns: str, name: str) -> bool:
    """算子入口存在性：torch_npu.* / tile_kernels.* 是普通函数；其余走 torch.ops 命名空间。
    （`dir(torch.ops...)` 对部分算子不可靠，所以逐层 getattr + op() 兜底。）"""
    try:
        if ns == "torch_npu":
            import torch_npu
            return hasattr(torch_npu, name)
        if ns.startswith("tile_kernels"):
            import importlib
            return hasattr(importlib.import_module(ns), name)
        import torch
        cur = torch.ops
        for part in ns.split("."):
            cur = getattr(cur, part)
        if hasattr(cur, name):
            return True
        try:                                   # 兜底：按字符串取（部分算子不出现在 dir 里）
            cur.op(name)
            return True
        except Exception:                      # noqa: BLE001
            return False
    except Exception:                          # noqa: BLE001
        return False


def tier0() -> dict:
    print("== Tier 0 · 环境（不碰卡）" + "=" * 40)
    items = []

    def add(ok, name, detail=""):
        items.append((name, ok, detail))
        print(f"  {'✅' if ok else '❌'} {name:<46} {detail}")

    py_ok = sys.version_info[:2] >= (3, 11)
    add(py_ok, f"python ≥3.11（torch 只有 cp311 包）", sys.version.split()[0])

    mods = {}
    for m in ("torch", "torch_npu", "yaml", "numpy"):
        try:
            mods[m] = __import__(m)
            add(True, f"import {m}", getattr(mods[m], "__version__", ""))
        except Exception as exc:               # noqa: BLE001
            add(False, f"import {m}", f"{type(exc).__name__}: {str(exc)[:60]}")
    if "torch_npu" not in mods:
        print("\n⛔ torch_npu 都 import 不了。最小条件里有它，请按顺序查：")
        print("   ① `source env_pack.sh`（它会把镜像自带/覆盖层的 torch 站点包拼进 PYTHONPATH）")
        print("   ② 目标机确实没有 torch：用 DS41_TORCH_SITE=<site-packages 路径> 指一套")
        print("      （950-DT 镜像自带的是 /usr/local/python3.11.16/lib/python3.11/site-packages）")
        print("   ③ 都拿不到 → 这台机器不满足最小条件（Tier 0 FAIL）")
        return {"ok": False, "items": items}

    cann = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME") or "?"
    ver = ""
    for f in (os.path.join(cann, "version.cfg"), os.path.join(cann, "opp/version.info")):
        if os.path.isfile(f):
            try:
                with open(f, errors="ignore") as fh:
                    ver = next((ln.strip() for ln in fh if "version" in ln.lower()), "")[:60]
            except OSError:
                pass
            break
    # 版本**只打印不判定**：口径是「算子能跑就行」
    add(os.path.isdir(cann), "CANN 路径（版本不校验）", f"{cann} {ver}")

    try:
        import cann_ops_transformer                # noqa: F401
        import cann_ops_transformer.ops            # noqa: F401
        import custom_ops                          # noqa: F401
        add(True, "import 厂商算子包 (custom_ops/cann_ops_transformer)", "")
    except Exception as exc:                       # noqa: BLE001
        add(False, "import 厂商算子包", f"{type(exc).__name__}: {str(exc)[:70]}")

    try:
        import tile_kernels.moe, tile_kernels.mhc, tile_kernels.engram   # noqa: F401
        add(True, "import tile_kernels (moe/mhc/engram)", "")
    except Exception as exc:                       # noqa: BLE001
        add(False, "import tile_kernels", f"{type(exc).__name__}: {str(exc)[:70]}")

    missing = []
    for ns, name in EXPECT:
        if not _has_op(ns, name):
            missing.append(f"{ns}.{name}")
    add(not missing, f"算子入口存在性（{len(EXPECT) - len(missing)}/{len(EXPECT)}）",
        "全在" if not missing else f"缺: {', '.join(missing[:6])}")

    try:
        n = mods["torch_npu"].npu.device_count()
        add(n > 0, "NPU 可见性", f"device_count={n}")
    except Exception as exc:                       # noqa: BLE001
        add(False, "NPU 可见性", f"{type(exc).__name__}: {str(exc)[:70]}")

    ok = all(o for _n, o, _d in items)
    print(f"\n  Tier 0：{'✅ PASS' if ok else '❌ FAIL'}")
    return {"ok": ok, "items": items, "missing": missing}


class OpHook:
    """挂在 exec_ops.run_op 上：逐算子计次/计时/查输出健康。"""

    def __init__(self):
        self.stat = {}
        self.order = []

    def __call__(self, orig):
        def wrapper(key, st, ctx, w):
            t0 = time.perf_counter()
            ok, err = True, ""
            hold = st.get("_shared") if isinstance(st.get("_shared"), dict) else {}
            try:
                ok = bool(orig(key, st, ctx, w))
            except BaseException as exc:           # noqa: BLE001 —— 记录后原样抛出（runner 自己也会兜）
                ok, err = False, f"{type(exc).__name__}: {str(exc)[:120]}"
            dt = time.perf_counter() - t0
            rec = self.stat.setdefault(key, {"call": 0, "fail": 0, "skip": 0, "sec": 0.0,
                                             "bad": "", "info": "", "layers": []})
            li = int(ctx.get("idx", -1))
            rec["call"] += 1
            rec["sec"] += dt
            if li not in rec["layers"]:
                rec["layers"].append(li)
            if not ok:
                rec["skip"] += 1
                if err and not rec["bad"]:
                    rec["bad"] = err
            if err:
                rec["fail"] += 1
                if not rec["bad"]:
                    rec["bad"] = err
            if ok and not rec["bad"]:
                bad, info = _health(st, hold)
                if bad:
                    rec["bad"] = bad
                if info:
                    rec["info"] = info
            if key not in self.order:
                self.order.append(key)
            return ok
        return wrapper


def _health(st, hold) -> tuple[str, str]:
    """看几个主输出有没有 NaN/Inf（只查前 HEALTH_ROWS 行，代价可控）。

    注意：**不能**用「数值量级」判失败 —— 合成权重是随机的，逐层放大到 1e6~1e13 是正常现象；
    这里只报告量级（info），失败判据只有 NaN/Inf（以及算子抛异常，那在上层）。
    """
    try:
        import torch
        amax, who = 0.0, ""
        for name in HEALTH_KEYS:
            v = st.get(name)
            if v is None:
                v = hold.get(name)
            if not isinstance(v, torch.Tensor) or not v.dtype.is_floating_point:
                continue
            t = v[:HEALTH_ROWS] if v.dim() >= 1 and v.shape[0] > HEALTH_ROWS else v
            t = t.float()
            if not torch.isfinite(t).all():
                return f"{name} 出现 NaN/Inf", ""
            am = float(t.abs().max())
            if am > amax:
                amax, who = am, name
        return "", (f"{who} absmax≈{amax:.2e}（合成权重下量级仅供参考）" if amax else "")
    except Exception:                              # noqa: BLE001 —— 健康检查自己不许拖垮自检
        return "", ""


def tier1(a, tmp: str) -> dict:
    print("\n== Tier 1 · 逐算子真调用（合成权重）" + "=" * 30)
    import torch
    import exec_ops

    # 默认**不喂** input_ids：合成权重下 engram 走"零占位"路径（exec_engram._ids_of 的既定分支）。
    # 喂真 ids 的话哈希行号 ~3.8e8 会超出合成模式缩小的 engram 表（engram_table_rows）→ 必报越界，
    # 那是"合成表 + 真 ids"这个本来就不成立的组合，不是算子问题。真链路验证走 --parity（真权重）。
    ids_path = a.ids_file
    if ids_path:
        print(f"  ids：{ids_path}（注意：合成权重下 engram 表是缩小的，哈希行号可能越界）")
    else:
        print("  ids：不喂（合成模式：engram 走零占位路径）")

    hook = OpHook()
    orig = exec_ops.run_op
    exec_ops.run_op = hook(orig)

    cfg = os.path.join(ROOT, "configs/mini_single.yaml")
    # 一定要带 --layers：不带的话 runner 会走「贪心下一个 token」分支（要 tokenizer/权重目录）
    layers = 40 if a.full else a.layers
    argv = [os.path.join(ROOT, "runner.py"), cfg,
            "--seq", str(a.seq), "--device", a.device,
            "--progress", "off" if a.quiet else "on",
            "--layers", str(layers), "--out", os.path.join(tmp, "out")]
    if ids_path:
        argv += ["--ids-file", ids_path]
    print(f"  mini 真跑：{os.path.basename(cfg)} T={a.seq} layers={layers} device={a.device}")
    print("  " + "-" * 60)
    old_argv, old_cwd = sys.argv, os.getcwd()
    rc = 0
    try:
        os.chdir(ROOT)
        sys.argv = argv
        try:
            from runner import main as run_main
            run_main()
        except SystemExit as exc:                  # runner 有错会非零退出
            rc = int(exc.code or 0)
    except Exception as exc:                       # noqa: BLE001
        print(f"\n  ⛔ mini 跑挂了：{type(exc).__name__}: {str(exc)[:300]}")
        rc = 1
    finally:
        sys.argv, _ = old_argv, os.chdir(old_cwd)
        exec_ops.run_op = orig

    return {"rc": rc, "stat": hook.stat, "order": hook.order}


def extra_cases(device: str) -> dict:
    """合成模式走不到的算子，这里单独用最小输入覆盖。

    `tile_kernels.engram.engram_hash`：合成权重下 engram 走零占位路径，不会调它；
    这里直接算一次真 n-gram 哈希（用仓库自带的 cache_engram_layout.pt，不需要模型目录），
    校验形状/有限性/行号落在官方表尺寸内。
    """
    print("\n== 补充用例（合成模式走不到的算子）" + "=" * 26)
    out = {}
    try:
        import torch
        import engram_layout as el
        ids = torch.arange(512, dtype=torch.long)
        h = el.ngram_hash_ids(None, ids, device=device)
        ok_shape = h.dim() == 3 and h.shape[-1] == 24
        row_max = int(h.max())
        limit = int(el.L["num_embeddings"][0])
        ok = ok_shape and row_max < limit and not bool(torch.isnan(h.float()).any())
        print(f"  {'✅' if ok else '❌'} engram_hash（ngram_hash_ids）  形状 {tuple(h.shape)}  "
              f"行号 max={row_max} < 表行数 {limit}")
        out["engram_hash"] = {"call": 1, "fail": 0 if ok else 1, "skip": 0, "sec": 0.0,
                              "bad": "" if ok else "形状/行号越界", "info": "", "layers": [-1]}
    except Exception as exc:                       # noqa: BLE001
        print(f"  ❌ engram_hash（ngram_hash_ids）  {type(exc).__name__}: {str(exc)[:110]}")
        out["engram_hash"] = {"call": 1, "fail": 1, "skip": 0, "sec": 0.0,
                              "bad": f"{type(exc).__name__}: {str(exc)[:110]}", "info": "", "layers": [-1]}
    return out


def print_table(stat: dict) -> tuple[bool, list]:
    """逐算子表 + 覆盖对照（opgraph 注册表里没跑到的算子显式列出来）。"""
    import opgraph
    import exec_ops
    print("\n== 算子自检表 ==")
    print(f"  {'算子':<16}{'调用':>5}{'失败':>5}{'跳过':>5}{'累计耗时':>10}  状态  备注")
    ok_all = True
    lines = []
    for key in sorted(set(list(opgraph.OPS) + list(exec_ops.SKIP) + list(stat))):
        rec = stat.get(key)
        if rec is None:
            note = exec_ops.SKIP.get(key) or UNCOVERED_WHY.get(key) or "本次层段未覆盖（角色专属）"
            print(f"  {key:<16}{'—':>5}{'—':>5}{'—':>5}{'—':>10}  ⚪    {note[:46]}")
            lines.append({"op": key, "call": 0, "status": "uncovered", "note": note})
            continue
        status, note = "✅", rec["bad"] or rec.get("info", "")
        if rec["fail"]:
            status, note = "❌", rec["bad"] or "执行报错"
            ok_all = False
        elif rec["skip"] and not rec["call"] - rec["skip"]:
            status, note = "⚪", "不可执行（缺上下文/后端关）"
        elif rec["bad"]:
            status, note = "⚠️", rec["bad"]
            ok_all = False
        ls = rec["layers"]
        lstr = f"L{ls[0]}.." if len(ls) > 4 else "L" + ",".join(map(str, ls[:4]))
        print(f"  {key:<16}{rec['call']:>5}{rec['fail']:>5}{rec['skip']:>5}"
              f"{rec['sec']:>9.2f}s  {status}    {lstr} {note[:40]}")
        lines.append({"op": key, "call": rec["call"], "fail": rec["fail"],
                      "skip": rec["skip"], "sec": round(rec["sec"], 3),
                      "layers": ls, "status": status, "note": note})
    n_ok = sum(1 for x in lines if x["status"] == "✅")
    n_bad = sum(1 for x in lines if x["status"] in ("❌", "⚠️"))
    n_unc = sum(1 for x in lines if x["status"] in ("⚪", "uncovered"))
    print(f"\n  汇总：✅ {n_ok} 个算子真跑过且输出健康；⚠️/❌ {n_bad}；⚪ {n_unc}（未覆盖/不可执行，见备注）")
    print("  注：耗时是墙钟（未逐算子同步），只用于量级参考；数值正确性由 --parity 或对拍工具负责。")
    return ok_all, lines


def parity(a, tmp: str) -> dict:
    """可选：真权重跑一遍 + 与官方参考逐层对拍（需要真权重 + 参考 dump）。"""
    ref = a.parity
    if not ref or not os.path.isdir(ref):
        print(f"\n== Tier 2 · 对拍：跳过（--parity 给的目录不存在：{ref}）")
        return {"skipped": True}
    cfg = os.path.join(ROOT, "configs/mini_real.yaml")
    ours = os.path.join(tmp, "collect")
    print(f"\n== Tier 2 · 真权重对拍（参考 {ref}）" + "=" * 22)
    cmd = [sys.executable, "-u", os.path.join(ROOT, "runner.py"), cfg,
           "--seq", str(a.seq), "--device", a.device, "--progress", "off",
           "--collect-layers", ours]
    if a.ids_file:
        cmd += ["--ids-file", a.ids_file]
    if a.weights_path:
        cmd += ["--weights-path", a.weights_path]
    print("  " + " ".join(cmd[:6]) + " …")
    p = subprocess.run(cmd, cwd=ROOT)
    if p.returncode != 0:
        print(f"  ⛔ 真权重跑失败（rc={p.returncode}）；看上面日志")
        return {"ok": False, "rc": p.returncode}
    q = subprocess.run([sys.executable, os.path.join(ROOT, "tools/compare_layers.py"),
                        "--ref", ref, "--ours", ours], cwd=ROOT)
    return {"ok": q.returncode == 0, "rc": q.returncode, "ours": ours}


def _spec_check():
    """Tier 0.5：模块规格门禁（CPU-only）。返回 (ok, 输出尾部)。"""
    try:
        cp = subprocess.run(
            [sys.executable, os.path.join(ROOT, "tools", "check_module_spec.py"),
             "--configs", os.path.join(ROOT, "configs", "mini_ds41_flash.yaml")],
            capture_output=True, text=True, timeout=120)
        out = (cp.stdout or "") + (cp.stderr or "")
        ok = cp.returncode == 0
        print(f"\n== Tier 0.5 · 模块规格（CPU） {'✅ PASS' if ok else '❌ FAIL'} ==")
        for line in out.strip().splitlines()[-6:]:
            print("  " + line)
        return ok, out
    except Exception as e:                       # noqa: BLE001
        print(f"\n== Tier 0.5 · 模块规格 ❌ FAIL（{type(e).__name__}: {e}）==")
        return False, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=25,
                    help="跑前 N 层（默认 25=覆盖全部算子种类；--full 时忽略）")
    ap.add_argument("--full", action="store_true", help="跑满 40 层")
    ap.add_argument("--device", default="npu:0", help="逻辑卡（配合 ASCEND_RT_VISIBLE_DEVICES 用）")
    ap.add_argument("--card", type=int, default=None,
                    help="物理卡号：内部设 ASCEND_RT_VISIBLE_DEVICES=<N> 后再以 npu:0 跑（绕开逻辑卡号≥1 的坑）")
    ap.add_argument("--ids-file", default="", help="自备 input_ids（不给我就现造一份随机 ids）")
    ap.add_argument("--weights-path", default="", help="真权重目录（--parity 用）")
    ap.add_argument("--parity", default="", help="官方 collect 参考目录（可选，跑 Tier 2 对拍）")
    ap.add_argument("--json", default="", help="把结果落一份机读 JSON")
    ap.add_argument("--quiet", action="store_true", help="关掉阶段进度条（只留逐层行与表格）")
    ap.add_argument("--tier0-only", action="store_true", help="只查环境（不碰卡），用于排障")
    ap.add_argument("--keep", action="store_true", help="保留临时目录（默认删）")
    a = ap.parse_args()

    # 物理卡 → 只暴露那一张，再以 npu:0 跑（这台机器逻辑卡号≥1 会 aicore fault）
    if a.card is not None and os.environ.get("DS41M_SELFCHECK_CARD") != str(a.card):
        env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(a.card), DS41M_SELFCHECK_CARD=str(a.card))
        rest, i = [], 1                            # 去掉 --card N，--device 换成 npu:0
        while i < len(sys.argv):
            if sys.argv[i] == "--card":
                i += 2
                continue
            if sys.argv[i] == "--device" and i + 1 < len(sys.argv):
                rest += ["--device", "npu:0"]
                i += 2
                continue
            rest.append(sys.argv[i])
            i += 1
        if "--device" not in rest:
            rest += ["--device", "npu:0"]
        print(f"[selfcheck] 物理卡 {a.card} → ASCEND_RT_VISIBLE_DEVICES={a.card}，以 npu:0 重入")
        os.execve(sys.executable, [sys.executable, sys.argv[0]] + rest, env)

    tmp = tempfile.mkdtemp(prefix="ds41m_selfcheck_")
    result = {"seq": a.seq, "full": a.full, "device": a.device, "card": a.card}
    try:
        t0 = tier0()
        result["tier0"] = {"ok": t0["ok"], "missing": t0.get("missing", [])}
        # Tier 0.5：模块规格门禁（CPU-only，~2s）——层序列/权重/接口/形状
        spec_ok, spec_out = _spec_check()
        result["spec"] = {"ok": spec_ok}
        if not spec_ok:
            print("\n" + spec_out)
            print("\n结论：❌ 模块规格门禁未过（tools/check_module_spec.py）")
            rc = 2
        if a.tier0_only:
            print(f"\n结论（--tier0-only）：{'✅ 环境就绪' if t0['ok'] else '❌ 环境缺件'}")
            rc = 0 if t0["ok"] else 2
        elif not t0["ok"]:
            print("\n结论：❌ Tier 0 未过 —— 按上面 ❌ 项处理（多半是环境没 source 或算子包缺失）")
            rc = 2
        elif not spec_ok:
            pass                    # 上面已打印模块规格失败
        else:
            t1 = tier1(a, tmp)
            result["run_rc"] = t1["rc"]
            t1["stat"].update(extra_cases(a.device))
            ok1, lines = print_table(t1["stat"])
            result["ops"] = lines
            rc = 0 if ok1 and t1["rc"] in (0,) else 1
            if a.parity:
                result["parity"] = parity(a, tmp)
            print("\n结论：" + ("✅ 算子自检通过（所有跑到的算子都成功且输出健康）" if rc == 0
                              else f"❌ 未通过（mini rc={t1['rc']}，见表里 ❌/⚠️ 项）"))
            print(f"  复现命令：python3.11 tools/selfcheck.py"
                  f"{' --full' if a.full else ''}{f' --card {a.card}' if a.card is not None else ''}")
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=1)
        print(f"  结果 JSON → {a.json}")
    sys.exit(rc)


if __name__ == "__main__":
    main()

#!/bin/bash
# 绑定直发验证矩阵：base（三个开关=0） vs combo（默认开）
#   逐点 final-row 对拍 + [r1] 稳态；日志/结果全在 $OUT 下
# 用法：bash tools/run_bind_matrix.sh <OUTDIR> [点列表...]
set -u
OUT=${1:?OUTDIR}; shift
if [ $# -gt 0 ]; then PTS=("$@"); else PTS=(
  "8k@0.90" "8k@0.97" "16k@0.97" "32k@0.97"
  "64k@0.60" "64k@0.97" "128k@0.97"
  "256k@0.60" "256k@0.90" "256k@0.97" ); fi
cd /mnt/xds/sfs/l00972766/ds41_mini || exit 2
source env_pack.sh >/dev/null 2>&1
export TMPDIR=${DS41M_TMPDIR:-/mnt/xds/sfs/l00972766/tmp}
mkdir -p "$OUT"; echo "OUT=$OUT pts=${#PTS[@]}"
for pt in "${PTS[@]}"; do
  tag=$(echo "$pt" | tr '@' '_' | tr '.' 'p')
  mkdir -p "$OUT/$tag"
  for arm in base combo; do
    if [ "$arm" = base ]; then
      E="DS41M_MQ_OUTBUF=0 DS41M_MQ_QBUF=0 DS41M_BIND_FAST=0"
    else
      E="DS41M_MQ_OUTBUF=1 DS41M_MQ_QBUF=1 DS41M_BIND_FAST=1"
    fi
    bash /mnt/xds/sfs/l00972766/tools/wait_npu_idle.sh 60 >/dev/null 2>&1
    env $E DS41M_HOST_OPT=1 DS41M_DUMP_FINAL_ROW="$OUT/row_${tag}_${arm}.pt" \
      timeout 1200 bash tools/run_hit.sh -L "$pt" --noprof-only --iters 3 \
      -o "$OUT/$tag/$arm" > "$OUT/${tag}_${arm}.log" 2>&1
    echo "$tag $arm rc=$?"
    sleep 5
  done
done
python3.11 - "$OUT" "${PTS[@]}" <<'PY'
import sys, os, torch, hashlib
OUT = sys.argv[1]; pts = sys.argv[2:]
bad = 0
for pt in pts:
    tag = pt.replace("@", "_").replace(".", "p")
    res = {}
    for arm in ("base", "combo"):
        p = f"{OUT}/row_{tag}_{arm}.pt"
        if os.path.exists(p):
            t = torch.load(p).float()
            res[arm] = hashlib.sha1(t.numpy().tobytes()).hexdigest()[:12]
    ok = res.get("base") is not None and res.get("base") == res.get("combo")
    if not ok:
        bad += 1
    print(f"{pt:<10} base={res.get('base')} combo={res.get('combo')} "
          f"{'OK' if ok else '*** MISMATCH ***'}", flush=True)
print(f"\n合计 {len(pts)} 点，不一致 {bad} 点")
PY

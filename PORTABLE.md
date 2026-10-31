# ds41_mini 便携包

本包把 **runtime**（厂商算子 py 包/二进制、deps、tilelang、deep_gemm）放在
`ds41_mini/runtime/` 下；`env_pack.sh` 会优先用包内 runtime，不依赖
`/mnt/xds/sfs/...` 共享盘。

## 目标机要求

- NPU 驱动 + 可用 CANN（镜像自带 `/usr/local/Ascend/...`；便携包**默认带 CANN overlay**，`--no-cann` 可省 12 GB）
- python3.11
- torch / torch_npu / transformers（**默认已打进包里**；目标镜像自带时可用 `--no-torch` 省 2 GB）
- 真权重目录（可选；合成权重不需要）。用构建器「设置 → 真权重目录」或
  `--weights-path` 指定。
- 缺 CANN/torch/厂商算子（含 MegaMoE）时：见 `PORTABLE_TROUBLESHOOTING.md`。

## 首次使用

```bash
cd ds41_mini
source ./env_pack.sh          # 会打印用的是包内 runtime 还是镜像 CANN
python3.11 tools/selfcheck.py --card 0   # 算子自检
```

## 常用

```bash
# 合成权重多点（8 卡）
bash tools/run_hit.sh --points-spec '8192@0.96' --noprof-only --iters 2 --cards 0,1,2,3,4,5,6,7

# 单卡（跳 MoE/engram）
DS41M_SKIP_MOE=1 DS41M_SKIP_ENGRAM=1 \
bash tools/run_hit.sh --points-spec '8192@0.96' --noprof-only --iters 2 --cards 0

# 真权重（路径按目标机改）
DS41M_MOEMODE=deepgemm python3.11 launch_mini.py \
  --config configs/mini_real.yaml --world 8 --cards 0,1,2,3,4,5,6,7 \
  --weights-path /data/DeepSeek-V4.1-Flash-cann-mxfp8-mxfp4 --seq 512 --decode 4 --decode-mode inc
```

## 不包含

- 真权重（数百 GB ~ TB）
- parity 官方参考大目录（只带了 `parity/ids_16k.pt`）
- 共享盘上的历史日志/trace

## 注意

- `env_pack.sh` 末尾会 `unset TMPDIR`；要覆盖请在 `source` **之后**再 export。
- 包内 `runtime/tilelang` 已把符号链接解引用；可直接整体拷贝。

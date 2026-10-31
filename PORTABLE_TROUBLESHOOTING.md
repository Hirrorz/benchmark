# ds41_mini 便携包：缺件 / 缺算子排查与安装

> 先记住一个总原则：`source ./env_pack.sh` 的输出就是“它到底用了哪套东西”。
> 排障第一步永远是看这几行：
> ```
> [env_pack] python: ...
> [env_pack] CANN: ...
> [env_pack] 厂商算子: N 个 vendor 已接入
> [env_pack] deps: ...
> [env_pack] torch 站点包: ...
> ```
> 任何一项不对，先修它，再跑 `tools/selfcheck.py`。

## 0. 文件分类（包里什么在哪）

```
ds41_mini/
  *.py                     # 引擎核心模块（靠同目录 import；不要随意移动）
  env_pack.sh              # 环境引导（source 用）
  configs/                 # yaml 配置
  tools/                   # 入口脚本、selfcheck、构建器 HTML、便携包生成器
  docs/                    # 设计/使用文档
  parity/                  # 便携包只带 ids_16k.pt（官方 ref 太大不带）
  runtime/                 # 便携包运行时（env_pack 优先用它）
    python/                #   厂商算子 Python 包（cann_ops_transformer 等）
    ascend/.../custom/     #   厂商算子二进制（vendors/<name>）
    pysite_full/           #   torch / torch_npu / transformers
    deps/                  #   yaml / sympy / ...
    tilelang/              #   tilelang + TileKernels + JIT 缓存
    deepgemm-pkg/          #   deep_gemm
    bin/ninja              #   tilelang JIT 需要
    ascend_full/           #   CANN overlay（默认带；--no-cann 可省）
  logs/ _out/              # 运行时生成（不打包）
  cache_engram_*.pt        # engram 哈希/布局缓存（小文件，保留）
  PORTABLE.md              # 便携包使用说明
  PORTABLE_TROUBLESHOOTING.md  # 本文件
```

## 1. 缺 CANN

**现象**：`⚠️ 没找到 CANN`，或 selfcheck Tier 0 报 `libascendcl.so` 缺失。

**处理**：
```bash
# 方式 A：指向镜像自带
export DS41_CANN=/usr/local/Ascend/ascend-toolkit/latest
source ./env_pack.sh

# 方式 B：用便携包里带的 overlay（推荐，和验证环境一致）
export DS41_CANN=$PWD/runtime/ascend_full/usr/local/Ascend/cann-9.2.0
source ./env_pack.sh
```
重新生成便携包时可加 `--with-cann`（默认已带）或 `--no-cann` 省 12 GB。
CANN 必须与厂商算子二进制匹配；**镜像 CANN 不一定能跑通厂商算子**，
以 `selfcheck.py` 为准。

## 2. 缺 torch / torch_npu / transformers

**现象**：`ModuleNotFoundError: No module named 'torch'`。

```bash
source ./env_pack.sh          # 确认打印了 torch 站点包路径
python3.11 -c "import torch, torch_npu; print(torch.__version__)"
```

- 便携包默认带 `runtime/pysite_full`；如果打的是 `--no-torch` 包，目标镜像必须自带。
- 指定其它站点包：
  ```bash
  export DS41_TORCH_SITE=/path/to/site-packages
  source ./env_pack.sh
  ```

## 3. 缺厂商算子（含 MegaMoE）的安装

厂商算子分两部分，**必须同时存在**：

1. **Python 包**：`runtime/python/cann_ops_transformer/`
   - MegaMoE 的入口：`cann_ops_transformer/ops/mc2/mega_moe.py`
   - 注册表：`cann_ops_transformer/ops/__init__.py`（`"mega_moe": "mc2.mega_moe"`）
2. **二进制**：`runtime/ascend/usr/local/Ascend/custom/megamoe/vendors/custom_transformer/`
   - `op_api/lib/libcust_opapi.so`（ACL 接口）
   - `op_impl/ai_core/...`（算子实现）
   - `op_proto/lib/...`（算子原型）

### 从可用机器拷贝（最常用）

```bash
SRC=/mnt/xds/sfs/l00972766/ds41_runtime     # 可用机器上的 runtime
DST=$PWD/runtime

# 1) 二进制：整类厂商算子（megamoe 必装；建议 compressor/matmul/vector 一起）
mkdir -p "$DST/ascend/usr/local/Ascend"
cp -a "$SRC/ascend/usr/local/Ascend/custom/megamoe" "$DST/ascend/usr/local/Ascend/custom/"
cp -a "$SRC/ascend/usr/local/Ascend/custom/compressor" "$DST/ascend/usr/local/Ascend/custom/" 2>/dev/null || true
cp -a "$SRC/ascend/usr/local/Ascend/custom/matmul" "$DST/ascend/usr/local/Ascend/custom/" 2>/dev/null || true
cp -a "$SRC/ascend/usr/local/Ascend/custom/vector" "$DST/ascend/usr/local/Ascend/custom/" 2>/dev/null || true

# 2) Python 包（cann_ops_transformer 及其依赖）
cp -a "$SRC/python/." "$DST/python/"

# 3) 重新引导（env_pack 会自动扫 vendors 并拼 ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH）
source ./env_pack.sh
```

### 验证 MegaMoE

```bash
source ./env_pack.sh
python3.11 - <<'PY'
from cann_ops_transformer.ops import mega_moe
print("mega_moe import OK:", mega_moe)
PY
python3.11 tools/selfcheck.py --card 0 --layers 1
```

环境变量最终应包含：

```bash
echo "$ASCEND_CUSTOM_OPP_PATH"   # 应含 .../custom_transformer
echo "$LD_LIBRARY_PATH"          # 应含 .../op_api/lib
```

### 如果 MegaMoE 仍缺

- 确认 `runtime/ascend/.../custom/megamoe/vendors/custom_transformer/op_api/lib/libcust_opapi.so` 存在；
- 确认 `ASCEND_CUSTOM_OPP_PATH` 里是该 `custom_transformer` 目录（不是它的父目录）；
- 确认 CANN 与厂商算子版本匹配（优先用便携包自带的 `runtime/ascend_full`）；
- `selfcheck.py --tier0-only` 看环境缺件，`--layers 1 --json out.json` 看具体错误。

## 4. 缺 tilelang / TileKernels / ninja

**现象**：`topk_gate`、`engram_hash` 一类 tilelang 算子报
`TypeError: can only concatenate str (not "NoneType") to str`，或 `tilelang` import 失败。

```bash
source ./env_pack.sh
echo "$TILELANG_RT $TILELANG_CACHE_DIR $CANNBOTDSL_CACHE_DIR"
command -v ninja
```

- 便携包布局：`runtime/tilelang/tilelang-merged2/tilelang` + `TileKernels-v2`；
- `ninja` 在 `runtime/bin/ninja`（env_pack 会自动加 PATH）；
- JIT 需要 `gcc/g++/make` 和 python3.11 头文件（`python3.11-dev`）；
- 缓存坏了可删掉重编：
  ```bash
  rm -rf "$TILELANG_CACHE_DIR" "$CANNBOTDSL_CACHE_DIR"
  ```

## 5. 缺 deep_gemm

**现象**：`ModuleNotFoundError: No module named 'deep_gemm'`（deepgemm 后端 / 真权重）。

```bash
source ./env_pack.sh
python3.11 -c "import deep_gemm; print(deep_gemm.__file__)"
```

便携包自带 `runtime/deepgemm-pkg`；缺了就从可用机器拷：
```bash
cp -a /mnt/xds/sfs/l00972766/rt/deepgemm-pkg "$PWD/runtime/"
```

## 6. 缺 yaml / sympy 等纯 Python 依赖

```bash
python3.11 -c "import yaml, sympy, requests"
```
便携包自带 `runtime/deps`；缺了：
```bash
pip install pyyaml sympy requests
# 或
cp -a /root/py311deps "$PWD/runtime/deps"
```

## 7. selfcheck 结果怎么读

```bash
python3.11 tools/selfcheck.py --card 0              # Tier0 + Tier1（25 层，覆盖全部算子种类）
python3.11 tools/selfcheck.py --card 0 --full       # 40 层
python3.11 tools/selfcheck.py --card 0 --layers 1 --json out.json   # 快速定位
python3.11 tools/selfcheck.py --tier0-only          # 只查环境，不碰卡
```

- `✅` 真跑过且输出健康；
- `⚪` 本次层段没覆盖 / 角色专属 / 被融合算子取代，**不是失败**；
- `❌` 看 note 的异常类型；`--json` 里有逐算子记录；
- 常见误报：
  - `mega_moe KeyError: 'topk_idx'` / `topk_gate TypeError ... NoneType`
    先确认 CANN 是不是包内 `runtime/ascend_full`（镜像 CANN 可能不匹配）；
    仍失败就按第 3/4 节查厂商算子与 tilelang 缓存。

## 8. 真权重 / ids / 参考

- 真权重不在包里：用构建器「设置 → 真权重目录」或 `--weights-path`。
- 便携包只带 `parity/ids_16k.pt`；官方 `ref` 目录太大没带，`--ref` 指向目标机自己的参考。
- 只跑合成权重时两者都不需要。

## 9. 重新生成便携包（在**有共享盘**的机器上）

```bash
bash tools/make_portable_kit.sh -o /data/kit          # 默认带 torch + CANN
bash tools/make_portable_kit.sh --no-cann -o /data/kit  # 目标镜像 CANN 已验证时
bash tools/make_portable_kit.sh --no-torch -o /data/kit # 目标镜像自带 torch 时
```

生成后先在本机模拟无共享盘：

```bash
cd /data/kit/ds41_mini
env -u DS41_PACK_RUNTIME -u DS41_CANN -u DS41_DEPS -u TILELANG_RT \
  DS41_SHARE=/nonexistent bash -c 'source ./env_pack.sh; python3.11 tools/selfcheck.py --card 0 --layers 1'
```

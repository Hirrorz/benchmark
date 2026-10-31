# DS4.1-Flash 真权重清单（从 model.safetensors.index.json + 各分片头自动生成）

源：`/mnt/xds/sfs/DeepSeek-V4.1-Flash-cann-mxfp8-mxfp4`  ·  96085 个 tensor  ·  48 分片  ·  114 种命名模式  ·  权重已量化(mxfp8/mxfp4)

| # | 模式 | tensor 数 | dtype/shape |
|---|---|---|---|
| 1 | `layers.{N}.ffn.experts.{N}.w1.weight` | 15360 | I8 [2304, 2560] |
| 2 | `layers.{N}.ffn.experts.{N}.w1.scale` | 15360 | F8_E8M0 [2304, 160] |
| 3 | `layers.{N}.ffn.experts.{N}.w3.weight` | 15360 | I8 [2304, 2560] |
| 4 | `layers.{N}.ffn.experts.{N}.w3.scale` | 15360 | F8_E8M0 [2304, 160] |
| 5 | `layers.{N}.ffn.experts.{N}.w2.weight` | 15360 | I8 [5120, 1152] |
| 6 | `layers.{N}.ffn.experts.{N}.w2.scale` | 15360 | F8_E8M0 [5120, 72] |
| 7 | `mtp.{N}.ffn.experts.{N}.w1.weight` | 384 | I8 [2304, 2560] |
| 8 | `mtp.{N}.ffn.experts.{N}.w1.scale` | 384 | F8_E8M0 [2304, 160] |
| 9 | `mtp.{N}.ffn.experts.{N}.w3.weight` | 384 | I8 [2304, 2560] |
| 10 | `mtp.{N}.ffn.experts.{N}.w3.scale` | 384 | F8_E8M0 [2304, 160] |
| 11 | `mtp.{N}.ffn.experts.{N}.w2.weight` | 384 | I8 [5120, 1152] |
| 12 | `mtp.{N}.ffn.experts.{N}.w2.scale` | 384 | F8_E8M0 [5120, 72] |
| 13 | `layers.{N}.hc_attn_base` | 40 | F32 [24] |
| 14 | `layers.{N}.hc_ffn_base` | 40 | F32 [24] |
| 15 | `layers.{N}.hc_attn_fn` | 40 | F32 [24, 20480] |
| 16 | `layers.{N}.hc_attn_scale` | 40 | F32 [3] |
| 17 | `layers.{N}.hc_ffn_fn` | 40 | F32 [24, 20480] |
| 18 | `layers.{N}.hc_ffn_scale` | 40 | F32 [3] |
| 19 | `layers.{N}.attn.attn_sink` | 40 | F32 [64] |
| 20 | `layers.{N}.attn.wq_a.weight` | 40 | F8_E4M3 [1280, 5120] |
| 21 | `layers.{N}.attn.wq_a.scale` | 40 | F8_E8M0 [1280, 160] |
| 22 | `layers.{N}.attn.wq_b.weight` | 40 | F8_E4M3 [32768, 1280] |
| 23 | `layers.{N}.attn.wq_b.scale` | 40 | F8_E8M0 [32768, 40] |
| 24 | `layers.{N}.attn.q_norm.weight` | 40 | BF16 [1280] |
| 25 | `layers.{N}.attn.wo_a.weight` | 40 | F8_E4M3 [8192, 4096] |
| 26 | `layers.{N}.attn.wo_a.scale` | 40 | F8_E8M0 [8192, 128] |
| 27 | `layers.{N}.attn.wkv.weight` | 40 | F8_E4M3 [512, 5120] |
| 28 | `layers.{N}.attn.wkv.scale` | 40 | F8_E8M0 [512, 160] |
| 29 | `layers.{N}.attn.kv_norm.weight` | 40 | BF16 [512] |
| 30 | `layers.{N}.attn.wo_b.weight` | 40 | F8_E4M3 [5120, 8192] |
| 31 | `layers.{N}.attn.wo_b.scale` | 40 | F8_E8M0 [5120, 256] |
| 32 | `layers.{N}.attn_norm.weight` | 40 | BF16 [5120] |
| 33 | `layers.{N}.ffn_norm.weight` | 40 | BF16 [5120] |
| 34 | `layers.{N}.ffn.shared_experts.w1.weight` | 40 | F8_E4M3 [2304, 5120] |
| 35 | `layers.{N}.ffn.shared_experts.w1.scale` | 40 | F8_E8M0 [2304, 160] |
| 36 | `layers.{N}.ffn.shared_experts.w3.weight` | 40 | F8_E4M3 [2304, 5120] |
| 37 | `layers.{N}.ffn.shared_experts.w3.scale` | 40 | F8_E8M0 [2304, 160] |
| 38 | `layers.{N}.ffn.shared_experts.w2.weight` | 40 | F8_E4M3 [5120, 2304] |
| 39 | `layers.{N}.ffn.shared_experts.w2.scale` | 40 | F8_E8M0 [5120, 72] |
| 40 | `layers.{N}.ffn.gate.bias` | 40 | F32 [384] |
| 41 | `layers.{N}.ffn.gate.bias_vl` | 40 | F32 [384] |
| 42 | `layers.{N}.ffn.gate.weight` | 40 | BF16 [384, 5120] |
| 43 | `vision.blocks.{N}.norm1.weight` | 32 | BF16 [1024] |
| 44 | `vision.blocks.{N}.attn.wqkv.weight` | 32 | BF16 [3072, 1024] |
| 45 | `vision.blocks.{N}.attn.wqkv.bias` | 32 | BF16 [3072] |
| 46 | `vision.blocks.{N}.attn.wo.weight` | 32 | BF16 [1024, 1024] |
| 47 | `vision.blocks.{N}.attn.wo.bias` | 32 | BF16 [1024] |
| 48 | `vision.blocks.{N}.norm2.weight` | 32 | BF16 [1024] |
| 49 | `vision.blocks.{N}.mlp.w1.weight` | 32 | BF16 [5632, 1024] |
| 50 | `vision.blocks.{N}.mlp.w2.weight` | 32 | BF16 [1024, 2816] |
| 51 | `layers.{N}.attn.indexer.wq_b.weight` | 8 | F8_E4M3 [4096, 1280] |
| 52 | `layers.{N}.attn.indexer.wq_b.scale` | 8 | F8_E8M0 [4096, 40] |
| 53 | `layers.{N}.attn.indexer.weights_proj.weight` | 8 | BF16 [32, 5120] |
| 54 | `layers.{N}.attn.compressor.wkv.weight` | 4 | BF16 [512, 5120] |
| 55 | `layers.{N}.attn.compressor.norm.weight` | 4 | BF16 [512] |
| 56 | `layers.{N}.attn.indexer.wk.weight` | 4 | BF16 [128, 512] |
| 57 | `layers.{N}.attn.indexer.k_norm.weight` | 4 | BF16 [128] |
| 58 | `layers.{N}.attn.compressor.wgate.weight` | 3 | BF16 [512, 5120] |
| 59 | `mtp.{N}.hc_attn_base` | 3 | F32 [24] |
| 60 | `mtp.{N}.hc_ffn_base` | 3 | F32 [24] |
| 61 | `mtp.{N}.hc_attn_fn` | 3 | F32 [24, 20480] |
| 62 | `mtp.{N}.hc_attn_scale` | 3 | F32 [3] |
| 63 | `mtp.{N}.hc_ffn_fn` | 3 | F32 [24, 20480] |
| 64 | `mtp.{N}.hc_ffn_scale` | 3 | F32 [3] |
| 65 | `mtp.{N}.attn.attn_sink` | 3 | F32 [64] |
| 66 | `mtp.{N}.attn.wq_a.weight` | 3 | F8_E4M3 [1280, 5120] |
| 67 | `mtp.{N}.attn.wq_a.scale` | 3 | F8_E8M0 [1280, 160] |
| 68 | `mtp.{N}.attn.wq_b.weight` | 3 | F8_E4M3 [32768, 1280] |
| 69 | `mtp.{N}.attn.wq_b.scale` | 3 | F8_E8M0 [32768, 40] |
| 70 | `mtp.{N}.attn.q_norm.weight` | 3 | BF16 [1280] |
| 71 | `mtp.{N}.attn.wo_a.weight` | 3 | F8_E4M3 [8192, 4096] |
| 72 | `mtp.{N}.attn.wo_a.scale` | 3 | F8_E8M0 [8192, 128] |
| 73 | `mtp.{N}.attn.wkv.weight` | 3 | F8_E4M3 [512, 5120] |
| 74 | `mtp.{N}.attn.wkv.scale` | 3 | F8_E8M0 [512, 160] |
| 75 | `mtp.{N}.attn.kv_norm.weight` | 3 | BF16 [512] |
| 76 | `mtp.{N}.attn.wo_b.weight` | 3 | F8_E4M3 [5120, 8192] |
| 77 | `mtp.{N}.attn.wo_b.scale` | 3 | F8_E8M0 [5120, 256] |
| 78 | `mtp.{N}.attn_norm.weight` | 3 | BF16 [5120] |
| 79 | `mtp.{N}.ffn_norm.weight` | 3 | BF16 [5120] |
| 80 | `mtp.{N}.ffn.shared_experts.w1.weight` | 3 | F8_E4M3 [2304, 5120] |
| 81 | `mtp.{N}.ffn.shared_experts.w1.scale` | 3 | F8_E8M0 [2304, 160] |
| 82 | `mtp.{N}.ffn.shared_experts.w3.weight` | 3 | F8_E4M3 [2304, 5120] |
| 83 | `mtp.{N}.ffn.shared_experts.w3.scale` | 3 | F8_E8M0 [2304, 160] |
| 84 | `mtp.{N}.ffn.shared_experts.w2.weight` | 3 | F8_E4M3 [5120, 2304] |
| 85 | `mtp.{N}.ffn.shared_experts.w2.scale` | 3 | F8_E8M0 [5120, 72] |
| 86 | `mtp.{N}.ffn.gate.bias` | 3 | F32 [128] |
| 87 | `mtp.{N}.ffn.gate.bias_vl` | 3 | F32 [128] |
| 88 | `mtp.{N}.ffn.gate.weight` | 3 | BF16 [128, 5120] |
| 89 | `layers.{N}.engram.embed.weight` | 2 | F8_E4M3 [384016682, 256]<br>F8_E4M3 [384006168, 256] |
| 90 | `layers.{N}.engram.embed.scale` | 2 | F8_E8M0 [384016682, 8]<br>F8_E8M0 [384006168, 8] |
| 91 | `layers.{N}.engram.q_weight` | 2 | BF16 [4, 5120] |
| 92 | `layers.{N}.engram.k_weight` | 2 | BF16 [4, 5120] |
| 93 | `layers.{N}.engram.wkv.weight` | 2 | F8_E4M3 [25600, 6144] |
| 94 | `layers.{N}.engram.wkv.scale` | 2 | F8_E8M0 [25600, 192] |
| 95 | `vision.patch_embed.proj.weight` | 1 | BF16 [1024, 588] |
| 96 | `vision.patch_embed.proj.bias` | 1 | BF16 [1024] |
| 97 | `vision.norm.weight` | 1 | BF16 [1024] |
| 98 | `aligner.w1.weight` | 1 | BF16 [5120, 9216] |
| 99 | `aligner.w1.bias` | 1 | BF16 [5120] |
| 100 | `aligner.w2.weight` | 1 | BF16 [5120, 5120] |
| 101 | `aligner.w2.bias` | 1 | BF16 [5120] |
| 102 | `embed.weight` | 1 | BF16 [129280, 5120] |
| 103 | `image_start` | 1 | BF16 [5120] |
| 104 | `image_end` | 1 | BF16 [5120] |
| 105 | `image_newline` | 1 | BF16 [5120] |
| 106 | `norm.weight` | 1 | BF16 [5120] |
| 107 | `head.weight` | 1 | BF16 [129280, 5120] |
| 108 | `mtp.{N}.main_proj.weight` | 1 | F8_E4M3 [5120, 15360] |
| 109 | `mtp.{N}.main_proj.scale` | 1 | F8_E8M0 [5120, 480] |
| 110 | `mtp.{N}.main_norm.weight` | 1 | BF16 [5120] |
| 111 | `mtp.{N}.norm.weight` | 1 | BF16 [5120] |
| 112 | `mtp.{N}.markov_head.embed.weight` | 1 | BF16 [129280, 256] |
| 113 | `mtp.{N}.markov_head.head.weight` | 1 | BF16 [129280, 256] |
| 114 | `mtp.{N}.confidence_head.proj.weight` | 1 | BF16 [1, 5376] |

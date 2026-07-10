# FA3 Single-Prefill Kernel Profiling: custom_mask vs block_expanding

## 1. 背景

FA3 single-prefill kernel 支持两种 mask 模式：

- **`custom_mask`（MaskMode::kCustom）**：接收外部传入的 packed-bitmask，在 kernel 内逐元素解包检查
- **`block_expanding`（MaskMode::kBlockExpanding）**：kernel 通过 `get_num_kv_tiles` 精确计算有效 KV 范围，在 CTA_KV 粒度跳过不相关的 tile

目标：通过 intra-kernel profiling 量化两种模式的微观性能差异。

## 2. Profiler 设置

### 2.1 基础设施

FlashInfer 提供了内核内 profiling 框架，由两部分组成：

| 组件 | 路径 | 作用 |
|------|------|------|
| CUDA 宏层 | `include/flashinfer/profiler.cuh` | `PROFILER_EVENT_START/END/INSTANT` 宏，条件编译（`-DFLASHINFER_ENABLE_PROFILER`）|
| Python 解码层 | `flashinfer/profiler/__init__.py` | `export_to_perfetto_trace()` 解码 buffer 生成 Perfetto trace |
| 驱动脚本 | `profiler/single_prefill_mask.py` | 运行两种 mask 模式，解码 buffer，打印统计 + 生成 trace |

### 2.2 Profiling 事件设计

在 `mma_f16` 主循环（per-KV-tile 迭代）内部插桩 5 种事件：

| 事件 | 含义 | 覆盖范围 |
|------|------|----------|
| `gemm-qk` | QK 矩阵乘 | `gemm</*init=*/true>(tiled_mma_qk, ...)` |
| `mask-apply` | LogitsTransform + mask（custom/block/causal） | 逐元素 transform + `apply_custom_mask` 或 boundary check |
| `softmax-merge` | online-softmax 更新 + pipeline 释放 | `attention_updater.update` + `cute::copy` 到 P |
| `gemm-pv` | PV 矩阵乘 | `gemm</*init=*/false>(tiled_mma_pv, ...)` |
| `write-o` | 结果写回 global memory | `collective_epilogue.store` |

### 2.3 运行方式

```bash
pip install git+https://github.com/flashinfer-ai/tg4perfetto.git  # 一次性安装
pip install --no-build-isolation -e . -v                           # 开发安装
python profiler/single_prefill_mask.py                              # 运行 profiling
```

## 3. 代码改动

### 3.1 内核层（CUDA）

**`include/flashinfer/attention/hopper/prefill_sm90.cuh`**

- `#include "../../profiler.cuh"` 提供 profiler 宏
- Perfiler init block：读取 `mainloop_params.additional_params.profiler_buffer`，初始化 `ProfilerClosure`（同上 autor openspace 中的设计，使用 `num_groups=1` 的单 group 模式）
- 调用 `mma_f16` 时传递 `profiler_closure`
- epilogue 前后用 `PROFILER_EVENT_START/END(kWriteO)` 包裹

**`include/flashinfer/attention/hopper/mainloop_mma.cuh`**

- `#include "../../profiler.cuh"`
- 定义 `SinglePrefillProfileEventType` 枚举和 `ProfilerClosure` 结构体（放在此处而非 `prefill_sm90.cuh`，因为 `mainloop_mma.cuh` 被 `mainloop.cuh` 间接 include，需要自包含）
- 函数签名添加 `ProfilerClosure& profiler_closure` 参数（`#ifdef FLASHINFER_ENABLE_PROFILER` 条件编译）
- 主循环内对每个 KV-tile 插桩：

```cpp
// 每个 KV-tile 迭代:
PROFILER_EVENT_START(kGemmQK);
gemm</*init=*/true>(tiled_mma_qk, ...);
PROFILER_EVENT_END(kGemmQK);

attention_updater.rescale_o(tOrO);
consumer_wait(pipeline_v, ...);

PROFILER_EVENT_START(kGemmPV);
gemm</*init=*/false>(tiled_mma_pv, ...);
PROFILER_EVENT_END(kGemmPV);

PROFILER_EVENT_START(kMaskApply);
// WarpScheduler barrier + LogitsTransform + apply_custom_mask
PROFILER_EVENT_END(kMaskApply);

PROFILER_EVENT_START(kSoftmaxMerge);
// attention_updater.update + pipeline release + copy to P
PROFILER_EVENT_END(kSoftmaxMerge);
```

### 3.2 JIT 编译层（Python）

**`csrc/single_prefill_sm90_customize_config.jinja`**

- `{% if use_profiler %}` 条件 include `<flashinfer/profiler.cuh>`
- 条件声明 `profiler_buffer` 参数和 `AdditionalParams` 字段

**`flashinfer/jit/attention/modules.py`**
 
- `gen_single_prefill_module()`、`gen_customize_single_prefill_module()`、`get_single_prefill_uri()` 等函数添加 `use_profiler` 参数
- URI hash 包含 `profiler_{true,false}` 确保缓存隔离
- 编译时添加 `-DFLASHINFER_ENABLE_PROFILER`

### 3.3 API 层（Python）

**`flashinfer/prefill.py`**

- `_get_single_prefill_profiler_module()`：缓存 profiler 版 JIT 模块
- `single_prefill_with_kv_cache_profiler()`：custom_mask 的 profiler API

**`flashinfer/dllm/block_extend.py`**

- `get_block_extend_module_with_offset()`：添加 `use_profiler` 参数
- `block_extend_attention_with_offset_profiler()`：block_expanding 的 profiler API

### 3.4 驱动脚本

**`profiler/single_prefill_mask.py`**

- 为 three production 场景（sparse/half_dense/dense）运行两种 mask
- `decode_buffer()` 解析 profiler buffer 的 tag/timestamp 编码
- `summarize()` 按事件类型输出 per-CTA 统计
- `export_to_perfetto_trace()`（tg4perfetto）+ `export_to_pftrace()`（perfetto pip 包）双通道 trace 导出

## 4. 测试配置

```
GPU:  H100 (SM90a)
dtype: bf16
head_dim: 128, num_qo_heads=32, num_kv_heads=8
CTA_Q=64, CTA_KV=128, dllm_block_size=32

qo_len=64, kv_len=8192, 3 种 q_offset:
  sparse_8192:  q_offset=0      — Q 在最开头，仅 1 个有效 KV tile
  half_dense:    q_offset=4096  — Q 在中间，31 个有效 KV tile
  dense:         q_offset=8128  — Q 在末尾，62 个有效 KV tile
```

## 5. 结果与结论

### 5.1 Per-tile 事件拆分（half_dense 场景，q_offset=4096）

| 事件 | custom_mask (clocks) | block_expanding (clocks) | 倍数 |
|------|---------------------|-------------------------|------|
| gemm-qk | 1,120 | 2,304 | 0.5x |
| **mask-apply** | **11,808** | **96** | **123x** |
| softmax-merge | 864 | 928 | 0.93x |
| gemm-pv | 2,016 | 992 | 2.0x |
| **per-tile 合计** | **~15,808** | **~4,320** | **3.7x** |

### 5.2 总体效率对比

| 场景 | custom_mask per-CTA | block_expanding per-CTA | 倍数 | tiles (cm/be) |
|------|--------------------|------------------------|------|----------------|
| sparse_8192 (off=0) | 672,896 | — (1 tile, 主循环外) | 161x* | 42 / 1 |
| half_dense (off=4096) | 678,112 | 134,592 | **5.0x** | 42 / 31 |
| dense (off=8128) | 676,000 | 268,544 | **2.5x** | 42 / 62 |

*注：sparse 场景的 block_expanding 仅 1 个 tile 在 mask loop 外处理，循环内无事件，wall-clock 差距 161x。

### 5.3 差距分解（half_dense）

```
5.0x = 1.35x (tile 数减少: 42→31) × 3.7x (per-tile 加速)

per-tile 3.7x 几乎全部来自 mask-apply:
  - mask-apply 减少 123x (11,808→96)
  - 其他事件总和几乎持平 (~4,000→4,224)
```

### 5.4 根因分析

**custom_mask 的性能瓶颈**：

`mask-apply` 占每个 tile 的 75%。在此阶段，kernel 必须对每个 KV-tile 内的 128 个 (q, k) 元素逐一执行 `apply_custom_mask()`——从 global memory 的 packed-bitmask 中提取对应 bit 并进行比较。由于 custom_mask 不了解 block 级结构，必须扫描所有 42 个 KV tile（= kv_len/CTA_KV × masking_step_factor），无论实际稀疏度如何。

**block_expanding 的优势来源**：

1. **Tile 级跳过（1.35x）**：`get_num_kv_tiles` 利用 block 结构精确计算每个 Q-tile 的有效 KV 范围，跳过完全不可见的 tile
2. **Per-tile 加速（3.7x）**：跳过 tile 内部的逐元素 mask 检查，仅对跨 block 边界的 tile 做 col_limit 边界截断
3. **mask-apply 降维（123x）**：O(128) 逐元素 packed-bitmask 解包被 O(1) tile 边界检查替代

### 5.5 建议

1. **生产环境优先使用 block_expanding**：在所有 q_offset 下均显著优于 custom_mask
2. **如必须使用 custom_mask**：考虑后端预计算 tile 级有效性信息，或在 kernel 内缓存 packed-bitmask 的 tile 级摘要以减少逐元素检查开销
3. **Profiler 框架已就绪**：`profiler/single_prefill_mask.py` + `-DFLASHINFER_ENABLE_PROFILER` 可复用于其他 kernel 性能诊断

## 6. 文件清单

| 文件 | 变更 |
|------|------|
| `include/flashinfer/attention/hopper/prefill_sm90.cuh` | profiler init block, kWriteO 事件 |
| `include/flashinfer/attention/hopper/mainloop_mma.cuh` | 事件枚举+closure 定义，per-tile 5 事件插桩 |
| `csrc/single_prefill_sm90_customize_config.jinja` | 条件 include/参数声明 |
| `flashinfer/jit/attention/modules.py` | `use_profiler` 参数贯通 |
| `flashinfer/prefill.py` | profiler 版 API |
| `flashinfer/dllm/block_extend.py` | profiler 版 block_expanding API |
| `profiler/single_prefill_mask.py` | 驱动脚本（解码+统计+trace） |
| `trace_*.perfetto-trace` (6 files) | 三种场景 × 两种 mask 的 Perfetto trace |
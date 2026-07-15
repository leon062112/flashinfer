# BLOCK_EXPANDING Mask 在 Prefill Plan 阶段的感知优化

## 背景

`BLOCK_EXPANDING` mask 模式（`MaskMode::kBlockExpanding = 4`）已在 CUDA kernel 层实现，其 mask 规则为：

```
mask[q, k] = (q_global / B) >= (kv_global / B)
```

其中 `B = dllm_block_size`（2 的幂次，如 32/64/128/256）。同一 block 内及更早 block 的 KV 位置可见，更晚 block 的 KV 位置**完全不可见**。

FA2 和 FA3 kernel 层均已在运行时跳过不可见的 KV tiles（`block_expanding_num_iterations` / `get_num_kv_tiles`），但 **plan 阶段对此 mask 模式一无所知**，导致：

1. **FA2**：split-kv 调度器将所有 KV 视为可见，产生不必要的 chunk 和过大的 workspace 分配
2. **FA3**：MinHeap 负载均衡调度器的 cost function 严重高估

### 典型问题场景

```
Q 序列: 128 tokens, dllm_block_size = 256
  → Q 全部在 block 0
  → 可见 KV = ceil(128/256)*256 = 256 tokens = 16 pages

KV 序列: 32768 tokens = 2048 pages
  → 优化前: plan 认为需要处理全部 2048 pages
  → 优化后: plan 只需要处理 16 pages
  → 减少: 99.2%
```

## 代码改动

共修改 **8 个文件**，新增 **61 行**，删除 **20 行**。

### FA2 路径

#### 1. `include/flashinfer/attention/scheduler.cuh`

**`PrefillSplitQOKVIndptr`**：新增 `mask_mode`、`dllm_block_size` 参数。在 `effective_kv_len_arr` 计算中添加 BLOCK_EXPANDING 分支：

```cpp
// 优化前
effective_kv_len_arr[i] =
    std::min(window_left >= 0 ? ceil_div(window_left + cta_tile_q, page_size) : kv_len_arr[i],
             kv_len_arr[i]);

// 优化后
if (mask_mode == 4 && dllm_block_size > 0) {
    // BLOCK_EXPANDING: KV 可见范围 = 最后 Q token 所在 block 的末尾
    int64_t qo_len_tokens = packed_qo_len_arr[i] / gqa_group_size;
    int64_t q_last_block = (qo_len_tokens - 1) / dllm_block_size;
    int64_t max_kv_tokens = (q_last_block + 1) * dllm_block_size;
    effective_kv_len_arr[i] = std::min(ceil_div(max_kv_tokens, page_size), kv_len_arr[i]);
} else {
    effective_kv_len_arr[i] = std::min(
        window_left >= 0 ? ceil_div(window_left + cta_tile_q, page_size) : kv_len_arr[i],
        kv_len_arr[i]);
}
```

**`PrefillPlan`**：新增 `mask_mode`、`dllm_block_size` 参数，透传到 `PrefillSplitQOKVIndptr`。

#### 2. `csrc/batch_prefill.cu` + `csrc/batch_prefill_jit_binding.cu`

TVM-FFI 绑定函数 `BatchPrefillWithKVCachePlan` 新增 `mask_mode=0`、`dllm_block_size=0` 参数（带默认值保证向后兼容）。

#### 3. `flashinfer/prefill.py`

两个 `plan()` 方法（Paged / Ragged）新增 `dllm_block_size: Optional[int] = None` 参数。FA2/FA3 的 args 构建中追加：

```python
if self._backend in ("fa2", "fa3"):
    args.append(mask_mode if mask_mode is not None else 0)
    args.append(dllm_block_size if dllm_block_size is not None else 0)
```

#### 4. `flashinfer/dllm/batch_block_extend.py`

两个 wrapper 的 `plan()` 传递给 inner wrapper：

```python
self._inner_wrapper.plan(..., dllm_block_size=self._dllm_block_size)
```

### FA3 路径

#### 5. `include/flashinfer/attention/scheduler.cuh`

**`PrefillSM90Plan`**：新增 `mask_mode`、`dllm_block_size` 参数。MinHeap 调度循环的 `effective_kv_len` 计算：

```cpp
// 优化前: 只处理 causal
int effective_kv_len =
    causal ? packed_causal_kv_end(...) : kv_len;

// 优化后: BLOCK_EXPANDING 优先于 causal
int effective_kv_len;
if (mask_mode == 4 && dllm_block_size > 0) {
    int q_tile_end = std::min((q_tile_idx + 1) * cta_tile_q, qo_len);
    int q_last_block = (q_tile_end - 1) / dllm_block_size;
    effective_kv_len = std::min((q_last_block + 1) * dllm_block_size, kv_len);
} else if (causal) {
    effective_kv_len = packed_causal_kv_end(...);
} else {
    effective_kv_len = kv_len;
}
```

#### 6. `csrc/batch_prefill_sm90.cu` + `csrc/batch_prefill_fp8_sm90.cu` + `csrc/batch_prefill_sm90_jit_binding.cu`

`BatchPrefillWithKVCacheSM90Plan` 新增 `mask_mode=0, dllm_block_size=0`，透传到 `PrefillSM90Plan`。

#### 7. `flashinfer/prefill.py`

与 FA2 共用同一段 args 追加逻辑（`if self._backend in ("fa2", "fa3")`）。

### 兼容性

- 所有新增参数均有**默认值**（`mask_mode=0`, `dllm_block_size=0`），`pod.py`、`cascade.py` 等现有调用者无需任何修改
- SM90 FP8 路径同样覆盖

## 改动前后对比

### FA2: padded_batch_size / workspace 对比

| 场景 (dllm_block_size=256) | 优化前 padded_batch | 优化后 padded_batch | workspace 减少 |
|---|---|---|---|
| Block 0 Q (128), KV=32K | 16 | **8** | s_smem -50% |
| Block 0 Q (64), KV=128K | 18 | **4** | s_smem -78% |
| Block 0 Q (128), KV=128K | 16 | **8** | s_smem -50% |
| Block 0-1 Q (384), KV=128K | 12 | 12 | 不变（Q 已覆盖足够范围） |
| Full range Q (32K), KV=32K | 1024 | 1024 | 不变（无优化空间） |

### FA3: MinHeap cost function 对比

| 场景 | 优化前 cost | 优化后 cost | 降低 |
|---|---|---|---|
| Block 0 Q, 32K KV (1 tile) | 33,024 | 384 | **-99%** |
| Block 0 Q, 128K KV (1 tile) | 131,328 | 320 | **-100%** |
| Block 0-1 Q, 32K KV (3 tiles) | 99,072 | 1,536 | **-98%** |
| Block 0 Q, block=64, 32K KV | 33,024 | 384 | **-99%** |
| Block 0 Q, block=1024, 32K KV (4 tiles) | 132,096 | 2,304 | **-98%** |

## 收益分析

### 1. CUDA Graph / fixed-batch 场景（最主要收益）

FA2 中 `padded_batch_size` 决定了 CUDA Graph 的 grid 大小。在 block 0 Q + 长 KV 场景下：

- **优化前**：padded_batch_size = 18（需要生成 18 个 sub-batch CTA 来处理 KV 分块）
- **优化后**：padded_batch_size = 4（只需要 4 个）
- **无效空转减少 78%**

对于解码类服务（每步生成一个新 token），如果 KV 缓存已经很长（如 128K），但 Q 始终在 block 0，每次推理的 grid 中都减少了大量空转 CTA。

### 2. workspace memory 压力降低

| 场景 | 优化前 s_smem | 优化后 s_smem | 减少 |
|---|---|---|---|
| Block 0 Q, KV=128K | 35.7 MB | 8.0 MB | -78% |
| Block 0 Q, KV=32K | 35.9 MB | 16.8 MB | -53% |

多请求并发时，减少的 memory 可用于更大的 batch size 或更长的 context。

### 3. FA3 负载均衡质量提升

FA3 的 MinHeap 调度器将 Q tiles 按 `cost_function(qo_tiles, effective_kv_len)` 分配到各 SM。优化后：

- 每个 Q tile 的 cost 准确反映其实际 KV 可见范围
- 前面 block 的 Q tiles（可见 KV 少）不再与后面 block 的 Q tiles 被错误地视为同等成本
- 避免了"轻量 tile 抢占一个完整 SM，重量 tile 堆积在尾部"的不均衡

### 4. 无性能退化场景

- `dllm_block_size = 0` 时完全等同于未优化（无额外开销）
- Q 覆盖全部 KV 范围时（如 chunk 大小等于 KV 长度），effective_kv_len = kv_len，cost 不变
- 单 prefill 路径（`block_extend_attention_with_offset`）不走 plan 阶段，不受影响

## 测试结果

全部 5 个测试通过（FA2 + FA3），精度无退化：

```
tests/attention/test_dllm_blockwise_mask_attention.py:
  test_dllm_precision_vs_custom_mask_fa2 ........ PASSED (max_diff=0.0)
  test_heterogeneous_prefix_batch ............... PASSED (max_diff≈0.0005)
  test_cascade_current_chunk_batch .............. PASSED (max_diff=0.0)
  test_cascade_precision_alignment .............. PASSED (max_diff=0.0)
  test_sglang_vs_block_extend_cascade ........... PASSED (max_diff≈0.001)
```

## 后续工作

| 优先级 | 事项 | 说明 |
|---|---|---|
| P2 | FA3 offset 支持 | 当前仅支持 q_offset=0, kv_offset=0，cascade attention 场景需透传 per-batch offset 数组到 plan |
| P3 | TwoStageHolisticPlan | 类似逻辑扩展到 holistic scheduler |
| P3 | MLA Plan | 类似逻辑扩展到 MLA scheduler（`packed_causal_kv_end` → block_expanding 分支） |
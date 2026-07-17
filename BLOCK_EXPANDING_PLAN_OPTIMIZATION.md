# BLOCK_EXPANDING Mask 在 Prefill Plan 阶段的感知优化

## 1. 背景

### 1.1 BLOCK_EXPANDING Mask 规则

`BLOCK_EXPANDING` mask（`MaskMode::kBlockExpanding = 4`）专为 Diffusion LLM（DLLM）设计，其 mask 规则为：

```
mask[q, k] = (q_global / B) >= (kv_global / B)
```

其中 `B = dllm_block_size`（2 的幂，如 32/64/128/256）。同一 block 内的 KV 双向可见，更早 block 可见，更晚 block **完全不可见**。

FA2 和 FA3 kernel 层已在运行时通过 tile 级跳过（`block_expanding_num_iterations` / `get_num_kv_tiles`）处理了不可见 KV。但 **plan 阶段对此 mask 模式一无所知**，导致调度器严重高估计算量。

### 1.2 典型问题场景

```
Q 序列: 128 tokens, dllm_block_size = 256
  → Q 全部在 block 0
  → 可见 KV = 256 tokens = 16 pages

KV 序列: 131072 tokens = 8192 pages
  → 优化前: plan 认为需要处理全部 8192 pages
  → 优化后: plan 只需要处理 16 pages (256 tokens)
  → 工作量减少 99.8%
```

## 2. 改动前各后端的调度策略

### 2.1 FA2：Split-QO-KV 调度

FA2 使用 **split-kv** 策略，`PrefillPlan` 流程为：

**第1步 — 确定 `cta_tile_q`**：根据平均 packed_qo_len 和 head_dim 查找最佳 Q tile 大小（通常 128）。

**第2步 — 计算 `effective_kv_len_arr`**：对每个 batch item，决定调度器认为的"有效 KV 长度"（以 page 为单位）：
- 有 sliding window：`min((window_left + cta_tile_q) / page_size, kv_len)`
- 无 sliding window：直接用 `kv_len`（**全部 KV 视为可见**）

**第3步 — 二分搜索 `kv_chunk_size`**（`PrefillBinarySearchKVChunkSize`）：在 `[min_kv_chunk_size, max_kv_len]` 范围内搜索最小的 chunk size，使得 `total_chunks = Σ ceil_div(effective_kv_len[i], kv_chunk_size)` 不超过可用 CTA 数量。

**第4步 — 生成 (Q tile, KV chunk) 子批次对**：每个 request 产生 `num_q_tiles × num_kv_chunks` 个子批。`merge_indptr` 和 `o_indptr` 用于 kernel 执行后合并 split 结果。

**第5步 — 计算 `padded_batch_size`**：决定 CUDA Graph 的 grid 大小上限。

**核心问题**：`effective_kv_len` 被高估。Block 0 Q + 128K KV 场景下，`effective_kv_len` 被设为 8192 pages，二分搜索产生 19 个 KV chunk，`padded_batch_size = 19`。但实际只需 16 pages → 0 个 chunk（无需 split-kv）→ `padded_batch_size = 2`。

### 2.2 FA3：MinHeap 负载均衡调度

FA3 使用 **MinHeap** 贪心调度器，`PrefillSM90Plan` 流程为：

**第1步 — 按 KV 长度降序排序 batch items**：长序列优先调度，但 batch 内每个 item 仍然独立处理。

**第2步 — Q tiles 逆序遍历**：每个 request 的 Q tile 从 `num_qo_tiles - 1` 到 0 逆序处理（后面的 tile 可见 KV 更多、cost 更大，先分配可避免尾部堆积）。

**第3步 — MinHeap 贪心分配**：维持一个大小为 N_SM 的最小堆，key = 每个 SM 已分配的累计 cost。每次从堆 pop 出 cost 最小的 SM，将当前 Q tile 分配给它，累加 `cost(cta_tile_q, effective_kv_len)` 后插回堆。

**第4步 — cost function**：`cost(qo_len, kv_len) = 2 × qo_len + kv_len`。

**第5步 — `effective_kv_len` 计算**：
- Causal 场景：`packed_causal_kv_end(qo_len, kv_len, tile_idx)` — 只算当前 Q tile 最后一个位置能看到的 KV 范围
- Non-causal 场景：直接用 `kv_len`（**全部 KV 视为可见**）

**核心问题**：BLOCK_EXPANDING 场景下 `causal=False`，`effective_kv_len` 直接用 `kv_len`。这导致所有 Q tiles 的 cost 全部相同（= `cost(128, 131072) = 131328`），MinHeap 退化为**轮询（round-robin）分配**——无法区分轻量 tile（block 0）和重量 tile（block N），负载不均衡。

### 2.3 改动前的代码路径

```
BatchBlockExtendRaggedOffsetWrapper.plan()
  → self._inner_wrapper.plan(..., mask_mode=4)      # ⚠️ mask_mode 传入但 plan 不知如何利用
    → C-side args 构建 ← 无 dllm_block_size          # ⚠️ 缺失关键参数
      → PrefillPlan() / PrefillSM90Plan()
        → effective_kv_len = kv_len (全部 KV)
```

## 3. Block-Expanding 改动方案

### 3.1 设计原则

1. **只在 plan 阶段修改**（不改 kernel）：kernel 已有 tile 级 mask 感知，plan 修改纯粹是调度质量优化
2. **默认值保证向后兼容**：`mask_mode=0`、`dllm_block_size=0` → 行为等同于改动前
3. **FA2/FA3 各自独立实现**：两个后端的调度器算法不同，在各分支中独立计算 `effective_kv_len`

### 3.2 FA2 effective_kv_len 计算

在 `PrefillSplitQOKVIndptr` 中，对每个 batch item 基于最后 Q token 所在 block 计算可见 KV：

```
last_Q_block = (qo_len_tokens - 1) / dllm_block_size
visible_KV_pages = ((last_Q_block + 1) * dllm_block_size) / page_size
effective_kv_len = min(visible_KV_pages, kv_len)
```

这个值被后续 `PrefillBinarySearchKVChunkSize` 接收，决定 `kv_chunk_size` 和每个 batch item 的 `num_kv_chunks`。

### 3.3 FA3 effective_kv_len 计算

在 `PrefillSM90Plan` 的 MinHeap 分配循环中，对每个 Q tile 用 tile 末尾位置计算可见 KV：

```
tile_end = min((tile_idx + 1) * cta_tile_q, qo_len)
last_Q_block = (tile_end - 1) / dllm_block_size
effective_kv_len = min((last_Q_block + 1) * dllm_block_size, kv_len)
```

Q tiles 逆序分配 + 差异化 cost 使得 MinHeap 能真正实现负载均衡。

### 3.4 参数传递链路

```
Python BatchBlockExtendRaggedOffsetWrapper
  ├─ self._dllm_block_size (构造函数)
  └─ plan() → self._inner_wrapper.plan(..., dllm_block_size=self._dllm_block_size)
       └─ Python args 构建:
          args.append(mask_mode if mask_mode is not None else 0)
          args.append(dllm_block_size if dllm_block_size is not None else 0)
            └─ C TVM-FFI: BatchPrefillWithKVCachePlan(..., mask_mode, dllm_block_size)
                 └─ PrefillPlan(..., mask_mode, dllm_block_size)
                      └─ PrefillSplitQOKVIndptr(..., mask_mode, dllm_block_size)

同样 FA3:
  C TVM-FFI: BatchPrefillWithKVCacheSM90Plan(..., mask_mode, dllm_block_size)
    └─ PrefillSM90Plan(..., mask_mode, dllm_block_size)
```

## 4. 具体代码改动

### 4.1 文件清单

| 文件 | 改动 | 说明 |
|------|:---:|------|
| `include/flashinfer/attention/scheduler.cuh` | +37/-9 | FA2 + FA3 调度器核心逻辑 |
| `csrc/batch_prefill.cu` | +4/-3 | FA2 plan TVM-FFI 绑定 |
| `csrc/batch_prefill_jit_binding.cu` | +2/-1 | FA2 JIT 绑定声明 |
| `csrc/batch_prefill_sm90.cu` | +3/-2 | FA3 plan TVM-FFI 绑定 |
| `csrc/batch_prefill_fp8_sm90.cu` | +3/-2 | FA3 FP8 plan TVM-FFI 绑定 |
| `csrc/batch_prefill_sm90_jit_binding.cu` | +2/-1 | FA3 JIT 绑定声明 |
| `flashinfer/prefill.py` | +6/-0 | `plan()` 新增 `dllm_block_size` 参数 |
| `flashinfer/dllm/batch_block_extend.py` | +2/-0 | BBE wrapper 传递 `dllm_block_size` |

### 4.2 Python 层

**`flashinfer/prefill.py`** — Paged 和 Ragged wrapper 的 `plan()` 新增参数：

```python
def plan(self, ..., dllm_block_size: Optional[int] = None):
    ...
    # FA2/FA3 args 构建末尾追加 (位置: fixed_split_size 之后)
    if self._backend in ("fa2", "fa3"):
        args.append(mask_mode if mask_mode is not None else 0)      # 新增
        args.append(dllm_block_size if dllm_block_size is not None else 0)  # 新增
```

**`flashinfer/dllm/batch_block_extend.py`** — 两个 BBE wrapper 各自传递 `dllm_block_size`：

```python
# BatchBlockExtendPagedOffsetWrapper.plan() / BatchBlockExtendRaggedOffsetWrapper.plan():
self._inner_wrapper.plan(
    ..., mask_mode=MaskMode.BLOCK_EXPANDING.value,
    dllm_block_size=self._dllm_block_size,  # 新增：启用 plan 层优化
)
```

### 4.3 C++ 层

**FA2 — `PrefillSplitQOKVIndptr`（`include/flashinfer/attention/scheduler.cuh:494`）**：

```cpp
inline auto PrefillSplitQOKVIndptr(
    ..., bool disable_split_kv,
    int64_t mask_mode,      // 新增
    int64_t dllm_block_size // 新增
) {
    // ... effective_kv_len_arr 计算:
    if (mask_mode == 4 && dllm_block_size > 0) {
        int64_t qo_len_tokens = packed_qo_len_arr[i] / int64_t(gqa_group_size);
        int64_t q_last_block = (qo_len_tokens > 0)
            ? (qo_len_tokens - 1) / dllm_block_size : 0;
        int64_t max_kv_global_tokens = (q_last_block + 1) * dllm_block_size;
        int64_t block_kv_end_pages =
            ceil_div(max_kv_global_tokens, (int64_t)page_size);
        effective_kv_len_arr[i] = std::min(block_kv_end_pages, kv_len_arr[i]);
    } else {
        effective_kv_len_arr[i] = std::min(
            window_left >= 0
                ? ceil_div(window_left + cta_tile_q, page_size)
                : kv_len_arr[i],
            kv_len_arr[i]);
    }
}

// PrefillPlan 签名同样新增两参数，透传给 PrefillSplitQOKVIndptr
inline cudaError_t PrefillPlan(
    ..., int32_t fixed_split_size, bool disable_split_kv,
    int64_t mask_mode,       // 新增
    int64_t dllm_block_size, // 新增
    int64_t num_colocated_ctas, cudaStream_t stream);
```

**FA3 — `PrefillSM90Plan`（`include/flashinfer/attention/scheduler.cuh:890`）**：

```cpp
inline cudaError_t PrefillSM90Plan(
    ..., bool causal, bool enable_cuda_graph, uint32_t sizeof_dtype_o,
    int64_t mask_mode,       // 新增
    int64_t dllm_block_size, // 新增
    cudaStream_t stream) {
    // ... MinHeap 分配循环 (Q tiles 逆序):
    for (int qo_tile_idx = num_qo_tiles - 1; qo_tile_idx >= 0; --qo_tile_idx) {
        auto [cta_idx, accum_cost] = cta_cost_heap.pop();

        int effective_kv_len;
        if (mask_mode == 4 && dllm_block_size > 0) {
            int q_tile_end = std::min(
                (qo_tile_idx + 1) * cta_tile_q, qo_len);
            int q_last_block = (q_tile_end > 0)
                ? (q_tile_end - 1) / (int)dllm_block_size : 0;
            int max_visible_kv =
                (q_last_block + 1) * (int)dllm_block_size;
            effective_kv_len = std::min(max_visible_kv, kv_len);
        } else if (causal) {
            effective_kv_len = packed_causal_kv_end(
                qo_len, kv_len, qo_tile_idx, cta_tile_q, num_qo_tiles, 1);
        } else {
            effective_kv_len = kv_len;
        }
        cta_cost_heap.insert({cta_idx,
            accum_cost + cost_function(cta_tile_q, effective_kv_len)});
    }
}
```

### 4.4 兼容性

| 参数 | 默认值 | 含义 |
|------|:---:|------|
| `mask_mode` | 0 | `MaskMode.NON_CAUSAL`，不触发 BLOCK_EXPANDING 剪枝 |
| `dllm_block_size` | 0 | 不触发 BLOCK_EXPANDING 剪枝 |

`pod.py`、`cascade.py` 等现有调用者不传这两个参数 → 行为完全不变。SM90 FP8 路径同样覆盖。

## 5. 性能收益

### 5.1 FA2: split-kv 场景

测试配置: `dllm_block_size=256`, `heads=32/8`, `head_dim=128`

| 场景 | 优化前 | 优化后 | 收益 |
|------|:---:|:---:|------|
| **Block0 Q(32), KV=128K, 1req** | | | |
| padded_batch_size | 19 | **2** | **-89%** |
| workspace (s offset) | 39.8 MB | **4.2 MB** | **-89%** |
| kernel time | 0.043 ms | **0.019 ms** | **-56%** |
| split_kv | true | true | |
| **Block0 Q(128), KV=32K, 4reqs** | | | 不变 |
| padded_batch_size | 16 | 16 | 无需 split-kv |
| **Full Q(4096), KV=32K, 4reqs** | | | 不变 |
| padded_batch_size | 512 | 512 | 全量可见 |

**适用条件**：Q 在早期 block + KV 极长 → split-kv 被触发 → `padded_batch_size` 和 workspace 大幅降低。对于 CUDA Graph 部署场景，更小的 `padded_batch_size` 意味着更少的空转 CTA，grid 利用率提升。

### 5.2 FA3: MinHeap 负载均衡

测试配置: `dllm_block_size=256`, `heads=32/8`, `head_dim=128`, `KV=128K`, 单请求

| Q 长度 | Q tiles | 优化前 kernel | 优化后 kernel | 加速比 |
|--------|:---:|:----------:|:----------:|:---:|
| 256 (block0) | 2 | 0.018 ms | 0.018 ms | 1.00x |
| 512 (block0-1) | 4 | 0.045 ms | 0.045 ms | 1.00x |
| 1024 (block0-3) | 8 | 0.121 ms | **0.111 ms** | **1.09x** |
| 2048 (block0-7) | 16 | 0.412 ms | **0.330 ms** | **1.25x** |
| 4096 (block0-15) | 32 | 1.307 ms | **1.142 ms** | **1.14x** |
| 8192 (block0-31) | 64 | 4.709 ms | **4.133 ms** | **1.14x** |

**机制分析**：

```
优化前 (effective_kv_len = kv_len = 131072 对所有 tile):
  Block 0 tile:  cost(128, 131072) = 131328     ← 高估
  Block 1 tile:  cost(128, 131072) = 131328     ← 高估
  ...
  Block 31 tile: cost(128, 131072) = 131328     ← 高估
  → 所有 tile cost 相同 → MinHeap 退化为轮询 → 负载不均衡

优化后 (effective_kv_len 按 block 精确计算):
  Block 0 tile:  cost(128, 256)   =  512       ← 轻量
  Block 1 tile:  cost(128, 512)   =  768       ← 轻量
  Block 7 tile:  cost(128, 2048)  = 2304       ← 中等
  Block 15 tile: cost(128, 4096)  = 4352       ← 中等
  Block 31 tile: cost(128, 8192)  = 8448       ← 重量
  → MinHeap 分布: SM0={block0, block31}, SM1={block1, block30}, ...
  → 每个 SM 的组合成本接近 → 负载更均衡
```

- **Q ≤ 512（≤ 4 tiles）**：tile 太少，差异不明显
- **Q = 1024~8192（8~64 tiles）**：加速 **9%~25%**，tile 越多负载越重要
- 效果在 ~16 tiles 达到峰值 25%，更多 tiles 时趋于 14%（大 tile 量下轮询本身也有一定均衡效果）

### 5.3 无退化场景

| 场景 | 原因 |
|------|------|
| Q 覆盖全部 KV 范围 | `effective_kv_len = kv_len`，cost 不变 |
| `dllm_block_size = 0` | 走 else 分支，完全等价 |
| FA2 非 split-kv | `effective_kv_len` 只影响 `num_kv_chunks`，非 split-kv 下始终为 1 |
| FA3 tiles < 4 | MinHeap 分配粒度不足，无明显差异 |

## 6. 后续工作

| 优先级 | 事项 | 说明 |
|:---:|---|------|
| P2 | FA3 offset 支持 | 仅 `q_offset=0`/`kv_offset=0`，cascade attention 需传入 per-batch offset 数组 |
| P3 | HolisticPlan | 扩展到 TwoStageHolisticPlan 调度器 |
| P3 | MLA Plan | 扩展到 MLA scheduler（block_expanding 分支） |
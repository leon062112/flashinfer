"""Test plan BLOCK_EXPANDING mask awareness optimization performance.

Run BEFORE and AFTER the two feature commits to compare plan behavior.
"""
import math
import time
import torch
from flashinfer.dllm import BatchBlockExtendRaggedOffsetWrapper


def get_fa2_plan_info(wrapper):
    """Extract key plan metrics from FA2 inner_wrapper's plan_info.
    FA2 PrefillPlanInfo::ToVector():
      [0] padded_batch_size, [1] total_num_rows, [3] cta_tile_q,
      [5] qo_tile_indices_offset, [6] kv_tile_indices_offset,
      [10] v_offset, [11] s_offset, [14] split_kv
    """
    pi = wrapper._inner_wrapper._plan_info
    return {
        "padded_batch_size": pi[0],
        "total_num_rows": pi[1],
        "cta_tile_q": pi[3],
        "qo_tile_indices_len": pi[5],
        "kv_tile_indices_len": pi[6],
        "v_offset": pi[10],
        "s_offset": pi[11],
        "split_kv": bool(pi[14]),
    }


def get_fa3_plan_info(wrapper):
    """Extract key plan metrics from FA3 inner_wrapper's plan_info.
    FA3 PrefillPlanSM90Info::ToVector():
      [0] qo_tile_indices_offset, [1] qo_indptr_offset, [2] kv_indptr_offset,
      [3] qo_len_offset, [4] kv_len_offset, [5] head_indices_offset,
      [6] work_indptr_offset, [7] batch_indices_offset, [8] same_schedule_for_all_heads
    """
    pi = wrapper._inner_wrapper._plan_info
    return {
        "qo_tile_indices_offset": pi[0],
        "qo_indptr_offset": pi[1],
        "kv_indptr_offset": pi[2],
        "qo_len_offset": pi[3],
        "kv_len_offset": pi[4],
        "head_indices_offset": pi[5],
        "work_indptr_offset": pi[6],
        "batch_indices_offset": pi[7],
        "same_schedule_for_all_heads": bool(pi[8]),
    }


def test_scenario(desc, num_requests, qo_len_per_req, kv_len_per_req,
                  num_heads, num_kv_heads, head_dim, dllm_block_size,
                  q_offset, warmup=5, bench_iters=50, backend="fa3"):
    """Simulate one incremental prefill step using BBE wrapper."""
    device = torch.device("cuda:0")
    dtype = torch.float16

    q_list = [torch.randn(qo_len_per_req, num_heads, head_dim, dtype=dtype, device=device)
              for _ in range(num_requests)]
    k_list = [torch.randn(kv_len_per_req, num_kv_heads, head_dim, dtype=dtype, device=device)
              for _ in range(num_requests)]
    v_list = [torch.randn(kv_len_per_req, num_kv_heads, head_dim, dtype=dtype, device=device)
              for _ in range(num_requests)]

    q = torch.cat(q_list, dim=0)
    k = torch.cat(k_list, dim=0)
    v = torch.cat(v_list, dim=0)

    qo_indptr = torch.tensor([i * qo_len_per_req for i in range(num_requests + 1)],
                             dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([i * kv_len_per_req for i in range(num_requests + 1)],
                             dtype=torch.int32, device=device)
    q_offsets = torch.full((num_requests,), q_offset, dtype=torch.int32, device=device)

    ws_size = 256 * 1024 * 1024

    # Plan
    t0 = time.perf_counter()
    wrapper = BatchBlockExtendRaggedOffsetWrapper(
        torch.empty(ws_size, dtype=torch.uint8, device=device),
        kv_layout="NHD", dllm_block_size=dllm_block_size, backend=backend,
    )
    wrapper.plan(
        qo_indptr=qo_indptr, kv_indptr=kv_indptr,
        num_qo_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, q_data_type=dtype,
        q_offsets=q_offsets,
    )
    torch.cuda.synchronize()
    plan_time = (time.perf_counter() - t0) * 1000

    if backend == "fa2":
        info = get_fa2_plan_info(wrapper)
        key_metric = f"padded_bs={info['padded_batch_size']}, kv_tiles={info['kv_tile_indices_len']}, split_kv={info['split_kv']}"
    else:
        info = get_fa3_plan_info(wrapper)
        key_metric = f"qo_tiles={info['qo_tile_indices_offset']}, kv_len_o={info['kv_len_offset']}, work={info['work_indptr_offset']}"

    # Kernel
    for _ in range(warmup):
        wrapper.run(q, k, v)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(bench_iters):
        wrapper.run(q, k, v)
    torch.cuda.synchronize()
    kernel_time = (time.perf_counter() - t0) / bench_iters * 1000

    print(f"  [{desc:<22}, {backend}] plan={plan_time:>8.3f}ms, kernel={kernel_time:>8.3f}ms | {key_metric}")

    del wrapper
    torch.cuda.empty_cache()

    result = {
        "plan_time_ms": plan_time,
        "kernel_time_ms": kernel_time,
    }
    result.update(info)
    return result


def main():
    print("=" * 80)
    print("BLOCK_EXPANDING Plan Optimization — Performance Test")
    print("=" * 80)

    DLLM_BLOCK = 256
    H = 32
    KH = 8
    D = 128

    # Key scenarios: early-block Q with long KV (where optimization matters most)
    scenarios = [
        # (name,           n_reqs, qo_len, kv_len,      q_offset, backend)
        ("Block0, short KV",    4, 128,   1024,        0,       "fa2"),
        ("Block0, mid KV",      4, 128,   16384,       0,       "fa2"),
        ("Block0, long KV",     4, 128,   32768,       0,       "fa2"),
        ("Block0-1, long KV",   4, 512,   32768,       0,       "fa2"),
        ("Full Q, long KV",     4, 4096,  32768,       0,       "fa2"),
        ("Block2, mid KV",      4, 256,   16384,       512,     "fa2"),
        ("Block0, short KV",    4, 128,   1024,        0,       "fa3"),
        ("Block0, mid KV",      4, 128,   16384,       0,       "fa3"),
        ("Block0, long KV",     4, 128,   32768,       0,       "fa3"),
        ("Block0-1, long KV",   4, 512,   32768,       0,       "fa3"),
        ("Full Q, long KV",     4, 4096,  32768,       0,       "fa3"),
        ("Block2, mid KV",      4, 256,   16384,       512,     "fa3"),
    ]

    results = {}
    for name, nr, ql, kl, qoff, be in scenarios:
        try:
            results[(name, be)] = test_scenario(
                name, nr, ql, kl, H, KH, D, DLLM_BLOCK, qoff, backend=be,
            )
        except Exception as e:
            print(f"  [{name}, {be}] ERROR: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*100}")
    print(f"Summary (dllm_block={DLLM_BLOCK}, heads={H}/{KH}, dim={D})")
    print(f"{'='*100}")
    print(f"{'Scenario':<22} {'BE':>4} {'Plan(ms)':>10} {'Kernel(ms)':>11} {'FA2(padded_bs/kv_tiles)':>26} {'FA3(qo_tiles/kv_len/work)':>26}")
    print(f"{'-'*22} ---- ---------- ----------- {'-'*26} {'-'*26}")

    for (name, be), r in results.items():
        if be == "fa2":
            metric = f"bs={r['padded_batch_size']}, kvt={r['kv_tile_indices_len']}, split={r['split_kv']}"
        else:
            metric = f"qt={r['qo_tile_indices_offset']}, kvl={r['kv_len_offset']}, w={r['work_indptr_offset']}"
        print(f"{name:<22} {be:>4} {r['plan_time_ms']:>10.3f} {r['kernel_time_ms']:>11.3f} {metric:<26} {'':>26}")

    print(f"\nKey observations:")
    print(f"  'Block0' with dllm_block={DLLM_BLOCK}: Q (128 tokens < 256) all in block 0")
    print(f"  - Only the first block's KV ({DLLM_BLOCK} tokens) is visible")
    print(f"  - Optimized plan should show: FA2 padded_batch_size ~4 instead of much larger")
    print(f"  - 'Full Q' (4096 tokens, q_offset=0): spans 16 blocks, no optimization expected")

    return results


if __name__ == "__main__":
    main()
"""Plan optimization benefit benchmark: KV >> visible window scenarios.

Key insight from scheduler analysis:
  effective_kv_len = min(max_visible_kv, kv_len)
  where max_visible_kv = (q_last_block + 1) * dllm_block_size

The optimization benefits are proportional to: (kv_len - effective_kv) / kv_len
= fraction of KV that can be pruned.

Our previous benchmarks had kv_len ≈ visible_window (small ratio).
This benchmark creates scenarios with KV >> visible window.
"""

import json, math, time, os, sys
import torch
import numpy as np
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper


def bench_long_kv_single(
    tokens_q=128,
    tokens_kv=32768,
    block_size=32,
    heads=32,
    kv_heads=8,
    hdim=128,
    warmup=10,
    bench=100,
    backend="auto",
):
    """
    Single request: short Q, very long KV.
    With block_extend and q_offset=0:
      q_last_block = 127 // 32 = 3
      visible_kv = 4 * 32 = 128 tokens
      effective_kv = 128, kv_len = 32768
      pruning ratio = (32768 - 128) / 32768 = 99.6%

    Without plan optimization, the scheduler sees kv_len=32768 (2048 pages).
    With plan optimization, effective_kv=128 (8 pages).
    """
    device = torch.device("cuda:0")
    dtype = torch.float16
    sm = 1.0 / math.sqrt(hdim)

    print(f"\n{'=' * 80}")
    print(f"Long KV Single: Q={tokens_q}, KV={tokens_kv}, block={block_size}")
    print(
        f"visible_kv={(tokens_q // block_size + 1) * block_size}, pruning_ratio={1 - (tokens_q // block_size + 1) * block_size / tokens_kv:.1%}"
    )
    print(f"{'=' * 80}")

    q = torch.randn(tokens_q, heads, hdim, dtype=dtype, device=device)
    k = torch.randn(tokens_kv, kv_heads, hdim, dtype=dtype, device=device)
    v = torch.randn(tokens_kv, kv_heads, hdim, dtype=dtype, device=device)

    qo = torch.tensor([0, tokens_q], dtype=torch.int32, device=device)
    kv_ip = torch.tensor([0, tokens_kv], dtype=torch.int32, device=device)
    q_off = torch.zeros(1, dtype=torch.int32, device=device)

    # Plan
    w = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device),
        kv_layout="NHD",
        block_extend=True,
        block_size=block_size,
        backend=backend,
    )
    w.plan(
        qo_indptr=qo,
        kv_indptr=kv_ip,
        num_qo_heads=heads,
        num_kv_heads=kv_heads,
        head_dim_qk=hdim,
        q_data_type=dtype,
        sm_scale=sm,
        q_offsets=q_off,
    )

    # cuda event timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        w.run(q, k, v)
    torch.cuda.synchronize()

    start.record()
    for _ in range(bench):
        w.run(q, k, v)
    end.record()
    torch.cuda.synchronize()

    gpu_ms = start.elapsed_time(end) / bench
    print(f"  GPU kernel: {gpu_ms:.4f} ms")

    torch.cuda.empty_cache()
    return {
        "gpu_ms": gpu_ms,
        "q_len": tokens_q,
        "kv_len": tokens_kv,
        "visible_kv": (tokens_q // block_size + 1) * block_size,
        "pruning_ratio": 1 - (tokens_q // block_size + 1) * block_size / tokens_kv,
    }


def bench_long_kv_multi(
    num_req=64,
    tokens_q=128,
    tokens_kv=32768,
    block_size=32,
    heads=32,
    kv_heads=8,
    hdim=128,
    warmup=10,
    bench=100,
    backend="auto",
):
    """
    Multi-request: each request has short Q, very long KV.
    All requests have q_offset=0 (same starting point).
    Scheduler must handle 64 requests, each with kv_len=32768 pages.
    Plan optimization prunes each to effective_kv=128 tokens.
    """
    device = torch.device("cuda:0")
    dtype = torch.float16
    sm = 1.0 / math.sqrt(hdim)

    visible_kv = (tokens_q // block_size + 1) * block_size
    pruning = 1 - visible_kv / tokens_kv

    print(f"\n{'=' * 80}")
    print(
        f"Long KV Multi: {num_req}reqs, Q={tokens_q}, KV={tokens_kv}, block={block_size}"
    )
    print(f"visible_kv={visible_kv}, pruning_ratio={pruning:.1%}")
    print(f"{'=' * 80}")

    all_q = [
        torch.randn(tokens_q, heads, hdim, dtype=dtype, device=device)
        for _ in range(num_req)
    ]
    all_k = [
        torch.randn(tokens_kv, kv_heads, hdim, dtype=dtype, device=device)
        for _ in range(num_req)
    ]
    all_v = [
        torch.randn(tokens_kv, kv_heads, hdim, dtype=dtype, device=device)
        for _ in range(num_req)
    ]

    q_cat = torch.cat(all_q, dim=0)
    k_cat = torch.cat(all_k, dim=0)
    v_cat = torch.cat(all_v, dim=0)

    qo = torch.tensor(
        [i * tokens_q for i in range(num_req + 1)], dtype=torch.int32, device=device
    )
    kv_ip = torch.tensor(
        [i * tokens_kv for i in range(num_req + 1)], dtype=torch.int32, device=device
    )
    q_off = torch.zeros(num_req, dtype=torch.int32, device=device)

    w = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device),
        kv_layout="NHD",
        block_extend=True,
        block_size=block_size,
        backend=backend,
    )
    w.plan(
        qo_indptr=qo,
        kv_indptr=kv_ip,
        num_qo_heads=heads,
        num_kv_heads=kv_heads,
        head_dim_qk=hdim,
        q_data_type=dtype,
        sm_scale=sm,
        q_offsets=q_off,
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        w.run(q_cat, k_cat, v_cat)
    torch.cuda.synchronize()

    start.record()
    for _ in range(bench):
        w.run(q_cat, k_cat, v_cat)
    end.record()
    torch.cuda.synchronize()

    gpu_ms = start.elapsed_time(end) / bench
    print(f"  GPU kernel: {gpu_ms:.4f} ms")

    torch.cuda.empty_cache()
    return {
        "gpu_ms": gpu_ms,
        "num_req": num_req,
        "q_len": tokens_q,
        "kv_len": tokens_kv,
        "visible_kv": visible_kv,
        "pruning_ratio": pruning,
    }


def bench_heterogeneous_q_offsets(
    num_req=64,
    tokens_q=128,
    tokens_kv=32768,
    block_size=32,
    heads=32,
    kv_heads=8,
    hdim=128,
    warmup=10,
    bench=100,
    backend="auto",
):
    """
    Heterogeneous q_offsets: different requests at different steps.
    - Early step requests: q_offset=0, visible_kv=(128/32+1)*32=128
    - Late step requests: q_offset=16384, visible_kv=(16384+128)/32*32=16512
    Effective_kv varies per request → scheduler should differentiate.
    """
    device = torch.device("cuda:0")
    dtype = torch.float16
    sm = 1.0 / math.sqrt(hdim)

    # Mix of q_offsets: half at step 0, half at step 128
    offsets = []
    for i in range(num_req):
        if i < num_req // 2:
            offsets.append(0)  # early step: only first few blocks visible
        else:
            offsets.append(16384)  # late step: most KV visible

    print(f"\n{'=' * 80}")
    print(f"Heterogeneous offsets: {num_req}reqs, Q={tokens_q}, KV={tokens_kv}")
    print(
        f"  {num_req // 2} early-step (offset=0, visible_kv={(tokens_q // block_size + 1) * block_size})"
    )
    print(
        f"  {num_req // 2} late-step (offset=16384, visible_kv={(16384 + tokens_q) // block_size * block_size + block_size})"
    )
    print(f"{'=' * 80}")

    all_q = [
        torch.randn(tokens_q, heads, hdim, dtype=dtype, device=device)
        for _ in range(num_req)
    ]
    all_k = [
        torch.randn(tokens_kv, kv_heads, hdim, dtype=dtype, device=device)
        for _ in range(num_req)
    ]
    all_v = [
        torch.randn(tokens_kv, kv_heads, hdim, dtype=dtype, device=device)
        for _ in range(num_req)
    ]

    q_cat = torch.cat(all_q, dim=0)
    k_cat = torch.cat(all_k, dim=0)
    v_cat = torch.cat(all_v, dim=0)

    qo = torch.tensor(
        [i * tokens_q for i in range(num_req + 1)], dtype=torch.int32, device=device
    )
    kv_ip = torch.tensor(
        [i * tokens_kv for i in range(num_req + 1)], dtype=torch.int32, device=device
    )
    q_off = torch.tensor(offsets, dtype=torch.int32, device=device)

    w = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device),
        kv_layout="NHD",
        block_extend=True,
        block_size=block_size,
        backend=backend,
    )
    w.plan(
        qo_indptr=qo,
        kv_indptr=kv_ip,
        num_qo_heads=heads,
        num_kv_heads=kv_heads,
        head_dim_qk=hdim,
        q_data_type=dtype,
        sm_scale=sm,
        q_offsets=q_off,
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        w.run(q_cat, k_cat, v_cat)
    torch.cuda.synchronize()

    start.record()
    for _ in range(bench):
        w.run(q_cat, k_cat, v_cat)
    end.record()
    torch.cuda.synchronize()

    gpu_ms = start.elapsed_time(end) / bench
    print(f"  GPU kernel: {gpu_ms:.4f} ms")

    torch.cuda.empty_cache()
    return {
        "gpu_ms": gpu_ms,
        "num_req": num_req,
        "q_len": tokens_q,
        "kv_len": tokens_kv,
        "offsets": {
            "early": (0, (tokens_q // block_size + 1) * block_size),
            "late": (16384, (16384 + tokens_q) // block_size * block_size + block_size),
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", type=str, default="auto", choices=["auto", "fa2", "fa3"]
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Backend: {args.backend}")
    print(f"Strategy: KV >> visible window (maximize pruning ratio)")
    print()

    results = {"gpu": torch.cuda.get_device_name(0), "backend": args.backend}

    # Case 1: Single request, long KV
    print("\n" + "=" * 80)
    print("CASE 1: Long KV Single - 128 Q tokens, 32768 KV tokens (pruning=99.6%)")
    print("=" * 80)
    sys.stdout.flush()
    r = bench_long_kv_single(
        tokens_q=128, tokens_kv=32768, block_size=32, backend=args.backend
    )
    results["single_long_kv"] = r
    torch.cuda.empty_cache()

    # Case 2: Multi-request, long KV
    print("\n" + "=" * 80)
    print("CASE 2: Long KV Multi - 64 reqs × (128Q, 32768KV)")
    print("=" * 80)
    sys.stdout.flush()
    r = bench_long_kv_multi(
        num_req=64, tokens_q=128, tokens_kv=32768, block_size=32, backend=args.backend
    )
    results["multi_long_kv"] = r
    torch.cuda.empty_cache()

    # Case 3: Heterogeneous q_offsets
    print("\n" + "=" * 80)
    print("CASE 3: Heterogeneous offsets - mixed early/late steps")
    print("=" * 80)
    sys.stdout.flush()
    r = bench_heterogeneous_q_offsets(
        num_req=64, tokens_q=128, tokens_kv=32768, block_size=32, backend=args.backend
    )
    results["heterogeneous_offsets"] = r
    torch.cuda.empty_cache()

    # Case 4: Extreme - 1M KV tokens, tiny Q
    print("\n" + "=" * 80)
    print("CASE 4: Extreme KV - 32 Q tokens, 131072 KV tokens (pruning=99.97%)")
    print("=" * 80)
    sys.stdout.flush()
    try:
        r = bench_long_kv_single(
            tokens_q=32, tokens_kv=131072, block_size=32, backend=args.backend
        )
        results["extreme_kv"] = r
    except RuntimeError as e:
        print(f"  SKIPPED (OOM?): {e}")
        results["extreme_kv"] = {"error": str(e)}
    torch.cuda.empty_cache()

    # Save
    output = args.output or f"/root/code/flashinfer/plan_opt_bench_{args.backend}.json"
    json.dump(results, open(output, "w"), indent=2, default=str)
    print(f"\nSaved to {output}")

    print(f"\n{'=' * 80}")
    print(f"SUMMARY ({args.backend})")
    print(f"{'=' * 80}")
    for k, v in results.items():
        if isinstance(v, dict) and "gpu_ms" in v:
            print(f"  {k}: {v['gpu_ms']:.4f} ms")


if __name__ == "__main__":
    main()

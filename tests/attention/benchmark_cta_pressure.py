"""FA3 MinHeap CTA pressure: mixed requests with cost differentiation after pruning"""

import json, math, time, os, sys
import torch
import numpy as np
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper


def bench_cta_pressure(
    num_req=200,
    q_len=128,
    kv_len=16384,
    block_size=32,
    heads=32,
    kv_heads=8,
    hdim=128,
    warmup=10,
    bench=50,
    backend="fa3",
):
    """
    FA3 MinHeap: 200 requests, 100 early + 100 late.

    Early (q_offset=0): visible = (128/32+1)*32 = 160, pruning = 1-160/16384 = 99.0%
    Late (q_offset=8192): visible = (8192+128)/32*32+32 = 8352, pruning = 49.0%

    Cost without pruning: 2*128 + 16384 = 16640 (all same)
    Cost with pruning: early=2*128+160=416, late=2*128+8352=8608

    Work items: 200 * ceil_div(128*4, 128) = 200 * 4 = 800
    SMs: ~78 → ~10 items/SM (moderate CTA pressure)
    """
    device = torch.device("cuda:0")
    dtype = torch.float16
    sm = 1.0 / math.sqrt(hdim)

    half = num_req // 2
    late_offset = 8192

    visible_early = (q_len // block_size + 1) * block_size
    visible_late = min(((late_offset + q_len) // block_size + 1) * block_size, kv_len)

    total_qo_tiles = num_req * max(1, q_len * heads // kv_heads // 128)

    print(f"\n{'=' * 80}")
    print(
        f"FA3 CTA Pressure: {num_req} reqs, Q={q_len}, KV={kv_len}, block={block_size}"
    )
    print(
        f"  {half} early (q_offset=0, visible={visible_early}, pruning={1 - visible_early / kv_len:.1%})"
    )
    print(
        f"  {half} late (q_offset={late_offset}, visible={visible_late}, pruning={1 - visible_late / kv_len:.1%})"
    )
    print(f"  Per-tile cost w/o pruning: 2*128+{kv_len}={2 * 128 + kv_len} (uniform)")
    print(
        f"  Per-tile cost w/ pruning:  early=2*128+{visible_early}={2 * 128 + visible_early}, late=2*128+{visible_late}={2 * 128 + visible_late}"
    )
    print(f"  ~{total_qo_tiles} Q tiles vs ~78 SMs → ~{total_qo_tiles // 78} items/SM")
    print(f"{'=' * 80}")

    # Memory estimate and check
    mem_per_kv = num_req * kv_len * kv_heads * hdim * 2  # fp16 bytes
    mem_per_q = num_req * q_len * heads * hdim * 2
    total_mem = (mem_per_kv * 2 + mem_per_q) / (1024**3)
    print(f"  Memory: ~{total_mem:.1f} GB (KV×2 + Q)")
    sys.stdout.flush()

    all_q = [
        torch.randn(q_len, heads, hdim, dtype=dtype, device=device)
        for _ in range(num_req)
    ]
    all_k = [
        torch.randn(kv_len, kv_heads, hdim, dtype=dtype, device=device)
        for _ in range(num_req)
    ]
    all_v = [
        torch.randn(kv_len, kv_heads, hdim, dtype=dtype, device=device)
        for _ in range(num_req)
    ]

    q_cat = torch.cat(all_q, dim=0)
    k_cat = torch.cat(all_k, dim=0)
    v_cat = torch.cat(all_v, dim=0)

    qo = torch.tensor(
        [i * q_len for i in range(num_req + 1)], dtype=torch.int32, device=device
    )
    kv_ip = torch.tensor(
        [i * kv_len for i in range(num_req + 1)], dtype=torch.int32, device=device
    )
    q_off = torch.tensor(
        [0 if i < half else late_offset for i in range(num_req)],
        dtype=torch.int32,
        device=device,
    )

    ws = torch.empty(2048 * 1024 * 1024, dtype=torch.uint8, device=device)

    print("  Planning...")
    sys.stdout.flush()
    w = BatchPrefillWithRaggedKVCacheWrapper(
        ws, kv_layout="NHD", block_extend=True, block_size=block_size, backend=backend
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

    print("  Warmup...")
    sys.stdout.flush()
    for _ in range(warmup):
        w.run(q_cat, k_cat, v_cat)
    torch.cuda.synchronize()

    print("  Benchmark...")
    sys.stdout.flush()
    start.record()
    for _ in range(bench):
        w.run(q_cat, k_cat, v_cat)
    end.record()
    torch.cuda.synchronize()

    gpu_ms = start.elapsed_time(end) / bench
    print(f"  GPU kernel: {gpu_ms:.4f} ms")
    torch.cuda.empty_cache()

    return {"gpu_ms": gpu_ms, "num_req": num_req, "q_len": q_len, "kv_len": kv_len}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="fa3", choices=["fa3"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Backend: {args.backend}")
    torch.cuda.empty_cache()

    # JIT warmup
    print("JIT warmup...")
    sys.stdout.flush()
    _ = bench_cta_pressure(
        num_req=4, q_len=128, kv_len=1024, block_size=32, bench=3, backend=args.backend
    )
    torch.cuda.empty_cache()

    # 200 requests
    r = bench_cta_pressure(
        num_req=200, q_len=128, kv_len=16384, block_size=32, backend=args.backend
    )

    output = args.output or f"/root/code/flashinfer/cta_pressure_{args.backend}.json"
    json.dump(
        {"gpu": torch.cuda.get_device_name(0), "backend": args.backend, "result": r},
        open(output, "w"),
        indent=2,
        default=str,
    )
    print(f"\nResult: {r['gpu_ms']:.4f} ms")


if __name__ == "__main__":
    main()

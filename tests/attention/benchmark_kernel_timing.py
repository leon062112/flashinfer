"""Fine-grained GPU kernel timing: feature/block-extend vs plan-opt

Uses torch.cuda.Event to isolate GPU kernel time from CPU overhead,
measuring whether the plan optimization (effective_kv pruning) actually
reduces the GPU work in each step.

Key metrics per step:
- gpu_kernel_ms: pure GPU execution time (cuda event elapsed)
- nblocks: number of CTAs launched (from workspace_size)
"""

import json, math, time, os, sys
import torch
import numpy as np
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper


def bench_single_kernel_timing(
    tokens=8192,
    block_size=32,
    heads=32,
    kv_heads=8,
    hdim=128,
    warmup=3,
    bench=30,
    backend="auto",
):
    """Single-request: measure GPU kernel time per step with cuda events."""
    device = torch.device("cuda:0")
    dtype = torch.float16
    sm = 1.0 / math.sqrt(hdim)

    q_full = torch.randn(tokens, heads, hdim, dtype=dtype, device=device)
    k_full = torch.randn(tokens, kv_heads, hdim, dtype=dtype, device=device)
    v_full = torch.randn(tokens, kv_heads, hdim, dtype=dtype, device=device)

    def split(t, cs):
        return [t[i * cs : (i + 1) * cs] for i in range(t.shape[0] // cs)]

    results = {}

    for cs in [32, 64, 128, 256, 512]:
        if tokens % cs != 0:
            continue
        nsteps = tokens // cs

        qs = split(q_full, cs)
        k_cumul = [k_full[: (i + 1) * cs] for i in range(nsteps)]
        v_cumul = [v_full[: (i + 1) * cs] for i in range(nsteps)]

        qo = torch.tensor([0, cs], dtype=torch.int32, device=device)
        kv_ip = [
            torch.tensor([0, (i + 1) * cs], dtype=torch.int32, device=device)
            for i in range(nsteps)
        ]
        q_off = [
            torch.tensor([i * cs], dtype=torch.int32, device=device)
            for i in range(nsteps)
        ]

        # Pre-create wrappers
        wrappers = []
        for i in range(nsteps):
            w = BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device),
                kv_layout="NHD",
                block_extend=True,
                block_size=block_size,
                backend=backend,
            )
            w.plan(
                qo_indptr=qo,
                kv_indptr=kv_ip[i],
                num_qo_heads=heads,
                num_kv_heads=kv_heads,
                head_dim_qk=hdim,
                q_data_type=dtype,
                sm_scale=sm,
                q_offsets=q_off[i],
            )
            wrappers.append(w)

        out = torch.empty(cs, heads, hdim, dtype=dtype, device=device)

        # Per-step cuda event timing
        step_times = {i: [] for i in range(nsteps)}
        start_events = {i: torch.cuda.Event(enable_timing=True) for i in range(nsteps)}
        end_events = {i: torch.cuda.Event(enable_timing=True) for i in range(nsteps)}

        # Warmup
        for _ in range(warmup):
            for i in range(nsteps):
                out.copy_(wrappers[i].run(qs[i], k_cumul[i], v_cumul[i]))

        # Benchmark with cuda events
        for _ in range(bench):
            for i in range(nsteps):
                start_events[i].record()
                out.copy_(wrappers[i].run(qs[i], k_cumul[i], v_cumul[i]))
                end_events[i].record()
                torch.cuda.synchronize()
                step_times[i].append(start_events[i].elapsed_time(end_events[i]))

        # Summary per step
        print(f"\n  chunk={cs} steps={nsteps} backend={backend}")
        print(f"  {'step':>5} {'gpu_kernel_ms':>14} {'kv_len':>8}")
        for i in range(nsteps):
            t = np.mean(step_times[i])
            kv_len = (i + 1) * cs
            print(f"  {i:>5} {t:>14.4f} {kv_len:>8}")

        # Aggregate
        all_times = [np.mean(step_times[i]) for i in range(nsteps)]
        first_step = np.mean(step_times[0]) if nsteps > 0 else 0
        last_step = np.mean(step_times[nsteps - 1]) if nsteps > 0 else 0
        mid_step = np.mean(step_times[nsteps // 2]) if nsteps > 0 else 0

        results[f"chunk{cs}"] = {
            "chunk": cs,
            "steps": nsteps,
            "total_gpu_ms": float(np.sum(all_times)),
            "mean_gpu_ms": float(np.mean(all_times)),
            "first_step_ms": float(first_step),
            "mid_step_ms": float(mid_step),
            "last_step_ms": float(last_step),
            "step_times": [float(t) for t in all_times],
        }

        print(
            f"  Total GPU: {np.sum(all_times):.3f}ms, Mean: {np.mean(all_times):.4f}ms"
        )
        print(
            f"  First: {first_step:.4f}ms, Mid: {mid_step:.4f}ms, Last: {last_step:.4f}ms"
        )

        del wrappers
        torch.cuda.empty_cache()

    return results


def bench_multi_kernel_timing(
    num_req=256,
    tokens=512,
    block_size=32,
    heads=32,
    kv_heads=8,
    hdim=128,
    warmup=3,
    bench=30,
    backend="auto",
):
    """Multi-request: measure GPU kernel time per step."""
    device = torch.device("cuda:0")
    dtype = torch.float16
    sm = 1.0 / math.sqrt(hdim)

    all_q = [
        torch.randn(tokens, heads, hdim, dtype=dtype, device=device)
        for _ in range(num_req)
    ]
    all_k = [
        torch.randn(tokens, kv_heads, hdim, dtype=dtype, device=device)
        for _ in range(num_req)
    ]
    all_v = [
        torch.randn(tokens, kv_heads, hdim, dtype=dtype, device=device)
        for _ in range(num_req)
    ]

    def split(t, cs):
        return [t[i * cs : (i + 1) * cs] for i in range(t.shape[0] // cs)]

    results = {}

    for cs in [32, 64, 128, 256, 512]:
        if tokens % cs != 0:
            continue
        nsteps = tokens // cs

        qs_be = [split(q, cs) for q in all_q]
        q_bufs = [
            torch.cat([qs_be[r][i] for r in range(num_req)], dim=0)
            for i in range(nsteps)
        ]
        k_bufs, v_bufs = [], []
        for i in range(nsteps):
            kv_len = (i + 1) * cs
            k_bufs.append(torch.cat([all_k[r][:kv_len] for r in range(num_req)], dim=0))
            v_bufs.append(torch.cat([all_v[r][:kv_len] for r in range(num_req)], dim=0))

        qo_ip = torch.tensor(
            [i * cs for i in range(num_req + 1)], dtype=torch.int32, device=device
        )
        kv_ip_list, q_off_list = [], []
        for i in range(nsteps):
            kv_len = (i + 1) * cs
            kv_ip_list.append(
                torch.tensor(
                    [r * kv_len for r in range(num_req + 1)],
                    dtype=torch.int32,
                    device=device,
                )
            )
            q_off_list.append(
                torch.full((num_req,), i * cs, dtype=torch.int32, device=device)
            )

        wrappers = []
        for i in range(nsteps):
            w = BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(384 * 1024 * 1024, dtype=torch.uint8, device=device),
                kv_layout="NHD",
                block_extend=True,
                block_size=block_size,
                backend=backend,
            )
            w.plan(
                qo_indptr=qo_ip,
                kv_indptr=kv_ip_list[i],
                num_qo_heads=heads,
                num_kv_heads=kv_heads,
                head_dim_qk=hdim,
                q_data_type=dtype,
                sm_scale=sm,
                q_offsets=q_off_list[i],
            )
            wrappers.append(w)

        out = torch.empty(num_req * cs, heads, hdim, dtype=dtype, device=device)

        step_times = {i: [] for i in range(nsteps)}
        start_events = {i: torch.cuda.Event(enable_timing=True) for i in range(nsteps)}
        end_events = {i: torch.cuda.Event(enable_timing=True) for i in range(nsteps)}

        for _ in range(warmup):
            for i in range(nsteps):
                out.copy_(wrappers[i].run(q_bufs[i], k_bufs[i], v_bufs[i]))
        torch.cuda.synchronize()

        for _ in range(bench):
            for i in range(nsteps):
                start_events[i].record()
                out.copy_(wrappers[i].run(q_bufs[i], k_bufs[i], v_bufs[i]))
                end_events[i].record()
                torch.cuda.synchronize()
                step_times[i].append(start_events[i].elapsed_time(end_events[i]))

        print(f"\n  chunk={cs} steps={nsteps} backend={backend}")
        print(f"  {'step':>5} {'gpu_kernel_ms':>14} {'kv_len':>10}")
        for i in range(nsteps):
            t = np.mean(step_times[i])
            kv_len = (i + 1) * cs * num_req
            print(f"  {i:>5} {t:>14.4f} {kv_len:>10}")

        all_times = [np.mean(step_times[i]) for i in range(nsteps)]
        first_step = np.mean(step_times[0]) if nsteps > 0 else 0
        last_step = np.mean(step_times[nsteps - 1]) if nsteps > 0 else 0
        mid_step = np.mean(step_times[nsteps // 2]) if nsteps > 0 else 0

        results[f"chunk{cs}"] = {
            "chunk": cs,
            "steps": nsteps,
            "total_gpu_ms": float(np.sum(all_times)),
            "mean_gpu_ms": float(np.mean(all_times)),
            "first_step_ms": float(first_step),
            "mid_step_ms": float(mid_step),
            "last_step_ms": float(last_step),
            "step_times": [float(t) for t in all_times],
        }

        print(
            f"  Total GPU: {np.sum(all_times):.3f}ms, Mean: {np.mean(all_times):.4f}ms"
        )
        print(
            f"  First: {first_step:.4f}ms, Mid: {mid_step:.4f}ms, Last: {last_step:.4f}ms"
        )

        del wrappers
        torch.cuda.empty_cache()

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="multi", choices=["single", "multi", "both"]
    )
    parser.add_argument(
        "--backend", type=str, default="auto", choices=["auto", "fa2", "fa3"]
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Backend: {args.backend}")
    print(f"Method: torch.cuda.Event (GPU kernel time only)")
    print()

    branch = os.path.basename(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    )

    all_results = {
        "gpu": torch.cuda.get_device_name(0),
        "backend": args.backend,
        "branch": branch,
    }

    if args.mode in ("single", "both"):
        print("=" * 80)
        print(f"SINGLE-REQUEST (tokens=8192) GPU kernel timing | {args.backend}")
        print("=" * 80)
        r = bench_single_kernel_timing(tokens=8192, backend=args.backend)
        all_results["single"] = r
        torch.cuda.empty_cache()

    if args.mode in ("multi", "both"):
        print("=" * 80)
        print(f"MULTI-REQUEST (256×512) GPU kernel timing | {args.backend}")
        print("=" * 80)
        r = bench_multi_kernel_timing(num_req=256, tokens=512, backend=args.backend)
        all_results["multi"] = r
        torch.cuda.empty_cache()

    # Save
    output = (
        args.output
        or f"/root/code/flashinfer/kernel_timing_{args.backend}_{args.mode}.json"
    )
    json.dump(all_results, open(output, "w"), indent=2, default=str)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()

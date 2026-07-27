"""Non-CUDA-Graph benchmark: feature/block-extend vs worktree-block-extend-plan-opt

Tests BBE multi-request incremental prefill WITHOUT CUDA Graph,
to expose plan() overhead differences.
"""

import json, math, time, sys
import torch
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper


def bench_multi_no_cg(
    num_req=256,
    tokens=512,
    block_size=32,
    heads=32,
    kv_heads=8,
    hdim=128,
    warmup=10,
    bench=50,
    backend="auto",
):
    """BBE multi-request, NO CUDA Graph: measure plan+run per step"""
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
    chunk_sizes = [32, 64, 128, 256, 512]

    for cs in chunk_sizes:
        if tokens % cs != 0:
            continue
        nsteps = tokens // cs

        qs_be = [split(q, cs) for q in all_q]
        q_be_bufs = [
            torch.cat([qs_be[r][i] for r in range(num_req)], dim=0)
            for i in range(nsteps)
        ]

        k_be_bufs, v_be_bufs = [], []
        for i in range(nsteps):
            kv_len = (i + 1) * cs
            k_be_bufs.append(
                torch.cat([all_k[r][:kv_len] for r in range(num_req)], dim=0)
            )
            v_be_bufs.append(
                torch.cat([all_v[r][:kv_len] for r in range(num_req)], dim=0)
            )

        qo_indptr = torch.tensor(
            [i * cs for i in range(num_req + 1)], dtype=torch.int32, device=device
        )
        kv_indptr_list = []
        q_offsets_list = []
        for i in range(nsteps):
            kv_len = (i + 1) * cs
            kv_indptr_list.append(
                torch.tensor(
                    [r * kv_len for r in range(num_req + 1)],
                    dtype=torch.int32,
                    device=device,
                )
            )
            q_offsets_list.append(
                torch.full((num_req,), i * cs, dtype=torch.int32, device=device)
            )

        # --- Approach A: plan+run per step (measures plan overhead) ---
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        output = torch.empty(num_req * cs, heads, hdim, dtype=dtype, device=device)

        def run_plan_per_step():
            """plan() every step (worst case)"""
            for i in range(nsteps):
                w = BatchPrefillWithRaggedKVCacheWrapper(
                    workspace,
                    kv_layout="NHD",
                    block_extend=True,
                    block_size=block_size,
                    backend=backend,
                )
                w.plan(
                    qo_indptr=qo_indptr,
                    kv_indptr=kv_indptr_list[i],
                    num_qo_heads=heads,
                    num_kv_heads=kv_heads,
                    head_dim_qk=hdim,
                    q_data_type=dtype,
                    sm_scale=sm,
                    q_offsets=q_offsets_list[i],
                )
                output.copy_(w.run(q_be_bufs[i], k_be_bufs[i], v_be_bufs[i]))

        # --- Approach B: pre-plan all steps (amortizes plan) ---
        wrappers = []
        plan_start = time.perf_counter()
        for i in range(nsteps):
            w = BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device),
                kv_layout="NHD",
                block_extend=True,
                block_size=block_size,
                backend=backend,
            )
            w.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr_list[i],
                num_qo_heads=heads,
                num_kv_heads=kv_heads,
                head_dim_qk=hdim,
                q_data_type=dtype,
                sm_scale=sm,
                q_offsets=q_offsets_list[i],
            )
            wrappers.append(w)
        plan_cost = time.perf_counter() - plan_start

        def run_pre_planned():
            """plan() done upfront, only run() per step"""
            for i in range(nsteps):
                output.copy_(wrappers[i].run(q_be_bufs[i], k_be_bufs[i], v_be_bufs[i]))

        # Warmup
        for _ in range(warmup):
            run_plan_per_step()
        torch.cuda.synchronize()
        for _ in range(warmup):
            run_pre_planned()
        torch.cuda.synchronize()

        # Benchmark Approach A (plan per step)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(bench):
            run_plan_per_step()
        torch.cuda.synchronize()
        plan_per_step_ms = (time.perf_counter() - t0) / bench * 1000

        # Benchmark Approach B (pre-planned)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(bench):
            run_pre_planned()
        torch.cuda.synchronize()
        pre_planned_ms = (time.perf_counter() - t0) / bench * 1000

        results[f"chunk{cs}"] = {
            "chunk": cs,
            "steps": nsteps,
            "plan_per_step_ms": plan_per_step_ms,
            "pre_planned_ms": pre_planned_ms,
            "plan_overhead_ms": plan_per_step_ms - pre_planned_ms,
            "plan_only_ms": plan_cost * 1000 / nsteps,  # avg plan() cost per step
            "plan_overhead_pct": (plan_per_step_ms - pre_planned_ms)
            / pre_planned_ms
            * 100,
        }

        print(
            f"  chunk={cs:>4} steps={nsteps:>3} | plan_per_step={plan_per_step_ms:>8.3f}ms | "
            f"pre_planned={pre_planned_ms:>8.3f}ms | "
            f"plan_overhead={plan_per_step_ms - pre_planned_ms:>7.3f}ms ({results[f'chunk{cs}']['plan_overhead_pct']:>5.1f}%) | "
            f"plan_only={plan_cost * 1000 / nsteps:>7.3f}ms/step"
        )

        del wrappers
        torch.cuda.empty_cache()

    return results


def bench_single_no_cg(
    tokens=8192,
    block_size=32,
    heads=32,
    kv_heads=8,
    hdim=128,
    warmup=10,
    bench=50,
    backend="auto",
):
    """BBE single-request, NO CUDA Graph"""
    device = torch.device("cuda:0")
    dtype = torch.float16
    sm = 1.0 / math.sqrt(hdim)

    q_full = torch.randn(tokens, heads, hdim, dtype=dtype, device=device)
    k_full = torch.randn(tokens, kv_heads, hdim, dtype=dtype, device=device)
    v_full = torch.randn(tokens, kv_heads, hdim, dtype=dtype, device=device)

    def split(t, cs):
        return [t[i * cs : (i + 1) * cs] for i in range(t.shape[0] // cs)]

    results = {}
    chunk_sizes = [32, 64, 128, 256, 512]

    for cs in chunk_sizes:
        if tokens % cs != 0:
            continue
        nsteps = tokens // cs

        qs = split(q_full, cs)
        k_cumul = [k_full[: (i + 1) * cs] for i in range(nsteps)]
        v_cumul = [v_full[: (i + 1) * cs] for i in range(nsteps)]

        qo = torch.tensor([0, cs], dtype=torch.int32, device=device)
        kv_ip = []
        q_off = []
        for i in range(nsteps):
            kv_len = (i + 1) * cs
            kv_ip.append(torch.tensor([0, kv_len], dtype=torch.int32, device=device))
            q_off.append(torch.tensor([i * cs], dtype=torch.int32, device=device))

        out = torch.empty(cs, heads, hdim, dtype=dtype, device=device)

        # Plan per step
        def run_plan_per_step():
            for i in range(nsteps):
                w = BatchPrefillWithRaggedKVCacheWrapper(
                    torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
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
                out.copy_(w.run(qs[i], k_cumul[i], v_cumul[i]))

        # Pre-planned
        wrappers = []
        plan_start = time.perf_counter()
        for i in range(nsteps):
            w = BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
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
        plan_cost = time.perf_counter() - plan_start

        def run_pre_planned():
            for i in range(nsteps):
                out.copy_(wrappers[i].run(qs[i], k_cumul[i], v_cumul[i]))

        for _ in range(warmup):
            run_plan_per_step()
        torch.cuda.synchronize()
        for _ in range(warmup):
            run_pre_planned()
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(bench):
            run_plan_per_step()
        torch.cuda.synchronize()
        plan_per_step_ms = (time.perf_counter() - t0) / bench * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(bench):
            run_pre_planned()
        torch.cuda.synchronize()
        pre_planned_ms = (time.perf_counter() - t0) / bench * 1000

        results[f"chunk{cs}"] = {
            "chunk": cs,
            "steps": nsteps,
            "plan_per_step_ms": plan_per_step_ms,
            "pre_planned_ms": pre_planned_ms,
            "plan_overhead_ms": plan_per_step_ms - pre_planned_ms,
            "plan_only_ms": plan_cost * 1000 / nsteps,
            "plan_overhead_pct": (plan_per_step_ms - pre_planned_ms)
            / pre_planned_ms
            * 100,
        }

        print(
            f"  chunk={cs:>4} steps={nsteps:>3} | plan_per_step={plan_per_step_ms:>8.3f}ms | "
            f"pre_planned={pre_planned_ms:>8.3f}ms | "
            f"plan_overhead={plan_per_step_ms - pre_planned_ms:>7.3f}ms ({results[f'chunk{cs}']['plan_overhead_pct']:>5.1f}%)"
        )

        del wrappers
        torch.cuda.empty_cache()

    return results


def main():
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--mode", type=str, default="multi", choices=["single", "multi", "both"]
    )
    args = parser.parse_args()

    import flashinfer

    branch = os.path.basename(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"FlashInfer: {flashinfer.__version__}")
    print(f"Branch: {branch}")
    print(f"Mode: NO CUDA GRAPH")
    print()

    all_results = {
        "gpu": torch.cuda.get_device_name(0),
        "branch": branch,
        "flashinfer_version": flashinfer.__version__,
    }

    if args.mode in ("single", "both"):
        print("=" * 80)
        print("Single-Request (tokens=8192) NO CUDA GRAPH")
        print("=" * 80)
        r = bench_single_no_cg(tokens=8192)
        all_results["single"] = r
        torch.cuda.empty_cache()

    if args.mode in ("multi", "both"):
        print("=" * 80)
        print("Multi-Request (256×512) NO CUDA GRAPH")
        print("=" * 80)
        r = bench_multi_no_cg(num_req=256, tokens=512)
        all_results["multi"] = r
        torch.cuda.empty_cache()

    print(f"\n{'=' * 80}")
    print(f"SUMMARY (no CG, {branch})")
    print(f"{'=' * 80}")
    for mode in ["single", "multi"]:
        if mode not in all_results:
            continue
        print(f"\n{mode.upper()}:")
        print(
            f"  {'chunk':>5} {'steps':>5} {'plan/step':>10} {'pre-planned':>10} {'overhead':>10} {'overhead%':>10}"
        )
        for k in sorted(all_results[mode].keys()):
            r = all_results[mode][k]
            print(
                f"  {r['chunk']:>5} {r['steps']:>5} {r['plan_per_step_ms']:>10.3f} {r['pre_planned_ms']:>10.3f} {r['plan_overhead_ms']:>10.3f} {r['plan_overhead_pct']:>9.1f}%"
            )

    if args.output:
        json.dump(all_results, open(args.output, "w"), indent=2, default=str)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

"""Profiler-based fine-grained BBE analysis: feature/block-extend vs plan-opt

Uses include/flashinfer/profiler.cuh to record per-step GPU kernel timings
via the persistent prefill template's built-in profiler events.

Measures:
- kRunner1/kRunner2 compute time per step
- kReduction time per step
- Number of active CTAs per step
"""

import json, math, time, sys, os
import torch
import numpy as np
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
from flashinfer.profiler import decode_tag

# Profiler event indices (from persistent.cuh)
EVENT_NAMES = {0: "kRunner1", 1: "kRunner2", 2: "kReduction"}


def decode_profiler(profiler_buffer, event_names=None):
    """Decode profiler buffer into per-step timing."""
    if event_names is None:
        event_names = EVENT_NAMES

    buf = profiler_buffer.cpu()
    header = buf[:1].view(dtype=torch.int32)
    nblocks = int(header[0])
    ngroups = int(header[1])

    events = []
    for i in range(1, len(buf)):
        if buf[i] == 0:
            continue
        tag = int(buf[i : i + 1].view(dtype=torch.uint32)[0])
        ts = int(buf[i : i + 1].view(dtype=torch.uint32)[1])
        block_idx, group_idx, event_idx, event_type, sm_id = decode_tag(
            tag, nblocks, ngroups
        )
        events.append(
            {
                "block": block_idx,
                "group": group_idx,
                "event": event_idx,
                "event_name": event_names.get(event_idx, f"ev{event_idx}"),
                "event_type": event_type,  # 0=begin, 1=end
                "timestamp": ts,
                "sm": sm_id,
            }
        )
    return events, nblocks, ngroups


def compute_event_durations(events):
    """Compute durations by matching begin/end pairs."""
    durations = {}  # (event_idx, block) -> duration
    pending = {}  # (event_idx, block) -> begin_timestamp

    for e in events:
        key = (e["event"], e["block"])
        if e["event_type"] == 0:  # begin
            pending[key] = e
        elif e["event_type"] == 1:  # end
            if key in pending:
                dur = e["timestamp"] - pending[key]["timestamp"]
                durations[key] = dur
                del pending[key]

    return durations


def summarize_durations(durations):
    """Summarize per-event-type durations across all blocks."""
    by_event = {}
    for (evt, blk), dur in durations.items():
        if evt not in by_event:
            by_event[evt] = []
        by_event[evt].append(dur)
    return by_event


def bench_multi_profiler(
    num_req=256,
    tokens=512,
    block_size=32,
    heads=32,
    kv_heads=8,
    hdim=128,
    warmup=3,
    bench=5,
    backend="auto",
    chunk_sizes=None,
):
    """BBE multi-request with profiler enabled, measure per-step kernel breakdown."""

    if chunk_sizes is None:
        chunk_sizes = [32, 128]

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

    for cs in chunk_sizes:
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

        ws_sz = 384 * 1024 * 1024
        # Profiler buffer: header + blocks × groups per step
        # We create one wrapper per step, allocate profiler buffer per step
        profiler_buffers = []
        wrappers = []

        print(
            f"\n  chunk={cs} steps={nsteps} | Creating wrappers with use_profiler=True..."
        )
        sys.stdout.flush()

        for i in range(nsteps):
            w = BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(ws_sz, dtype=torch.uint8, device=device),
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
                use_profiler=True,
            )
            wrappers.append(w)

        # Allocate profiler buffer (generous size)
        max_blocks = 8192
        profiler_size = 1 + max_blocks * 2  # header + blocks×groups entries (groups=1)
        for i in range(nsteps):
            profiler_buffers.append(
                torch.zeros(profiler_size, dtype=torch.uint64, device=device)
            )

        out = torch.empty(num_req * cs, heads, hdim, dtype=dtype, device=device)

        # Collect profiler data for each step
        step_profiler_data = {i: [] for i in range(nsteps)}

        for it in range(bench):
            torch.cuda.synchronize()
            for i in range(nsteps):
                out.copy_(
                    wrappers[i].run(
                        q_bufs[i],
                        k_bufs[i],
                        v_bufs[i],
                        profiler_buffer=profiler_buffers[i],
                    )
                )
                torch.cuda.synchronize()

                # Decode
                events, nblocks, ngroups = decode_profiler(profiler_buffers[i].clone())
                durations = compute_event_durations(events)
                summary = summarize_durations(durations)

                step_data = {
                    "step": i,
                    "nblocks": nblocks,
                    "runner1_durs": summary.get(0, []),
                    "runner2_durs": summary.get(1, []),
                    "reduction_durs": summary.get(2, []),
                }
                step_profiler_data[i].append(step_data)

                # Reset buffer
                profiler_buffers[i].zero_()

            if it == 0:
                print(
                    f"    iter {it}: step 0 nblocks={step_profiler_data[0][0]['nblocks']}, "
                    f"step {nsteps - 1} nblocks={step_profiler_data[nsteps - 1][0]['nblocks']}"
                )

        # Aggregate across iterations
        step_summary = {}
        for i in range(nsteps):
            all_r1 = []
            all_rd = []
            all_nblocks = []
            for d in step_profiler_data[i]:
                all_r1.extend(d["runner1_durs"])
                all_rd.extend(d["reduction_durs"])
                all_nblocks.append(d["nblocks"])

            step_summary[i] = {
                "nblocks": int(np.mean(all_nblocks)),
                "runner1_total": float(np.sum(all_r1))
                / bench,  # total GPU cycles across all CTAs
                "runner1_mean": float(np.mean(all_r1)) if all_r1 else 0,
                "runner1_max": float(np.max(all_r1)) if all_r1 else 0,
                "reduction_total": float(np.sum(all_rd)) / bench,
                "reduction_mean": float(np.mean(all_rd)) if all_rd else 0,
            }

        results[f"chunk{cs}"] = {"steps": nsteps, "step_summary": step_summary}

        # Print summary
        print(
            f"\n    {'step':>5} {'nblocks':>7} {'r1_total(cyc)':>14} {'r1_mean(cyc)':>14} {'red_total(cyc)':>14}"
        )

        for i in range(nsteps):
            s = step_summary[i]
            print(
                f"    {i:>5} {s['nblocks']:>7} {s['runner1_total']:>14.0f} {s['runner1_mean']:>14.0f} {s['reduction_total']:>14.0f}"
            )

        del wrappers, profiler_buffers
        torch.cuda.empty_cache()

    return results


def bench_single_profiler(
    tokens=8192,
    block_size=32,
    heads=32,
    kv_heads=8,
    hdim=128,
    warmup=3,
    bench=5,
    backend="auto",
    chunk_sizes=None,
):
    """BBE single-request with profiler enabled."""

    if chunk_sizes is None:
        chunk_sizes = [32, 256]

    device = torch.device("cuda:0")
    dtype = torch.float16
    sm = 1.0 / math.sqrt(hdim)

    q_full = torch.randn(tokens, heads, hdim, dtype=dtype, device=device)
    k_full = torch.randn(tokens, kv_heads, hdim, dtype=dtype, device=device)
    v_full = torch.randn(tokens, kv_heads, hdim, dtype=dtype, device=device)

    def split(t, cs):
        return [t[i * cs : (i + 1) * cs] for i in range(t.shape[0] // cs)]

    results = {}

    for cs in chunk_sizes:
        if tokens % cs != 0:
            continue
        nsteps = tokens // cs

        qs = split(q_full, cs)
        k_cumul = [k_full[: (i + 1) * cs] for i in range(nsteps)]
        v_cumul = [v_full[: (i + 1) * cs] for i in range(nsteps)]

        qo = torch.tensor([0, cs], dtype=torch.int32, device=device)
        kv_ip, q_off = [], []
        for i in range(nsteps):
            kv_len = (i + 1) * cs
            kv_ip.append(torch.tensor([0, kv_len], dtype=torch.int32, device=device))
            q_off.append(torch.tensor([i * cs], dtype=torch.int32, device=device))

        ws_sz = 256 * 1024 * 1024
        wrappers = []
        profiler_bufs = []

        print(
            f"\n  chunk={cs} steps={nsteps} | Creating wrappers with use_profiler=True..."
        )
        sys.stdout.flush()

        for i in range(nsteps):
            w = BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(ws_sz, dtype=torch.uint8, device=device),
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
                use_profiler=True,
            )
            wrappers.append(w)

        max_blocks = 4096
        prof_sz = 1 + max_blocks * 2
        for i in range(nsteps):
            profiler_bufs.append(
                torch.zeros(prof_sz, dtype=torch.uint64, device=device)
            )

        out = torch.empty(cs, heads, hdim, dtype=dtype, device=device)
        step_data = {i: [] for i in range(nsteps)}

        for it in range(bench):
            torch.cuda.synchronize()
            for i in range(nsteps):
                out.copy_(
                    wrappers[i].run(
                        qs[i], k_cumul[i], v_cumul[i], profiler_buffer=profiler_bufs[i]
                    )
                )
                torch.cuda.synchronize()

                events, nblocks, ngroups = decode_profiler(profiler_bufs[i].clone())
                durations = compute_event_durations(events)
                summary = summarize_durations(durations)

                step_data[i].append(
                    {
                        "nblocks": nblocks,
                        "runner1_durs": summary.get(0, []),
                        "runner2_durs": summary.get(1, []),
                        "reduction_durs": summary.get(2, []),
                    }
                )
                profiler_bufs[i].zero_()

            if it == 0:
                mid = nsteps // 2
                last = nsteps - 1
                print(
                    f"    iter {it}: step 0 nblocks={step_data[0][0]['nblocks']}, "
                    f"step {mid} nblocks={step_data[mid][0]['nblocks']}, "
                    f"step {last} nblocks={step_data[last][0]['nblocks']}"
                )

        step_summary = {}
        for i in range(nsteps):
            all_r1, all_rd, all_nb = [], [], []
            for d in step_data[i]:
                all_r1.extend(d["runner1_durs"])
                all_rd.extend(d["reduction_durs"])
                all_nb.append(d["nblocks"])
            step_summary[i] = {
                "nblocks": int(np.mean(all_nb)),
                "runner1_total": float(np.sum(all_r1)) / bench,
                "runner1_mean": float(np.mean(all_r1)) if all_r1 else 0,
                "runner1_max": float(np.max(all_r1)) if all_r1 else 0,
                "reduction_total": float(np.sum(all_rd)) / bench,
            }

        results[f"chunk{cs}"] = {"steps": nsteps, "step_summary": step_summary}

        print(
            f"\n    {'step':>5} {'nblocks':>7} {'r1_total(cyc)':>14} {'r1_mean(cyc)':>14}"
        )
        for i in sorted(step_summary.keys()):
            s = step_summary[i]
            print(
                f"    {i:>5} {s['nblocks']:>7} {s['runner1_total']:>14.0f} {s['runner1_mean']:>14.0f}"
            )

        del wrappers, profiler_bufs
        torch.cuda.empty_cache()

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="single", choices=["single", "multi", "both"]
    )
    parser.add_argument(
        "--backend", type=str, default="auto", choices=["auto", "fa2", "fa3"]
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Backend: {args.backend}")
    print(f"Mode: GPU Profiler (FLASHINFER_ENABLE_PROFILER)")
    print()

    all_results = {"gpu": torch.cuda.get_device_name(0), "backend": args.backend}

    if args.mode in ("single", "both"):
        print("=" * 80)
        print(f"SINGLE-REQUEST (tokens=8192) Profiler | backend={args.backend}")
        print("=" * 80)
        sys.stdout.flush()
        r = bench_single_profiler(tokens=8192, backend=args.backend)
        all_results["single"] = r
        torch.cuda.empty_cache()

    if args.mode in ("multi", "both"):
        print("=" * 80)
        print(f"MULTI-REQUEST (256×512) Profiler | backend={args.backend}")
        print("=" * 80)
        sys.stdout.flush()
        r = bench_multi_profiler(num_req=256, tokens=512, backend=args.backend)
        all_results["multi"] = r
        torch.cuda.empty_cache()

    # Save
    output = (
        args.output
        or f"/root/code/flashinfer/profiler_results_{os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))}_{args.backend}_{args.mode}.json"
    )

    class NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
            return super().default(o)

    json.dump(all_results, open(output, "w"), indent=2, cls=NpEncoder)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()

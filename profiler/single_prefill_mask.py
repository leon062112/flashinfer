"""Profile FA3 single-prefill: block_expanding vs custom_mask via the in-kernel
profiler (`include/flashinfer/profiler.cuh`).

The FA3 single-prefill kernel is instrumented (behind
`FLASHINFER_ENABLE_PROFILER`) to emit, per Q-tile per CTA, two events bracketing
the `mma_f16` mainloop call (kMma=0) and the epilogue store (kEpilogue=1). This
script runs both masks under that profiler on the same problem and decodes the
buffer two ways:

  1. a per-CTA `kMma`-duration summary (median / p90 / max) printed as a table,
  2. a Perfetto trace dumped for visual inspection.

The hypothesis (from the wall-clock sweep): `custom_mask` pays the full kv_len
KV sweep on every CTA regardless of sparsity (kMma ~flat across CTAs), while
`block_expanding` shortens the per-Q-tile KV-tile count via `get_num_kv_tiles`
so most CTAs' kMma collapses (only the trailing dense block is expensive).
"""

import argparse
import math
from collections import defaultdict

import torch

import flashinfer
from flashinfer.dllm import block_extend_attention_with_offset_profiler
from flashinfer.prefill import single_prefill_with_kv_cache_profiler

try:
    from flashinfer.profiler import export_to_perfetto_trace
except Exception as _e:  # tg4perfetto (perfetto export) optional
    export_to_perfetto_trace = None
    print(f"[note] flashinfer perfetto export (tg4perfetto) unavailable: {_e}")

# Prefer the pure-python `perfetto` package (pip-installable, no protoc needed)
# to write a Perfetto trace viewable in ui.perfetto.dev.
_pftrace_ok = False
try:
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import (
        TracePacket,
        TrackEvent,
    )
    from perfetto.trace_builder.proto_builder import TraceProtoBuilder

    _pftrace_ok = True
except Exception as _e:
    print(f"[note] perfetto trace export (perfetto pkg) unavailable: {_e}")

# Event indices (SinglePrefillProfileEventType in mainloop_mma.cuh):
#   0: tma-load-k, 1: tma-load-v (producer), 2: gemm-qk, 3: mask-apply,
#   4: softmax-merge, 5: gemm-pv, 6: write-o (consumer)
EVENT_NAMES = ["tma-load-k", "tma-load-v", "gemm-qk", "mask-apply", "softmax-merge", "gemm-pv", "write-o"]


def export_to_pftrace(durations, num_blocks, file_name, sm_hint=None):
    """Write a Perfetto `.pftrace` viewable in https://ui.perfetto.dev.

    One track per CTA; on each track we emit per-tile event slices using the
    raw GPU-clock timestamps decoded from the profiler buffer (the cuts are
    implicit via BEGIN/END pairing). Timestamps are GPU clocks (32-bit
    ``%globaltimer_lo``), so the x-axis is monotonic-but-not-wallclock — fine for
    relative timing within a kernel.
    """
    if not _pftrace_ok:
        print(f"   (pftrace export skipped: perfetto pkg not available)")
        return
    b = TraceProtoBuilder()
    seq = 1
    for blk in range(num_blocks):
        p = b.add_packet()
        p.trusted_packet_sequence_id = seq
        td = p.track_descriptor
        td.uuid = 1000 + blk
        name = f"CTA{blk}"
        if sm_hint and blk in sm_hint:
            name += f"/SM{sm_hint[blk]}"
        td.name = name
    total_slices = 0
    for event_idx in range(len(EVENT_NAMES)):
        for blk, evs in durations[event_idx].items():
            for (begin, end) in evs:
                if begin == end:  # INSTANT event — use zero-length marker
                    pe = b.add_packet()
                    pe.trusted_packet_sequence_id = seq
                    pe.timestamp = begin
                    te = pe.track_event
                    te.track_uuid = 1000 + blk
                    te.type = TrackEvent.Type.TYPE_INSTANT
                    te.name = EVENT_NAMES[event_idx]
                else:
                    pb = b.add_packet()
                    pb.trusted_packet_sequence_id = seq
                    pb.timestamp = begin
                    te = pb.track_event
                    te.track_uuid = 1000 + blk
                    te.type = TrackEvent.Type.TYPE_SLICE_BEGIN
                    te.name = EVENT_NAMES[event_idx]
                    pe = b.add_packet()
                    pe.trusted_packet_sequence_id = seq
                    pe.timestamp = end
                    te2 = pe.track_event
                    te2.track_uuid = 1000 + blk
                    te2.type = TrackEvent.Type.TYPE_SLICE_END
                total_slices += 1
    with open(file_name, "wb") as f:
        f.write(b.serialize())
    n_ctas = sum(1 for d in durations.values() for b in d)
    print(f"   pftrace written: {file_name}  ({n_ctas} CTAs, {total_slices} slices)")


def make_block_mask(qo_len, kv_len, dllm_block_size, q_offset, device):
    q_pos = torch.arange(qo_len, device=device) + q_offset
    k_pos = torch.arange(kv_len, device=device)
    mask_2d = (
        (q_pos.unsqueeze(1) // dllm_block_size)
        >= (k_pos.unsqueeze(0) // dllm_block_size)
    ).to(torch.uint8)
    return mask_2d


def pack_mask(mask_2d):
    from flashinfer.quantization import packbits

    return packbits(mask_2d.contiguous().view(-1), bitorder="little")


def decode_buffer(profiler_buffer):
    """Decode the profiler buffer into per-CTA event spans (in GPU clocks).

    Buffer layout (profiler.cuh): entry[0] = (num_blocks, num_groups) header;
    subsequent entries are (tag, timestamp) encoded as low/high uint32 of a
    uint64. We pair BEGIN/END events per (block_idx, event_idx) to get spans.
    INSTANT events are recorded as zero-duration slices.

    Returns (num_blocks, spans, sm_hint) where spans[event_idx][block]
    is a list of (begin, end) GPU-clock timestamp tuples, and sm_hint[block] is
    the SM id (if present in the tag).
    """
    buf = profiler_buffer.cpu()
    num_blocks, num_groups = buf[:1].view(dtype=torch.int32).tolist()
    num_blocks, num_groups = int(num_blocks), int(num_groups)

    num_events = len(EVENT_NAMES)
    open_ts = {i: {} for i in range(num_events)}
    spans = {i: defaultdict(list) for i in range(num_events)}
    sm_hint = {}

    raw = buf.view(dtype=torch.uint32).reshape(-1, 2)  # rows of (tag, timestamp)
    for i in range(1, len(buf)):
        tag, timestamp = raw[i].tolist()
        if tag == 0 and timestamp == 0:
            continue
        event_type = tag & 0x3
        event_idx = (tag >> 2) & 0x3FF
        block_group_idx = (tag >> 12) & 0xFFF
        sm_id = (tag >> 24) & 0xFF
        if event_idx >= num_events:
            continue
        block_idx = block_group_idx // num_groups if num_groups else block_group_idx
        sm_hint[block_idx] = sm_id
        if event_type == 0:  # BEGIN
            open_ts[event_idx][block_idx] = timestamp
        elif event_type == 1 and block_idx in open_ts[event_idx]:  # END
            begin = open_ts[event_idx].pop(block_idx)
            spans[event_idx][block_idx].append((begin, timestamp))
    return num_blocks, spans, sm_hint


def durations_from_spans(spans):
    """Helper: convert (begin,end) spans to durations for the table summarizer."""
    return {ev: {b: [e - s for (s, e) in vs] for b, vs in d.items()}
            for ev, d in spans.items()}


def summarize(durations, label, num_blocks):
    """Print per-event-type duration stats and compute per-CTA total kMma clocks.
    Returns per-block total kMma clocks (gemm-qk + softmax-update + gemm-pv summed)."""
    lines = []
    for ev_idx, ev_name in enumerate(EVENT_NAMES):
        ev_d = durations[ev_idx]
        if not ev_d:
            lines.append(f"    {ev_name}: no events decoded")
            continue
        all_d = sorted(v for ds in ev_d.values() for v in ds)
        n = len(all_d)
        med = all_d[n // 2]
        p90 = all_d[int(n * 0.9)]
        mx = all_d[-1]
        lines.append(f"    {ev_name}: CTAs={len(ev_d)}/{num_blocks}  "
                     f"median={med}  p90={p90}  max={mx}  n={n}")
    if lines:
        print(f"  [{label}] per-event-type breakdown:")
        for l in lines:
            print(l)

    # Per-CTA total: sum gemm-qk + mask-apply + softmax-merge + gemm-pv per block.
    total_per_block = defaultdict(int)
    for ev_idx in range(4):  # gemm-qk, mask-apply, softmax-merge, gemm-pv
        for blk, ds in durations[ev_idx].items():
            total_per_block[blk] += sum(ds)
    all_total = sorted(total_per_block.values())
    if all_total:
        n = len(all_total)
        med = all_total[n // 2]
        p90 = all_total[int(n * 0.9)]
        mx = all_total[-1]
        sample = [(b, total_per_block[b]) for b in sorted(total_per_block)[:3]]
        print(f"  [{label}] per-CTA total kernels clocks: median={med} p90={p90} max={mx}  "
              f"(n={n} CTAs)")
        print("        per-CTA total kernels (block_idx -> clocks): " +
              ", ".join(f"{b}->{d}" for b, d in sample[:3]))
        print()
    return all_total


def profile_config(name, qo_len, kv_len, num_heads, num_kv_heads, head_dim,
                   dllm_block_size, q_offset, dtype, profiler_buffer_size, device):
    print(f"\n=== {name}  qo={qo_len} kv={kv_len} B={dllm_block_size} "
          f"off={q_offset} h={num_heads}/{num_kv_heads} d={head_dim} ===")
    sm_scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    mask_2d = make_block_mask(qo_len, kv_len, dllm_block_size, q_offset, device)
    packed = pack_mask(mask_2d)

    # --- custom_mask ---
    buf_cm = torch.zeros((profiler_buffer_size,), dtype=torch.uint64, device=device)
    # warmup (builds + caches the profiler module)
    single_prefill_with_kv_cache_profiler(q, k, v, packed, buf_cm, sm_scale=sm_scale)
    buf_cm.zero_()
    single_prefill_with_kv_cache_profiler(q, k, v, packed, buf_cm, sm_scale=sm_scale)
    nb_cm, spans_cm, sm_cm = decode_buffer(buf_cm)
    cm_d = summarize(durations_from_spans(spans_cm), "custom_mask", nb_cm)
    if export_to_perfetto_trace is not None:
        export_to_perfetto_trace(buf_cm, EVENT_NAMES, f"trace_custom_mask_{name}.perfetto-trace")
    else:
        export_to_pftrace(spans_cm, nb_cm, f"trace_custom_mask_{name}.perfetto-trace", sm_hint=sm_cm)

    # --- block_expanding ---
    buf_be = torch.zeros((profiler_buffer_size,), dtype=torch.uint64, device=device)
    block_extend_attention_with_offset_profiler(
        q, k, v, dllm_block_size=dllm_block_size,
        profiler_buffer=buf_be, q_offset=q_offset, sm_scale=sm_scale,
    )
    buf_be.zero_()
    block_extend_attention_with_offset_profiler(
        q, k, v, dllm_block_size=dllm_block_size,
        profiler_buffer=buf_be, q_offset=q_offset, sm_scale=sm_scale,
    )
    nb_be, spans_be, sm_be = decode_buffer(buf_be)
    be_d = summarize(durations_from_spans(spans_be), "block_expanding", nb_be)
    if export_to_perfetto_trace is not None:
        export_to_perfetto_trace(buf_be, EVENT_NAMES, f"trace_block_expanding_{name}.perfetto-trace")
    else:
        export_to_pftrace(spans_be, nb_be, f"trace_block_expanding_{name}.perfetto-trace", sm_hint=sm_be)

    if cm_d and be_d:
        cm_med = cm_d[len(cm_d) // 2]
        be_med = be_d[len(be_d) // 2]
        print(f"  -> kMma median clocks  custom_mask={cm_med}  block_expanding={be_med}  "
              f"ratio={cm_med / be_med if be_med else float('nan'):.1f}x")
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-size", type=int, default=1 << 20)  # 1M uint64 = 8 MiB
    args = parser.parse_args()

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    H, KH, D = 32, 8, 128

    # Config 1: the 61x wall-clock case (sparse, kv=8192, off=0).
    profile_config("sparse_8192", 64, 8192, H, KH, D, 32, 0, dtype, args.buffer_size, device)
    # Config 2: half-dense (off=4096).
    profile_config("half_dense", 64, 8192, H, KH, D, 32, 4096, dtype, args.buffer_size, device)
    # Config 3: dense (off=8128) — block_expanding degrades to plain causal.
    profile_config("dense", 64, 8192, H, KH, D, 32, 8128, dtype, args.buffer_size, device)


if __name__ == "__main__":
    main()
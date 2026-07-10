"""Profile block_expanding vs custom_mask on FA3 single prefill.

Goal: quantify the potential advantage of the procedural block_expanding mask
(MaskMode::kBlockExpanding, computed inline as per-Q-tile kv_valid_end, whole-
block skip, no bitmask read) over the element-wise custom_mask (MaskMode::kCustom,
bit-packed [qo_len,kv_len] uint8 read per KV element via `mask[off>>3]>>(off&7)&1`).

Both run on the same FA3 single-prefill backend (SM90/Hopper). They produce
numerically identical attention (smoke-checked: diff 0.0), so any timing gap is
purely the mask evaluation / memory cost.

Two sweeps:
  (A) KV-length sweep   — fixed Q/B/heads, vary kv_len. Expect block_expanding's
      skip advantage to grow with kv_len (less bitmask traffic, fewer useless FMAs).
  (B) Typical configs    — representative GQA/MQA/head_dim/long-seq shapes from
      tests/attention/test_dllm_blockwise_mask_attention.py.

Timing: flashinfer.testing.bench_gpu_time — CUPTI if cupti-python is importable,
else CUDA events (auto-fallback). L2 is flushed between iters (cold L2) so the
bitmask-reload penalty of custom_mask is visible.
"""

import argparse
import math
import sys
import time

import torch

import flashinfer
from flashinfer.dllm import block_extend_attention_with_offset
from flashinfer.testing import bench_gpu_time


# ---------------------------------------------------------------------------
# Fluent callables that close over all tensors/params so bench_gpu_time can
# invoke them with no args.
# ---------------------------------------------------------------------------
def make_custom_mask_call(q, k, v, mask, sm_scale):
    def _call():
        return flashinfer.single_prefill_with_kv_cache(
            q, k, v, custom_mask=mask, sm_scale=sm_scale, backend="fa3"
        )

    return _call


def make_block_expanding_call(q, k, v, dllm_block_size, q_offset, sm_scale):
    def _call():
        return block_extend_attention_with_offset(
            q, k, v,
            dllm_block_size=dllm_block_size,
            q_offset=q_offset,
            sm_scale=sm_scale,
            backend="fa3",
        )

    return _call


def make_block_mask(qo_len, kv_len, dllm_block_size, q_offset, device):
    """Build the bit-packed custom_mask equivalent to the block_expanding rule."""
    q_pos = torch.arange(qo_len, device=device) + q_offset
    k_pos = torch.arange(kv_len, device=device)
    mask_2d = (
        (q_pos.unsqueeze(1) // dllm_block_size)
        >= (k_pos.unsqueeze(0) // dllm_block_size)
    ).to(torch.uint8)
    return mask_2d


def time_kernel(call_fn, enable_cupti, repeat_time_ms=300, dry_run_time_ms=80):
    # bench_gpu_time returns a list of per-iter times in *ms* (not a (median,std)
    # tuple). Aggregate to median + std, convert to microseconds.
    # CUPTI needs a CUDA 13 driver; if init fails at runtime, fall back to CUDA
    # events for the rest of the run.
    global _cupti_ok
    use_cupti = enable_cupti and _cupti_ok
    try:
        times_ms = bench_gpu_time(
            call_fn,
            dry_run_time_ms=dry_run_time_ms,
            repeat_time_ms=repeat_time_ms,
            enable_cupti=use_cupti,
        )
    except Exception as e:
        msg = str(e)
        if use_cupti and ("CUDA" in msg or "NotSupported" in msg or "cupti" in msg.lower()):
            _cupti_ok = False
            print(f"[cupti unavailable, falling back to CUDA events: {e}]")
            return time_kernel(call_fn, False, repeat_time_ms, dry_run_time_ms)
        raise
    times_us = torch.tensor(times_ms, dtype=torch.float64) * 1e3
    return times_us.median().item(), float(times_us.std(unbiased=False).item())


_cupti_ok = True


def fmt(x):
    return f"{x:>10.2f}"


# ---------------------------------------------------------------------------
# Sweep A: KV-length sweep
# ---------------------------------------------------------------------------
def run_kv_length_sweep(device, dtype, enable_cupti):
    print("\n" + "=" * 118)
    print(" SWEEP A — KV-length sweep  (FA3 single prefill, block_expanding vs custom_mask)")
    print("=" * 118)
    qo_len = 64
    num_heads, num_kv_heads, head_dim = 32, 8, 128
    dllm_block_size = 32
    q_offset = 0
    sm_scale = 1.0 / math.sqrt(head_dim)
    kv_lens = [256, 512, 1024, 2048, 4096, 8192, 16384]

    header = (
        f"{'kv_len':>7} {'B':>4} {'qo':>4} "
        f"{'be_us':>10} {'be_std':>9} "
        f"{'cm_us':>10} {'cm_std':>9} "
        f"{'speedup':>9} {'be_MFlops/s':>13} {'cm_MFlops/s':>13} {'mask_KB':>9}"
    )
    print(header)
    print("-" * len(header))

    for kv_len in kv_lens:
        q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        mask = make_block_mask(qo_len, kv_len, dllm_block_size, q_offset, device)

        # correctness sanity (cheap): only first & long configs
        be_call = make_block_expanding_call(q, k, v, dllm_block_size, q_offset, sm_scale)
        cm_call = make_custom_mask_call(q, k, v, mask, sm_scale)

        be_us, be_std = time_kernel(be_call, enable_cupti)
        cm_us, cm_std = time_kernel(cm_call, enable_cupti)

        speedup = cm_us / be_us if be_us > 0 else float("nan")
        # flops = 2 * qo_len * num_heads * kv_len * (2*head_dim) (QK + AV)
        flops = 2 * qo_len * num_heads * kv_len * (2 * head_dim)
        be_mfl = flops / (be_us * 1e-6) / 1e6 if be_us > 0 else float("nan")
        cm_mfl = flops / (cm_us * 1e-6) / 1e6 if cm_us > 0 else float("nan")
        mask_kb = mask.numel() / 8 / 1024.0  # bit-packed -> bytes -> KB

        print(
            f"{kv_len:>7} {dllm_block_size:>4} {qo_len:>4} "
            f"{fmt(be_us)} {be_std:>9.2f} "
            f"{fmt(cm_us)} {cm_std:>9.2f} "
            f"{speedup:>8.2f}x {be_mfl:>13.1f} {cm_mfl:>13.1f} {mask_kb:>9.1f}"
        )
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Sweep B: typical configs
# ---------------------------------------------------------------------------
def run_typical_configs(device, dtype, enable_cupti):
    print("\n" + "=" * 118)
    print(" SWEEP B — Typical configs  (FA3 single prefill, block_expanding vs custom_mask)")
    print("=" * 118)

    # (name, qo_len, kv_len, num_heads, num_kv_heads, head_dim, dllm_block_size, q_offset)
    configs = [
        ("GQA 32/8 d128",       64,  2048, 32, 8, 128, 32, 0),
        ("GQA 32/8 d128 long",  128, 8192, 32, 8, 128, 32, 0),
        ("MQA 32/1 d128",       64,  2048, 32, 1, 128, 32, 0),
        ("GQA 32/4 d128",       64,  2048, 32, 4, 128, 32, 0),
        ("GQA 32/8 d64",        64,  2048, 32, 8, 64,  32, 0),
        ("GQA 32/8 B16",        64,  2048, 32, 8, 128, 16, 0),
        ("GQA 32/8 B64",        64,  2048, 32, 8, 128, 64, 0),
        ("GQA 32/8 B128",       64,  2048, 32, 8, 128, 128, 0),
        ("incremental step off",64,  2048, 32, 8, 128, 32, 1984),
        ("non-aligned off",     33,  2050, 32, 8, 128, 64, 17),
    ]

    header = (
        f"{'config':<24} {'qo':>4} {'kv':>6} {'B':>4} {'off':>6} "
        f"{'be_us':>10} {'cm_us':>10} {'speedup':>9} {'be_MFlops/s':>13} {'cm_MFlops/s':>13}"
    )
    print(header)
    print("-" * len(header))

    for (name, qo_len, kv_len, num_heads, num_kv_heads, head_dim, dllm_block_size, q_offset) in configs:
        sm_scale = 1.0 / math.sqrt(head_dim)
        q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        mask = make_block_mask(qo_len, kv_len, dllm_block_size, q_offset, device)

        be_call = make_block_expanding_call(q, k, v, dllm_block_size, q_offset, sm_scale)
        cm_call = make_custom_mask_call(q, k, v, mask, sm_scale)

        be_us, _ = time_kernel(be_call, enable_cupti)
        cm_us, _ = time_kernel(cm_call, enable_cupti)

        speedup = cm_us / be_us if be_us > 0 else float("nan")
        flops = 2 * qo_len * num_heads * kv_len * (2 * head_dim)
        be_mfl = flops / (be_us * 1e-6) / 1e6 if be_us > 0 else float("nan")
        cm_mfl = flops / (cm_us * 1e-6) / 1e6 if cm_us > 0 else float("nan")

        print(
            f"{name:<24} {qo_len:>4} {kv_len:>6} {dllm_block_size:>4} {q_offset:>6} "
            f"{fmt(be_us)} {fmt(cm_us)} {speedup:>8.2f}x {be_mfl:>13.1f} {cm_mfl:>13.1f}"
        )
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Sweep C: sparsity sweep — vary q_offset to change how much KV each Q tile
# actually reaches. Demonstrates block_expanding's cost scales with the
# *effective* work (kv_valid_end), NOT total kv_len; custom_mask always pays
# full kv_len. Also includes a dense causal FA3 reference as the upper bound.
# ---------------------------------------------------------------------------
def run_sparsity_sweep(device, dtype, enable_cupti):
    print("\n" + "=" * 118)
    print(" SWEEP C — Sparsity sweep  (vary q_offset; + dense causal FA3 reference)")
    print("=" * 118)
    qo_len, kv_len = 64, 8192
    num_heads, num_kv_heads, head_dim = 32, 8, 128
    dllm_block_size = 32
    sm_scale = 1.0 / math.sqrt(head_dim)

    # q_offset goes 0 (attend ~1 block) -> large (attend most of kv_len).
    q_offsets = [0, 1024, 2048, 4096, 6144, 7680, 8128]

    header = (
        f"{'q_off':>6} {'eff_kv':>7} {'frac':>6} "
        f"{'be_us':>10} {'cm_us':>10} {'speedup':>9} "
        f"{'causal_us':>10}"
    )
    print(header)
    print("-" * len(header))

    q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    # dense causal FA3 reference (upper bound: every Q reads all KV up to its row)
    causal_call = _make_causal_call(q, k, v, sm_scale)
    causal_us, _ = time_kernel(causal_call, enable_cupti)

    for q_offset in q_offsets:
        mask = make_block_mask(qo_len, kv_len, dllm_block_size, q_offset, device)
        be_call = make_block_expanding_call(q, k, v, dllm_block_size, q_offset, sm_scale)
        cm_call = make_custom_mask_call(q, k, v, mask, sm_scale)

        be_us, _ = time_kernel(be_call, enable_cupti)
        cm_us, _ = time_kernel(cm_call, enable_cupti)

        eff_kv = int(mask.sum(1).max().item())  # most any single Q attends to
        frac = eff_kv / kv_len
        speedup = cm_us / be_us if be_us > 0 else float("nan")

        print(
            f"{q_offset:>6} {eff_kv:>7} {frac:>5.1%} "
            f"{fmt(be_us)} {fmt(cm_us)} {speedup:>8.2f}x "
            f"{fmt(causal_us)}"
        )
        torch.cuda.empty_cache()


def _make_causal_call(q, k, v, sm_scale):
    def _call():
        return flashinfer.single_prefill_with_kv_cache(
            q, k, v, causal=True, sm_scale=sm_scale, backend="fa3"
        )

    return _call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--sweep", default="all", choices=["all", "kv", "typical", "sparsity"])
    args = parser.parse_args()

    from flashinfer.utils import is_sm90a_supported

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"flashinfer {flashinfer.__version__}  device {torch.cuda.get_device_name(0)}  "
          f"cc {torch.cuda.get_device_capability(0)}  dtype {args.dtype}")
    if not is_sm90a_supported(device):
        print("ERROR: FA3 requires SM90a (Hopper). This GPU is not SM90a.", file=sys.stderr)
        sys.exit(1)

    # CUPTI auto-fallback inside bench_gpu_time; detect here for reporting only.
    try:
        import cupti  # noqa: F401
        enable_cupti = True
        timing = "CUPTI"
    except Exception:
        enable_cupti = False
        timing = "CUDA-events (cupti-python not installed)"

    # Probe CUPTI actually initializes (CUDA 13 driver required). If not, the
    # per-kernel timer will auto-fall back to CUDA events; reflect that here.
    global _cupti_ok
    if enable_cupti:
        try:
            import cupti as _cupti_probe
            _cupti_probe.activity_enable(_cupti_probe.ActivityKind.RUNTIME)
            _cupti_probe.activity_disable(_cupti_probe.ActivityKind.RUNTIME)
        except Exception:
            _cupti_ok = False
            timing = "CUDA-events (CUPTI needs CUDA 13 driver)"
    print(f"timing backend: {timing}")
    print("all times are median GPU kernel us (cold L2, l2-flush between iters)\n")

    t0 = time.time()
    if args.sweep in ("all", "kv"):
        run_kv_length_sweep(device, dtype, enable_cupti)
    if args.sweep in ("all", "typical"):
        run_typical_configs(device, dtype, enable_cupti)
    if args.sweep in ("all", "sparsity"):
        run_sparsity_sweep(device, dtype, enable_cupti)
    print(f"\nelapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
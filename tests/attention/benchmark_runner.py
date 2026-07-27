"""DLLM Block Extend 三路分支性能对比 Benchmark Runner

Usage: python benchmark_runner.py --output results.json

测试 4 组配置:
  1. Block Extend vs Custom Mask Single-Request Incremental Prefill (tokens=8192, CUDA Graph)
  2. Block Extend vs Custom Mask Multi-Request BatchPrefill (256 reqs × 512 tokens, CUDA Graph)
  3. Block Extend vs Cascade Attention Single-Request (tokens=8192, CUDA Graph)
  4. Block Extend vs Cascade Attention Multi-Request BatchPrefill (256 reqs × 512 tokens, CUDA Graph)
"""

import json
import sys
import os
import traceback

# Add parent dir to path so we can import the benchmark module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check if this branch supports block_extend
HAS_BLOCK_EXTEND = False
try:
    import flashinfer
    from flashinfer.prefill import _prepare_block_extend_offset

    HAS_BLOCK_EXTEND = True
except (ImportError, AttributeError):
    pass


def run_safe(func, *args, **kwargs):
    """Run a function safely, returning (success, result_or_error)"""
    try:
        torch = __import__("torch")
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def extract_metrics(results):
    """从测试结果字典中提取关键比对指标"""
    cascade = results.get("cascade_baseline", {})
    custom_mask = results.get("custom_mask_baseline", {})

    cascade_time = cascade.get("time_cg_ms", None)
    cm_time = custom_mask.get("time_cg_ms", None)

    def get_best(prefix):
        keys = [k for k in results if k.startswith(prefix)]
        best = None
        for k in keys:
            r = results[k]
            if best is None or r["time_cg_ms"] < best["time_cg_ms"]:
                best = r
        if best:
            return {
                "chunk_size": best["chunk_size"],
                "num_steps": best["num_steps"],
                "time_ms": best["time_cg_ms"],
                "ms_per_step": best["time_cg_ms"] / best["num_steps"]
                if best["num_steps"]
                else None,
            }
        return None

    metrics = {
        "baseline1_cascade_ms": cascade_time,
        "baseline2_custom_mask_ms": cm_time,
        "bbe_best": get_best("bbe_chunk"),
        "v2_best": get_best("v2_chunk"),
    }

    # Compute speedups
    bbe = metrics["bbe_best"]
    if bbe and bbe["time_ms"]:
        bbe["vs_cascade_speedup"] = (
            cascade_time / bbe["time_ms"] if cascade_time else None
        )
        bbe["vs_custom_mask_speedup"] = cm_time / bbe["time_ms"] if cm_time else None

    v2 = metrics["v2_best"]
    if v2 and v2["time_ms"]:
        v2["vs_cascade_speedup"] = (
            cascade_time / v2["time_ms"] if cascade_time else None
        )
        v2["vs_custom_mask_speedup"] = cm_time / v2["time_ms"] if cm_time else None

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"FlashInfer: {flashinfer.__version__}")
    print(f"block_extend support: {HAS_BLOCK_EXTEND}")
    print()

    from test_dllm_blockwise_mask_attention_benchmark import (
        test_incremental_singlereq_prefill_step_by_step_with_cg,
        test_incremental_batchprefill_step_by_step_with_cg,
    )

    all_results = {
        "gpu": torch.cuda.get_device_name(0),
        "flashinfer_version": flashinfer.__version__,
        "has_block_extend": HAS_BLOCK_EXTEND,
    }

    # ============================================================
    # Test 1+3: Single-Request Incremental Prefill (tokens=8192)
    # ============================================================
    print("=" * 80)
    print("TEST: Single-Request Incremental Prefill (tokens=8192, CUDA Graph)")
    print("=" * 80)
    sys.stdout.flush()

    ok, result = run_safe(
        test_incremental_singlereq_prefill_step_by_step_with_cg,
        tokens_per_request=8192,
        dllm_block_size=32,
        chunk_sizes=[32, 64, 128, 256, 512],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        warmup_iters=10,
        bench_iters=100,
        verbose=False,
        backend="auto",
    )
    torch.cuda.empty_cache()

    if ok:
        all_results["single"] = extract_metrics(result)
    else:
        print(f"  SKIPPED: {result}")
        all_results["single"] = {"error": str(result)}

    # ============================================================
    # Test 2+4: Multi-Request BatchPrefill (256 reqs × 512 tokens)
    # ============================================================
    print("\n" + "=" * 80)
    print("TEST: Multi-Request BatchPrefill (256 reqs × 512 tokens, CUDA Graph)")
    print("=" * 80)
    sys.stdout.flush()

    ok, result = run_safe(
        test_incremental_batchprefill_step_by_step_with_cg,
        num_requests=256,
        tokens_per_request=512,
        dllm_block_size=32,
        chunk_sizes=[32, 64, 128, 256, 512],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        warmup_iters=10,
        bench_iters=100,
        verbose=False,
        backend="auto",
    )
    torch.cuda.empty_cache()

    if ok:
        all_results["multi"] = extract_metrics(result)
    else:
        print(f"  SKIPPED: {result}")
        all_results["multi"] = {"error": str(result)}

    # ============================================================
    # Print compact summary
    # ============================================================
    print("\n" + "=" * 80)
    print("COMPACT SUMMARY")
    print("=" * 80)

    for test_key in ["single", "multi"]:
        r = all_results.get(test_key, {})
        if "error" in r:
            print(f"\n[{test_key}] ERROR: {r['error']}")
            continue

        test_label = "Single 8192" if test_key == "single" else "Multi 256×512"
        print(f"\n--- {test_label} ---")
        print(
            f"  Cascade (Base1):     {r.get('baseline1_cascade_ms', 'N/A'):>10} ms"
            if r.get("baseline1_cascade_ms")
            else f"  Cascade (Base1):     N/A"
        )
        print(
            f"  CustomMask (Base2):  {r.get('baseline2_custom_mask_ms', 'N/A'):>10} ms"
            if r.get("baseline2_custom_mask_ms")
            else f"  CustomMask (Base2):  N/A"
        )

        bbe = r.get("bbe_best")
        if bbe:
            print(
                f"  BBE (chunk={bbe['chunk_size']}):    {bbe['time_ms']:>10.3f} ms  ({bbe.get('vs_cascade_speedup', 0):.2f}x vs Cascade, {bbe.get('vs_custom_mask_speedup', 0):.2f}x vs CustomMask)"
            )

        v2 = r.get("v2_best")
        if v2:
            print(
                f"  V2  (chunk={v2['chunk_size']}):    {v2['time_ms']:>10.3f} ms  ({v2.get('vs_cascade_speedup', 0):.2f}x vs Cascade, {v2.get('vs_custom_mask_speedup', 0):.2f}x vs CustomMask)"
            )

    if args.output:
        with open(args.output, "w") as f:
            # Custom JSON encoder for non-serializable types
            class Encoder(json.JSONEncoder):
                def default(self, o):
                    return str(o)

            json.dump(all_results, f, indent=2, cls=Encoder)
        print(f"\nResults saved to {args.output}")

    return all_results


if __name__ == "__main__":
    main()

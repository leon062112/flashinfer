"""细粒度 Profiling: BBE plan optimization vs Custom Mask, 逐项拆分耗时

分析维度:
  1. plan() 时间 vs run() 时间占比
  2. BBE vs Custom Mask 的 kernel 时间差异
  3. effective_kv 剪枝对 tile 数和执行时间的影响
"""

import torch
import math
from flashinfer import (
    BatchPrefillWithRaggedKVCacheWrapper,
    single_prefill_with_kv_cache,
)

torch.manual_seed(42)
dev = torch.device("cuda:0")
fp16 = torch.float16
NH, NKH, HD = 32, 8, 128
BS = 32  # dllm_block_size
SM_SCALE = 1.0 / math.sqrt(HD)


def bench(fn, warmup=30, iters=200, sync_after=True):
    for _ in range(warmup):
        fn()
    if sync_after:
        torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return sum(times) / len(times), min(times)


def make_custom_mask(qo_len, kv_len, q_offset=0):
    q_pos = torch.arange(qo_len, device=dev) + q_offset
    k_pos = torch.arange(kv_len, device=dev)
    return (q_pos.unsqueeze(1) // BS) >= (k_pos.unsqueeze(0) // BS)


def header(msg):
    print(f"\n{'─' * 70}")
    print(f"  {msg}")
    print(f"{'─' * 70}")


# ============ Case A: Single-req 8192, per-step at different kv_len ============
header("Case A: Single-req 8192 — per-step BBE vs CustomMask profile")

for chunk in [32, 64, 128]:
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    header(f"  chunk_size={chunk}")

    for label, kv_len, q_offs in [
        ("early kv=chunk ", chunk, 0),
        ("mid   kv=4096 ", 4096, 4096 - chunk),
        ("late  kv=8192 ", 8192, 8192 - chunk),
    ]:
        qo_len = chunk
        q = torch.randn(qo_len, NH, HD, dtype=fp16, device=dev)
        k = torch.randn(kv_len, NKH, HD, dtype=fp16, device=dev)
        v = torch.randn(kv_len, NKH, HD, dtype=fp16, device=dev)
        qip = torch.tensor([0, qo_len], dtype=torch.int32, device=dev)
        kip = torch.tensor([0, kv_len], dtype=torch.int32, device=dev)
        qoff = torch.tensor([q_offs], dtype=torch.int32, device=dev)
        cmask = make_custom_mask(qo_len, kv_len, q_offs)

        # --- plan() time: BBE vs CM ---
        def plan_bbe():
            w = BatchPrefillWithRaggedKVCacheWrapper(
                ws, kv_layout="NHD", backend="fa2", block_extend=True, block_size=BS
            )
            w.plan(
                qo_indptr=qip,
                kv_indptr=kip,
                num_qo_heads=NH,
                num_kv_heads=NKH,
                head_dim_qk=HD,
                q_offsets=qoff,
            )

        def plan_cm():
            w = BatchPrefillWithRaggedKVCacheWrapper(ws, kv_layout="NHD", backend="fa2")
            w.plan(
                qo_indptr=qip,
                kv_indptr=kip,
                num_qo_heads=NH,
                num_kv_heads=NKH,
                head_dim_qk=HD,
                custom_mask=cmask,
                causal=False,
                sm_scale=SM_SCALE,
            )

        # Create permanent wrappers for run() profiling
        w_bbe = BatchPrefillWithRaggedKVCacheWrapper(
            ws, kv_layout="NHD", backend="fa2", block_extend=True, block_size=BS
        )
        w_bbe.plan(
            qo_indptr=qip,
            kv_indptr=kip,
            num_qo_heads=NH,
            num_kv_heads=NKH,
            head_dim_qk=HD,
            q_offsets=qoff,
        )
        w_cm = BatchPrefillWithRaggedKVCacheWrapper(ws, kv_layout="NHD", backend="fa2")
        w_cm.plan(
            qo_indptr=qip,
            kv_indptr=kip,
            num_qo_heads=NH,
            num_kv_heads=NKH,
            head_dim_qk=HD,
            custom_mask=cmask,
            causal=False,
            sm_scale=SM_SCALE,
        )

        # Run single_prefill as additional baseline
        avg_single, min_single = bench(
            lambda: single_prefill_with_kv_cache(
                q, k, v, custom_mask=cmask, sm_scale=SM_SCALE
            )
        )

        avg_plan_bbe, min_plan_bbe = bench(plan_bbe)
        avg_plan_cm, min_plan_cm = bench(plan_cm)
        avg_run_bbe, min_run_bbe = bench(lambda: w_bbe.run(q, k, v))
        avg_run_cm, min_run_cm = bench(lambda: w_cm.run(q, k, v))

        bbe_tot = avg_plan_bbe + avg_run_bbe
        cm_tot = avg_plan_cm + avg_run_cm

        print(f"  [{label}] qo={qo_len:4d} kv={kv_len:5d} offs={q_offs:4d}")
        print(
            f"    BBE:   plan={avg_plan_bbe:7.3f}ms ({avg_plan_bbe / bbe_tot * 100:4.0f}%)  "
            f"run={avg_run_bbe:7.3f}ms ({avg_run_bbe / bbe_tot * 100:4.0f}%)  total={bbe_tot:.3f}ms"
        )
        print(
            f"    CustM: plan={avg_plan_cm:7.3f}ms ({avg_plan_cm / cm_tot * 100:4.0f}%)  "
            f"run={avg_run_cm:7.3f}ms ({avg_run_cm / cm_tot * 100:4.0f}%)  total={cm_tot:.3f}ms"
        )
        print(f"    Single: run={avg_single:7.3f}ms")
        print(
            f"    Δplan = {avg_plan_bbe - avg_plan_cm:+.3f}ms  Δrun = {avg_run_bbe - avg_run_cm:+.3f}ms  "
            f"run(BBE/CM) = {avg_run_bbe / max(avg_run_cm, 0.001):.2f}x"
        )


# ============ Case B: Multi-req 256×512, per-step ============
header("Case B: Multi-req 256×512 — per-step BBE vs CustomMask profile")
NR = 256
TPR = 512

all_q = [torch.randn(TPR, NH, HD, dtype=fp16, device=dev) for _ in range(NR)]
all_k = [torch.randn(TPR, NKH, HD, dtype=fp16, device=dev) for _ in range(NR)]
all_v = [torch.randn(TPR, NKH, HD, dtype=fp16, device=dev) for _ in range(NR)]

for chunk in [128, 256]:
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    header(f"  chunk_size={chunk}")

    q_list = [all_q[i][:chunk] for i in range(NR)]
    k_list = [all_k[i][:chunk] for i in range(NR)]
    v_list = [all_v[i][:chunk] for i in range(NR)]
    q_batch = torch.cat(q_list, dim=0)
    k_batch = torch.cat(k_list, dim=0)
    v_batch = torch.cat(v_list, dim=0)

    qip = torch.tensor(
        [i * chunk for i in range(NR + 1)], dtype=torch.int32, device=dev
    )
    kip = torch.tensor(
        [i * chunk for i in range(NR + 1)], dtype=torch.int32, device=dev
    )
    qoff = torch.zeros(NR, dtype=torch.int32, device=dev)
    cmask = make_custom_mask(chunk, chunk, 0).flatten().repeat(NR)

    def plan_bbe():
        w = BatchPrefillWithRaggedKVCacheWrapper(
            ws, kv_layout="NHD", backend="fa2", block_extend=True, block_size=BS
        )
        w.plan(
            qo_indptr=qip,
            kv_indptr=kip,
            num_qo_heads=NH,
            num_kv_heads=NKH,
            head_dim_qk=HD,
            q_offsets=qoff,
        )

    def plan_cm():
        w = BatchPrefillWithRaggedKVCacheWrapper(ws, kv_layout="NHD", backend="fa2")
        w.plan(
            qo_indptr=qip,
            kv_indptr=kip,
            num_qo_heads=NH,
            num_kv_heads=NKH,
            head_dim_qk=HD,
            custom_mask=cmask,
            causal=False,
            sm_scale=SM_SCALE,
        )

    w_bbe = BatchPrefillWithRaggedKVCacheWrapper(
        ws, kv_layout="NHD", backend="fa2", block_extend=True, block_size=BS
    )
    w_bbe.plan(
        qo_indptr=qip,
        kv_indptr=kip,
        num_qo_heads=NH,
        num_kv_heads=NKH,
        head_dim_qk=HD,
        q_offsets=qoff,
    )
    w_cm = BatchPrefillWithRaggedKVCacheWrapper(ws, kv_layout="NHD", backend="fa2")
    w_cm.plan(
        qo_indptr=qip,
        kv_indptr=kip,
        num_qo_heads=NH,
        num_kv_heads=NKH,
        head_dim_qk=HD,
        custom_mask=cmask,
        causal=False,
        sm_scale=SM_SCALE,
    )

    avg_plan_bbe, min_plan_bbe = bench(plan_bbe)
    avg_plan_cm, min_plan_cm = bench(plan_cm)
    avg_run_bbe, min_run_bbe = bench(lambda: w_bbe.run(q_batch, k_batch, v_batch))
    avg_run_cm, min_run_cm = bench(lambda: w_cm.run(q_batch, k_batch, v_batch))

    bbe_tot = avg_plan_bbe + avg_run_bbe
    cm_tot = avg_plan_cm + avg_run_cm

    print(f"  [batch={NR}, Q=KV={chunk}]")
    print(
        f"    BBE:   plan={avg_plan_bbe:7.3f}ms ({avg_plan_bbe / bbe_tot * 100:4.0f}%)  "
        f"run={avg_run_bbe:7.3f}ms ({avg_run_bbe / bbe_tot * 100:4.0f}%)  total={bbe_tot:.3f}ms"
    )
    print(
        f"    CustM: plan={avg_plan_cm:7.3f}ms ({avg_plan_cm / cm_tot * 100:4.0f}%)  "
        f"run={avg_run_cm:7.3f}ms ({avg_run_cm / cm_tot * 100:4.0f}%)  total={cm_tot:.3f}ms"
    )
    print(
        f"    Δplan = {avg_plan_bbe - avg_plan_cm:+.3f}ms  Δrun = {avg_run_bbe - avg_run_cm:+.3f}ms  "
        f"run(BBE/CM) = {avg_run_bbe / max(avg_run_cm, 0.001):.2f}x"
    )


# ============ Case C: effective_kv pruning impact ============
header("Case C: effective_kv pruning — tile & time vs kv_len (Q in block 0)")

for kv_len in [512, 1024, 2048, 4096, 8192]:
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    qo_len = 32
    q = torch.randn(qo_len, NH, HD, dtype=fp16, device=dev)
    k = torch.randn(kv_len, NKH, HD, dtype=fp16, device=dev)
    v = torch.randn(kv_len, NKH, HD, dtype=fp16, device=dev)
    qip = torch.tensor([0, qo_len], dtype=torch.int32, device=dev)
    kip = torch.tensor([0, kv_len], dtype=torch.int32, device=dev)
    qoff = torch.tensor([0], dtype=torch.int32, device=dev)
    cmask = make_custom_mask(qo_len, kv_len, 0)

    w_bbe = BatchPrefillWithRaggedKVCacheWrapper(
        ws, kv_layout="NHD", backend="fa2", block_extend=True, block_size=BS
    )
    w_bbe.plan(
        qo_indptr=qip,
        kv_indptr=kip,
        num_qo_heads=NH,
        num_kv_heads=NKH,
        head_dim_qk=HD,
        q_offsets=qoff,
    )
    w_cm = BatchPrefillWithRaggedKVCacheWrapper(ws, kv_layout="NHD", backend="fa2")
    w_cm.plan(
        qo_indptr=qip,
        kv_indptr=kip,
        num_qo_heads=NH,
        num_kv_heads=NKH,
        head_dim_qk=HD,
        custom_mask=cmask,
        causal=False,
        sm_scale=SM_SCALE,
    )

    run_bbe, _ = bench(lambda: w_bbe.run(q, k, v))
    run_cm, _ = bench(lambda: w_cm.run(q, k, v))
    eff_kv = min(BS, kv_len)
    speedup = run_cm / max(run_bbe, 0.001)

    print(
        f"  kv_len={kv_len:5d}  eff_kv={eff_kv:4d} ({eff_kv / kv_len * 100:5.1f}%)  "
        f"BBE={run_bbe:.3f}ms  CM={run_cm:.3f}ms  speedup={speedup:5.2f}x"
    )

print(f"\n{'─' * 70}")
print("Done.")

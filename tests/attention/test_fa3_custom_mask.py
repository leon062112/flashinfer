"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math

import pytest
import torch

import flashinfer
from flashinfer.utils import is_sm90a_supported


def _make_causal_mask(qo_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
    """Causal mask: (qo_idx >= kv_idx - (kv_len - qo_len))."""
    mask = torch.tril(
        torch.full((qo_len, kv_len), True, device=device),
        diagonal=(kv_len - qo_len),
    )
    return mask.to(torch.uint8)


def _make_block_extend_mask(
    qo_len: int,
    kv_len: int,
    block_size: int,
    q_offset: int,
    kv_offset: int,
    device: torch.device,
) -> torch.Tensor:
    """Block-extend mask: q_block >= k_block where block = (pos + offset) // B."""
    q_pos = torch.arange(qo_len, device=device) + q_offset
    k_pos = torch.arange(kv_len, device=device) + kv_offset
    mask = (q_pos.unsqueeze(1) // block_size) >= (k_pos.unsqueeze(0) // block_size)
    return mask.to(torch.uint8)


def _make_random_mask(
    qo_len: int, kv_len: int, density: float, device: torch.device
) -> torch.Tensor:
    """Random mask with given density of True entries."""
    mask = torch.rand(qo_len, kv_len, device=device) < density
    return mask.to(torch.uint8)


def _pytorch_sdpa_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_2d: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Reference: PyTorch SDPA with the same mask, used as ground truth."""
    # SDPA expects GQA broadcastable KV: [batch, heads, seq, dim]
    q_ref = q.unsqueeze(0).transpose(1, 2)   # [1, Hq, Q, D]
    k_ref = k.unsqueeze(0).transpose(1, 2)   # [1, Hkv, K, D]
    v_ref = v.unsqueeze(0).transpose(1, 2)   # [1, Hkv, K, D]
    # mask: True=keep, False=mask; SDPA expects float additive mask
    attn_mask = torch.where(mask_2d.bool(), 0.0, float("-inf"))
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Q, K]
    # SDPA handles GQA broadcasting
    o_ref = torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, attn_mask=attn_mask, scale=sm_scale, is_causal=False,
    )
    return o_ref.squeeze(0).transpose(0, 1)  # [Q, Hq, D]


# ── Shape sets ──────────────────────────────────────────────────────────────
# (qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim)
SMALL_SHAPES = [
    (32, 64, 8, 8, 128),
    (33, 97, 4, 1, 128),
    (64, 128, 32, 8, 128),
    (64, 64, 16, 4, 64),
]

LARGE_SHAPES = [
    (128, 2048, 32, 8, 128),
    (256, 512, 16, 4, 128),
    (512, 512, 8, 8, 256),
]

# ── Fixture: warmup JIT ─────────────────────────────────────────────────────
@pytest.fixture(scope="module", autouse=True)
def warmup_fa3_jit():
    """Pre-compile FA3 single-prefill kernels to avoid per-test JIT overhead."""
    if not is_sm90a_supported(torch.device("cuda:0")):
        return
    flashinfer.jit.attention.gen_single_prefill_module(
        "fa3",
        torch.float16,
        torch.float16,
        torch.float16,
        head_dim_qk=128,
        head_dim_vo=128,
        pos_encoding_mode=0,
    ).build_and_load()
    flashinfer.jit.attention.gen_single_prefill_module(
        "fa3",
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        head_dim_qk=128,
        head_dim_vo=128,
        pos_encoding_mode=0,
    ).build_and_load()


# ── Tests ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SMALL_SHAPES)
def test_fa3_custom_mask_causal_vs_fa2(dtype, shape):
    """FA3 custom_mask with causal pattern should match FA2 custom_mask."""
    if not is_sm90a_supported(torch.device("cuda:0")):
        pytest.skip("SM90A is not supported")

    qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim = shape
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")

    tol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    sm_scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(42)
    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=dtype, device="cuda:0")
    k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda:0")
    v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda:0")

    mask_2d = _make_causal_mask(qo_len, kv_len, q.device)

    o_fa2 = flashinfer.single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa2",
    )
    o_fa3 = flashinfer.single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa3",
    )
    torch.testing.assert_close(o_fa3, o_fa2, rtol=tol, atol=tol)
    torch.cuda.empty_cache()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SMALL_SHAPES)
def test_fa3_custom_mask_block_extend_vs_fa2(dtype, shape):
    """FA3 custom_mask with block-extend pattern should match FA2."""
    if not is_sm90a_supported(torch.device("cuda:0")):
        pytest.skip("SM90A is not supported")

    qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim = shape
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")

    tol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    sm_scale = 1.0 / math.sqrt(head_dim)

    rock = 137
    torch.manual_seed(rock)
    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=dtype, device="cuda:0")
    k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda:0")
    v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda:0")

    for block_size in [16, 32, 64]:
        for q_offset in [0, 64]:
            mask_2d = _make_block_extend_mask(
                qo_len, kv_len, block_size, q_offset=q_offset, kv_offset=0,
                device=q.device,
            )
            o_fa2 = flashinfer.single_prefill_with_kv_cache(
                q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa2",
            )
            o_fa3 = flashinfer.single_prefill_with_kv_cache(
                q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa3",
            )
            torch.testing.assert_close(
                o_fa3, o_fa2, rtol=tol, atol=tol,
                msg=f"block_size={block_size}, q_offset={q_offset}",
            )
    torch.cuda.empty_cache()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SMALL_SHAPES)
def test_fa3_custom_mask_random_vs_fa2(dtype, shape):
    """FA3 custom_mask with random mask pattern should match FA2."""
    if not is_sm90a_supported(torch.device("cuda:0")):
        pytest.skip("SM90A is not supported")

    qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim = shape
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")

    tol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    sm_scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(99)
    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=dtype, device="cuda:0")
    k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda:0")
    v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda:0")

    for density in [0.3, 0.7, 0.95]:
        mask_2d = _make_random_mask(qo_len, kv_len, density, q.device)
        o_fa2 = flashinfer.single_prefill_with_kv_cache(
            q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa2",
        )
        o_fa3 = flashinfer.single_prefill_with_kv_cache(
            q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa3",
        )
        torch.testing.assert_close(
            o_fa3, o_fa2, rtol=tol, atol=tol,
            msg=f"random density={density}",
        )
    torch.cuda.empty_cache()


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("shape", SMALL_SHAPES)
def test_fa3_custom_mask_vs_pytorch_sdpa(dtype, shape):
    """FA3 custom_mask should match PyTorch SDPA reference (ground truth)."""
    if not is_sm90a_supported(torch.device("cuda:0")):
        pytest.skip("SM90A is not supported")

    qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim = shape
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")

    sm_scale = 1.0 / math.sqrt(head_dim)
    tol = 3e-2  # SDPA vs FlashInfer may differ slightly

    torch.manual_seed(777)
    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=dtype, device="cuda:0")
    k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda:0")
    v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda:0")

    # Test with block-extend mask
    mask_2d = _make_block_extend_mask(
        qo_len, kv_len, block_size=32, q_offset=32, kv_offset=0, device=q.device,
    )
    o_ref = _pytorch_sdpa_ref(q, k, v, mask_2d, sm_scale)
    o_fa3 = flashinfer.single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa3",
    )
    torch.testing.assert_close(o_fa3, o_ref, rtol=tol, atol=tol)

    # Test with random mask
    mask_2d = _make_random_mask(qo_len, kv_len, 0.5, q.device)
    o_ref = _pytorch_sdpa_ref(q, k, v, mask_2d, sm_scale)
    o_fa3 = flashinfer.single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa3",
    )
    torch.testing.assert_close(o_fa3, o_ref, rtol=tol, atol=tol)
    torch.cuda.empty_cache()


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("shape", LARGE_SHAPES)
def test_fa3_custom_mask_large_shapes_vs_fa2(dtype, shape):
    """FA3 custom_mask on larger/more-challenging shapes should match FA2."""
    if not is_sm90a_supported(torch.device("cuda:0")):
        pytest.skip("SM90A is not supported")

    qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim = shape
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")

    sm_scale = 1.0 / math.sqrt(head_dim)
    tol = 2e-2

    torch.manual_seed(42)
    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=dtype, device="cuda:0")
    k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda:0")
    v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda:0")

    # Causal mask on a long KV
    mask_2d = _make_causal_mask(qo_len, kv_len, q.device)
    o_fa2 = flashinfer.single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa2",
    )
    o_fa3 = flashinfer.single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa3",
    )
    torch.testing.assert_close(o_fa3, o_fa2, rtol=tol, atol=tol)

    # Block-extend mask with offset
    mask_2d = _make_block_extend_mask(
        qo_len, kv_len, block_size=64, q_offset=0, kv_offset=0, device=q.device,
    )
    o_fa2 = flashinfer.single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa2",
    )
    o_fa3 = flashinfer.single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa3",
    )
    torch.testing.assert_close(o_fa3, o_fa2, rtol=tol, atol=tol)
    torch.cuda.empty_cache()


def test_fa3_custom_mask_all_masked():
    """FA3 custom_mask with all-zero mask: output should be all zeros."""
    if not is_sm90a_supported(torch.device("cuda:0")):
        pytest.skip("SM90A is not supported")

    qo_len, kv_len, head_dim = 64, 128, 128
    sm_scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(1)
    q = torch.randn(qo_len, 4, head_dim, dtype=torch.float16, device="cuda:0")
    k = torch.randn(kv_len, 4, head_dim, dtype=torch.float16, device="cuda:0")
    v = torch.randn(kv_len, 4, head_dim, dtype=torch.float16, device="cuda:0")

    mask_all_false = torch.zeros(qo_len, kv_len, dtype=torch.uint8, device="cuda:0")
    o_fa3 = flashinfer.single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask_all_false, sm_scale=sm_scale, backend="fa3",
    )
    assert torch.all(o_fa3 == 0), "All-masked output should be zero"
    torch.cuda.empty_cache()


def test_fa3_custom_mask_all_unmasked():
    """FA3 custom_mask with all-ones mask: output should match causal=False."""
    if not is_sm90a_supported(torch.device("cuda:0")):
        pytest.skip("SM90A is not supported")

    qo_len, kv_len, head_dim = 64, 64, 128
    sm_scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(2)
    q = torch.randn(qo_len, 4, head_dim, dtype=torch.float16, device="cuda:0")
    k = torch.randn(kv_len, 4, head_dim, dtype=torch.float16, device="cuda:0")
    v = torch.randn(kv_len, 4, head_dim, dtype=torch.float16, device="cuda:0")

    # All-ones custom mask (no masking) via FA3 custom_mask
    mask_all_true = torch.ones(qo_len, kv_len, dtype=torch.uint8, device="cuda:0")
    o_custom = flashinfer.single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask_all_true, sm_scale=sm_scale, backend="fa3",
    )

    # Non-causal (no masking) via FA3 causal=False
    o_nocausal = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=False, sm_scale=sm_scale, backend="fa3",
    )
    torch.testing.assert_close(o_custom, o_nocausal, rtol=1e-2, atol=1e-2)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
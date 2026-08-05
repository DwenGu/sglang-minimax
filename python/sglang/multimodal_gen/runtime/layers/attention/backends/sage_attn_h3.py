"""MiniMax-H3-specialized fused SageAttention2 varlen kernels.

The upstream SageAttention2 varlen path materializes three temporary tensors
before launching attention: centered K, quantized Q, and an FP16 copy of V.
H3 always reaches this backend as packed, non-causal THD tensors with head
dimension 128.  The kernels below keep the same per-block INT8 quantization,
but fuse K centering into K quantization and fuse Q quantization into the
attention kernel.  BF16 V is converted to FP16 in registers.

This module intentionally does not cache cu_seqlens-derived metadata or
workspaces.  That optimization is independent and can be evaluated later.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_DEFAULT_BLOCK_M = 64
_BLOCK_N = 32
_HEAD_DIM = 128
_LOG2_E = 1.4426950408889634


def _kernel_config() -> tuple[int, int, int, int]:
    """Return the H20/H3-tuned config, allowing explicit lab overrides."""
    block_m = int(os.getenv("SGLANG_SAGE2_H3_BLOCK_M", str(_DEFAULT_BLOCK_M)))
    block_n = int(os.getenv("SGLANG_SAGE2_H3_BLOCK_N", str(_BLOCK_N)))
    num_warps = int(os.getenv("SGLANG_SAGE2_H3_NUM_WARPS", "4"))
    num_stages = int(os.getenv("SGLANG_SAGE2_H3_NUM_STAGES", "5"))
    if block_m not in (32, 64, 128):
        raise ValueError("SGLANG_SAGE2_H3_BLOCK_M must be 32, 64, or 128")
    if block_n not in (32, 64, 128):
        raise ValueError("SGLANG_SAGE2_H3_BLOCK_N must be 32, 64, or 128")
    if num_warps not in (4, 8):
        raise ValueError("SGLANG_SAGE2_H3_NUM_WARPS must be 4 or 8")
    if num_stages not in (2, 3, 4, 5):
        raise ValueError("SGLANG_SAGE2_H3_NUM_STAGES must be between 2 and 5")
    return block_m, block_n, num_warps, num_stages


def is_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
) -> bool:
    """Whether the tensors match the specialized MiniMax-H3 kernel contract."""
    return (
        not is_causal
        and query.is_cuda
        and query.ndim == key.ndim == value.ndim == 3
        and query.dtype == key.dtype == value.dtype
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.shape == key.shape == value.shape
        and query.shape[-1] == _HEAD_DIM
        and query.stride(-1) == key.stride(-1) == value.stride(-1) == 1
    )


@triton.jit
def _centered_k_quant_kernel(
    K,
    KMean,
    KInt8,
    KScale,
    CuSeqLens,
    CuSeqLensScale,
    stride_kn,
    stride_kh,
    stride_kmn,
    stride_kmh,
    stride_kon,
    stride_koh,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_n = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)

    seq_start = tl.load(CuSeqLens + batch)
    seq_end = tl.load(CuSeqLens + batch + 1)
    seq_len = seq_end - seq_start
    if block_n * BLOCK_N >= seq_len:
        return

    scale_start = tl.load(CuSeqLensScale + batch)
    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    valid = offs_n[:, None] < seq_len

    k_ptrs = (
        K
        + seq_start * stride_kn
        + head * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :]
    )
    km_ptrs = KMean + head * stride_kmh + offs_d[None, :] * stride_kmn
    ko_ptrs = (
        KInt8
        + seq_start * stride_kon
        + head * stride_koh
        + offs_n[:, None] * stride_kon
        + offs_d[None, :]
    )

    # Upstream computes K - mean as a standalone tensor.  Keep the centered
    # values in registers instead and immediately reduce/quantize them.
    k = tl.load(k_ptrs, mask=valid, other=0.0).to(tl.float32)
    km = tl.load(km_ptrs).to(tl.float32)
    centered = tl.where(valid, k - km, 0.0)
    scale = tl.maximum(tl.max(tl.abs(centered)) / 127.0, 1.0e-8)
    quant = centered / scale
    quant += 0.5 * tl.where(quant >= 0.0, 1.0, -1.0)
    tl.store(ko_ptrs, quant.to(tl.int8), mask=valid)
    tl.store(KScale + (scale_start + block_n) * H + head, scale)


@triton.jit
def _fused_q_attn_inner(
    acc,
    l_i,
    m_i,
    q,
    q_scale,
    kv_len,
    K_ptrs,
    KScale_ptr,
    V_ptrs,
    stride_kn,
    stride_vn,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    offs_n: tl.constexpr,
):
    for start_n in range(0, kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_valid = offs_n[None, :] < (kv_len - start_n)
        k = tl.load(K_ptrs, mask=k_valid, other=0)
        k_scale = tl.load(KScale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)
        qk += tl.where(k_valid, 0.0, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # Avoid the full-tensor BF16 -> FP16 copy made by upstream Sage2.
        v_valid = offs_n[:, None] < (kv_len - start_n)
        v = tl.load(V_ptrs, mask=v_valid, other=0.0).to(tl.float16)
        acc += tl.dot(p.to(tl.float16), v, out_dtype=tl.float16)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        KScale_ptr += H
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i


@triton.jit
def _fused_q_attn_kernel(
    Q,
    KInt8,
    V,
    CuSeqLens,
    KScale,
    CuSeqLensKScale,
    Out,
    stride_qn,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_on,
    stride_oh,
    sm_scale_log2,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_m = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)

    seq_start = tl.load(CuSeqLens + batch)
    seq_end = tl.load(CuSeqLens + batch + 1)
    seq_len = seq_end - seq_start
    if block_m * BLOCK_M >= seq_len:
        return

    k_scale_start = tl.load(CuSeqLensKScale + batch)
    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_valid = offs_m[:, None] < seq_len

    q_ptrs = (
        Q
        + seq_start * stride_qn
        + head * stride_qh
        + offs_m[:, None] * stride_qn
        + offs_d[None, :]
    )
    k_ptrs = (
        KInt8
        + seq_start * stride_kn
        + head * stride_kh
        + offs_n[None, :] * stride_kn
        + offs_d[:, None]
    )
    v_ptrs = (
        V
        + seq_start * stride_vn
        + head * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_d[None, :]
    )
    out_ptrs = (
        Out
        + seq_start * stride_on
        + head * stride_oh
        + offs_m[:, None] * stride_on
        + offs_d[None, :]
    )

    # Q has exactly one consumer.  Quantize it in the same CTA that consumes
    # it, removing both the Q INT8/scale buffers and the Q quantization launch.
    q = tl.load(q_ptrs, mask=q_valid, other=0.0).to(tl.float32)
    q *= sm_scale_log2
    q_scale = tl.maximum(tl.max(tl.abs(q)) / 127.0, 1.0e-8)
    q_quant = q / q_scale
    q_quant += 0.5 * tl.where(q_quant >= 0.0, 1.0, -1.0)
    q_int8 = q_quant.to(tl.int8)

    k_scale_ptr = KScale + k_scale_start * H + head
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    acc, l_i = _fused_q_attn_inner(
        acc,
        l_i,
        m_i,
        q_int8,
        q_scale,
        seq_len,
        k_ptrs,
        k_scale_ptr,
        v_ptrs,
        stride_kn,
        stride_vn,
        H,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        offs_n,
    )
    tl.store(out_ptrs, (acc / l_i[:, None]).to(Out.type.element_ty), mask=q_valid)


def sageattn2_h3_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Run fused SageAttention2 for MiniMax-H3 packed self-attention."""
    if not is_supported(query, key, value, is_causal=False):
        raise ValueError("unsupported input for the MiniMax-H3 fused Sage2 kernel")
    if not cu_seqlens.is_cuda or not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be a contiguous CUDA tensor")

    torch.cuda.set_device(value.device)
    heads = query.shape[1]
    batches = cu_seqlens.shape[0] - 1
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(_HEAD_DIM)

    # Deliberately recomputed on every call: metadata caching is option 3 and
    # is not part of this optimization pass.
    block_m, block_n, num_warps, num_stages = _kernel_config()
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    k_scale_lens = (seq_lens + block_n - 1) // block_n
    cu_seqlens_k_scale = F.pad(torch.cumsum(k_scale_lens, dim=0), (1, 0), value=0)

    key_mean = key.mean(dim=0, keepdim=True)
    key_int8 = torch.empty_like(key, dtype=torch.int8)
    # Avoid synchronizing on cu_seqlens_k_scale[-1].item().  The packed token
    # count determines an allocation upper bound; unused trailing scales are
    # harmless and are never addressed by the kernels.
    max_scale_blocks = triton.cdiv(key.shape[0], block_n) + batches
    key_scale = torch.empty(
        (max_scale_blocks, heads), device=key.device, dtype=torch.float32
    )

    k_grid = (triton.cdiv(max_seqlen, block_n), heads, batches)
    _centered_k_quant_kernel[k_grid](
        key,
        key_mean,
        key_int8,
        key_scale,
        cu_seqlens,
        cu_seqlens_k_scale,
        key.stride(0),
        key.stride(1),
        key_mean.stride(2),
        key_mean.stride(1),
        key_int8.stride(0),
        key_int8.stride(1),
        H=heads,
        HEAD_DIM=_HEAD_DIM,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )

    output = torch.empty_like(query)
    attn_grid = (triton.cdiv(max_seqlen, block_m), heads, batches)
    _fused_q_attn_kernel[attn_grid](
        query,
        key_int8,
        value,
        cu_seqlens,
        key_scale,
        cu_seqlens_k_scale,
        output,
        query.stride(0),
        query.stride(1),
        key_int8.stride(0),
        key_int8.stride(1),
        value.stride(0),
        value.stride(1),
        output.stride(0),
        output.stride(1),
        sm_scale_log2=sm_scale * _LOG2_E,
        H=heads,
        HEAD_DIM=_HEAD_DIM,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output

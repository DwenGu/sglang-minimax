"""MiniMax-H3 packed adapter for the native SageAttention2 SM90 CUDA op.

SageAttention2 v2.2.0 exposes only a dense SM90 QK-INT8/PV-FP8 CUDA kernel.
MiniMax-H3 carries one live packed interval plus a short alignment-padding
interval.  This adapter preserves the packed boundaries by preprocessing and
launching the native CUDA op independently for each interval, while writing
directly into one packed output tensor.

The TMA-based upstream kernel needs a tensor-map descriptor with one fixed
sequence extent per launch, so separate launches are intentional.  For the H3
layout the second launch is only 12 tokens and the long 37,748-token interval
dominates runtime.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch


def _nvtx_enabled() -> bool:
    return os.getenv(
        "SGLANG_DIFFUSION_ATTENTION_NVTX",
        os.getenv("SGLANG_SAGEATTENTION_NVTX", "0"),
    ).strip().lower() in {
        "1",
        "true",
    }


@contextmanager
def _attention_range(name: str, *, enabled: bool):
    """Emit the same sub-operation label to NVTX and the Chrome trace."""
    if not enabled:
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        with torch.profiler.record_function(name):
            yield
    finally:
        torch.cuda.nvtx.range_pop()


def is_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
    cu_seqlens_host: tuple[int, ...] | None,
) -> bool:
    if cu_seqlens_host is None or len(cu_seqlens_host) < 2:
        return False
    if tuple(cu_seqlens_host)[0] != 0 or tuple(cu_seqlens_host)[-1] != query.shape[0]:
        return False
    if any(end < start for start, end in zip(cu_seqlens_host, cu_seqlens_host[1:])):
        return False
    return (
        not is_causal
        and query.is_cuda
        and query.device == key.device == value.device
        and torch.cuda.get_device_capability(query.device) == (9, 0)
        and query.ndim == key.ndim == value.ndim == 3
        and query.dtype == key.dtype == value.dtype
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.shape == key.shape == value.shape
        and query.shape[-1] == 128
        and query.stride(-1) == key.stride(-1) == value.stride(-1) == 1
    )


@torch.inference_mode()
def sageattn2_h3_sm90_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens_host: tuple[int, ...],
    sm_scale: float,
) -> torch.Tensor:
    """Run the native SM90 CUDA attention kernel over packed H3 intervals."""
    if not is_supported(
        query,
        key,
        value,
        is_causal=False,
        cu_seqlens_host=cu_seqlens_host,
    ):
        raise ValueError("unsupported input for the H3 SM90 CUDA adapter")

    # These are the same official preprocessing functions used by
    # sageattn_qk_int8_pv_fp8_cuda_sm90. Q/K per-thread quantization is Triton;
    # V FP8 transpose/quantization and the attention body are native CUDA.
    from sageattention.core import per_channel_fp8, per_thread_int8_triton
    from sageattention import sm90_compile

    annotate = _nvtx_enabled()
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    for interval_index, (start, end) in enumerate(
        zip(cu_seqlens_host, cu_seqlens_host[1:])
    ):
        if start == end:
            continue
        interval_prefix = f"sageattention2.h3_sm90_cuda.interval_{interval_index}"
        with _attention_range(interval_prefix, enabled=annotate):
            q = query[start:end].unsqueeze(0)
            k = key[start:end].unsqueeze(0)
            v = value[start:end].unsqueeze(0)

            # Match the official smooth_k=True behavior. K centering is a
            # constant logit shift in exact attention and improves K
            # quantization accuracy.
            with _attention_range(
                f"{interval_prefix}.key_mean", enabled=annotate
            ):
                key_mean = k.mean(dim=1, keepdim=True)
            with _attention_range(
                f"{interval_prefix}.qk_int8_quant_triton", enabled=annotate
            ):
                q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(
                    q,
                    k,
                    key_mean,
                    tensor_layout="NHD",
                    BLKQ=64,
                    WARPQ=16,
                    BLKK=128,
                    WARPK=128,
                )

            # The native kernel consumes FP8 V through a TMA descriptor whose
            # sequence extent must cover complete 128-token tiles.
            length = end - start
            v_pad_len = (-length) % 128
            if v_pad_len:
                with _attention_range(
                    f"{interval_prefix}.v_padding", enabled=annotate
                ):
                    v = torch.cat(
                        (
                            v,
                            torch.zeros(
                                (1, v_pad_len, v.shape[2], v.shape[3]),
                                dtype=v.dtype,
                                device=v.device,
                            ),
                        ),
                        dim=1,
                    )
            with _attention_range(
                f"{interval_prefix}.v_fp8_quant_cuda", enabled=annotate
            ):
                v_fp8, v_scale, _ = per_channel_fp8(
                    v,
                    tensor_layout="NHD",
                    smooth_v=False,
                )

            interval_output = output[start:end].unsqueeze(0)
            with _attention_range(
                f"{interval_prefix}.attention_cuda", enabled=annotate
            ):
                sm90_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
                    q_int8,
                    k_int8,
                    v_fp8,
                    interval_output,
                    q_scale,
                    k_scale,
                    v_scale,
                    0,  # NHD
                    0,  # non-causal
                    3,  # per-thread Q/K quantization
                    sm_scale,
                    0,  # return_lse=False
                )
    return output

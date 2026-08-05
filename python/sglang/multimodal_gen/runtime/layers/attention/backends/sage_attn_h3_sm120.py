"""MiniMax-H3 packed adapter for SageAttention2's SM120 CUDA path.

SageAttention2 v2.2.0 exposes a dense SM120 dispatcher, while MiniMax-H3
reaches the diffusion backend as packed THD tensors.  Preserve every packed
boundary by invoking the official dense CUDA path once per interval.  This
module intentionally avoids the SageAttention2 Triton varlen implementation,
which upstream marks as unavailable on SM120.

Unlike the SM90 adapter, this implementation cannot write directly into a
caller-provided output buffer because the public SM120 API allocates its own
output.  The interval result is therefore copied into one packed output.  A
native SM120 varlen/output-buffer API would be the next optimization point.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.layers.attention.nvtx import attention_nvtx_range


def is_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
    cu_seqlens_host: tuple[int, ...] | None,
) -> bool:
    """Return whether inputs satisfy the conservative H3/SM120 contract."""
    if cu_seqlens_host is None or len(cu_seqlens_host) < 2:
        return False
    if cu_seqlens_host[0] != 0 or cu_seqlens_host[-1] != query.shape[0]:
        return False
    if any(end < start for start, end in zip(cu_seqlens_host, cu_seqlens_host[1:])):
        return False
    return (
        not is_causal
        and query.is_cuda
        and query.device == key.device == value.device
        and torch.cuda.get_device_capability(query.device) == (12, 0)
        and query.ndim == key.ndim == value.ndim == 3
        and query.dtype == key.dtype == value.dtype
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.shape == key.shape == value.shape
        and query.shape[-1] == 128
        and query.stride(-1) == key.stride(-1) == value.stride(-1) == 1
    )


@torch.inference_mode()
def sageattn2_h3_sm120_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens_host: tuple[int, ...],
    sm_scale: float,
) -> torch.Tensor:
    """Run SageAttention2's official SM120 dense CUDA path per H3 interval."""
    if not is_supported(
        query,
        key,
        value,
        is_causal=False,
        cu_seqlens_host=cu_seqlens_host,
    ):
        raise ValueError("unsupported input for the H3 SM120 CUDA adapter")

    # On compute capability 12.0 the public SageAttention2 v2.2.0 dispatcher
    # selects sageattn_qk_int8_pv_fp8_cuda with per-warp CUDA Q/K quantization
    # and fp32+fp16 accumulation.  Do not call sageattn_varlen here: it is a
    # Triton implementation and is not an SM120-safe fallback.
    from sageattention import sageattn

    output = torch.empty_like(query)
    for interval_index, (start, end) in enumerate(
        zip(cu_seqlens_host, cu_seqlens_host[1:])
    ):
        if start == end:
            continue
        interval_prefix = f"sageattention2.h3_sm120_cuda.interval_{interval_index}"
        with attention_nvtx_range(interval_prefix):
            with attention_nvtx_range(f"{interval_prefix}.dense_cuda_pipeline"):
                interval_output = sageattn(
                    query[start:end].unsqueeze(0),
                    key[start:end].unsqueeze(0),
                    value[start:end].unsqueeze(0),
                    tensor_layout="NHD",
                    is_causal=False,
                    sm_scale=sm_scale,
                    return_lse=False,
                )
            with attention_nvtx_range(f"{interval_prefix}.output_copy"):
                output[start:end].copy_(interval_output.squeeze(0))
    return output

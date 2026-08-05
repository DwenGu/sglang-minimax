# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0


import os
from contextlib import contextmanager

import torch
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (  # FlashAttentionMetadata,
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from sageattention import sageattn, sageattn_varlen

logger = init_logger(__name__)


def _sageattention_variant() -> str:
    variant = os.getenv("SGLANG_SAGEATTENTION_VARIANT", "").strip().lower()
    aliases = {
        "1": "1",
        "v1": "1",
        "sageattention1": "1",
        "2": "2",
        "v2": "2",
        "sageattention2": "2",
    }
    if variant not in aliases:
        raise RuntimeError(
            "SGLANG_SAGEATTENTION_VARIANT must select SageAttention 1 or 2; "
            f"got {variant!r}"
        )
    return aliases[variant]


@contextmanager
def _profile_range(name: str, *, nvtx_enabled: bool = False):
    """Annotate attention in both Chrome traces and Nsight/NVTX captures."""
    from sglang.multimodal_gen.runtime.utils.profiler import SGLDiffusionProfiler

    profiler_enabled = (
        torch.cuda.is_available() and SGLDiffusionProfiler.get_instance() is not None
    )
    if nvtx_enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        if profiler_enabled:
            with torch.profiler.record_function(name):
                yield
        else:
            yield
    finally:
        if nvtx_enabled:
            torch.cuda.nvtx.range_pop()


def _sage2_h3_fused_enabled() -> bool:
    value = os.getenv("SGLANG_SAGEATTENTION_FUSED_VARLEN", "0").strip().lower()
    if value not in {"0", "1", "false", "true"}:
        raise RuntimeError(
            f"SGLANG_SAGEATTENTION_FUSED_VARLEN must be 0/1/false/true; got {value!r}"
        )
    return value in {"1", "true"}


def _sage2_h3_sm90_cuda_enabled() -> bool:
    value = os.getenv("SGLANG_SAGEATTENTION_SM90_CUDA", "0").strip().lower()
    if value not in {"0", "1", "false", "true"}:
        raise RuntimeError(
            f"SGLANG_SAGEATTENTION_SM90_CUDA must be 0/1/false/true; got {value!r}"
        )
    return value in {"1", "true"}


def _sageattention_nvtx_enabled() -> bool:
    value = os.getenv(
        "SGLANG_DIFFUSION_ATTENTION_NVTX",
        os.getenv("SGLANG_SAGEATTENTION_NVTX", "0"),
    ).strip().lower()
    if value not in {"0", "1", "false", "true"}:
        raise RuntimeError(
            "SGLANG_DIFFUSION_ATTENTION_NVTX must be 0/1/false/true; "
            f"got {value!r}"
        )
    return value in {"1", "true"}


class SageAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SAGE_ATTN

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl


class SageAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)
        self.variant = _sageattention_variant()
        self.use_h3_fused_varlen = False
        self.use_h3_sm90_cuda = False
        self.use_nvtx = _sageattention_nvtx_enabled()
        if self.variant == "2":
            try:
                from sageattention import sageattn_qk_int8_pv_fp16_triton
            except ImportError as exc:
                raise RuntimeError(
                    "SageAttention2 was selected, but its v2 Triton kernel is "
                    "not available in the active sageattention package"
                ) from exc
            self._sageattn2 = sageattn_qk_int8_pv_fp16_triton
            self.use_h3_fused_varlen = _sage2_h3_fused_enabled()
            self.use_h3_sm90_cuda = _sage2_h3_sm90_cuda_enabled()
        logger.info_once(
            f"Using SageAttention{self.variant} for diffusion attention "
            f"(head_size={head_size})"
        )
        if self.use_h3_fused_varlen:
            logger.info_once(
                "Using the H20-tuned MiniMax-H3 fused SageAttention2 varlen kernel"
            )
        if self.use_h3_sm90_cuda:
            logger.info_once(
                "Using the native SageAttention2 SM90 CUDA kernel for MiniMax-H3"
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor:
        if self.variant == "1" and return_softmax_lse:
            raise NotImplementedError("SageAttention1 does not return softmax LSE")
        attention_fn = sageattn if self.variant == "1" else self._sageattn2
        with _profile_range(
            f"sageattention{self.variant}.dense", nvtx_enabled=self.use_nvtx
        ):
            output = attention_fn(
                query,
                key,
                value,
                # since input is (batch_size, seq_len, head_num, head_dim)
                tensor_layout="NHD",
                is_causal=self.causal,
                sm_scale=self.softmax_scale,
                return_lse=return_softmax_lse,
            )
        if return_softmax_lse:
            output, softmax_lse = output
            return output, softmax_lse
        return output

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if self.variant == "1":
            with _profile_range(
                "sageattention1.varlen", nvtx_enabled=self.use_nvtx
            ):
                return sageattn_varlen(
                    query,
                    key,
                    value,
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen,
                    max_seqlen,
                    is_causal=self.causal,
                    sm_scale=self.softmax_scale,
                )

        if self.use_h3_sm90_cuda:
            from sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn_h3_sm90 import (
                is_supported as is_sm90_supported,
                sageattn2_h3_sm90_cuda,
            )

            if is_sm90_supported(
                query,
                key,
                value,
                is_causal=self.causal,
                cu_seqlens_host=cu_seqlens_host,
            ):
                with _profile_range(
                    "sageattention2.h3_sm90_cuda", nvtx_enabled=self.use_nvtx
                ):
                    return sageattn2_h3_sm90_cuda(
                        query,
                        key,
                        value,
                        cu_seqlens_host=cu_seqlens_host,
                        sm_scale=self.softmax_scale,
                    )

        if self.use_h3_fused_varlen:
            from sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn_h3 import (
                is_supported,
                sageattn2_h3_varlen,
            )

            if is_supported(query, key, value, is_causal=self.causal):
                with _profile_range(
                    "sageattention2.h3_fused_varlen", nvtx_enabled=self.use_nvtx
                ):
                    return sageattn2_h3_varlen(
                        query,
                        key,
                        value,
                        cu_seqlens,
                        max_seqlen,
                        sm_scale=self.softmax_scale,
                    )

        # Preserve upstream SageAttention2 as a safe fallback for unsupported
        # shapes/dtypes and for A/B validation with the fusion disabled.
        with _profile_range(
            "sageattention2.varlen", nvtx_enabled=self.use_nvtx
        ):
            return sageattn_varlen(
                query,
                key,
                value,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                is_causal=self.causal,
                sm_scale=self.softmax_scale,
            )

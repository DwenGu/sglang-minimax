# SPDX-License-Identifier: Apache-2.0
"""Low-precision Ulysses attention using FlashInfer's prepared QKV API.

The H3 request prepares one communicator/workspace shared by its serialized
attention layers. FlashInfer owns statistics, quantization and input exchange;
stock SageAttention2 consumes the encoded result. The framework keeps its
existing output All2All and ordinary Sage fallback for unsupported routes.
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from math import prod as math_prod

import torch
from flashinfer.comm import UlyssesCommunicator, UlyssesQKV, UlyssesQKVWorkspace

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn import (
    SageAttentionImpl,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _a2a_staging_buffer,
)
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum

logger = logging.getLogger(__name__)

_HEAD_DIM = 128
_WORLD_SIZES = (2, 4, 8)
_SAGE_NHD, _SAGE_NON_CAUSAL, _SAGE_PER_WARP, _SAGE_NO_LSE = 0, 0, 2, 0


@functools.cache
def _sage_kernel(layout: str):
    if layout == "sage2_sm90":
        from sageattention import _qattn_sm90

        return _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf
    if layout == "sage2_sm89_sm120":
        from sageattention import _qattn_sm89

        return _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf
    raise ValueError(f"Unsupported FlashInfer QKV layout: {layout}")


@dataclass
class _PreparedQKV:
    comm: UlyssesCommunicator
    workspace: UlyssesQKVWorkspace
    encoded: UlyssesQKV | None = None


@contextmanager
def prepare_ulysses_qkv(impls, *, qkv_shape, used_sequence, dtype, device):
    """Collective request preparation; no per-layer cache or initialization."""
    if any(getattr(impl, "_prepared_qkv", None) is not None for impl in impls):
        raise RuntimeError("Overlapping requests cannot share a Lowp QKV workspace")
    with UlyssesCommunicator(
        get_sp_group().ulysses_group,
        # The original floating Q/K/V total is a conservative byte budget for
        # both Sage2 layouts. Preparation allocates only the actual wire size.
        max_bytes=3 * math_prod(qkv_shape) * dtype.itemsize,
        dtype=dtype,
        backend="nccl",
        device=device,
    ) as comm:
        workspace = comm.prepare_qkv(qkv_shape, used_sequence=used_sequence)
        state = _PreparedQKV(comm, workspace)
        for impl in impls:
            impl._prepared_qkv = state
        try:
            yield
        finally:
            for impl in impls:
                impl._prepared_qkv = None


class UlyssesLowpV2GBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [_HEAD_DIM]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.ULYSSES_LOWP_V2G

    @staticmethod
    def get_impl_cls() -> type[UlyssesLowpV2GImpl]:
        return UlyssesLowpV2GImpl


class UlyssesLowpV2GImpl(SageAttentionImpl):
    """``forward`` / ``forward_varlen`` are the inherited BF16 SageAttention2
    path (non-Ulysses components, fallbacks).  ``forward_varlen_ulysses`` is
    the low-precision path the MiniMax-H3 attention core calls BEFORE its own
    input all-to-all; it returns the post-exchange attention output
    ``[S_global, h_local, D]`` (the model's output all-to-all follows), or
    ``None`` to take the regular path."""

    _fallback_logged = False
    _prepared_qkv: _PreparedQKV | None = None

    def _route(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None,
        max_seqlen: int,
    ) -> tuple[int, int] | str:
        if torch.is_grad_enabled() or torch.cuda.is_current_stream_capturing():
            return "eager_only"
        if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
            return "shape"
        if (
            q.dtype not in (torch.bfloat16, torch.float16)
            or k.dtype != q.dtype
            or v.dtype != q.dtype
        ):
            return "dtype"
        local_sequence, num_heads, head_dim = q.shape
        if head_dim != _HEAD_DIM:
            return "head_dim"
        world_size = get_ulysses_parallel_world_size()
        if world_size not in _WORLD_SIZES or num_heads % world_size:
            return "world_size"
        global_sequence = local_sequence * world_size
        if (
            cu_seqlens_host is None
            or len(cu_seqlens_host) != 3
            or cu_seqlens_host[0] != 0
            or cu_seqlens_host[2] != global_sequence
            or not 0 < cu_seqlens_host[1] <= global_sequence
            or max_seqlen != cu_seqlens_host[1]
        ):
            return "cu_seqlens"
        if any(t.stride(-1) != 1 for t in (q, k, v)):
            return "stride"
        return world_size, int(cu_seqlens_host[1])

    def forward_varlen_ulysses(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens_host: tuple[int, ...] | None,
        max_seqlen: int,
    ) -> torch.Tensor | None:
        route = self._route(q, k, v, cu_seqlens_host, max_seqlen)
        if isinstance(route, str):
            if not UlyssesLowpV2GImpl._fallback_logged:
                UlyssesLowpV2GImpl._fallback_logged = True
                logger.info(
                    "ulysses_lowp_v2g: taking the BF16 Sage path (%s) before any "
                    "low-precision collective",
                    route,
                )
            return None
        world_size, used = route
        state = self._prepared_qkv
        if state is None:
            raise RuntimeError(
                "Lowp Ulysses must be prepared before attention; use the H3 "
                "denoise loop or model.prepare_ulysses_attention() on every rank"
            )
        if used != state.workspace.used_sequence:
            raise ValueError(
                "Request used_sequence differs from the prepared QKV workspace"
            )
        local_sequence, num_heads, head_dim = q.shape
        local_heads = num_heads // world_size
        global_sequence = local_sequence * world_size
        state.encoded = state.comm.scatter_qkv(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            workspace=state.workspace,
            out=state.encoded,
        )
        encoded = state.encoded
        with torch.cuda.nvtx.range("lowp_sage_attn"):
            out = _a2a_staging_buffer(
                "lowp_attn_out",
                (1, global_sequence, local_heads, head_dim),
                encoded.input_dtype,
                q.device,
            )
            if used < global_sequence:
                out[:, used:].zero_()
            _sage_kernel(encoded.layout)(
                encoded.q[:, :used],
                encoded.k[:, :used],
                encoded.v,
                out[:, :used],
                encoded.q_scale,
                encoded.k_scale,
                encoded.v_scale,
                _SAGE_NHD,
                _SAGE_NON_CAUSAL,
                _SAGE_PER_WARP,
                float(self.softmax_scale),
                _SAGE_NO_LSE,
            )
        return out[0]

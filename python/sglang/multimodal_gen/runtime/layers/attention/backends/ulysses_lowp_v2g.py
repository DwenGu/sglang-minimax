# SPDX-License-Identifier: Apache-2.0
"""Low-precision Ulysses all-to-all attention for MiniMax-H3 (V2-G).

Quantizes Q/K to INT8 and V to FP8 on SageAttention2's GLOBAL 32/64-token
grids BEFORE the sequence->head all-to-all, so the exchange moves ~half the
bytes of the BF16 path, and the receiver hands the pre-quantized operands
straight to SageAttention's SM120 kernel.  Every quantization primitive comes
from ``flashinfer.comm.ulysses_lowp``; the attention kernel is the stock
SageAttention ``_qattn_sm89`` binding (no fork, no patch).

Routing is decided by the packed-sequence padding alone: ``flashinfer``
picks stats protocol 3 (ALIGN-128 fast path) when the local shard is a whole
number of 128-token blocks and protocol 2 (boundary machinery) otherwise --
both produce byte-identical payloads wherever both are legal.  Pad the packed
global sequence to ``flashinfer.comm.ulysses_lowp.required_alignment(P, 3)``
(``SGLANG_MINIMAX_H3_PACKED_ALIGNMENT=128*P``) to take the fast path.

Non-Ulysses calls and any request the route guards decline run plain
SageAttention2 (inherited BF16 path).  All guards read rank-uniform state, so
the whole Ulysses group makes the same decision without a collective.
"""

from __future__ import annotations

import logging

import flashinfer.comm.ulysses_lowp as lowp
import torch
import torch.distributed as dist
from sageattention import _qattn_sm89
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_ulysses_parallel_rank,
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
    _usp_all_to_all_single,
)
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum

logger = logging.getLogger(__name__)

_HEAD_DIM = 128
_WORLD_SIZES = (2, 4, 8)
_SAGE_NHD, _SAGE_NON_CAUSAL, _SAGE_PER_WARP, _SAGE_NO_LSE = 0, 0, 2, 0


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

    def _route(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None,
        max_seqlen: int,
    ) -> tuple[int, int, int] | str:
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
        return world_size, get_ulysses_parallel_rank(), int(cu_seqlens_host[1])

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
        world_size, rank, used = route
        local_sequence, num_heads, head_dim = q.shape
        local_heads = num_heads // world_size
        global_sequence = local_sequence * world_size
        device = q.device
        q4, k4, v4 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)

        with torch.cuda.nvtx.range("lowp_local_stats"):
            send, ctx = lowp.local_stats(
                q4, k4, v4, rank=rank, world_size=world_size, used_sequence=used
            )
            gathered = _a2a_staging_buffer(
                "lowp_stats_gather", (world_size * send.numel(),), torch.float32, device
            )
        with torch.cuda.nvtx.range("lowp_stats_allgather"):
            dist.all_gather_into_tensor(
                gathered, send, group=get_sp_group().ulysses_group
            )
        with torch.cuda.nvtx.range("lowp_finalize_stats"):
            stats = lowp.finalize_stats(gathered, ctx, k4)

        spec = lowp.payload_spec(
            batch_size=1,
            local_sequence=local_sequence,
            num_heads=num_heads,
            head_dim=head_dim,
            world_size=world_size,
        )
        with torch.cuda.nvtx.range("lowp_quant_pack"):
            send_u8 = _a2a_staging_buffer(
                "lowp_qkv_send",
                (world_size, int(spec["chunk_bytes"])),
                torch.uint8,
                device,
            )
            lowp.quant_and_pack(q4, k4, v4, stats, out=send_u8)
        with torch.cuda.nvtx.range("lowp_input_a2a"):
            recv_u8 = _usp_all_to_all_single(send_u8, role="lowp_qkv_recv")

        with torch.cuda.nvtx.range("lowp_unpack"):
            q_width, k_width = lowp.scale_widths(used)
            q_int8 = _a2a_staging_buffer(
                "lowp_q_global",
                (1, global_sequence, local_heads, head_dim),
                torch.int8,
                device,
            )
            k_int8 = _a2a_staging_buffer(
                "lowp_k_global",
                (1, global_sequence, local_heads, head_dim),
                torch.int8,
                device,
            )
            v_fp8 = _a2a_staging_buffer(
                "lowp_v_packed",
                (1, head_dim, local_heads, int(spec["padded_sequence"])),
                torch.float8_e4m3fn,
                device,
            )
            q_scale = _a2a_staging_buffer(
                "lowp_q_scale", (1, local_heads, q_width), torch.float32, device
            )
            k_scale = _a2a_staging_buffer(
                "lowp_k_scale", (1, local_heads, k_width), torch.float32, device
            )
            lowp.unpack_for_sage(
                recv_u8,
                batch_size=1,
                local_sequence=local_sequence,
                local_heads=local_heads,
                head_dim=head_dim,
                world_size=world_size,
                aligned=None,
                scale_sequence=used,
                out=(q_int8, k_int8, v_fp8, q_scale, k_scale),
            )
            head_start = rank * local_heads
            v_scale = stats.v_scale_global[
                :, head_start : head_start + local_heads
            ].contiguous()

        with torch.cuda.nvtx.range("lowp_sage_attn"):
            out = _a2a_staging_buffer(
                "lowp_attn_out",
                (1, global_sequence, local_heads, head_dim),
                q.dtype,
                device,
            )
            if used < global_sequence:
                out[:, used:].zero_()
            _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
                q_int8[:, :used],
                k_int8[:, :used],
                v_fp8,
                out[:, :used],
                q_scale,
                k_scale,
                v_scale,
                _SAGE_NHD,
                _SAGE_NON_CAUSAL,
                _SAGE_PER_WARP,
                float(self.softmax_scale),
                _SAGE_NO_LSE,
            )
        return out[0]

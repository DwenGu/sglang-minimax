#!/usr/bin/env python3
"""Benchmark native SageAttention2 SM90 CUDA against the H3 packed adapter.

The default shape is the exact 4-GPU MiniMax-H3 attention layout observed for
the 1344x768, 124-frame validation request.  The native API is measured on the
37,748-token live dense interval; the adapter additionally preserves and
computes the 12-token packed padding interval.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Callable


DEPLOY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY_ROOT / "sglang" / "python"))
sys.path.insert(0, str(DEPLOY_ROOT / "sageattention" / "2.2.0"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sageattention import sageattn  # noqa: E402
from sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn_h3 import (  # noqa: E402
    sageattn2_h3_varlen,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn_h3_sm90 import (  # noqa: E402
    sageattn2_h3_sm90_cuda,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--total-tokens", type=int, default=37_760)
    parser.add_argument("--used-tokens", type=int, default=37_748)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-live-cosine", type=float, default=0.99999)
    parser.add_argument("--require-adapter-speedup", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def benchmark(
    fn: Callable[[], torch.Tensor], warmup: int, repeats: int
) -> tuple[float, float, torch.Tensor]:
    output = None
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        output = fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    assert output is not None
    return statistics.median(samples), statistics.mean(samples), output


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return F.cosine_similarity(
        left.float().flatten(), right.float().flatten(), dim=0
    ).item()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 < args.used_tokens < args.total_tokens:
        raise ValueError("used_tokens must be between zero and total_tokens")
    if args.head_dim != 128:
        raise ValueError("the native SM90 kernel requires head_dim=128")
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError("warmup and repeats must be positive")

    torch.cuda.set_device(args.device)
    capability = torch.cuda.get_device_capability(args.device)
    if capability != (9, 0):
        raise RuntimeError(f"SM90 is required, got {capability}")

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    shape = (args.total_tokens, args.heads, args.head_dim)
    query = torch.randn(
        shape, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    key = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    value = torch.randn(
        shape, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    boundaries = (0, args.used_tokens, args.total_tokens)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
    max_seqlen = max(args.used_tokens, args.total_tokens - args.used_tokens)
    sm_scale = args.head_dim**-0.5

    def native_dense_live() -> torch.Tensor:
        return sageattn(
            query[: args.used_tokens].unsqueeze(0),
            key[: args.used_tokens].unsqueeze(0),
            value[: args.used_tokens].unsqueeze(0),
            tensor_layout="NHD",
            is_causal=False,
            sm_scale=sm_scale,
        )

    def h3_fused_triton() -> torch.Tensor:
        return sageattn2_h3_varlen(
            query,
            key,
            value,
            cu_seqlens,
            max_seqlen,
            sm_scale=sm_scale,
        )

    def h3_sm90_adapter() -> torch.Tensor:
        return sageattn2_h3_sm90_cuda(
            query,
            key,
            value,
            cu_seqlens_host=boundaries,
            sm_scale=sm_scale,
        )

    native_median, native_mean, native_output = benchmark(
        native_dense_live, args.warmup, args.repeats
    )
    fused_median, fused_mean, fused_output = benchmark(
        h3_fused_triton, args.warmup, args.repeats
    )
    adapter_median, adapter_mean, adapter_output = benchmark(
        h3_sm90_adapter, args.warmup, args.repeats
    )

    live_cosine = cosine(
        native_output.squeeze(0), adapter_output[: args.used_tokens]
    )
    adapter_vs_fused = cosine(adapter_output, fused_output)
    finite = bool(torch.isfinite(adapter_output).all().item())
    overhead = (adapter_median / native_median - 1.0) * 100.0
    speedup = fused_median / adapter_median

    print("SageAttention2 SM90 native vs MiniMax-H3 packed adapter")
    print(f"GPU:          {torch.cuda.get_device_name(args.device)}")
    print(f"Shape:        {list(shape)} BF16")
    print(f"cu_seqlens:   {list(boundaries)}")
    print(f"Warmup/runs:  {args.warmup}/{args.repeats}")
    print()
    print(f"{'Path':<24} {'Median ms':>12} {'Mean ms':>12}")
    print(f"{'native dense live':<24} {native_median:>12.3f} {native_mean:>12.3f}")
    print(f"{'H3 fused Triton':<24} {fused_median:>12.3f} {fused_mean:>12.3f}")
    print(f"{'H3 SM90 adapter':<24} {adapter_median:>12.3f} {adapter_mean:>12.3f}")
    print()
    print(f"Adapter overhead vs native live: {overhead:+.3f}%")
    print(f"Adapter speedup vs fused Triton: {speedup:.4f}x")
    print(f"Native/adapter live cosine:      {live_cosine:.8f}")
    print(f"Adapter/fused packed cosine:     {adapter_vs_fused:.8f}")
    print(f"Adapter output finite:           {finite}")

    passed = finite and live_cosine >= args.min_live_cosine
    if args.require_adapter_speedup:
        passed = passed and adapter_median < fused_median
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

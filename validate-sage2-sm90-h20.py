#!/usr/bin/env python3
"""Validate the native SageAttention2 SM90 CUDA path on NVIDIA H20.

This test exercises the public ``sageattn()`` dispatcher after the v2.2.0
SM90 fake-registration name-collision fix. It compares the dispatched SM90
QK-INT8/PV-FP8 CUDA implementation with PyTorch SDPA and the official dense
QK-INT8/PV-FP16 Triton implementation.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
from pathlib import Path
from typing import Callable


DEPLOY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY_ROOT / "sageattention" / "2.2.0"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sageattention import (  # noqa: E402
    sageattn,
    sageattn_qk_int8_pv_fp16_triton,
)
import sageattention.sm90_compile as sm90_compile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--lengths",
        default="12,128,1024,4096,37748",
        help="Comma-separated dense sequence lengths.",
    )
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-cuda-cosine", type=float, default=0.99)
    parser.add_argument(
        "--require-long-speedup",
        action="store_true",
        help="Require CUDA to beat Triton for the largest requested length.",
    )
    return parser.parse_args()


@torch.inference_mode()
def benchmark(
    fn: Callable[[], torch.Tensor], warmup: int, repeats: int
) -> tuple[float, float, torch.Tensor]:
    output = None
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        output = fn()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end))
    assert output is not None
    return statistics.median(values), statistics.mean(values), output


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return F.cosine_similarity(
        left.float().flatten(), right.float().flatten(), dim=0
    ).item()


def main() -> int:
    args = parse_args()
    lengths = [int(value) for value in args.lengths.split(",") if value.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("all lengths must be positive")
    if args.warmup <= 0 or args.repeats <= 0:
        raise ValueError("warmup and repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.cuda.set_device(args.device)
    capability = torch.cuda.get_device_capability(args.device)
    if capability != (9, 0):
        raise RuntimeError(f"SM90 is required, got capability {capability}")

    real_wrapper = getattr(
        sm90_compile,
        "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf",
    )
    fake_wrapper = getattr(
        sm90_compile,
        "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_fake_impl",
        None,
    )
    if not isinstance(real_wrapper, torch.library.CustomOpDef):
        raise RuntimeError(
            "SM90 real custom-op wrapper is not bound; the v2.2.0 fake "
            "registration may still be shadowing it"
        )
    if fake_wrapper is None:
        raise RuntimeError("SM90 fake implementation is not separately registered")

    print("SageAttention2 native SM90 CUDA validation")
    print(f"GPU:             {torch.cuda.get_device_name(args.device)}")
    print(f"Capability:      {capability}")
    print(f"Heads/head_dim:  {args.heads}/{args.head_dim}")
    print(f"Lengths:         {lengths}")
    print(f"Warmup/repeats:  {args.warmup}/{args.repeats}")
    print(f"Real wrapper:    {real_wrapper}")
    print()
    print(
        f"{'N':>7} {'SDPA ms':>10} {'Triton ms':>11} {'SM90 ms':>10} "
        f"{'CUDA vs Tri':>12} {'Tri cosine':>12} {'CUDA cosine':>13} {'Finite':>8}"
    )

    passed = True
    largest_cuda_faster = False
    for index, length in enumerate(lengths):
        generator = torch.Generator(device=args.device).manual_seed(args.seed + length)
        shape = (1, length, args.heads, args.head_dim)
        query = torch.randn(
            shape, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        key = torch.randn(
            shape, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        value = torch.randn(
            shape, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        sm_scale = args.head_dim**-0.5

        def sdpa() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                is_causal=False,
                scale=sm_scale,
            ).transpose(1, 2)

        def triton_dense() -> torch.Tensor:
            return sageattn_qk_int8_pv_fp16_triton(
                query,
                key,
                value,
                tensor_layout="NHD",
                is_causal=False,
                sm_scale=sm_scale,
            )

        def sm90_cuda() -> torch.Tensor:
            # On SM90 the public dispatcher selects the native FP8 CUDA op.
            return sageattn(
                query,
                key,
                value,
                tensor_layout="NHD",
                is_causal=False,
                sm_scale=sm_scale,
            )

        # Alternate timing order across lengths to reduce persistent bias.
        if index % 2 == 0:
            sdpa_median, _, reference = benchmark(sdpa, args.warmup, args.repeats)
            triton_median, _, triton_output = benchmark(
                triton_dense, args.warmup, args.repeats
            )
            cuda_median, _, cuda_output = benchmark(
                sm90_cuda, args.warmup, args.repeats
            )
        else:
            cuda_median, _, cuda_output = benchmark(
                sm90_cuda, args.warmup, args.repeats
            )
            triton_median, _, triton_output = benchmark(
                triton_dense, args.warmup, args.repeats
            )
            sdpa_median, _, reference = benchmark(sdpa, args.warmup, args.repeats)

        triton_cosine = cosine(triton_output, reference)
        cuda_cosine = cosine(cuda_output, reference)
        cuda_finite = bool(torch.isfinite(cuda_output).all().item())
        delta = (cuda_median / triton_median - 1.0) * 100.0
        print(
            f"{length:>7d} {sdpa_median:>10.3f} {triton_median:>11.3f} "
            f"{cuda_median:>10.3f} {delta:>+11.2f}% "
            f"{triton_cosine:>12.8f} {cuda_cosine:>13.8f} "
            f"{str(cuda_finite):>8}"
        )
        passed = passed and cuda_finite and cuda_cosine >= args.min_cuda_cosine
        if length == max(lengths):
            largest_cuda_faster = cuda_median < triton_median

        del reference, triton_output, cuda_output
        gc.collect()
        torch.cuda.empty_cache()

    if args.require_long_speedup:
        passed = passed and largest_cuda_faster
    print(f"\nLargest-length CUDA faster: {largest_cuda_faster}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

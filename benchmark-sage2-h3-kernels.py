#!/usr/bin/env python3
"""Compare upstream SageAttention2 varlen with the H3 fused implementation.

The defaults reproduce the 4-GPU Ulysses attention shape observed for the
1344x768, 124-frame MiniMax-H3 t2va request used in this deployment:

  text rows:       38
  audio rows:     207 * 2 = 414
  video rows:      37 * (48 / 2) * (84 / 2) = 37,296
  used rows:       37,748
  alignment pad:   12
  packed Q/K/V:    [37,760, 14, 128] BF16
  cu_seqlens:      [0, 37,748, 37,760]

This is a CUDA microbenchmark, not a conventional CPU unit test. It performs
warmup before timing so Triton JIT compilation is excluded. CUDA Events time
GPU work; host-side Python launch overhead is intentionally not included.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Callable


DEPLOY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY_ROOT / "sglang" / "python"))
sys.path.insert(0, str(DEPLOY_ROOT / "sageattention" / "2.2.0"))

import torch  # noqa: E402
from sageattention import sageattn_varlen  # noqa: E402
from sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn_h3 import (  # noqa: E402
    sageattn2_h3_varlen,
)


DEFAULT_TOTAL_TOKENS = 37_760
DEFAULT_USED_TOKENS = 37_748
DEFAULT_HEADS = 14
DEFAULT_HEAD_DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--total-tokens", type=int, default=DEFAULT_TOTAL_TOKENS)
    parser.add_argument("--used-tokens", type=int, default=DEFAULT_USED_TOKENS)
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument(
        "--require-speedup",
        action="store_true",
        help="Exit nonzero unless fused median latency is lower than upstream.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optionally write the full result as JSON.",
    )
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
        "stdev_ms": statistics.pstdev(values),
    }


@torch.inference_mode()
def warmup(fn: Callable[[], torch.Tensor], iterations: int) -> None:
    output = None
    for _ in range(iterations):
        output = fn()
    torch.cuda.synchronize()
    del output


@torch.inference_mode()
def measure_latency(
    fn: Callable[[], torch.Tensor], iterations: int
) -> tuple[list[float], torch.Tensor]:
    values: list[float] = []
    output = None
    for _ in range(iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        output = fn()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end))
    assert output is not None
    return values, output


@torch.inference_mode()
def measure_incremental_peak_bytes(fn: Callable[[], torch.Tensor]) -> int:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    incremental_peak = torch.cuda.max_memory_allocated() - baseline
    del output
    torch.cuda.synchronize()
    return incremental_peak


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup < 1 or args.repeats < 1 or args.rounds < 1:
        raise ValueError("warmup, repeats, and rounds must all be positive")
    if not 0 < args.used_tokens < args.total_tokens:
        raise ValueError("used_tokens must be between 0 and total_tokens")
    if args.head_dim != 128:
        raise ValueError("the H3 fused kernel requires head_dim=128")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    dtype = getattr(torch, args.dtype)
    padding_tokens = args.total_tokens - args.used_tokens
    cu_seqlens = torch.tensor(
        [0, args.used_tokens, args.total_tokens],
        dtype=torch.int32,
        device=device,
    )
    max_seqlen = max(args.used_tokens, padding_tokens)
    sm_scale = args.head_dim**-0.5

    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = (args.total_tokens, args.heads, args.head_dim)
    query = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    key = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    value = torch.randn(shape, dtype=dtype, device=device, generator=generator)

    def upstream() -> torch.Tensor:
        return sageattn_varlen(
            query,
            key,
            value,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            is_causal=False,
            sm_scale=sm_scale,
        )

    def h3_fused() -> torch.Tensor:
        return sageattn2_h3_varlen(
            query,
            key,
            value,
            cu_seqlens,
            max_seqlen,
            sm_scale=sm_scale,
        )

    print("MiniMax-H3 SageAttention2 kernel comparison")
    print(f"GPU:          {torch.cuda.get_device_name(args.device)}")
    print(f"Capability:   {torch.cuda.get_device_capability(args.device)}")
    print(f"Shape:        {list(shape)}")
    print(f"Dtype:        {dtype}")
    print(f"cu_seqlens:   {[0, args.used_tokens, args.total_tokens]}")
    print(f"Warmup:       {args.warmup} per implementation")
    print(f"Timed calls:  {args.repeats * args.rounds} per implementation")
    print()

    # Compile and warm both implementations before collecting any samples.
    warmup(upstream, args.warmup)
    warmup(h3_fused, args.warmup)

    upstream_times: list[float] = []
    fused_times: list[float] = []
    upstream_output = None
    fused_output = None
    # Alternate order by round to reduce bias from clock/temperature drift.
    for round_index in range(args.rounds):
        if round_index % 2 == 0:
            values, upstream_output = measure_latency(upstream, args.repeats)
            upstream_times.extend(values)
            values, fused_output = measure_latency(h3_fused, args.repeats)
            fused_times.extend(values)
        else:
            values, fused_output = measure_latency(h3_fused, args.repeats)
            fused_times.extend(values)
            values, upstream_output = measure_latency(upstream, args.repeats)
            upstream_times.extend(values)

    assert upstream_output is not None and fused_output is not None
    upstream_f32 = upstream_output.float().flatten()
    fused_f32 = fused_output.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(
        upstream_f32, fused_f32, dim=0
    ).item()
    mean_abs_error = (upstream_f32 - fused_f32).abs().mean().item()
    max_abs_error = (upstream_f32 - fused_f32).abs().max().item()
    upstream_finite = bool(torch.isfinite(upstream_output).all().item())
    fused_finite = bool(torch.isfinite(fused_output).all().item())
    del upstream_f32, fused_f32, upstream_output, fused_output

    upstream_peak = measure_incremental_peak_bytes(upstream)
    fused_peak = measure_incremental_peak_bytes(h3_fused)
    upstream_summary = summarize(upstream_times)
    fused_summary = summarize(fused_times)
    latency_reduction_pct = (
        1.0 - fused_summary["median_ms"] / upstream_summary["median_ms"]
    ) * 100.0
    speedup = upstream_summary["median_ms"] / fused_summary["median_ms"]
    memory_reduction_bytes = upstream_peak - fused_peak

    print("Latency (CUDA Event, milliseconds)")
    print(
        f"{'Implementation':<28} {'Median':>10} {'Mean':>10} "
        f"{'P95':>10} {'Min':>10} {'Max':>10}"
    )
    for name, summary in (
        ("Upstream sageattn_varlen", upstream_summary),
        ("H3 fused sageattn2", fused_summary),
    ):
        print(
            f"{name:<28} {summary['median_ms']:>10.3f} "
            f"{summary['mean_ms']:>10.3f} {summary['p95_ms']:>10.3f} "
            f"{summary['min_ms']:>10.3f} {summary['max_ms']:>10.3f}"
        )
    print()
    print(f"Median latency reduction: {latency_reduction_pct:+.3f}%")
    print(f"Median speedup:           {speedup:.4f}x")
    print()
    print("Incremental peak allocated memory")
    print(f"Upstream:                 {upstream_peak / 2**20:.1f} MiB")
    print(f"H3 fused:                 {fused_peak / 2**20:.1f} MiB")
    print(f"Reduction:                {memory_reduction_bytes / 2**20:.1f} MiB")
    print()
    print("Numerical comparison")
    print(f"Upstream finite:          {upstream_finite}")
    print(f"H3 fused finite:          {fused_finite}")
    print(f"Cosine similarity:        {cosine:.8f}")
    print(f"Mean absolute error:      {mean_abs_error:.8f}")
    print(f"Max absolute error:       {max_abs_error:.8f}")

    passed = upstream_finite and fused_finite and cosine >= args.min_cosine
    if args.require_speedup:
        passed = passed and fused_summary["median_ms"] < upstream_summary["median_ms"]

    result = {
        "passed": passed,
        "environment": {
            "gpu": torch.cuda.get_device_name(args.device),
            "capability": list(torch.cuda.get_device_capability(args.device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "input": {
            "shape": list(shape),
            "dtype": args.dtype,
            "cu_seqlens": [0, args.used_tokens, args.total_tokens],
            "max_seqlen": max_seqlen,
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "rounds": args.rounds,
        },
        "upstream": {
            **upstream_summary,
            "incremental_peak_bytes": upstream_peak,
        },
        "h3_fused": {
            **fused_summary,
            "incremental_peak_bytes": fused_peak,
        },
        "comparison": {
            "median_latency_reduction_pct": latency_reduction_pct,
            "median_speedup": speedup,
            "memory_reduction_bytes": memory_reduction_bytes,
            "cosine_similarity": cosine,
            "mean_absolute_error": mean_abs_error,
            "max_absolute_error": max_abs_error,
            "upstream_finite": upstream_finite,
            "h3_fused_finite": fused_finite,
        },
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON result:              {args.output_json}")

    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

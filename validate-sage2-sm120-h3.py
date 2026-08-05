#!/usr/bin/env python3
"""Validate SageAttention2 SM120 CUDA with MiniMax-H3 packed exact shapes.

The test first compares the official dense SM120 dispatcher with PyTorch SDPA
on tractable sequence lengths.  It then executes the MiniMax-H3 packed layout
``[37760, 14, 128]`` with boundaries ``[0, 37748, 37760]``, reports CUDA-event
latency and peak memory, and writes a machine-readable JSON result.

Run this only after SageAttention2 has been compiled for ``sm_120``.  The
companion ``prepare-and-validate-sage2-sm120.sh`` performs that build and calls
this script with the correct source path.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--total-tokens", type=int, default=37_760)
    parser.add_argument("--used-tokens", type=int, default=37_748)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--reference-lengths",
        default="12,127,511",
        help="Comma-separated small lengths used for SDPA numerical checks.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument(
        "--sageattention-root",
        type=Path,
        help="SageAttention source/install root to prepend to sys.path.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=SCRIPT_DIR / "run" / "sm120-h3-validation.json",
    )
    parser.add_argument(
        "--allow-missing-torch-sm120-arch",
        action="store_true",
        help="Do not fail when torch.cuda.get_arch_list omits sm_120.",
    )
    return parser.parse_args()


def parse_version(value: str | None) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", value or "")
    if match is None:
        raise RuntimeError(f"cannot parse CUDA version {value!r}")
    return int(match.group(1)), int(match.group(2))


@torch.inference_mode()
def benchmark(
    fn: Callable[[], torch.Tensor], warmup: int, repeats: int
) -> tuple[list[float], torch.Tensor]:
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
    return samples, output


def timing_summary(samples: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return F.cosine_similarity(
        left.float().flatten(), right.float().flatten(), dim=0
    ).item()


def mib(value: int) -> float:
    return value / 1024**2


def main() -> int:
    args = parse_args()
    if args.sageattention_root is not None:
        source_root = args.sageattention_root.expanduser().resolve()
        if not (source_root / "sageattention").is_dir():
            raise RuntimeError(f"invalid SageAttention root: {source_root}")
        sys.path.insert(0, str(source_root))
    else:
        source_root = None

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 < args.used_tokens < args.total_tokens:
        raise ValueError("used_tokens must be between zero and total_tokens")
    if args.head_dim != 128:
        raise ValueError("the validated MiniMax-H3 contract requires head_dim=128")
    if args.heads <= 0 or args.warmup < 1 or args.repeats < 1:
        raise ValueError("heads, warmup, and repeats must be positive")

    reference_lengths = [
        int(value) for value in args.reference_lengths.split(",") if value.strip()
    ]
    if not reference_lengths or any(length <= 0 for length in reference_lengths):
        raise ValueError("all reference lengths must be positive")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        raise RuntimeError(f"SM120 is required, got compute capability {capability}")
    cuda_version = parse_version(torch.version.cuda)
    if cuda_version < (12, 8):
        raise RuntimeError(
            f"CUDA >= 12.8 is required for SM120, got {torch.version.cuda}"
        )

    torch_arches = torch.cuda.get_arch_list()
    if "sm_120" not in torch_arches and not args.allow_missing_torch_sm120_arch:
        raise RuntimeError(
            "this PyTorch build does not advertise sm_120; install an SM120-capable "
            f"build or use --allow-missing-torch-sm120-arch after verification: {torch_arches}"
        )

    from sageattention import sageattn
    import sageattention
    import sageattention._qattn_sm89 as sm120_extension

    sage_module_path = Path(sageattention.__file__).resolve()
    if source_root is not None and source_root not in sage_module_path.parents:
        raise RuntimeError(
            f"imported SageAttention from {sage_module_path}, expected {source_root}"
        )

    extension_path = Path(sm120_extension.__file__).resolve()
    print("SageAttention2 SM120 × MiniMax-H3 validation")
    print(f"GPU:                {torch.cuda.get_device_name(device)}")
    print(f"Compute capability: {capability}")
    print(f"PyTorch/CUDA:       {torch.__version__} / {torch.version.cuda}")
    print(f"Torch arch list:    {torch_arches}")
    print(f"SageAttention:      {sage_module_path}")
    print(f"CUDA extension:     {extension_path}")
    print()

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "result": "FAIL",
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "device": args.device,
            "compute_capability": list(capability),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_arch_list": torch_arches,
            "sageattention_module": str(sage_module_path),
            "sm120_extension": str(extension_path),
        },
        "reference_checks": [],
    }

    passed = True
    sm_scale = args.head_dim**-0.5
    print("Small-shape numerical checks")
    print(
        f"{'N':>7} {'SDPA median':>13} {'Sage median':>13} {'cosine':>12} {'finite':>8}"
    )
    for length in reference_lengths:
        generator = torch.Generator(device=device).manual_seed(args.seed + length)
        shape = (1, length, args.heads, args.head_dim)
        query = torch.randn(
            shape, device=device, dtype=torch.bfloat16, generator=generator
        )
        key = torch.randn(
            shape, device=device, dtype=torch.bfloat16, generator=generator
        )
        value = torch.randn(
            shape, device=device, dtype=torch.bfloat16, generator=generator
        )

        def sdpa() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                is_causal=False,
                scale=sm_scale,
            ).transpose(1, 2)

        def sage_dense() -> torch.Tensor:
            # The public dispatcher must select the official SM120 CUDA path:
            # per-warp CUDA Q/K quantization and FP8 V/attention.
            return sageattn(
                query,
                key,
                value,
                tensor_layout="NHD",
                is_causal=False,
                sm_scale=sm_scale,
                return_lse=False,
            )

        sdpa_samples, reference = benchmark(sdpa, args.warmup, args.repeats)
        sage_samples, sage_output = benchmark(sage_dense, args.warmup, args.repeats)
        similarity = cosine(sage_output, reference)
        finite = bool(torch.isfinite(sage_output).all().item())
        check_passed = finite and similarity >= args.min_cosine
        passed = passed and check_passed
        reference_result = {
            "length": length,
            "shape": list(shape),
            "sdpa": timing_summary(sdpa_samples),
            "sageattention2_sm120": timing_summary(sage_samples),
            "cosine": similarity,
            "finite": finite,
            "passed": check_passed,
        }
        report["reference_checks"].append(reference_result)
        print(
            f"{length:>7d} {statistics.median(sdpa_samples):>13.3f} "
            f"{statistics.median(sage_samples):>13.3f} {similarity:>12.8f} "
            f"{str(finite):>8}"
        )
        del query, key, value, reference, sage_output
        gc.collect()
        torch.cuda.empty_cache()

    boundaries = (0, args.used_tokens, args.total_tokens)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    exact_shape = (args.total_tokens, args.heads, args.head_dim)
    query = torch.randn(
        exact_shape, device=device, dtype=torch.bfloat16, generator=generator
    )
    key = torch.randn(
        exact_shape, device=device, dtype=torch.bfloat16, generator=generator
    )
    value = torch.randn(
        exact_shape, device=device, dtype=torch.bfloat16, generator=generator
    )

    def dense_interval(start: int, end: int) -> torch.Tensor:
        return sageattn(
            query[start:end].unsqueeze(0),
            key[start:end].unsqueeze(0),
            value[start:end].unsqueeze(0),
            tensor_layout="NHD",
            is_causal=False,
            sm_scale=sm_scale,
            return_lse=False,
        )

    def packed_h3() -> torch.Tensor:
        output = torch.empty_like(query)
        for start, end in zip(boundaries, boundaries[1:]):
            interval_output = dense_interval(start, end)
            output[start:end].copy_(interval_output.squeeze(0))
        return output

    allocated_before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    packed_samples, packed_output = benchmark(packed_h3, args.warmup, args.repeats)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    exact_finite = bool(torch.isfinite(packed_output).all().item())
    passed = passed and exact_finite

    interval_results = []
    for start, end in zip(boundaries, boundaries[1:]):
        samples, interval_output = benchmark(
            lambda start=start, end=end: dense_interval(start, end),
            args.warmup,
            args.repeats,
        )
        interval_results.append(
            {
                "start": start,
                "end": end,
                "tokens": end - start,
                **timing_summary(samples),
            }
        )
        del interval_output

    exact_result = {
        "shape": list(exact_shape),
        "dtype": "bfloat16",
        "cu_seqlens": list(boundaries),
        "packed_adapter": timing_summary(packed_samples),
        "intervals": interval_results,
        "finite": exact_finite,
        "memory_mib": {
            "allocated_before_adapter": mib(allocated_before),
            "peak_allocated": mib(peak_allocated),
            "peak_reserved": mib(peak_reserved),
            "additional_peak_allocated": mib(peak_allocated - allocated_before),
        },
    }
    report["exact_h3"] = exact_result
    report["result"] = "PASS" if passed else "FAIL"

    print()
    print("Exact MiniMax-H3 packed shape")
    print(f"Shape:              {list(exact_shape)} BF16")
    print(f"cu_seqlens:         {list(boundaries)}")
    print(
        "Packed median/mean: "
        f"{statistics.median(packed_samples):.3f} / "
        f"{statistics.mean(packed_samples):.3f} ms"
    )
    for interval in interval_results:
        print(
            f"Interval {interval['start']}:{interval['end']} "
            f"({interval['tokens']} tokens): {interval['median_ms']:.3f} ms median"
        )
    print(f"Peak allocated:     {mib(peak_allocated):.1f} MiB")
    print(f"Peak reserved:      {mib(peak_reserved):.1f} MiB")
    print(f"Output finite:      {exact_finite}")

    output_path = args.json_output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"JSON report:        {output_path}")
    print(f"RESULT:             {report['result']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

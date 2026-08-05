"""Opt-in NVTX/record-function annotations for diffusion attention."""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch


def attention_nvtx_enabled() -> bool:
    """Return whether detailed diffusion-attention annotations are enabled."""
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


@contextmanager
def attention_nvtx_range(name: str):
    """Emit a matching CPU record-function and GPU NVTX range when enabled."""
    if not attention_nvtx_enabled():
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        with torch.profiler.record_function(name):
            yield
    finally:
        torch.cuda.nvtx.range_pop()

"""Opt-in NVTX-only instrumentation for the timeline experiment."""

import os

if os.environ.get("H3_SAGE2_QK_CUDA") == "1":
    import functools

    from sageattention import core

    def force_cuda(name):
        original = getattr(core, name)

        @functools.wraps(original)
        def wrapped(q, k, v, **kwargs):
            kwargs["qk_quant_gran"] = "per_warp"
            return original(q, k, v, **kwargs)

        setattr(core, name, wrapped)

    force_cuda("sageattn_qk_int8_pv_fp8_cuda_sm90")
    force_cuda("sageattn_qk_int8_pv_fp8_cuda")
    print(f"SAGE2_QK_CUDA_PER_WARP_ACTIVE pid={os.getpid()}", flush=True)

if os.environ.get("H3_TIMELINE_NVTX") == "1":
    import functools

    import torch
    from sageattention import core

    from sglang.multimodal_gen.runtime.layers import usp
    from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
        FlashAttentionImpl,
    )

    def annotate(module, name, label):
        original = getattr(module, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            shapes = ";".join(
                str(tuple(a.shape)) for a in args[:3] if isinstance(a, torch.Tensor)
            )
            torch.cuda.nvtx.range_push(f"timeline::{label} [{shapes}]")
            try:
                return original(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()

        setattr(module, name, wrapped)

    annotate(usp, "_usp_input_all_to_all_packed_qkv", "bf16_input_a2a")
    annotate(usp, "_usp_output_all_to_all", "bf16_output_a2a")
    annotate(FlashAttentionImpl, "forward", "bf16_attention")
    annotate(core, "per_warp_int8_cuda", "stock_qk_quant")
    annotate(core, "per_thread_int8_triton", "stock_qk_quant_per_thread")
    annotate(core, "per_channel_fp8", "stock_v_quant")
    # A machine built only for SM120 need not have the SM90 extension.
    # Only annotate extensions that core actually imported successfully.
    for enabled, module, entry in [
        (
            "SM90_ENABLED",
            "sm90_compile",
            "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf",
        ),
        (
            "SM89_ENABLED",
            "sm89_compile",
            "qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf",
        ),
    ]:
        if getattr(core, enabled, False):
            annotate(
                getattr(core, module)._qattn_sm90
                if module == "sm90_compile"
                else getattr(core, module)._qattn_sm89,
                entry,
                "sage2_kernel",
            )
    print(f"TIMELINE_NVTX_ACTIVE pid={os.getpid()}", flush=True)

"""Inspect one fixed module: GPU0, denoising_step_1, transformer.blocks.25.attn."""

import csv
import re
import sqlite3

from pre_attention import LABELS, union_ns
from run import SCRATCH


def short(name):
    if name.startswith("nccl"):
        return name.split("(")[0]
    if any(
        k in name for k in ("qk_int8_sv_f8_attn_kernel", "qk_int_sv_f8_attn_kernel")
    ):
        gran = re.search(r"\(QuantGranularity\)(\d)", name).group(1)
        return f"qk_int8_sv_f8_attn_kernel (QuantGranularity={gran})"
    if "direct_copy_kernel" in name:
        return "aten::copy (" + ("BF16" if "BFloat16" in name else "float/Other") + ")"
    if "CatArrayBatchedCopy" in name:
        return "aten::CatArrayBatchedCopy_vectorized"
    if "reduce_kernel" in name:
        kind = (
            "mean"
            if "MeanOps" in name
            else (
                "max"
                if "MaxNanFunctor" in name
                else ("min" if "MinNanFunctor" in name else "Other")
            )
        )
        return f"aten::reduce_kernel ({kind})"
    if "FillFunctor" in name:
        return "aten::fill"
    if "CUDAFunctor_add" in name:
        return "aten::add"
    if "elementwise_kernel" in name:
        return "aten::elementwise_kernel"
    m = re.search(r"(?:flashinfer::ulysses_lowp::)?(\w+Kernel)(<[^>]*>)?", name)
    if m:
        return m.group(1) + (m.group(2) or "")
    return name.split("(")[0].replace("void ", "")


FUNCTIONAL_LABELS = [
    "Q: amax + INT8 quantization",
    "K: separate statistics + INT8 quantization",
    "V: separate FP8 quantization",
    "K/V: fused statistics",
    "Statistics helpers and finalization",
    "Statistics AllGather",
    "Input All2All",
    "Separate sending QKV pack",
    "Receiver BF16 contiguous copies",
    "V: separate padding and transpose",
    "Receiver Lowp unpack and tail clearing",
]


def functional_metrics(module):
    """Disjoint functional categories for the fixed pre-attention GPU window."""
    backend = module["backend"]
    sums = [0.0] * len(FUNCTIONAL_LABELS)
    for k in module["kernels"]:
        if k["phase"] != "Preprocessing":
            continue
        stage, name = k["stage"], k["name"]
        if "QuantInt8Kernel" in name:
            i = 0 if "(bool)0, (bool)0" in name else 1
        elif "GroupedAmaxKernel" in name or "QuantInt8GroupScalePackKernel" in name:
            i = 0 if "(bool)0" in name else 1
        elif "MeanOps" in name:
            i = 1
        elif "MeanScaleKernel" in name or "QuantVFP8WithScalePackKernel" in name:
            i = 2
        elif "KSumVAmax" in name:
            i = 3
        elif stage == "lowp_stats_allgather":
            i = 5
        elif "ncclDevKernel_SendRecv" in name:
            i = 6
        elif "_pack_qkv_destination_major_kernel" in name:
            i = 7
        elif "direct_copy_kernel" in name and backend == "sage2":
            i = 8
        elif backend == "sage2" and (
            "TransposePadPermuteKernel" in name
            or "FillFunctor" in name
            or "CatArrayBatchedCopy" in name
        ):
            i = 9
        elif stage in [
            "lowp_local_stats",
            "lowp_finalize_stats",
            "lowp_quant_pack",
        ]:
            i = 4
        elif backend == "lowp" and ("UnpackForSage" in name or "FillFunctor" in name):
            i = 10
        else:
            raise AssertionError((stage, name))
        sums[i] += k["us"]
    return sums


def inspect(up, backend, *, steps=4):
    work = SCRATCH / f"analysis_case1_up{up}_{backend}_{steps}steps"
    conn = sqlite3.connect(work / "trace.sqlite")
    pid = conn.execute(
        "SELECT DISTINCT p.pid FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN PROCESSES p USING(globalPid) WHERE k.deviceId=0"
    ).fetchone()[0]
    with (work / "projection_nvtx_gpu_proj_trace.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if int(r["PID"]) == pid]
    for r in rows:
        r["label"] = r["Name"].lstrip(":")
        for k in [
            "Orig Start (ns)",
            "Orig Duration (ns)",
            "Projected Start (ns)",
            "Projected Duration (ns)",
        ]:
            r[k] = int(r[k])
    step = next(r for r in rows if r["label"] == "denoising_step_1")

    def inside(r, outer):
        return (
            outer["Orig Start (ns)"]
            <= r["Orig Start (ns)"]
            < outer["Orig Start (ns)"] + outer["Orig Duration (ns)"]
        )

    matches = [
        r
        for r in rows
        if r["label"].startswith(
            "MiniMaxH3DenoisingStage.transformer.blocks.25.attn in="
        )
        and inside(r, step)
    ]
    assert len(matches) == 1
    module = matches[0]
    selected = [r for r in rows if inside(r, module)]
    first = next(r for r in selected if r["label"].split(" [")[0] == LABELS[backend][0])
    attn = next(r for r in selected if r["label"].startswith("timeline::sage2_kernel"))
    begin = module["Projected Start (ns)"]
    end = begin + module["Projected Duration (ns)"]
    pre = first["Projected Start (ns)"]
    core = attn["Projected Start (ns)"]
    core_end = core + attn["Projected Duration (ns)"]
    ks = conn.execute(
        "SELECT k.start,k.end,k.streamId,k.correlationId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id WHERE deviceId=0 AND k.start>=? AND k.end<=? ORDER BY k.start",
        (begin, end),
    ).fetchall()
    kernels = []
    labels = LABELS[backend] + ["timeline::bf16_output_a2a", "timeline::sage2_kernel"]
    for a, b, stream, corr, name in ks:
        phase = (
            "Upstream QKV / RoPE"
            if a < pre
            else (
                "Preprocessing"
                if a < core
                else ("Attention kernel" if a < core_end else "Downstream output")
            )
        )
        containing = [
            r
            for r in selected
            if any(r["label"].startswith(x) for x in labels)
            and r["Projected Start (ns)"] <= a
            and b <= r["Projected Start (ns)"] + r["Projected Duration (ns)"]
        ]
        stage = (
            min(containing, key=lambda r: r["Projected Duration (ns)"])["label"].split(
                " ["
            )[0]
            if containing
            else "No separate marker"
        )
        kernels.append(
            dict(
                phase=phase,
                stage=stage,
                name=name,
                short=short(name),
                start_ns=a,
                end_ns=b,
                offset_us=(a - begin) / 1e3,
                pre_offset_us=(a - pre) / 1e3,
                us=(b - a) / 1e3,
                stream=stream,
                correlation_id=corr,
            )
        )
    assert (
        sum(
            any(
                n in k["name"]
                for n in ("qk_int8_sv_f8_attn_kernel", "qk_int_sv_f8_attn_kernel")
            )
            for k in kernels
        )
        == 1
    )
    pre_ops = [k for k in kernels if k["phase"] == "Preprocessing"]
    result = dict(
        up=up,
        backend=backend,
        pid=pid,
        module=module["label"],
        step="denoising_step_1",
        module_begin_ns=begin,
        module_end_ns=end,
        pre_begin_ns=pre,
        attention_begin_ns=core,
        module_span_us=(end - begin) / 1e3,
        pre_span_us=(core - pre) / 1e3,
        attention_us=(core_end - core) / 1e3,
        pre_active_us=union_ns([(k["start_ns"], k["end_ns"]) for k in pre_ops]) / 1e3,
        kernels=kernels,
    )
    conn.close()
    return result

"""Compare pre-attention ranges and complete GPU windows from existing traces."""

import bisect
import csv
import sqlite3
from collections import defaultdict

from run import SCRATCH

LABELS = {
    "lowp": [
        "lowp_local_stats",
        "lowp_stats_allgather",
        "lowp_finalize_stats",
        "lowp_quant_pack",
        "lowp_input_a2a",
        "lowp_unpack",
    ],
    "sage2": [
        "timeline::bf16_input_a2a",
        "timeline::stock_qk_quant",
        "timeline::stock_v_quant",
    ],
}


def union_ns(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return sum(b - a for a, b in merged)


def analyze(up, backend, *, steps=4, expected_calls=150):
    work = SCRATCH / f"analysis_case1_up{up}_{backend}_{steps}steps"
    conn = sqlite3.connect(work / "trace.sqlite")
    mapping = dict(
        conn.execute(
            "SELECT DISTINCT p.pid,k.deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN PROCESSES p USING(globalPid)"
        )
    )
    with (work / "projection_nvtx_gpu_proj_trace.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = r["Name"].lstrip(":").split(" [")[0]
        r["device"] = mapping.get(int(r["PID"]))
        for key in [
            "Orig Start (ns)",
            "Orig Duration (ns)",
            "Projected Start (ns)",
            "Projected Duration (ns)",
        ]:
            r[key] = int(r[key])
    results = []
    for device in range(8):
        loop = next(
            r for r in rows if r["device"] == device and r["label"] == "denoising_loop"
        )
        start = loop["Orig Start (ns)"]
        stop = start + loop["Orig Duration (ns)"]
        selected = [
            r
            for r in rows
            if r["device"] == device and start <= r["Orig Start (ns)"] < stop
        ]
        stages = {
            label: sorted(
                [r for r in selected if r["label"] == label],
                key=lambda r: r["Orig Start (ns)"],
            )
            for label in LABELS[backend]
        }
        attn = sorted(
            [r for r in selected if r["label"] == "timeline::sage2_kernel"],
            key=lambda r: r["Orig Start (ns)"],
        )
        assert len(attn) == expected_calls and all(
            len(rs) == expected_calls for rs in stages.values()
        )
        first = stages[LABELS[backend][0]]
        kernels = conn.execute(
            "SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id WHERE deviceId=? ORDER BY k.start",
            (device,),
        ).fetchall()
        starts = [k[0] for k in kernels]
        byname = defaultdict(lambda: [0, 0.0])
        span_ns = cpu_ns = active_ns = 0
        per_call = []
        for i, (f, a) in enumerate(zip(first, attn)):
            assert f["Orig Start (ns)"] < a["Orig Start (ns)"]
            if i + 1 < expected_calls:
                assert a["Orig Start (ns)"] < first[i + 1]["Orig Start (ns)"]
            begin = f["Projected Start (ns)"]
            end = a["Projected Start (ns)"]
            assert begin < end
            span_ns += end - begin
            cpu_ns += a["Orig Start (ns)"] - f["Orig Start (ns)"]
            ops = kernels[
                bisect.bisect_left(starts, begin) : bisect.bisect_left(starts, end)
            ]
            assert not any("qk_int8_sv_f8_attn_kernel" in n for _, _, n in ops)
            active_ns += union_ns([(x, min(y, end)) for x, y, _ in ops])
            for x, y, n in ops:
                byname[n][0] += 1
                byname[n][1] += (min(y, end) - x) / 1e6
            per_call.append((end - begin) / 1e6)
        results.append(
            dict(
                device=device,
                calls=expected_calls,
                stages_ms={
                    label: sum(r["Projected Duration (ns)"] for r in rs) / 1e6
                    for label, rs in stages.items()
                },
                gpu_span_ms=span_ns / 1e6,
                gpu_active_union_ms=active_ns / 1e6,
                cpu_span_ms=cpu_ns / 1e6,
                span_first_ms=per_call[0],
                span_median_ms=sorted(per_call)[len(per_call) // 2],
                span_max_ms=max(per_call),
                kernels=[
                    dict(name=n, count=v[0], ms=v[1])
                    for n, v in sorted(byname.items(), key=lambda x: -x[1][1])
                ],
            )
        )
    conn.close()
    return dict(up=up, backend=backend, devices=results)

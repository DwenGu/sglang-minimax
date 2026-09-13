"""Analyze captures in scratch storage; publish only one Markdown report."""

import csv
import sqlite3
import subprocess

from run import SCRATCH


def timeline(path, *, expected_updates=3):
    work = SCRATCH / ("analysis_" + path.stem)
    work.mkdir(exist_ok=True)
    db = work / "trace.sqlite"
    projection = work / "projection_nvtx_gpu_proj_trace.csv"
    with (work / "nsys.log").open("a") as log:
        if not db.exists():
            subprocess.run(
                ["nsys", "export", "--type=sqlite", "--output=" + str(db), str(path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if not projection.exists():
            subprocess.run(
                [
                    "nsys",
                    "stats",
                    "--report",
                    "nvtx_gpu_proj_trace",
                    "--format",
                    "csv",
                    "--output",
                    str(work / "projection"),
                    str(db),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    with sqlite3.connect(db) as conn:
        devices = dict(
            conn.execute(
                "SELECT DISTINCT p.pid,k.deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN PROCESSES p USING(globalPid)"
            )
        )
        with projection.open() as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["label"] = r["Name"].lstrip(":").split(" [")[0]
            r["device"] = devices.get(int(r["PID"]))
        loops = {r["device"]: r for r in rows if r["label"] == "denoising_loop"}
        assert set(loops) == set(range(8)), loops.keys()
        steps = [r for r in rows if r["label"].startswith("denoising_step_")]
        assert len(steps) == 8 * expected_updates, len(steps)
        loop = loops[0]
        begin = int(loop["Orig Start (ns)"])
        end = begin + int(loop["Orig Duration (ns)"])
        stages = {}
        for r in rows:
            if (
                r["device"] == 0
                and begin <= int(r["Orig Start (ns)"]) < end
                and r["label"].startswith(("lowp_", "timeline::"))
            ):
                v = stages.setdefault(r["label"], dict(count=0, gpu_ms=0.0))
                v["count"] += 1
                v["gpu_ms"] += int(r["Projected Duration (ns)"]) / 1e6
        required = (
            "lowp_input_a2a" if "_lowp_" in path.name else "timeline::bf16_input_a2a"
        )
        assert stages[required]["count"] == 50 * expected_updates, stages
        gpu_begin = int(loop["Projected Start (ns)"])
        gpu_end = gpu_begin + int(loop["Projected Duration (ns)"])
        kernels = conn.execute(
            "SELECT s.value, SUM(k.end-k.start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id WHERE k.deviceId=0 AND k.start>=? AND k.end<=? GROUP BY s.value ORDER BY 2 DESC",
            (gpu_begin, gpu_end),
        ).fetchall()
        categories = {"Sage2": 0.0, "BF16 attention": 0.0, "GEMM": 0.0, "Other": 0.0}
        for name, ms in kernels:
            n = name.lower()
            if any(
                k in n
                for k in ("qk_int8_sv_f8_attn_kernel", "qk_int_sv_f8_attn_kernel")
            ):
                category = "Sage2"
            elif "flash::" in n or "fmha" in n:
                category = "BF16 attention"
            elif any(s in n for s in ["gemm", "cutlass", "cublas", "wgmma", "nvjet"]):
                category = "GEMM"
            else:
                category = "Other"
            categories[category] += ms
        return dict(
            name=path.stem,
            updates_per_gpu=expected_updates,
            attention_calls=50 * expected_updates,
            loop_ms=(gpu_end - gpu_begin) / 1e6,
            stages=stages,
            top_kernels=kernels[:10],
            kernel_categories=categories,
            nvtx=conn.execute("SELECT count(*) FROM NVTX_EVENTS").fetchone()[0],
        )

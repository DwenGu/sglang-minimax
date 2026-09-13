"""Portable UP4/8 comparison: 50-step videos, 3-step NVTX traces, seed 2101."""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("results")
SCRATCH = Path("work")
MODEL_PATH = "/models/MiniMax-H3"
PORT = 30041
PROMPTS = [
    "A hummingbird hovering beside a bright red hibiscus flower, wings blurred in slow motion, macro close-up, sunlit garden background",
    "A street musician playing an acoustic guitar on a rainy evening sidewalk, warm streetlight reflections on wet pavement, close-up on hands and face",
    "Waves crashing against dark volcanic rocks at dusk, sea spray backlit by the setting sun, distant seabirds circling.",
]


def run_group(up, backend, label, trace=False, *, trace_steps=4):
    name = f"up{up}_{label}_" + ("trace" if trace else "video")
    work = SCRATCH / name
    if (work / "done.json").exists():
        return
    work.mkdir(parents=True, exist_ok=True)
    targets = ROOT / ("timelines" if trace else "videos")
    suffix = f"_{trace_steps}steps.nsys-rep" if trace else "_50steps.mp4"
    if list(targets.glob(f"case*_up{up}_{label}{suffix}")):
        raise FileExistsError("Existing results; choose a fresh output/scratch pair")
    port = PORT
    session = "h32101_" + name
    command = [
        "sglang",
        "serve",
        "--model-path",
        MODEL_PATH,
        "--port",
        str(port),
        "--num-gpus",
        "8",
        "--tp-size",
        str(8 // up),
        "--ulysses-degree",
        str(up),
        "--ring-degree",
        "1",
        "--attention-backend",
        backend,
        "--model-variant",
        "fl2va",
        "--performance-mode",
        "speed",
        "--use-fsdp-inference",
        "false",
        "--minimax-h3-adaln-online",
        "true",
        "--enable-torch-compile",
        "false",
        "--enable-layerwise-nvtx-marker",
        str(trace).lower(),
        "--warmup-mode",
        "server",
    ]
    env = dict(
        os.environ,
        H3_TIMELINE_NVTX=str(int(trace)),
        H3_SAGE2_QK_CUDA=str(int(backend == "sage_attn")),
    )
    if trace or backend == "sage_attn":
        env["PYTHONPATH"] = str(HERE / "nvtx") + os.pathsep + env.get("PYTHONPATH", "")

    def nsys(*args):
        with (work / "nsys.log").open("a") as log:
            subprocess.run(
                ["nsys", *args], stdout=log, stderr=subprocess.STDOUT, check=True
            )

    def request(case, steps, tag):
        request_dir = work / tag
        with (work / (tag + ".log")).open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "request.py"),
                    "--base-url",
                    f"http://127.0.0.1:{port}",
                    "--evidence",
                    str(request_dir),
                    "--server-workdir",
                    str(work),
                    "--short-edge",
                    "704",
                    "--seconds",
                    "5",
                    "--steps",
                    str(steps),
                    "--seed",
                    "2101",
                    "--prompt",
                    PROMPTS[case],
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        summary = json.loads((request_dir / "summary.json").read_text())
        files = [
            Path(f["path"]) for f in summary["files"] if f["path"].endswith(".mp4")
        ]
        assert len(files) == 1, summary
        return files[0]

    launch_command = (
        ["nsys", "launch", "--session-new=" + session, "--trace=cuda,nvtx"]
        if trace
        else []
    ) + command
    print(name + ": starting", flush=True)
    records = []
    with (work / "server.log").open("w") as log:
        proc = subprocess.Popen(
            launch_command,
            cwd=work,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 1200
        while True:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=3
                ) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(name + ": server failed to become ready")
            time.sleep(3)
        print(name + ": warming up", flush=True)
        request(0, trace_steps if trace else 50, "warmup").unlink()
        for case in range(1) if trace else range(3):
            stem = f"case{case + 1}_up{up}_{label}"
            print(name + ": " + stem, flush=True)
            offset = (work / "server.log").stat().st_size
            if trace:
                nsys(
                    "start",
                    "--session=" + session,
                    "--sample=none",
                    "--cpuctxsw=none",
                    "--output="
                    + str(ROOT / "timelines" / (stem + f"_{trace_steps}steps")),
                )
            video = request(case, trace_steps if trace else 50, f"case{case + 1}")
            if trace:
                nsys("stop", "--session=" + session)
                video.unlink()
            else:
                target = ROOT / "videos" / (stem + "_50steps.mp4")
                shutil.move(video, target)
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-i", str(target), "-f", "null", "-"],
                    check=True,
                )
            with (work / "server.log").open("rb") as log:
                log.seek(offset)
                content = re.sub(
                    r"\x1b\[[0-9;]*m", "", log.read().decode(errors="replace")
                )
            pipeline = re.findall(
                r"Pixel data generated successfully in ([0-9.]+) seconds", content
            )
            denoise = re.findall(
                r"\[MiniMaxH3DenoisingStage\] finished in ([0-9.]+) seconds", content
            )
            records.append(
                dict(
                    case=case + 1,
                    up=up,
                    backend=label,
                    trace=trace,
                    steps=trace_steps if trace else 50,
                    pipeline=float(pipeline[-1]),
                    denoise=float(denoise[-1]),
                )
            )
            (work / "records.json").write_text(json.dumps(records, indent=2))
    finally:
        if trace:
            nsys("shutdown", "--session=" + session, "--kill=sigterm")
        elif proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        deadline = time.monotonic() + 60
        while subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip():
            if time.monotonic() > deadline:
                raise RuntimeError("GPU processes remain after shutdown")
            time.sleep(2)
    (work / "done.json").write_text(json.dumps(records, indent=2))
    print(name + ": done", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="8 GPU H3 validation: 50-step videos, 3-step CUDA/NVTX traces"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--up", type=int, choices=[4, 8], default=8)
    parser.add_argument("--mode", choices=["all", "videos", "timelines"], default="all")
    args = parser.parse_args()
    ROOT, SCRATCH = args.output.resolve(), args.scratch.resolve()
    MODEL_PATH, PORT = args.model_path, args.port
    if ROOT == SCRATCH or ROOT in SCRATCH.parents or SCRATCH in ROOT.parents:
        parser.error("output and scratch must be separate, non-nested directories")
    for folder in ["videos", "timelines"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    import flashinfer
    import flashinfer.comm.ulysses_lowp as lowp
    import sageattention
    import torch

    import sglang

    caps = [
        torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())
    ]
    if len(caps) != 8 or len(set(caps)) != 1 or caps[0] not in [(9, 0), (12, 0)]:
        raise RuntimeError(f"Requires 8 visible homogeneous SM90 or SM120 GPUs: {caps}")
    if not lowp.capability("cuda")["supported"]:
        raise RuntimeError("FlashInfer layout is unavailable")
    packages = {}
    for module in [flashinfer, sageattention, sglang]:
        location = Path(module.__file__).resolve()
        repo = next(p for p in location.parents if (p / ".git").exists())
        packages[module.__name__] = dict(
            file=str(location),
            head=subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip(),
            status=subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"], text=True
            ),
        )
    environment = dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        devices=[torch.cuda.get_device_name(i) for i in range(8)],
        capabilities=caps,
        packages=packages,
        model_path=MODEL_PATH,
        up=args.up,
    )
    environment_path = SCRATCH / "environment.json"
    encoded = json.dumps(environment, indent=2)
    if environment_path.exists() and environment_path.read_text() != encoded:
        raise RuntimeError("Environment differs from existing scratch; use a new batch")
    environment_path.write_text(encoded)
    for backend, label in [
        ("ulysses_lowp_v2g", "lowp"),
        ("sage_attn", "sage2"),
        ("fa", "bf16"),
    ]:
        if args.mode in ["all", "videos"]:
            run_group(args.up, backend, label)
        if args.mode in ["all", "timelines"]:
            run_group(args.up, backend, label, trace=True, trace_steps=3)
    print("ALL CAPTURES COMPLETE", flush=True)

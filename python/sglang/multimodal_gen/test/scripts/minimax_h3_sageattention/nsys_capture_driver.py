#!/usr/bin/env python3
"""Run one warmed MiniMax-H3 request inside a parent-process NVTX range."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(
    os.environ.get("SGLANG_REPO_ROOT", str(Path(__file__).resolve().parents[6]))
)
DEPLOY_ROOT = Path(
    os.environ.get(
        "MINIMAX_H3_DEPLOY_ROOT",
        str(REPO_ROOT / "artifacts" / "minimax_h3_sageattention"),
    )
)
ATTENTION_MODE = os.environ.get("ATTENTION_MODE", "baseline")
RUN_TAG = os.environ.get("RUN_TAG", f"{ATTENTION_MODE}-nsys")
CAPTURE_RANGE = os.environ.get("NSYS_CAPTURE_RANGE", "minimax_h3_nsys_request")
PORT = int(os.environ.get("PORT", "30010"))
READY_TIMEOUT_SECONDS = int(os.environ.get("NSYS_READY_TIMEOUT_SECONDS", "900"))


def wait_until_ready(server: subprocess.Popen[bytes], log_path: Path) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    ready_message = "The server is fired up and ready to roll!"
    health_url = f"http://127.0.0.1:{PORT}/health"
    direct_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    while time.monotonic() < deadline:
        return_code = server.poll()
        if return_code is not None:
            raise RuntimeError(
                f"server exited during startup with code {return_code}: {log_path}"
            )

        if log_path.exists() and ready_message in log_path.read_text(errors="replace"):
            try:
                with direct_opener.open(health_url, timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                pass
        time.sleep(2)

    raise TimeoutError(
        f"server did not become ready within {READY_TIMEOUT_SECONDS}s: {log_path}"
    )


def stop_server(server: subprocess.Popen[bytes]) -> None:
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=60)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"server did not stop after SIGTERM (PID {server.pid})")


def main() -> int:
    log_path = DEPLOY_ROOT / "logs" / f"server-{RUN_TAG}.log"
    output_file = os.environ.get(
        "OUTPUT_FILE",
        str(DEPLOY_ROOT / "outputs" / f"minimax-h3-{RUN_TAG}.mp4"),
    )
    child_env = os.environ.copy()
    child_env.setdefault("SGLANG_DIFFUSION_ATTENTION_NVTX", "1")
    for proxy_bypass_variable in ("NO_PROXY", "no_proxy"):
        current_bypass = child_env.get(proxy_bypass_variable, "")
        child_env[proxy_bypass_variable] = (
            f"{current_bypass},127.0.0.1,localhost,0.0.0.0".lstrip(",")
        )

    with log_path.open("wb") as server_log:
        server = subprocess.Popen(
            [str(SCRIPT_DIR / "start_server.sh"), "--enable-layerwise-nvtx-marker"],
            cwd=REPO_ROOT,
            env=child_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            print(
                f"Waiting for warmed {ATTENTION_MODE} server (PID {server.pid})...",
                flush=True,
            )
            wait_until_ready(server, log_path)
            print(f"Starting nsys capture range: {CAPTURE_RANGE}", flush=True)

            # Nsight Systems 2025.5 in this image records NVTX correctly but
            # does not reliably trigger capture-range=nvtx. The CUDA Profiler
            # API is the tested capture switch; NVTX remains the analysis label.
            import torch

            torch.cuda.init()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push(CAPTURE_RANGE)
            try:
                validation_env = child_env | {
                    "ATTENTION_MODE": ATTENTION_MODE,
                    "RUN_TAG": RUN_TAG,
                    "PROFILE_TIMELINE": "0",
                    "NUM_INFERENCE_STEPS": os.environ.get("NUM_INFERENCE_STEPS", "3"),
                    "OUTPUT_FILE": output_file,
                }
                result = subprocess.run(
                    [str(SCRIPT_DIR / "validate_t2va.sh")],
                    cwd=REPO_ROOT,
                    env=validation_env,
                    check=False,
                )
            finally:
                torch.cuda.nvtx.range_pop()
                torch.cuda.cudart().cudaProfilerStop()
                print(f"Stopped nsys capture range: {CAPTURE_RANGE}", flush=True)

            return result.returncode
        finally:
            stop_server(server)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"nsys capture driver failed: {error}", file=sys.stderr, flush=True)
        raise

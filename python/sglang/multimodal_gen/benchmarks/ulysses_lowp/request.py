"""Submit a reproducible t2va request, with terminal-state and timeout handling."""

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--evidence", type=Path, required=True, help="New directory for this request"
    )
    parser.add_argument(
        "--server-workdir", type=Path, help="Server cwd, if accessible on this machine"
    )
    parser.add_argument("--short-edge", type=int, choices=(704, 768), default=768)
    parser.add_argument("--steps", type=int, choices=(3, 4, 50), default=50)
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture a diagnostic trace; exclude this request from timings",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    args.evidence.mkdir(parents=True, exist_ok=False)
    payload = dict(
        model="MiniMaxAI/MiniMax-H3",
        prompt=args.prompt,
        seconds=args.seconds,
        task="t2va",
        conditions=[],
        target=dict(
            short_edge=args.short_edge,
            aspect_ratio="16:9",
            duration_seconds=float(args.seconds),
        ),
        num_outputs_per_prompt=1,
        num_inference_steps=args.steps,
        flow_shift=12.0,
        audio_flow_shift=3.0,
        seed=args.seed,
    )
    if args.profile:
        payload.update(profile=True, num_profiled_timesteps=1)

    def save(name, obj):
        with (args.evidence / name).open("x") as f:
            json.dump(obj, f, indent=2)

    save("request.json", payload)
    if args.dry_run:
        print(args.evidence / "request.json")
        return
    start = time.monotonic()

    def call(path, body=None):
        remaining = args.timeout - (time.monotonic() - start)
        if remaining <= 0:
            raise TimeoutError("request deadline exceeded")
        request = urllib.request.Request(
            args.base_url.rstrip("/") + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer sglang-anything",
            },
        )
        with urllib.request.urlopen(request, timeout=min(60, remaining)) as response:
            return json.load(response)

    try:
        job = call("/v1/videos", payload)
        save("submitted.json", job)
        job_id = urllib.parse.quote(str(job["id"]), safe="")
        poll = 0
        while True:
            state = call("/v1/videos/" + job_id)
            save(f"status-{poll:04d}.json", state)
            poll += 1
            status = state.get("status")
            if status == "completed":
                break
            if status in (
                "failed",
                "error",
                "cancelled",
                "canceled",
                "expired",
                "rejected",
            ):
                raise RuntimeError(f"job reached failure state: {status}")
            remaining = args.timeout - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError("job polling deadline exceeded")
            time.sleep(min(10, remaining))
        summary = dict(
            job_id=job["id"],
            elapsed_seconds=time.monotonic() - start,
            video_status="completed; playback and lowp profile verification pending",
        )
        paths = state.get("output_file_paths") or state.get("output_paths") or []
        if state.get("file_path"):
            paths = [state["file_path"], *paths]
        summary["reported_paths"] = paths
        summary["files"] = []
        for name in dict.fromkeys(paths):
            path = Path(name)
            if not path.is_absolute():
                if args.server_workdir is None:
                    continue
                path = args.server_workdir / path
            if path.is_file():
                with path.open("rb") as f:
                    digest = hashlib.file_digest(f, "sha256").hexdigest()
                summary["files"].append(dict(path=str(path.resolve()), sha256=digest))
        save("summary.json", summary)
        print(json.dumps(summary, indent=2))
    except Exception as error:
        save(
            "failure.json",
            dict(error=str(error), elapsed_seconds=time.monotonic() - start),
        )
        raise


if __name__ == "__main__":
    main()

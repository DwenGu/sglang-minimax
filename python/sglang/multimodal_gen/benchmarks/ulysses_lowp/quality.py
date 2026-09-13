"""Full-video decoding and SSIM; similarity is not a quality ranking."""

import re
import subprocess


def decoded_hash(path):
    return subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "hash",
            "-hash",
            "sha256",
            "-",
        ],
        text=True,
    ).strip()


def ssim(first, second):
    p = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "info",
            "-i",
            str(first),
            "-i",
            str(second),
            "-lavfi",
            "ssim",
            "-an",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(re.findall(r"All:([0-9.]+)", p.stderr)[-1])

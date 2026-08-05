#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import av


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} VIDEO.mp4", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"video is missing or empty: {path}")

    with av.open(str(path)) as container:
        video_streams = list(container.streams.video)
        audio_streams = list(container.streams.audio)
        if not video_streams:
            raise SystemExit("MP4 has no video stream")
        if not audio_streams:
            raise SystemExit("MP4 has no audio stream")

        video = video_streams[0]
        audio = audio_streams[0]
        duration = (
            float(container.duration / av.time_base)
            if container.duration is not None
            else None
        )
        result = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "duration_seconds": duration,
            "video": {
                "codec": video.codec_context.name,
                "width": video.codec_context.width,
                "height": video.codec_context.height,
                "fps": float(video.average_rate) if video.average_rate else None,
                "frames": video.frames,
            },
            "audio": {
                "codec": audio.codec_context.name,
                "sample_rate": audio.codec_context.sample_rate,
                "channels": audio.codec_context.channels,
            },
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))

        if video.codec_context.name != "h264":
            raise SystemExit("unexpected video codec")
        if (video.codec_context.width, video.codec_context.height) != (1344, 768):
            raise SystemExit("unexpected video dimensions")
        if audio.codec_context.name != "aac":
            raise SystemExit("unexpected audio codec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

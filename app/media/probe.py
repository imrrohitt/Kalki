from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from app.config import settings


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str | None
    has_audio: bool
    audio_codec: str | None


class MediaError(Exception):
    pass


def probe_video(path: str) -> VideoInfo:
    cmd = [
        settings.ffprobe_path,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise MediaError("ffprobe not found. Install FFmpeg.") from exc
    except subprocess.CalledProcessError as exc:
        raise MediaError(f"ffprobe failed: {exc.stderr}") from exc

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video is None:
        raise MediaError("No video stream found.")

    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)

    fps = 30.0
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "30/1"
    if isinstance(rate, str) and "/" in rate:
        num, den = rate.split("/", 1)
        if float(den) != 0:
            fps = float(num) / float(den)

    return VideoInfo(
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        video_codec=video.get("codec_name"),
        has_audio=audio is not None,
        audio_codec=audio.get("codec_name") if audio else None,
    )


def validate_video(info: VideoInfo, max_duration_sec: float) -> None:
    if info.duration <= 0:
        raise MediaError("Could not determine video duration.")
    if info.duration > max_duration_sec:
        raise MediaError(
            f"Video too long ({info.duration:.1f}s). Max is {max_duration_sec:.0f}s."
        )
    if not info.has_audio:
        raise MediaError("Video has no audio stream.")

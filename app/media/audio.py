from __future__ import annotations

import subprocess
from pathlib import Path

from app.config import settings
from app.media.probe import MediaError


def extract_audio(video_path: str, audio_path: str) -> str:
    Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        settings.ffmpeg_path,
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise MediaError("ffmpeg not found. Install FFmpeg.") from exc
    except subprocess.CalledProcessError as exc:
        raise MediaError(f"audio extraction failed: {exc.stderr[-500:]}") from exc
    return audio_path

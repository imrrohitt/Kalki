from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from broll_search_api.config import settings


logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".m4v"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".svg"}


def _run(cmd: list[str]) -> bool:
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if completed.returncode != 0:
            logger.warning("ffmpeg failed: %s", completed.stderr[-400:])
            return False
        return True
    except (OSError, subprocess.TimeoutExpired):
        logger.exception("ffmpeg exec failed")
        return False


def prepare_clip(source: Path, dest_dir: Path) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{source.stem}_broll.mp4"
    suffix = source.suffix.lower()
    work = source
    if suffix == ".svg":
        png = dest_dir / f"{source.stem}.png"
        svg_ok = _run(
            [
                settings.ffmpeg_path,
                "-y",
                "-i",
                str(source),
                str(png),
            ]
        )
        if not svg_ok or not png.exists():
            return None
        work = png
        suffix = ".png"
    vf = (
        f"scale={settings.output_width}:{settings.output_height}:"
        "force_original_aspect_ratio=increase,"
        f"crop={settings.output_width}:{settings.output_height}"
    )
    if suffix in IMAGE_SUFFIXES or work.suffix.lower() in IMAGE_SUFFIXES:
        cmd = [
            settings.ffmpeg_path,
            "-y",
            "-loop",
            "1",
            "-i",
            str(work),
            "-t",
            str(settings.broll_clip_seconds),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(out),
        ]
    elif suffix in VIDEO_SUFFIXES or suffix in {".m3u8", ".mpd"}:
        cmd = [
            settings.ffmpeg_path,
            "-y",
            "-i",
            str(source),
            "-t",
            str(settings.broll_clip_seconds),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-an",
            str(out),
        ]
    else:
        return None

    if not _run(cmd):
        return None
    return out if out.exists() else None

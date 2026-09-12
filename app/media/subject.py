from __future__ import annotations

import logging
import statistics
import subprocess

from app.config import settings
from app.media.probe import probe_video

logger = logging.getLogger(__name__)

# Downsampled grid used to find the first row that is not the backdrop.
_COLS = 64
_ROWS = 112


def detect_head_top(
    source: str,
    *,
    width: int,
    height: int,
    timestamps: list[float] | None = None,
) -> int:
    """Y (output pixels) of the top of the talking head after the 9:16 crop.

    Captions should sit just above this line, in the empty wall space.
    """
    times = timestamps or _sample_times(source)
    tops: list[int] = []
    vf = (
        f"scale={_COLS}:{_ROWS}:force_original_aspect_ratio=increase,"
        f"crop={_COLS}:{_ROWS},format=gray"
    )
    for t in times:
        raw = _grab_gray(source, t, vf)
        if raw is None:
            continue
        row = _first_subject_row(raw, _COLS, _ROWS)
        if row is None:
            continue
        tops.append(int(round(row / _ROWS * height)))
    if not tops:
        fallback = int(height * 0.34)
        logger.info("head-top detect failed; using y=%s", fallback)
        return fallback
    y = int(statistics.median(tops))
    y = max(int(height * 0.22), min(y, int(height * 0.58)))
    logger.info("head-top y=%s (from %s samples)", y, len(tops))
    return y


def _sample_times(source: str) -> list[float]:
    try:
        duration = max(probe_video(source).duration, 1.0)
    except Exception:
        duration = 8.0
    picks = (0.12, 0.28, 0.48)
    times = [round(duration * p, 2) for p in picks]
    return [t for t in times if 0.05 < t < duration - 0.05] or [0.4]


def _grab_gray(source: str, timestamp: float, vf: str) -> bytes | None:
    cmd = [
        settings.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.2f}",
        "-i",
        source,
        "-vf",
        vf,
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except OSError:
        return None
    data = proc.stdout or b""
    if proc.returncode != 0 or len(data) < _COLS * _ROWS:
        return None
    return data[: _COLS * _ROWS]


def _backdrop_value(pixels: bytes, cols: int, rows: int) -> int:
    """Wall color from the side strips, skipping the ceiling band."""
    samples: list[int] = []
    y0 = max(6, rows // 10)
    y1 = max(y0 + 4, rows // 3)
    edge = max(4, cols // 10)
    for y in range(y0, y1):
        row = pixels[y * cols : (y + 1) * cols]
        samples.extend(row[:edge])
        samples.extend(row[-edge:])
    if not samples:
        return 128
    return sorted(samples)[len(samples) // 2]


def _first_subject_row(pixels: bytes, cols: int, rows: int) -> int | None:
    """First row where the center leaves the wall (hair), skipping the ceiling."""
    bg = _backdrop_value(pixels, cols, rows)
    threshold = 28
    start = max(6, rows // 10)
    left = cols // 5
    right = 4 * cols // 5
    seen_wall = False
    wall_rows = 0
    for y in range(start, rows - 6):
        row = pixels[y * cols : (y + 1) * cols]
        center = row[left:right]
        if not center:
            continue
        changed = sum(1 for p in center if abs(p - bg) > threshold) / len(center)
        if not seen_wall:
            if changed < 0.18:
                wall_rows += 1
                if wall_rows >= 2:
                    seen_wall = True
            else:
                wall_rows = 0
            continue
        if changed >= 0.45:
            return y
    return None

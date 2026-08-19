from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from broll_search_api.config import settings


logger = logging.getLogger(__name__)


@dataclass
class FileProbe:
    ok: bool
    kind: str
    mime: str
    bytes: int
    width: int = 0
    height: int = 0
    duration: float = 0.0
    codec: str = ""
    reason: str = ""


def sniff_mime(path: Path) -> str:
    data = path.read_bytes()[:64]
    if not data:
        return "empty"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data[4:8] == b"ftyp":
        return "video/mp4"
    if data.startswith(b"\x1aE\xdf\xa3"):
        return "video/webm"
    head = data.lstrip()[:40].lower()
    if head.startswith(b"<svg") or (head.startswith(b"<?xml") and b"svg" in path.read_bytes()[:400].lower()):
        return "image/svg+xml"
    if head.startswith(b"<!doct") or head.startswith(b"<html") or head.startswith(b"<head"):
        return "text/html"
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".ogv": "video/ogg",
    }.get(suffix, "application/octet-stream")


def probe_file(path: Path) -> FileProbe:
    if not path.exists() or not path.is_file():
        return FileProbe(ok=False, kind="missing", mime="", bytes=0, reason="file missing")
    size = path.stat().st_size
    if size < 512:
        return FileProbe(ok=False, kind="tiny", mime="", bytes=size, reason="file too small")
    mime = sniff_mime(path)
    if mime == "text/html":
        return FileProbe(ok=False, kind="html", mime=mime, bytes=size, reason="downloaded HTML, not media")
    if mime == "empty":
        return FileProbe(ok=False, kind="empty", mime=mime, bytes=size, reason="empty file")

    kind = "image" if mime.startswith("image/") else "video" if mime.startswith("video/") else "file"
    probe = FileProbe(ok=True, kind=kind, mime=mime, bytes=size)

    if mime == "image/svg+xml":
        return probe

    cmd = [
        settings.ffprobe_path,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffprobe failed for %s: %s", path, exc)
        return probe

    if result.returncode != 0:
        if kind == "image":
            return probe
        probe.ok = False
        probe.reason = "ffprobe could not read media"
        return probe

    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return probe

    streams = data.get("streams") or []
    visual = next((s for s in streams if s.get("codec_type") in {"video", "image"}), None)
    if visual:
        probe.width = int(visual.get("width") or 0)
        probe.height = int(visual.get("height") or 0)
        probe.codec = str(visual.get("codec_name") or "")
    fmt = data.get("format") or {}
    try:
        probe.duration = float(fmt.get("duration") or visual.get("duration") or 0)
    except (TypeError, ValueError, AttributeError):
        probe.duration = 0.0

    if kind == "image" and probe.width and probe.width < 80:
        probe.ok = False
        probe.reason = f"image too small ({probe.width}x{probe.height})"
    if kind == "video" and probe.duration and probe.duration < 0.2:
        probe.ok = False
        probe.reason = "video duration too short"
    return probe

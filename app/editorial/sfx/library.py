from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import ROOT_DIR, settings
from app.editorial.models import SfxKind


@dataclass(frozen=True)
class SfxClip:
    filename: str
    trim: float
    gain: float


# Short editorial hits only. Skip meme/long clips (glass, beep, iPhone, glitch).
CLIPS: dict[SfxKind, SfxClip] = {
    "whoosh": SfxClip("whoosh-effect-3-225188.mp3", trim=0.52, gain=0.22),
    "swoosh": SfxClip("clean-fast-swooshaiff-14784.mp3", trim=0.48, gain=0.24),
    "impact": SfxClip("arrow-impact-87260.mp3", trim=0.42, gain=0.30),
    "hit": SfxClip("fast-impact-blow-2655.mp3", trim=0.40, gain=0.26),
}


def sfx_dir() -> Path:
    path = Path(settings.sfx_dir)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def resolve_clip(kind: SfxKind) -> tuple[Path, SfxClip] | None:
    clip = CLIPS.get(kind)
    if clip is None:
        return None
    path = sfx_dir() / clip.filename
    if not path.exists():
        return None
    return path, clip

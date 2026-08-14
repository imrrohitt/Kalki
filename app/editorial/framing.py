from __future__ import annotations

from app.config import settings
from app.editorial.models import SubjectBox, VisualContext


# Typical talking-head in 9:16: head+shoulders, not a full-body wide shot.
DEFAULT_BOX = SubjectBox(x=0.18, y=0.16, w=0.64, h=0.50)


def max_safe_scale(box: SubjectBox, *, margin: float = 0.10) -> float:
    """Largest zoom that keeps the subject bbox inside the frame with padding."""
    usable = 1.0 - 2.0 * margin
    by_w = usable / max(box.w, 0.25)
    by_h = usable / max(box.h, 0.25)
    return round(min(by_w, by_h, settings.zoom_max_scale), 3)


def default_visual() -> VisualContext:
    box = DEFAULT_BOX
    safe = max_safe_scale(box)
    tightness = max(box.w, box.h)
    return VisualContext(
        face_scale=round(tightness, 3),
        face_cx=box.cx,
        face_cy=box.cy,
        current_zoom=1.0,
        bbox=box,
        max_safe_scale=safe,
    )


def scale_from_intensity(intensity: float, max_safe: float) -> float:
    """intensity 0 → 1.0; 1 → max_safe. Floor a small push if intensity > 0."""
    intensity = max(0.0, min(1.0, intensity))
    if intensity <= 0.02:
        return 1.0
    headroom = max(0.0, max_safe - 1.0)
    return round(1.0 + headroom * intensity, 3)

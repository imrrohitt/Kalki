from __future__ import annotations

from app.captions.validation import CaptionValidationError, validate_caption_timeline
from app.captions.models import CaptionTimeline
from app.timeline.models import EditTimeline


class EditPlanValidationError(ValueError):
    pass


def validate_edit_timeline(
    timeline: EditTimeline,
    video_duration: float,
) -> EditTimeline:
    caption_timeline = CaptionTimeline(captions=list(timeline.captions))
    try:
        validate_caption_timeline(caption_timeline, video_duration)
    except CaptionValidationError as exc:
        raise EditPlanValidationError(str(exc)) from exc

    errors: list[str] = []
    last_end = -1.0
    ordered = sorted(timeline.zooms, key=lambda z: z.start)
    for index, zoom in enumerate(ordered):
        if zoom.end > video_duration + 0.08:
            errors.append(
                f"zooms[{index}].end ({zoom.end}) exceeds video duration ({video_duration})"
            )
        if zoom.target_scale > 1.35:
            errors.append(f"zooms[{index}].target_scale too large ({zoom.target_scale})")
        if zoom.start < last_end - 0.05:
            errors.append(
                f"zooms[{index}] overlaps previous zoom ending at {last_end:.3f}"
            )
        last_end = max(last_end, zoom.span_end)

    last_g = -1.0
    for index, graphic in enumerate(sorted(timeline.graphics, key=lambda g: g.start)):
        if graphic.end > video_duration + 0.12:
            errors.append(
                f"graphics[{index}].end ({graphic.end}) exceeds video duration ({video_duration})"
            )
        if graphic.start < last_g - 0.02:
            errors.append(
                f"graphics[{index}] overlaps previous graphic ending at {last_g:.3f}"
            )
        last_g = graphic.end

    if errors:
        raise EditPlanValidationError("; ".join(errors))
    return timeline

from __future__ import annotations

from pydantic import ValidationError

from app.captions.models import CaptionTimeline


class CaptionValidationError(ValueError):
    pass


def validate_caption_timeline(
    data: dict | CaptionTimeline,
    video_duration: float,
) -> CaptionTimeline:
    timeline = (
        data
        if isinstance(data, CaptionTimeline)
        else CaptionTimeline.model_validate(data)
    )

    errors: list[str] = []
    for index, caption in enumerate(timeline.captions):
        if caption.end > video_duration + 0.05:
            errors.append(
                f"captions[{index}].end ({caption.end}) exceeds video duration ({video_duration})"
            )
        if caption.end - caption.start < 0.12:
            errors.append(
                f"captions[{index}] duration too short ({caption.end - caption.start:.3f}s)"
            )

    if errors:
        raise CaptionValidationError("; ".join(errors))

    return timeline


def format_validation_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return exc.json()
    return str(exc)

from types import SimpleNamespace

import pytest

from app.captions.models import Caption, CaptionTimeline, CaptionWord
from app.captions.validation import CaptionValidationError, validate_caption_timeline
from app.media.probe import MediaError, validate_video


def test_valid_timeline():
    data = {
        "version": "1.0",
        "style": "dynamic_social",
        "captions": [
            {
                "start": 0.2,
                "end": 1.4,
                "text": "AI IS",
                "position": "bottom_center",
                "animation": "pop",
                "words": [
                    {"text": "AI", "start": 0.2, "end": 0.55, "emphasis": True},
                    {"text": "IS", "start": 0.55, "end": 0.8, "emphasis": False},
                ],
            }
        ],
    }
    timeline = validate_caption_timeline(data, video_duration=10.0)
    assert isinstance(timeline, CaptionTimeline)
    assert timeline.captions[0].words[0].emphasis is True


def test_reject_negative_timestamp():
    with pytest.raises(Exception):
        Caption(start=-1, end=1, text="NOPE", words=[])


def test_reject_end_before_start():
    with pytest.raises(Exception):
        CaptionWord(text="AI", start=1.0, end=0.5)


def test_reject_word_outside_caption():
    with pytest.raises(Exception):
        Caption(
            start=1.0,
            end=2.0,
            text="AI",
            words=[CaptionWord(text="AI", start=0.5, end=0.8)],
        )


def test_reject_caption_outside_duration():
    timeline = CaptionTimeline(
        captions=[
            Caption(
                start=0.0,
                end=2.0,
                text="HELLO WORLD",
                words=[
                    CaptionWord(text="HELLO", start=0.0, end=0.5),
                    CaptionWord(text="WORLD", start=0.5, end=1.0),
                ],
            )
        ]
    )
    with pytest.raises(CaptionValidationError):
        validate_caption_timeline(timeline, video_duration=1.0)


def _info(duration: float) -> SimpleNamespace:
    return SimpleNamespace(duration=duration, has_audio=True)


def test_duration_threshold_from_env_seconds():
    validate_video(_info(130.8), max_duration_sec=180)
    validate_video(_info(400.0), max_duration_sec=0)


def test_reject_video_over_duration_threshold():
    with pytest.raises(MediaError, match="Max is 60s"):
        validate_video(_info(90.0), max_duration_sec=60.0)

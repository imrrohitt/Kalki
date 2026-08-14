import pytest

from app.editorial.analyzer import merge_annotations
from app.editorial.context import build_sentence_windows
from app.editorial.heuristic import heuristic_analyze
from app.editorial.models import (
    LlmAnalysisPayload,
    LlmSentenceAnnotation,
    SentenceSignals,
    ZoomMotion,
)
from app.transcription.models import Segment, Transcript, Word


def _transcript(sentences: list[tuple[str, float]]) -> Transcript:
    """sentences: (text, start). 0.6s pause between sentences."""
    words: list[Word] = []
    t = 0.0
    for text, start in sentences:
        t = start
        tokens = text.split()
        for token in tokens:
            words.append(Word(word=token, start=t, end=t + 0.18, probability=0.9))
            t += 0.2
    return Transcript(
        language="en",
        duration=t,
        segments=[
            Segment(
                start=words[0].start,
                end=words[-1].end,
                text=" ".join(w.word for w in words),
                words=words,
            )
        ],
    )


def test_context_windows_include_neighbors():
    transcript = _transcript(
        [
            ("Most developers think performance is about CPU.", 0.0),
            ("But that's actually wrong.", 3.0),
            ("The real bottleneck is usually database queries.", 6.0),
        ]
    )
    windows = build_sentence_windows(transcript)
    assert len(windows) == 3
    mid = windows[1]
    assert mid.context.previous is not None
    assert "CPU" in mid.context.previous
    assert "wrong" in mid.context.current.lower()
    assert mid.context.next is not None
    assert "bottleneck" in mid.context.next
    assert mid.prosody.pause_before is not None
    assert mid.prosody.pause_before >= 0.45


def test_heuristic_labels_setup_reversal_reveal():
    transcript = _transcript(
        [
            ("I thought the company was doing $10M a year.", 0.0),
            ("I was completely wrong.", 4.0),
            ("They were actually doing $100M.", 8.0),
        ]
    )
    analysis = heuristic_analyze(build_sentence_windows(transcript))
    roles = [s.editorial_role for s in analysis.sentences]
    assert roles[0] == "assumption"
    assert roles[1] == "contradiction"
    assert roles[2] == "reveal"
    assert any(p.pattern.startswith("setup") for p in analysis.story_patterns)


def test_generic_and_cta_are_low_interest():
    transcript = _transcript(
        [
            ("Today we're going to talk about databases.", 0.0),
            ("If you want to learn this subscribe.", 4.0),
        ]
    )
    analysis = heuristic_analyze(build_sentence_windows(transcript))
    assert analysis.sentences[0].editorial_role == "generic"
    assert analysis.sentences[1].editorial_role == "cta"
    assert analysis.sentences[0].visual_interest < 0.2


def test_merge_keeps_transcript_timestamps():
    transcript = _transcript([("But that's actually wrong.", 1.2)])
    windows = build_sentence_windows(transcript)
    payload = LlmAnalysisPayload(
        sentences=[
            LlmSentenceAnnotation(
                sentence_id=0,
                editorial_role="contrast",
                signals=SentenceSignals(contrast=0.94, reveal=0.7),
                visual_interest=0.88,
                story_position="climax",
                confidence=0.91,
                zoom=ZoomMotion(
                    apply=True,
                    intensity=0.7,
                    delay_ms=180,
                    ease_in_ms=560,
                    hold_ms=700,
                    ease_out_ms=480,
                ),
            )
        ]
    )
    analysis = merge_annotations(windows, payload)
    assert analysis.sentences[0].start == windows[0].start
    assert analysis.sentences[0].end == windows[0].end
    assert analysis.sentences[0].editorial_role == "contrast"
    assert analysis.sentences[0].text == windows[0].text
    assert analysis.sentences[0].zoom.apply is True
    assert analysis.sentences[0].zoom.ease_in_ms == 560


def test_windows_follow_whisper_segments_without_punctuation():
    transcript = Transcript(
        language="hi",
        duration=2.0,
        segments=[
            Segment(
                start=0.0,
                end=0.5,
                text="hello world",
                words=[
                    Word(word="hello", start=0.0, end=0.2, probability=0.9),
                    Word(word="world", start=0.2, end=0.5, probability=0.9),
                ],
            ),
            Segment(
                start=0.55,
                end=1.0,
                text="this continues",
                words=[
                    Word(word="this", start=0.55, end=0.7, probability=0.9),
                    Word(word="continues", start=0.7, end=1.0, probability=0.9),
                ],
            ),
        ],
    )
    windows = build_sentence_windows(transcript)
    assert len(windows) == 2
    assert windows[0].text == "hello world"
    assert windows[1].text == "this continues"
    assert max(w.end - w.start for w in windows) < 1.0


def test_edit_plan_rejects_overlapping_zooms():
    from app.captions.models import Caption, CaptionWord
    from app.editorial.models import ZoomDecision
    from app.timeline.models import EditTimeline
    from app.timeline.validator import EditPlanValidationError, validate_edit_timeline

    timeline = EditTimeline(
        captions=[
            Caption(
                start=0.0,
                end=2.0,
                text="HELLO",
                words=[CaptionWord(text="HELLO", start=0.0, end=1.0)],
            )
        ],
        zooms=[
            ZoomDecision(
                start=0.2,
                end=1.5,
                intent="reveal",
                style="fast_punch",
                target_scale=1.16,
                sentence_id=0,
            ),
            ZoomDecision(
                start=1.0,
                end=2.0,
                intent="emphasis",
                style="slow_punch",
                target_scale=1.12,
                sentence_id=1,
            ),
        ],
    )
    with pytest.raises(EditPlanValidationError, match="overlaps"):
        validate_edit_timeline(timeline, video_duration=10.0)

from app.editorial.context import build_sentence_windows
from app.editorial.framing import default_visual, max_safe_scale, scale_from_intensity
from app.editorial.heuristic import heuristic_analyze
from app.editorial.models import SubjectBox, VisualContext, ZoomDecision
from app.editorial.zoom.agent import ZoomDecisionEngine
from app.renderer.zoom import build_zoom_filtergraph, zoom_segments
from app.transcription.models import Segment, Transcript, Word


def _transcript(sentences: list[tuple[str, float]]) -> Transcript:
    words: list[Word] = []
    t = 0.0
    for text, start in sentences:
        t = start
        for token in text.split():
            words.append(Word(word=token, start=t, end=t + 0.18, probability=0.9))
            t += 0.2
    return Transcript(
        language="en",
        duration=max(t, 12.0),
        segments=[
            Segment(
                start=words[0].start,
                end=words[-1].end,
                text=" ".join(w.word for w in words),
                words=words,
            )
        ],
    )


def test_bbox_caps_zoom_amount():
    box = SubjectBox(x=0.12, y=0.10, w=0.76, h=0.62)
    safe = max_safe_scale(box)
    assert safe <= 1.20
    assert scale_from_intensity(1.0, safe) == safe
    assert scale_from_intensity(0.0, safe) == 1.0


def test_story_pattern_escalates_zoom():
    transcript = _transcript(
        [
            ("I thought the company was doing $10M a year.", 0.0),
            ("I was completely wrong.", 4.0),
            ("They were actually doing $100M.", 8.0),
        ]
    )
    analysis = heuristic_analyze(build_sentence_windows(transcript))
    zooms = ZoomDecisionEngine().decide(analysis, video_duration=12.0)
    by_intent = {z.intent: z for z in zooms}
    assert "assumption" not in by_intent
    assert by_intent["reveal"].target_scale > by_intent["contradiction"].target_scale
    assert by_intent["reveal"].start > analysis.sentences[2].start
    assert by_intent["reveal"].ease_in >= 0.45
    assert by_intent["contradiction"].ease_in >= 0.45
    visual = default_visual()
    assert by_intent["reveal"].target_scale <= visual.max_safe_scale + 0.001


def test_cta_and_generic_do_not_zoom():
    transcript = _transcript(
        [
            ("Today we're going to talk about databases.", 0.0),
            ("If you want to learn this subscribe.", 4.0),
        ]
    )
    analysis = heuristic_analyze(build_sentence_windows(transcript))
    zooms = ZoomDecisionEngine().decide(analysis, video_duration=8.0)
    assert zooms == []


def test_tight_framing_skips_zoom():
    transcript = _transcript([("I would NEVER do this.", 0.0)])
    analysis = heuristic_analyze(build_sentence_windows(transcript))
    zooms = ZoomDecisionEngine().decide(
        analysis,
        video_duration=4.0,
        visual=VisualContext(
            face_scale=0.95,
            current_zoom=1.2,
            bbox=SubjectBox(x=0.05, y=0.08, w=0.90, h=0.78),
            max_safe_scale=1.02,
        ),
    )
    assert zooms == []


def test_question_then_stronger_answer():
    transcript = _transcript(
        [
            ("So what does that mean?", 0.0),
            ("It means you're wasting money.", 3.5),
        ]
    )
    analysis = heuristic_analyze(build_sentence_windows(transcript))
    zooms = ZoomDecisionEngine().decide(analysis, video_duration=8.0)
    assert len(zooms) == 2
    assert zooms[0].intent == "question"
    assert zooms[1].intent == "answer"
    assert zooms[1].target_scale > zooms[0].target_scale
    assert zooms[0].ease_in >= 0.45


def test_zoom_duration_is_capped(monkeypatch):
    monkeypatch.setattr("app.editorial.zoom.planner.settings.zoom_max_duration_sec", 3.2)
    transcript = _transcript(
        [("The real reason is database queries are the bottleneck here.", 0.0)]
    )
    analysis = heuristic_analyze(build_sentence_windows(transcript))
    analysis.sentences[0].end = 20.0
    analysis.sentences[0].editorial_role = "key_insight"
    analysis.sentences[0].visual_interest = 0.9
    analysis.sentences[0].zoom.apply = True
    analysis.sentences[0].zoom.intensity = 0.6
    zooms = ZoomDecisionEngine().decide(analysis, video_duration=20.0)
    assert zooms
    assert max(z.span_end - z.start for z in zooms) <= 3.21


def test_zoom_filter_eases_instead_of_snapping():
    zooms = [
        ZoomDecision(
            start=2.0,
            end=3.2,
            intent="reveal",
            style="slow_punch",
            target_scale=1.16,
            easing="ease_in_out",
            ease_in=0.5,
            hold=0.7,
            ease_out=0.45,
            sentence_id=3,
            anchor_x=0.5,
            anchor_y=0.41,
        )
    ]
    pieces = zoom_segments(zooms)
    zoomed = [p for p in pieces if p.scale > 1.001]
    scales = [p.scale for p in zoomed]
    assert len(scales) >= 6
    assert scales[0] < scales[max(range(len(scales)), key=lambda i: scales[i])]
    assert max(scales) == 1.16
    assert min(scales) < 1.08
    vf = build_zoom_filtergraph(zooms, 1080, 1920)
    assert vf is not None
    assert "trim=start=2.000" in vf
    assert "concat=" in vf
    assert "scale=trunc(iw*1.160/2)*2" in vf
    assert vf.count("scale=trunc(iw*") >= 6
    assert "ffmpeg" not in vf.lower()

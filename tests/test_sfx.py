from app.editorial.heuristic import heuristic_analyze
from app.editorial.context import build_sentence_windows
from app.editorial.models import GraphicBeat, SfxHit
from app.editorial.sfx.library import resolve_clip
from app.editorial.sfx.planner import plan_sfx
from app.renderer.sfx_mix import build_sfx_mix
from app.transcription.models import Segment, Transcript, Word


def _transcript() -> Transcript:
    words = [
        Word(word="Why", start=0.1, end=0.4, probability=0.9),
        Word(word="RAG", start=0.4, end=0.8, probability=0.9),
        Word(word="beats", start=0.8, end=1.1, probability=0.9),
        Word(word="fine-tuning", start=4.0, end=4.5, probability=0.9),
        Word(word="costs", start=4.5, end=4.9, probability=0.9),
        Word(word="Subscribe", start=12.0, end=12.4, probability=0.9),
        Word(word="now", start=12.4, end=12.7, probability=0.9),
    ]
    return Transcript(
        language="en",
        duration=14.0,
        segments=[
            Segment(start=0.1, end=12.7, text=" ".join(w.word for w in words), words=words)
        ],
    )


def test_library_resolves_whoosh_and_impact():
    whoosh = resolve_clip("whoosh")
    impact = resolve_clip("impact")
    assert whoosh is not None and whoosh[0].exists()
    assert impact is not None and impact[0].exists()


def test_plan_sfx_hits_hook_and_vs():
    analysis = heuristic_analyze(build_sentence_windows(_transcript()))
    graphics = [
        GraphicBeat(start=0.1, end=4.0, kind="bullets", title="Why RAG", kicker="HOOK"),
        GraphicBeat(
            start=4.0,
            end=10.0,
            kind="vs_split",
            title="RAG vs Fine-tuning",
            left="RAG",
            right="Fine-tuning",
        ),
        GraphicBeat(start=10.0, end=13.5, kind="stat", title="10x cheaper"),
    ]
    hits = plan_sfx(analysis, graphics, video_duration=14.0)
    assert hits
    assert hits[0].kind == "impact"
    kinds = {h.kind for h in hits}
    assert "swoosh" in kinds or "hit" in kinds
    times = [h.at for h in hits]
    assert times == sorted(times)
    for prev, cur in zip(times, times[1:]):
        assert cur - prev >= 1.6


def test_sfx_mix_builds_adelay_graph():
    hits = [
        SfxHit(at=0.2, kind="impact", reason="hook"),
        SfxHit(at=4.0, kind="swoosh", reason="compare"),
    ]
    files, graph = build_sfx_mix(hits, voice_has_audio=True, video_duration=8.0)
    assert len(files) == 2
    assert "adelay=200|200" in graph
    assert "adelay=4000|4000" in graph
    assert "[aout]" in graph
    assert "amix=inputs=3" in graph

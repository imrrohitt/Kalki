from app.editorial.context import build_sentence_windows
from app.editorial.heuristic import heuristic_analyze
from app.editorial.scenes.planner import plan_scenes
from app.editorial.transcript.repair import align_words, heuristic_repair
from app.media.probe import probe_audio
from app.transcription.models import Segment, Transcript, Word


def _transcript() -> Transcript:
    return Transcript(
        language="en",
        duration=12.0,
        segments=[
            Segment(
                start=0.0,
                end=4.0,
                text="Why are we using RAKA instead of fine tunning",
                words=[
                    Word(word="Why", start=0.0, end=0.3, probability=0.9),
                    Word(word="are", start=0.3, end=0.5, probability=0.9),
                    Word(word="we", start=0.5, end=0.7, probability=0.9),
                    Word(word="using", start=0.7, end=1.0, probability=0.9),
                    Word(word="RAKA", start=1.0, end=1.5, probability=0.9),
                    Word(word="instead", start=1.5, end=2.0, probability=0.9),
                    Word(word="of", start=2.0, end=2.2, probability=0.9),
                    Word(word="fine", start=2.2, end=2.6, probability=0.9),
                    Word(word="tunning", start=2.6, end=3.2, probability=0.9),
                ],
            ),
            Segment(
                start=4.2,
                end=8.0,
                text="First retrieve then generate the answer",
                words=[
                    Word(word="First", start=4.2, end=4.5, probability=0.9),
                    Word(word="retrieve", start=4.5, end=5.1, probability=0.9),
                    Word(word="then", start=5.1, end=5.4, probability=0.9),
                    Word(word="generate", start=5.4, end=6.0, probability=0.9),
                    Word(word="the", start=6.0, end=6.2, probability=0.9),
                    Word(word="answer", start=6.2, end=6.8, probability=0.9),
                ],
            ),
        ],
    )


def test_heuristic_repair_fixes_raka_and_fine_tuning():
    repaired = heuristic_repair(_transcript())
    blob = " ".join(seg.text for seg in repaired.segments)
    assert "RAG" in blob
    assert "Fine-tuning" in blob or "fine-tuning" in blob.lower()
    assert repaired.segments[0].words
    assert repaired.segments[0].words[0].start == 0.0


def test_align_words_keeps_timing_when_counts_match():
    original = [
        Word(word="a", start=0.0, end=0.2, probability=0.8),
        Word(word="b", start=0.2, end=0.4, probability=0.8),
    ]
    aligned = align_words(original, "Hello World")
    assert [w.word for w in aligned] == ["Hello", "World"]
    assert aligned[0].start == 0.0
    assert aligned[1].end == 0.4


def test_plan_scenes_opens_with_a_hook(tmp_path):
    analysis = heuristic_analyze(build_sentence_windows(_transcript()))
    beats = plan_scenes(analysis, video_duration=12.0)
    assert beats
    assert beats[0].kicker == "HOOK"
    assert beats[0].start == 0.0
    kinds = {b.kind for b in beats}
    assert kinds & {"vs_split", "process", "diagram", "stat", "topic", "bullets", "quote"}
    assert beats[0].nodes
    assert len(beats[0].bullets) >= 3


def test_ass_full_canvas_places_graphics_and_bottom_captions(tmp_path):
    from app.captions.models import Caption, CaptionTimeline, CaptionWord
    from app.editorial.models import GraphicBeat
    from app.renderer.ass import write_ass_file
    from app.renderer.design import CAPTION_Y_FULL

    path = tmp_path / "full.ass"
    write_ass_file(
        CaptionTimeline(
            captions=[
                Caption(
                    start=0.2,
                    end=1.8,
                    text="Why RAG",
                    words=[
                        CaptionWord(text="Why", start=0.2, end=0.6),
                        CaptionWord(text="RAG", start=0.6, end=1.8, emphasis=True),
                    ],
                )
            ]
        ),
        str(path),
        graphics=[
            GraphicBeat(
                start=0.0,
                end=3.0,
                kind="diagram",
                title="How RAG answers",
                kicker="HOOK",
                chips=["Query", "Retrieve", "Generate"],
            )
        ],
        layout="full",
        video_duration=3.0,
        theme="paper",
    )
    text = path.read_text()
    assert "Audio Motion Reel" in text
    assert "How RAG answers" in text
    assert "Query" in text
    assert "Retrieve" in text
    assert rf"\pos(540,{CAPTION_Y_FULL})" in text
    assert r"\pos(540,720)" not in text
    assert "GTitle" in text
    assert "GNode" in text
    assert r"\move(" in text
    # Boxed nodes + dotted spine, not a title floating in empty space.
    assert text.count("GHair") >= 4 or text.count("\\p1") >= 8


def test_probe_audio_reads_wav(tmp_path):
    import subprocess

    from app.config import settings

    wav = tmp_path / "tone.wav"
    subprocess.run(
        [
            settings.ffmpeg_path,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1.5",
            str(wav),
        ],
        check=True,
        capture_output=True,
    )
    info = probe_audio(str(wav))
    assert info.duration >= 1.4
    assert info.channels >= 1


def test_full_canvas_payload_keeps_nodes_and_four_bullets():
    from app.editorial.graphics.agent import _beats_from_payload
    from app.editorial.models import GraphicNode

    beats = _beats_from_payload(
        {
            "graphics": [
                {
                    "start": 0,
                    "end": 8,
                    "kind": "diagram",
                    "title": "How RAG answers a question",
                    "kicker": "FLOW",
                    "nodes": [
                        {"label": "Question", "sub": "user asks"},
                        {"label": "Retrieve", "sub": "pull chunks"},
                        {"label": "Generate", "sub": "write answer"},
                    ],
                    "bullets": [
                        "Re-index when files change",
                        "No GPU retrain every time",
                        "Fine-tune if data is frozen",
                        "Same LLM, two update paths",
                    ],
                }
            ]
        },
        10.0,
        canvas="full",
    )
    assert len(beats) == 1
    assert [n.label for n in beats[0].nodes] == ["Question", "Retrieve", "Generate"]
    assert isinstance(beats[0].nodes[0], GraphicNode)
    assert len(beats[0].bullets) == 4

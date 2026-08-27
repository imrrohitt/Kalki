from app.asr import fix_asr_text
from app.captions.heuristic import heuristic_caption_timeline
from app.editorial.context import build_sentence_windows
from app.editorial.graphics.agent import _beats_from_payload
from app.editorial.graphics.glyphs import resolve_glyph
from app.editorial.graphics.planner import cover_duration, plan_graphics
from app.editorial.graphics.terms import extract_terms
from app.editorial.heuristic import heuristic_analyze
from app.editorial.models import GraphicBeat
from app.renderer.ass import write_ass_file
from app.transcription.models import Segment, Transcript, Word


def _transcript(sentences: list[tuple[str, float]]) -> Transcript:
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


def test_asr_fix_maps_raka_to_rag():
    assert "RAG" in fix_asr_text("Why are we using RAKA system")
    from app.asr import stabilize_copy

    assert "Fine-tuning" in stabilize_copy("fine tunning vs RAKA")
    assert "RAG" in stabilize_copy("THE RAKA\nSYSTEM")
    assert "LoRA" in stabilize_copy("use lora adapters")
    assert "quantization" in stabilize_copy("quantisation of weights")
    terms = extract_terms("choose RAKA over the fine tuning")
    names = [t[0] for t in terms]
    assert "RAG" in names
    assert "Fine-tuning" in names


def test_graphics_plan_emits_vs_and_does_not_overlap():
    transcript = _transcript(
        [
            ("If you are giving AI interviews about LLM", 0.0),
            ("Why do we prefer the RAKA system over the fine tuning", 4.0),
            ("Fine tuning means you train an existing LLM on domain data", 8.0),
            ("There is LoRA and parameter efficient fine tuning", 12.0),
            ("RAG is cheaper when documents change and you re-index", 16.0),
        ]
    )
    analysis = heuristic_analyze(build_sentence_windows(transcript))
    beats = plan_graphics(analysis, video_duration=22.0)
    assert beats
    kinds = {b.kind for b in beats}
    assert "vs_split" in kinds
    assert any(b.bullets for b in beats)
    ends = -1.0
    for beat in beats:
        assert beat.end > beat.start
        assert beat.start >= ends - 0.02
        ends = beat.end
        assert beat.end <= 22.12


def test_heuristic_captions_group_short_phrases():
    transcript = _transcript(
        [
            ("If you are giving AI interviews", 0.0),
            ("fine tuning is costly", 3.0),
        ]
    )
    timeline = heuristic_caption_timeline(transcript, video_duration=8.0)
    assert timeline.captions
    assert all(len(c.text.split()) <= 4 for c in timeline.captions)
    assert any("RAG" in c.text or "fine" in c.text.lower() or "AI" in c.text for c in timeline.captions)


def test_ass_split_includes_motion_and_seam_captions(tmp_path):
    from app.captions.models import Caption, CaptionTimeline, CaptionWord

    timeline = CaptionTimeline(
        captions=[
            Caption(
                start=0.2,
                end=1.8,
                text="Why RAG",
                position="center",
                words=[
                    CaptionWord(text="Why", start=0.2, end=0.6),
                    CaptionWord(text="RAG", start=0.6, end=1.8, emphasis=True),
                ],
            )
        ]
    )
    from app.editorial.models import GraphicBullet

    beats = [
        GraphicBeat(
            start=0.2,
            end=2.4,
            kind="vs_split",
            title="RAG vs Fine-tuning",
            left="RAG",
            right="Fine-tuning",
            kicker="THE TRADEOFF",
            motion="slide_up",
            bullets=[
                GraphicBullet(icon="📄", text="First pointer", delay_ms=0),
                GraphicBullet(icon="🧠", text="Second pointer", delay_ms=480),
                GraphicBullet(icon="💡", text="Third pointer", delay_ms=960),
            ],
        )
    ]
    path = tmp_path / "out.ass"
    write_ass_file(
        timeline,
        str(path),
        graphics=beats,
        split_layout=True,
        video_duration=3.0,
    )
    text = path.read_text()
    assert r"\pos(540,720)" in text
    assert r"\an8\pos(540,730)" in text
    assert r"\fad(" in text
    assert "GTitle" in text
    assert "GVs" in text
    assert "THE TRADEOFF" in text
    assert "Why" in text
    assert "RAG" in text
    assert r"\an8\pos(540,730)" in text or r"\pos(540,720)" in text
    assert r"\fsp" in text
    assert r"\move(" in text
    assert r"\t(" in text
    assert "Fine-tuning" in text
    assert r"\p1" in text
    assert r"\clip(" in text
    assert "GShape" not in text or "GLine" in text
    assert r"\fs112" in text
    assert r"\fscx124" in text or r"\fscy124" in text
    assert r"\alpha&HFF&" in text


def test_overlay_emphasis_is_bigger_and_pops(tmp_path):
    from app.captions.models import Caption, CaptionTimeline, CaptionWord
    from app.renderer.ass import write_ass_file

    path = tmp_path / "kinetic.ass"
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
        split_layout=False,
        video_duration=2.0,
        theme="tech",
    )
    text = path.read_text()
    assert r"\fs72" in text
    assert r"\fs94" in text
    assert r"\an8\pos(540," in text
    assert r"\clip(" in text
    assert r"\t(" in text
    assert "WHY" in text
    assert "RAG" in text
    # Emphasis uses the theme accent, not the body white.
    assert r"\c&H00FFC94F&" in text


def test_overlay_captions_sit_above_head_and_wrap(tmp_path):
    from app.captions.models import Caption, CaptionTimeline, CaptionWord
    from app.renderer.ass import write_ass_file

    path = tmp_path / "above.ass"
    write_ass_file(
        CaptionTimeline(
            captions=[
                Caption(
                    start=0.2,
                    end=2.0,
                    text="And Follow Compliance Like GDPR",
                    words=[
                        CaptionWord(text="And", start=0.2, end=0.4),
                        CaptionWord(text="Follow", start=0.4, end=0.7),
                        CaptionWord(text="Compliance", start=0.7, end=1.2),
                        CaptionWord(text="Like", start=1.2, end=1.5),
                        CaptionWord(text="GDPR", start=1.5, end=2.0, emphasis=True),
                    ],
                )
            ]
        ),
        str(path),
        split_layout=False,
        video_duration=2.0,
        head_top=480,
    )
    text = path.read_text()
    dialogue = [ln for ln in text.splitlines() if ln.startswith("Dialogue: 5,")][0]
    assert r"\an8\pos(540,96)" in dialogue
    assert r"\N" in dialogue
    assert "GDPR" in dialogue


def test_short_headline_stays_on_one_line(tmp_path):
    from app.captions.models import CaptionTimeline
    from app.renderer.ass import write_ass_file

    path = tmp_path / "title.ass"
    write_ass_file(
        CaptionTimeline(captions=[]),
        str(path),
        graphics=[
            GraphicBeat(
                start=0.0,
                end=2.0,
                kind="process",
                title="Quantization Defined",
                kicker="STEPS",
            )
        ],
        split_layout=True,
        video_duration=2.0,
    )
    text = path.read_text()
    assert "Quantization Defined" in text
    assert "Dialogue: 3," in text
    title_events = [ln for ln in text.splitlines() if "GTitle" in ln]
    assert any("Quantization Defined" in ln for ln in title_events)
    assert not any(ln.endswith("Defined") and "Quantization" not in ln for ln in title_events)


def test_themes_change_palette_and_background(tmp_path):
    from app.captions.models import CaptionTimeline
    from app.renderer.design import THEMES, get_theme
    from app.renderer.split import build_split_filtergraph

    beat = GraphicBeat(start=0.0, end=3.0, kind="bullets", title="90% fail", kicker="HOOK")
    outputs = {}
    for name in THEMES:
        path = tmp_path / f"{name}.ass"
        write_ass_file(
            CaptionTimeline(captions=[]),
            str(path),
            graphics=[beat],
            split_layout=True,
            video_duration=3.0,
            theme=name,
        )
        outputs[name] = path.read_text()
    assert THEMES["noir"].ink in outputs["noir"]
    assert THEMES["noir"].accent in outputs["noir"]
    assert THEMES["paper"].accent in outputs["paper"]
    assert outputs["noir"] != outputs["paper"]
    # Unknown themes fall back to the default palette.
    assert get_theme("bogus").name == "paper"

    graph = build_split_filtergraph(
        width=1080, height=1920, fps=30,
        ass_escaped="x.ass", fonts_dir="/fonts", theme="tech",
    )
    assert THEMES["tech"].bg_top in graph
    assert "drawgrid" in graph
    paper_graph = build_split_filtergraph(
        width=1080, height=1920, fps=30,
        ass_escaped="x.ass", fonts_dir="/fonts",
    )
    assert "drawgrid" not in paper_graph
    assert THEMES["paper"].bg_top in paper_graph


def test_cover_duration_closes_gaps():
    beats = [
        GraphicBeat(start=1.0, end=4.0, kind="bullets", title="One"),
        GraphicBeat(start=8.0, end=11.0, kind="bullets", title="Two"),
    ]
    covered = cover_duration(beats, 15.0)
    assert covered[0].start == 0.0
    assert covered[0].end == 8.0
    assert covered[1].start == 8.0
    assert covered[1].end == 15.0


def test_payload_clips_copy_and_maps_glyph():
    beats = _beats_from_payload(
        {
            "graphics": [
                {
                    "start": 0,
                    "end": 6,
                    "kind": "bullets",
                    "title": "This title is way too long for the 720 pixel canvas and must clip",
                    "icon": "🧠",
                    "bullets": [
                        {"text": "first point that is also much too long for one line of type", "delay_ms": 0},
                        {"text": "second", "delay_ms": 520},
                        {"text": "third should drop", "delay_ms": 960},
                    ],
                }
            ]
        },
        10.0,
    )
    assert len(beats) == 1
    assert beats[0].glyph == "brain"
    assert len(beats[0].title) <= 40
    assert len(beats[0].bullets) == 2
    assert len(beats[0].bullets[0].text) <= 32
    assert resolve_glyph("⚖️", "vs_split") == "scale"


def test_payload_stabilizes_misspeaks():
    beats = _beats_from_payload(
        {
            "graphics": [
                {
                    "start": 0,
                    "end": 6,
                    "kind": "vs_split",
                    "title": "RAKA vs fine tunning",
                    "left": "raka",
                    "right": "fine tuning",
                    "bullets": [{"text": "use lora and destillation", "delay_ms": 0}],
                }
            ]
        },
        10.0,
    )
    assert beats[0].title == "RAG vs Fine-tuning"
    assert beats[0].left == "RAG"
    assert "Fine-tun" in beats[0].right
    assert beats[0].bullets == []

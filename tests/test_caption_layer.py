from pathlib import Path

import numpy as np

from app.captions.models import Caption, CaptionTimeline, CaptionWord
from app.editorial.models import SfxHit
from app.renderer.caption_layer import BRIGHT_INK, WHITE, CaptionLayer
from app.renderer.soundtrack import build_soundtrack, plan_accents, synthesize_music_bed


def _cap(start, end, text, treatment, emphasis=(), mood="neutral", icon="none"):
    tokens = text.replace("\n", " ").split()
    step = (end - start) / (len(tokens) + 1)
    return Caption(
        start=start,
        end=end,
        text=text,
        treatment=treatment,
        mood=mood,
        icon=icon,
        words=[
            CaptionWord(text=t, start=start + i * step, end=start + (i + 1) * step, emphasis=t in emphasis)
            for i, t in enumerate(tokens)
        ],
    )


def _timeline():
    return CaptionTimeline(
        captions=[
            _cap(0.0, 1.5, "I recently switched\nto senior AI", "stack"),
            _cap(1.5, 3.0, "giving interviews", "mix", ("interviews",)),
            _cap(3.0, 4.5, "core depth", "oval"),
            _cap(4.5, 6.0, "rejections", "tape"),
            _cap(6.0, 7.5, "the trade-offs", "underline", ("trade-offs",)),
        ]
    )


def test_caption_layer_draws_every_treatment_above_the_head():
    layer = CaptionLayer(_timeline(), width=1080, height=1920, fps=30, head_top=640)
    assert layer.band_top + layer.band_height <= 640 + 200
    for t in (1.2, 2.8, 4.2, 5.8, 7.3):
        band = layer.frame_rgba(t)
        assert band.shape == (layer.band_height, 1080, 4)
        alpha = band[..., 3]
        assert alpha.max() == 255, t
        ys, xs = np.nonzero(alpha > 32)
        # Centered composition with real margins.
        assert abs((xs.min() + xs.max()) / 2 - 540) < 60, t
        assert xs.min() > 40 and xs.max() < 1040, t


def test_caption_layer_frames_are_empty_between_captions_and_reuse_static_frames():
    timeline = CaptionTimeline(captions=[_cap(1.0, 2.0, "hello there", "plain")])
    layer = CaptionLayer(timeline, width=540, height=960, fps=10, head_top=320)
    frames = list(layer.iter_frames(30))
    assert len(frames) == 30
    assert not any(frames[0])
    assert any(frames[15])
    assert frames[18] is frames[19]
    assert not any(frames[25])


def test_accents_are_sparse_and_never_per_word():
    timeline = CaptionTimeline(
        captions=[_cap(i * 1.0, i * 1.0 + 0.9, "word here", "plain") for i in range(60)]
        + [_cap(61.0, 62.0, "core depth", "oval"), _cap(63.0, 64.0, "the trade-offs", "underline", ("trade-offs",))]
    )
    hits = plan_accents(timeline, video_duration=70.0)
    assert hits[0].kind == "riser" and hits[0].at == 0.0
    assert len(hits) == 2  # hook + one sparkle; the underline is too close to the oval


def test_pop_mood_uses_frames_reveal_not_rise():
    """A surprise/excited caption 'blinks' in (scale-bounce) instead of rising."""
    plain = CaptionTimeline(captions=[_cap(0.0, 1.5, "an ordinary line", "plain")])
    excited = CaptionTimeline(
        captions=[_cap(0.0, 1.5, "you won't believe", "serif", mood="excited")]
    )
    modes_plain = {e.mode for e, _, _ in CaptionLayer(plain, width=1080, height=1920, head_top=640)._states(0.3, 0)}
    modes_pop = {e.mode for e, _, _ in CaptionLayer(excited, width=1080, height=1920, head_top=640)._states(0.3, 0)}
    assert modes_plain == {"rise"}
    assert "frames" in modes_pop and "rise" not in modes_pop


def test_pop_only_touches_emphasis_word_in_an_ordinary_line():
    timeline = CaptionTimeline(
        captions=[_cap(0.0, 2.0, "we got surprising results", "mix", ("surprising",), mood="surprise")]
    )
    layer = CaptionLayer(timeline, width=1080, height=1920, head_top=640)
    els = layer._layout(timeline.captions[0], 0)
    # Exactly one word (the emphasized one) pops; the rest still rise.
    assert sum(1 for e in els if e.mode == "frames") == 1
    assert sum(1 for e in els if e.mode == "rise") == 3


def test_icon_badge_renders_a_distinct_glyph_per_icon():
    for icon in ("money", "growth", "idea", "video", "social", "check"):
        timeline = CaptionTimeline(captions=[_cap(0.0, 1.5, "context line here", "plain", icon=icon)])
        layer = CaptionLayer(timeline, width=1080, height=1920, head_top=640)
        band = layer.frame_rgba(0.9)
        assert band[..., 3].max() > 0, icon
        # The badge sits left of the text block, inside the caption band.
        ys, xs = np.nonzero(band[..., 3] > 20)
        assert xs.min() < layer.width * 0.35, icon


def test_icon_never_appears_on_a_money_chip_line():
    """The chip pill is already the 'notice this' signal — a redundant money
    badge on top of it would double up on the same figure."""
    from app.captions.craft import CaptionDraft, LineNote, assign_styles
    from app.transcription.models import Word

    text = "well so last year I earned $500 today"
    words = [Word(word=w, start=i * 1.0, end=i * 1.0 + 0.9) for i, w in enumerate(text.split())]
    drafts = [
        CaptionDraft(first=0, last=3, text="well so last year"),
        CaptionDraft(first=4, last=7, text="I earned $500 today"),
    ]
    notes = [LineNote(), LineNote(key="$500", weight=3, kind="term", icon="money")]
    out = assign_styles(drafts, notes, words)
    assert out[1].style == "chip"
    assert out[1].icon == "none"


def test_caption_color_adapts_to_its_own_background_not_the_whole_video():
    """A talking head that walks from shade into open sky needs both looks in
    the SAME render — one flat guess for the whole video would leave half the
    captions invisible."""
    timeline = CaptionTimeline(
        captions=[
            _cap(0.0, 1.5, "in the shade", "plain"),
            _cap(5.0, 6.5, "core depth", "oval"),
            _cap(10.0, 11.5, "in bright sky", "plain"),
            _cap(15.0, 16.5, "core depth", "oval"),
        ]
    )
    # 0-8s dark background, 8s+ bright background.
    times = [float(t) for t in range(0, 20)]
    values = [40.0 if t < 8 else 220.0 for t in times]
    layer = CaptionLayer(timeline, width=1080, height=1920, head_top=640, bg_luma_times=times, bg_luma_values=values)

    dark_body = _dominant_color(layer, timeline.captions[0])
    bright_body = _dominant_color(layer, timeline.captions[2])
    assert dark_body == WHITE
    assert bright_body == BRIGHT_INK

    dark_hair = _dominant_color(layer, timeline.captions[1], hairline=True)
    bright_hair = _dominant_color(layer, timeline.captions[3], hairline=True)
    assert dark_hair != bright_hair  # the oval ring also flips, not just body text


def _dominant_color(layer: CaptionLayer, cap: Caption, *, hairline: bool = False):
    band = layer.frame_rgba(min(cap.start + 0.6, cap.end - 0.15))
    alpha = band[..., 3]
    # Fully opaque glyph pixels only — anti-aliased edges blend toward the
    # transparent shadow halo and would pull the average off the true color.
    mask = alpha > 250 if not hairline else (alpha > 10) & (alpha < 200)
    ys, xs = np.nonzero(mask)
    assert len(ys) > 0
    pixels = band[ys, xs, :3].astype(np.float64)
    return tuple(int(round(v)) for v in pixels.mean(axis=0))


def test_soundtrack_graph_ducks_music_under_voice(tmp_path: Path):
    bed = synthesize_music_bed("focused_tech", 4.0, tmp_path / "bed.wav")
    assert bed.stat().st_size > 100_000
    files, graph = build_soundtrack(
        voice_input=0,
        first_extra_input=2,
        hits=[SfxHit(at=0.0, kind="riser", gain=0.14), SfxHit(at=1.0, kind="shimmer", gain=0.1)],
        music_path=bed,
        video_duration=4.0,
        cache_dir=tmp_path,
    )
    assert files[0] == bed
    assert "sidechaincompress" in graph and "loudnorm" in graph
    assert graph.endswith("[aout]")

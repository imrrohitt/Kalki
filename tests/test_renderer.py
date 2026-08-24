import subprocess
from pathlib import Path

import pytest

from app.captions.models import Caption, CaptionTimeline, CaptionWord
from app.config import settings
from app.editorial.models import ZoomDecision
from app.renderer.ffmpeg_renderer import FFmpegRenderer


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    out = tmp_path / "src.mp4"
    cmd = [
        settings.ffmpeg_path,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=720x1280:d=3",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=3",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def test_renderer_writes_mp4(tiny_video: Path, tmp_path: Path):
    timeline = CaptionTimeline(
        captions=[
            Caption(
                start=0.2,
                end=1.5,
                text="HELLO WORLD",
                position="bottom_center",
                animation="pop",
                words=[
                    CaptionWord(text="HELLO", start=0.2, end=0.7, emphasis=True),
                    CaptionWord(text="WORLD", start=0.7, end=1.5, emphasis=False),
                ],
            ),
            Caption(
                start=1.6,
                end=2.8,
                text="MORE CAPTIONS HERE",
                position="bottom_center",
                animation="pop",
                words=[
                    CaptionWord(text="MORE", start=1.6, end=2.0, emphasis=False),
                    CaptionWord(text="CAPTIONS", start=2.0, end=2.4, emphasis=True),
                    CaptionWord(text="HERE", start=2.4, end=2.8, emphasis=False),
                ],
            ),
        ]
    )
    out = tmp_path / "out.mp4"
    renderer = FFmpegRenderer(font_path=str(settings.font_path), split_layout=False)
    renderer.render(str(tiny_video), timeline, str(out))
    assert out.exists()
    assert out.stat().st_size > 1000
    assert out.with_suffix(".ass").exists()


def test_renderer_applies_zoom_filter(tiny_video: Path, tmp_path: Path):
    timeline = CaptionTimeline(
        captions=[
            Caption(
                start=0.2,
                end=2.4,
                text="THE REVEAL",
                words=[
                    CaptionWord(text="THE", start=0.2, end=0.6),
                    CaptionWord(text="REVEAL", start=0.6, end=2.4, emphasis=True),
                ],
            )
        ]
    )
    out = tmp_path / "zoomed.mp4"
    renderer = FFmpegRenderer(font_path=str(settings.font_path), split_layout=False)
    renderer.render(
        str(tiny_video),
        timeline,
        str(out),
        zooms=[
            ZoomDecision(
                start=0.4,
                end=2.0,
                intent="reveal",
                style="fast_punch",
                target_scale=1.16,
                easing="ease_out",
                sentence_id=0,
            )
        ],
    )
    assert out.exists()
    assert out.stat().st_size > 1000


def _pixel_rgb(video: Path, at_sec: float, x: int = 0, y: int = 0, width: int = 1080) -> tuple[int, int, int]:
    raw = subprocess.run(
        [
            settings.ffmpeg_path,
            "-ss",
            f"{at_sec:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-an",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    idx = (y * width + x) * 3
    return raw.stdout[idx], raw.stdout[idx + 1], raw.stdout[idx + 2]


def _top_left_rgb(video: Path, at_sec: float) -> tuple[int, int, int]:
    return _pixel_rgb(video, at_sec, 0, 0)


def test_zoom_is_visible_on_pixels(tmp_path: Path):
    src = tmp_path / "marker.mp4"
    subprocess.run(
        [
            settings.ffmpeg_path,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=1080x1920:d=2,drawbox=x=0:y=0:w=120:h=120:color=red:t=fill",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(src),
        ],
        check=True,
        capture_output=True,
    )
    out = tmp_path / "zoomed.mp4"
    FFmpegRenderer(font_path=str(settings.font_path), split_layout=False).render(
        str(src),
        CaptionTimeline(captions=[]),
        str(out),
        zooms=[
            ZoomDecision(
                start=0.2,
                end=1.6,
                intent="reveal",
                style="hold",
                target_scale=1.35,
                easing="linear",
                ease_in=0.4,
                hold=0.9,
                ease_out=0.3,
                release_duration=0.3,
                sentence_id=0,
            )
        ],
    )
    r, g, b = _top_left_rgb(out, 1.0)
    # Center-crop zoom must throw away the red corner → white.
    assert g > 80 and b > 80, f"top-left still red rgb=({r},{g},{b}); zoom filter is a no-op"


def test_split_layout_light_top_canvas(tmp_path: Path):
    src = tmp_path / "talk.mp4"
    subprocess.run(
        [
            settings.ffmpeg_path,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=720x1280:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(src),
        ],
        check=True,
        capture_output=True,
    )
    from app.editorial.models import GraphicBeat

    out = tmp_path / "split.mp4"
    FFmpegRenderer(font_path=str(settings.font_path), split_layout=True).render(
        str(src),
        CaptionTimeline(
            captions=[
                Caption(
                    start=0.2,
                    end=1.6,
                    text="Why RAG",
                    words=[
                        CaptionWord(text="Why", start=0.2, end=0.7),
                        CaptionWord(text="RAG", start=0.7, end=1.6, emphasis=True),
                    ],
                )
            ]
        ),
        str(out),
        graphics=[
            GraphicBeat(
                start=0.2,
                end=1.8,
                kind="term_card",
                title="RAG",
                subtitle="retrieve, then generate",
                kicker="CONCEPT",
            )
        ],
        video_duration=2.0,
    )
    r, g, b = _pixel_rgb(out, 1.0, x=80, y=20)
    assert r > 180 and g > 170 and b > 160, f"top canvas should stay light rgb=({r},{g},{b})"
    assert out.with_suffix(".ass").read_text().count("\\fad(") >= 1


def test_split_filtergraph_fits_head_into_bottom_panel():
    from app.renderer.split import build_split_filtergraph, head_panel_filter

    graph = build_split_filtergraph(
        width=1080,
        height=1920,
        fps=30,
        ass_escaped="x.ass",
        fonts_dir="/fonts",
        zoom_graph="SHOULD_BE_IGNORED",
    )
    assert "scale=1080:1200:force_original_aspect_ratio=decrease" in graph
    assert "overlay=(W-w)/2:(H-h)/2" in graph
    assert "boxblur=" in graph
    assert "crop=1080:960:0:0" not in graph
    assert "scale=1080:1920" not in graph
    assert "SHOULD_BE_IGNORED" not in graph
    assert "vstack=inputs=2" in graph
    assert head_panel_filter(1080, 1200) in graph


def test_renderer_mixes_sfx(tiny_video: Path, tmp_path: Path):
    from app.editorial.models import SfxHit

    out = tmp_path / "sfx.mp4"
    FFmpegRenderer(font_path=str(settings.font_path), split_layout=False).render(
        str(tiny_video),
        CaptionTimeline(
            captions=[
                Caption(
                    start=0.2,
                    end=1.4,
                    text="HOOK",
                    words=[CaptionWord(text="HOOK", start=0.2, end=1.4, emphasis=True)],
                )
            ]
        ),
        str(out),
        sfx=[SfxHit(at=0.15, kind="impact", reason="hook")],
        video_duration=3.0,
    )
    assert out.exists()
    assert out.stat().st_size > 1000


def test_audio_reel_renderer_writes_mp4(tmp_path: Path):
    from app.editorial.models import GraphicBeat, SfxHit

    wav = tmp_path / "voice.wav"
    subprocess.run(
        [
            settings.ffmpeg_path,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            str(wav),
        ],
        check=True,
        capture_output=True,
    )
    out = tmp_path / "reel.mp4"
    FFmpegRenderer(font_path=str(settings.font_path)).render_audio_reel(
        source_audio=str(wav),
        caption_timeline=CaptionTimeline(
            captions=[
                Caption(
                    start=0.2,
                    end=1.6,
                    text="Why RAG",
                    words=[
                        CaptionWord(text="Why", start=0.2, end=0.7),
                        CaptionWord(text="RAG", start=0.7, end=1.6, emphasis=True),
                    ],
                )
            ]
        ),
        output_path=str(out),
        graphics=[
            GraphicBeat(
                start=0.0,
                end=2.0,
                kind="diagram",
                title="How RAG Works",
                kicker="HOOK",
                chips=["Ask", "Retrieve", "Answer"],
            )
        ],
        sfx=[SfxHit(at=0.12, kind="impact", reason="hook")],
        audio_duration=2.0,
        theme="paper",
    )
    assert out.exists()
    assert out.stat().st_size > 1000
    ass = out.with_suffix(".ass").read_text()
    assert "How RAG Works" in ass
    assert r"\pos(540,1748)" in ass
    assert "GNode" in ass

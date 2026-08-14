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
    renderer = FFmpegRenderer(font_path=str(settings.font_path))
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
    renderer = FFmpegRenderer(font_path=str(settings.font_path))
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


def _top_left_rgb(video: Path, at_sec: float) -> tuple[int, int, int]:
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
    return raw.stdout[0], raw.stdout[1], raw.stdout[2]


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
    FFmpegRenderer(font_path=str(settings.font_path)).render(
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

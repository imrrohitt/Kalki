import subprocess
from pathlib import Path

import pytest

from app.captions.models import Caption, CaptionTimeline, CaptionWord
from app.config import settings
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

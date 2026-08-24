"""Render representative split-layout scenes for visual QA.

Usage: python scripts/preview_graphics.py <source_video> [out_dir] [theme]
Renders a 26s reel with one card per graphic kind, then dumps frames.
Themes: paper (default) | noir | tech | ivory
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.captions.models import Caption, CaptionTimeline, CaptionWord
from app.editorial.models import GraphicBeat, GraphicBullet
from app.renderer.ffmpeg_renderer import FFmpegRenderer


def beats() -> list[GraphicBeat]:
    return [
        GraphicBeat(
            start=0.0, end=5.0, kind="bullets",
            kicker="HOOK", title="90% of AI projects fail",
            bullets=[
                GraphicBullet(text="Most teams pick the wrong tool", delay_ms=0),
                GraphicBullet(text="The fix is simpler than you think", delay_ms=480),
            ],
        ),
        GraphicBeat(
            start=5.0, end=10.0, kind="vs_split",
            kicker="THE TRADEOFF", title="Which one should you use?",
            left="RAG", right="Fine-tuning",
            subtitle="fresh docs vs trained weights",
        ),
        GraphicBeat(
            start=10.0, end=15.0, kind="stat",
            kicker="THE NUMBER", title="73%",
            subtitle="of teams start with RAG",
        ),
        GraphicBeat(
            start=15.0, end=20.0, kind="process",
            kicker="RAG PATH", title="How RAG answers a question",
            chips=["Documents", "Retrieve", "Generate"],
        ),
        GraphicBeat(
            start=20.0, end=26.0, kind="quote",
            kicker="THE TAKE", title="The best model is the one you never retrain",
            subtitle="every platform engineer, eventually",
        ),
    ]


def captions() -> CaptionTimeline:
    caps = []
    for t, text, hot in [
        (0.4, "most AI projects fail", "fail"),
        (5.4, "RAG or fine-tuning", "RAG"),
        (10.4, "73 percent choose RAG", "73"),
        (15.4, "here is the pipeline", "pipeline"),
        (20.4, "never retrain blindly", "never"),
    ]:
        words = []
        wt = t
        for w in text.split():
            words.append(
                CaptionWord(text=w, start=wt, end=wt + 0.3, emphasis=(w == hot))
            )
            wt += 0.32
        caps.append(Caption(start=t, end=t + 2.2, text=text, words=words))
    return CaptionTimeline(captions=caps)


def main() -> None:
    src = sys.argv[1]
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "storage/preview")
    theme = sys.argv[3] if len(sys.argv) > 3 else None
    out_dir.mkdir(parents=True, exist_ok=True)
    trimmed = out_dir / "src26.mp4"
    if not trimmed.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-t", "26", "-c", "copy", str(trimmed)],
            check=True, capture_output=True,
        )
    out = out_dir / "preview.mp4"
    FFmpegRenderer(split_layout=True).render(
        str(trimmed), captions(), str(out),
        graphics=beats(), video_duration=26.0, theme=theme,
    )
    for name, at in [
        ("01_hook_mid", 0.35), ("02_hook_set", 3.0),
        ("03_vs_mid", 5.45), ("04_vs_set", 8.0),
        ("05_stat_mid", 10.5), ("06_stat_set", 13.0),
        ("07_proc_mid", 16.0), ("08_proc_set", 18.5),
        ("09_quote_set", 23.0),
    ]:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(at), "-i", str(out),
             "-frames:v", "1", str(out_dir / f"{name}.png")],
            check=True, capture_output=True,
        )
    print(f"wrote {out} and frames in {out_dir}")


if __name__ == "__main__":
    main()

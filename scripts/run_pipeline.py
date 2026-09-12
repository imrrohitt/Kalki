"""Run the full pipeline (transcribe -> editorial -> captions -> graphics -> render)
on a local file without the API server.

Usage: python scripts/run_pipeline.py <source_video> [out_dir] [theme] [split]
Themes: paper (default) | noir | tech | ivory
split: false (default) | true
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.captions.agent import CaptionAgentService
from app.editorial.decisions import EditorialIntelligenceEngine
from app.editorial.framing import default_visual
from app.media.audio import extract_audio
from app.media.probe import probe_video
from app.renderer.ffmpeg_renderer import FFmpegRenderer
from app.transcription.faster_whisper_provider import FasterWhisperProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_pipeline")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "split"}


async def main() -> None:
    src = sys.argv[1]
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "storage/real_run")
    theme = sys.argv[3] if len(sys.argv) > 3 else None
    split_layout = _truthy(sys.argv[4]) if len(sys.argv) > 4 else False
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe_video(src)
    log.info(
        "source %dx%d display %dx%d %.1fs split=%s",
        info.width,
        info.height,
        info.display_width,
        info.display_height,
        info.duration,
        split_layout,
    )

    audio = out_dir / "audio.wav"
    t0 = time.perf_counter()
    extract_audio(src, str(audio))
    transcript = await FasterWhisperProvider().transcribe(str(audio))
    (out_dir / "transcript.json").write_text(
        transcript.model_dump_json(indent=2), encoding="utf-8"
    )
    log.info(
        "transcribed %s words in %.1fs",
        sum(len(s.words) for s in transcript.segments),
        time.perf_counter() - t0,
    )

    duration = min(info.duration, transcript.duration or info.duration)
    visual = default_visual()
    editorial = EditorialIntelligenceEngine()

    analysis = await editorial.analyze(
        transcript=transcript, video_duration=duration, job_id="preview", visual=visual
    )
    log.info("editorial: %s sentences", len(analysis.sentences))

    timeline = await CaptionAgentService().generate(
        transcript=transcript, video_duration=duration, job_id="preview"
    )
    (out_dir / "captions.json").write_text(
        timeline.model_dump_json(indent=2), encoding="utf-8"
    )
    log.info("captions: %s groups", len(timeline.captions))

    plan = await editorial.plan(
        analysis,
        timeline,
        video_duration=duration,
        visual=visual,
        job_id="preview",
        split_layout=split_layout,
    )
    (out_dir / "edit_plan.json").write_text(
        plan.model_dump_json(indent=2), encoding="utf-8"
    )
    log.info("plan: %s graphics, %s sfx", len(plan.graphics), len(plan.sfx))

    t_render = time.perf_counter()
    FFmpegRenderer(split_layout=split_layout).render(
        source_video=src,
        caption_timeline=timeline,
        output_path=str(out_dir / "output.mp4"),
        zooms=[],
        graphics=plan.graphics,
        sfx=plan.sfx,
        video_duration=duration,
        theme=theme,
        split_layout=split_layout,
    )
    log.info("rendered in %.1fs", time.perf_counter() - t_render)
    audio.unlink(missing_ok=True)
    print(f"DONE {out_dir / 'output.mp4'}")


if __name__ == "__main__":
    asyncio.run(main())

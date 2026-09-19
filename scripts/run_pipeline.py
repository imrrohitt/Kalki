"""Run the full pipeline on a local file without the API server.

Full-frame (default): transcribe -> caption director -> premium captions + soundtrack.
Split: transcribe -> editorial -> captions -> graphics -> render.

Usage: python scripts/run_pipeline.py <source_video> [out_dir] [theme] [split] [reference.md] [caption_style]
Themes: paper (default) | noir | tech | ivory  (split layout only)
split: false (default) | true
reference.md: optional creator transcript/translation used as ground truth for meaning
caption_style: classic (default) | premium  (full-frame only)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.captions.agent import CaptionAgentService
from app.captions.director import CaptionDirector
from app.editorial.decisions import EditorialIntelligenceEngine
from app.editorial.framing import default_visual
from app.media.audio import extract_audio
from app.media.probe import probe_video
from app.renderer.ffmpeg_renderer import FFmpegRenderer
from app.renderer.soundtrack import plan_accents
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
    if not split_layout:
        reference = Path(sys.argv[5]).read_text(encoding="utf-8") if len(sys.argv) > 5 else ""
        caption_style = sys.argv[6] if len(sys.argv) > 6 else "classic"
        result = await CaptionDirector().direct(
            transcript,
            video_duration=info.duration,
            job_id="local",
            reference_text=reference,
            caption_style=caption_style,
        )
        (out_dir / "brief.json").write_text(
            json.dumps(result.brief.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "captions.json").write_text(
            result.timeline.model_dump_json(indent=2), encoding="utf-8"
        )
        (out_dir / "design.json").write_text(
            json.dumps(result.design(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("director: %s captions %s", len(result.timeline.captions), result.metrics)
        accents = plan_accents(result.timeline, video_duration=info.duration)
        t_render = time.perf_counter()
        FFmpegRenderer(split_layout=False).render_overlay_reel(
            source_video=src,
            caption_timeline=result.timeline,
            output_path=str(out_dir / "output.mp4"),
            accents=accents,
            music_mood=result.brief.music_mood,
            video_duration=info.duration,
            caption_style=caption_style,
        )
        log.info("rendered in %.1fs", time.perf_counter() - t_render)
        audio.unlink(missing_ok=True)
        print(f"DONE {out_dir / 'output.mp4'}")
        return
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

from __future__ import annotations

import json
import logging
import time

from app.captions.agent import CaptionAgentService
from app.config import settings
from app.editorial.decisions import EditorialIntelligenceEngine
from app.editorial.framing import default_visual
from app.media.audio import extract_audio
from app.media.probe import probe_video, validate_video
from app.pipeline.jobs import JobStatus, job_store
from app.renderer.ffmpeg_renderer import FFmpegRenderer
from app.transcription.faster_whisper_provider import FasterWhisperProvider

logger = logging.getLogger(__name__)


def _jid(job_id: str) -> str:
    return job_id[:8]


class Pipeline:
    def __init__(
        self,
        stt: FasterWhisperProvider | None = None,
        caption_agent: CaptionAgentService | None = None,
        renderer: FFmpegRenderer | None = None,
        editorial: EditorialIntelligenceEngine | None = None,
    ) -> None:
        self.stt = stt or FasterWhisperProvider()
        self.caption_agent = caption_agent or CaptionAgentService()
        self.renderer = renderer or FFmpegRenderer()
        self.editorial = editorial or EditorialIntelligenceEngine()

    async def run(self, job_id: str) -> None:
        job = job_store.get(job_id)
        if job is None:
            logger.error("job %s not found; skipping pipeline", job_id)
            return

        job_dir = settings.storage_path / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        audio_path = job_dir / "audio.wav"
        transcript_path = job_dir / "transcript.json"
        editorial_path = job_dir / "editorial.json"
        captions_path = job_dir / "captions.json"
        edit_plan_path = job_dir / "edit_plan.json"
        output_path = job_dir / "output.mp4"

        jid = _jid(job_id)
        t0 = time.perf_counter()
        logger.info("[%s] pipeline started", jid)
        try:
            job.set_stage(JobStatus.validating)
            logger.info("[%s] validating video", jid)
            info = probe_video(job.source_path)
            validate_video(info, settings.max_video_duration_sec)
            logger.info(
                "[%s] source %dx%d %.0ffps %.1fs %s%s",
                jid,
                info.width,
                info.height,
                info.fps,
                info.duration,
                info.video_codec or "video",
                f"+{info.audio_codec}" if info.audio_codec else "",
            )

            job.set_stage(JobStatus.extracting_audio)
            logger.info("[%s] extracting audio", jid)
            t_audio = time.perf_counter()
            extract_audio(job.source_path, str(audio_path))
            logger.info("[%s] audio extracted (%.1fs)", jid, time.perf_counter() - t_audio)

            job.set_stage(JobStatus.transcribing)
            logger.info("[%s] transcribing", jid)
            t_stt = time.perf_counter()
            transcript = await self.stt.transcribe(str(audio_path))
            job.metrics["transcription_time_ms"] = int(
                (time.perf_counter() - t_stt) * 1000
            )
            word_count = sum(len(seg.words) for seg in transcript.segments)
            logger.info(
                "[%s] transcript %s: %s words, %s segments (%.1fs)",
                jid,
                transcript.language or "unknown",
                word_count,
                len(transcript.segments),
                time.perf_counter() - t_stt,
            )
            transcript_path.write_text(
                transcript.model_dump_json(indent=2),
                encoding="utf-8",
            )
            job.transcript_path = str(transcript_path)

            duration = info.duration
            if transcript.duration:
                duration = min(duration, transcript.duration)
            visual = default_visual()

            job.set_stage(JobStatus.analyzing_editorial)
            logger.info("[%s] editorial analysis", jid)
            t_editorial = time.perf_counter()
            analysis = await self.editorial.analyze(
                transcript=transcript,
                video_duration=duration,
                job_id=job_id,
                visual=visual,
            )
            job.metrics["editorial_time_ms"] = int(
                (time.perf_counter() - t_editorial) * 1000
            )
            logger.info(
                "[%s] editorial: %s sentences (%.1fs)",
                jid,
                len(analysis.sentences),
                time.perf_counter() - t_editorial,
            )
            editorial_path.write_text(
                analysis.model_dump_json(indent=2),
                encoding="utf-8",
            )
            job.editorial_path = str(editorial_path)

            job.set_stage(JobStatus.generating_captions)
            logger.info("[%s] generating captions", jid)
            t_agent = time.perf_counter()
            timeline = await self.caption_agent.generate(
                transcript=transcript,
                video_duration=duration,
                job_id=job_id,
            )
            job.metrics["agent_time_ms"] = int(
                (time.perf_counter() - t_agent) * 1000
            )
            logger.info(
                "[%s] captions: %s groups (%.1fs)",
                jid,
                len(timeline.captions),
                time.perf_counter() - t_agent,
            )
            captions_path.write_text(
                timeline.model_dump_json(indent=2),
                encoding="utf-8",
            )
            job.captions_path = str(captions_path)

            job.set_stage(JobStatus.planning_edits)
            logger.info("[%s] planning edits (zooms + graphics + sfx)", jid)
            t_plan = time.perf_counter()
            edit_plan = await self.editorial.plan(
                analysis,
                timeline,
                video_duration=duration,
                visual=visual,
                job_id=job_id,
            )
            job.metrics["zoom_count"] = len(edit_plan.zooms)
            job.metrics["sfx_count"] = len(edit_plan.sfx)
            logger.info(
                "[%s] edit plan: %s zooms, %s graphic cards, %s sfx (%.1fs)",
                jid,
                len(edit_plan.zooms),
                len(edit_plan.graphics),
                len(edit_plan.sfx),
                time.perf_counter() - t_plan,
            )
            edit_plan_path.write_text(
                edit_plan.model_dump_json(indent=2),
                encoding="utf-8",
            )
            job.edit_plan_path = str(edit_plan_path)

            job.set_stage(JobStatus.rendering)
            logger.info("[%s] rendering %dx%d", jid, settings.output_width, settings.output_height)
            t_render = time.perf_counter()
            self.renderer.render(
                source_video=job.source_path,
                caption_timeline=timeline,
                output_path=str(output_path),
                zooms=edit_plan.zooms,
                graphics=edit_plan.graphics,
                sfx=edit_plan.sfx,
                video_duration=duration,
            )
            job.metrics["graphic_count"] = len(edit_plan.graphics)
            job.metrics["render_time_ms"] = int(
                (time.perf_counter() - t_render) * 1000
            )
            logger.info("[%s] render done (%.1fs)", jid, time.perf_counter() - t_render)

            job.result_path = str(output_path)
            job.metrics["total_processing_time_ms"] = int(
                (time.perf_counter() - t0) * 1000
            )
            job.metrics["stt_api_cost"] = 0
            job.set_stage(JobStatus.completed)
            logger.info("[%s] completed in %.1fs", jid, time.perf_counter() - t0)

            if audio_path.exists():
                audio_path.unlink()

        except Exception as exc:  # noqa: BLE001 - surface as job failure
            job.set_stage(JobStatus.failed)
            job.error = str(exc)
            job.metrics["total_processing_time_ms"] = int(
                (time.perf_counter() - t0) * 1000
            )
            logger.exception("[%s] failed at %s: %s", jid, job.stage, exc)
            (job_dir / "error.json").write_text(
                json.dumps({"error": str(exc)}, indent=2),
                encoding="utf-8",
            )
            if audio_path.exists():
                audio_path.unlink(missing_ok=True)

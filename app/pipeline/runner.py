from __future__ import annotations

import json
import time
from pathlib import Path

from app.captions.agent import CaptionAgentService
from app.config import settings
from app.media.audio import extract_audio
from app.media.probe import probe_video, validate_video
from app.pipeline.jobs import Job, JobStatus, job_store
from app.renderer.ffmpeg_renderer import FFmpegRenderer
from app.transcription.faster_whisper_provider import FasterWhisperProvider


class Pipeline:
    def __init__(
        self,
        stt: FasterWhisperProvider | None = None,
        caption_agent: CaptionAgentService | None = None,
        renderer: FFmpegRenderer | None = None,
    ) -> None:
        self.stt = stt or FasterWhisperProvider()
        self.caption_agent = caption_agent or CaptionAgentService()
        self.renderer = renderer or FFmpegRenderer()

    async def run(self, job_id: str) -> None:
        job = job_store.get(job_id)
        if job is None:
            return

        job_dir = settings.storage_path / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        audio_path = job_dir / "audio.wav"
        transcript_path = job_dir / "transcript.json"
        captions_path = job_dir / "captions.json"
        output_path = job_dir / "output.mp4"

        t0 = time.perf_counter()
        try:
            job.set_stage(JobStatus.validating)
            info = probe_video(job.source_path)
            validate_video(info, settings.max_video_duration_sec)

            job.set_stage(JobStatus.extracting_audio)
            extract_audio(job.source_path, str(audio_path))

            job.set_stage(JobStatus.transcribing)
            t_stt = time.perf_counter()
            transcript = await self.stt.transcribe(str(audio_path))
            job.metrics["transcription_time_ms"] = int(
                (time.perf_counter() - t_stt) * 1000
            )
            transcript_path.write_text(
                transcript.model_dump_json(indent=2),
                encoding="utf-8",
            )
            job.transcript_path = str(transcript_path)

            duration = info.duration
            if transcript.duration:
                duration = min(duration, transcript.duration)

            job.set_stage(JobStatus.generating_captions)
            t_agent = time.perf_counter()
            timeline = await self.caption_agent.generate(
                transcript=transcript,
                video_duration=duration,
                job_id=job_id,
            )
            job.metrics["agent_time_ms"] = int(
                (time.perf_counter() - t_agent) * 1000
            )
            captions_path.write_text(
                timeline.model_dump_json(indent=2),
                encoding="utf-8",
            )
            job.captions_path = str(captions_path)

            job.set_stage(JobStatus.rendering)
            t_render = time.perf_counter()
            self.renderer.render(
                source_video=job.source_path,
                caption_timeline=timeline,
                output_path=str(output_path),
            )
            job.metrics["render_time_ms"] = int(
                (time.perf_counter() - t_render) * 1000
            )

            job.result_path = str(output_path)
            job.metrics["total_processing_time_ms"] = int(
                (time.perf_counter() - t0) * 1000
            )
            job.metrics["stt_api_cost"] = 0
            job.set_stage(JobStatus.completed)

            # cleanup extracted audio
            if audio_path.exists():
                audio_path.unlink()

        except Exception as exc:  # noqa: BLE001 - surface as job failure
            job.set_stage(JobStatus.failed)
            job.error = str(exc)
            job.metrics["total_processing_time_ms"] = int(
                (time.perf_counter() - t0) * 1000
            )
            (job_dir / "error.json").write_text(
                json.dumps({"error": str(exc)}, indent=2),
                encoding="utf-8",
            )
            if audio_path.exists():
                audio_path.unlink(missing_ok=True)

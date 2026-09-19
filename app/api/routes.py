from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.pipeline.jobs import Job, JobStatus, job_store
from app.pipeline.runner import Pipeline
from app.renderer.design import THEMES
from app.transcription.markdown import transcript_from_upload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")
_pipeline: Pipeline | None = None

AUDIO_SUFFIXES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".flac",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
}

CAPTION_STYLES = {"classic", "premium"}


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


def _job_payload(job: Job, *, include_progress: bool = False) -> dict:
    payload = {
        "job_id": job.job_id,
        "status": job.status.value,
        "kind": job.kind,
        "theme": job.theme or settings.graphics_theme,
        "job_dir": str(job.job_dir),
        "output_path": str(job.output_file),
    }
    if job.kind == "video":
        payload["split_screen"] = job.split_layout
        payload["caption_style"] = job.caption_style
    if include_progress:
        payload["stage"] = job.stage
        payload["progress"] = job.progress
        payload["metrics"] = job.metrics
    if job.error:
        payload["error"] = job.error
    return payload


def _store_transcript(job: Job, file: UploadFile) -> None:
    raw = file.file.read()
    if not raw.strip():
        raise HTTPException(status_code=400, detail="Empty transcript file")
    name = Path(file.filename or "transcription.md").name
    dest = job.job_dir / name
    dest.write_bytes(raw)
    try:
        transcript = transcript_from_upload(raw, name)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid transcript: {exc}") from exc
    path = job.job_dir / "transcript.json"
    path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    job.transcript_path = str(path)
    job.skip_stt = True
    job.persist()


def _store_source(job: Job, file: UploadFile, suffix: str) -> None:
    dest = job.job_dir / f"source{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    job.source_path = str(dest)
    job.set_stage(JobStatus.uploaded)


@router.get("/themes")
async def list_themes():
    return {
        "default": settings.graphics_theme,
        "themes": sorted(THEMES.keys()),
    }


@router.post("/videos")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    transcript: UploadFile | None = File(None),
    theme: str = Query("", description="Motion-graphics theme: paper|noir|tech|ivory"),
    split_screen: bool = Query(
        False,
        description="If true, graphics panel above the speaker. Default is full-frame talking-head.",
    ),
    caption_style: str = Query(
        "classic",
        description=(
            "Full-frame captions only. 'classic' (default) is the existing caption "
            "director look, unchanged. 'premium' additionally uses a black CTA bubble "
            "on direct asks (follow/subscribe), an occasional hand-marker font line, "
            "and — when the framing shows enough of the speaker — an occasional "
            "chest-area placement."
        ),
    ),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    theme = theme.strip().lower()
    if theme and theme not in THEMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown theme '{theme}'. Available: {sorted(THEMES.keys())}",
        )
    caption_style = caption_style.strip().lower() or "classic"
    if caption_style not in CAPTION_STYLES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown caption_style '{caption_style}'. Available: {sorted(CAPTION_STYLES)}",
        )

    suffix = Path(file.filename).suffix or ".mp4"
    job = job_store.create(source_path="")
    job.theme = theme
    job.split_layout = split_screen
    job.caption_style = caption_style
    _store_source(job, file, suffix)
    if transcript is not None and transcript.filename:
        _store_transcript(job, transcript)

    size_mb = Path(job.source_path).stat().st_size / (1024 * 1024)
    logger.info(
        "[%s] uploaded %s (%.1f MB) theme=%s split_screen=%s caption_style=%s dir=%s",
        job.job_id[:8],
        file.filename,
        size_mb,
        theme or settings.graphics_theme,
        split_screen,
        caption_style,
        job.job_dir,
    )

    background_tasks.add_task(get_pipeline().run, job.job_id)
    return _job_payload(job)


@router.post("/reels")
async def upload_audio_reel(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    theme: str = Query("", description="Motion-graphics theme: paper|noir|tech|ivory"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    theme = theme.strip().lower()
    if theme and theme not in THEMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown theme '{theme}'. Available: {sorted(THEMES.keys())}",
        )

    suffix = Path(file.filename).suffix.lower() or ".wav"
    if suffix not in AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected an audio file ({', '.join(sorted(AUDIO_SUFFIXES))})",
        )

    job = job_store.create(source_path="", kind="audio_reel")
    job.theme = theme
    _store_source(job, file, suffix)

    size_mb = Path(job.source_path).stat().st_size / (1024 * 1024)
    logger.info(
        "[%s] uploaded audio reel %s (%.1f MB) dir=%s",
        job.job_id[:8],
        file.filename,
        size_mb,
        job.job_dir,
    )

    background_tasks.add_task(get_pipeline().run, job.job_id)
    return _job_payload(job)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return _job_payload(job, include_progress=True)


@router.get("/jobs/{job_id}/result")
async def get_result(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.completed:
        raise HTTPException(status_code=409, detail=f"Job not ready: {job.status.value}")
    path = Path(job.result_path) if job.result_path else job.output_file
    if not path.exists():
        path = job.output_file
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result file missing")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")

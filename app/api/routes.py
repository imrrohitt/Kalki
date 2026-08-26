from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.pipeline.jobs import JobStatus, job_store
from app.pipeline.runner import Pipeline
from app.renderer.design import THEMES

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


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


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
    theme: str = Query("", description="Motion-graphics theme: paper|noir|tech|ivory"),
    split_screen: bool = Query(
        False,
        description="If true, graphics panel above the speaker. Default is full-frame talking-head.",
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

    suffix = Path(file.filename).suffix or ".mp4"
    uploads = settings.storage_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    job = job_store.create(source_path="")
    job.theme = theme
    job.split_layout = split_screen
    dest = uploads / f"{job.job_id}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    job.source_path = str(dest)
    job.set_stage(JobStatus.uploaded)
    size_mb = dest.stat().st_size / (1024 * 1024)
    logger.info(
        "[%s] uploaded %s (%.1f MB) theme=%s split_screen=%s",
        job.job_id[:8],
        file.filename,
        size_mb,
        theme or settings.graphics_theme,
        split_screen,
    )

    background_tasks.add_task(get_pipeline().run, job.job_id)

    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "kind": job.kind,
        "theme": job.theme or settings.graphics_theme,
        "split_screen": job.split_layout,
    }


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

    uploads = settings.storage_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    job = job_store.create(source_path="", kind="audio_reel")
    job.theme = theme
    dest = uploads / f"{job.job_id}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    job.source_path = str(dest)
    job.set_stage(JobStatus.uploaded)
    size_mb = dest.stat().st_size / (1024 * 1024)
    logger.info(
        "[%s] uploaded audio reel %s (%.1f MB)",
        job.job_id[:8],
        file.filename,
        size_mb,
    )

    background_tasks.add_task(get_pipeline().run, job.job_id)

    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "kind": job.kind,
        "theme": job.theme or settings.graphics_theme,
    }


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    payload = {
        "job_id": job.job_id,
        "status": job.status.value,
        "kind": job.kind,
        "theme": job.theme or settings.graphics_theme,
        "stage": job.stage,
        "progress": job.progress,
        "metrics": job.metrics,
    }
    if job.kind == "video":
        payload["split_screen"] = job.split_layout
    if job.error:
        payload["error"] = job.error
    return payload


@router.get("/jobs/{job_id}/result")
async def get_result(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.completed or not job.result_path:
        raise HTTPException(status_code=409, detail=f"Job not ready: {job.status.value}")
    path = Path(job.result_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result file missing")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")

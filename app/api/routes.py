from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.pipeline.jobs import JobStatus, job_store
from app.pipeline.runner import Pipeline

router = APIRouter(prefix="/api/v1")
_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


@router.post("/videos")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix or ".mp4"
    uploads = settings.storage_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    job = job_store.create(source_path="")
    dest = uploads / f"{job.job_id}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    job.source_path = str(dest)
    job.set_stage(JobStatus.uploaded)

    background_tasks.add_task(get_pipeline().run, job.job_id)

    return {"job_id": job.job_id, "status": job.status.value}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    payload = {
        "job_id": job.job_id,
        "status": job.status.value,
        "stage": job.stage,
        "progress": job.progress,
        "metrics": job.metrics,
    }
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

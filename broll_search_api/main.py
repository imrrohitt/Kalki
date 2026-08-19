from __future__ import annotations

import logging
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse

from broll_search_api.agent import inspect_url, run_search
from broll_search_api.config import settings
from broll_search_api.jobs import job_store
from broll_search_api.models import InspectRequest, SearchRequest


logger = logging.getLogger(__name__)

app = FastAPI(
    title="B-roll Search API",
    version="0.1.0",
    description=(
        "Search APIs first, Playwright fallback for Google/Bing and JS pages. "
        "Standalone — not mounted on the editor API. "
        "A media URL is not a reuse license; copyrighted hosts are blocked."
    ),
)


def _ensure_storage() -> None:
    settings.storage_path.mkdir(parents=True, exist_ok=True)
    (settings.storage_path / "jobs").mkdir(parents=True, exist_ok=True)
    settings.library_path.mkdir(parents=True, exist_ok=True)


_ensure_storage()


async def _run_job(job_id: str, request: SearchRequest) -> None:
    job_store.set_stage(job_id, "running", "searching")
    try:
        result = await run_search(request, job_id=job_id)
        job_store.complete(job_id, result)
    except Exception as exc:
        logger.exception("B-roll search job %s failed", job_id)
        job_store.fail(job_id, str(exc))


@app.get("/health")
async def health() -> dict[str, str]:
    playwright_ok = "missing"
    try:
        import playwright  # noqa: F401

        playwright_ok = "ok"
    except Exception:
        playwright_ok = "missing"
    return {"status": "ok", "playwright": playwright_ok}


@app.post("/api/v1/broll/search")
async def start_search(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    wait: bool = False,
):
    if not request.transcript.strip() and not request.queries:
        raise HTTPException(status_code=400, detail="Provide transcript or queries")
    job = job_store.create()
    if wait:
        await _run_job(job.job_id, request)
        done = job_store.get(job.job_id)
        if done is None:
            raise HTTPException(status_code=500, detail="Job missing after run")
        return done.model_dump()
    background_tasks.add_task(_run_job, job.job_id, request)
    return {"job_id": job.job_id, "status": job.status}


@app.get("/api/v1/broll/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.model_dump()


@app.post("/api/v1/broll/inspect")
async def inspect(request: InspectRequest):
    try:
        result = await inspect_url(str(request.url), download=request.download)
    except Exception as exc:
        logger.exception("Inspect failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return result.model_dump()


@app.get("/api/v1/broll/library")
async def list_library():
    root = settings.library_path
    jobs = []
    if root.exists():
        for job_dir in sorted(root.iterdir()):
            if not job_dir.is_dir():
                continue
            accepted = job_dir / "accepted"
            files = [p.name for p in accepted.iterdir()] if accepted.exists() else []
            jobs.append({"job_id": job_dir.name, "files": files, "path": str(accepted)})
    return {"library_dir": str(root), "jobs": jobs}


@app.get("/api/v1/broll/library/{job_id}/{filename}")
async def get_library_asset(job_id: str, filename: str):
    safe = Path(filename).name
    for folder in ("accepted", "clips"):
        path = settings.library_path / job_id / folder / safe
        if path.exists() and path.is_file():
            media = "video/mp4" if path.suffix.lower() == ".mp4" else "application/octet-stream"
            return FileResponse(path, filename=safe, media_type=media)
    raise HTTPException(status_code=404, detail="Asset not found")


@app.get("/api/v1/broll/jobs/{job_id}/assets/{filename}")
async def get_asset(job_id: str, filename: str):
    if job_store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    safe = Path(filename).name
    for folder in ("assets", "clips"):
        path = settings.storage_path / "jobs" / job_id / folder / safe
        if path.exists() and path.is_file():
            media = "video/mp4" if path.suffix.lower() == ".mp4" else "application/octet-stream"
            return FileResponse(path, filename=safe, media_type=media)
    raise HTTPException(status_code=404, detail="Asset not found")

from __future__ import annotations

import uuid
from threading import Lock

from broll_search_api.models import SearchJob, SearchJobResult


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, SearchJob] = {}
        self._lock = Lock()

    def create(self) -> SearchJob:
        job = SearchJob(job_id=uuid.uuid4().hex[:12], status="queued", stage="queued")
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> SearchJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def set_stage(self, job_id: str, status: str, stage: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status  # type: ignore[assignment]
            job.stage = stage

    def complete(self, job_id: str, result: SearchJobResult) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "completed"
            job.stage = "completed"
            job.result = result

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "failed"
            job.stage = "failed"
            job.error = error


job_store = JobStore()

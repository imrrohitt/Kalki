from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    uploaded = "uploaded"
    validating = "validating"
    extracting_audio = "extracting_audio"
    transcribing = "transcribing"
    repairing_transcript = "repairing_transcript"
    analyzing_editorial = "analyzing_editorial"
    generating_captions = "generating_captions"
    planning_edits = "planning_edits"
    rendering = "rendering"
    completed = "completed"
    failed = "failed"


STAGE_PROGRESS = {
    JobStatus.uploaded: 5,
    JobStatus.validating: 10,
    JobStatus.extracting_audio: 25,
    JobStatus.transcribing: 45,
    JobStatus.repairing_transcript: 52,
    JobStatus.analyzing_editorial: 58,
    JobStatus.generating_captions: 72,
    JobStatus.planning_edits: 82,
    JobStatus.rendering: 90,
    JobStatus.completed: 100,
    JobStatus.failed: 100,
}


@dataclass
class Job:
    job_id: str
    source_path: str
    kind: str = "video"
    theme: str = ""
    status: JobStatus = JobStatus.uploaded
    stage: str = JobStatus.uploaded.value
    progress: int = 5
    error: str | None = None
    result_path: str | None = None
    transcript_path: str | None = None
    captions_path: str | None = None
    editorial_path: str | None = None
    edit_plan_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def set_stage(self, status: JobStatus) -> None:
        self.status = status
        self.stage = status.value
        self.progress = STAGE_PROGRESS[status]


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self, source_path: str, *, kind: str = "video") -> Job:
        job = Job(job_id=str(uuid.uuid4()), source_path=source_path, kind=kind)
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)


job_store = JobStore()

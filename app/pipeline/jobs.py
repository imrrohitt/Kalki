from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
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


def jobs_root() -> Path:
    from app.config import settings

    return settings.storage_path / "jobs"


@dataclass
class Job:
    job_id: str
    source_path: str
    kind: str = "video"
    theme: str = ""
    split_layout: bool = False
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

    @property
    def job_dir(self) -> Path:
        return jobs_root() / self.job_id

    @property
    def output_file(self) -> Path:
        return self.job_dir / "output.mp4"

    def set_stage(self, status: JobStatus) -> None:
        self.status = status
        self.stage = status.value
        self.progress = STAGE_PROGRESS[status]
        self.persist()

    def persist(self) -> None:
        self.job_dir.mkdir(parents=True, exist_ok=True)
        (self.job_dir / "job.json").write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "source_path": self.source_path,
            "kind": self.kind,
            "theme": self.theme,
            "split_layout": self.split_layout,
            "status": self.status.value,
            "stage": self.stage,
            "progress": self.progress,
            "error": self.error,
            "result_path": self.result_path,
            "transcript_path": self.transcript_path,
            "captions_path": self.captions_path,
            "editorial_path": self.editorial_path,
            "edit_plan_path": self.edit_plan_path,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        status = JobStatus(data.get("status", JobStatus.uploaded.value))
        return cls(
            job_id=data["job_id"],
            source_path=data.get("source_path", ""),
            kind=data.get("kind", "video"),
            theme=data.get("theme", ""),
            split_layout=bool(data.get("split_layout", False)),
            status=status,
            stage=data.get("stage", status.value),
            progress=int(data.get("progress", STAGE_PROGRESS[status])),
            error=data.get("error"),
            result_path=data.get("result_path"),
            transcript_path=data.get("transcript_path"),
            captions_path=data.get("captions_path"),
            editorial_path=data.get("editorial_path"),
            edit_plan_path=data.get("edit_plan_path"),
            metrics=dict(data.get("metrics") or {}),
        )


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self, source_path: str, *, kind: str = "video") -> Job:
        job = Job(job_id=str(uuid.uuid4()), source_path=source_path, kind=kind)
        job.job_dir.mkdir(parents=True, exist_ok=True)
        self._jobs[job.job_id] = job
        job.persist()
        return job

    def get(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is not None:
            return job
        path = jobs_root() / job_id / "job.json"
        if not path.exists():
            return None
        try:
            job = Job.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return None
        self._jobs[job_id] = job
        return job


job_store = JobStore()

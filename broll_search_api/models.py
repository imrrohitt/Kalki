from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


MediaKind = Literal["video", "image", "stream", "embed", "download"]
MediaVia = Literal["dom", "network", "og", "download_link", "api"]
LicenseStatus = Literal["reusable", "copyrighted", "unknown"]
JobStatus = Literal["queued", "running", "completed", "failed"]
CandidateAction = Literal["downloaded", "skipped", "blocked", "rejected", "accepted"]


class SearchRequest(BaseModel):
    transcript: str = ""
    queries: list[str] | None = None
    max_queries: int = Field(4, ge=1, le=8)
    max_results_per_query: int = Field(5, ge=1, le=12)
    max_pages_to_open: int = Field(6, ge=0, le=15)
    download: bool = True
    prepare_ffmpeg: bool = True
    review: bool = True
    max_downloads: int = Field(10, ge=1, le=20)
    engines: list[str] = Field(
        default_factory=lambda: ["wikimedia", "duckduckgo", "bing", "google"]
    )


class InspectRequest(BaseModel):
    url: HttpUrl
    download: bool = False


class TopicQuery(BaseModel):
    event: str
    queries: list[str] = Field(default_factory=list)
    media_need: str = ""


class SearchHit(BaseModel):
    engine: str
    query: str
    title: str
    url: str
    snippet: str = ""
    recency_hint: str = ""
    layer: Literal["api", "playwright"] = "api"
    search_url: str = ""


class DetectedMedia(BaseModel):
    kind: MediaKind
    url: str
    page_url: str
    via: MediaVia
    mime: str | None = None
    title: str = ""
    width: int | None = None
    height: int | None = None
    page_license_hint: str = ""


class RightsDecision(BaseModel):
    status: LicenseStatus
    label: str = ""
    reason: str = ""
    source_domain: str = ""


class AssetVerdict(BaseModel):
    keep: bool
    score: float = Field(0.0, ge=0.0, le=1.0)
    reason: str = ""
    checks: list[str] = Field(default_factory=list)


class BrollCandidate(BaseModel):
    title: str
    page_url: str
    media_url: str
    kind: MediaKind
    via: MediaVia
    mime: str | None = None
    rights: RightsDecision
    action: CandidateAction
    rank: float = 0.0
    local_path: str | None = None
    ffmpeg_path: str | None = None
    verdict: AssetVerdict | None = None


class SearchJobResult(BaseModel):
    topics: list[TopicQuery] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    hits: list[SearchHit] = Field(default_factory=list)
    candidates: list[BrollCandidate] = Field(default_factory=list)
    downloaded: int = 0
    accepted: int = 0
    rejected: int = 0
    blocked: int = 0
    skipped: int = 0
    used_playwright: bool = False
    library_dir: str | None = None
    visited_pages: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    note: str = (
        "Search APIs run first. Playwright is the fallback/browser layer. "
        "Only reusable-licensed assets are downloaded. "
        "Finding a video URL does not grant reuse rights."
    )


class SearchJob(BaseModel):
    job_id: str
    status: JobStatus
    stage: str = "queued"
    error: str | None = None
    result: SearchJobResult | None = None

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.captions.models import Caption, CaptionTimeline, CaptionWord
from app.main import create_app
from app.pipeline.jobs import job_store
from app.pipeline.runner import Pipeline
from app.renderer.ffmpeg_renderer import FFmpegRenderer
from app.transcription.models import Segment, Transcript, Word
import app.api.routes as routes


def _make_video(path: Path) -> None:
    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=1280x720:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=2",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_upload_job_status_and_result(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    from app.config import Settings

    test_settings = Settings(
        storage_dir=str(tmp_path / "storage"),
        caption_font_path="assets/fonts/Montserrat-Bold.ttf",
    )
    monkeypatch.setattr("app.config.settings", test_settings)
    monkeypatch.setattr("app.api.routes.settings", test_settings)
    monkeypatch.setattr("app.pipeline.runner.settings", test_settings)

    fake_transcript = Transcript(
        language="en",
        language_probability=0.99,
        duration=2.0,
        segments=[
            Segment(
                start=0.0,
                end=1.5,
                text="hello world",
                words=[
                    Word(word="hello", start=0.0, end=0.5, probability=0.9),
                    Word(word="world", start=0.5, end=1.0, probability=0.9),
                ],
            )
        ],
    )
    fake_timeline = CaptionTimeline(
        captions=[
            Caption(
                start=0.0,
                end=1.2,
                text="HELLO WORLD",
                words=[
                    CaptionWord(text="HELLO", start=0.0, end=0.5, emphasis=True),
                    CaptionWord(text="WORLD", start=0.5, end=1.0, emphasis=False),
                ],
            )
        ]
    )

    stt = MagicMock()
    stt.transcribe = AsyncMock(return_value=fake_transcript)
    agent = MagicMock()
    agent.generate = AsyncMock(return_value=fake_timeline)
    routes._pipeline = Pipeline(
        stt=stt,
        caption_agent=agent,
        renderer=FFmpegRenderer(font_path=str(test_settings.font_path)),
    )

    app = create_app()
    video = tmp_path / "clip.mp4"
    _make_video(video)

    with TestClient(app) as client:
        with video.open("rb") as f:
            resp = client.post(
                "/api/v1/videos",
                files={"file": ("clip.mp4", f, "video/mp4")},
            )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        status = None
        for _ in range(40):
            status = client.get(f"/api/v1/jobs/{job_id}").json()
            if status["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)

        assert status is not None
        assert status["status"] == "completed", status
        assert status["progress"] == 100
        assert "zoom_count" in status["metrics"]

        result = client.get(f"/api/v1/jobs/{job_id}/result")
        assert result.status_code == 200
        assert "video" in result.headers["content-type"]


def test_job_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    from app.config import Settings

    test_settings = Settings(storage_dir=str(tmp_path / "storage"))
    monkeypatch.setattr("app.config.settings", test_settings)
    monkeypatch.setattr("app.api.routes.settings", test_settings)
    routes._pipeline = Pipeline(
        stt=MagicMock(),
        caption_agent=MagicMock(),
        renderer=MagicMock(),
    )
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/api/v1/jobs/does-not-exist").status_code == 404
        _ = job_store

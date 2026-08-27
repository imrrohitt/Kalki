import time
import uuid
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
        renderer=FFmpegRenderer(
            font_path=str(test_settings.font_path),
            split_layout=True,
        ),
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
        body = resp.json()
        assert body["split_screen"] is False
        job_id = body["job_id"]
        uuid.UUID(job_id)
        assert Path(body["job_dir"]).name == job_id
        assert body["output_path"].endswith(f"{job_id}/output.mp4")

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
        assert Path(body["output_path"]).is_file()
        assert Path(body["output_path"]).name == "output.mp4"

        result = client.get(f"/api/v1/jobs/{job_id}/result")
        assert result.status_code == 200
        assert "video" in result.headers["content-type"]


def _make_audio(path: Path) -> None:
    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=2",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_upload_audio_reel_job_status_and_result(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    from app.config import Settings

    test_settings = Settings(
        storage_dir=str(tmp_path / "storage"),
        caption_font_path="assets/fonts/Montserrat-Bold.ttf",
        transcript_repair_llm_enabled=False,
        scenes_llm_enabled=False,
        editorial_llm_enabled=False,
        sfx_llm_enabled=False,
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
                text="why rag beats fine tuning",
                words=[
                    Word(word="why", start=0.0, end=0.4, probability=0.9),
                    Word(word="rag", start=0.4, end=0.8, probability=0.9),
                    Word(word="beats", start=0.8, end=1.1, probability=0.9),
                    Word(word="fine", start=1.1, end=1.3, probability=0.9),
                    Word(word="tuning", start=1.3, end=1.6, probability=0.9),
                ],
            )
        ],
    )
    fake_timeline = CaptionTimeline(
        captions=[
            Caption(
                start=0.0,
                end=1.2,
                text="Why RAG",
                words=[
                    CaptionWord(text="Why", start=0.0, end=0.5, emphasis=True),
                    CaptionWord(text="RAG", start=0.5, end=1.0, emphasis=False),
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
    audio = tmp_path / "talk.wav"
    _make_audio(audio)

    with TestClient(app) as client:
        with audio.open("rb") as f:
            resp = client.post(
                "/api/v1/reels",
                files={"file": ("talk.wav", f, "audio/wav")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["kind"] == "audio_reel"
        job_id = body["job_id"]

        status = None
        for _ in range(60):
            status = client.get(f"/api/v1/jobs/{job_id}").json()
            if status["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)

        assert status is not None
        assert status["status"] == "completed", status
        assert status["kind"] == "audio_reel"
        assert status["metrics"].get("graphic_count", 0) >= 1

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


def test_videos_query_params_theme_and_split_screen(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    from app.config import Settings

    test_settings = Settings(storage_dir=str(tmp_path / "storage"))
    monkeypatch.setattr("app.config.settings", test_settings)
    monkeypatch.setattr("app.api.routes.settings", test_settings)
    routes._pipeline = MagicMock()
    routes._pipeline.run = AsyncMock()
    app = create_app()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not-a-real-video")

    with TestClient(app) as client:
        with video.open("rb") as f:
            default = client.post(
                "/api/v1/videos?theme=tech",
                files={"file": ("clip.mp4", f, "video/mp4")},
            )
        assert default.status_code == 200
        body = default.json()
        uuid.UUID(body["job_id"])
        assert body["theme"] == "tech"
        assert body["split_screen"] is False
        assert body["job_dir"].endswith(body["job_id"])
        assert body["output_path"] == str(Path(body["job_dir"]) / "output.mp4")
        job_dir = Path(body["job_dir"])
        assert job_dir.is_dir()
        assert (job_dir / "source.mp4").exists()
        assert (job_dir / "job.json").exists()

        job = job_store.get(body["job_id"])
        assert job is not None
        assert job.theme == "tech"
        assert job.split_layout is False

        status = client.get(f"/api/v1/jobs/{body['job_id']}").json()
        assert status["theme"] == "tech"
        assert status["split_screen"] is False

        with video.open("rb") as f:
            split = client.post(
                "/api/v1/videos?theme=noir&split_screen=true",
                files={"file": ("clip.mp4", f, "video/mp4")},
            )
        assert split.status_code == 200
        body = split.json()
        assert body["theme"] == "noir"
        assert body["split_screen"] is True

        job = job_store.get(body["job_id"])
        assert job is not None
        assert job.split_layout is True

        # Same UUID is still readable after the in-memory map is cleared.
        job_store._jobs.clear()
        revived = client.get(f"/api/v1/jobs/{body['job_id']}").json()
        assert revived["job_id"] == body["job_id"]
        assert revived["split_screen"] is True
        assert revived["theme"] == "noir"

        with video.open("rb") as f:
            bad = client.post(
                "/api/v1/videos?theme=bogus",
                files={"file": ("clip.mp4", f, "video/mp4")},
            )
        assert bad.status_code == 400

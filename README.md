# AI Reel Editor — Captions MVP

Upload video → faster-whisper → ADK Caption Agent (Groq via LiteLLM) → FFmpeg → 9:16 MP4.

## Setup

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set LLM_API_KEY / LLM_MODEL for Groq
```

Requires system FFmpeg with `drawtext` (on macOS: `brew install ffmpeg-full`) and `ffprobe`.

Set in `.env`:

```env
FFMPEG_PATH=/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg
FFPROBE_PATH=/opt/homebrew/opt/ffmpeg-full/bin/ffprobe
```

## Run API

```bash
uvicorn app.main:app --reload --port 8000
```

- `POST /api/v1/videos` — upload video (max 60s)
- `GET /api/v1/jobs/{job_id}` — status
- `GET /api/v1/jobs/{job_id}/result` — final MP4

## Tests

```bash
pytest -q
```

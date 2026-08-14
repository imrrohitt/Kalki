# AI Reel Editor — Captions MVP

Upload video → faster-whisper → Google ADK `LlmAgent` (DeepSeek via LiteLLM) → FFmpeg → 9:16 MP4.

Captions are produced by an autonomous ADK agent with a `submit_caption_groups` tool, not a one-shot completion call. Set `LLM_API_KEY` in `.env`.

## Setup

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set LLM_API_KEY / LLM_MODEL
```

Requires system FFmpeg with `drawtext` (on macOS: `brew install ffmpeg-full`) and `ffprobe`.

Set in `.env`:

```env
FFMPEG_PATH=/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg
FFPROBE_PATH=/opt/homebrew/opt/ffmpeg-full/bin/ffprobe
MAX_VIDEO_DURATION_SEC=0
CAPTION_CHUNK_SECONDS=0
```

`MAX_VIDEO_DURATION_SEC` is the upload length threshold in seconds. `0` means no limit. `CAPTION_CHUNK_SECONDS` is how long each caption-agent window is; `0` sends the whole transcript in one request (faster on DeepSeek, which has high per-request latency).

## Run API

```bash
uvicorn app.main:app --reload --port 8000
```

- `POST /api/v1/videos` — upload video (length limited by `MAX_VIDEO_DURATION_SEC`; `0` = no limit)
- `GET /api/v1/jobs/{job_id}` — status
- `GET /api/v1/jobs/{job_id}/result` — final MP4

## Tests

```bash
pytest -q
```

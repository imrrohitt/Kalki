"""Run the audio-only reel pipeline on a local file.

Usage: python scripts/run_reel.py <source_audio> [out_dir] [theme]
Themes: paper (default) | noir | tech | ivory
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.pipeline.jobs import JobStatus, job_store
from app.pipeline.runner import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_reel")


async def main() -> None:
    src = Path(sys.argv[1]).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"audio not found: {src}")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "storage/reel_run")
    theme = sys.argv[3] if len(sys.argv) > 3 else ""
    out_dir.mkdir(parents=True, exist_ok=True)

    dest = out_dir / f"source{src.suffix or '.wav'}"
    if dest.resolve() != src:
        shutil.copy2(src, dest)

    job = job_store.create(source_path=str(dest), kind="audio_reel")
    job.theme = theme
    job.set_stage(JobStatus.uploaded)
    log.info("job %s theme=%s source=%s", job.job_id[:8], theme or settings.graphics_theme, dest)

    pipeline = Pipeline()
    await pipeline.run(job.job_id)

    job = job_store.get(job.job_id)
    if job is None or job.status != JobStatus.completed or not job.result_path:
        raise SystemExit(f"failed: {job.status.value if job else 'missing'} {job.error if job else ''}")

    final = out_dir / "output.mp4"
    shutil.copy2(job.result_path, final)
    artifacts = Path(job.result_path).parent
    for name in (
        "transcript.raw.json",
        "transcript.json",
        "editorial.json",
        "captions.json",
        "edit_plan.json",
        "output.ass",
    ):
        src_art = artifacts / name
        if src_art.exists():
            shutil.copy2(src_art, out_dir / name)
    log.info("done in %.1fs → %s", (job.metrics or {}).get("total_processing_time_ms", 0) / 1000, final)
    print(f"DONE {final}")


if __name__ == "__main__":
    asyncio.run(main())

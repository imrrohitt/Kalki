from __future__ import annotations

import logging
import mimetypes
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
from playwright.async_api import Page

from broll_search_api.config import settings
from broll_search_api.models import DetectedMedia


logger = logging.getLogger(__name__)

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _filename_from_url(url: str, fallback: str) -> str:
    path = unquote(urlparse(url).path)
    name = Path(path).name or fallback
    name = SAFE_NAME.sub("_", name).strip("._") or fallback
    return name[:120]


async def download_http(url: str, dest_dir: Path, suggested: str | None = None) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": settings.user_agent, "Accept": "*/*"}
    timeout = httpx.Timeout(40.0, connect=15.0)
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                length = int(response.headers.get("content-length") or 0)
                if length and length > settings.max_asset_bytes:
                    logger.warning("Skip %s: content-length %s", url, length)
                    return None
                mime = (response.headers.get("content-type") or "").split(";")[0].strip()
                name = suggested or _filename_from_url(url, "asset")
                if "." not in name:
                    ext = mimetypes.guess_extension(mime) or ""
                    if ext:
                        name = name + ext
                dest = dest_dir / name
                if dest.exists():
                    dest = dest_dir / f"{dest.stem}_{abs(hash(url)) % 99999}{dest.suffix}"
                written = 0
                with dest.open("wb") as handle:
                    async for chunk in response.aiter_bytes(64 * 1024):
                        written += len(chunk)
                        if written > settings.max_asset_bytes:
                            handle.close()
                            dest.unlink(missing_ok=True)
                            logger.warning("Skip %s: exceeded size cap", url)
                            return None
                        handle.write(chunk)
                if written == 0:
                    dest.unlink(missing_ok=True)
                    return None
                head = dest.read_bytes()[:32].lstrip().lower()
                if head.startswith(b"<!doct") or head.startswith(b"<html"):
                    dest.unlink(missing_ok=True)
                    logger.warning("Skip %s: HTML body, not media", url)
                    return None
                return dest
    except Exception:
        logger.exception("HTTP download failed for %s", url)
        return None


async def download_via_playwright(page: Page, media: DetectedMedia, dest_dir: Path) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        async with page.expect_download(timeout=8000) as pending:
            await page.locator("a[download], a:has-text('Download')").first.click(timeout=4000)
        download = await pending.value
        name = SAFE_NAME.sub("_", download.suggested_filename or "download")
        dest = dest_dir / name
        await download.save_as(str(dest))
        return dest if dest.exists() and dest.stat().st_size > 0 else None
    except Exception:
        logger.info("No native download event on %s", media.page_url)
        return None

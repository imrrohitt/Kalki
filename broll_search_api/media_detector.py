from __future__ import annotations

import logging
from urllib.parse import urlparse

from playwright.async_api import Page, Response

from broll_search_api.browser import dismiss_consent, goto
from broll_search_api.models import DetectedMedia, MediaKind, MediaVia
from broll_search_api.search import normalize_url


logger = logging.getLogger(__name__)

VIDEO_EXT = {".mp4", ".webm", ".mov", ".m4v", ".mkv"}
STREAM_EXT = {".m3u8", ".mpd"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".avif"}
DOWNLOAD_EXT = VIDEO_EXT | IMAGE_EXT | {".zip"}

PAGE_EXTRACT_JS = """
() => {
  const abs = (u) => {
    try { return new URL(u, document.baseURI).href; } catch { return ""; }
  };
  const videos = [...document.querySelectorAll("video, source")].map((el) => ({
    src: el.currentSrc || el.src || el.getAttribute("src") || "",
    type: el.getAttribute("type") || "",
  }));
  const images = [...document.querySelectorAll("img")].slice(0, 40).map((el) => ({
    src: el.currentSrc || el.src || el.getAttribute("src") || "",
    alt: el.alt || "",
    width: el.naturalWidth || el.width || 0,
    height: el.naturalHeight || el.height || 0,
  }));
  const og = {
    image: document.querySelector('meta[property="og:image"]')?.content || "",
    video: document.querySelector('meta[property="og:video"]')?.content
      || document.querySelector('meta[property="og:video:url"]')?.content || "",
    title: document.querySelector('meta[property="og:title"]')?.content
      || document.title || "",
  };
  const license = document.querySelector('link[rel="license"]')?.href
    || document.querySelector(".licensetpl_short")?.textContent
    || document.querySelector("[itemprop='license']")?.content
    || document.querySelector("[itemprop='license']")?.href
    || document.querySelector(".licensetpl a")?.textContent
    || "";
  const downloads = [...document.querySelectorAll("a[download], a[href$='.mp4'], a[href$='.webm'], a[href$='.mov'], a[href$='.zip']")].map((a) => ({
    href: a.href,
    text: (a.innerText || "").trim(),
    download: a.getAttribute("download") || "",
  }));
  const embeds = [...document.querySelectorAll("iframe")].map((el) => el.src || "").filter(Boolean);
  return {videos, images, og, license, downloads, embeds, canonical: abs(location.href)};
}
"""


def _ext(url: str) -> str:
    path = urlparse(url).path.lower()
    if "." not in path.rsplit("/", 1)[-1]:
        return ""
    return "." + path.rsplit(".", 1)[-1]


def _kind_from_url(url: str, mime: str | None = None) -> MediaKind | None:
    mime = (mime or "").split(";")[0].strip().lower()
    ext = _ext(url)
    if mime.startswith("video/") or ext in VIDEO_EXT:
        return "video"
    if "mpegurl" in mime or mime == "application/vnd.apple.mpegurl" or ext in STREAM_EXT:
        return "stream"
    if mime.startswith("image/") or ext in IMAGE_EXT:
        return "image"
    if ext in DOWNLOAD_EXT:
        return "download"
    return None


class NetworkTap:
    def __init__(self, page_url: str) -> None:
        self.page_url = page_url
        self.found: list[DetectedMedia] = []
        self._seen: set[str] = set()

    def attach(self, page: Page) -> None:
        page.on("response", self._on_response)

    def _add(self, url: str, mime: str | None, via: MediaVia, kind: MediaKind | None = None) -> None:
        url = normalize_url(url)
        if not url or url in self._seen:
            return
        kind = kind or _kind_from_url(url, mime)
        if kind is None:
            return
        self._seen.add(url)
        self.found.append(
            DetectedMedia(
                kind=kind,
                url=url,
                page_url=self.page_url,
                via=via,
                mime=mime,
            )
        )

    def _on_response(self, response: Response) -> None:
        try:
            url = response.url
            headers = response.headers or {}
            mime = headers.get("content-type")
            self._add(url, mime, "network")
        except Exception:
            return


async def inspect_page(page: Page, url: str) -> list[DetectedMedia]:
    tap = NetworkTap(url)
    tap.attach(page)
    try:
        await goto(page, url)
        await dismiss_consent(page)
    except Exception:
        logger.exception("Failed to open %s", url)
        return tap.found

    try:
        payload = await page.evaluate(PAGE_EXTRACT_JS)
    except Exception:
        logger.exception("DOM extract failed for %s", url)
        payload = {}

    page_url = payload.get("canonical") or url
    title = (payload.get("og") or {}).get("title") or ""
    license_hint = (payload.get("license") or "").strip()
    collected: list[DetectedMedia] = list(tap.found)
    seen = {item.url for item in collected}

    def add(kind: MediaKind, media_url: str, via: MediaVia, mime: str | None = None, width: int | None = None, height: int | None = None) -> None:
        media_url = normalize_url(media_url)
        if not media_url or media_url in seen:
            return
        seen.add(media_url)
        collected.append(
            DetectedMedia(
                kind=kind,
                url=media_url,
                page_url=page_url,
                via=via,
                mime=mime,
                title=title,
                width=width,
                height=height,
                page_license_hint=license_hint,
            )
        )

    for video in payload.get("videos") or []:
        src = video.get("src") or ""
        kind = _kind_from_url(src, video.get("type")) or "video"
        add(kind, src, "dom", video.get("type") or None)

    for image in payload.get("images") or []:
        width = int(image.get("width") or 0) or None
        height = int(image.get("height") or 0) or None
        if width and width < 160:
            continue
        add("image", image.get("src") or "", "dom", width=width, height=height)

    og = payload.get("og") or {}
    if og.get("image"):
        add("image", og["image"], "og")
    if og.get("video"):
        kind = _kind_from_url(og["video"]) or "video"
        add(kind, og["video"], "og")

    for item in payload.get("downloads") or []:
        href = item.get("href") or ""
        kind = _kind_from_url(href) or "download"
        add(kind, href, "download_link")

    for embed in payload.get("embeds") or []:
        add("embed", embed, "dom")

    for item in collected:
        if not item.title:
            item.title = title
        if license_hint and not item.page_license_hint:
            item.page_license_hint = license_hint

    return collected

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

from playwright.async_api import Page

from broll_search_api.browser import launch_browser
from broll_search_api.config import settings
from broll_search_api.downloader import download_http, download_via_playwright
from broll_search_api.ffmpeg_prep import prepare_clip
from broll_search_api.media_detector import inspect_page
from broll_search_api.models import (
    BrollCandidate,
    DetectedMedia,
    SearchHit,
    SearchJobResult,
    SearchRequest,
)
from broll_search_api.query_gen import flatten_queries, generate_topics
from broll_search_api.reviewer import AssetReviewAgent, PagePickAgent, place_in_library
from broll_search_api.rights import classify, rank
from broll_search_api.search import (
    needs_playwright_search,
    run_api_searches,
    run_playwright_searches,
)


logger = logging.getLogger(__name__)


def _job_dir(job_id: str) -> Path:
    path = settings.storage_path / "jobs" / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dedupe_media(items: list[DetectedMedia]) -> list[DetectedMedia]:
    seen: set[str] = set()
    out: list[DetectedMedia] = []
    for item in items:
        key = item.url.split("?")[0]
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _select_for_download(candidates: list[BrollCandidate], limit: int = 8) -> list[BrollCandidate]:
    reusable = [item for item in candidates if item.rights.status == "reusable"]
    reusable.sort(key=lambda item: (item.kind != "video", -item.rank))
    picked: list[BrollCandidate] = []
    seen_urls: set[str] = set()
    for item in reusable:
        if item.media_url in seen_urls:
            continue
        seen_urls.add(item.media_url)
        picked.append(item)
        if len(picked) >= limit:
            break
    return picked


def _candidates_from_media(detected: list[DetectedMedia], query_blob: str) -> list[BrollCandidate]:
    candidates: list[BrollCandidate] = []
    for media in detected:
        rights = classify(media)
        action = "blocked" if rights.status == "copyrighted" else "skipped"
        candidates.append(
            BrollCandidate(
                title=media.title or media.page_url,
                page_url=media.page_url,
                media_url=media.url,
                kind=media.kind,
                via=media.via,
                mime=media.mime,
                rights=rights,
                action=action,
                rank=rank(media, rights, query_blob),
            )
        )
    return candidates


async def _download_reusable(
    page: Page | None,
    candidates: list[BrollCandidate],
    assets_dir: Path,
    clips_dir: Path,
    prepare: bool,
    limit: int = 10,
) -> None:
    for candidate in _select_for_download(candidates, limit=limit):
        local: Path | None = None
        if page is not None and candidate.via == "download_link":
            try:
                await page.goto(candidate.page_url, wait_until="domcontentloaded", timeout=20000)
                local = await download_via_playwright(
                    page,
                    DetectedMedia(
                        kind=candidate.kind,
                        url=candidate.media_url,
                        page_url=candidate.page_url,
                        via=candidate.via,
                    ),
                    assets_dir,
                )
            except Exception:
                local = None
        if local is None and candidate.kind != "stream":
            local = await download_http(candidate.media_url, assets_dir)
        if local is None:
            continue
        candidate.action = "downloaded"
        candidate.local_path = str(local)
        if prepare:
            clip = prepare_clip(local, clips_dir)
            if clip is not None:
                candidate.ffmpeg_path = str(clip)


async def run_search(request: SearchRequest, job_id: str | None = None) -> SearchJobResult:
    job_id = job_id or uuid.uuid4().hex[:12]
    dest = _job_dir(job_id)

    if request.queries:
        topics = []
        queries = [q.strip() for q in request.queries if q.strip()][: request.max_queries * 2]
    else:
        topics = await generate_topics(request.transcript, request.max_queries)
        queries = flatten_queries(topics, request.max_queries)
    if not queries:
        return SearchJobResult(topics=topics, note="No search queries could be generated.")

    hits: list[SearchHit] = []
    detected: list[DetectedMedia] = []
    used_playwright = False
    visited_pages: list[str] = []

    api_hits, wiki_media, api_counts = await run_api_searches(
        queries,
        request.engines,
        request.max_results_per_query,
    )
    hits.extend(api_hits)
    detected.extend(wiki_media)

    news_pages = [hit for hit in hits if hit.engine != "wikimedia"][: request.max_pages_to_open]
    need_browser = needs_playwright_search(
        request.engines, api_counts, request.max_results_per_query
    ) or bool(news_pages)

    if need_browser:
        used_playwright = True
        async with launch_browser() as session:
            if needs_playwright_search(request.engines, api_counts, request.max_results_per_query):
                pw_hits = await run_playwright_searches(
                    session,
                    queries,
                    request.engines,
                    request.max_results_per_query,
                    api_counts,
                )
                seen = {hit.url for hit in hits}
                for hit in pw_hits:
                    if hit.url in seen:
                        continue
                    seen.add(hit.url)
                    hits.append(hit)

            page_hits = [hit for hit in hits if hit.engine != "wikimedia"]
            if request.max_pages_to_open > 0 and page_hits:
                news_pages = await PagePickAgent().pick(
                    page_hits,
                    request.transcript,
                    queries,
                    limit=min(5, request.max_pages_to_open),
                )
            else:
                news_pages = []
            visited_pages = [hit.url for hit in news_pages]

            sem = asyncio.Semaphore(2)

            async def inspect_hit(hit: SearchHit) -> list[DetectedMedia]:
                async with sem:
                    page = await session.new_page()
                    try:
                        found = await inspect_page(page, hit.url)
                        for item in found:
                            if not item.title:
                                item.title = hit.title
                        return found
                    finally:
                        await page.close()

            if news_pages:
                pages = await asyncio.gather(
                    *[inspect_hit(hit) for hit in news_pages],
                    return_exceptions=True,
                )
                for batch in pages:
                    if isinstance(batch, Exception):
                        logger.warning("Page inspect failed: %s", batch)
                        continue
                    detected.extend(batch)

            detected = _dedupe_media(detected)
            query_blob = " ".join(queries)
            candidates = _candidates_from_media(detected, query_blob)

            if request.download:
                page = await session.new_page()
                try:
                    await _download_reusable(
                        page,
                        candidates,
                        dest / "incoming",
                        dest / "clips",
                        request.prepare_ffmpeg,
                        limit=request.max_downloads,
                    )
                finally:
                    await page.close()
    else:
        detected = _dedupe_media(detected)
        candidates = _candidates_from_media(detected, " ".join(queries))
        if request.download:
            await _download_reusable(
                None,
                candidates,
                dest / "incoming",
                dest / "clips",
                request.prepare_ffmpeg,
                limit=request.max_downloads,
            )

    library_dir = None
    if request.download and request.review:
        await AssetReviewAgent().review(
            candidates,
            request.transcript,
            queries,
            topics,
        )
        library_dir = str(
            place_in_library(job_id, candidates, dest / "rejected")
        )
    elif request.download:
        for candidate in candidates:
            if candidate.action == "downloaded":
                candidate.action = "accepted"
        library_dir = str(place_in_library(job_id, candidates, dest / "rejected"))

    if library_dir:
        Path(library_dir).mkdir(parents=True, exist_ok=True)
        (Path(library_dir) / "sources.json").write_text(
            json.dumps(
                {
                    "visited_pages": visited_pages,
                    "copyrighted_not_downloaded": [
                        {
                            "title": c.title,
                            "page_url": c.page_url,
                            "media_url": c.media_url,
                            "kind": c.kind,
                        }
                        for c in candidates
                        if c.action == "blocked"
                    ][:30],
                    "note": "Copyrighted media is listed, not downloaded.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    downloaded = sum(1 for item in candidates if item.local_path)
    accepted = sum(1 for item in candidates if item.action == "accepted")
    rejected = sum(1 for item in candidates if item.action == "rejected")
    blocked = sum(1 for item in candidates if item.action == "blocked")
    skipped = sum(1 for item in candidates if item.action == "skipped")
    candidates.sort(key=lambda item: (item.action != "accepted", -(item.verdict.score if item.verdict else item.rank)))
    warnings: list[str] = []
    if "google" in request.engines and not any(hit.engine == "google" for hit in hits):
        warnings.append(
            "Google blocked headless Playwright (captcha / unusual traffic). "
            "Bing/DuckDuckGo are used to open top result pages instead."
        )
    if "bing" in request.engines and not any(hit.engine == "bing" for hit in hits):
        warnings.append("Bing Playwright returned no extractable result URLs.")
    warnings.append(
        "Copyrighted news/social/stock media is not downloaded. "
        "Those URLs are listed in sources.json for licensing, not saved as files."
    )
    return SearchJobResult(
        topics=topics,
        queries=queries,
        hits=hits,
        candidates=candidates,
        downloaded=downloaded,
        accepted=accepted,
        rejected=rejected,
        blocked=blocked,
        skipped=skipped,
        used_playwright=used_playwright,
        library_dir=library_dir,
        visited_pages=visited_pages,
        warnings=warnings,
    )


async def inspect_url(url: str, download: bool = False) -> SearchJobResult:
    job_id = uuid.uuid4().hex[:12]
    dest = _job_dir(job_id)
    candidates: list[BrollCandidate] = []
    async with launch_browser() as session:
        page = await session.new_page()
        try:
            detected = await inspect_page(page, url)
            for media in _dedupe_media(detected):
                rights = classify(media)
                action = "blocked" if rights.status == "copyrighted" else "skipped"
                candidate = BrollCandidate(
                    title=media.title or media.page_url,
                    page_url=media.page_url,
                    media_url=media.url,
                    kind=media.kind,
                    via=media.via,
                    mime=media.mime,
                    rights=rights,
                    action=action,
                    rank=rank(media, rights, url),
                )
                if download and rights.status == "reusable" and media.kind != "stream":
                    local = await download_http(media.url, dest / "incoming")
                    if local is not None:
                        candidate.action = "downloaded"
                        candidate.local_path = str(local)
                        clip = prepare_clip(local, dest / "clips")
                        if clip is not None:
                            candidate.ffmpeg_path = str(clip)
                candidates.append(candidate)
        finally:
            await page.close()
    library_dir = None
    if download:
        await AssetReviewAgent().review(candidates, url, [url], [])
        library_dir = str(place_in_library(job_id, candidates, dest / "rejected"))
    return SearchJobResult(
        queries=[url],
        hits=[SearchHit(engine="direct", query=url, title=url, url=url, layer="playwright")],
        candidates=candidates,
        downloaded=sum(1 for item in candidates if item.local_path),
        accepted=sum(1 for item in candidates if item.action == "accepted"),
        rejected=sum(1 for item in candidates if item.action == "rejected"),
        blocked=sum(1 for item in candidates if item.action == "blocked"),
        skipped=sum(1 for item in candidates if item.action == "skipped"),
        used_playwright=True,
        library_dir=library_dir,
    )

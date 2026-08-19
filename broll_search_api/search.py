from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
import base64

import httpx
from playwright.async_api import Page

from broll_search_api.browser import BrowserSession, dismiss_consent, goto
from broll_search_api.config import settings
from broll_search_api.models import DetectedMedia, SearchHit
from broll_search_api.rights import hostname, license_from_text


logger = logging.getLogger(__name__)

TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
}

API_ENGINES = {"wikimedia", "duckduckgo"}
PLAYWRIGHT_ENGINES = {"google", "bing", "bing_news", "duckduckgo"}
SKIP_HOSTS = {
    "duckduckgo.com",
    "bing.com",
    "microsoft.com",
    "google.com",
    "google.co.in",
    "accounts.google.com",
    "consent.google.com",
}


def google_search_url(query: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(query)}&hl=en&pws=0"


def bing_search_url(query: str) -> str:
    return f"https://www.bing.com/search?q={quote_plus(query)}"


def bing_news_url(query: str) -> str:
    return f"https://www.bing.com/news/search?q={quote_plus(query)}"


def duckduckgo_search_url(query: str) -> str:
    return f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"


def unwrap_search_redirect(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if "bing.com" in host and "/ck/" in (parsed.path or ""):
        token = parse_qs(parsed.query).get("u", [""])[0]
        if token.startswith("a1"):
            token = token[2:]
        token = token.replace("-", "+").replace("_", "/")
        pad = "=" * ((4 - len(token) % 4) % 4)
        try:
            decoded = base64.b64decode(token + pad).decode("utf-8", errors="ignore")
            if decoded.startswith("http"):
                return decoded
        except Exception:
            return url
    if "duckduckgo.com" in host and "uddg" in (parsed.query or ""):
        return unquote(parse_qs(parsed.query).get("uddg", [""])[0])
    return url


def normalize_url(url: str) -> str:
    url = unwrap_search_redirect((url or "").strip())
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ""
    query = "&".join(
        part
        for part in (parsed.query or "").split("&")
        if part and part.split("=", 1)[0].lower() not in TRACKING_PARAMS
    )
    return parsed._replace(query=query, fragment="").geturl()


async def _eval_links(page: Page, selectors: list[str]) -> list[dict[str, str]]:
    script = """
    (sels) => {
      const out = [];
      const seen = new Set();
      for (const sel of sels) {
        for (const el of document.querySelectorAll(sel)) {
          const href = el.href || "";
          const title = (el.innerText || el.textContent || "").trim();
          if (!href || seen.has(href)) continue;
          seen.add(href);
          const card = el.closest("li, article, .result, .news-card, .b_algo, div.g") || el.parentElement;
          const snippet = card ? (card.innerText || "").trim().slice(0, 280) : "";
          out.push({href, title, snippet});
        }
      }
      return out;
    }
    """
    try:
        return await page.evaluate(script, selectors)
    except Exception:
        return []


def _to_hits(
    engine: str,
    query: str,
    rows: list[dict[str, str]],
    limit: int,
    layer: str,
    search_url: str = "",
) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for row in rows:
        url = normalize_url(row.get("href") or "")
        host = hostname(url)
        if not url or host in SKIP_HOSTS or any(host.endswith("." + h) for h in SKIP_HOSTS):
            continue
        title = (row.get("title") or "").strip() or host
        snippet = (row.get("snippet") or "").strip()
        recency = ""
        match = re.search(r"\b(\d+\s*(?:m|h|d|w|hour|hours|day|days|min)s? ago)\b", snippet, re.I)
        if match:
            recency = match.group(1)
        hits.append(
            SearchHit(
                engine=engine,
                query=query,
                title=title[:200],
                url=url,
                snippet=snippet[:280],
                recency_hint=recency,
                layer=layer,  # type: ignore[arg-type]
                search_url=search_url,
            )
        )
        if len(hits) >= limit:
            break
    return hits


def _http_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }


async def search_duckduckgo_api(query: str, limit: int) -> list[SearchHit]:
    search_url = duckduckgo_search_url(query)
    try:
        async with httpx.AsyncClient(timeout=20.0, headers=_http_headers(), follow_redirects=True) as client:
            response = await client.get(search_url)
            response.raise_for_status()
            html = response.text
    except Exception:
        logger.exception("DuckDuckGo API search failed for %s", query)
        return []

    rows: list[dict[str, str]] = []
    for match in re.finditer(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        re.I | re.S,
    ):
        title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
        rows.append({"href": match.group(1), "title": title, "snippet": ""})
    return _to_hits("duckduckgo", query, rows, limit, "api", search_url)


async def search_duckduckgo_playwright(session: BrowserSession, query: str, limit: int) -> list[SearchHit]:
    search_url = duckduckgo_search_url(query)
    page = await session.new_page()
    try:
        await goto(page, search_url)
        await dismiss_consent(page)
        rows = await _eval_links(page, ["a.result__a", ".result__title a", "a.result-link"])
        return _to_hits("duckduckgo", query, rows, limit, "playwright", search_url)
    except Exception:
        logger.exception("DuckDuckGo Playwright search failed for %s", query)
        return []
    finally:
        await page.close()


async def search_google_playwright(session: BrowserSession, query: str, limit: int) -> list[SearchHit]:
    search_url = google_search_url(query)
    page = await session.new_page()
    try:
        await goto(page, search_url)
        await dismiss_consent(page)
        await page.wait_for_load_state("domcontentloaded")
        if "/sorry/" in page.url:
            logger.warning("Google blocked automated search for %s", query)
            return []
        rows = await _eval_links(page, ["div#search a:has(h3)", "a h3", "div.g a"])
        if not rows:
            # Type-into-box fallback if the results URL was blocked/redirected.
            await goto(page, "https://www.google.com")
            await dismiss_consent(page)
            box = page.locator("textarea[name='q'], input[name='q']").first
            await box.fill(query)
            await page.keyboard.press("Enter")
            await page.wait_for_load_state("domcontentloaded")
            rows = await _eval_links(page, ["div#search a:has(h3)", "a h3"])
        return _to_hits("google", query, rows, limit, "playwright", page.url or search_url)
    except Exception:
        logger.exception("Google Playwright search failed for %s", query)
        return []
    finally:
        await page.close()


async def search_bing_playwright(session: BrowserSession, query: str, limit: int, news: bool = False) -> list[SearchHit]:
    search_url = bing_news_url(query) if news else bing_search_url(query)
    engine = "bing_news" if news else "bing"
    page = await session.new_page()
    try:
        await goto(page, search_url)
        await dismiss_consent(page)
        await page.wait_for_load_state("domcontentloaded")
        selectors = (
            ["a.title", "a[class*='title']", ".news-card a", "div.t_s a", "li.b_algo a"]
            if news
            else ["li.b_algo h2 a", "li.b_algo a", "h2 a"]
        )
        rows = await _eval_links(page, selectors)
        return _to_hits(engine, query, rows, limit, "playwright", page.url or search_url)
    except Exception:
        logger.exception("Bing Playwright search failed for %s", query)
        return []
    finally:
        await page.close()


def _strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value or "").strip()


async def search_wikimedia(query: str, limit: int) -> list[tuple[SearchHit, DetectedMedia]]:
    params = {
        "action": "query",
        "format": "json",
        "origin": "*",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": "1920",
    }
    found: list[tuple[SearchHit, DetectedMedia]] = []
    pages: Any = []
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=20.0, headers=_http_headers(), follow_redirects=True) as client:
                response = await client.get("https://commons.wikimedia.org/w/api.php", params=params)
                if response.status_code == 429:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                pages = ((response.json().get("query") or {}).get("pages") or {}).values()
                last_error = None
                break
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(0.8)
    else:
        pages = []
    if last_error is not None and not pages:
        logger.warning("Wikimedia search failed for %s: %s", query, last_error)
        return []

    for page in pages:
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        media_url = info.get("url") or info.get("thumburl") or ""
        mime = (info.get("mime") or "").lower()
        if not media_url:
            continue
        size = int(info.get("size") or 0)
        if mime.startswith("audio/") or mime in {"application/ogg", "application/x-ogg"}:
            continue
        if size and size > settings.max_asset_bytes:
            continue
        width = int(info.get("thumbwidth") or info.get("width") or 0)
        height = int(info.get("thumbheight") or info.get("height") or 0)
        if size and size < 4000:
            continue
        meta: dict[str, Any] = info.get("extmetadata") or {}
        license_name = _strip_html((meta.get("LicenseShortName") or {}).get("value") or "")
        usage = _strip_html((meta.get("UsageTerms") or {}).get("value") or "")
        title = _strip_html(page.get("title") or "")
        page_url = "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_")
        kind = "video" if mime.startswith("video/") or mime in {"application/ogg", "application/x-ogm"} else "image"
        license_hint = " ".join(part for part in (license_name, usage) if part)
        status, _ = license_from_text(license_hint or "wikimedia")
        if status == "copyrighted":
            continue
        hit = SearchHit(
            engine="wikimedia",
            query=query,
            title=title.replace("File:", ""),
            url=page_url,
            snippet=license_hint,
            layer="api",
            search_url="https://commons.wikimedia.org/w/api.php",
        )
        media = DetectedMedia(
            kind=kind,
            url=normalize_url(media_url),
            page_url=page_url,
            via="api",
            mime=mime or None,
            title=title.replace("File:", ""),
            width=width or None,
            height=height or None,
            page_license_hint=license_hint or "Wikimedia Commons",
        )
        found.append((hit, media))
        if len(found) >= limit:
            break
    return found


def _extend(hits: list[SearchHit], seen: set[str], incoming: list[SearchHit]) -> None:
    for hit in incoming:
        if hit.url in seen:
            continue
        seen.add(hit.url)
        hits.append(hit)


async def run_api_searches(
    queries: list[str],
    engines: list[str],
    per_query: int,
) -> tuple[list[SearchHit], list[DetectedMedia], dict[str, int]]:
    hits: list[SearchHit] = []
    wiki_media: list[DetectedMedia] = []
    seen: set[str] = set()
    counts: dict[str, int] = {q: 0 for q in queries}

    for query in queries:
        before = len(hits)
        if "duckduckgo" in engines:
            _extend(hits, seen, await search_duckduckgo_api(query, per_query))
        if "wikimedia" in engines:
            for hit, media in await search_wikimedia(query, per_query):
                if media.url in seen:
                    continue
                if hit.url not in seen:
                    seen.add(hit.url)
                    hits.append(hit)
                seen.add(media.url)
                wiki_media.append(media)
            await asyncio.sleep(0.35)
        counts[query] = len(hits) - before
    return hits, wiki_media, counts


async def run_playwright_searches(
    session: BrowserSession,
    queries: list[str],
    engines: list[str],
    per_query: int,
    api_counts: dict[str, int],
) -> list[SearchHit]:
    hits: list[SearchHit] = []
    seen: set[str] = set()
    for query in queries:
        api_ok = api_counts.get(query, 0) >= min(2, per_query)
        if "duckduckgo" in engines and not api_ok:
            _extend(hits, seen, await search_duckduckgo_playwright(session, query, per_query))
        if "bing" in engines:
            _extend(hits, seen, await search_bing_playwright(session, query, per_query, news=False))
        if "bing_news" in engines:
            _extend(hits, seen, await search_bing_playwright(session, query, per_query, news=True))
        if "google" in engines:
            _extend(hits, seen, await search_google_playwright(session, query, per_query))
    return hits


def needs_playwright_search(engines: list[str], api_counts: dict[str, int], per_query: int) -> bool:
    if any(engine in PLAYWRIGHT_ENGINES and engine not in API_ENGINES for engine in engines):
        return True
    if "duckduckgo" in engines and any(count < min(2, per_query) for count in api_counts.values()):
        return True
    return False

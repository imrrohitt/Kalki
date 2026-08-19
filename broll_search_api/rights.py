from __future__ import annotations

from urllib.parse import urlparse

from broll_search_api.models import DetectedMedia, LicenseStatus, RightsDecision


REUSABLE_DOMAIN_SUFFIXES = (
    "wikimedia.org",
    "wikipedia.org",
    "wikimediafoundation.org",
    "archive.org",
    "openverse.org",
    "creativecommons.org",
    "pexels.com",
    "pixabay.com",
    "unsplash.com",
    "nasa.gov",
    "noaa.gov",
    "usgs.gov",
    "nih.gov",
    "cdc.gov",
    "loc.gov",
    "si.edu",
    "metmuseum.org",
    "europeana.eu",
    "nappy.co",
    "coverr.co",
    "mixkit.co",
    "videvo.net",
)

BLOCKED_MEDIA_HOSTS = (
    "youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
    "googlevideo.com",
    "vimeo.com",
    "dailymotion.com",
    "tiktok.com",
    "instagram.com",
    "cdninstagram.com",
    "facebook.com",
    "fbcdn.net",
    "twitter.com",
    "x.com",
    "twimg.com",
    "cnn.com",
    "bbc.co.uk",
    "bbc.com",
    "bbci.co.uk",
    "reuters.com",
    "apnews.com",
    "ap.org",
    "nytimes.com",
    "wsj.com",
    "bloomberg.com",
    "ft.com",
    "theguardian.com",
    "washingtonpost.com",
    "latimes.com",
    "forbes.com",
    "techcrunch.com",
    "theverge.com",
    "wired.com",
    "arstechnica.com",
    "ndtv.com",
    "indiatimes.com",
    "indianexpress.com",
    "hindustantimes.com",
    "news18.com",
    "gettyimages.com",
    "shutterstock.com",
    "alamy.com",
    "adobe.com",
)

REUSABLE_LICENSE_MARKERS = (
    "cc0",
    "cc-0",
    "cc by",
    "cc-by",
    "cc by-sa",
    "cc-by-sa",
    "cc by-nc",
    "creative commons",
    "public domain",
    "pdm",
    "wikimedia",
    "gnu fdl",
    "gfdl",
    "pexels license",
    "pixabay license",
    "unsplash license",
)

REJECT_LICENSE_MARKERS = (
    "all rights reserved",
    "getty",
    "shutterstock",
    "associated press",
    "reuters",
    "afp",
    "strictly for editorial",
    "not for commercial",
    "drm",
)


def hostname(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _matches_suffix(host: str, suffixes: tuple[str, ...]) -> bool:
    return any(host == s or host.endswith("." + s) for s in suffixes)


def _norm_text(value: str) -> str:
    return " ".join((value or "").lower().replace("_", " ").split())


def license_from_text(text: str) -> tuple[LicenseStatus | None, str]:
    blob = _norm_text(text)
    if not blob:
        return None, ""
    if any(marker in blob for marker in REJECT_LICENSE_MARKERS):
        return "copyrighted", text.strip()[:120]
    if any(marker in blob for marker in REUSABLE_LICENSE_MARKERS):
        return "reusable", text.strip()[:120]
    return None, text.strip()[:120]


def classify(media: DetectedMedia) -> RightsDecision:
    media_host = hostname(media.url)
    page_host = hostname(media.page_url)
    hint_status, hint_label = license_from_text(media.page_license_hint)

    if media.kind in {"embed", "stream"} and _matches_suffix(media_host, BLOCKED_MEDIA_HOSTS):
        return RightsDecision(
            status="copyrighted",
            label=hint_label or "embedded/streamed third-party media",
            reason="Blocked host for streams/embeds. A media URL is not a reuse license.",
            source_domain=media_host or page_host,
        )

    if _matches_suffix(media_host, BLOCKED_MEDIA_HOSTS) or _matches_suffix(
        page_host, BLOCKED_MEDIA_HOSTS
    ):
        return RightsDecision(
            status="copyrighted",
            label=hint_label or "news/stock/social host",
            reason="Host is treated as copyrighted news, social, or stock media.",
            source_domain=media_host or page_host,
        )

    if hint_status == "copyrighted":
        return RightsDecision(
            status="copyrighted",
            label=hint_label,
            reason="Page license text is not reusable.",
            source_domain=media_host or page_host,
        )

    if _matches_suffix(media_host, REUSABLE_DOMAIN_SUFFIXES) or _matches_suffix(
        page_host, REUSABLE_DOMAIN_SUFFIXES
    ):
        return RightsDecision(
            status="reusable",
            label=hint_label or "allowlisted reusable-media host",
            reason="Source is on the reusable-media allowlist.",
            source_domain=media_host or page_host,
        )

    if hint_status == "reusable":
        return RightsDecision(
            status="reusable",
            label=hint_label,
            reason="Page advertised a Creative Commons / public-domain license.",
            source_domain=media_host or page_host,
        )

    return RightsDecision(
        status="unknown",
        label=hint_label or "no license found",
        reason="No reusable license detected. Unknown media is not downloaded.",
        source_domain=media_host or page_host,
    )


def rank(media: DetectedMedia, rights: RightsDecision, query: str) -> float:
    if rights.status != "reusable":
        return 0.0
    score = 40.0
    if media.kind == "video":
        score += 25.0
    elif media.kind == "image":
        score += 12.0
    elif media.kind == "download":
        score += 18.0
    if media.via in {"api", "download_link"}:
        score += 8.0
    blob = f"{media.title} {media.url} {media.page_url}".lower()
    for token in query.lower().split():
        if len(token) > 2 and token in blob:
            score += 3.0
    return round(score, 2)

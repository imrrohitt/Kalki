from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any

from broll_search_api.config import settings
from broll_search_api.models import TopicQuery


logger = logging.getLogger(__name__)

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "have", "has",
    "was", "were", "are", "but", "not", "you", "your", "our", "their",
    "just", "about", "into", "over", "after", "before", "today", "aaj",
    "hai", "hain", "ka", "ki", "ke", "ne", "se", "mein", "me", "ko",
}


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _fallback_topics(transcript: str, max_queries: int) -> list[TopicQuery]:
    year = datetime.now().year
    words = re.findall(r"[A-Za-z][A-Za-z0-9.+-]{2,}", transcript)
    keep: list[str] = []
    seen: set[str] = set()
    for word in words:
        key = word.lower()
        if key in STOPWORDS or key in seen:
            continue
        seen.add(key)
        keep.append(word)
        if len(keep) >= 8:
            break
    core = " ".join(keep[:6]) or transcript.strip()[:80]
    event = core[:120] or "current event"
    queries = [
        f"{core} {year}",
        f"{core} news",
        f"{core} Wikimedia Commons",
        f"{core} official",
    ]
    return [TopicQuery(event=event, queries=queries[: max(2, max_queries)], media_need="relevant product, logo, or event footage")]


def _parse_topics(payload: dict[str, Any], max_queries: int) -> list[TopicQuery]:
    raw = payload.get("topics") or []
    topics: list[TopicQuery] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        queries = [str(q).strip() for q in (item.get("queries") or []) if str(q).strip()]
        event = str(item.get("event") or "").strip()
        if not event and not queries:
            continue
        topics.append(
            TopicQuery(
                event=event or (queries[0] if queries else "topic"),
                queries=queries[:max_queries],
                media_need=str(item.get("media_need") or "").strip(),
            )
        )
        if len(topics) >= max_queries:
            break
    return topics


async def generate_topics(transcript: str, max_queries: int = 4) -> list[TopicQuery]:
    transcript = " ".join(transcript.split())
    if not transcript:
        return []

    api_key = settings.llm_api_key.replace("Bearer ", "").strip()
    if not api_key:
        return _fallback_topics(transcript, max_queries)

    year = datetime.now().year
    prompt = f"""You generate B-roll search queries from a spoken video transcript.

Return ONLY JSON:
{{
  "topics": [
    {{
      "event": "short description of the news or visual moment",
      "queries": ["search query 1", "search query 2"],
      "media_need": "what should appear on screen"
    }}
  ]
}}

Rules:
- Prefer searchable proper nouns: products, companies, places, events.
- Include one recency query with the year {year}.
- Include one reusable-media query, e.g. adding "Wikimedia Commons" or "public domain".
- Max {max_queries} topics, 2 queries each.
- No commentary.

Transcript:
{transcript[:4000]}
"""

    try:
        import litellm

        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
            os.environ.setdefault("DEEPSEEK_API_KEY", api_key)
        model = settings.llm_model
        if "deepseek" in model.lower() and model.lower().startswith("openai/"):
            model = "deepseek/" + model.split("/", 1)[1]
        complete_kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "api_base": (settings.llm_base_url or "").rstrip("/") or None,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": 45,
            "drop_params": True,
        }
        if "deepseek" in model.lower():
            complete_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        response = litellm.completion(**complete_kwargs)
        text = response.choices[0].message.content or ""
        topics = _parse_topics(_extract_json(text), max_queries)
        if topics:
            return topics
    except Exception:
        logger.exception("LLM query generation failed; using keyword fallback")

    return _fallback_topics(transcript, max_queries)


def flatten_queries(topics: list[TopicQuery], max_queries: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for topic in topics:
        for query in topic.queries:
            key = query.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(query.strip())
            if len(out) >= max_queries * 2:
                return out
    return out

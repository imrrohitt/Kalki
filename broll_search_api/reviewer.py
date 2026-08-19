from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from broll_search_api.config import settings
from broll_search_api.models import AssetVerdict, BrollCandidate, TopicQuery
from broll_search_api.probe import FileProbe, probe_file
from broll_search_api.query_gen import _extract_json


logger = logging.getLogger(__name__)


def _lite_complete(prompt: str) -> str:
    api_key = settings.llm_api_key.replace("Bearer ", "").strip()
    if not api_key:
        return ""
    import litellm

    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("DEEPSEEK_API_KEY", api_key)
    model = settings.llm_model
    if "deepseek" in model.lower() and model.lower().startswith("openai/"):
        model = "deepseek/" + model.split("/", 1)[1]
    kwargs: dict = {
        "model": model,
        "api_key": api_key,
        "api_base": (settings.llm_base_url or "").rstrip("/") or None,
        "messages": [{"role": "user", "content": prompt}],
        "timeout": 45,
        "drop_params": True,
    }
    if "deepseek" in model.lower():
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    response = litellm.completion(**kwargs)
    return response.choices[0].message.content or ""


class PagePickAgent:
    """Picks the best search-result pages to open for B-roll."""

    async def pick(
        self,
        hits: list,
        transcript: str,
        queries: list[str],
        limit: int = 5,
    ) -> list:
        unique: list = []
        seen: set[str] = set()
        for hit in hits:
            host = (hit.url or "").split("/")[2] if "://" in hit.url else hit.url
            if not hit.url or host in seen:
                continue
            seen.add(host)
            unique.append(hit)
            if len(unique) >= 12:
                break
        if len(unique) <= limit:
            return unique[:limit]
        payload = [
            {"id": i, "title": h.title, "url": h.url, "snippet": h.snippet, "engine": h.engine}
            for i, h in enumerate(unique)
        ]
        prompt = f"""You pick the {limit} best websites to open for B-roll research.

Topic:
{transcript[:1500] or ", ".join(queries)}

Results JSON:
{json.dumps(payload, ensure_ascii=False)}

Return ONLY JSON:
{{"ids": [0, 2, 4]}}

Prefer official product pages, explainers, and pages likely to have logos/screenshots/diagrams.
Skip homepages with no article, login walls, and unrelated results.
Exactly {limit} ids if possible.
"""
        try:
            text = _lite_complete(prompt)
            ids = (_extract_json(text) or {}).get("ids") or []
            picked = [unique[int(i)] for i in ids if str(i).isdigit() and 0 <= int(i) < len(unique)]
            if picked:
                return picked[:limit]
        except Exception:
            logger.exception("Page pick LLM failed")
        return unique[:limit]


def _heuristic_verdict(
    candidate: BrollCandidate,
    probe: FileProbe,
    transcript: str,
    queries: list[str],
    topics: list[TopicQuery],
) -> AssetVerdict:
    if not probe.ok:
        return AssetVerdict(keep=False, score=0.0, reason=probe.reason or "failed technical probe", checks=["probe_fail"])
    blob = " ".join(
        [
            candidate.title,
            candidate.media_url,
            Path(candidate.local_path or "").name,
            candidate.rights.label,
        ]
    ).lower()
    needles = [t.lower() for t in queries]
    for topic in topics:
        needles.extend(topic.event.lower().split())
        if topic.media_need:
            needles.append(topic.media_need.lower())
    for token in transcript.replace(",", " ").split():
        if token[:1].isupper() and len(token) > 2:
            needles.append(token.lower())
    hits = {n for n in needles if len(n) > 2 and n in blob}
    score = min(1.0, 0.2 + 0.2 * len(hits))
    keep = candidate.rights.status == "reusable" and len(hits) >= 2
    reason = "title/filename matches topic" if keep else "weak or no topic overlap"
    return AssetVerdict(keep=keep, score=round(score, 2), reason=reason, checks=["heuristic"])


class AssetReviewAgent:
    """Decides whether a downloaded file is usable B-roll for this transcript."""

    async def review(
        self,
        candidates: list[BrollCandidate],
        transcript: str,
        queries: list[str],
        topics: list[TopicQuery],
    ) -> list[BrollCandidate]:
        pending: list[tuple[int, BrollCandidate, FileProbe]] = []
        for idx, candidate in enumerate(candidates):
            if candidate.action != "downloaded" or not candidate.local_path:
                continue
            probe = probe_file(Path(candidate.local_path))
            if not probe.ok:
                candidate.action = "rejected"
                candidate.verdict = AssetVerdict(
                    keep=False,
                    score=0.0,
                    reason=probe.reason or "invalid media file",
                    checks=["probe_fail"],
                )
                continue
            pending.append((idx, candidate, probe))

        if not pending:
            return candidates

        llm_map = await self._ask_llm(pending, transcript, queries, topics)
        for idx, candidate, probe in pending:
            verdict = llm_map.get(idx)
            if verdict is None:
                verdict = _heuristic_verdict(candidate, probe, transcript, queries, topics)
            candidate.verdict = verdict
            candidate.action = "accepted" if verdict.keep else "rejected"
        return candidates

    async def _ask_llm(
        self,
        pending: list[tuple[int, BrollCandidate, FileProbe]],
        transcript: str,
        queries: list[str],
        topics: list[TopicQuery],
    ) -> dict[int, AssetVerdict]:
        needs = "; ".join(f"{t.event}: {t.media_need}" for t in topics if t.event) or "relevant B-roll"
        payload = []
        for idx, candidate, probe in pending:
            payload.append(
                {
                    "id": idx,
                    "kind": candidate.kind,
                    "title": candidate.title,
                    "filename": Path(candidate.local_path or "").name,
                    "mime": probe.mime,
                    "bytes": probe.bytes,
                    "width": probe.width,
                    "height": probe.height,
                    "duration": probe.duration,
                    "license": candidate.rights.label,
                    "page_url": candidate.page_url,
                }
            )
        prompt = f"""You are BrollAssetAgent. Keep only files that are correct B-roll for this spoken video.

Transcript:
{transcript[:2500] or "(queries only)"}

Search queries: {", ".join(queries)}
Visual needs: {needs}

Assets JSON:
{json.dumps(payload, ensure_ascii=False)}

Return ONLY JSON:
{{
  "verdicts": [
    {{"id": 0, "keep": true, "score": 0.86, "reason": "official OpenAI logo matches the topic"}}
  ]
}}

Keep if: the file would actually appear on screen for this sentence — NVIDIA GPU/chip, Apple Silicon/MacBook hardware, or the named product.
Reject if: wrong company, software tutorial, power-adapter/demo coincidence, generic unrelated photo, UI icon, search-page screenshot, broken file, watermarked news still.
When unsure, reject. Keep score must be >= 0.75.
Every id must appear once.
"""
        try:
            text = _lite_complete(prompt)
            if not text:
                return {}
            data = _extract_json(text)
        except Exception:
            logger.exception("Asset review LLM failed")
            return {}

        out: dict[int, AssetVerdict] = {}
        for item in data.get("verdicts") or []:
            if not isinstance(item, dict):
                continue
            try:
                asset_id = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            keep = bool(item.get("keep"))
            try:
                score = float(item.get("score") or (0.8 if keep else 0.2))
            except (TypeError, ValueError):
                score = 0.8 if keep else 0.2
            out[asset_id] = AssetVerdict(
                keep=keep,
                score=max(0.0, min(1.0, score)),
                reason=str(item.get("reason") or "").strip()[:240],
                checks=["llm_agent"],
            )
        return out


def place_in_library(
    job_id: str,
    candidates: list[BrollCandidate],
    rejected_dir: Path,
) -> Path:
    library = settings.library_path / job_id
    accepted_dir = library / "accepted"
    clips_dir = library / "clips"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        src = Path(candidate.local_path) if candidate.local_path else None
        if src is None or not src.exists():
            continue
        if candidate.action == "accepted":
            dest = accepted_dir / src.name
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
            candidate.local_path = str(dest)
            if candidate.ffmpeg_path:
                clip = Path(candidate.ffmpeg_path)
                if clip.exists():
                    clip_dest = clips_dir / clip.name
                    shutil.copy2(clip, clip_dest)
                    candidate.ffmpeg_path = str(clip_dest)
        elif candidate.action == "rejected":
            dest = rejected_dir / src.name
            try:
                shutil.move(str(src), dest)
                candidate.local_path = str(dest)
            except OSError:
                logger.warning("Could not move rejected asset %s", src)

    manifest = {
        "job_id": job_id,
        "accepted": [
            {
                "title": c.title,
                "kind": c.kind,
                "path": c.local_path,
                "clip": c.ffmpeg_path,
                "reason": c.verdict.reason if c.verdict else "",
                "score": c.verdict.score if c.verdict else 0,
            }
            for c in candidates
            if c.action == "accepted"
        ],
        "blocked_copyrighted": [
            {
                "title": c.title,
                "kind": c.kind,
                "page_url": c.page_url,
                "media_url": c.media_url,
                "reason": c.rights.reason,
            }
            for c in candidates
            if c.action == "blocked"
        ][:40],
    }
    (library / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return library

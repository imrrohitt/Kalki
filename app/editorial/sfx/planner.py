from __future__ import annotations

from app.editorial.models import EditorialAnalysis, GraphicBeat, SfxHit, SfxKind

_KIND_RANK: dict[SfxKind, int] = {"whoosh": 1, "swoosh": 2, "hit": 3, "impact": 4}
_MAX_HITS = 14
_MIN_GAP = 1.65


def _clip_time(at: float, duration: float) -> float | None:
    if duration <= 0.2:
        return round(max(0.04, at), 3)
    if at >= duration - 0.12:
        return None
    return round(max(0.04, at), 3)


def _dedupe(hits: list[SfxHit], duration: float) -> list[SfxHit]:
    ordered = sorted(hits, key=lambda h: (h.at, -_KIND_RANK.get(h.kind, 0)))
    kept: list[SfxHit] = []
    for hit in ordered:
        at = _clip_time(hit.at, duration)
        if at is None:
            continue
        hit = hit.model_copy(update={"at": at})
        if kept and at - kept[-1].at < _MIN_GAP:
            if _KIND_RANK.get(hit.kind, 0) > _KIND_RANK.get(kept[-1].kind, 0):
                kept[-1] = hit
            continue
        kept.append(hit)
        if len(kept) >= _MAX_HITS:
            break
    return kept


def plan_sfx(
    analysis: EditorialAnalysis,
    graphics: list[GraphicBeat],
    *,
    video_duration: float,
) -> list[SfxHit]:
    hits: list[SfxHit] = []
    hook = next((s for s in analysis.sentences if s.editorial_role == "hook"), None)
    if hook:
        hits.append(SfxHit(at=hook.start, kind="impact", gain=0.32, reason="hook"))
    elif graphics:
        hits.append(SfxHit(at=graphics[0].start, kind="impact", gain=0.32, reason="open"))

    for graphic in graphics:
        start = graphic.start
        kicker = (graphic.kicker or "").upper()
        if graphic.kind == "vs_split":
            hits.append(SfxHit(at=start + 0.06, kind="swoosh", gain=0.24, reason="compare"))
        elif graphic.kind == "stat":
            hits.append(SfxHit(at=start + 0.08, kind="hit", gain=0.26, reason="number"))
        elif graphic.kind in {"process", "diagram", "chip_row"}:
            hits.append(SfxHit(at=start + 0.05, kind="whoosh", gain=0.22, reason="steps"))
        elif kicker in {"HOOK", "REVEAL"}:
            hits.append(SfxHit(at=start, kind="impact", gain=0.30, reason=kicker.lower()))

    for sentence in analysis.sentences:
        role = sentence.editorial_role
        if role in {"reveal", "answer"}:
            hits.append(SfxHit(at=sentence.start, kind="hit", gain=0.26, reason=role))
        elif role == "contrast":
            hits.append(SfxHit(at=sentence.start, kind="swoosh", gain=0.22, reason="contrast"))
        elif role == "important_number":
            hits.append(SfxHit(at=sentence.start, kind="hit", gain=0.26, reason="number"))
        elif role == "cta":
            hits.append(SfxHit(at=sentence.start, kind="whoosh", gain=0.22, reason="cta"))

    return _dedupe(hits, video_duration)

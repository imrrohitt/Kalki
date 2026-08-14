from __future__ import annotations

from app.config import settings
from app.editorial.framing import default_visual, scale_from_intensity
from app.editorial.models import (
    EditorialAnalysis,
    VisualContext,
    ZoomDecision,
    ZoomScore,
    ZoomStyle,
)
from app.editorial.zoom.mapping import HIGH_PRIORITY_ROLES, SKIP_ROLES


def _style_for(ease_in: float, hold: float, intensity: float) -> ZoomStyle:
    if hold >= 0.75 and intensity >= 0.6:
        return "hold"
    if ease_in >= 0.55:
        return "slow_punch"
    return "fast_punch"


def _fit_span(
    ease_in: float,
    hold: float,
    ease_out: float,
    max_span: float,
    min_in: float,
    min_out: float,
) -> tuple[float, float, float] | None:
    ease_in = max(min_in, ease_in)
    ease_out = max(min_out, ease_out)
    hold = max(0.22, hold)
    total = ease_in + hold + ease_out
    if total <= max_span:
        return ease_in, hold, ease_out
    overflow = total - max_span
    hold = max(0.22, hold - overflow)
    total = ease_in + hold + ease_out
    if total <= max_span:
        return ease_in, hold, ease_out
    overflow = total - max_span
    ease_out = max(min_out, ease_out - overflow)
    total = ease_in + hold + ease_out
    if total <= max_span:
        return ease_in, hold, ease_out
    overflow = total - max_span
    ease_in = max(min_in, ease_in - overflow)
    if ease_in + hold + ease_out > max_span + 0.001:
        return None
    return ease_in, hold, ease_out


class ZoomPlanner:
    def plan(
        self,
        analysis: EditorialAnalysis,
        scores: list[ZoomScore],
        *,
        video_duration: float,
        visual: VisualContext | None = None,
    ) -> list[ZoomDecision]:
        visual = visual or default_visual()
        max_safe = visual.max_safe_scale
        min_gap = settings.zoom_min_gap_sec
        min_in = settings.zoom_min_ease_in_sec
        min_out = settings.zoom_min_ease_out_sec
        decisions: list[ZoomDecision] = []
        last_end = -10.0

        score_by_id = {s.sentence_id: s for s in scores}
        ordered = sorted(analysis.sentences, key=lambda s: s.start)

        if max_safe <= 1.04:
            for score in scores:
                score.action = "skip"
                score.reason = "subject bbox too tight to zoom"
            return []

        for sentence in ordered:
            score = score_by_id.get(sentence.sentence_id)
            motion = sentence.zoom
            if score is None or not motion.apply:
                if score is not None and not motion.apply:
                    score.action = "skip"
                    score.reason = "motion.apply=false"
                continue
            if sentence.editorial_role in SKIP_ROLES:
                score.action = "skip"
                score.reason = f"role={sentence.editorial_role}"
                continue
            tightness = max(visual.bbox.w, visual.bbox.h, visual.face_scale) * visual.current_zoom
            if tightness >= 0.88:
                score.action = "skip"
                score.reason = "framing too tight"
                continue

            scale = min(
                scale_from_intensity(motion.intensity, max_safe),
                settings.zoom_max_scale,
            )
            if scale <= 1.001:
                score.action = "skip"
                score.reason = "intensity maps to no zoom"
                continue

            delay = max(0.0, motion.delay_ms / 1000.0)
            start = max(0.0, sentence.start + delay)
            remaining = video_duration - start
            if remaining < min_in + min_out + 0.22:
                score.action = "skip"
                score.reason = "not enough room to ease"
                continue

            available = min(
                settings.zoom_max_duration_sec,
                remaining,
                max(min_in + min_out + 0.22, sentence.end - start + 0.45),
            )
            fitted = _fit_span(
                motion.ease_in_ms / 1000.0,
                motion.hold_ms / 1000.0,
                motion.ease_out_ms / 1000.0,
                available,
                min_in,
                min_out,
            )
            if fitted is None:
                score.action = "skip"
                score.reason = "cannot ease within duration cap"
                continue
            ease_in, hold, ease_out = fitted
            span = ease_in + hold + ease_out
            peak_end = start + ease_in + hold
            span_end = start + span

            gap = start - last_end
            score.previous_zoom_distance = round(max(0.0, gap), 3)
            allow_close = (
                score.final_score >= 0.88
                and sentence.editorial_role in HIGH_PRIORITY_ROLES
            )
            if decisions and gap < min_gap and not allow_close:
                score.action = "skip"
                score.reason = f"too soon after previous zoom ({gap:.2f}s)"
                continue
            if decisions and start < last_end - 0.02:
                score.action = "skip"
                score.reason = "overlaps previous ease-out"
                continue

            decisions.append(
                ZoomDecision(
                    start=round(start, 3),
                    end=round(peak_end, 3),
                    intent=sentence.editorial_role,
                    style=_style_for(ease_in, hold, motion.intensity),
                    target_scale=round(scale, 3),
                    easing="ease_in_out",
                    ease_in=round(ease_in, 3),
                    hold=round(hold, 3),
                    ease_out=round(ease_out, 3),
                    release_duration=round(ease_out, 3),
                    anchor_x=round(visual.bbox.cx, 3),
                    anchor_y=round(visual.bbox.cy, 3),
                    confidence=sentence.confidence,
                    sentence_id=sentence.sentence_id,
                    score=score.final_score,
                )
            )
            last_end = span_end
            score.action = "apply"
            score.reason = "planned"

        return decisions

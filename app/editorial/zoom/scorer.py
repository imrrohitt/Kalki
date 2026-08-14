from __future__ import annotations

from app.config import settings
from app.editorial.models import (
    EditorialAnalysis,
    EditorialSentence,
    VisualContext,
    ZoomAction,
    ZoomScore,
)
from app.editorial.zoom.mapping import (
    HIGH_PRIORITY_ROLES,
    ROLE_SEMANTIC_WEIGHT,
    SKIP_ROLES,
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _semantic_score(sentence: EditorialSentence) -> float:
    role_w = ROLE_SEMANTIC_WEIGHT.get(sentence.editorial_role, 0.2)
    return _clamp(
        0.42 * role_w
        + 0.33 * sentence.visual_interest
        + 0.25 * sentence.signals.peak()
    )


def _prosody_score(sentence: EditorialSentence) -> float:
    prosody = sentence.prosody
    score = 0.35
    if prosody.pause_before is not None and prosody.pause_before >= 0.45:
        score += 0.22
    if prosody.pause_after is not None and prosody.pause_after >= 0.35:
        score += 0.18
    if prosody.speaking_rate is not None and prosody.speaking_rate <= 2.2:
        score += 0.12
    if prosody.pitch_change is not None:
        score += 0.2 * _clamp(prosody.pitch_change)
    if prosody.loudness is not None:
        score += 0.15 * _clamp(prosody.loudness)
    return _clamp(score)


def _visual_score(visual: VisualContext) -> float:
    tightness = max(visual.bbox.w, visual.bbox.h, visual.face_scale) * visual.current_zoom
    if visual.max_safe_scale <= 1.05:
        return 0.22
    if tightness >= 0.88:
        return 0.28
    if tightness >= 0.75:
        return 0.48
    return 0.78


class ZoomScorer:
    def score(
        self,
        analysis: EditorialAnalysis,
        *,
        visual: VisualContext | None = None,
    ) -> list[ZoomScore]:
        visual = visual or VisualContext()
        visual_base = _visual_score(visual)
        threshold = settings.zoom_score_threshold
        results: list[ZoomScore] = []

        for sentence in analysis.sentences:
            semantic = _semantic_score(sentence)
            prosody = _prosody_score(sentence)
            candidate = sentence.editorial_role not in SKIP_ROLES and sentence.zoom.apply
            blended = _clamp(0.55 * semantic + 0.25 * visual_base + 0.20 * prosody)

            action: ZoomAction = "apply"
            reason = "pass"
            if not sentence.zoom.apply:
                action = "skip"
                reason = "motion.apply=false"
            elif not candidate:
                action = "skip"
                reason = f"role={sentence.editorial_role}"
            elif visual_base < 0.35:
                action = "skip"
                reason = "framing too tight"
            elif blended < threshold and sentence.editorial_role not in HIGH_PRIORITY_ROLES:
                action = "skip"
                reason = f"score {blended:.2f} < {threshold:.2f}"
            elif blended < threshold * 0.85:
                action = "skip"
                reason = f"score {blended:.2f} below priority floor"

            results.append(
                ZoomScore(
                    sentence_id=sentence.sentence_id,
                    candidate=candidate and action == "apply",
                    intent=sentence.editorial_role,
                    semantic_score=round(semantic, 3),
                    visual_score=round(visual_base, 3),
                    prosody_score=round(prosody, 3),
                    previous_zoom_distance=0.0,
                    final_score=round(blended, 3),
                    action=action,
                    reason=reason,
                )
            )
        return results

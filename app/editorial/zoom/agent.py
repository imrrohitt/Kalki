from __future__ import annotations

from app.editorial.models import EditorialAnalysis, VisualContext, ZoomDecision
from app.editorial.zoom.planner import ZoomPlanner
from app.editorial.zoom.scorer import ZoomScorer


class ZoomDecisionEngine:
    """First consumer of Editorial Intelligence. Maps intent → zoom, never keywords → ffmpeg."""

    def __init__(
        self,
        scorer: ZoomScorer | None = None,
        planner: ZoomPlanner | None = None,
    ) -> None:
        self.scorer = scorer or ZoomScorer()
        self.planner = planner or ZoomPlanner()

    def decide(
        self,
        analysis: EditorialAnalysis,
        *,
        video_duration: float,
        visual: VisualContext | None = None,
    ) -> list[ZoomDecision]:
        scores = self.scorer.score(analysis, visual=visual)
        return self.planner.plan(
            analysis,
            scores,
            video_duration=video_duration,
            visual=visual,
        )

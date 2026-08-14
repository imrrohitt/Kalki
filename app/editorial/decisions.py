from __future__ import annotations

from app.captions.models import CaptionTimeline
from app.editorial.analyzer import EditorialAnalyzer
from app.editorial.broll.agent import BrollAgent
from app.editorial.cuts.agent import CutAgent
from app.editorial.models import EditorialAnalysis, VisualContext
from app.editorial.zoom.agent import ZoomDecisionEngine
from app.timeline.models import EditTimeline
from app.timeline.validator import validate_edit_timeline
from app.transcription.models import Transcript


class EditorialIntelligenceEngine:
    """Understand rhetorical context, then decide what should change visually."""

    def __init__(
        self,
        analyzer: EditorialAnalyzer | None = None,
        zoom_engine: ZoomDecisionEngine | None = None,
        broll_agent: BrollAgent | None = None,
        cut_agent: CutAgent | None = None,
    ) -> None:
        self.analyzer = analyzer or EditorialAnalyzer()
        self.zoom_engine = zoom_engine or ZoomDecisionEngine()
        self.broll_agent = broll_agent or BrollAgent()
        self.cut_agent = cut_agent or CutAgent()

    async def analyze(
        self,
        transcript: Transcript,
        video_duration: float,
        job_id: str,
        visual: VisualContext | None = None,
    ) -> EditorialAnalysis:
        return await self.analyzer.analyze(
            transcript=transcript,
            video_duration=video_duration,
            job_id=job_id,
            visual=visual,
        )

    async def plan(
        self,
        analysis: EditorialAnalysis,
        captions: CaptionTimeline,
        *,
        video_duration: float,
        visual: VisualContext | None = None,
    ) -> EditTimeline:
        zooms = self.zoom_engine.decide(
            analysis,
            video_duration=video_duration,
            visual=visual,
        )
        broll = await self.broll_agent.plan(analysis)
        cuts = await self.cut_agent.plan(analysis)
        timeline = EditTimeline(
            captions=list(captions.captions),
            zooms=zooms,
            broll=broll,
            cuts=cuts,
        )
        return validate_edit_timeline(timeline, video_duration)

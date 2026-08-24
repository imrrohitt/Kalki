from __future__ import annotations

import logging

from app.captions.models import CaptionTimeline
from app.config import settings
from app.editorial.analyzer import EditorialAnalyzer
from app.editorial.broll.agent import BrollAgent
from app.editorial.cuts.agent import CutAgent
from app.editorial.graphics.agent import GraphicsAgent
from app.editorial.models import EditorialAnalysis, VisualContext
from app.editorial.scenes.agent import SceneDirectorAgent
from app.editorial.sfx.agent import SfxAgent
from app.editorial.zoom.agent import ZoomDecisionEngine
from app.timeline.models import EditTimeline
from app.timeline.validator import validate_edit_timeline
from app.transcription.models import Transcript

logger = logging.getLogger(__name__)


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
        self.graphics_agent = GraphicsAgent()
        self.scene_director = SceneDirectorAgent()
        self.sfx_agent = SfxAgent()

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
        job_id: str = "edit",
    ) -> EditTimeline:
        jid = job_id[:8]
        if settings.split_layout_enabled:
            zooms = []
            logger.info("[%s] zooms: skipped (split layout keeps the full face)", jid)
        else:
            zooms = self.zoom_engine.decide(
                analysis,
                video_duration=video_duration,
                visual=visual,
            )
            logger.info("[%s] zooms: %s", jid, len(zooms))
        broll = await self.broll_agent.plan(analysis)
        cuts = await self.cut_agent.plan(analysis)
        graphics = await self.graphics_agent.plan(
            analysis,
            video_duration=video_duration,
            job_id=job_id,
        )
        logger.info("[%s] graphics: %s cards", jid, len(graphics))
        sfx = await self.sfx_agent.plan(
            analysis,
            graphics,
            video_duration=video_duration,
            job_id=job_id,
        )
        logger.info("[%s] sfx: %s hits", jid, len(sfx))
        timeline = EditTimeline(
            captions=list(captions.captions),
            zooms=zooms,
            broll=broll,
            cuts=cuts,
            graphics=graphics,
            sfx=sfx,
        )
        return validate_edit_timeline(timeline, video_duration)

    async def plan_reel(
        self,
        analysis: EditorialAnalysis,
        captions: CaptionTimeline,
        *,
        video_duration: float,
        job_id: str = "reel",
    ) -> EditTimeline:
        jid = job_id[:8]
        graphics = await self.scene_director.plan(
            analysis,
            video_duration=video_duration,
            job_id=job_id,
        )
        logger.info("[%s] scenes: %s cards", jid, len(graphics))
        sfx = await self.sfx_agent.plan(
            analysis,
            graphics,
            video_duration=video_duration,
            job_id=job_id,
        )
        logger.info("[%s] sfx: %s hits", jid, len(sfx))
        timeline = EditTimeline(
            captions=list(captions.captions),
            graphics=graphics,
            sfx=sfx,
        )
        return validate_edit_timeline(timeline, video_duration)

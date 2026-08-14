"""Cut agent — consumes EditorialAnalysis later. Not implemented in this pass."""

from __future__ import annotations

from app.editorial.models import CutDecision, EditorialAnalysis


class CutAgent:
    async def plan(self, analysis: EditorialAnalysis) -> list[CutDecision]:
        _ = analysis
        return []

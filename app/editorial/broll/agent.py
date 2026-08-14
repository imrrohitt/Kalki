"""B-roll agent — consumes EditorialAnalysis later. Not implemented in this pass."""

from __future__ import annotations

from app.editorial.models import BrollDecision, EditorialAnalysis


class BrollAgent:
    async def plan(self, analysis: EditorialAnalysis) -> list[BrollDecision]:
        _ = analysis
        return []

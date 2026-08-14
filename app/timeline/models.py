from __future__ import annotations

from pydantic import BaseModel, Field

from app.captions.models import Caption
from app.editorial.models import BrollDecision, CutDecision, TransitionDecision, ZoomDecision


class EditTimeline(BaseModel):
    """Unified edit plan. Captions, zooms, and later cuts/b-roll/transitions."""

    version: str = "1.0"
    captions: list[Caption] = Field(default_factory=list)
    zooms: list[ZoomDecision] = Field(default_factory=list)
    cuts: list[CutDecision] = Field(default_factory=list)
    broll: list[BrollDecision] = Field(default_factory=list)
    transitions: list[TransitionDecision] = Field(default_factory=list)

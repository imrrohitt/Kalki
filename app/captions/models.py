from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


CaptionPosition = Literal["top", "center", "bottom_center"]
CaptionAnimation = Literal["none", "pop"]
CaptionStyle = Literal["dynamic_social"]
CaptionTreatment = Literal[
    "plain",
    "mix",
    "serif",
    "quote",
    "oval",
    "underline",
    "blob",
    "stack",
    "tape",
    "chip",
    "bubble",
]
# AI-judged context, not a look. Mood drives the reveal motion (a real
# surprise or an excited beat "blinks" in instead of the usual smooth rise);
# icon is a small contextual badge; cta flags a line genuinely asking the
# viewer to do something (drives the "bubble" treatment in the premium
# caption style). All default to the quiet case — most lines carry none.
CaptionMood = Literal["neutral", "surprise", "excited", "happy", "serious", "urgent"]
CaptionIcon = Literal[
    "none", "money", "growth", "idea", "video", "social", "check",
    "warning", "time", "target", "fire", "heart", "star", "lock", "question",
]
# Render-time-only variety for the premium caption style — never set by the
# LLM, chosen deterministically by the renderer from timing/geometry it alone
# knows about (font rotation, and whether this video's framing even shows a
# chest/torso area to place text in).
CaptionFont = Literal["default", "marker"]
CaptionArea = Literal["head", "chest"]


class CaptionWord(BaseModel):
    text: str
    start: float
    end: float
    emphasis: bool = False

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("word text must not be empty")
        return value

    @model_validator(mode="after")
    def check_times(self) -> CaptionWord:
        if self.start < 0:
            raise ValueError("word.start must be >= 0")
        if self.end <= self.start:
            raise ValueError("word.end must be > word.start")
        return self


class Caption(BaseModel):
    start: float
    end: float
    text: str
    position: CaptionPosition = "bottom_center"
    animation: CaptionAnimation = "pop"
    treatment: CaptionTreatment = "plain"
    mood: CaptionMood = "neutral"
    icon: CaptionIcon = "none"
    cta: bool = False
    font_variant: CaptionFont = "default"
    screen_area: CaptionArea = "head"
    words: list[CaptionWord] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("caption text must not be empty")
        return value

    @model_validator(mode="after")
    def check_times(self) -> Caption:
        if self.start < 0:
            raise ValueError("caption.start must be >= 0")
        if self.end <= self.start:
            raise ValueError("caption.end must be > caption.start")
        for word in self.words:
            if word.start < self.start - 1e-3:
                raise ValueError("word.start must be >= caption.start")
            if word.end > self.end + 1e-3:
                raise ValueError("word.end must be <= caption.end")
        return self


class CaptionTimeline(BaseModel):
    version: str = "1.0"
    style: CaptionStyle = "dynamic_social"
    captions: list[Caption] = Field(default_factory=list)

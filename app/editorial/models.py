from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


EditorialRole = Literal[
    "emphasis",
    "surprise",
    "contrast",
    "strong_opinion",
    "key_insight",
    "important_number",
    "question",
    "answer",
    "story_climax",
    "reveal",
    "warning",
    "hook",
    "cta",
    "emotional_peak",
    "humor",
    "transition",
    "generic",
    "assumption",
    "contradiction",
]

StoryPosition = Literal["setup", "development", "climax", "resolution", "none"]
ZoomStyle = Literal["slow_punch", "fast_punch", "hold", "punch_release", "none"]
ZoomEasing = Literal["linear", "ease_in", "ease_out", "ease_in_out"]
ZoomAction = Literal["apply", "skip"]


class SentenceSignals(BaseModel):
    emphasis: float = Field(0.0, ge=0.0, le=1.0)
    surprise: float = Field(0.0, ge=0.0, le=1.0)
    contrast: float = Field(0.0, ge=0.0, le=1.0)
    emotion: float = Field(0.0, ge=0.0, le=1.0)
    humor: float = Field(0.0, ge=0.0, le=1.0)
    question: float = Field(0.0, ge=0.0, le=1.0)
    reveal: float = Field(0.0, ge=0.0, le=1.0)
    warning: float = Field(0.0, ge=0.0, le=1.0)
    cta: float = Field(0.0, ge=0.0, le=1.0)

    def peak(self) -> float:
        return max(
            self.emphasis,
            self.surprise,
            self.contrast,
            self.emotion,
            self.humor,
            self.question,
            self.reveal,
            self.warning,
            self.cta,
        )


class ProsodySignals(BaseModel):
    """Timestamp-derived for now; loudness/pitch filled when audio analysis lands."""

    speaking_rate: float | None = None
    pause_before: float | None = None
    pause_after: float | None = None
    loudness: float | None = None
    pitch_change: float | None = None
    stressed_word: str | None = None


class ContextWindow(BaseModel):
    previous: str | None = None
    current: str
    next: str | None = None


class SentenceWindow(BaseModel):
    """Transcript sentence plus neighbors. No editorial judgment yet."""

    sentence_id: int
    start: float
    end: float
    text: str
    word_ids: list[int] = Field(default_factory=list)
    context: ContextWindow
    prosody: ProsodySignals = Field(default_factory=ProsodySignals)

    @model_validator(mode="after")
    def check_times(self) -> SentenceWindow:
        if self.end <= self.start:
            raise ValueError("sentence.end must be > start")
        return self


class SubjectBox(BaseModel):
    """Normalized subject bbox in the 9:16 frame (0-1)."""

    x: float = Field(0.18, ge=0.0, le=1.0)
    y: float = Field(0.16, ge=0.0, le=1.0)
    w: float = Field(0.64, ge=0.05, le=1.0)
    h: float = Field(0.50, ge=0.05, le=1.0)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


class ZoomMotion(BaseModel):
    """LLM-owned camera move. Renderer eases this; it must not snap."""

    apply: bool = False
    intensity: float = Field(0.0, ge=0.0, le=1.0)
    delay_ms: int = Field(0, ge=0, le=1200)
    ease_in_ms: int = Field(480, ge=0, le=2000)
    hold_ms: int = Field(600, ge=0, le=2500)
    ease_out_ms: int = Field(420, ge=0, le=2000)


class EditorialSentence(BaseModel):
    sentence_id: int
    start: float
    end: float
    text: str
    word_ids: list[int] = Field(default_factory=list)
    context: ContextWindow
    signals: SentenceSignals = Field(default_factory=SentenceSignals)
    editorial_role: EditorialRole = "generic"
    visual_interest: float = Field(0.0, ge=0.0, le=1.0)
    story_position: StoryPosition = "none"
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    prosody: ProsodySignals = Field(default_factory=ProsodySignals)
    zoom: ZoomMotion = Field(default_factory=ZoomMotion)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("sentence text must not be empty")
        return value


class StoryPattern(BaseModel):
    pattern: str
    sentence_ids: list[int]
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class EditorialAnalysis(BaseModel):
    """Central Editorial Intelligence output. Zoom/cut/b-roll consume this."""

    version: str = "1.0"
    sentences: list[EditorialSentence] = Field(default_factory=list)
    story_patterns: list[StoryPattern] = Field(default_factory=list)


class VisualContext(BaseModel):
    """Framing constraints. Face metrics stay optional until a detector exists."""

    face_scale: float = Field(0.55, ge=0.0, le=1.0)
    face_cx: float = Field(0.5, ge=0.0, le=1.0)
    face_cy: float = Field(0.5, ge=0.0, le=1.0)
    current_zoom: float = Field(1.0, ge=1.0, le=2.0)
    bbox: SubjectBox = Field(default_factory=SubjectBox)
    max_safe_scale: float = Field(1.22, ge=1.0, le=1.5)


class ZoomScore(BaseModel):
    sentence_id: int
    candidate: bool
    intent: EditorialRole
    semantic_score: float = Field(0.0, ge=0.0, le=1.0)
    visual_score: float = Field(0.0, ge=0.0, le=1.0)
    prosody_score: float = Field(0.0, ge=0.0, le=1.0)
    previous_zoom_distance: float = 0.0
    final_score: float = Field(0.0, ge=0.0, le=1.0)
    action: ZoomAction
    reason: str = ""


class ZoomDecision(BaseModel):
    """Renderer contract produced from LLM timing, clamped by subject bbox."""

    start: float
    end: float
    intent: EditorialRole
    style: ZoomStyle = "slow_punch"
    target_scale: float = Field(1.0, ge=1.0, le=1.4)
    easing: ZoomEasing = "ease_in_out"
    ease_in: float = Field(0.48, ge=0.0, le=2.5)
    hold: float = Field(0.55, ge=0.0, le=3.0)
    ease_out: float = Field(0.42, ge=0.0, le=2.5)
    release_duration: float = Field(0.42, ge=0.0, le=2.5)
    anchor_x: float = Field(0.5, ge=0.0, le=1.0)
    anchor_y: float = Field(0.42, ge=0.0, le=1.0)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    sentence_id: int
    score: float = Field(0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_times(self) -> ZoomDecision:
        if self.end <= self.start:
            raise ValueError("zoom.end must be > start")
        self.release_duration = self.ease_out
        return self

    @property
    def span_end(self) -> float:
        return self.start + self.ease_in + self.hold + self.ease_out


class CutDecision(BaseModel):
    start: float
    end: float
    intent: str
    confidence: float = 0.0


class BrollDecision(BaseModel):
    start: float
    end: float
    intent: str
    query: str = ""
    confidence: float = 0.0


GraphicKind = Literal[
    "term_card",
    "vs_split",
    "stat",
    "chip_row",
    "process",
    "quote",
    "topic",
    "bullets",
]
GraphicMotion = Literal["fade", "slide_up", "scale_in"]
SfxKind = Literal["whoosh", "swoosh", "impact", "hit"]


class SfxHit(BaseModel):
    """One timed sound effect under the voice track."""

    at: float = Field(..., ge=0.0)
    kind: SfxKind
    gain: float = Field(0.26, ge=0.05, le=0.8)
    reason: str = ""


class GraphicBullet(BaseModel):
    """One line that builds in after the title."""

    text: str
    icon: str = ""
    delay_ms: int = Field(0, ge=0, le=4000)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("bullet text must not be empty")
        return value


class GraphicBeat(BaseModel):
    """One motion card in the top half of a split reel."""

    start: float
    end: float
    kind: GraphicKind = "term_card"
    title: str
    subtitle: str = ""
    kicker: str = ""
    icon: str = ""
    glyph: str = ""
    chips: list[str] = Field(default_factory=list)
    bullets: list[GraphicBullet] = Field(default_factory=list)
    left: str = ""
    right: str = ""
    motion: GraphicMotion = "slide_up"
    confidence: float = Field(0.6, ge=0.0, le=1.0)

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("graphic title must not be empty")
        return value

    @model_validator(mode="after")
    def check_times(self) -> GraphicBeat:
        if self.end <= self.start:
            raise ValueError("graphic.end must be > start")
        return self


class TransitionDecision(BaseModel):
    at: float
    kind: str = "cut"
    confidence: float = 0.0


class LlmSentenceAnnotation(BaseModel):
    sentence_id: int
    editorial_role: EditorialRole
    signals: SentenceSignals = Field(default_factory=SentenceSignals)
    visual_interest: float = Field(0.5, ge=0.0, le=1.0)
    story_position: StoryPosition = "none"
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    zoom: ZoomMotion | None = None


class LlmAnalysisPayload(BaseModel):
    sentences: list[LlmSentenceAnnotation] = Field(default_factory=list)
    story_patterns: list[StoryPattern] = Field(default_factory=list)

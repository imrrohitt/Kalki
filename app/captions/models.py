from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


CaptionPosition = Literal["top", "center", "bottom_center"]
CaptionAnimation = Literal["none", "pop"]
CaptionStyle = Literal["dynamic_social"]


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

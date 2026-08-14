from pydantic import BaseModel, Field


class Word(BaseModel):
    word: str
    start: float
    end: float
    probability: float = 0.0


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    language: str | None = None
    language_probability: float | None = None
    duration: float | None = None
    segments: list[Segment] = Field(default_factory=list)

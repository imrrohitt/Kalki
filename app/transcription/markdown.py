from __future__ import annotations

import json
import re

from app.transcription.models import Segment, Transcript, Word

_RANGE = re.compile(
    r"^\s*\*{0,2}\s*(\d{1,2}):(\d{2})\s*[-–—]\s*(\d{1,2}):(\d{2})\s*\*{0,2}\s*$"
)
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’+\-/]*")


def _mmss(minutes: str, seconds: str) -> float:
    return int(minutes) * 60.0 + int(seconds)


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in text.replace("\n", " ").split():
        token = raw.strip(".,;:!?\"“”()[]{}")
        if not token:
            continue
        if token in {"—", "-", "–"}:
            continue
        tokens.append(token)
    return tokens or _WORD.findall(text)


def _words_for_span(text: str, start: float, end: float) -> list[Word]:
    tokens = _tokenize(text)
    if not tokens:
        return []
    span = max(end - start, 0.12 * len(tokens))
    step = span / len(tokens)
    words: list[Word] = []
    for i, token in enumerate(tokens):
        t0 = start + step * i
        t1 = start + step * (i + 1)
        words.append(
            Word(
                word=token,
                start=round(t0, 3),
                end=round(max(t1, t0 + 0.06), 3),
                probability=1.0,
            )
        )
    if words:
        words[-1].end = round(max(words[-1].end, end), 3)
    return words


def parse_timestamped_markdown(text: str, *, duration: float | None = None) -> Transcript:
    """Parse `**mm:ss - mm:ss**` blocks of English copy into a timed transcript."""
    blocks: list[tuple[float, float, list[str]]] = []
    current: tuple[float, float, list[str]] | None = None
    for line in (text or "").splitlines():
        stripped = line.strip()
        match = _RANGE.match(stripped.replace("*", ""))
        if match and stripped.startswith("*"):
            if current is not None:
                blocks.append(current)
            start = _mmss(match.group(1), match.group(2))
            end = _mmss(match.group(3), match.group(4))
            current = (start, max(end, start + 0.2), [])
            continue
        if match and current is None:
            start = _mmss(match.group(1), match.group(2))
            end = _mmss(match.group(3), match.group(4))
            current = (start, max(end, start + 0.2), [])
            continue
        if current is None:
            continue
        if stripped and not stripped.lower().startswith("here is the timestamped"):
            current[2].append(stripped)

    if current is not None:
        blocks.append(current)

    segments: list[Segment] = []
    for start, end, lines in blocks:
        body = " ".join(lines).strip()
        if not body:
            continue
        words = _words_for_span(body, start, end)
        if not words:
            continue
        segments.append(
            Segment(
                start=float(words[0].start),
                end=float(words[-1].end),
                text=body,
                words=words,
            )
        )

    if not segments:
        raise ValueError("No timestamped caption blocks found in transcript")

    last = segments[-1].end
    return Transcript(
        language="en",
        language_probability=1.0,
        duration=float(duration) if duration else last,
        segments=segments,
    )


def transcript_from_upload(payload: bytes, filename: str, *, duration: float | None = None) -> Transcript:
    name = (filename or "").lower()
    text = payload.decode("utf-8")
    if name.endswith(".json"):
        data = json.loads(text)
        return Transcript.model_validate(data)
    return parse_timestamped_markdown(text, duration=duration)

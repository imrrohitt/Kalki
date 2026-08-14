from __future__ import annotations

import re

from app.editorial.models import (
    EditorialAnalysis,
    EditorialRole,
    EditorialSentence,
    SentenceSignals,
    SentenceWindow,
    StoryPattern,
    StoryPosition,
    ZoomMotion,
)


_NUMBER = re.compile(r"\b\d+(\.\d+)?\s*(%|x|million|billion|k)\b", re.I)
_MONEY = re.compile(r"\$\s?\d|\b\d+\s*(million|billion|percent|%)\b", re.I)

_HOOK = re.compile(
    r"\b(nobody tells you|most people don't|here's the thing|wait for it|"
    r"you won't believe|secret|what if i told)\b",
    re.I,
)
_CTA = re.compile(
    r"\b(subscribe|follow me|link in|comment below|if you want to (learn|get)|"
    r"smash that|hit (that )?follow|check out)\b",
    re.I,
)
_WARNING = re.compile(
    r"\b(don't|do not|never|mistake|avoid|stop doing|warning|careful)\b", re.I
)
_REVEAL = re.compile(
    r"\b(actually|turns out|the (real|truth) (reason|is)|in reality|"
    r"what really happened|the twist)\b",
    re.I,
)
_CONTRAST = re.compile(
    r"^(but|however|instead|except|yet)\b|\b(that's actually wrong|"
    r"that's not|not the case|on the contrary)\b",
    re.I,
)
_SURPRISE = re.compile(
    r"\b(crazy|insane|shocked|unbelievable|no way|wild|suddenly)\b", re.I
)
_OPINION = re.compile(r"\b(i would never|never do this|hate|always|must)\b", re.I)
_INSIGHT = re.compile(
    r"\b(the real reason|that's why|the point is|what matters|"
    r"the bottleneck|the key)\b",
    re.I,
)
_ASSUMPTION = re.compile(
    r"\b(i thought|i used to think|people think|most developers think|"
    r"everyone assumes|it seemed)\b",
    re.I,
)
_CONTRADICTION = re.compile(
    r"\b(i was (completely )?wrong|that's (actually )?wrong|"
    r"not true|i was mistaken)\b",
    re.I,
)
_EMOTION = re.compile(
    r"\b(shocked|devastated|excited|terrified|proud|heartbroken)\b", re.I
)
_HUMOR = re.compile(r"\b(joke|funny|lol|hilarious|kidding)\b", re.I)
_GENERIC = re.compile(
    r"\b(today (we are|we're) going to|welcome back|in this video|"
    r"let's get started|hey guys)\b",
    re.I,
)
_TRANSITION = re.compile(
    r"\b(now here's|moving on|next|so anyway|here's the interesting)\b", re.I
)
_EMPHASIS = re.compile(
    r"\b(most important|the most|really|critical|essential|pay attention)\b", re.I
)


def _role_and_signals(window: SentenceWindow) -> tuple[EditorialRole, SentenceSignals, float]:
    text = window.context.current
    prev = window.context.previous or ""
    signals = SentenceSignals()
    role: EditorialRole = "generic"
    interest = 0.2

    if _GENERIC.search(text):
        return "generic", signals, 0.05
    if _CTA.search(text):
        signals.cta = 0.9
        return "cta", signals, 0.1

    if text.rstrip().endswith("?"):
        signals.question = 0.86
        role = "question"
        interest = 0.45
    if prev.rstrip().endswith("?") and not text.rstrip().endswith("?"):
        signals.emphasis = max(signals.emphasis, 0.7)
        role = "answer"
        interest = 0.78

    checks: list[tuple[re.Pattern[str], EditorialRole, str, float]] = [
        (_HOOK, "hook", "emphasis", 0.88),
        (_ASSUMPTION, "assumption", "emphasis", 0.4),
        (_CONTRADICTION, "contradiction", "contrast", 0.82),
        (_CONTRAST, "contrast", "contrast", 0.84),
        (_REVEAL, "reveal", "reveal", 0.9),
        (_WARNING, "warning", "warning", 0.86),
        (_SURPRISE, "surprise", "surprise", 0.8),
        (_OPINION, "strong_opinion", "emphasis", 0.84),
        (_INSIGHT, "key_insight", "emphasis", 0.8),
        (_EMOTION, "emotional_peak", "emotion", 0.78),
        (_HUMOR, "humor", "humor", 0.7),
        (_TRANSITION, "transition", "emphasis", 0.5),
        (_EMPHASIS, "emphasis", "emphasis", 0.76),
    ]
    for pattern, candidate, signal_name, score in checks:
        if pattern.search(text):
            setattr(signals, signal_name, max(getattr(signals, signal_name), score))
            locked = role in {"assumption", "cta", "question", "contradiction"}
            if role == "generic" or (not locked and score >= interest):
                role = candidate
                interest = max(interest, score)

    if _NUMBER.search(text) or _MONEY.search(text):
        signals.emphasis = max(signals.emphasis, 0.72)
        if role in {"generic", "assumption", "reveal"}:
            if role != "assumption":
                role = "important_number" if role == "generic" else role
        interest = max(interest, 0.8)

    if window.prosody.pause_before and window.prosody.pause_before >= 0.5:
        signals.emphasis = max(signals.emphasis, 0.55)
        interest = min(1.0, interest + 0.08)
    if window.prosody.pause_after and window.prosody.pause_after >= 0.4:
        signals.emphasis = max(signals.emphasis, 0.5)

    return role, signals, min(1.0, interest)


def _zoom_for_role(role: EditorialRole) -> ZoomMotion:
    """Fallback camera when the LLM is off. Same timing language the prompt uses."""
    if role in {"generic", "cta", "assumption"}:
        return ZoomMotion(apply=False, intensity=0.0)
    presets: dict[EditorialRole, dict[str, int | float]] = {
        "reveal": dict(intensity=0.88, delay_ms=180, ease_in_ms=560, hold_ms=720, ease_out_ms=500),
        "story_climax": dict(intensity=0.82, delay_ms=80, ease_in_ms=600, hold_ms=700, ease_out_ms=480),
        "warning": dict(intensity=0.80, delay_ms=40, ease_in_ms=500, hold_ms=600, ease_out_ms=440),
        "surprise": dict(intensity=0.75, delay_ms=20, ease_in_ms=480, hold_ms=520, ease_out_ms=430),
        "answer": dict(intensity=0.72, delay_ms=80, ease_in_ms=520, hold_ms=650, ease_out_ms=460),
        "strong_opinion": dict(intensity=0.70, delay_ms=40, ease_in_ms=500, hold_ms=580, ease_out_ms=440),
        "important_number": dict(intensity=0.68, delay_ms=100, ease_in_ms=540, hold_ms=800, ease_out_ms=420),
        "hook": dict(intensity=0.62, delay_ms=0, ease_in_ms=520, hold_ms=550, ease_out_ms=450),
        "key_insight": dict(intensity=0.60, delay_ms=60, ease_in_ms=560, hold_ms=620, ease_out_ms=460),
        "emotional_peak": dict(intensity=0.58, delay_ms=80, ease_in_ms=620, hold_ms=650, ease_out_ms=500),
        "humor": dict(intensity=0.55, delay_ms=0, ease_in_ms=480, hold_ms=380, ease_out_ms=400),
        "contradiction": dict(intensity=0.52, delay_ms=60, ease_in_ms=540, hold_ms=520, ease_out_ms=460),
        "contrast": dict(intensity=0.50, delay_ms=40, ease_in_ms=540, hold_ms=500, ease_out_ms=450),
        "emphasis": dict(intensity=0.48, delay_ms=40, ease_in_ms=520, hold_ms=480, ease_out_ms=430),
        "question": dict(intensity=0.32, delay_ms=0, ease_in_ms=680, hold_ms=400, ease_out_ms=520),
        "transition": dict(intensity=0.28, delay_ms=0, ease_in_ms=640, hold_ms=400, ease_out_ms=500),
    }
    raw = presets.get(role)
    if raw is None:
        return ZoomMotion(apply=False, intensity=0.0)
    return ZoomMotion(apply=True, **raw)


def _story_position(role: EditorialRole) -> StoryPosition:
    if role in {"hook", "assumption"}:
        return "setup"
    if role in {"reveal", "story_climax", "important_number"}:
        return "climax"
    if role in {"cta"}:
        return "resolution"
    if role == "generic":
        return "none"
    return "development"


def detect_story_patterns(sentences: list[EditorialSentence]) -> list[StoryPattern]:
    patterns: list[StoryPattern] = []
    roles = [s.editorial_role for s in sentences]
    for i in range(len(roles) - 2):
        triple = (roles[i], roles[i + 1], roles[i + 2])
        if triple[0] in {"assumption", "generic", "emphasis"} and triple[1] in {
            "contradiction",
            "contrast",
            "surprise",
            "emotional_peak",
        } and triple[2] in {"reveal", "important_number", "key_insight", "story_climax"}:
            patterns.append(
                StoryPattern(
                    pattern="setup → reversal → reveal",
                    sentence_ids=[i, i + 1, i + 2],
                    confidence=0.82,
                )
            )
        if triple[0] == "question" and triple[1] in {"answer", "reveal", "key_insight"}:
            patterns.append(
                StoryPattern(
                    pattern="question → answer",
                    sentence_ids=[i, i + 1],
                    confidence=0.8,
                )
            )
    for i in range(len(roles) - 1):
        if roles[i] == "question" and roles[i + 1] in {
            "answer",
            "reveal",
            "key_insight",
            "important_number",
        }:
            already = any(i in p.sentence_ids and i + 1 in p.sentence_ids for p in patterns)
            if not already:
                patterns.append(
                    StoryPattern(
                        pattern="question → answer",
                        sentence_ids=[i, i + 1],
                        confidence=0.78,
                    )
                )
    return patterns


def heuristic_analyze(windows: list[SentenceWindow]) -> EditorialAnalysis:
    sentences: list[EditorialSentence] = []
    for window in windows:
        role, signals, interest = _role_and_signals(window)
        sentences.append(
            EditorialSentence(
                sentence_id=window.sentence_id,
                start=window.start,
                end=window.end,
                text=window.text,
                word_ids=window.word_ids,
                context=window.context,
                signals=signals,
                editorial_role=role,
                visual_interest=round(interest, 3),
                story_position=_story_position(role),
                confidence=0.45 if role == "generic" else 0.62,
                prosody=window.prosody,
                zoom=_zoom_for_role(role),
            )
        )
    return EditorialAnalysis(
        sentences=sentences,
        story_patterns=detect_story_patterns(sentences),
    )

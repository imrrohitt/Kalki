from __future__ import annotations

from app.editorial.graphics.terms import (
    extract_number,
    extract_terms,
    fix_asr_text,
    is_contrast,
    is_process,
)
from app.editorial.models import EditorialAnalysis, EditorialSentence, GraphicBeat, GraphicBullet

MIN_HOLD = 4.2
MAX_HOLD = 7.4
GAP = 0.14
TOPIC_TITLE = "RAG vs Fine-tuning"
TOPIC_SUB = "same model, two ways to update it"


def _span(sentence: EditorialSentence) -> tuple[float, float]:
    start = float(sentence.start)
    end = max(float(sentence.end), start + MIN_HOLD)
    if end - start > MAX_HOLD:
        end = start + MAX_HOLD
    return start, end


def _bullets(items: list[tuple[str, str]]) -> list[GraphicBullet]:
    out: list[GraphicBullet] = []
    for i, (icon, text) in enumerate(items[:3]):
        out.append(GraphicBullet(icon=icon, text=text, delay_ms=i * 480))
    return out


def _beat_for_sentence(sentence: EditorialSentence) -> GraphicBeat | None:
    text = fix_asr_text(sentence.text)
    terms = extract_terms(text)
    role = sentence.editorial_role
    start, end = _span(sentence)

    if role in {"cta", "generic"} and not terms:
        return None

    if is_contrast(sentence) or (
        any(t[0] == "RAG" for t in terms) and any(t[0] == "Fine-tuning" for t in terms)
    ):
        return GraphicBeat(
            start=start,
            end=end,
            kind="vs_split",
            title="Which one should you use?",
            left="RAG",
            right="Fine-tuning",
            subtitle="fresh docs vs trained weights",
            kicker="THE TRADEOFF",
            icon="⚖️",
            bullets=_bullets(
                [
                    ("📄", "RAG reads your latest documents"),
                    ("🧠", "Fine-tuning rewrites model weights"),
                    ("💡", "Pick based on how often data changes"),
                ]
            ),
            motion="slide_up",
            confidence=0.86,
        )

    number = extract_number(text)
    if role == "important_number" and number:
        return GraphicBeat(
            start=start,
            end=end,
            kind="stat",
            title=number,
            subtitle=terms[0][0] if terms else "key figure",
            kicker="NUMBER",
            icon="📊",
            motion="scale_in",
            confidence=0.8,
        )

    if any(t[0] in {"LoRA", "PEFT"} for t in terms):
        return GraphicBeat(
            start=start,
            end=end,
            kind="bullets",
            title="Fine-tune without a full retrain",
            kicker="PEFT",
            icon="⚙️",
            bullets=_bullets(
                [
                    ("🧩", "LoRA adds small adapter layers"),
                    ("⚙️", "PEFT trains a slice, not the whole net"),
                    ("💰", "Cheaper than updating every weight"),
                ]
            ),
            motion="slide_up",
            confidence=0.8,
        )

    if any(t[0] == "RAG" for t in terms) and is_process(text):
        return GraphicBeat(
            start=start,
            end=end,
            kind="process",
            title="How RAG answers a question",
            kicker="RAG PATH",
            icon="🔍",
            chips=["Documents", "Retrieve", "Generate"],
            bullets=_bullets(
                [
                    ("📂", "Keep company docs in a store"),
                    ("🔍", "Retrieve the chunks that match"),
                    ("✨", "The LLM writes the answer from those"),
                ]
            ),
            motion="slide_up",
            confidence=0.82,
        )

    if any(t[0] == "Fine-tuning" for t in terms) and is_process(text):
        return GraphicBeat(
            start=start,
            end=end,
            kind="process",
            title="How fine-tuning actually works",
            kicker="FINE-TUNE PATH",
            icon="🧠",
            chips=["Base LLM", "Domain data", "New weights"],
            bullets=_bullets(
                [
                    ("🧠", "Start from an existing LLM"),
                    ("📁", "Train it on your domain examples"),
                    ("⚙️", "The weights themselves change"),
                ]
            ),
            motion="slide_up",
            confidence=0.8,
        )

    if any(t[0] == "RAG" for t in terms) and "cost" in text.lower():
        return GraphicBeat(
            start=start,
            end=end,
            kind="bullets",
            title="RAG stays cheap as data moves",
            kicker="COST",
            icon="💰",
            bullets=_bullets(
                [
                    ("🔁", "Re-index docs instead of retraining"),
                    ("💰", "No giant GPU run every update"),
                    ("📄", "Best when documents keep changing"),
                ]
            ),
            motion="slide_up",
            confidence=0.8,
        )

    if any(t[0] == "Fine-tuning" for t in terms) and "cost" in text.lower():
        return GraphicBeat(
            start=start,
            end=end,
            kind="bullets",
            title="Fine-tuning gets expensive fast",
            kicker="COST",
            icon="💰",
            bullets=_bullets(
                [
                    ("💰", "You pay GPU time to retrain"),
                    ("🔁", "Every doc change means another run"),
                    ("📂", "Use it when the data is stable"),
                ]
            ),
            motion="slide_up",
            confidence=0.8,
        )

    if terms:
        canonical, subtitle = terms[0]
        icon = {
            "RAG": "🔍",
            "Fine-tuning": "🧠",
            "LLM": "💡",
            "Documents": "📄",
            "Domain data": "📁",
            "Cost": "💰",
            "Latest data": "🔁",
            "Static data": "📂",
            "LoRA": "🧩",
            "PEFT": "⚙️",
        }.get(canonical, "✨")
        title = {
            "RAG": "What RAG actually does",
            "Fine-tuning": "What fine-tuning changes",
            "LLM": "Same base model, two paths",
            "Documents": "Keep knowledge in the documents",
            "Domain data": "Train on your domain examples",
            "Cost": "What this choice actually costs",
            "Latest data": "When the data keeps changing",
            "Static data": "When the knowledge stays put",
            "LoRA": "LoRA trains a small adapter",
            "PEFT": "PEFT skips a full retrain",
        }.get(canonical, subtitle[:48] if len(subtitle) > 12 else canonical)
        extra = [t[0] for t in terms[1:3]]
        kicker = "CONCEPT"
        if role in {"hook", "question"}:
            kicker = "HOOK"
        elif role in {"key_insight", "reveal"}:
            kicker = "INSIGHT"
        lines = [(icon, subtitle)]
        if extra:
            lines.append(("✨", "Also in play: " + " · ".join(extra)))
        if role == "question":
            lines.append(("🎯", "This is the question to nail in interviews"))
        return GraphicBeat(
            start=start,
            end=end,
            kind="bullets",
            title=title,
            subtitle=subtitle,
            kicker=kicker,
            icon=icon,
            chips=extra,
            bullets=_bullets(lines),
            motion="slide_up",
            confidence=0.74,
        )

    if role in {"hook", "key_insight", "reveal", "strong_opinion"}:
        snippet = text.strip()
        if len(snippet) > 48:
            snippet = snippet[:45].rsplit(" ", 1)[0] + "…"
        return GraphicBeat(
            start=start,
            end=end,
            kind="bullets",
            title=snippet,
            kicker="TAKE",
            icon="💡",
            bullets=_bullets(
                [
                    ("💡", snippet),
                    ("🎯", "Hold this point while he explains it"),
                ]
            ),
            motion="slide_up",
            confidence=0.6,
        )
    return None


def _no_overlap(beats: list[GraphicBeat]) -> list[GraphicBeat]:
    ordered = sorted(beats, key=lambda b: (b.start, -b.confidence))
    kept: list[GraphicBeat] = []
    last_end = -1.0
    for beat in ordered:
        start = max(beat.start, last_end + GAP)
        end = beat.end
        if end - start < MIN_HOLD * 0.62:
            continue
        if end - start > MAX_HOLD:
            end = start + MAX_HOLD
        kept.append(beat.model_copy(update={"start": round(start, 3), "end": round(end, 3)}))
        last_end = end
    return kept


def cover_duration(beats: list[GraphicBeat], duration: float) -> list[GraphicBeat]:
    """Stretch cards so the top canvas has no empty gaps."""
    if not beats or duration <= 0:
        return beats
    ordered = sorted(beats, key=lambda b: b.start)
    covered: list[GraphicBeat] = []
    for i, beat in enumerate(ordered):
        start = 0.0 if i == 0 else round(beat.start, 3)
        if i + 1 < len(ordered):
            end = round(ordered[i + 1].start, 3)
        else:
            end = round(duration, 3)
        if end - start < 0.5:
            continue
        covered.append(beat.model_copy(update={"start": start, "end": end}))
    return covered


def _dedupe_titles(beats: list[GraphicBeat]) -> list[GraphicBeat]:
    kept: list[GraphicBeat] = []
    last_title = ""
    last_at = -99.0
    for beat in beats:
        if beat.kind in {"vs_split", "process", "stat"}:
            kept.append(beat)
            last_title = beat.title
            last_at = beat.start
            continue
        if beat.title == last_title and beat.start - last_at < 9.0:
            continue
        kept.append(beat)
        last_title = beat.title
        last_at = beat.start
    return kept


def _clamp_to_duration(beats: list[GraphicBeat], duration: float) -> list[GraphicBeat]:
    kept: list[GraphicBeat] = []
    for beat in beats:
        end = min(beat.end, duration)
        if end - beat.start < 0.9:
            continue
        kept.append(beat.model_copy(update={"end": round(end, 3)}))
    return kept


def plan_graphics(
    analysis: EditorialAnalysis,
    *,
    video_duration: float,
) -> list[GraphicBeat]:
    raw: list[GraphicBeat] = []
    for sentence in analysis.sentences:
        beat = _beat_for_sentence(sentence)
        if beat is not None:
            raw.append(beat)
    beats = _no_overlap(raw)
    beats = _dedupe_titles(beats)
    beats = _clamp_to_duration(beats, video_duration)
    beats = cover_duration(beats, video_duration)
    if not beats and video_duration > 1:
        beats = [
            GraphicBeat(
                start=0.0,
                end=min(video_duration, 5.0),
                kind="topic",
                title=TOPIC_TITLE,
                subtitle=TOPIC_SUB,
                kicker="TOPIC",
                icon="⚖️",
                bullets=_bullets(
                    [
                        ("🔍", "RAG retrieves, then the model writes"),
                        ("🧠", "Fine-tuning changes the model itself"),
                        ("💡", "Same LLM, two ways to update it"),
                    ]
                ),
                motion="slide_up",
                confidence=0.5,
            )
        ]
    return cover_duration(_clamp_to_duration(beats, video_duration), video_duration)

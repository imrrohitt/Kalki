from __future__ import annotations

from app.asr import fix_asr_text, stabilize_copy
from app.captions.models import Caption, CaptionTimeline, CaptionWord
from app.captions.validation import validate_caption_timeline
from app.transcription.models import Transcript, Word

# One on-screen group. Longer LLM dumps freeze the last line for the rest of the video.
MAX_CAPTION_WORDS = 6
MAX_CAPTION_SEC = 3.8
MAX_CAPTION_HOLD = 0.85

_KEEP_PHRASES = (
    "fine tuning",
    "fine-tuning",
    "rag system",
    "raka system",
    "ai interviews",
    "domain data",
    "existing llm",
    "static data",
    "latest data",
    "lora",
    "peft",
)

_EMPHASIS = {
    "rag",
    "raka",
    "llm",
    "lora",
    "peft",
    "fine-tuning",
    "tuning",
    "ai",
    "cost",
    "documents",
}


def _flatten(transcript: Transcript) -> list[Word]:
    words: list[Word] = []
    for seg in transcript.segments:
        for w in seg.words:
            token = fix_asr_text(w.word.strip())
            if not token:
                continue
            end = float(w.end) if w.end > w.start else float(w.start) + 0.08
            words.append(Word(word=token, start=float(w.start), end=end, probability=w.probability))
    return words


def _join_preview(words: list[Word], n: int) -> str:
    return " ".join(w.word.lower() for w in words[:n])


def _group_words(words: list[Word]) -> list[list[Word]]:
    groups: list[list[Word]] = []
    i = 0
    n = len(words)
    while i < n:
        remaining = words[i:]
        take = 3
        preview3 = _join_preview(remaining, 3)
        preview2 = _join_preview(remaining, 2)
        if any(preview3.startswith(p) or preview3 == p for p in _KEEP_PHRASES):
            take = min(3, len(remaining))
        elif any(preview2.startswith(p) or preview2 == p for p in _KEEP_PHRASES):
            take = min(2, len(remaining))
        else:
            take = 2 if len(remaining) >= 2 else 1
            if len(remaining) >= 3:
                gap = remaining[2].start - remaining[1].end
                if gap < 0.18 and len(remaining[2].word) <= 4:
                    take = 3
            if len(remaining) >= 2:
                gap0 = remaining[1].start - remaining[0].end
                if gap0 >= 0.38:
                    take = 1
        chunk = remaining[:take]
        # Don't leave a dangling last word if we can fold it.
        if i + take == n - 1 and take <= 2:
            chunk = remaining
            i = n
        else:
            i += len(chunk)
        groups.append(chunk)
    return groups


def _time_slice_words(words: list[Word]) -> list[list[Word]]:
    packs: list[list[Word]] = []
    current: list[Word] = []
    t0 = float(words[0].start) if words else 0.0
    for w in words:
        if current and (float(w.end) - t0) > MAX_CAPTION_SEC:
            packs.append(current)
            current = [w]
            t0 = float(w.start)
        else:
            if not current:
                t0 = float(w.start)
            current.append(w)
    if current:
        packs.append(current)
    return packs


def packs_for_words(words: list[Word]) -> list[list[Word]]:
    if not words:
        return []
    duration = float(words[-1].end) - float(words[0].start)
    if len(words) <= MAX_CAPTION_WORDS and duration <= MAX_CAPTION_SEC:
        return [words]
    grouped = _group_words(words) if len(words) > 1 else [words]
    out: list[list[Word]] = []
    for group in grouped:
        span = float(group[-1].end) - float(group[0].start)
        if span <= MAX_CAPTION_SEC:
            out.append(group)
        elif len(group) == 1:
            w = group[0]
            out.append(
                [
                    Word(
                        word=w.word,
                        start=float(w.start),
                        end=min(float(w.end), float(w.start) + MAX_CAPTION_SEC),
                        probability=w.probability,
                    )
                ]
            )
        else:
            out.extend(_time_slice_words(group))
    return out


def explode_caption_timeline(timeline: CaptionTimeline) -> CaptionTimeline:
    """Split captions that would sit on screen for many seconds without changing."""
    exploded: list[Caption] = []
    for cap in timeline.captions:
        words = list(cap.words)
        if not words:
            display = (cap.text or "").replace("\\n", " ").replace("\n", " ").strip()
            tokens = [tok for tok in display.split() if tok]
            if tokens:
                span = max(float(cap.end) - float(cap.start), 0.08 * len(tokens))
                t0 = float(cap.start)
                words = [
                    CaptionWord(
                        text=tok,
                        start=t0 + span * i / len(tokens),
                        end=t0 + span * (i + 1) / len(tokens),
                    )
                    for i, tok in enumerate(tokens)
                ]
        duration = float(cap.end) - float(cap.start)
        if not words:
            exploded.append(cap)
            continue
        if len(words) <= MAX_CAPTION_WORDS and duration <= MAX_CAPTION_SEC:
            exploded.append(cap)
            continue
        asr = [
            Word(word=w.text, start=float(w.start), end=float(w.end), probability=1.0)
            for w in words
        ]
        emphasis_at = {
            (w.text.lower(), round(float(w.start), 2)): w.emphasis for w in words
        }
        for pack in packs_for_words(asr):
            start = float(pack[0].start)
            end = max(float(pack[-1].end), start + 0.25)
            pack_words = [
                CaptionWord(
                    text=w.word,
                    start=max(float(w.start), start),
                    end=min(max(float(w.end), max(float(w.start), start) + 0.05), end),
                    emphasis=emphasis_at.get(
                        (w.word.lower(), round(float(w.start), 2)), False
                    ),
                )
                for w in pack
            ]
            exploded.append(
                Caption(
                    start=start,
                    end=end,
                    text=stabilize_copy(" ".join(w.word for w in pack)) or pack[0].word,
                    position=cap.position,
                    animation=cap.animation,
                    words=pack_words,
                )
            )
    return CaptionTimeline(
        version=timeline.version,
        style=timeline.style,
        captions=_clamp_caption_holds(exploded),
    )


def _clamp_caption_holds(captions: list[Caption]) -> list[Caption]:
    ordered = sorted(captions, key=lambda c: (c.start, c.end))
    clamped: list[Caption] = []
    for i, cap in enumerate(ordered):
        spoken_end = max((float(w.end) for w in cap.words), default=float(cap.end))
        nxt = (
            float(ordered[i + 1].start)
            if i + 1 < len(ordered)
            else spoken_end + MAX_CAPTION_HOLD
        )
        end = min(
            max(float(cap.end), spoken_end + 0.12),
            spoken_end + MAX_CAPTION_HOLD,
            nxt - 0.03,
        )
        if end <= float(cap.start):
            end = float(cap.start) + 0.22
        words = []
        for w in cap.words:
            w_end = min(float(w.end), end)
            w_start = min(max(float(w.start), float(cap.start)), w_end - 0.05)
            if w_end <= w_start:
                w_end = w_start + 0.05
            words.append(
                CaptionWord(
                    text=w.text,
                    start=w_start,
                    end=min(w_end, end),
                    emphasis=w.emphasis,
                )
            )
        clamped.append(
            Caption(
                start=cap.start,
                end=end,
                text=cap.text,
                position=cap.position,
                animation=cap.animation,
                words=words,
            )
        )
    return clamped


def heuristic_caption_timeline(
    transcript: Transcript,
    video_duration: float,
) -> CaptionTimeline:
    words = _flatten(transcript)
    if not words:
        raise RuntimeError("No words in transcript for caption generation")
    captions: list[Caption] = []
    for group in _group_words(words):
        start = group[0].start
        end = max(max(w.end for w in group), start + 0.25)
        captions.append(
            Caption(
                start=start,
                end=end,
                text=stabilize_copy(" ".join(w.word for w in group)),
                position="center",
                animation="pop",
                words=[
                    CaptionWord(
                        text=stabilize_copy(w.word) or w.word,
                        start=max(w.start, start),
                        end=min(max(w.end, max(w.start, start) + 0.05), end),
                        emphasis=w.word.lower().strip(".,!?") in _EMPHASIS,
                    )
                    for w in group
                ],
            )
        )
    data = CaptionTimeline(captions=captions)
    return validate_caption_timeline(data, video_duration)

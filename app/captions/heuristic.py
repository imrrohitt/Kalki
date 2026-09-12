from __future__ import annotations

from app.asr import fix_asr_text, stabilize_copy
from app.captions.copy import caption_spoken_case
from app.captions.models import Caption, CaptionTimeline, CaptionTreatment, CaptionWord
from app.captions.validation import validate_caption_timeline
from app.transcription.models import Transcript, Word

_SPECIAL = {"oval", "blob", "underline"}
_SPECIAL_GAP = 10.0

# One on-screen group. Longer LLM dumps freeze the last line for the rest of the video.
MAX_CAPTION_WORDS = 6
MAX_CAPTION_SEC = 3.8
MAX_CAPTION_HOLD = 0.85

_KEEP_PHRASES = (
    "fine tuning",
    "fine-tuning",
    "rag system",
    "raka system",
    "ai engineer",
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
    "engineer",
    "cost",
    "documents",
    "gdpr",
    "creator",
    "interviews",
    "interviewing",
    "strategy",
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


def explode_caption_timeline(
    timeline: CaptionTimeline,
    video_duration: float | None = None,
) -> CaptionTimeline:
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
                    treatment=cap.treatment,
                    words=pack_words,
                )
            )
    return CaptionTimeline(
        version=timeline.version,
        style=timeline.style,
        captions=_clamp_caption_holds(exploded, video_duration=video_duration),
    )


def _clamp_caption_holds(
    captions: list[Caption],
    video_duration: float | None = None,
) -> list[Caption]:
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
        if video_duration and video_duration > 0:
            end = min(end, float(video_duration))
        end = max(end, float(cap.start) + 0.22)
        if video_duration and video_duration > 0:
            end = min(end, float(video_duration))
        if end <= float(cap.start):
            end = min(float(cap.start) + 0.22, float(video_duration or cap.start + 0.22))
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
                treatment=cap.treatment,
                words=words,
            )
        )
    return clamped


def pace_caption_treatments(timeline: CaptionTimeline) -> CaptionTimeline:
    """Keep ovals/blobs rare so the reel stays happening, not surprising."""
    last = -999.0
    captions: list[Caption] = []
    for cap in timeline.captions:
        treatment: CaptionTreatment = cap.treatment
        if treatment in _SPECIAL:
            if float(cap.start) - last < _SPECIAL_GAP:
                treatment = "serif"
            else:
                last = float(cap.start)
        if treatment == "plain" and any(w.emphasis for w in cap.words):
            treatment = "mix"
        captions.append(cap.model_copy(update={"treatment": treatment}))
    return CaptionTimeline(
        version=timeline.version,
        style=timeline.style,
        captions=captions,
    )


_FILLER = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "so", "it",
    "is", "my", "just", "just", "when", "into", "into", "from", "as", "at",
    "i", "me", "we", "you", "they", "them", "this", "that", "there", "here",
}

_OVAL_TERMS = (
    "rag",
    "fine-tuning",
    "fine tuning",
    "llm",
    "ai engineer",
    "lora",
    "peft",
)


def _token_count(caption: Caption) -> int:
    return len([t for t in caption.text.replace("\n", " ").split() if t])


def _merge_captions(a: Caption, b: Caption) -> Caption:
    text = f"{a.text.replace(chr(10), ' ')} {b.text.replace(chr(10), ' ')}".strip()
    words = list(a.words) + [
        w.model_copy(update={"start": max(float(w.start), float(a.start))})
        for w in b.words
    ]
    treatment = a.treatment
    if b.treatment in _SPECIAL or (b.treatment == "mix" and a.treatment == "plain"):
        treatment = b.treatment
    return Caption(
        start=a.start,
        end=max(float(a.end), float(b.end)),
        text=text,
        position=a.position,
        animation=a.animation,
        treatment=treatment,
        words=words,
    )


def coalesce_short_captions(timeline: CaptionTimeline) -> CaptionTimeline:
    """Join 1–2 word leftovers so lines read like the reference reels."""
    caps = list(timeline.captions)
    if not caps:
        return timeline
    out: list[Caption] = []
    current = caps[0]
    for nxt in caps[1:]:
        gap = float(nxt.start) - float(current.end)
        n = _token_count(current) + _token_count(nxt)
        tiny = _token_count(current) <= 2 or _token_count(nxt) <= 2
        if tiny and gap <= 0.55 and n <= 4:
            current = _merge_captions(current, nxt)
            continue
        out.append(current)
        current = nxt
    out.append(current)
    return CaptionTimeline(
        version=timeline.version, style=timeline.style, captions=out
    )


def _stem(token: str) -> str:
    return token.strip(".,!?;:\"'“”").lower()


def _mark_unique_words(caption: Caption) -> list[CaptionWord]:
    marked: list[CaptionWord] = []
    for word in caption.words:
        stem = _stem(word.text)
        hot = bool(word.emphasis) or stem in _EMPHASIS
        if stem in _FILLER:
            hot = False
        marked.append(word.model_copy(update={"emphasis": hot}))
    return marked


def decorate_caption_treatments(timeline: CaptionTimeline) -> CaptionTimeline:
    """Promote unique phrases to serif/oval without decorating filler."""
    last_special = -999.0
    captions: list[Caption] = []
    for cap in timeline.captions:
        blob = cap.text.replace("\n", " ").lower()
        words = _mark_unique_words(cap)
        has_hot = any(w.emphasis for w in words)
        treatment = cap.treatment
        if any(term in blob for term in _OVAL_TERMS):
            if float(cap.start) - last_special >= _SPECIAL_GAP:
                treatment = "oval"
                last_special = float(cap.start)
            elif treatment in {"plain", "mix"}:
                treatment = "serif"
        elif treatment == "mix" and not has_hot:
            treatment = "plain"
        elif treatment == "serif" and not has_hot:
            treatment = "plain"
        elif treatment == "plain" and has_hot:
            treatment = "mix"
        captions.append(cap.model_copy(update={"treatment": treatment, "words": words}))
    return CaptionTimeline(
        version=timeline.version, style=timeline.style, captions=captions
    )


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
        raw_words = [
            CaptionWord(
                text=stabilize_copy(w.word) or w.word,
                start=max(w.start, start),
                end=min(max(w.end, max(w.start, start) + 0.05), end),
                emphasis=w.word.lower().strip(".,!?") in _EMPHASIS,
            )
            for w in group
        ]
        treatment: CaptionTreatment = "mix" if any(w.emphasis for w in raw_words) else "plain"
        text = caption_spoken_case(
            stabilize_copy(" ".join(w.word for w in group)),
            treatment=treatment,
        )
        cased = [
            w.model_copy(update={"text": caption_spoken_case(w.text, treatment="plain") or w.text})
            for w in raw_words
        ]
        captions.append(
            Caption(
                start=start,
                end=end,
                text=text,
                position="center",
                animation="pop",
                treatment=treatment,
                words=cased,
            )
        )
    data = CaptionTimeline(captions=captions)
    timeline = coalesce_short_captions(data)
    timeline = decorate_caption_treatments(timeline)
    timeline = pace_caption_treatments(timeline)
    return validate_caption_timeline(timeline, video_duration)

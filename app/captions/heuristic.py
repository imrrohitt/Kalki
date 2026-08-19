from __future__ import annotations

from app.asr import fix_asr_text, stabilize_copy
from app.captions.models import Caption, CaptionTimeline, CaptionWord
from app.captions.validation import validate_caption_timeline
from app.transcription.models import Transcript, Word

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

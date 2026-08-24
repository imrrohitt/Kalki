from __future__ import annotations

from app.asr import fix_asr_text, stabilize_copy
from app.transcription.models import Segment, Transcript, Word


def align_words(original: list[Word], new_text: str) -> list[Word]:
    """Map a corrected sentence onto the original word timings."""
    tokens = [t for t in stabilize_copy(new_text).split() if t]
    if not tokens:
        return list(original)
    if not original:
        return [
            Word(word=token, start=0.0, end=0.12 * (i + 1), probability=0.5)
            for i, token in enumerate(tokens)
        ]
    if len(tokens) == len(original):
        return [
            Word(
                word=token,
                start=src.start,
                end=src.end,
                probability=src.probability,
            )
            for token, src in zip(tokens, original, strict=True)
        ]
    t0 = float(original[0].start)
    t1 = float(original[-1].end)
    span = max(t1 - t0, 0.08 * len(tokens))
    step = span / len(tokens)
    return [
        Word(
            word=token,
            start=round(t0 + i * step, 3),
            end=round(t0 + (i + 1) * step, 3),
            probability=0.7,
        )
        for i, token in enumerate(tokens)
    ]


def apply_segment_repairs(
    transcript: Transcript,
    repairs: dict[int, str],
) -> Transcript:
    segments: list[Segment] = []
    for i, seg in enumerate(transcript.segments):
        text = repairs.get(i)
        if not text:
            words = [
                w.model_copy(update={"word": fix_asr_text(w.word)})
                for w in seg.words
            ]
            cleaned = stabilize_copy(seg.text) or " ".join(w.word for w in words)
            segments.append(
                Segment(start=seg.start, end=seg.end, text=cleaned, words=words)
            )
            continue
        words = align_words(seg.words, text)
        if not words:
            continue
        segments.append(
            Segment(
                start=words[0].start,
                end=words[-1].end,
                text=" ".join(w.word for w in words),
                words=words,
            )
        )
    return transcript.model_copy(update={"segments": segments})


def heuristic_repair(transcript: Transcript) -> Transcript:
    repairs: dict[int, str] = {}
    for i, seg in enumerate(transcript.segments):
        source = seg.text or " ".join(w.word for w in seg.words)
        cleaned = stabilize_copy(source)
        if cleaned and cleaned != source.strip():
            repairs[i] = cleaned
    if not repairs:
        return apply_segment_repairs(transcript, {})
    return apply_segment_repairs(transcript, repairs)

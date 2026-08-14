from __future__ import annotations

import re

from app.editorial.models import ContextWindow, ProsodySignals, SentenceWindow
from app.transcription.models import Transcript, Word


_SENTENCE_END = re.compile(r"[.?!…]+$")
_PAUSE_GAP_SEC = 0.45
_MAX_WORDS = 16
_MAX_DURATION_SEC = 4.5


def flatten_words(transcript: Transcript) -> list[Word]:
    words: list[Word] = []
    for segment in transcript.segments:
        for word in segment.words:
            token = word.word.strip()
            if not token:
                continue
            end = float(word.end) if word.end > word.start else float(word.start) + 0.08
            words.append(
                Word(
                    word=token,
                    start=float(word.start),
                    end=end,
                    probability=word.probability,
                )
            )
    return words


def _should_break(
    prev: Word,
    current: Word,
    *,
    segment_changed: bool,
    group_len: int,
    group_start: float,
) -> bool:
    if segment_changed:
        return True
    if current.start - prev.end >= _PAUSE_GAP_SEC:
        return True
    if _SENTENCE_END.search(prev.word):
        return True
    if group_len >= _MAX_WORDS:
        return True
    if current.start - group_start >= _MAX_DURATION_SEC and group_len >= 4:
        return True
    return False


def _join_text(words: list[Word]) -> str:
    parts: list[str] = []
    for word in words:
        token = word.word
        if parts and token in {",", ".", "?", "!", ";", ":"}:
            parts[-1] = parts[-1] + token
        elif parts and token.startswith("'"):
            parts[-1] = parts[-1] + token
        else:
            parts.append(token)
    text = " ".join(parts).strip()
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    return text


def _prosody(
    words: list[Word],
    previous_end: float | None,
    next_start: float | None,
) -> ProsodySignals:
    start = words[0].start
    end = words[-1].end
    duration = max(end - start, 0.08)
    pause_before = None if previous_end is None else max(0.0, start - previous_end)
    pause_after = None if next_start is None else max(0.0, next_start - end)
    stressed = None
    if pause_after is not None and pause_after >= 0.35:
        stressed = words[-1].word
    return ProsodySignals(
        speaking_rate=round(len(words) / duration, 3),
        pause_before=None if pause_before is None else round(pause_before, 3),
        pause_after=None if pause_after is None else round(pause_after, 3),
        stressed_word=stressed,
    )


def build_sentence_windows(transcript: Transcript) -> list[SentenceWindow]:
    indexed: list[tuple[int, int, Word]] = []
    global_i = 0
    for seg_i, segment in enumerate(transcript.segments):
        for word in segment.words:
            token = word.word.strip()
            if not token:
                continue
            end = float(word.end) if word.end > word.start else float(word.start) + 0.08
            indexed.append(
                (
                    global_i,
                    seg_i,
                    Word(
                        word=token,
                        start=float(word.start),
                        end=end,
                        probability=word.probability,
                    ),
                )
            )
            global_i += 1
    if not indexed:
        return []

    groups: list[list[tuple[int, Word]]] = []
    first_id, first_seg, first_word = indexed[0]
    current: list[tuple[int, Word]] = [(first_id, first_word)]
    current_seg = first_seg
    for word_id, seg_i, word in indexed[1:]:
        prev = current[-1][1]
        if _should_break(
            prev,
            word,
            segment_changed=seg_i != current_seg,
            group_len=len(current),
            group_start=current[0][1].start,
        ):
            groups.append(current)
            current = [(word_id, word)]
            current_seg = seg_i
        else:
            current.append((word_id, word))
            current_seg = seg_i
    if current:
        groups.append(current)

    windows: list[SentenceWindow] = []
    texts: list[str] = []
    packed: list[tuple[list[int], list[Word]]] = []
    for group in groups:
        ids = [item[0] for item in group]
        group_words = [item[1] for item in group]
        packed.append((ids, group_words))
        texts.append(_join_text(group_words))

    for sentence_id, ((word_ids, group_words), text) in enumerate(zip(packed, texts)):
        previous = texts[sentence_id - 1] if sentence_id else None
        nxt = texts[sentence_id + 1] if sentence_id + 1 < len(texts) else None
        previous_end = packed[sentence_id - 1][1][-1].end if sentence_id else None
        next_start = packed[sentence_id + 1][1][0].start if sentence_id + 1 < len(packed) else None
        windows.append(
            SentenceWindow(
                sentence_id=sentence_id,
                start=group_words[0].start,
                end=group_words[-1].end,
                text=text,
                word_ids=word_ids,
                context=ContextWindow(previous=previous, current=text, next=nxt),
                prosody=_prosody(group_words, previous_end, next_start),
            )
        )
    return windows

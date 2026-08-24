from __future__ import annotations

import re

from app.asr import stabilize_copy
from app.editorial.graphics.planner import cover_duration
from app.editorial.graphics.terms import extract_number, extract_terms, is_contrast, is_process
from app.editorial.models import (
    EditorialAnalysis,
    EditorialSentence,
    GraphicBeat,
    GraphicBullet,
    GraphicNode,
)


MIN_HOLD = 5.2
MAX_HOLD = 9.5
GAP = 0.10

_QUESTION = re.compile(r"\?$|^(why|how|what|when|should|is|are|do)\b", re.I)


def _clip(text: str, max_chars: int) -> str:
    text = stabilize_copy(text)
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars + 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.strip(" ,.;:?") or text[:max_chars]


def _headline(text: str, max_words: int = 7) -> str:
    words = stabilize_copy(text).strip(" \"'").split()
    if not words:
        return "Listen to this"
    if words[0].lower() in {"so", "and", "but", "well", "okay", "um", "uh"}:
        words = words[1:] or words
    title = " ".join(words[:max_words]).strip(" ,.;:")
    return _clip(title, 44)


def _chunks(text: str, n: int, width: int) -> list[str]:
    words = stabilize_copy(text).split()
    if not words:
        return []
    if len(words) <= n:
        return [_clip(" ".join(words), width)]
    size = max(1, (len(words) + n - 1) // n)
    out: list[str] = []
    for i in range(0, len(words), size):
        piece = _clip(" ".join(words[i : i + size]), width)
        if piece:
            out.append(piece)
        if len(out) >= n:
            break
    return out


def _nodes_from(text: str, terms: list[tuple[str, str]], chips: list[str] | None = None) -> list[GraphicNode]:
    if chips:
        cleaned = [_clip(c, 18) for c in chips if str(c).strip()]
        if len(cleaned) >= 2:
            return [GraphicNode(label=c) for c in cleaned[:5]]
    if len(terms) >= 2:
        return [
            GraphicNode(label=_clip(name, 18), sub=_clip(sub, 36))
            for name, sub in terms[:4]
        ]
    parts = _chunks(text, 3, 20)
    if len(parts) >= 2:
        return [GraphicNode(label=p) for p in parts]
    if terms:
        return [GraphicNode(label=_clip(terms[0][0], 18), sub=_clip(terms[0][1], 36))]
    if parts:
        return [GraphicNode(label=parts[0])]
    return []


def _bullets_from_sentence(sentence: EditorialSentence, extra: list[str] | None = None) -> list[GraphicBullet]:
    text = stabilize_copy(sentence.text)
    terms = extract_terms(text)
    lines: list[str] = []
    for name, sub in terms[:3]:
        lines.append(_clip(f"{name}: {sub}", 44))
    lines.extend(_clip(x, 44) for x in (extra or []) if x)
    if len(lines) < 3:
        lines.extend(_chunks(text, 4, 44))
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        key = line.lower()
        if key in seen or not line:
            continue
        seen.add(key)
        unique.append(line)
    return [
        GraphicBullet(text=line, delay_ms=i * 380)
        for i, line in enumerate(unique[:5])
    ]


def _span(sentence: EditorialSentence) -> tuple[float, float]:
    start = float(sentence.start)
    end = max(float(sentence.end), start + MIN_HOLD)
    if end - start > MAX_HOLD:
        end = start + MAX_HOLD
    return start, end


def enrich_scene(beat: GraphicBeat, sentence: EditorialSentence | None = None) -> GraphicBeat:
    """Guarantee boxed nodes + several supporting lines for a full 9:16 scene."""
    text = sentence.text if sentence is not None else beat.title
    terms = extract_terms(text)
    nodes = list(beat.nodes)
    if len(nodes) < 2:
        nodes = _nodes_from(text, terms, beat.chips or None)
    if beat.kind == "vs_split":
        left = beat.left or (terms[0][0] if terms else "A")
        right = beat.right or (terms[1][0] if len(terms) > 1 else "B")
        if len(nodes) < 2:
            nodes = [
                GraphicNode(label=_clip(left, 18), sub=_clip(beat.subtitle, 36)),
                GraphicNode(label=_clip(right, 18)),
            ]
    if len(nodes) == 1 and beat.subtitle:
        nodes.append(GraphicNode(label=_clip(beat.subtitle, 18)))
    chips = beat.chips or [n.label for n in nodes]
    bullets = list(beat.bullets)
    if len(bullets) < 3 and sentence is not None:
        bullets = _bullets_from_sentence(sentence, extra=[n.sub for n in nodes if n.sub])
    elif len(bullets) < 3:
        extras = [n.sub or n.label for n in nodes]
        bullets = [
            GraphicBullet(text=_clip(x, 44), delay_ms=i * 380)
            for i, x in enumerate(extras[:5])
            if x
        ]
    return beat.model_copy(update={"nodes": nodes[:5], "chips": chips[:6], "bullets": bullets[:5]})


def _beat_for_sentence(sentence: EditorialSentence, *, hook: bool = False) -> GraphicBeat:
    text = stabilize_copy(sentence.text)
    terms = extract_terms(text)
    role = sentence.editorial_role
    start, end = _span(sentence)
    number = extract_number(text)
    kicker = "HOOK" if hook or role == "hook" else "INSIGHT"
    if role == "question" or _QUESTION.search(text):
        kicker = "HOOK" if hook else "QUESTION"
    elif role == "important_number" or number:
        kicker = "NUMBER"
    elif role in {"contrast", "contradiction"}:
        kicker = "COMPARE"
    elif role == "cta":
        kicker = "CTA"

    if number and (hook or role in {"important_number", "hook", "key_insight"}):
        beat = GraphicBeat(
            start=start,
            end=end,
            kind="stat",
            title=number,
            subtitle=_clip(text, 42),
            kicker=kicker,
            nodes=_nodes_from(text, terms),
            bullets=_bullets_from_sentence(sentence),
            motion="scale_in",
            confidence=0.8,
        )
        return enrich_scene(beat, sentence)

    if is_contrast(sentence) and len(terms) >= 2:
        beat = GraphicBeat(
            start=start,
            end=end,
            kind="vs_split",
            title=_headline(text, 6) or "Which path wins?",
            left=_clip(terms[0][0], 18),
            right=_clip(terms[1][0], 18),
            subtitle=_clip(terms[0][1], 36),
            kicker="HOOK" if hook else "COMPARE",
            nodes=[
                GraphicNode(label=_clip(terms[0][0], 18), sub=_clip(terms[0][1], 36)),
                GraphicNode(label=_clip(terms[1][0], 18), sub=_clip(terms[1][1], 36)),
            ],
            bullets=_bullets_from_sentence(sentence),
            motion="slide_up",
            confidence=0.84,
        )
        return enrich_scene(beat, sentence)

    if is_process(text) or (role in {"key_insight", "answer"} and terms):
        chips = [t[0] for t in terms[:4]]
        if len(chips) < 2:
            chips = [_clip(w, 14) for w in text.split()[:4]]
        kind = "diagram" if hook or role in {"key_insight", "answer"} else "process"
        beat = GraphicBeat(
            start=start,
            end=end,
            kind=kind,
            title=_headline(text),
            kicker="HOOK" if hook else "STEPS",
            chips=chips[:5],
            nodes=_nodes_from(text, terms, chips),
            bullets=_bullets_from_sentence(sentence),
            motion="slide_up",
            confidence=0.78,
        )
        return enrich_scene(beat, sentence)

    if role == "question" or _QUESTION.search(text):
        beat = GraphicBeat(
            start=start,
            end=end,
            kind="quote" if hook else "topic",
            title=_headline(text, 8),
            kicker=kicker,
            nodes=_nodes_from(text, terms),
            bullets=_bullets_from_sentence(sentence),
            motion="slide_up",
            confidence=0.76,
        )
        return enrich_scene(beat, sentence)

    beat = GraphicBeat(
        start=start,
        end=end,
        kind="topic" if hook else "bullets",
        title=_headline(text),
        kicker=kicker,
        nodes=_nodes_from(text, terms),
        bullets=_bullets_from_sentence(sentence),
        motion="slide_up" if not hook else "scale_in",
        confidence=0.7 if hook else 0.62,
    )
    return enrich_scene(beat, sentence)


def _no_overlap(beats: list[GraphicBeat], duration: float) -> list[GraphicBeat]:
    ordered = sorted(beats, key=lambda b: (b.start, -b.confidence))
    kept: list[GraphicBeat] = []
    last_end = -1.0
    for beat in ordered:
        start = max(beat.start, last_end + GAP)
        end = min(beat.end, duration)
        if end - start < MIN_HOLD * 0.5:
            continue
        if end - start > MAX_HOLD:
            end = start + MAX_HOLD
        kept.append(beat.model_copy(update={"start": round(start, 3), "end": round(end, 3)}))
        last_end = end
    return kept


def plan_scenes(
    analysis: EditorialAnalysis,
    *,
    video_duration: float,
) -> list[GraphicBeat]:
    sentences = list(analysis.sentences)
    if not sentences:
        return [
            GraphicBeat(
                start=0.0,
                end=min(video_duration, 5.0),
                kind="topic",
                title="The idea in one line",
                kicker="HOOK",
                nodes=[
                    GraphicNode(label="Listen", sub="the point of this clip"),
                    GraphicNode(label="Map it", sub="boxes, then the takeaway"),
                    GraphicNode(label="Remember", sub="one idea to keep"),
                ],
                bullets=[
                    GraphicBullet(text="A scene, not a caption", delay_ms=0),
                    GraphicBullet(text="Follow the speaker's idea", delay_ms=380),
                    GraphicBullet(text="Leave with one clear take", delay_ms=760),
                ],
                motion="scale_in",
                confidence=0.4,
            )
        ]
    raw: list[GraphicBeat] = []
    raw.append(_beat_for_sentence(sentences[0], hook=True))
    for sentence in sentences[1:]:
        if sentence.editorial_role in {"generic", "transition"} and not extract_terms(
            sentence.text
        ):
            continue
        raw.append(_beat_for_sentence(sentence))
    beats = _no_overlap(raw, video_duration)
    if not beats:
        beats = [_beat_for_sentence(sentences[0], hook=True)]
    return cover_duration(beats, video_duration) or beats

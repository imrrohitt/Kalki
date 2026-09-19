"""Caption director: a DeepSeek agent that reads the talk, then writes, reviews and
designs premium on-screen captions in the reference reel style.

Four passes, each JSON in / JSON out, each with one job:

1. brief    — understand the whole talk (thinking on): topic, hook, music mood,
               key ideas, glossary, and a corrected English transcript.
2. script   — per ~25 s of speech, split each corrected segment into short spoken
               caption lines. Text only; deterministic checks catch condensed,
               invented or over-long output and ask for a retry.
3. review   — read all lines in order and fix meaning and flow across splits.
4. annotate — judge every line: key word, importance weight, kind (payoff, concept,
               term, drama, quote). `craft.assign_styles` turns that into the look
               with a consistent rhythm.

Timing comes from Whisper: each segment's lines are aligned onto that segment's
timed words (`app.captions.craft.align_phrases`). Rarity, hook, density and
minimum on-screen time are enforced in `app.captions.craft`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from openai import AsyncOpenAI

from app.asr import stabilize_copy
from app.captions.craft import (
    FILLER,
    ICONS,
    LINE_KINDS,
    MOODS,
    MUSIC_MOODS,
    STYLE_NAMES,
    CaptionDraft,
    LineNote,
    align_phrases,
    assign_styles,
    drafts_to_timeline,
    enforce_design_rules,
    norm_token,
)
from app.captions.models import CaptionTimeline
from app.config import settings
from app.transcription.models import Transcript, Word

logger = logging.getLogger(__name__)

CHUNK_SECONDS = 25.0
MAX_SCREEN_WORDS = 5
SECONDS_PER_CAPTION = 1.45

BRIEF_SYSTEM = """You are the story editor of a premium Instagram Reels studio.
You receive a machine transcript of a talking-head video. The speaker may talk in
Hindi, Hinglish or English; the transcript is Whisper's English rendering and it
contains slips (wrong words, broken grammar, mistranslated phrases, misheard names).

Understand what the speaker really means, then return JSON only:
{
  "topic": "short topic",
  "summary": "2 sentences on what the viewer learns",
  "audience": "who this is for",
  "tone": "calm | warm | energetic | serious | playful",
  "music_mood": one of %(moods)s,
  "hook": "the strongest opening idea in <= 6 words, faithful to the speaker",
  "key_ideas": ["3-6 short phrases the speaker actually says that deserve design emphasis"],
  "glossary": [{"heard": "cloud", "correct": "Claude"}],
  "segments": [{"id": 0, "text": "clean, faithful English for this segment"}]
}

Rules for `segments`:
- One entry per input segment id, same ids, same order.
- Fix mishearings from context (e.g. "cloud" next to "cursor" -> "Claude",
  "rag" -> "RAG"). Fix grammar. Keep the speaker's first-person voice and wording.
- Never add claims, numbers or examples the speaker did not make.
- Keep the transcript's polarity (did / didn't, can / can't) unless the context
  makes the opposite unmistakable.
- If `reference_translation_from_creator` is present, it is the creator's own
  accurate translation: trust it over the machine transcript for meaning, while
  keeping one entry per machine segment.
""" % {"moods": " | ".join(MUSIC_MOODS)}

SCRIPT_SYSTEM = """You write the caption lines of a premium Instagram Reels edit. Lines appear
one at a time above the speaker's head, in sync with speech.

You get part of a talking-head video, one entry per speech segment: the corrected
transcript `text`, the raw machine transcript `asr` (it carries the real spoken order),
and how many seconds the segment is spoken. Split each segment into its caption lines.

Return JSON only:
{"segments": [{"id": 0, "lines": ["When I started", "giving interviews"]}]}

Rules (machine-checked; violations are sent back to you):
- Lines follow the speaker's words in spoken order. Read in sequence they say everything
  the segment says. Tighten wording and fix grammar, but never drop an idea and never add
  words, examples or claims the speaker did not say.
- Each line is ONE breath: 1-4 words (5 max). About one line per 1.2-1.5 s of speech:
  a 6 s segment gets 4-5 lines, a 20 s segment gets 14-17.
- Split at natural phrase breaks. Never leave a lonely fragment such as "and I", "of the",
  "you should"; articles and prepositions stay with their words.
- Spoken sentence case: capital only at a sentence start, for I, names (Claude, Cursor)
  and acronyms (AI, RAG, LLM). Never Title Case. No trailing full stops, no emojis.
- Drop pure filler ("so what happens is", "basically") and greetings that carry no idea.
"""

REVIEW_SYSTEM = """You are the senior copy reviewer of a premium Reels caption edit. You get the
corrected transcript, the brief, and every caption line in order with its time. Lines
render one after another above the speaker's head.

Fix meaning and flow only. Return JSON only:
{"edits": [{"i": 12, "text": "..."}], "notes": "one line"}

Check every line:
1. Meaning: it must match what is said at that moment. Fix mistranslations and invented
   words. Names and terms must be right (Claude, Cursor, RAG, AI engineer).
2. Flow: read in order, lines must read as the speaker's sentences split into natural
   spoken phrases. No words duplicated across neighbours. Fix broken grammar such as
   "will be successfully completed" -> "will go well".
3. Casing: spoken sentence case, no Title Case, no ALL CAPS except acronyms.

Only include lines you actually change. Never add, remove, merge or reorder lines. Keep
each line 1-5 words and never move words from a neighbouring line into it.
"""

ANNOTATE_SYSTEM = """You are the art director of a premium Reels caption edit. You get every
caption line in order with its time, plus the brief. Judge what each line means to the
viewer AND how the speaker is actually delivering it; the studio's design system turns
your judgement into the look (cream serif words, serif lines, hand-drawn ovals,
underlines, tape stickers) and the motion (a still rise, or a quick "blink" pop for a
real high) and the rare contextual icon.

Return JSON only:
{"lines": [{"i": 0, "key": "interviews", "weight": 1, "kind": "normal", "mood": "neutral", "icon": "none"}]}

For EVERY line:
- `key`: the single most meaningful word, or 2-word term, copied exactly from the line
  ("giving interviews" -> "interviews", "a RAG application" -> "RAG application",
  "got the role" -> "role"). "" when the line is pure connective speech ("and I think").
- `weight`:
    0 connective / filler ("when I started", "of the", "you talk about")
    1 ordinary content
    2 important: a result, contrast, or a term the viewer should notice
    3 THE key idea or payoff of its sentence; at most about one line in eight
- `kind`:
    payoff   a result or punchline ("got the role", "not a big problem")
    concept  the core idea of a beat, 1-3 words ("core depth")
    term     a concrete technical term the lesson is about ("trade-offs")
    drama    one emotionally charged word ("rejections", "mistake")
    quote    a rule of thumb or a quoted line
    normal   everything else
- `mood`: the genuine emotional charge of THIS line's delivery, judged from the words and
  where it sits in the story — not decoration, a real read of tone:
    neutral   ordinary matter-of-fact delivery — the vast majority of lines
    surprise  a twist, a shocking number, an unexpected reveal ("I got rejected 40 times")
    excited   a visible high, a win the speaker is clearly hyped about ("and it blew up")
    happy     warm, encouraging, a positive turn
    serious   a warning or a hard truth the viewer must take seriously
    urgent    time pressure, "right now" energy, a deadline
  Word-level highlighting and mood-driven motion are what make a caption edit feel alive
  instead of a static subtitle track — viewers track the movement. Use `surprise` and
  `excited` generously wherever the delivery genuinely earns it: roughly one line in every
  8-10 across the reel, not once or twice in the whole video. Every beat of a talk has SOME
  texture — a mild "huh" moment, a small win, a turn — don't reserve these for only the
  single biggest moment; a flat reel of all-neutral lines is the failure mode, not overuse.
  Still judge honestly: don't mark a line surprise/excited just because it has a number or
  an exclamation mark in the transcript, and never two in a row on the same small beat.
- `icon`: name one whenever a line is genuinely about it — these should show up
  regularly, not just once or twice:
    money    a specific amount, price, or earnings figure
    growth   scaling, results, a metric going up, momentum building
    idea     a key insight, a realization, an "aha", a tip worth remembering
    video    making/posting video content, YouTube, Reels, a content platform
    social   DMs, comments, outreach, LinkedIn/Twitter/Instagram engagement
    check    a completed step, a proof point, a "that's it" confirmation, a done task
    warning  a mistake, a risk, a red flag, "don't do this"
    time     a deadline, a duration ("in 2 minutes"), urgency around time
    target   a goal, an objective, aiming for something specific
    fire     something trending, viral, hyped, blowing up
    heart    something personal, emotional, heartfelt, about relationships or people
    star     quality, being the best, premium, a rating or achievement
    lock     privacy, security, confidentiality, keeping something secret
    question a doubt, a rhetorical question, genuine uncertainty
  Aim for roughly one icon every 6-10 lines where the topic genuinely fits one of these —
  "none" is still right when a line truly isn't about any of them, but don't default to
  "none" out of caution; a caption edit with icons scattered through it reads as designed,
  one with none at all reads as unfinished.

Judge generously but honestly: weight, mood, and icon should each reflect what the line
actually is. The goal is a caption track that feels hand-edited by someone who was paying
close attention to the whole talk — varied, textured, never flat, never random.
"""


@dataclass
class ReelBrief:
    topic: str = ""
    summary: str = ""
    audience: str = ""
    tone: str = "warm"
    music_mood: str = "warm_inspiring"
    hook: str = ""
    key_ideas: list[str] = field(default_factory=list)
    glossary: list[dict[str, str]] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "summary": self.summary,
            "audience": self.audience,
            "tone": self.tone,
            "music_mood": self.music_mood,
            "hook": self.hook,
            "key_ideas": self.key_ideas,
            "glossary": self.glossary,
            "segments": self.segments,
        }


@dataclass
class DirectorResult:
    timeline: CaptionTimeline
    brief: ReelBrief
    drafts: list[CaptionDraft]
    notes: list["LineNote"] = field(default_factory=list)
    review_notes: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def design(self) -> list[dict[str, Any]]:
        """Per line: what it says, what it means, and how it is drawn."""
        return [
            {
                "text": d.text,
                "sub": d.sub,
                "style": d.style,
                "emphasis": d.emphasis,
                "mood": d.mood,
                "icon": d.icon,
                "key": note.key,
                "weight": note.weight,
                "kind": note.kind,
            }
            for d, note in zip(self.drafts, list(self.notes) + [LineNote()] * len(self.drafts))
        ]


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def segment_words(transcript: Transcript) -> tuple[list[Word], list[tuple[int, int]]]:
    """All timed words, plus the [lo, hi) word range of every segment."""
    words: list[Word] = []
    ranges: list[tuple[int, int]] = []
    for seg in transcript.segments:
        lo = len(words)
        for w in seg.words:
            token = w.word.strip()
            if not token:
                continue
            end = float(w.end) if w.end > w.start else float(w.start) + 0.08
            words.append(
                Word(word=token, start=float(w.start), end=end, probability=w.probability)
            )
        ranges.append((lo, len(words)))
    return words, ranges


def chunk_segments(transcript: Transcript, chunk_seconds: float = CHUNK_SECONDS) -> list[list[int]]:
    chunks: list[list[int]] = []
    current: list[int] = []
    t0 = 0.0
    for i, seg in enumerate(transcript.segments):
        if not seg.words:
            continue
        if current and seg.end - t0 > chunk_seconds:
            chunks.append(current)
            current = []
        if not current:
            t0 = seg.start
        current.append(i)
    if current:
        chunks.append(current)
    return chunks


def _content(text: str) -> list[str]:
    return [t for t in (norm_token(x) for x in text.split()) if len(t) >= 3 and t not in FILLER]


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _lines_of(item: Any) -> list[str]:
    raw = item.get("lines") if isinstance(item, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for line in raw:
        text = line.get("text") if isinstance(line, dict) else line
        text = str(text or "").strip()
        if text:
            out.append(text)
    return out


def script_problems(data: dict[str, Any], segments: list[dict[str, Any]]) -> list[str]:
    """Hard errors raise; soft problems are returned for a retry."""
    got = data.get("segments")
    if not isinstance(got, list):
        raise ValueError("`segments` must be a list")
    by_id: dict[int, list[str]] = {}
    for item in got:
        try:
            by_id[int(item.get("id"))] = _lines_of(item)
        except (TypeError, ValueError, AttributeError):
            continue
    missing = [s["id"] for s in segments if not by_id.get(s["id"])]
    if missing:
        raise ValueError(f"segments {missing} have no lines; every segment id needs lines")

    problems: list[str] = []
    for seg in segments:
        lines = by_id[seg["id"]]
        sid = seg["id"]
        wanted = max(1, round(seg["seconds"] / SECONDS_PER_CAPTION))
        if seg["spoken_words"] >= 4 and len(lines) < max(1, int(wanted * 0.7)):
            problems.append(
                f"segment {sid} is spoken for {seg['seconds']}s but has {len(lines)} lines; use about {wanted}"
            )
        said = {_stem(t) for t in _content(seg["text"] + " " + seg["asr"])}
        shown = [_stem(t) for t in _content(" ".join(lines))]
        source = [_stem(t) for t in _content(seg["text"])]
        dropped = [t for t in dict.fromkeys(source) if t not in set(shown)]
        if source and len(dropped) > max(2, int(len(set(source)) * 0.3)):
            problems.append(f"segment {sid} drops spoken words: {', '.join(dropped[:8])}")
        invented = [t for t in dict.fromkeys(shown) if t not in said]
        # Grammar fixes add small words; a new long word is usually a new idea.
        if len(invented) > max(1, int(len(set(shown)) * 0.2)) or any(len(t) >= 7 for t in invented):
            problems.append(f"segment {sid} adds words the speaker did not say: {', '.join(invented[:6])}")
        for line in lines:
            if len(line.split()) > MAX_SCREEN_WORDS:
                problems.append(f'segment {sid} line "{line}" has {len(line.split())} words (max {MAX_SCREEN_WORDS})')
    return problems


def annotation_problems(data: dict[str, Any], lines: list[str]) -> list[str]:
    items = data.get("lines")
    if not isinstance(items, list):
        raise ValueError("`lines` must be a list")
    seen: dict[int, dict[str, Any]] = {}
    for item in items:
        try:
            seen[int(item.get("i"))] = item
        except (TypeError, ValueError, AttributeError):
            continue
    missing = [i for i in range(len(lines)) if i not in seen]
    if len(missing) > len(lines) // 10:
        raise ValueError(f"lines {missing[:20]} are missing; annotate every line")
    problems: list[str] = []
    for i, item in seen.items():
        if not 0 <= i < len(lines):
            continue
        key = str(item.get("key") or "").strip()
        if key and norm_token(key) not in norm_token(lines[i]):
            problems.append(f'line {i} "{lines[i]}": key "{key}" is not in the line')
    heavy = sum(1 for item in seen.values() if _as_int(item.get("weight")) >= 3)
    if heavy > max(2, len(lines) // 6):
        problems.append(f"{heavy} lines have weight 3; keep weight 3 for about one line in eight")

    popped = sum(1 for item in seen.values() if str(item.get("mood") or "neutral") in {"surprise", "excited"})
    if popped > max(4, len(lines) // 5):
        problems.append(
            f"{popped} lines are marked surprise/excited; that is too many to land as highlights — "
            "keep it to roughly one in every 8-10 lines"
        )
    elif len(lines) >= 15 and popped == 0:
        problems.append(
            "every single line is mood=neutral; a real talk has SOME texture — find the lines with "
            "genuine surprise, a small win, or a turn and mark them surprise/excited (roughly one in 8-10)"
        )
    iconed = sum(1 for item in seen.values() if str(item.get("icon") or "none") != "none")
    if iconed > max(6, len(lines) // 4):
        problems.append(
            f"{iconed} lines have an icon; that reads as clutter — keep it to roughly one in every 6-10 lines"
        )
    elif len(lines) >= 15 and iconed == 0:
        problems.append(
            "every single line has icon=none; look again for lines genuinely about money, growth, an "
            "idea, video/social platforms, a warning, time, a goal, hype, something personal, quality, "
            "privacy, or a doubt, and give roughly one in every 6-10 lines a matching icon"
        )
    return problems[:15]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _edit_is_safe(before: str, after: str, lines: list[str], i: int) -> bool:
    """Reviewer may polish a line, not grow it or copy its neighbours' words."""
    if not after or after == before:
        return False
    if len(after.split()) > MAX_SCREEN_WORDS or len(after.split()) > len(before.split()) + 2:
        return False
    new = set(_content(after)) - set(_content(before))
    for j in (i - 1, i + 1):
        if new and 0 <= j < len(lines):
            neighbour = set(_content(lines[j]))
            if new & neighbour:
                return False
    return True


class CaptionDirector:
    def __init__(self, *, client: AsyncOpenAI | None = None, model: str | None = None) -> None:
        self.model = model or settings.director_model
        self.client = client or AsyncOpenAI(
            api_key=settings.deepseek_api_key or "missing",
            base_url=settings.deepseek_base_url,
            timeout=300,
            max_retries=3,
        )

    async def _ask(
        self,
        system: str,
        user: dict[str, Any],
        *,
        label: str,
        max_tokens: int = 8000,
        think: bool = False,
        attempts: int = 3,
        validate: Callable[[dict[str, Any]], list[str] | None] | None = None,
    ) -> dict[str, Any]:
        """JSON call with validation feedback. `validate` raises for unusable
        output and returns soft problems, which are retried while attempts remain."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            t0 = time.perf_counter()
            text = ""
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=32000 if think else max_tokens,
                    temperature=0.4,
                    extra_body={"thinking": {"type": "enabled" if think else "disabled"}},
                )
                text = resp.choices[0].message.content or ""
                data = _extract_json(text)
                soft = validate(data) if validate is not None else None
                if soft and attempt < attempts:
                    raise ValueError("; ".join(soft))
                logger.info(
                    "director %s ok (attempt %s, %.1fs%s)",
                    label,
                    attempt,
                    time.perf_counter() - t0,
                    f", {len(soft)} minor issues kept" if soft else "",
                )
                return data
            except Exception as exc:  # noqa: BLE001 - retried with feedback
                last_error = exc
                logger.warning("director %s attempt %s: %s", label, attempt, str(exc)[:600])
                if attempt < attempts and text:
                    messages = messages[:2] + [
                        {"role": "assistant", "content": text[:16000]},
                        {
                            "role": "user",
                            "content": f"Fix these problems and return the full corrected JSON only: {exc}",
                        },
                    ]
                elif attempt < attempts:
                    await asyncio.sleep(1.5 * attempt)
        raise RuntimeError(f"director {label} failed: {last_error}")

    # ------------------------------------------------------------------ brief
    async def brief(self, transcript: Transcript, *, reference_text: str = "") -> ReelBrief:
        segments = [
            {"id": i, "start": round(s.start, 1), "text": stabilize_copy(s.text)}
            for i, s in enumerate(transcript.segments)
        ]
        payload: dict[str, Any] = {
            "spoken_language": transcript.spoken_language or transcript.language,
            "segments": segments,
        }
        if reference_text.strip():
            payload["reference_translation_from_creator"] = reference_text.strip()[:12000]

        def validate(data: dict[str, Any]) -> None:
            got = data.get("segments")
            if not isinstance(got, list) or len(got) != len(segments):
                raise ValueError(
                    f"segments must have exactly {len(segments)} entries, got "
                    f"{len(got) if isinstance(got, list) else 'none'}"
                )
            return None

        data = await self._ask(BRIEF_SYSTEM, payload, label="brief", think=True, validate=validate)
        by_id: dict[int, str] = {}
        for item in data.get("segments") or []:
            try:
                by_id[int(item.get("id"))] = str(item.get("text") or "").strip()
            except (TypeError, ValueError, AttributeError):
                continue
        mood = str(data.get("music_mood") or "").strip()
        return ReelBrief(
            topic=str(data.get("topic") or ""),
            summary=str(data.get("summary") or ""),
            audience=str(data.get("audience") or ""),
            tone=str(data.get("tone") or "warm"),
            music_mood=mood if mood in MUSIC_MOODS else "warm_inspiring",
            hook=str(data.get("hook") or ""),
            key_ideas=[str(x) for x in (data.get("key_ideas") or []) if str(x).strip()][:8],
            glossary=[
                {"heard": str(g.get("heard", "")), "correct": str(g.get("correct", ""))}
                for g in (data.get("glossary") or [])
                if isinstance(g, dict)
            ][:30],
            segments=[by_id.get(i) or stabilize_copy(s.text) for i, s in enumerate(transcript.segments)],
        )

    # ----------------------------------------------------------------- script
    async def script_chunk(
        self, transcript: Transcript, seg_ids: list[int], brief: ReelBrief
    ) -> dict[int, list[str]]:
        segs = transcript.segments
        segments = [
            {
                "id": i,
                "seconds": round(segs[i].end - segs[i].start, 1),
                "spoken_words": len(segs[i].words),
                "asr": stabilize_copy(segs[i].text),
                "text": brief.segments[i],
            }
            for i in seg_ids
        ]
        payload = {
            "brief": {"topic": brief.topic, "glossary": brief.glossary},
            "segments": segments,
        }
        data = await self._ask(
            SCRIPT_SYSTEM,
            payload,
            label=f"script[s{seg_ids[0]}-s{seg_ids[-1]}]",
            validate=lambda d: script_problems(d, segments),
        )
        out: dict[int, list[str]] = {}
        for item in data.get("segments") or []:
            try:
                sid = int(item.get("id"))
            except (TypeError, ValueError, AttributeError):
                continue
            if sid in seg_ids:
                out[sid] = _lines_of(item)
        return out

    # ----------------------------------------------------------------- review
    async def review(
        self, drafts: list[CaptionDraft], words: list[Word], brief: ReelBrief
    ) -> tuple[list[CaptionDraft], str]:
        lines = [d.text for d in drafts]
        payload = {
            "brief": {"topic": brief.topic, "glossary": brief.glossary},
            "corrected_transcript": " ".join(brief.segments),
            "lines": [
                {"i": i, "t": round(words[d.first].start, 1), "text": d.text}
                for i, d in enumerate(drafts)
            ],
        }
        data = await self._ask(REVIEW_SYSTEM, payload, label="review", attempts=2)
        out = list(drafts)
        changed = 0
        for edit in data.get("edits") or []:
            try:
                i = int(edit.get("i"))
            except (TypeError, ValueError, AttributeError):
                continue
            if not 0 <= i < len(out):
                continue
            after = str(edit.get("text") or "").strip()
            if _edit_is_safe(out[i].text, after, lines, i):
                out[i] = replace(out[i], text=after)
                changed += 1
        return out, f"{changed} copy edits. {data.get('notes') or ''}".strip()

    # --------------------------------------------------------------- annotate
    async def annotate(
        self, drafts: list[CaptionDraft], words: list[Word], brief: ReelBrief
    ) -> list[LineNote]:
        lines = [d.text for d in drafts]
        payload = {
            "brief": {"topic": brief.topic, "hook": brief.hook, "key_ideas": brief.key_ideas},
            "lines": [
                {"i": i, "t": round(words[d.first].start, 1), "text": d.text}
                for i, d in enumerate(drafts)
            ],
        }
        data = await self._ask(
            ANNOTATE_SYSTEM,
            payload,
            label="annotate",
            max_tokens=12000,
            validate=lambda d: annotation_problems(d, lines),
        )
        notes = [LineNote() for _ in drafts]
        for item in data.get("lines") or []:
            i = _as_int(item.get("i") if isinstance(item, dict) else None, -1)
            if not 0 <= i < len(notes):
                continue
            kind = str(item.get("kind") or "normal").strip().lower()
            mood = str(item.get("mood") or "neutral").strip().lower()
            icon = str(item.get("icon") or "none").strip().lower()
            notes[i] = LineNote(
                key=str(item.get("key") or "").strip(),
                weight=max(0, min(3, _as_int(item.get("weight"), 1))),
                kind=kind if kind in LINE_KINDS else "normal",
                mood=mood if mood in MOODS else "neutral",
                icon=icon if icon in ICONS else "none",
            )
        moody = sum(1 for note in notes if note.mood != "neutral")
        iconed = sum(1 for note in notes if note.icon != "none")
        logger.info("director annotate: %s lines with a mood, %s with an icon", moody, iconed)
        return notes

    # ------------------------------------------------------------------- run
    async def direct(
        self,
        transcript: Transcript,
        *,
        video_duration: float,
        job_id: str = "director",
        reference_text: str = "",
        audio_path: str | None = None,
    ) -> DirectorResult:
        """`audio_path` (16kHz mono PCM wav) is optional but recommended: it is
        the one signal in this whole pipeline read from the actual voice
        rather than the transcript — a real loudness spike becomes a "blink"
        pop or a colored highlight even on a line the text alone reads flat.
        """
        jid = job_id[:8]
        words, seg_ranges = segment_words(transcript)
        if not words:
            raise RuntimeError("No timed words in transcript")
        metrics: dict[str, Any] = {}

        loud_times = loud_db = None
        if audio_path:
            from app.captions.acoustics import try_voice_loudness_track

            loud_times, loud_db = try_voice_loudness_track(audio_path)
            if loud_times is not None:
                logger.info("[%s] voice loudness track: %s samples", jid, len(loud_times))

        t0 = time.perf_counter()
        brief = await self.brief(transcript, reference_text=reference_text)
        metrics["brief_ms"] = int((time.perf_counter() - t0) * 1000)
        logger.info("[%s] director brief: %r mood=%s hook=%r", jid, brief.topic, brief.music_mood, brief.hook)

        t1 = time.perf_counter()
        chunks = chunk_segments(transcript)
        sem = asyncio.Semaphore(4)

        async def run_chunk(seg_ids: list[int]) -> dict[int, list[str]]:
            async with sem:
                return await self.script_chunk(transcript, seg_ids, brief)

        script: dict[int, list[str]] = {}
        for part in await asyncio.gather(*[run_chunk(ids) for ids in chunks]):
            script.update(part)

        drafts: list[CaptionDraft] = []
        for sid, (lo, hi) in enumerate(seg_ranges):
            if hi <= lo:
                continue
            lines = script.get(sid) or [brief.segments[sid] or transcript.segments[sid].text]
            prev_first = -1
            for (first, last), text in zip(align_phrases(lines, words[lo:hi]), lines):
                if first == prev_first and drafts:
                    # More lines than spoken words: fold into the previous line.
                    prev = drafts[-1]
                    drafts[-1] = replace(prev, last=lo + last, text=f"{prev.text} {text}")
                    continue
                prev_first = first
                drafts.append(CaptionDraft(first=lo + first, last=lo + last, text=text))
        metrics["script_ms"] = int((time.perf_counter() - t1) * 1000)
        logger.info("[%s] director scripted %s lines in %s chunk(s)", jid, len(drafts), len(chunks))

        notes = ""
        t2 = time.perf_counter()
        try:
            drafts, notes = await self.review(drafts, words, brief)
            logger.info("[%s] director review: %s", jid, notes)
        except Exception as exc:  # noqa: BLE001 - review is an improvement pass
            logger.warning("[%s] director review skipped: %s", jid, exc)
        metrics["review_ms"] = int((time.perf_counter() - t2) * 1000)

        t3 = time.perf_counter()
        try:
            line_notes = await self.annotate(drafts, words, brief)
        except Exception as exc:  # noqa: BLE001 - craft still designs from heuristics
            logger.warning("[%s] director annotate failed, heuristic weights: %s", jid, exc)
            line_notes = [LineNote() for _ in drafts]
        drafts = assign_styles(drafts, line_notes, words, loud_times=loud_times, loud_db=loud_db)
        metrics["design_ms"] = int((time.perf_counter() - t3) * 1000)

        drafts = enforce_design_rules(drafts, words)
        logger.info(
            "[%s] mood/icon after rarity gating: %s pop, %s icon",
            jid,
            sum(1 for d in drafts if d.mood != "neutral"),
            sum(1 for d in drafts if d.icon != "none"),
        )
        timeline = drafts_to_timeline(drafts, words, video_duration=video_duration)
        return DirectorResult(
            timeline=timeline,
            brief=brief,
            drafts=drafts,
            notes=line_notes,
            review_notes=notes,
            metrics=metrics,
        )

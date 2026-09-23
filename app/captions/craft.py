"""Deterministic caption craft: coverage, timing, word reveal, and design restraint.

The director agent decides *what* a caption says and how it should look. This
module makes sure the result is always renderable and always tasteful: every
spoken word is covered once, captions follow the voice, special treatments stay
rare, and word reveal times come from the real ASR timings.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from app.asr import stabilize_copy
from app.captions.models import Caption, CaptionTimeline, CaptionWord
from app.transcription.models import Word

STYLE_NAMES = ("plain", "mix", "serif", "stack", "quote", "oval", "underline", "tape", "chip", "bubble")
MUSIC_MOODS = (
    "warm_inspiring",
    "focused_tech",
    "calm_reflective",
    "confident_upbeat",
    "serious_story",
)

# Seconds between two captions of the same special style.
STYLE_MIN_GAP = {
    "oval": 18.0,
    "underline": 14.0,
    "tape": 50.0,
    "quote": 12.0,
    "stack": 25.0,
    "serif": 4.5,
    "chip": 11.0,
    "bubble": 18.0,
}
# Big cream moments never sit back to back.
BIG_STYLES = {"serif", "quote", "oval", "tape", "stack", "chip", "bubble"}
BIG_MIN_GAP = 2.4
MAX_WORDS = {
    "plain": 5,
    "mix": 5,
    "serif": 4,
    "stack": 4,
    "quote": 5,
    "oval": 3,
    "underline": 4,
    "tape": 3,
    "chip": 4,
    "bubble": 4,
}
MAX_CAPTION_SECONDS = 3.4

# A line naming a concrete figure (money, percent, a round number) gets the
# premium colored pill instead of plain white sans — it is the number a
# scroller remembers.
MONEY_RE = re.compile(
    r"(\$|₹|€|£)\s?\d|\d[\d,]*\s?(lakh|crore|million|billion|percent|%)\b",
    re.I,
)

ACRONYMS = {
    "AI", "RAG", "LLM", "LLMS", "API", "APIS", "GPU", "CPU", "SQL", "UI", "UX", "ML",
    "CEO", "CTO", "IT", "AWS", "GCP", "GDPR", "PEFT", "LORA", "NLP", "SDK", "OK",
    "HR", "DSA", "SDE", "CV", "PR", "US", "USA", "UK", "PDF", "JSON", "HTTP",
    "SEO", "SEM", "CRM", "ROI", "KPI", "B2B", "B2C", "UPI", "DM", "DMS", "CTA",
    "SaaS", "MVP", "QA", "OS", "PC", "TV", "FAQ", "URL", "RAM", "SSD",
}
FILLER = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "to", "of", "in", "on",
    "at", "for", "and", "or", "but", "so", "that", "this", "it", "its", "i", "you",
    "we", "they", "he", "she", "my", "your", "very", "really", "just", "thing",
    "things", "like", "basically", "actually", "then", "there", "here", "what",
    "how", "all", "about", "with", "do", "did", "does", "have", "has", "had",
}
GREETINGS = {"hi", "hello", "hey", "namaste", "guys", "everyone", "friends"}
_WORD_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class CaptionDraft:
    first: int
    last: int
    text: str
    sub: str = ""
    style: str = "plain"
    emphasis: str = ""
    # AI-judged context, not a look: mood drives the reveal motion (a "blink"
    # pop for a surprise or an excited beat); icon is a small contextual badge
    # (money, growth, a key idea, a platform mention...). Both come from the
    # director's annotate pass — see LineNote — and stay "neutral"/"" for the
    # vast majority of lines.
    mood: str = "neutral"
    icon: str = "none"
    cta: bool = False

    @property
    def tokens(self) -> list[str]:
        return self.text.split()

    @property
    def sub_tokens(self) -> list[str]:
        return self.sub.split()


def norm_token(token: str) -> str:
    return _WORD_RE.sub("", token.lower())


def fix_case(text: str) -> str:
    """Spoken sentence case guard: no shouting, no Title Case.

    A short all-caps token (<=5 letters: WHO, MCA, ROI, API...) is almost
    always a real acronym the director capitalized on purpose, even when it
    is not in the fixed ACRONYMS list — the list can't cover every proper
    noun or industry term. Only a longer all-caps word is fixed as likely
    accidental shouting.
    """
    tokens = text.split()
    if not tokens:
        return text
    out: list[str] = []
    for tok in tokens:
        core = re.sub(r"[^A-Za-z]", "", tok)
        if len(core) > 5 and core.isupper() and core.upper() not in ACRONYMS:
            tok = tok.lower()
        out.append(tok)
    alpha = [re.sub(r"[^A-Za-z]", "", t) for t in out]
    # "I", acronyms, and short all-caps tokens are capitalized on purpose;
    # judge only the ordinary words for a Title Case run.
    is_acronym_like = lambda a: a.upper() in ACRONYMS or (len(a) <= 5 and a.isupper())
    judged = [a for a in alpha if a and not is_acronym_like(a) and a != "I"]
    titled = [a for a in judged if a[0].isupper()]
    if len(out) >= 3 and len(judged) >= 2 and len(titled) == len(judged):
        out = [out[0]] + [
            t
            if is_acronym_like(re.sub(r"[^A-Za-z]", "", t))
            or t in {"I", "I'm", "I've", "I'll", "I'd"}
            else t.lower()
            for t in out[1:]
        ]
    return " ".join(out)


def clean_copy(text: str) -> str:
    text = stabilize_copy(text.replace("\n", " "))
    text = text.strip().strip("\"“”")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.;:\u2014\u2013-]+$", "", text).strip()
    return fix_case(text)


def _content_word(tokens: list[str]) -> str:
    best = ""
    for tok in tokens:
        core = norm_token(tok)
        if core and core not in FILLER and len(core) >= len(norm_token(best)):
            best = tok.strip(",?!")
    return best


def _emphasis_ok(text: str, emphasis: str) -> bool:
    if not emphasis:
        return False
    tokens = [norm_token(t) for t in text.split()]
    wanted = [norm_token(t) for t in emphasis.split() if norm_token(t)]
    if not wanted or len(wanted) > 2:
        return False
    if all(w in FILLER for w in wanted):
        return False
    for i in range(len(tokens) - len(wanted) + 1):
        if tokens[i : i + len(wanted)] == wanted:
            return True
    return False


def _downgrade(d: CaptionDraft) -> CaptionDraft:
    """One step toward plain, keeping the line's best word in serif."""
    if d.style == "oval":
        style = "serif" if len(d.tokens) <= MAX_WORDS["serif"] else "mix"
        return replace(d, style=style)
    if d.style in {"tape", "quote", "chip", "bubble"}:
        return replace(d, style="serif")
    if d.style == "stack":
        merged = f"{d.text} {d.sub}".strip()
        return replace(d, text=merged, sub="", style="mix", emphasis=_content_word(d.tokens))
    if d.style in {"serif", "underline"}:
        emphasis = d.emphasis if _emphasis_ok(d.text, d.emphasis) else _content_word(d.tokens)
        return replace(d, style="mix" if emphasis else "plain", emphasis=emphasis)
    return replace(d, style="plain", emphasis="")


def _normalize_coverage(drafts: list[CaptionDraft], n_words: int) -> list[CaptionDraft]:
    ordered = sorted((d for d in drafts if d.text.strip()), key=lambda d: (d.first, d.last))
    fixed: list[CaptionDraft] = []
    cursor = 0
    for d in ordered:
        first = max(d.first, cursor)
        last = min(max(d.last, first), n_words - 1)
        if first > n_words - 1 or last < first:
            continue
        if fixed and first > cursor:
            # Uncovered ids between groups belong to the previous caption.
            fixed[-1] = replace(fixed[-1], last=first - 1)
        elif not fixed and first > 0:
            first = 0
        fixed.append(replace(d, first=first, last=last))
        cursor = last + 1
    if fixed and cursor < n_words:
        fixed[-1] = replace(fixed[-1], last=n_words - 1)
    return fixed


def _split_long(d: CaptionDraft, words: list[Word]) -> list[CaptionDraft]:
    span = words[d.last].end - words[d.first].start
    tokens = d.tokens
    if d.style == "stack" or len(tokens) < 4 or d.last == d.first:
        return [d]
    if span <= MAX_CAPTION_SECONDS and len(tokens) <= MAX_WORDS.get(d.style, 5):
        return [d]
    half = len(tokens) // 2
    mid_t = words[d.first].start + span / 2
    cut = d.first
    for i in range(d.first, d.last):
        if words[i + 1].start <= mid_t:
            cut = i + 1
    cut = min(max(cut, d.first + 1), d.last)
    left_text = " ".join(tokens[:half])
    right_text = " ".join(tokens[half:])

    def part(first: int, last: int, text: str, style: str, *, carry: bool) -> CaptionDraft:
        emphasis = d.emphasis if _emphasis_ok(text, d.emphasis) else ""
        if style == "mix" and not emphasis:
            style = "plain"
        return CaptionDraft(
            first=first,
            last=last,
            text=text,
            style=style,
            emphasis=emphasis,
            # A split line keeps its mood/icon on the first half only — a
            # badge or a pop on both fragments of one broken sentence reads
            # as a glitch, not a highlight.
            mood=d.mood if carry else "neutral",
            icon=d.icon if carry else "none",
            cta=d.cta if carry else False,
        )

    if d.style in BIG_STYLES:
        # The half that carries the content keeps the serif; the other reads plain.
        left_weight = len([t for t in tokens[:half] if norm_token(t) not in FILLER])
        right_weight = len([t for t in tokens[half:] if norm_token(t) not in FILLER])
        left_style, right_style = ("serif", "plain") if left_weight > right_weight else ("plain", "serif")
    elif d.style in {"mix", "underline"}:
        left_style = right_style = "mix"
    else:
        left_style = right_style = "plain"
    left = part(d.first, cut - 1, left_text, left_style, carry=True)
    right = part(cut, d.last, right_text, right_style, carry=False)
    return _split_long(left, words) + _split_long(right, words)


def _hook_index(
    drafts: list[CaptionDraft], words: list[Word], notes: list[LineNote] | None = None
) -> int:
    """The strongest line of the first 2.5 s (designed lines win)."""
    if not drafts:
        return -1
    t0 = words[drafts[0].first].start
    window = [i for i, d in enumerate(drafts[:3]) if words[d.first].start - t0 <= 2.5] or [0]
    for i in window:
        if drafts[i].style in {"serif", "stack", "quote", "oval", "tape"}:
            return i

    def content(i: int) -> int:
        return len([
            t for t in drafts[i].tokens
            if len(norm_token(t)) >= 3 and norm_token(t) not in FILLER | GREETINGS
        ])

    # The opener stays the hook unless it is only a greeting or filler.
    for i in window:
        if content(i) > 0:
            return i
    return window[0]


def enforce_design_rules(drafts: list[CaptionDraft], words: list[Word]) -> list[CaptionDraft]:
    if not words:
        return []
    cleaned: list[CaptionDraft] = []
    for d in drafts:
        style = d.style if d.style in STYLE_NAMES else "plain"
        text = clean_copy(d.text)
        sub = clean_copy(d.sub) if d.sub else ""
        if not text:
            continue
        if style == "stack" and not sub:
            style = "serif"
        if style != "stack":
            sub = ""
        emphasis = clean_copy(d.emphasis).strip(",?!") if d.emphasis else ""
        cleaned.append(
            CaptionDraft(
                first=d.first, last=d.last, text=text, sub=sub, style=style, emphasis=emphasis,
                mood=d.mood, icon=d.icon, cta=d.cta,
            )
        )

    covered = _normalize_coverage(cleaned, len(words))
    split: list[CaptionDraft] = []
    for d in covered:
        split.extend(_split_long(d, words))

    hook_index = _hook_index(split, words)
    out: list[CaptionDraft] = []
    last_style_at: dict[str, float] = {}
    last_big_at = -99.0
    decorated_terms: set[str] = set()
    for index, d in enumerate(split):
        t = words[d.first].start
        # Length limits per look.
        while len(d.tokens) > MAX_WORDS.get(d.style, 5) and d.style != "plain":
            d = _downgrade(d)
        if d.style in {"mix", "underline"} and not _emphasis_ok(d.text, d.emphasis):
            if d.style == "mix":
                d = replace(d, style="plain", emphasis="")
            else:
                d = replace(d, emphasis="")
        if d.style == "plain" and d.emphasis:
            d = replace(d, emphasis="")
        # Hook: the strongest line of the first seconds is a designed moment.
        if index == hook_index and d.style in {"plain", "mix"} and len(d.tokens) <= MAX_WORDS["serif"]:
            d = replace(d, style="serif", emphasis="")
        # Rarity.
        for _ in range(4):
            gap = STYLE_MIN_GAP.get(d.style)
            too_soon = gap is not None and t - last_style_at.get(d.style, -99.0) < gap
            big_clash = d.style in BIG_STYLES and t - last_big_at < BIG_MIN_GAP and index > 0
            if not (too_soon or big_clash):
                break
            d = _downgrade(d)
        # A hand-drawn accent on the same term twice looks templated.
        if d.style in {"oval", "underline", "tape"}:
            term = norm_token(d.emphasis or d.text)
            if term in decorated_terms:
                d = _downgrade(d)
            else:
                decorated_terms.add(term)
        if d.style in STYLE_MIN_GAP:
            last_style_at[d.style] = t
        if d.style in BIG_STYLES:
            last_big_at = t
        out.append(d)
    return _merge_flashes(_lift_plain_runs(out), words)


def _lift_plain_runs(drafts: list[CaptionDraft], max_run: int = 3) -> list[CaptionDraft]:
    """Floor for the look: a long run of plain lines gets one cream word."""
    out = list(drafts)
    run: list[int] = []
    for i, d in enumerate(out + [CaptionDraft(first=0, last=0, text="_", style="mix")]):
        if d.style == "plain" and i < len(out):
            run.append(i)
            continue
        while len(run) > max_run:
            window = run[: max_run + 1]
            best = max(window, key=lambda k: len(norm_token(_content_word(out[k].tokens))))
            word = _content_word(out[best].tokens)
            if len(norm_token(word)) >= 4:
                out[best] = replace(out[best], style="mix", emphasis=word)
                run = run[run.index(best) + 1 :]
            else:
                run = run[max_run + 1 :]
        run = []
    return out


def _merge_flashes(drafts: list[CaptionDraft], words: list[Word], floor: float = 0.5) -> list[CaptionDraft]:
    """Fold a line the voice rushes through into its neighbour when both still fit."""
    out = list(drafts)
    i = 0
    while i < len(out):
        d = out[i]
        span = words[d.last].end - words[d.first].start
        if span >= floor or d.style == "stack" or len(out) == 1:
            i += 1
            continue
        candidates = []
        if i + 1 < len(out) and out[i + 1].style != "stack":
            candidates.append(i + 1)
        if i > 0 and out[i - 1].style != "stack":
            candidates.append(i - 1)
        merged = False
        for j in candidates:
            a, b = (out[i], out[j]) if j > i else (out[j], out[i])
            tokens = a.tokens + b.tokens
            starts_sentence = b.tokens and b.tokens[0][:1].isupper() and norm_token(b.tokens[0]) != "i"
            if a.text.rstrip().endswith((",", ".", "?", "!")) or starts_sentence:
                continue
            if len(tokens) > MAX_WORDS["mix"]:
                continue
            keep = a if a.style != "plain" else b
            style = keep.style if keep.style in {"plain", "mix", "serif"} else "mix"
            emphasis = keep.emphasis if style == "mix" else ""
            if style == "serif" and len(tokens) > MAX_WORDS["serif"]:
                style, emphasis = "mix", _content_word(keep.tokens)
            if style == "mix" and not _emphasis_ok(" ".join(tokens), emphasis):
                emphasis = _content_word(tokens)
                style = "mix" if emphasis else "plain"
            lo, hi = min(i, j), max(i, j)
            out[lo : hi + 1] = [
                CaptionDraft(
                    first=a.first,
                    last=b.last,
                    text=" ".join(tokens),
                    style=style,
                    emphasis=emphasis,
                    mood=keep.mood,
                    icon=keep.icon,
                    cta=keep.cta,
                )
            ]
            merged = True
            i = lo
            break
        if not merged:
            i += 1
    return out


LINE_KINDS = ("payoff", "concept", "term", "drama", "quote", "normal")
# Genuine emotional beats the delivery calls for. Most lines are "neutral" —
# these exist so a real surprise or a real high can move, not every line.
MOODS = ("neutral", "surprise", "excited", "happy", "serious", "urgent")
MOODS_THAT_POP = {"surprise", "excited"}
# A vocabulary of contextual badges covering the ground a creator/business talk
# actually covers. "none" is still the right answer for most lines — these are
# accents, not decoration on every line.
ICONS = (
    "none", "money", "growth", "idea", "video", "social", "check",
    "warning", "time", "target", "fire", "heart", "star", "lock", "question",
)
# Minimum seconds between two pops / two icon badges, independent of caption
# style — these are motion and iconography, not the cream-word rarity above.
POP_MIN_GAP = 7.0
ICON_MIN_GAP = 13.0
# A line the speaker's own voice gets noticeably louder on — real prosodic
# emphasis, not a text guess — is at least as strong a "highlight this" signal
# as anything the LLM reads off the words alone.
VOCAL_EMPHASIS_MIN_GAP = 5.0
VOCAL_EMPHASIS_MAX_FRACTION = 0.18
# The opening stretch that decides whether someone keeps watching. It gets
# first claim on rare accents at a slightly lower bar than the rest of the
# reel — a richer, more varied hook, never invented onto filler.
HOOK_WINDOW_SECONDS = 30.0


@dataclass(frozen=True)
class LineNote:
    """The director's judgement of one line: what it says, what it means, and
    the context signals — mood, icon, cta — that the annotate prompt reads
    straight off the speaker's words and delivery."""

    key: str = ""
    weight: int = 1
    kind: str = "normal"
    mood: str = "neutral"
    icon: str = "none"
    cta: bool = False


def _strip_greeting(text: str) -> str:
    tokens = text.split()
    while len(tokens) > 2 and norm_token(tokens[0]) in GREETINGS:
        tokens = tokens[1:]
    if tokens and tokens != text.split():
        tokens[0] = tokens[0][:1].upper() + tokens[0][1:]
    return " ".join(tokens)


def _key_for(text: str, key: str) -> str:
    """The emphasis that fits the line: the key, or its strongest 1-2 words."""
    if _emphasis_ok(text, key):
        return key
    parts = [t for t in key.split() if norm_token(t) and norm_token(t) not in FILLER]
    for size in (2, 1):
        for k in range(len(parts) - size, -1, -1):
            candidate = " ".join(parts[k : k + size])
            if _emphasis_ok(text, candidate):
                return candidate
    return ""


def assign_styles(
    drafts: list[CaptionDraft],
    notes: list[LineNote],
    words: list[Word],
    *,
    loud_times=None,
    loud_db=None,
    caption_style: str = "classic",
) -> list[CaptionDraft]:
    """Turn line judgements into the reference look with a steady rhythm.

    Placed in priority order so the strongest moments win the spacing: the hook,
    then hand-drawn accents on weight-3 ideas, then serif payoffs, then cream serif
    words on meaningful lines. Everything else stays plain white sans.

    `loud_times`/`loud_db` are the optional voice-loudness track from
    `app.captions.acoustics` — the one signal here that comes from the actual
    audio rather than the transcript. A line the speaker's own voice gets
    genuinely louder on is treated as real emphasis alongside the LLM's mood.
    """
    if not drafts:
        return []
    notes = list(notes) + [LineNote()] * max(0, len(drafts) - len(notes))
    n = len(drafts)
    # Editorial is a deliberately undecorated look — plain/mix/serif/stack and
    # a rare CTA bubble, no hand-drawn oval/underline/tape/quote or colored
    # chip. Those stay exclusive to classic/premium.
    decorative_enabled = caption_style != "editorial"
    texts = [_strip_greeting(d.text) if i <= 1 else d.text for i, d in enumerate(drafts)]
    times = [words[d.first].start for d in drafts]
    styles = ["plain"] * n
    emphasis = [""] * n
    stacked: dict[int, int] = {}  # hook index -> absorbed next index

    loud = [False] * n
    if loud_times is not None and loud_db is not None and len(loud_times):
        from app.captions.acoustics import flag_vocal_emphasis

        spans = [(words[d.first].start, words[d.last].end) for d in drafts]
        loud = flag_vocal_emphasis(
            spans,
            loud_times,
            loud_db,
            min_gap_s=VOCAL_EMPHASIS_MIN_GAP,
            max_fraction=VOCAL_EMPHASIS_MAX_FRACTION,
        )

    def clear(i: int, style: str) -> bool:
        gap = STYLE_MIN_GAP.get(style, 0.0)
        for j in range(n):
            if j == i or styles[j] == "plain":
                continue
            dt = abs(times[i] - times[j])
            if styles[j] == style and dt < gap:
                return False
            designed = {*BIG_STYLES, "underline"}
            if style in designed and styles[j] in designed and dt < BIG_MIN_GAP + 0.6:
                return False
        return True

    # 1. Hook.
    hook = _hook_index(drafts, words, notes)
    if 0 <= hook < n:
        n_words = len(texts[hook].split())
        if (
            hook + 1 < n
            and n_words <= 3
            and len(texts[hook + 1].split()) <= 4
            and times[hook + 1] - times[hook] <= 1.8
            and not texts[hook].rstrip().endswith((".", "?", "!"))
        ):
            styles[hook] = "stack"
            stacked[hook] = hook + 1
        elif n_words <= MAX_WORDS["serif"]:
            styles[hook] = "serif"
        else:
            key = _key_for(texts[hook], notes[hook].key) or _content_word(texts[hook].split())
            styles[hook], emphasis[hook] = ("mix", key) if key else ("plain", "")

    def free(i: int) -> bool:
        return styles[i] == "plain" and i not in stacked.values()

    # 2. Premium colored chip: a concrete figure (money/percent/round number)
    #    a scroller remembers, OR a line that is genuinely a highlight — the
    #    LLM read real surprise/excitement in it, or the speaker's own voice
    #    got noticeably louder right here. A specific figure always gets first
    #    claim on the slot (pass 2a); emotional/vocal highlights only fill
    #    slots a figure isn't using nearby (pass 2b) — a real dollar amount
    #    should never lose its pill to an unrelated nearby highlight.
    if decorative_enabled:
        for i in range(n):
            if free(i) and MONEY_RE.search(texts[i]):
                if len(texts[i].split()) <= MAX_WORDS["chip"] and clear(i, "chip"):
                    styles[i] = "chip"
        for i in range(n):
            if not free(i):
                continue
            is_vocal_high = loud[i] and notes[i].weight >= 1
            is_llm_high = notes[i].mood in MOODS_THAT_POP and notes[i].weight >= 2
            if not (is_vocal_high or is_llm_high):
                continue
            if len(texts[i].split()) <= MAX_WORDS["chip"] and clear(i, "chip"):
                styles[i] = "chip"

    # 2.5. Premium/editorial only: a black CTA bubble on the rare line that is
    #      genuinely asking the viewer to do something right now — follow,
    #      subscribe, comment, share. Judged by the LLM (LineNote.cta), same
    #      pattern as mood/icon; never fires in the classic style.
    if caption_style in {"premium", "editorial"}:
        for i in range(n):
            if not free(i) or not notes[i].cta:
                continue
            if len(texts[i].split()) <= MAX_WORDS["bubble"] and clear(i, "bubble"):
                styles[i] = "bubble"

    # 3. Hand-drawn accents on the key ideas.
    accent_for = {"concept": "oval", "term": "underline", "drama": "tape", "quote": "quote"}
    order = sorted(range(n), key=lambda i: (-notes[i].weight, times[i]))
    if decorative_enabled:
        for i in order:
            note = notes[i]
            if not free(i) or note.weight < 2 or note.kind not in accent_for:
                continue
            if note.weight == 2 and note.kind not in {"drama", "quote", "term"}:
                continue
            n_words = len(texts[i].split())
            key = _key_for(texts[i], note.key)
            candidates = [accent_for[note.kind]]
            if note.kind == "concept":
                candidates.append("underline")
            for style in candidates:
                fits = {
                    "oval": n_words <= MAX_WORDS["oval"],
                    "underline": n_words <= MAX_WORDS["underline"] and bool(key),
                    "tape": n_words <= MAX_WORDS["tape"] and bool(key),
                    "quote": 2 <= n_words <= MAX_WORDS["quote"],
                }[style]
                if fits and clear(i, style):
                    styles[i] = style
                    emphasis[i] = key if style == "underline" else ""
                    break

    # 4. Serif payoffs.
    for i in order:
        note = notes[i]
        if not free(i) or note.weight < 2:
            continue
        if note.kind == "normal" and note.weight < 3:
            continue
        if len(texts[i].split()) <= MAX_WORDS["serif"] and clear(i, "serif"):
            styles[i] = "serif"

    # 4.5 Front-load the hook: the first ~30s decides whether someone keeps
    #     watching, so it gets first claim on the rare hand-drawn accents at a
    #     slightly lower bar than the rest of the reel — never invented onto
    #     filler, just given priority when a real candidate already exists.
    hook_window = [i for i in range(n) if times[i] - times[0] < HOOK_WINDOW_SECONDS]

    def hook_rank(i: int) -> tuple[int, int, float]:
        content = [t for t in texts[i].split() if norm_token(t) not in FILLER]
        return (notes[i].weight, len(content), -times[i])

    if decorative_enabled:
        for style, limit in (("oval", MAX_WORDS["oval"]), ("underline", MAX_WORDS["underline"])):
            if any(styles[i] == style for i in hook_window):
                continue
            for i in sorted(hook_window, key=hook_rank, reverse=True):
                if not free(i) or notes[i].weight < 1:
                    continue
                key = _key_for(texts[i], notes[i].key) or _content_word(texts[i].split())
                if len(texts[i].split()) > limit or len(norm_token(key)) < 4:
                    continue
                if not clear(i, style):
                    continue
                styles[i] = style
                emphasis[i] = key if style == "underline" else ""
                break

    # 5. Fill to the reel's rhythm: the judged lines are often too few for the
    #    look, so the strongest free lines take the remaining slots.
    seconds = max(times[-1] - times[0], 1.0)

    def rank(i: int) -> tuple[int, int, float]:
        content = [t for t in texts[i].split() if norm_token(t) not in FILLER]
        return (notes[i].weight, len(content), -times[i])

    fill_targets = (
        [
            ("oval", round(seconds / 26), MAX_WORDS["oval"]),
            ("underline", round(seconds / 20), MAX_WORDS["underline"]),
        ]
        if decorative_enabled
        else []
    ) + [("serif", round(seconds / 7), MAX_WORDS["serif"])]
    for style, target, limit in fill_targets:
        have = sum(1 for s in styles if s == style)
        for i in sorted(range(n), key=rank, reverse=True):
            if have >= target:
                break
            if not free(i) or notes[i].weight < 1:
                continue
            words_in = texts[i].split()
            if len(words_in) > limit:
                continue
            key = _key_for(texts[i], notes[i].key) or _content_word(words_in)
            # A hand-drawn accent needs a real word to sit on.
            if len(norm_token(key)) < (5 if style != "serif" else 4):
                continue
            if not clear(i, style):
                continue
            styles[i] = style
            emphasis[i] = key if style == "underline" else ""
            have += 1

    # 6. Cream serif words.
    for i in range(n):
        if not free(i):
            continue
        key = _key_for(texts[i], notes[i].key)
        if not key:
            continue
        prev_plain = i == 0 or styles[i - 1] == "plain"
        if notes[i].weight >= 2 or prev_plain or len(norm_token(key)) >= 7:
            styles[i], emphasis[i] = "mix", key

    # 7. Cream words carry the reel, but past ~45% of lines they stop reading as
    #    emphasis. Drop the weakest back to plain white sans.
    mixes = [i for i in range(n) if styles[i] == "mix"]
    budget = int(0.45 * n)
    if len(mixes) > budget:
        weakest = sorted(
            mixes, key=lambda i: (notes[i].weight, len(norm_token(emphasis[i])))
        )
        for i in weakest[: len(mixes) - budget]:
            styles[i], emphasis[i] = "plain", ""

    # 8. Mood: a real surprise or a real high gets a pop/"blink" reveal instead
    #    of the usual smooth rise — never on a plain connective line, never two
    #    in quick succession, so the motion still reads as a highlight. A
    #    genuine vocal-loudness spike counts too, even when the LLM's
    #    text-only read of that line was neutral — the voice knows something
    #    the transcript alone doesn't.
    mood = ["neutral"] * n
    last_pop = -99.0
    for i in order:
        note = notes[i]
        wants_pop = note.mood in MOODS_THAT_POP or (loud[i] and note.weight >= 1)
        if (
            wants_pop
            and styles[i] != "plain"
            and i not in stacked
            and times[i] - last_pop >= POP_MIN_GAP
        ):
            mood[i] = note.mood if note.mood in MOODS_THAT_POP else "excited"
            last_pop = times[i]

    # 9. Icon: a small contextual badge on the rare line that is genuinely
    #    about money, growth, an idea, a platform, outreach, or a proof point.
    icon = ["none"] * n
    last_icon = -99.0
    for i in order:
        note = notes[i]
        if (
            note.icon in ICONS
            and note.icon != "none"
            and note.weight >= 2
            and i not in stacked
            and styles[i] != "chip"  # the pill already is the "notice this" signal
            and times[i] - last_icon >= ICON_MIN_GAP
        ):
            icon[i] = note.icon
            last_icon = times[i]

    # The hook gets one icon at a slightly lower weight bar too, same
    # reasoning as the accents above — only if the director actually flagged
    # one in that window; never invented.
    if not any(icon[i] != "none" for i in hook_window):
        for i in sorted(hook_window, key=lambda j: times[j]):
            note = notes[i]
            if (
                note.icon in ICONS
                and note.icon != "none"
                and note.weight >= 1
                and i not in stacked
                and styles[i] != "chip"
            ):
                icon[i] = note.icon
                break

    out: list[CaptionDraft] = []
    absorbed = set(stacked.values())
    for i, d in enumerate(drafts):
        if i in absorbed:
            continue
        if i in stacked:
            nxt = drafts[stacked[i]]
            out.append(
                replace(
                    d, text=texts[i], sub=nxt.text, last=nxt.last, style="stack", emphasis="",
                    mood=mood[i], icon=icon[i], cta=notes[i].cta,
                )
            )
            continue
        out.append(
            replace(
                d, text=texts[i], style=styles[i], emphasis=emphasis[i],
                mood=mood[i], icon=icon[i], cta=notes[i].cta,
            )
        )
    return out


def align_phrases(phrases: list[str], asr: list[Word]) -> list[tuple[int, int]]:
    """Map consecutive caption phrases onto a segment's timed words.

    Returns one local (first, last) word range per phrase. Tokens that match ASR
    words anchor the mapping; the rest are interpolated by position. Every phrase
    gets at least one word while words last; extra phrases share the final word.
    """
    n_words = len(asr)
    n_phrases = len(phrases)
    if n_phrases == 0 or n_words == 0:
        return [(0, max(0, n_words - 1))] * n_phrases
    tokens: list[str] = []
    first_token: list[int] = []
    for phrase in phrases:
        first_token.append(len(tokens))
        tokens.extend(phrase.split() or ["_"])
    a = [norm_token(t) for t in tokens]
    b = [norm_token(w.word) for w in asr]
    anchors: list[tuple[float, float]] = [(-1.0, -0.5)]
    for block in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(block.size):
            if a[block.a + k]:
                anchors.append((float(block.a + k), float(block.b + k)))
    anchors.append((float(len(tokens)), n_words - 0.5))
    estimate = [0.0] * len(tokens)
    for (i0, p0), (i1, p1) in zip(anchors, anchors[1:]):
        lo, hi = int(i0), int(i1)
        for i in range(max(lo, 0), min(hi + 1, len(tokens))):
            span = (i1 - i0) or 1.0
            estimate[i] = p0 + (p1 - p0) * (i - i0) / span
    starts: list[int] = []
    for k in range(n_phrases):
        s = int(math.floor(estimate[first_token[k]] + 0.5))
        if k == 0:
            s = 0
        else:
            s = max(s, starts[-1] + 1)
            s = min(s, n_words - (n_phrases - k)) if n_phrases <= n_words else min(s, n_words - 1)
            s = max(s, starts[-1] + (1 if n_phrases <= n_words else 0))
        starts.append(min(max(s, 0), n_words - 1))
    spans: list[tuple[int, int]] = []
    for k, s in enumerate(starts):
        nxt = starts[k + 1] if k + 1 < n_phrases else n_words
        spans.append((s, max(s, nxt - 1)))
    return spans


def align_token_times(tokens: list[str], asr: list[Word]) -> list[float]:
    """Start time for each display token, anchored on matching ASR words."""
    if not tokens:
        return []
    if not asr:
        return [0.0] * len(tokens)
    t_start = asr[0].start
    t_end = max(asr[-1].end, t_start + 0.1)
    a = [norm_token(t) for t in tokens]
    b = [norm_token(w.word) for w in asr]
    anchors: dict[int, float] = {}
    for block in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(block.size):
            if a[block.a + k]:
                anchors[block.a + k] = asr[block.b + k].start
    times: list[float | None] = [anchors.get(i) for i in range(len(tokens))]
    times[0] = t_start
    # Interpolate the gaps between anchors.
    i = 0
    n = len(times)
    while i < n:
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < n and times[j] is None:
            j += 1
        left = times[i - 1] if i > 0 else t_start
        right = times[j] if j < n else t_end
        assert left is not None and right is not None
        steps = j - i + 1
        for k in range(i, j):
            times[k] = left + (right - left) * (k - i + 1) / steps
        i = j
    result: list[float] = []
    prev = t_start
    for value in times:
        v = max(float(value if value is not None else prev), prev)
        result.append(v)
        prev = v
    return result


def _emphasis_flags(tokens: list[str], emphasis: str) -> list[bool]:
    flags = [False] * len(tokens)
    wanted = [norm_token(t) for t in emphasis.split() if norm_token(t)]
    if not wanted:
        return flags
    normed = [norm_token(t) for t in tokens]
    for i in range(len(normed) - len(wanted) + 1):
        if normed[i : i + len(wanted)] == wanted:
            for k in range(len(wanted)):
                flags[i + k] = True
            break
    return flags


MIN_ON_SCREEN = 0.72
REVEAL_WINDOW = 0.45
# Captions may lead their first spoken word by this much when a flash needs room.
MAX_LEAD = 0.35


def caption_spans(
    drafts: list[CaptionDraft], words: list[Word], *, video_duration: float
) -> list[tuple[float, float]]:
    """On-screen [start, end) per draft: follows the voice, holds through short
    pauses, and borrows time from neighbours so no caption flashes."""
    limit = video_duration if video_duration > 0 else (words[-1].end + 1.0 if words else 0.0)
    starts = [words[d.first].start for d in drafts]
    ends: list[float] = []
    for i, d in enumerate(drafts):
        natural_end = words[d.last].end + 0.28
        if i + 1 < len(drafts):
            nxt = starts[i + 1]
            end = nxt if nxt - natural_end < 0.8 else natural_end + 0.35
            ends.append(min(end, nxt))
        else:
            ends.append(min(words[d.last].end + 0.9, limit))
    n = len(drafts)
    # Water-fill the minimum on-screen time. Each pass pushes a boundary later
    # (or earlier) by what the neighbour can spare; repeating it lets the need
    # travel along a run of rushed captions to wherever the slack actually is.
    for _ in range(12):
        moved = False
        for i in range(n):
            need = MIN_ON_SCREEN - (ends[i] - starts[i])
            if need <= 1e-4:
                continue
            # Extend into silence after the caption.
            ceiling = starts[i + 1] if i + 1 < n else limit
            grow = min(need, max(0.0, ceiling - ends[i]))
            if grow > 0:
                ends[i] += grow
                need -= grow
                moved = True
            # Push the next boundary later; the next caption may pass the need on.
            if need > 1e-4 and i + 1 < n and abs(ends[i] - starts[i + 1]) < 1e-6:
                room = (ends[i + 1] - starts[i + 1]) - 0.4 * MIN_ON_SCREEN
                shift = min(need, max(0.0, room))
                if shift > 0:
                    ends[i] += shift
                    starts[i + 1] += shift
                    need -= shift
                    moved = True
            # Or start earlier, taking from the previous caption.
            if need > 1e-4 and i > 0:
                room = (ends[i - 1] - starts[i - 1]) - 0.4 * MIN_ON_SCREEN
                lead = min(need, max(0.0, room), MAX_LEAD)
                if lead > 0:
                    if abs(ends[i - 1] - starts[i]) < 1e-6:
                        ends[i - 1] -= lead
                    starts[i] -= lead
                    moved = True
        if not moved:
            break
    spans: list[tuple[float, float]] = []
    prev_end = 0.0
    for s, e in zip(starts, ends):
        s = max(s, prev_end)
        e = min(max(e, s + 0.2), limit) if limit > s else s + 0.2
        spans.append((s, e))
        prev_end = e
    return spans


def drafts_to_timeline(
    drafts: list[CaptionDraft], words: list[Word], *, video_duration: float
) -> CaptionTimeline:
    captions: list[Caption] = []
    spans = caption_spans(drafts, words, video_duration=video_duration)
    for index, d in enumerate(drafts):
        asr = words[d.first : d.last + 1]
        if not asr:
            continue
        start, end = spans[index]
        if end <= start:
            continue

        line0 = d.tokens
        line1 = d.sub_tokens if d.style == "stack" else []
        tokens = line0 + line1
        times = align_token_times(tokens, asr)
        # Words build in with the voice, but the full line lands quickly so the
        # viewer reads phrases, not a word ticker.
        window = min(REVEAL_WINDOW, 0.5 * (end - start))
        last_t = max(times[-1] - start, 1e-6) if times else 1e-6
        if last_t > window:
            times = [start + (t - start) * window / last_t for t in times]
        flags = _emphasis_flags(line0, d.emphasis) + [False] * len(line1)
        reveal_ceiling = max(start, end - 0.18)
        cap_words: list[CaptionWord] = []
        for k, tok in enumerate(tokens):
            w_start = min(max(times[k], start), reveal_ceiling)
            w_end = min(times[k + 1], end) if k + 1 < len(tokens) else end
            w_end = max(w_end, w_start + 0.01)
            if w_end > end:
                w_start, w_end = min(w_start, end - 0.01), end
            cap_words.append(
                CaptionWord(text=tok, start=w_start, end=w_end, emphasis=flags[k])
            )
        text = " ".join(line0) + ("\n" + " ".join(line1) if line1 else "")
        captions.append(
            Caption(
                start=start,
                end=end,
                text=text,
                treatment=d.style,  # type: ignore[arg-type]
                words=cap_words,
                mood=d.mood,
                icon=d.icon,
                cta=d.cta,
            )
        )
    return CaptionTimeline(captions=captions)

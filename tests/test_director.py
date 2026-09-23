import asyncio
import json
from types import SimpleNamespace

from app.captions.craft import (
    CaptionDraft,
    LineNote,
    align_phrases,
    assign_styles,
    caption_spans,
    drafts_to_timeline,
    enforce_design_rules,
    fix_case,
)
from app.captions.director import CaptionDirector, script_problems
from app.transcription.models import Segment, Transcript, Word


def _words(text: str, start: float = 0.0, step: float = 0.35) -> list[Word]:
    return [
        Word(word=tok, start=start + i * step, end=start + (i + 1) * step - 0.02)
        for i, tok in enumerate(text.split())
    ]


def test_align_phrases_maps_lines_onto_spoken_words():
    asr = _words("Hi, I have recently switched to Senior AI Engineer's role.")
    spans = align_phrases(["I recently switched", "to senior AI engineer role"], asr)
    assert spans == [(0, 4), (5, 9)]


def test_align_phrases_gives_every_line_a_word():
    asr = _words("one two three")
    spans = align_phrases(["alpha", "beta", "gamma"], asr)
    assert [s[0] for s in spans] == [0, 1, 2]


def test_caption_spans_never_flash():
    words = _words("a b c d e f", step=0.12)
    drafts = [CaptionDraft(first=i, last=i, text=w.word) for i, w in enumerate(words)]
    merged = enforce_design_rules(drafts, words)
    spans = caption_spans(merged, words, video_duration=3.0)
    assert all(e > s for s, e in spans)
    assert all(b[0] >= a[1] - 1e-6 for a, b in zip(spans, spans[1:]))


def test_fix_case_removes_title_case_and_shouting():
    assert fix_case("When I Started Interviewing") == "When I started interviewing"
    # Long accidental caps get fixed; short tokens are trusted as real acronyms
    # even when they are not on the fixed ACRONYMS list (WHO, MCA, ROI, ...).
    assert fix_case("SERIOUSLY stop that") == "seriously stop that"
    assert fix_case("I did MCA then") == "I did MCA then"
    assert fix_case("associated with WHO") == "associated with WHO"


def test_enforce_rules_keeps_decorations_rare():
    words = _words(" ".join(f"w{i}" for i in range(40)), step=0.5)
    drafts = [
        CaptionDraft(first=i, last=i + 1, text="core depth", style="oval")
        for i in range(0, 40, 2)
    ]
    out = enforce_design_rules(drafts, words)
    assert sum(1 for d in out if d.style == "oval") == 1


def test_assign_styles_promotes_a_line_the_voice_gets_loud_on():
    """The LLM never hears tone — a genuine vocal-loudness spike is treated as
    real emphasis even when the text-only annotate pass called it neutral."""
    import numpy as np

    text = ["so", "anyway", "and", "then", "wait", "what", "happened", "next", "was", "wild"]
    words = _words(" ".join(text), step=1.0)
    drafts = [CaptionDraft(first=i, last=i, text=t) for i, t in enumerate(text)]
    # Every line reads flat to the LLM — but "wait" earns an ordinary cream
    # emphasis on its own merit, which is what makes the pop visible at all.
    notes = [LineNote(weight=1) for _ in text]
    notes[4] = LineNote(key="wait", weight=2)

    times = np.arange(0, 12, 0.2, dtype=np.float32)
    db = np.full_like(times, -32.0)
    db[(times >= 4.0) & (times < 4.9)] += 14.0  # "wait" (span 4.0-4.9) is genuinely loud

    out = assign_styles(drafts, notes, words, loud_times=times, loud_db=db)
    loud_draft = next(d for d in out if d.text == "wait")
    assert loud_draft.mood == "excited"
    assert loud_draft.style != "plain"


def test_assign_styles_builds_rhythm_from_notes():
    text = [
        "I recently switched",
        "to senior AI engineer",
        "When I started",
        "giving interviews",
        "I faced a lot of",
        "rejections",
        "I didn't have",
        "the core depth",
        "and got the role",
    ]
    words = _words(" ".join(text), step=1.0)
    drafts, cursor = [], 0
    for line in text:
        n = len(line.split())
        drafts.append(CaptionDraft(first=cursor, last=cursor + n - 1, text=line))
        cursor += n
    notes = [
        LineNote("switched", 2, "payoff"),
        LineNote("AI", 1, "term"),
        LineNote("", 0),
        LineNote("interviews", 1),
        LineNote("", 0),
        LineNote("rejections", 3, "drama"),
        LineNote("", 0),
        LineNote("core depth", 3, "concept"),
        LineNote("role", 3, "payoff"),
    ]
    styles = {d.text: d.style for d in assign_styles(drafts, notes, words)}
    assert styles["I recently switched"] in {"serif", "stack"}
    assert styles["rejections"] == "tape"
    assert styles["the core depth"] == "oval"
    # Content lines are designed, connective speech stays plain white sans.
    assert styles["giving interviews"] != "plain"
    assert styles["When I started"] == "plain"
    assert styles["I faced a lot of"] == "plain"


def test_assign_styles_editorial_suppresses_decorations_but_keeps_serif_and_mix():
    """The editorial theme is a deliberately clean, undecorated look — no
    hand-drawn oval/underline/tape/quote or colored chip — while the plain
    serif payoff and cream-word rhythm (which the renderer re-styles as
    italic for this theme) still fire exactly as in classic."""
    text = [
        "I recently switched",
        "to senior AI engineer",
        "When I started",
        "giving interviews",
        "I faced a lot of",
        "rejections",
        "I didn't have",
        "the core depth",
        "and got the role",
    ]
    words = _words(" ".join(text), step=1.0)
    drafts, cursor = [], 0
    for line in text:
        n = len(line.split())
        drafts.append(CaptionDraft(first=cursor, last=cursor + n - 1, text=line))
        cursor += n
    notes = [
        LineNote("switched", 2, "payoff"),
        LineNote("AI", 1, "term"),
        LineNote("", 0),
        LineNote("interviews", 1),
        LineNote("", 0),
        LineNote("rejections", 3, "drama"),
        LineNote("", 0),
        LineNote("core depth", 3, "concept"),
        LineNote("role", 3, "payoff"),
    ]
    out = assign_styles(drafts, notes, words, caption_style="editorial")
    styles = {d.style for d in out}
    assert styles <= {"plain", "mix", "serif", "stack", "bubble"}
    assert "oval" not in styles and "underline" not in styles
    assert "tape" not in styles and "chip" not in styles and "quote" not in styles
    # The payoff/concept lines still get *some* emphasis — just re-expressed
    # through the undecorated vocabulary instead of dropped entirely.
    assert {d.text: d.style for d in out}["the core depth"] != "plain"


def test_assign_styles_editorial_still_allows_cta_bubble():
    """Editorial keeps the same CTA bubble as premium — it's the one
    decoration the reference look still carries."""
    text = ["thanks for watching", "please follow for more"]
    words = _words(" ".join(text), step=1.0)
    drafts = [
        CaptionDraft(first=0, last=2, text="thanks for watching"),
        CaptionDraft(first=3, last=6, text="please follow for more"),
    ]
    notes = [LineNote(), LineNote(cta=True)]
    out_editorial = assign_styles(drafts, notes, words, caption_style="editorial")
    assert out_editorial[1].style == "bubble"
    out_classic = assign_styles(drafts, notes, words, caption_style="classic")
    assert out_classic[1].style != "bubble"


def test_drafts_to_timeline_reveals_whole_line_quickly():
    words = _words("when I started giving interviews", step=0.5)
    drafts = [CaptionDraft(first=0, last=4, text="When I started giving interviews", style="plain")]
    timeline = drafts_to_timeline(drafts, words, video_duration=4.0)
    cap = timeline.captions[0]
    assert cap.words[-1].start - cap.start <= 0.56


def test_script_problems_flags_condensed_and_invented_lines():
    segments = [
        {
            "id": 0,
            "seconds": 6.0,
            "spoken_words": 16,
            "asr": "But when the interviewer asked me about their breakdowns and related things",
            "text": "But when the interviewer asked me about their breakdowns and related things",
        }
    ]
    data = {"segments": [{"id": 0, "lines": ["asked about trade-offs"]}]}
    problems = " ".join(script_problems(data, segments))
    assert "lines; use about" in problems
    assert "trade" in problems


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def create(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        self.calls.append(system[:40])
        for marker, payload in self.responses:
            if marker in system:
                content = payload(kwargs) if callable(payload) else payload
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content)))]
                )
        raise AssertionError(f"unexpected prompt: {system[:60]}")


def test_director_runs_brief_script_review_annotate():
    words = [
        Word(word=w, start=i * 0.4, end=i * 0.4 + 0.35)
        for i, w in enumerate("I have recently switched to Senior AI Engineer's role".split())
    ]
    transcript = Transcript(
        language="en",
        spoken_language="hi",
        segments=[Segment(start=0.0, end=3.6, text=" ".join(w.word for w in words), words=words)],
    )
    completions = _FakeCompletions(
        [
            (
                "story editor",
                {
                    "topic": "AI interviews",
                    "music_mood": "focused_tech",
                    "hook": "I recently switched",
                    "key_ideas": ["Senior AI Engineer"],
                    "segments": [{"id": 0, "text": "I recently switched to a Senior AI Engineer role."}],
                },
            ),
            (
                "caption lines",
                {"segments": [{"id": 0, "lines": ["I recently switched", "to senior AI engineer role"]}]},
            ),
            ("copy reviewer", {"edits": [], "notes": "fine"}),
            (
                "art director",
                {
                    "lines": [
                        {"i": 0, "key": "switched", "weight": 3, "kind": "payoff", "mood": "excited", "icon": "none"},
                        {"i": 1, "key": "AI", "weight": 2, "kind": "term", "mood": "neutral", "icon": "growth"},
                    ]
                },
            ),
        ]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = asyncio.run(
        CaptionDirector(client=client, model="fake").direct(transcript, video_duration=4.0)
    )
    caps = result.timeline.captions
    assert result.brief.music_mood == "focused_tech"
    # The LLM's mood/icon judgement made it all the way to the rendered caption.
    assert caps[0].mood == "excited"
    assert caps[1].icon == "growth"
    assert [c.text for c in caps] == ["I recently switched", "to senior AI engineer role"]
    assert caps[0].treatment in {"serif", "stack"}
    assert caps[-1].end <= 4.0
    assert len(completions.calls) == 4

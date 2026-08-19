from __future__ import annotations

GLYPHS = frozenset(
    {
        "search",
        "brain",
        "scale",
        "gear",
        "chart",
        "bolt",
        "docs",
        "loop",
        "target",
        "spark",
        "shield",
        "clock",
    }
)

_EMOJI_GLYPH = {
    "📄": "docs",
    "🔍": "search",
    "🧠": "brain",
    "⚙️": "gear",
    "💰": "chart",
    "🔁": "loop",
    "📂": "docs",
    "💡": "spark",
    "⚖️": "scale",
    "🚀": "bolt",
    "📊": "chart",
    "🎯": "target",
    "✨": "spark",
    "🧩": "gear",
    "🔐": "shield",
    "⏱️": "clock",
    "📁": "docs",
}

_KIND_GLYPH = {
    "vs_split": "scale",
    "process": "loop",
    "stat": "chart",
    "quote": "spark",
    "topic": "target",
    "chip_row": "loop",
    "term_card": "spark",
    "bullets": "spark",
}


def resolve_glyph(icon: str = "", kind: str = "bullets", glyph: str = "") -> str:
    raw = (glyph or icon or "").strip()
    key = raw.lower()
    if key in GLYPHS:
        return key
    if raw in _EMOJI_GLYPH:
        return _EMOJI_GLYPH[raw]
    return _KIND_GLYPH.get(kind, "spark")

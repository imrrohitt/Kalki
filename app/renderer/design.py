"""Design tokens for the split-reel motion canvas.

Single source of truth for typography, color, spacing, and motion so the
graphics read as one system instead of per-card improvisation.

ASS colors are &HAABBGGRR (alpha first, then blue/green/red). FFmpeg colors
below are plain 0xRRGGBB.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------- canvas
CANVAS_W = 1080
MARGIN_X = 72          # ~6.7% horizontal safe margin
TOP_Y = 64             # first baseline of the card
SAFE_BOTTOM_PAD = 72   # keep type off the seam
USABLE_W = CANVAS_W - 2 * MARGIN_X

# Full-canvas audio reel (no talking head). Fill top, middle, and bottom.
FULL_SAFE_Y = 1680
CAPTION_Y_FULL = 1748
FULL_MARGIN_X = 56
FULL_USABLE_W = CANVAS_W - 2 * FULL_MARGIN_X

# ---------------------------------------------------------------- themes
# Each theme is one coherent premium palette. Neutral carries 85%+ of the
# frame; accent is reserved for the kicker, indices, stats, and drawn rules.
# ASS colors are &HAABBGGRR; bg_* / grid_color are FFmpeg 0xRRGGBB.


@dataclass(frozen=True)
class Theme:
    name: str
    bg_top: str
    bg_bottom: str
    ink: str
    muted: str
    faint: str
    accent: str
    hair: str                  # hairline rules (ink at ~30% opacity)
    caption_emphasis: str      # keyword color for captions over video
    caption_outline: str       # stays dark on every theme (captions sit on video)
    grid: bool = False         # very subtle technical grid on the canvas
    grid_color: str = "0xFFFFFF"
    grid_opacity: float = 0.05


THEMES: dict[str, Theme] = {
    # Warm editorial: cream paper, warm near-black ink, burnt-orange accent.
    "paper": Theme(
        name="paper",
        bg_top="0xF7F2E9",
        bg_bottom="0xEDE5D5",
        ink="&H0014181C",              # #1C1814
        muted="&H005C646B",            # #6B645C
        faint="&H00968E85",            # #858E96 warm gray
        accent="&H001A5AE8",           # #E85A1A burnt orange
        hair="&HB414181C",
        caption_emphasis="&H002F7AFF",  # #FF7A2F
        caption_outline="&H0014181C",
    ),
    # Dark cinematic: near-black with a soft top light, warm off-white type,
    # amber-gold accent. Documentary / film-title energy.
    "noir": Theme(
        name="noir",
        bg_top="0x1A1B20",
        bg_bottom="0x0D0E11",
        ink="&H00E8EFF2",              # #F2EFE8 warm off-white
        muted="&H009CA4A8",            # #A8A49C
        faint="&H00636A6E",            # #6E6A63
        accent="&H003CB0F0",           # #F0B03C amber gold
        hair="&HB4E8EFF2",
        caption_emphasis="&H004DC2FF",  # #FFC24D
        caption_outline="&H00101114",
    ),
    # Technical: dark slate, cool grays, cyan accent, fine grid.
    "tech": Theme(
        name="tech",
        bg_top="0x131A22",
        bg_bottom="0x0B0F14",
        ink="&H00F5EFE9",              # #E9EFF5 cool white
        muted="&H00A5988B",            # #8B98A5
        faint="&H0075695D",            # #5D6975
        accent="&H00F8BD38",           # #38BDF8 cyan
        hair="&HB4F5EFE9",
        caption_emphasis="&H00FFC94F",  # #4FC9FF
        caption_outline="&H00101114",
        grid=True,
        grid_color="0xE9EFF5",
        grid_opacity=0.05,
    ),
    # Minimal light: gallery off-white, neutral ink, royal-blue accent.
    "ivory": Theme(
        name="ivory",
        bg_top="0xFBFBF9",
        bg_bottom="0xF0F0EA",
        ink="&H001C1A19",              # #191A1C
        muted="&H0076706D",            # #6D7076
        faint="&H00A39D9A",            # #9A9DA3
        accent="&H00E65B2E",           # #2E5BE6 royal blue
        hair="&HB41C1A19",
        caption_emphasis="&H00FF794D",  # #4D79FF
        caption_outline="&H00191A1C",
    ),
}

DEFAULT_THEME = "paper"


def get_theme(name: str | None) -> Theme:
    return THEMES.get((name or "").strip().lower(), THEMES[DEFAULT_THEME])


# Back-compat aliases (default paper palette).
_PAPER = THEMES["paper"]
BG_GRADIENT_TOP = _PAPER.bg_top
BG_GRADIENT_BOTTOM = _PAPER.bg_bottom
INK = _PAPER.ink
MUTED = _PAPER.muted
FAINT = _PAPER.faint
ACCENT = _PAPER.accent
HAIR_INK = _PAPER.hair
CAPTION_EMPHASIS = _PAPER.caption_emphasis

# ---------------------------------------------------------------- fonts
FONT_REGULAR = "Montserrat"          # Bold flag off
FONT_MEDIUM = "Montserrat Medium"
FONT_SEMIBOLD = "Montserrat SemiBold"
FONT_BOLD = "Montserrat"             # Bold flag on

# ---------------------------------------------------------------- type scale
FS_KICKER = 33
FS_DISPLAY = 118       # short hook headlines go display-size
FS_TITLE = 92          # fitted down as needed, never below FS_TITLE_MIN
FS_TITLE_MIN = 56
FS_SUBTITLE = 44
FS_STAT = 168
FS_BULLET = 54
FS_INDEX = 33          # bullet / step numerals
FS_VS_LABEL = 72
FS_VS_BADGE = 30
FS_CHIP = 46
FS_QUOTE = 62
FS_META = 36
FS_TITLE_FULL = 68
FS_NODE = 40
FS_NODE_SUB = 30
FS_BULLET_FULL = 40

TRACK_KICKER = 3.4     # uppercase labels get open tracking
TRACK_TITLE = -0.5     # large headlines get slightly tight tracking
TRACK_BADGE = 2.4

LINE_HEIGHT_TITLE = 1.10

# ---------------------------------------------------------------- motion
# Entrances ease out (fast in, soft settle); exits are short fades.
# \t accel < 1 is ease-out, > 1 is ease-in.
DUR_FAST = 240
DUR_BASE = 400
DUR_SLOW = 700

ACCEL_OUT = 0.28
ACCEL_SOFT = 0.5
ACCEL_IN = 2.4

RISE_PX = 26           # masked title rise
DRIFT_PX = 14          # secondary fade-rise drift
SLIDE_PX = 22          # vs-split horizontal slide

STAGGER_MS = 80        # sibling stagger inside a group
FADE_OUT_MS = 170

STAT_COUNT_MS = 640    # total count-up time
STAT_COUNT_STEPS = 12

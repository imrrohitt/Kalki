"""Premium caption layer for full-frame talking-head reels.

Type is rasterized with Pillow (real font metrics, baseline-aligned mixed fonts,
soft photographic shadows, supersampled strokes), animated per frame, and
streamed to FFmpeg as a transparent RGBA band that is overlaid above the
speaker's head.

The look follows the reference reels:
- plain      white geometric sans, words appear as they are spoken
- mix        white sans with one cream serif word (acronyms stay sans, in cream)
- serif      a whole line of cream serif that fades up softly
- stack      a big cream serif hook with a white sans line hanging below
- quote      cream serif in curly quotes
- oval       cream serif circled by a thin hand-drawn oval with two sparkles
- underline  a hairline rule with a sparkle at its centre
- tape       dark serif on a torn sage paper-tape sticker
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.captions.models import Caption, CaptionTimeline, CaptionWord
from app.config import ROOT_DIR

FONT_DIR = ROOT_DIR / "assets" / "fonts"
SANS_FONT = str(FONT_DIR / "Montserrat-Bold.ttf")
SERIF_FONT = str(FONT_DIR / "PlayfairDisplay-Regular.ttf")
# The editorial theme's emphasis word — a genuine italic serif, not a filter
# tilt, matching the clean "creator commentary" reference look.
SERIF_ITALIC_FONT = str(FONT_DIR / "PlayfairDisplay-Italic.ttf")

WHITE = (255, 255, 255)
CREAM = (255, 246, 194)
TAPE_FILL = (176, 196, 170)
TAPE_INK = (38, 26, 19)
HAIRLINE = (240, 238, 232)
# On a bright wall or sky, white-on-white (or a pale cream hairline) goes
# invisible. Each caption samples the real background under it and, when it
# reads bright, flips to this dark set instead — same warm identity, same
# hue family (a deep espresso-gold in place of pale cream), inverted for
# contrast. Chip/tape/badges already sit on a solid fill and never need this.
BRIGHT_INK = (26, 24, 22)
BRIGHT_GOLD = (120, 84, 18)
BRIGHT_HAIRLINE = (46, 42, 36)
# The editorial theme's accent family — a clean, light mint-green swapped in
# for the usual cream/gold, on a dark or bright background respectively. Body
# text stays plain white/ink either way; only the emphasis word, the cursor
# that follows it, and the CTA bubble's outline pick up the green.
EDITORIAL_GREEN = (176, 255, 196)
EDITORIAL_GREEN_BRIGHT = (18, 92, 48)
# Mean band luminance (0-255) above which a caption is judged to sit on a
# bright background.
BRIGHT_LUMA_THRESHOLD = 150.0
# Premium colored pill for a line naming a concrete figure (money, %, a round
# number) — alternated so a two-chip video doesn't repeat the same color.
CHIP_COLORS = [(22, 34, 49), (140, 72, 40), (33, 58, 47)]  # ink-navy, terracotta, forest
CHIP_TEXT = (255, 250, 236)
CHIP_SPARK = (255, 250, 232)
# The premium theme's "subscribe button" moment — a near-black badge with a
# thin cream outline and a star, for the rare line that is a direct ask of
# the viewer (follow/subscribe/comment). Deliberately distinct from the chip
# pill so it reads as a call-to-action, not another stat callout.
BUBBLE_FILL = (15, 15, 17)
BUBBLE_OUTLINE = (232, 224, 200)
BUBBLE_TEXT = (250, 247, 238)
# A genuine hand-marker accent for the premium theme — a real script font,
# not a filter effect, used sparingly for a casual, personal-note beat.
MARKER_FONT = str(FONT_DIR / "PermanentMarker-Regular.ttf")

# Font sizes in px on a 1080-wide frame.
FS_SANS = 58
FS_SERIF_INLINE = 80
FS_ACRONYM = 68
FS_SERIF = 90
FS_HOOK = 104
FS_SUB = 56
FS_TAPE = 94
FS_CHIP = 64
FS_BUBBLE = 62
# Script/marker fonts run visually smaller than a sans/serif at the same
# point size — sized up so the "handwritten" line reads at equal weight.
FS_MARKER = 96
MAX_LINE_FRAC = 0.84

WORD_IN = 0.16
LINE_IN = 0.30
EXIT = 0.08
# A "blink" pop for a line the director judged as a real surprise or an
# excited beat: a quick overshoot-and-settle instead of the usual smooth rise.
POP_SCALES = (0.4, 0.75, 1.28, 0.92, 1.08, 1.0)
POP_DUR = 0.24

SERIF_LINE_STYLES = {"serif", "oval", "quote"}
DECORATED = {"oval", "underline", "tape", "blob"}
# Editorial theme: words snap in fully typed (no rise/slide) with a blinking
# text cursor trailing the most recent one — a genuine typewriter reveal
# rather than a filter. Applies to the per-word treatments only; a whole-line
# serif statement keeps its soft fade, same as classic/premium.
TYPEWRITER_TREATMENTS = {"plain", "mix", "underline", "stack"}
CURSOR_ON = 0.22
CURSOR_OFF = 0.16
CURSOR_MAX_S = 0.85  # stop blinking after this long even on a long pause
CURSOR_GAP = 4.0  # px at u=1, between the word's last glyph and the cursor


@lru_cache(maxsize=128)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, max(8, int(size)))


def _ease_out(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1.0 - (1.0 - x) ** 3


def _premultiplied(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:4]
    return arr


def _is_acronym(token: str) -> bool:
    core = "".join(ch for ch in token if ch.isalnum())
    return 1 < len(core) <= 5 and core.isupper()


# --------------------------------------------------------------------- sprites
@dataclass
class TextSprite:
    arr: np.ndarray
    ox: int  # baseline origin inside arr
    oy: int
    advance: float
    ink_top: float  # relative to baseline (negative = above)
    ink_bottom: float
    space: float


def text_sprite(
    text: str,
    font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int],
    *,
    u: float,
    shadow: float,
    stroke: int = 0,
) -> TextSprite:
    left, top, right, bottom = font.getbbox(text, anchor="ls", stroke_width=stroke)
    blur = 7.0 * u
    pad = int(math.ceil(blur * 2.6 + 8 * u)) + 2
    w = int(right - left) + 2 * pad
    h = int(bottom - top) + 2 * pad
    ox, oy = pad - left, pad - top
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).text(
        (ox, oy), text, font=font, fill=255, anchor="ls", stroke_width=stroke, stroke_fill=255
    )
    m = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
    out = np.zeros((h, w, 4), dtype=np.float32)
    if shadow > 0:
        wide = np.asarray(mask.filter(ImageFilter.GaussianBlur(blur)), np.float32) / 255.0
        tight = np.asarray(
            mask.filter(ImageFilter.GaussianBlur(max(1.0, 1.7 * u))), np.float32
        ) / 255.0
        wide = np.roll(wide, max(1, round(3 * u)), axis=0)
        tight = np.roll(tight, max(1, round(1.4 * u)), axis=0)
        alpha = 1.0 - (1.0 - np.clip(wide * 0.52 * shadow, 0, 1)) * (
            1.0 - np.clip(tight * 0.38 * shadow, 0, 1)
        )
        if shadow > 1.15:
            glow = np.asarray(
                mask.filter(ImageFilter.GaussianBlur(18 * u)), np.float32
            ) / 255.0
            alpha = 1.0 - (1.0 - alpha) * (1.0 - np.clip(glow * 0.30, 0, 1))
        out[..., 3] = alpha
    col = np.array(color, dtype=np.float32) / 255.0
    out[..., :3] = col * m + out[..., :3] * (1.0 - m)
    out[..., 3:4] = m + out[..., 3:4] * (1.0 - m)
    return TextSprite(
        arr=out,
        ox=ox,
        oy=oy,
        advance=float(font.getlength(text)),
        ink_top=float(top),
        ink_bottom=float(bottom),
        space=float(font.getlength(" ")),
    )


def _astroid(size: int, color: tuple[int, int, int], *, glow: bool = True) -> np.ndarray:
    """Four-point sparkle with concave sides, supersampled."""
    ss = 4
    r = size * ss
    dim = int(r * 2.9)
    img = Image.new("L", (dim, dim), 0)
    c = dim / 2
    pts = []
    for k in range(160):
        th = 2 * math.pi * k / 160
        x = r * math.copysign(abs(math.cos(th)) ** 3, math.cos(th))
        y = r * math.copysign(abs(math.sin(th)) ** 3, math.sin(th))
        pts.append((c + x, c + y))
    ImageDraw.Draw(img).polygon(pts, fill=255)
    small = img.resize((dim // ss, dim // ss), Image.LANCZOS)
    m = np.asarray(small, np.float32) / 255.0
    alpha = m.copy()
    if glow:
        g = np.asarray(small.filter(ImageFilter.GaussianBlur(size * 0.35)), np.float32) / 255.0
        alpha = 1.0 - (1.0 - alpha) * (1.0 - g * 0.45)
    out = np.zeros(m.shape + (4,), np.float32)
    col = np.array(color, np.float32) / 255.0
    out[..., :3] = col * alpha[..., None]
    out[..., 3] = alpha
    return out


def _trim_alpha(arr: np.ndarray, pad: int = 2) -> np.ndarray:
    """Crop to the ink's own bounding box — text_sprite pads generously for a
    shadow blur radius even when shadow is off, which would otherwise make a
    tiny glyph (like a currency sign) look empty inside a small icon badge."""
    ys, xs = np.nonzero(arr[..., 3] > 0.01)
    if len(ys) == 0:
        return arr
    y0, y1 = max(0, ys.min() - pad), min(arr.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(arr.shape[1], xs.max() + 1 + pad)
    return arr[y0:y1, x0:x1]


def _star5(size: int, color: tuple[int, int, int]) -> np.ndarray:
    """A proper 5-point star (quality/rating) — distinct from the 4-point
    sparkle used for decorative accents."""
    ss = 4
    r = size * ss
    dim = int(r * 2.6)
    img = Image.new("L", (dim, dim), 0)
    c = dim / 2
    inner = r * 0.42
    pts = []
    for k in range(10):
        rad = r if k % 2 == 0 else inner
        th = -math.pi / 2 + k * math.pi / 5
        pts.append((c + rad * math.cos(th), c + rad * math.sin(th)))
    ImageDraw.Draw(img).polygon(pts, fill=255)
    small = img.resize((dim // ss, dim // ss), Image.LANCZOS)
    m = np.asarray(small, np.float32) / 255.0
    out = np.zeros(m.shape + (4,), np.float32)
    col = np.array(color, np.float32) / 255.0
    out[..., :3] = col * m[..., None]
    out[..., 3] = m
    return out


def _scaled(arr: np.ndarray, scale: float) -> np.ndarray:
    h, w = arr.shape[:2]
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGBA")
    return np.asarray(img.resize((nw, nh), Image.LANCZOS), np.float32) / 255.0


# -------------------------------------------------------------------- elements
@dataclass
class Element:
    t_in: float
    dur: float
    t_out: float
    mode: str  # rise | wipe | grow | frames
    arr: np.ndarray | None = None
    x: int = 0  # top-left (rise/wipe/grow) or centre (frames)
    y: int = 0
    rise: float = 0.0
    frames: list[np.ndarray] = field(default_factory=list)
    uid: int = 0

    def state(self, t: float) -> tuple[float, float] | None:
        if t < self.t_in or t >= self.t_out:
            return None
        p = 1.0 if self.dur <= 0 else min(1.0, (t - self.t_in) / self.dur)
        e = min(1.0, max(0.0, (self.t_out - t) / EXIT))
        return p, e


def _blend(canvas: np.ndarray, src: np.ndarray, x: int, y: int, alpha: float) -> tuple[int, int, int, int] | None:
    h, w = src.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1 or alpha <= 0.002:
        return None
    part = src[y0 - y : y1 - y, x0 - x : x1 - x]
    if alpha < 0.999:
        part = part * alpha
    dst = canvas[y0:y1, x0:x1]
    dst *= 1.0 - part[..., 3:4]
    dst += part
    return x0, y0, x1, y1


# ---------------------------------------------------------------------- layout
@dataclass
class _Token:
    word: CaptionWord
    text: str
    font_path: str
    size: int
    color: tuple[int, int, int]
    shadow: bool = True


class CaptionLayer:
    def __init__(
        self,
        timeline: CaptionTimeline,
        *,
        width: int,
        height: int,
        fps: int = 30,
        head_top: int | None = None,
        bright_background: bool = False,
        bg_luma_times: list[float] | None = None,
        bg_luma_values: list[float] | None = None,
        caption_style: str = "classic",
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.u = width / 1080.0
        self.caption_style = caption_style
        # Fallback for when no per-time track is available (tests, or a
        # detection failure): one flat guess for the whole reel, same as
        # before. When the track is present each caption judges its own
        # moment instead — a video that walks from shade into open sky needs
        # both looks, not one global guess.
        self._default_bright = bright_background
        self.shadow = 1.4 if bright_background else 1.0
        self._luma_times = np.asarray(bg_luma_times, np.float32) if bg_luma_times else None
        self._luma_values = np.asarray(bg_luma_values, np.float32) if bg_luma_values else None
        self.captions: list[Caption] = sorted(timeline.captions, key=lambda c: c.start)
        u = self.u
        head = head_top if head_top else int(height * 0.30)
        baseline = head - int(165 * u)
        baseline = max(int(height * 0.12), min(baseline, int(height * 0.25)))
        baseline = max(baseline, int(250 * u))
        top = max(0, baseline - int(250 * u))
        bottom = min(height, baseline + int(190 * u))

        # Premium theme only: an occasional line sits below the face — on the
        # chest/upper-torso area — instead of always above the head, so a
        # talking-head reel doesn't read from the exact same spot every beat.
        # Only when the framing actually shows that much of the speaker (a
        # tight face-filling close-up has nowhere to put it), and only ever a
        # heuristic offset below the hairline — there is no chin/chest
        # detector, so this is deliberately conservative.
        self.baseline_chest: int | None = None
        chest_enabled = False
        if caption_style == "premium" and head < height * 0.46:
            chest_y = int(head + 460 * u)
            if chest_y + int(170 * u) < height * 0.94:
                chest_enabled = True
                top = min(top, chest_y - int(120 * u))
                bottom = max(bottom, chest_y + int(150 * u))

        self.band_top = max(0, top - (top % 2))
        self.band_height = (bottom - self.band_top) + ((bottom - self.band_top) % 2)
        self.baseline = baseline - self.band_top
        if chest_enabled:
            self.baseline_chest = int(head + 460 * u) - self.band_top
        self._cache: dict[int, list[Element]] = {}
        self._uid = 0
        self._plan_premium_variety(chest_enabled)

    def _plan_premium_variety(self, chest_enabled: bool) -> None:
        """Deterministic, timing-based variety for the premium theme — never
        an LLM call, since these are purely spatial/typographic rotations the
        renderer alone has the geometry to decide (font rotation, and whether
        this video's own framing shows a chest area at all). Sets
        `font_variant`/`screen_area` directly on the sorted Caption objects,
        which `_tokens`/`_layout` already read. Rare and spaced out, same
        restraint as every other accent in this file. A no-op entirely in the
        classic style."""
        if self.caption_style != "premium" or not self.captions:
            return
        MARKER_MIN_GAP = 15.0
        CHEST_MIN_GAP = 9.0
        RESTYLABLE = {"plain", "mix", "serif"}
        last_marker = -1e9
        last_chest = -1e9
        for i, cap in enumerate(self.captions):
            if i == 0:
                continue  # keep the hook consistent
            t = cap.start
            words = len((cap.text or "").split())
            # A hand-marker beat: a short, undecorated line, well after the
            # last one — a casual aside, not competing with an oval/chip/tape
            # moment that already has its own distinct look.
            if (
                cap.treatment in RESTYLABLE
                and 1 <= words <= 4
                and t - last_marker >= MARKER_MIN_GAP
            ):
                cap.font_variant = "marker"
                last_marker = t
                continue  # one accent per line — don't also send it to the chest
            if chest_enabled and t - last_chest >= CHEST_MIN_GAP:
                cap.screen_area = "chest"
                last_chest = t

    # ------------------------------------------------------------ public api
    def iter_frames(self, n_frames: int) -> Iterator[bytes]:
        zero = bytes(self.width * self.band_height * 4)
        prev_key: tuple | None = None
        prev = zero
        cursor = 0
        for n in range(n_frames):
            t = n / self.fps
            while cursor < len(self.captions) and self.captions[cursor].end <= t:
                self._cache.pop(cursor, None)
                cursor += 1
            states = self._states(t, cursor)
            if not states:
                prev_key, prev = (), zero
                yield zero
                continue
            key = tuple(
                (el.uid, round(p * 48), round(e * 16)) for el, p, e in states
            )
            if key != prev_key:
                prev = self._compose(states).tobytes()
                prev_key = key
            yield prev

    def frame_rgba(self, t: float) -> np.ndarray:
        """Straight-alpha uint8 band at time t (for previews and tests)."""
        cursor = 0
        while cursor < len(self.captions) and self.captions[cursor].end <= t:
            cursor += 1
        states = self._states(t, cursor)
        if not states:
            return np.zeros((self.band_height, self.width, 4), np.uint8)
        return self._compose(states)

    # ------------------------------------------------------------- internals
    def _states(self, t: float, cursor: int) -> list[tuple[Element, float, float]]:
        out: list[tuple[Element, float, float]] = []
        i = cursor
        while i < len(self.captions) and self.captions[i].start <= t:
            cap = self.captions[i]
            if cap.end > t:
                if i not in self._cache:
                    self._cache[i] = self._layout(cap, i)
                for el in self._cache[i]:
                    st = el.state(t)
                    if st is not None:
                        out.append((el, st[0], st[1]))
            i += 1
        return out

    def _compose(self, states: list[tuple[Element, float, float]]) -> np.ndarray:
        canvas = np.zeros((self.band_height, self.width, 4), np.float32)
        box = [self.width, self.band_height, 0, 0]
        for el, p, e in states:
            drawn = self._draw(canvas, el, p, e)
            if drawn:
                box = [min(box[0], drawn[0]), min(box[1], drawn[1]), max(box[2], drawn[2]), max(box[3], drawn[3])]
        out = np.zeros((self.band_height, self.width, 4), np.uint8)
        if box[0] >= box[2] or box[1] >= box[3]:
            return out
        x0, y0, x1, y1 = box
        region = canvas[y0:y1, x0:x1]
        a = region[..., 3:4]
        rgb = region[..., :3] / np.maximum(a, 1e-5)
        out[y0:y1, x0:x1, :3] = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
        out[y0:y1, x0:x1, 3] = np.clip(a[..., 0] * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return out

    def _draw(self, canvas: np.ndarray, el: Element, p: float, e: float):
        eased = _ease_out(p)
        if el.mode == "rise":
            assert el.arr is not None
            dy = int(round(el.rise * (1.0 - eased)))
            return _blend(canvas, el.arr, el.x, el.y + dy, eased * e)
        if el.mode == "wipe":
            assert el.arr is not None
            w = max(1, int(round(el.arr.shape[1] * eased)))
            return _blend(canvas, el.arr[:, :w], el.x, el.y, e)
        if el.mode == "grow":
            assert el.arr is not None
            full = el.arr.shape[1]
            w = max(2, int(round(full * eased)))
            x0 = (full - w) // 2
            return _blend(canvas, el.arr[:, x0 : x0 + w], el.x + x0, el.y, e)
        if el.mode == "frames":
            idx = min(len(el.frames) - 1, int(p * (len(el.frames) - 1) + 0.5))
            arr = el.frames[idx]
            h, w = arr.shape[:2]
            return _blend(canvas, arr, el.x - w // 2, el.y - h // 2, e if p >= 1 else e * min(1.0, 0.35 + p))
        return None

    def _element(self, **kwargs) -> Element:
        self._uid += 1
        return Element(uid=self._uid, **kwargs)

    def _is_bright(self, cap: Caption) -> bool:
        """Real background intelligence: the mean luminance of the video
        under THIS caption's own on-screen window, not a single guess for
        the whole reel — a talking head walking from shade into open sky
        needs white text in one and dark ink in the other."""
        if self._luma_times is None or self._luma_values is None or len(self._luma_times) == 0:
            return self._default_bright
        mask = (self._luma_times >= cap.start) & (self._luma_times <= cap.end)
        if not mask.any():
            idx = int(np.argmin(np.abs(self._luma_times - cap.start)))
            sample = float(self._luma_values[idx])
        else:
            sample = float(self._luma_values[mask].mean())
        return sample > BRIGHT_LUMA_THRESHOLD

    def _palette_for(self, bright: bool) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        """(body, accent, hairline) — the dark set on a bright wall or sky,
        the usual cream-and-white set otherwise. Editorial swaps the accent
        for its clean light-green family; body text stays plain white/ink."""
        if self.caption_style == "editorial":
            accent = EDITORIAL_GREEN_BRIGHT if bright else EDITORIAL_GREEN
            return (BRIGHT_INK if bright else WHITE), accent, accent
        if bright:
            return BRIGHT_INK, BRIGHT_GOLD, BRIGHT_HAIRLINE
        return WHITE, CREAM, HAIRLINE

    def _tokens(
        self,
        cap: Caption,
        index: int,
        palette: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    ) -> list[list[_Token]]:
        body_color, accent_color, _hair = palette
        treatment = cap.treatment or "plain"
        if treatment == "blob":
            treatment = "tape"
        words = list(cap.words) or [
            CaptionWord(text=t, start=cap.start, end=cap.end) for t in cap.text.split()
        ]
        raw_lines = (cap.text or "").replace("\\n", "\n").split("\n")
        counts = [len(line.split()) for line in raw_lines if line.split()]
        if sum(counts) != len(words):
            counts = [len(words)]
        hook = index == 0 or cap.start < 1.2
        single_emphasis = sum(1 for w in words if w.emphasis) == 1
        lines: list[list[_Token]] = []
        k = 0
        for li, count in enumerate(counts):
            line: list[_Token] = []
            for word in words[k : k + count]:
                text = word.text
                if getattr(cap, "font_variant", "default") == "marker" and treatment in {"plain", "mix", "serif"}:
                    # The premium theme's hand-marker beat overrides the usual
                    # font for this whole line — a casual, personal-note look.
                    tok = _Token(word, text, MARKER_FONT, FS_MARKER, accent_color)
                elif treatment == "tape":
                    tok = _Token(word, text, SERIF_FONT, FS_TAPE, TAPE_INK, shadow=False)
                elif treatment == "chip":
                    tok = _Token(word, text, SANS_FONT, FS_CHIP, CHIP_TEXT, shadow=False)
                elif treatment == "bubble":
                    tok = _Token(word, text, SANS_FONT, FS_BUBBLE, BUBBLE_TEXT, shadow=False)
                elif treatment == "stack":
                    if li == 0:
                        tok = _Token(word, text, SERIF_FONT, FS_HOOK, accent_color)
                    else:
                        tok = _Token(word, text, SANS_FONT, FS_SUB, body_color)
                elif treatment in SERIF_LINE_STYLES:
                    tok = _Token(word, text, SERIF_FONT, FS_HOOK if hook else FS_SERIF, accent_color)
                elif word.emphasis and treatment in {"mix", "underline", "plain"}:
                    if single_emphasis and _is_acronym(text):
                        tok = _Token(word, text, SANS_FONT, FS_ACRONYM, accent_color)
                    else:
                        emph_font = (
                            SERIF_ITALIC_FONT if self.caption_style == "editorial" else SERIF_FONT
                        )
                        tok = _Token(word, text, emph_font, FS_SERIF_INLINE, accent_color)
                else:
                    tok = _Token(word, text, SANS_FONT, FS_SANS, body_color)
                line.append(tok)
            k += count
            if treatment == "quote" and line:
                line[0].text = "“" + line[0].text
                line[-1].text = line[-1].text + "”"
            lines.append(line)
        return lines

    def _build_line(self, tokens: list[_Token], scale: float) -> tuple[list[TextSprite], list[float], float]:
        u = self.u
        sprites = [
            text_sprite(
                tok.text,
                _font(tok.font_path, round(tok.size * u * scale)),
                tok.color,
                u=u,
                shadow=self.shadow if tok.shadow else 0.0,
                # Playfair hairlines break up under video compression.
                stroke=1 if tok.font_path in (SERIF_FONT, SERIF_ITALIC_FONT) and u >= 0.9 else 0,
            )
            for tok in tokens
        ]
        xs: list[float] = []
        x = 0.0
        for i, sp in enumerate(sprites):
            if i > 0:
                x += min(sprites[i - 1].space, sp.space) * 1.02
            xs.append(x)
            x += sp.advance
        return sprites, xs, x

    def _layout(self, cap: Caption, index: int) -> list[Element]:
        u = self.u
        treatment = cap.treatment or "plain"
        if treatment == "blob":
            treatment = "tape"
        # Real background intelligence, judged fresh for this caption's own
        # on-screen window: white-on-white is a real failure mode on a bright
        # wall or open sky, so a bright moment gets the dark-ink/deep-gold
        # set instead of the usual white/cream — same shadow logic, inverted
        # so the halo still reads as contrast rather than a glow that blends
        # straight into the background.
        bright = self._is_bright(cap)
        self.shadow = 1.4 if bright else 1.0
        palette = self._palette_for(bright)
        lines = self._tokens(cap, index, palette)
        max_w = self.width * MAX_LINE_FRAC
        if treatment in {"tape", "chip", "bubble"}:
            max_w -= 2 * 64 * u
        elements: list[Element] = []
        use_chest = cap.screen_area == "chest" and self.baseline_chest is not None
        baseline = float(self.baseline_chest if use_chest else self.baseline)
        line_meta: list[tuple[float, float, float, float, float]] = []  # x0, x1, ink_top, ink_bottom, size
        prev_size = 0.0
        # Editorial theme: (t_in, cursor_x, cursor_y, color, font_path, size_px)
        # per word, in reveal order, consumed after the loop to build the
        # trailing blinking cursor.
        typewriter_marks: list[tuple[float, float, float, tuple, str, float]] = []
        for li, tokens in enumerate(lines):
            if not tokens:
                continue
            sprites, xs, width = self._build_line(tokens, 1.0)
            scale = 1.0
            if width > max_w:
                scale = max(0.55, max_w / width)
                sprites, xs, width = self._build_line(tokens, scale)
            size = max(tok.size for tok in tokens) * u * scale
            if li > 0:
                baseline += 1.10 * size + 0.55 * prev_size
            prev_size = size
            x_left = (self.width - width) / 2.0
            whole_line = treatment in SERIF_LINE_STYLES or treatment in {"tape", "chip", "bubble"} or (
                treatment == "stack" and li == 0
            )
            typewriter = (
                self.caption_style == "editorial"
                and treatment in TYPEWRITER_TREATMENTS
                and not whole_line
            )
            # A pop applies to the whole phrase when it reveals as one unit
            # (serif/oval/tape/chip), or only to its emphasized cream word
            # inside an ordinary sentence — the rest of the line still just
            # rises, so the "blink" reads as a highlight, not a glitch.
            pop_line = cap.mood in {"surprise", "excited"}
            for k, (tok, sp, dx) in enumerate(zip(tokens, sprites, xs)):
                if whole_line:
                    t_in = cap.start + 0.05 * k + (0.14 if treatment in {"tape", "chip", "bubble"} else 0.0)
                    dur, rise = LINE_IN, 16 * u
                elif typewriter:
                    t_in = max(cap.start, tok.word.start)
                    dur, rise = 0.0, 0.0
                else:
                    t_in = max(cap.start, tok.word.start)
                    serif = tok.font_path in (SERIF_FONT, SERIF_ITALIC_FONT)
                    dur, rise = (0.22, 14 * u) if serif else (WORD_IN, 11 * u)
                t_in = min(t_in, max(cap.start, cap.end - 0.2))
                if typewriter:
                    typewriter_marks.append(
                        (t_in, x_left + dx + sp.advance, baseline, tok.color, tok.font_path, tok.size * scale)
                    )
                if pop_line and (whole_line or tok.word.emphasis):
                    cx = x_left + dx - sp.ox + sp.arr.shape[1] / 2
                    cy = baseline - sp.oy + sp.arr.shape[0] / 2
                    elements.append(
                        self._element(
                            t_in=t_in,
                            dur=POP_DUR,
                            t_out=cap.end,
                            mode="frames",
                            frames=[_scaled(sp.arr, s) for s in POP_SCALES],
                            x=int(round(cx)),
                            y=int(round(cy)),
                        )
                    )
                else:
                    elements.append(
                        self._element(
                            t_in=t_in,
                            dur=dur,
                            t_out=cap.end,
                            mode="rise",
                            arr=sp.arr,
                            x=int(round(x_left + dx - sp.ox)),
                            y=int(round(baseline - sp.oy)),
                            rise=rise,
                        )
                    )
            ink_top = baseline + min(sp.ink_top for sp in sprites)
            ink_bottom = baseline + max(sp.ink_bottom for sp in sprites)
            line_meta.append((x_left, x_left + width, ink_top, ink_bottom, size))

        if typewriter_marks:
            elements.extend(self._typewriter_cursor(cap, typewriter_marks))

        if not line_meta:
            return elements
        first = line_meta[0]
        if treatment == "oval":
            elements[:0] = self._oval(cap, first, palette[2])
        elif treatment == "underline":
            elements.extend(self._underline(cap, line_meta[-1], lines[-1], palette[2]))
        elif treatment == "tape":
            elements[:0] = self._tape(cap, first, index)
        elif treatment == "chip":
            elements[:0] = self._chip(cap, first, index)
        elif treatment == "bubble":
            elements[:0] = self._bubble(cap, first)
        if cap.icon and cap.icon != "none":
            elements[:0] = self._icon_badge(cap, first, cap.icon)
        return elements

    def _typewriter_cursor(
        self,
        cap: Caption,
        marks: list[tuple[float, float, float, tuple[int, int, int], str, float]],
    ) -> list[Element]:
        """A blinking text cursor trailing each word as it's 'typed' — the
        editorial theme's signature reveal. Built from a real glyph in the
        same font/size/color as the word it follows, so it sits on the exact
        baseline rather than a drawn rectangle. Implemented as a run of
        zero-duration elements covering only the blink-ON windows — the
        blink-OFF gaps between them simply have nothing to draw, so no change
        to the element/draw machinery is needed."""
        u = self.u
        elements: list[Element] = []
        for i, (t_in, cx, cy, color, font_path, size_px) in enumerate(marks):
            t_next = marks[i + 1][0] if i + 1 < len(marks) else cap.end
            t_end = min(t_next, t_in + CURSOR_MAX_S, cap.end)
            if t_end <= t_in:
                continue
            sp = text_sprite("|", _font(font_path, round(size_px * u)), color, u=u, shadow=self.shadow)
            x = int(round(cx + CURSOR_GAP * u - sp.ox))
            y = int(round(cy - sp.oy))
            t = t_in
            on = True
            while t < t_end - 1e-6:
                span = CURSOR_ON if on else CURSOR_OFF
                seg_end = min(t + span, t_end)
                if on:
                    elements.append(
                        self._element(
                            t_in=t, dur=0.0, t_out=seg_end, mode="rise",
                            arr=sp.arr, x=x, y=y, rise=0.0,
                        )
                    )
                t = seg_end
                on = not on
        return elements

    # ---------------------------------------------------------- decorations
    def _oval(self, cap: Caption, meta, hair: tuple[int, int, int] = HAIRLINE) -> list[Element]:
        u = self.u
        x0, x1, top, bottom, size = meta
        cx = (x0 + x1) / 2
        cy = (top + bottom) / 2
        rx = (x1 - x0) / 2 + 58 * u
        ry = (bottom - top) / 2 + 60 * u
        rx = min(rx, self.width / 2 - 14 * u)
        tilt = -4.0
        ss = 3
        stroke = max(2.4, 3.0 * u)
        pad = int(30 * u)
        W = int(2 * rx + 2 * pad)
        H = int(2 * ry + 2 * pad)
        steps = 14
        start_deg = 205.0
        sweep_total = 372.0
        frames: list[np.ndarray] = []
        for s in range(1, steps + 1):
            sweep = sweep_total * s / steps
            img = Image.new("L", (W * ss, H * ss), 0)
            d = ImageDraw.Draw(img)
            box = [pad * ss, pad * ss, (W - pad) * ss, (H - pad) * ss]
            d.arc(box, start=start_deg, end=start_deg + sweep, fill=255, width=int(stroke * ss))
            img = img.rotate(tilt, resample=Image.BICUBIC)
            small = img.resize((W, H), Image.LANCZOS)
            m = np.asarray(small, np.float32) / 255.0
            shade = np.asarray(small.filter(ImageFilter.GaussianBlur(3 * u)), np.float32) / 255.0
            alpha = 1.0 - (1.0 - m * 0.96) * (1.0 - shade * 0.34 * self.shadow)
            arr = np.zeros((H, W, 4), np.float32)
            col = np.array(hair, np.float32) / 255.0
            arr[..., :3] = col * (m * 0.96)[..., None]
            arr[..., 3] = alpha
            frames.append(arr)
        t0 = cap.start + 0.06
        els = [
            self._element(
                t_in=t0, dur=0.42, t_out=cap.end, mode="frames",
                frames=frames, x=int(cx), y=int(cy),
            )
        ]
        spark_a = _astroid(max(8, int(25 * u)), hair)
        spark_b = _astroid(max(9, int(31 * u)), hair)
        rad = math.radians(tilt)
        for arr, (ex, ey), delay in (
            (spark_a, (-0.93 * rx, -0.50 * ry), 0.34),
            (spark_b, (0.90 * rx, 0.58 * ry), 0.44),
        ):
            px = cx + ex * math.cos(rad) - ey * math.sin(rad)
            py = cy + ex * math.sin(rad) + ey * math.cos(rad)
            pops = [_scaled(arr, s) for s in (0.25, 0.6, 0.95, 1.18, 1.08, 1.0)]
            els.append(
                self._element(
                    t_in=t0 + delay, dur=0.30, t_out=cap.end, mode="frames",
                    frames=pops, x=int(px), y=int(py),
                )
            )
        return els

    def _underline(
        self, cap: Caption, meta, tokens: list[_Token], hair: tuple[int, int, int] = HAIRLINE
    ) -> list[Element]:
        u = self.u
        x0, x1, top, bottom, size = meta
        width = (x1 - x0) + 70 * u
        y = bottom + 24 * u
        thick = max(1.6, 2.2 * u)
        gap = 22 * u
        ss = 4
        pad = int(10 * u)
        W = int(width + 2 * pad)
        H = int(2 * pad + thick * 3)
        img = Image.new("L", (W * ss, H * ss), 0)
        d = ImageDraw.Draw(img)
        mid = W * ss / 2
        yy = H * ss / 2
        half_t = thick * ss / 2
        d.rectangle([pad * ss, yy - half_t, mid - gap * ss, yy + half_t], fill=255)
        d.rectangle([mid + gap * ss, yy - half_t, (W - pad) * ss, yy + half_t], fill=255)
        small = img.resize((W, H), Image.LANCZOS)
        m = np.asarray(small, np.float32) / 255.0
        shade = np.asarray(small.filter(ImageFilter.GaussianBlur(2.5 * u)), np.float32) / 255.0
        arr = np.zeros((H, W, 4), np.float32)
        arr[..., :3] = (np.array(hair, np.float32) / 255.0) * m[..., None]
        arr[..., 3] = 1.0 - (1.0 - m) * (1.0 - shade * 0.35 * self.shadow)
        emph = [tok.word.start for tok in tokens if tok.word.emphasis]
        t0 = max(cap.start + 0.12, min(emph) if emph else cap.start + 0.12)
        t0 = min(t0, max(cap.start, cap.end - 0.45))
        cx = (x0 + x1) / 2
        star = _astroid(max(7, int(17 * u)), hair)
        pops = [_scaled(star, s) for s in (0.25, 0.6, 0.95, 1.2, 1.06, 1.0)]
        return [
            self._element(
                t_in=t0, dur=0.36, t_out=cap.end, mode="grow",
                arr=arr, x=int(cx - W / 2), y=int(y - H / 2),
            ),
            self._element(
                t_in=t0 + 0.22, dur=0.28, t_out=cap.end, mode="frames",
                frames=pops, x=int(cx), y=int(y),
            ),
        ]

    def _icon_glyph(self, icon: str, size: int, u: float) -> np.ndarray | None:
        """A small premultiplied-RGBA glyph: a currency/@ character for the
        text-based icons, a hand-drawn shape for the rest."""
        color = CHIP_SPARK
        if icon in {"money", "social", "question"}:
            char = {"social": "@", "question": "?"}.get(icon, "₹")
            sprite = text_sprite(char, _font(SANS_FONT, max(10, int(size * 1.4))), color, u=u, shadow=0.0)
            return _trim_alpha(sprite.arr)
        if icon == "idea":
            return _astroid(max(6, size // 2), color, glow=False)
        if icon == "star":
            return _star5(max(6, int(size * 0.56)), color)
        ss = 4
        D = max(8, size)
        img = Image.new("L", (D * ss, D * ss), 0)
        d = ImageDraw.Draw(img)
        cx, cy = D * ss / 2, D * ss / 2
        stroke = max(2, int(D * ss * 0.09))
        if icon == "growth":
            w, h = D * ss * 0.5, D * ss * 0.5
            d.polygon(
                [
                    (cx, cy - h * 0.55), (cx - w * 0.42, cy - h * 0.02), (cx - w * 0.16, cy - h * 0.02),
                    (cx - w * 0.16, cy + h * 0.55), (cx + w * 0.16, cy + h * 0.55),
                    (cx + w * 0.16, cy - h * 0.02), (cx + w * 0.42, cy - h * 0.02),
                ],
                fill=255,
            )
        elif icon == "video":
            r = D * ss * 0.34
            d.polygon([(cx - r * 0.55, cy - r), (cx - r * 0.55, cy + r), (cx + r * 1.05, cy)], fill=255)
        elif icon == "check":
            pts = [(D * ss * 0.22, D * ss * 0.52), (D * ss * 0.42, D * ss * 0.74), (D * ss * 0.80, D * ss * 0.26)]
            d.line(pts, fill=255, width=max(2, int(D * ss * 0.13)), joint="curve")
        elif icon == "warning":
            r = D * ss * 0.40
            top = (cx, cy - r)
            bl = (cx - r * 0.92, cy + r * 0.72)
            br = (cx + r * 0.92, cy + r * 0.72)
            d.polygon([top, bl, br], outline=255, width=stroke)
            d.line([(cx, cy - r * 0.18), (cx, cy + r * 0.22)], fill=255, width=stroke)
            d.ellipse(
                [cx - stroke * 0.7, cy + r * 0.42, cx + stroke * 0.7, cy + r * 0.42 + stroke * 1.4],
                fill=255,
            )
        elif icon == "time":
            r = D * ss * 0.36
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=stroke)
            d.line([(cx, cy), (cx, cy - r * 0.55)], fill=255, width=stroke)
            d.line([(cx, cy), (cx + r * 0.42, cy + r * 0.20)], fill=255, width=stroke)
        elif icon == "target":
            for frac in (1.0, 0.55):
                rr = D * ss * 0.38 * frac
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=255, width=stroke)
            dot = D * ss * 0.08
            d.ellipse([cx - dot, cy - dot, cx + dot, cy + dot], fill=255)
        elif icon == "fire":
            r = D * ss * 0.40
            d.polygon(
                [
                    (cx, cy - r), (cx + r * 0.55, cy - r * 0.15), (cx + r * 0.42, cy + r * 0.35),
                    (cx + r * 0.18, cy + r * 0.55), (cx + r * 0.30, cy + r * 0.05),
                    (cx, cy + r * 0.35), (cx - r * 0.30, cy + r * 0.05), (cx - r * 0.18, cy + r * 0.55),
                    (cx - r * 0.42, cy + r * 0.35), (cx - r * 0.55, cy - r * 0.15),
                ],
                fill=255,
            )
        elif icon == "heart":
            r = D * ss * 0.24
            d.ellipse([cx - 2 * r + r * 0.15, cy - r * 1.05, cx + r * 0.15, cy + r * 0.95], fill=255)
            d.ellipse([cx - r * 0.15, cy - r * 1.05, cx + 2 * r - r * 0.15, cy + r * 0.95], fill=255)
            d.polygon(
                [
                    (cx - 1.85 * r, cy + r * 0.25), (cx + 1.85 * r, cy + r * 0.25), (cx, cy + r * 2.05),
                ],
                fill=255,
            )
        elif icon == "lock":
            bw, bh = D * ss * 0.58, D * ss * 0.46
            bx0, by0 = cx - bw / 2, cy - bh / 2 + D * ss * 0.10
            d.rounded_rectangle([bx0, by0, bx0 + bw, by0 + bh], radius=D * ss * 0.08, fill=255)
            sh_r = D * ss * 0.20
            d.arc(
                [cx - sh_r, by0 - 2 * sh_r + D * ss * 0.06, cx + sh_r, by0 + D * ss * 0.06],
                start=180, end=360, fill=255, width=stroke,
            )
            dot = D * ss * 0.055
            d.ellipse([cx - dot, cy + bh * 0.06 - dot, cx + dot, cy + bh * 0.06 + dot], fill=0)
        else:
            return None
        small = img.resize((D, D), Image.LANCZOS)
        m = np.asarray(small, np.float32) / 255.0
        out = np.zeros((D, D, 4), np.float32)
        col = np.array(color, np.float32) / 255.0
        out[..., :3] = col[None, None, :] * m[..., None]
        out[..., 3] = m
        return out

    def _icon_badge(self, cap: Caption, meta, icon: str) -> list[Element]:
        """A small round badge beside the line — sitting to its left, level
        with the text, so it never needs headroom the caption band doesn't
        have. A contextual accent the director chose, not decoration on
        every line."""
        u = self.u
        x0, x1, top, bottom, size = meta
        diameter = max(44, int(size * 0.80))
        gap = int(16 * u)
        cx = x0 - gap - diameter / 2
        cy = (top + bottom) / 2

        ss = 4
        pad = int(10 * u)
        D = diameter + 2 * pad
        mask = Image.new("L", (D * ss, D * ss), 0)
        ImageDraw.Draw(mask).ellipse([pad * ss, pad * ss, (D - pad) * ss, (D - pad) * ss], fill=255)
        small = mask.resize((D, D), Image.LANCZOS)
        m = np.asarray(small, np.float32) / 255.0
        shade = np.asarray(small.filter(ImageFilter.GaussianBlur(8 * u)), np.float32) / 255.0
        shade = np.roll(shade, max(1, int(5 * u)), axis=0)
        color = CHIP_COLORS[0]
        arr = np.zeros((D, D, 4), np.float32)
        col = np.array(color, np.float32) / 255.0
        arr[..., :3] = col[None, None, :] * m[..., None]
        arr[..., 3] = m + (shade * 0.30 * self.shadow) * (1.0 - m)

        glyph = self._icon_glyph(icon, int(diameter * 0.5), u)
        if glyph is not None:
            gh, gw = glyph.shape[:2]
            if gw <= D and gh <= D:
                gx, gy = (D - gw) // 2, (D - gh) // 2
                region = arr[gy : gy + gh, gx : gx + gw]
                ga = glyph[..., 3:4]
                region[..., :3] = glyph[..., :3] + region[..., :3] * (1.0 - ga)
                region[..., 3:4] = glyph[..., 3:4] + region[..., 3:4] * (1.0 - ga)

        frames = [_scaled(arr, s) for s in POP_SCALES]
        return [
            self._element(
                t_in=cap.start + 0.04, dur=POP_DUR, t_out=cap.end, mode="frames",
                frames=frames, x=int(cx), y=int(cy),
            )
        ]

    def _chip(self, cap: Caption, meta, index: int) -> list[Element]:
        """Solid rounded pill behind a concrete-figure line, plus a small
        sparkle at its corner — a premium highlight, not a torn sticker."""
        u = self.u
        x0, x1, top, bottom, size = meta
        pad_x = int(38 * u)
        pad_y = int(22 * u)
        w = int((x1 - x0) + 2 * pad_x)
        h = int((bottom - top) + 2 * pad_y)
        cx = (x0 + x1) / 2
        cy = (top + bottom) / 2
        radius = int(h * 0.40)
        ss = 3
        pad = int(16 * u)
        W = w + 2 * pad
        H = h + 2 * pad
        mask = Image.new("L", (W * ss, H * ss), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [pad * ss, pad * ss, (W - pad) * ss, (H - pad) * ss],
            radius=radius * ss,
            fill=255,
        )
        small = mask.resize((W, H), Image.LANCZOS)
        m = np.asarray(small, np.float32) / 255.0
        shade = np.asarray(small.filter(ImageFilter.GaussianBlur(11 * u)), np.float32) / 255.0
        shade = np.roll(shade, max(1, int(6 * u)), axis=0)
        color = CHIP_COLORS[index % len(CHIP_COLORS)]
        arr = np.zeros((H, W, 4), np.float32)
        col = np.array(color, np.float32) / 255.0
        arr[..., :3] = col[None, None, :] * m[..., None]
        arr[..., 3] = m + (shade * 0.32 * self.shadow) * (1.0 - m)

        els = [
            self._element(
                t_in=cap.start, dur=0.22, t_out=cap.end, mode="wipe",
                arr=arr, x=int(cx - W / 2), y=int(cy - H / 2),
            )
        ]
        spark = _astroid(max(7, int(16 * u)), CHIP_SPARK)
        pops = [_scaled(spark, s) for s in (0.2, 0.55, 0.9, 1.15, 1.0)]
        sx = cx + w / 2 - pad_x * 0.30
        sy = cy - h / 2 + pad_y * 0.30
        els.append(
            self._element(
                t_in=cap.start + 0.16, dur=0.24, t_out=cap.end, mode="frames",
                frames=pops, x=int(sx), y=int(sy),
            )
        )
        return els

    def _bubble(self, cap: Caption, meta) -> list[Element]:
        """The premium theme's CTA badge: a near-black rounded pill with a
        thin cream outline and a star — a direct ask of the viewer, styled
        like a subscribe/follow prompt rather than a stat callout."""
        u = self.u
        x0, x1, top, bottom, size = meta
        pad_x = int(40 * u)
        pad_y = int(24 * u)
        w = int((x1 - x0) + 2 * pad_x)
        h = int((bottom - top) + 2 * pad_y)
        cx = (x0 + x1) / 2
        cy = (top + bottom) / 2
        radius = int(h * 0.44)
        ss = 3
        pad = int(18 * u)
        W = w + 2 * pad
        H = h + 2 * pad
        stroke = max(2, int(2.6 * u))
        fill_mask = Image.new("L", (W * ss, H * ss), 0)
        ImageDraw.Draw(fill_mask).rounded_rectangle(
            [pad * ss, pad * ss, (W - pad) * ss, (H - pad) * ss], radius=radius * ss, fill=255,
        )
        line_mask = Image.new("L", (W * ss, H * ss), 0)
        ImageDraw.Draw(line_mask).rounded_rectangle(
            [pad * ss, pad * ss, (W - pad) * ss, (H - pad) * ss],
            radius=radius * ss, outline=255, width=stroke * ss,
        )
        fill_small = fill_mask.resize((W, H), Image.LANCZOS)
        line_small = line_mask.resize((W, H), Image.LANCZOS)
        fm = np.asarray(fill_small, np.float32) / 255.0
        lm = np.asarray(line_small, np.float32) / 255.0
        shade = np.asarray(fill_small.filter(ImageFilter.GaussianBlur(11 * u)), np.float32) / 255.0
        shade = np.roll(shade, max(1, int(6 * u)), axis=0)
        arr = np.zeros((H, W, 4), np.float32)
        outline_color = EDITORIAL_GREEN if self.caption_style == "editorial" else BUBBLE_OUTLINE
        fill_col = np.array(BUBBLE_FILL, np.float32) / 255.0
        line_col = np.array(outline_color, np.float32) / 255.0
        arr[..., :3] = fill_col[None, None, :] * fm[..., None]
        arr[..., 3] = fm + (shade * 0.34 * self.shadow) * (1.0 - fm)
        # Outline drawn over the fill, premultiplied-over.
        a = lm[..., None]
        arr[..., :3] = line_col[None, None, :] * a + arr[..., :3] * (1.0 - a)
        arr[..., 3:4] = np.maximum(arr[..., 3:4], a)

        els = [
            self._element(
                t_in=cap.start, dur=0.24, t_out=cap.end, mode="wipe",
                arr=arr, x=int(cx - W / 2), y=int(cy - H / 2),
            )
        ]
        star = _star5(max(7, int(15 * u)), outline_color)
        pops = [_scaled(star, s) for s in (0.2, 0.6, 1.05, 0.85, 1.0)]
        sx = cx - w / 2 + pad_x * 0.30
        sy = cy - h / 2 + pad_y * 0.30
        els.append(
            self._element(
                t_in=cap.start + 0.18, dur=0.26, t_out=cap.end, mode="frames",
                frames=pops, x=int(sx), y=int(sy),
            )
        )
        return els

    def _tape(self, cap: Caption, meta, index: int) -> list[Element]:
        u = self.u
        x0, x1, top, bottom, size = meta
        rng = random.Random(1000 + index)
        w = (x1 - x0) + 2 * 64 * u
        h = max((bottom - top) + 2 * 40 * u, 1.45 * size)
        cx = (x0 + x1) / 2
        cy = (top + bottom) / 2 + 2 * u
        ss = 2
        pad = int(26 * u)
        W = int(w + 2 * pad)
        H = int(h + 2 * pad)
        pts: list[tuple[float, float]] = []
        left, right = pad, W - pad
        upper, lower = pad, H - pad
        tooth = 7 * u
        # top edge, slight wobble
        n_top = max(6, int(w / (60 * u)))
        for i in range(n_top + 1):
            pts.append((left + (right - left) * i / n_top, upper + rng.uniform(-1.4, 1.4) * u))
        # right torn edge
        y = upper
        while y < lower:
            pts.append((right + rng.uniform(-tooth, tooth * 0.6), y))
            y += rng.uniform(5, 10) * u
        for i in range(n_top, -1, -1):
            pts.append((left + (right - left) * i / n_top, lower + rng.uniform(-1.4, 1.4) * u))
        y = lower
        while y > upper:
            pts.append((left + rng.uniform(-tooth * 0.6, tooth), y))
            y -= rng.uniform(5, 10) * u
        mask = Image.new("L", (W * ss, H * ss), 0)
        ImageDraw.Draw(mask).polygon([(px * ss, py * ss) for px, py in pts], fill=255)
        mask = mask.rotate(-1.6, resample=Image.BICUBIC).resize((W, H), Image.LANCZOS)
        m = np.asarray(mask, np.float32) / 255.0
        noise = np.random.default_rng(index).normal(0, 0.018, size=m.shape).astype(np.float32)
        shade = np.asarray(mask.filter(ImageFilter.GaussianBlur(9 * u)), np.float32) / 255.0
        shade = np.roll(shade, max(1, int(5 * u)), axis=0)
        fill = (np.array(TAPE_FILL, np.float32) / 255.0)[None, None, :] * (1.0 + noise[..., None])
        a_fill = m * 0.97
        arr = np.zeros((H, W, 4), np.float32)
        arr[..., :3] = np.clip(fill, 0, 1) * a_fill[..., None]
        arr[..., 3] = a_fill + (shade * 0.28) * (1.0 - a_fill)
        return [
            self._element(
                t_in=cap.start, dur=0.26, t_out=cap.end, mode="wipe",
                arr=arr, x=int(cx - W / 2), y=int(cy - H / 2),
            )
        ]

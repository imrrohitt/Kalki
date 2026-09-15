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

WHITE = (255, 255, 255)
CREAM = (255, 246, 194)
TAPE_FILL = (176, 196, 170)
TAPE_INK = (38, 26, 19)
HAIRLINE = (240, 238, 232)

# Font sizes in px on a 1080-wide frame.
FS_SANS = 58
FS_SERIF_INLINE = 80
FS_ACRONYM = 68
FS_SERIF = 90
FS_HOOK = 104
FS_SUB = 56
FS_TAPE = 94
MAX_LINE_FRAC = 0.84

WORD_IN = 0.16
LINE_IN = 0.30
EXIT = 0.08

SERIF_LINE_STYLES = {"serif", "oval", "quote"}
DECORATED = {"oval", "underline", "tape", "blob"}


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
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.u = width / 1080.0
        self.shadow = 1.4 if bright_background else 1.0
        self.captions: list[Caption] = sorted(timeline.captions, key=lambda c: c.start)
        u = self.u
        head = head_top if head_top else int(height * 0.30)
        baseline = head - int(165 * u)
        baseline = max(int(height * 0.12), min(baseline, int(height * 0.25)))
        baseline = max(baseline, int(250 * u))
        top = max(0, baseline - int(250 * u))
        bottom = min(height, baseline + int(190 * u))
        self.band_top = top - (top % 2)
        self.band_height = (bottom - self.band_top) + ((bottom - self.band_top) % 2)
        self.baseline = baseline - self.band_top
        self._cache: dict[int, list[Element]] = {}
        self._uid = 0

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

    def _tokens(self, cap: Caption, index: int) -> list[list[_Token]]:
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
                if treatment == "tape":
                    tok = _Token(word, text, SERIF_FONT, FS_TAPE, TAPE_INK, shadow=False)
                elif treatment == "stack":
                    if li == 0:
                        tok = _Token(word, text, SERIF_FONT, FS_HOOK, CREAM)
                    else:
                        tok = _Token(word, text, SANS_FONT, FS_SUB, WHITE)
                elif treatment in SERIF_LINE_STYLES:
                    tok = _Token(word, text, SERIF_FONT, FS_HOOK if hook else FS_SERIF, CREAM)
                elif word.emphasis and treatment in {"mix", "underline", "plain"}:
                    if single_emphasis and _is_acronym(text):
                        tok = _Token(word, text, SANS_FONT, FS_ACRONYM, CREAM)
                    else:
                        tok = _Token(word, text, SERIF_FONT, FS_SERIF_INLINE, CREAM)
                else:
                    tok = _Token(word, text, SANS_FONT, FS_SANS, WHITE)
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
                # Playfair Regular hairlines break up under video compression.
                stroke=1 if tok.font_path == SERIF_FONT and u >= 0.9 else 0,
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
        lines = self._tokens(cap, index)
        max_w = self.width * MAX_LINE_FRAC
        if treatment == "tape":
            max_w -= 2 * 64 * u
        elements: list[Element] = []
        baseline = float(self.baseline)
        line_meta: list[tuple[float, float, float, float, float]] = []  # x0, x1, ink_top, ink_bottom, size
        prev_size = 0.0
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
            whole_line = treatment in SERIF_LINE_STYLES or treatment == "tape" or (
                treatment == "stack" and li == 0
            )
            for k, (tok, sp, dx) in enumerate(zip(tokens, sprites, xs)):
                if whole_line:
                    t_in = cap.start + 0.05 * k + (0.14 if treatment == "tape" else 0.0)
                    dur, rise = LINE_IN, 16 * u
                else:
                    t_in = max(cap.start, tok.word.start)
                    serif = tok.font_path == SERIF_FONT
                    dur, rise = (0.22, 14 * u) if serif else (WORD_IN, 11 * u)
                t_in = min(t_in, max(cap.start, cap.end - 0.2))
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

        if not line_meta:
            return elements
        first = line_meta[0]
        if treatment == "oval":
            elements[:0] = self._oval(cap, first)
        elif treatment == "underline":
            elements.extend(self._underline(cap, line_meta[-1], lines[-1]))
        elif treatment == "tape":
            elements[:0] = self._tape(cap, first, index)
        return elements

    # ---------------------------------------------------------- decorations
    def _oval(self, cap: Caption, meta) -> list[Element]:
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
            col = np.array(HAIRLINE, np.float32) / 255.0
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
        spark_a = _astroid(max(8, int(25 * u)), HAIRLINE)
        spark_b = _astroid(max(9, int(31 * u)), HAIRLINE)
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

    def _underline(self, cap: Caption, meta, tokens: list[_Token]) -> list[Element]:
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
        arr[..., :3] = (np.array(HAIRLINE, np.float32) / 255.0) * m[..., None]
        arr[..., 3] = 1.0 - (1.0 - m) * (1.0 - shade * 0.35 * self.shadow)
        emph = [tok.word.start for tok in tokens if tok.word.emphasis]
        t0 = max(cap.start + 0.12, min(emph) if emph else cap.start + 0.12)
        t0 = min(t0, max(cap.start, cap.end - 0.45))
        cx = (x0 + x1) / 2
        star = _astroid(max(7, int(17 * u)), HAIRLINE)
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


def band_is_bright(frames: list[np.ndarray]) -> bool:
    """True when the wall behind the captions is light enough to need more shadow."""
    if not frames:
        return False
    lum = [float(np.mean(f[..., :3] @ np.array([0.2126, 0.7152, 0.0722]))) for f in frames]
    return float(np.median(lum)) > 125.0

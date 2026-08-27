from __future__ import annotations

import re
from pathlib import Path

from app.captions.heuristic import explode_caption_timeline
from app.captions.models import Caption, CaptionTimeline, CaptionWord
from app.editorial.models import GraphicBeat, GraphicBullet, GraphicNode
from app.renderer import design as D
from app.renderer.split import split_panel_sizes


def _ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        s += 1
        cs = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _caption_words(caption: Caption) -> list[CaptionWord]:
    if caption.words:
        return list(caption.words)
    display = (caption.text or "").replace("\\n", " ").strip()
    return [
        CaptionWord(text=token, start=caption.start, end=caption.end)
        for token in display.split()
        if token.strip()
    ]


def _wrap_caption_words(
    caption: Caption, *, max_chars: int, max_lines: int = 3
) -> list[list[CaptionWord]]:
    """Greedy wrap so a line never runs off the 1080-wide frame."""
    words = _caption_words(caption)
    if not words:
        return []
    lines: list[list[CaptionWord]] = []
    current: list[CaptionWord] = []
    for word in words:
        trial = " ".join(w.text for w in current + [word])
        if current and len(trial) > max_chars:
            lines.append(current)
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(current)
    # If we overflow max_lines, widen the wrap until it fits, then the
    # font-size fitter shrinks type so nothing runs off the frame.
    if len(lines) > max_lines:
        for chars in range(max_chars + 2, max_chars + 18, 2):
            wider: list[list[CaptionWord]] = []
            cur: list[CaptionWord] = []
            for word in words:
                trial = " ".join(w.text for w in cur + [word])
                if cur and len(trial) > chars:
                    wider.append(cur)
                    cur = [word]
                else:
                    cur.append(word)
            if cur:
                wider.append(cur)
            if len(wider) <= max_lines:
                return [line for line in wider if line]
        lines = wider
    return [line for line in lines if line]


def _word_delay_ms(caption: Caption, word: CaptionWord, index: int) -> int:
    delta = int(round((float(word.start) - float(caption.start)) * 1000))
    if delta >= 40:
        return min(max(delta, 0), 900)
    return min(index * 55, 360)


def _word_motion(
    *,
    delay: int,
    emphasis: bool,
    accent: str,
    body_color: str,
    body_fs: int,
    hot_fs: int,
    peak: int = 124,
) -> str:
    """Per-word pop. Emphasis lands bigger, with a bounce, in the accent color."""
    if emphasis:
        t1 = delay
        t2 = delay + 140
        t3 = delay + 260
        return (
            rf"{{\fs{hot_fs}\b1\c{accent}&\fscx78\fscy78\alpha&HFF&"
            rf"\t({t1},{t2},0.35,\alpha&H00&\fscx{peak}\fscy{peak})"
            rf"\t({t2},{t3},0.5,\fscx100\fscy100)}}"
        )
    t1 = delay
    t2 = delay + 120
    return (
        rf"{{\fs{body_fs}\c{body_color}&\fscx88\fscy88\alpha&HFF&"
        rf"\t({t1},{t2},0.4,\alpha&H00&\fscx100\fscy100)}}"
    )


def _kinetic_line(
    line_words: list[CaptionWord],
    caption: Caption,
    *,
    uppercase: bool,
    accent: str,
    body_color: str,
    body_fs: int,
    hot_fs: int,
    index_offset: int,
    peak: int = 124,
) -> str:
    parts: list[str] = []
    for i, word in enumerate(line_words):
        token = word.text.upper() if uppercase else word.text
        delay = _word_delay_ms(caption, word, index_offset + i)
        parts.append(
            _word_motion(
                delay=delay,
                emphasis=word.emphasis,
                accent=accent,
                body_color=body_color,
                body_fs=body_fs,
                hot_fs=hot_fs,
                peak=peak,
            )
            + _escape_ass(token)
        )
    return " ".join(parts)


# Overlay / split / full: (body fs, emphasis fs). Overlay sits above the
# head so type is smaller and wrap is tighter; emphasis still reads bigger.
_CAPTION_FS = {
    "overlay": (72, 94),
    "split": (78, 112),
    "full": (70, 102),
}


def _overlay_caption_y(head_top: int, height: int = 1920) -> int:
    """Top of the caption block, in the empty wall space above the head."""
    _ = height
    return max(64, min(96, head_top // 5))


def _caption_pop(uppercase: bool, *, split: bool = False) -> str:
    # Exit fade only — word pops handle the entrance so the line does not
    # compound a second scale animation.
    if uppercase:
        return r"{\fad(0,110)}"
    if split:
        return r"{\fad(0,90)}"
    return r"{\fad(0,100)}"


def _styled_caption_text(
    caption: Caption,
    *,
    uppercase: bool,
    split: bool = False,
    layout: str = "overlay",
    seam_y: int = 720,
    theme: D.Theme | None = None,
    overlay_y: int = 360,
    head_top: int = 520,
) -> str:
    th = theme or D.get_theme(None)
    layout = layout or ("split" if split else "overlay")
    stacked = layout in {"split", "full"}
    body_fs, hot_fs = _CAPTION_FS.get(layout, _CAPTION_FS["overlay"])
    max_chars = 14 if stacked else 16
    wrapped = _wrap_caption_words(caption, max_chars=max_chars, max_lines=3)
    longest = max(
        (" ".join(w.text for w in line) for line in wrapped),
        default="",
        key=len,
    )
    if layout == "overlay" and longest:
        usable = D.CANVAS_W - 200
        body_fs = _fit_fs(longest.upper() if uppercase else longest, body_fs, usable)
        hot_fs = min(hot_fs, body_fs + 22)
        hot_fs = _fit_fs(
            max((w.text for line in wrapped for w in line if w.emphasis), default="W"),
            hot_fs,
            usable,
        )
    if layout == "full":
        accent = th.accent
        body_color = th.ink if th.name in {"paper", "ivory"} else "&H00FFFFFF"
    else:
        accent = th.caption_emphasis
        body_color = "&H00FFFFFF"
    peak = 112 if layout == "overlay" else 124
    chunks: list[str] = []
    offset = 0
    for line_words in wrapped:
        chunks.append(
            _kinetic_line(
                line_words,
                caption,
                uppercase=uppercase,
                accent=accent,
                body_color=body_color,
                body_fs=body_fs,
                hot_fs=hot_fs,
                index_offset=offset,
                peak=peak,
            )
        )
        offset += len(line_words)
    body = r"\N".join(chunks)
    anim = _caption_pop(uppercase, split=stacked)
    if layout == "split":
        return rf"{{\an8\pos(540,{seam_y + 10})}}" + anim + body
    if layout == "full":
        return rf"{{\an8\pos(540,{D.CAPTION_Y_FULL})}}" + anim + body
    clip_bottom = max(overlay_y + 80, head_top - 24)
    return (
        rf"{{\an8\pos(540,{overlay_y})"
        rf"\clip(56,16,1024,{clip_bottom})}}"
        + anim
        + body
    )


def _dedupe_overlaps(captions: list[Caption]) -> list[Caption]:
    ordered = sorted(captions, key=lambda c: (c.start, c.end))
    fixed: list[Caption] = []
    last_end = -1.0
    for cap in ordered:
        start = max(float(cap.start), last_end + 0.04)
        end = float(cap.end)
        if end - start < 0.22:
            end = start + 0.22
        words = []
        for w in cap.words:
            w_start = max(float(w.start), start)
            w_end = min(max(float(w.end), w_start + 0.05), end)
            words.append(
                CaptionWord(
                    text=w.text,
                    start=w_start,
                    end=w_end,
                    emphasis=w.emphasis,
                )
            )
        fixed.append(
            Caption(
                start=start,
                end=end,
                text=cap.text,
                position=cap.position,
                animation=cap.animation,
                words=words,
            )
        )
        last_end = end
    return fixed


def _text_width(text: str, fs: int) -> int:
    return int(0.54 * fs * max(len(text), 1))


def _fit_fs(text: str, max_fs: int, usable: int = D.USABLE_W) -> int:
    fs = max_fs
    while fs > D.FS_TITLE_MIN and _text_width(text, fs) > usable:
        fs -= 2
    return fs


def _wrap_title(text: str, limit: int = 16) -> list[str]:
    """Prefer one line. Only wrap long 6+ word headlines."""
    words = text.split()
    if not words:
        return [text]
    if len(words) <= 5 or len(text) <= 28:
        return [text]
    mid = max(2, (len(words) + 1) // 2)
    first, second = " ".join(words[:mid]), " ".join(words[mid:])
    if not second:
        return [first]
    return [first, second]


def _clip(safe_y: int) -> str:
    return rf"\clip(0,8,{D.CANVAS_W},{safe_y})"


def _t(t1: int, t2: int, accel: float, tags: str) -> str:
    return rf"\t({t1},{t2},{accel},{tags})"


# ------------------------------------------------------------------ motion
# One restrained vocabulary: masked rise for headlines, fade-rise for
# secondary type, scale-settle for emphasis, drawn lines for structure.
# Entrances ease out via \t acceleration; exits are short fades.


def _mask_rise(
    x: int,
    y: int,
    *,
    an: int = 7,
    fs: int,
    x1: int,
    x2: int,
    safe_y: int,
    dur: int = D.DUR_BASE,
) -> str:
    """Editorial mask reveal: type rises into a clip window that opens upward."""
    y1 = max(y - 12, 0)
    y2 = min(y + int(fs * 1.35), safe_y)
    return (
        rf"{{\an{an}\move({x},{y + D.RISE_PX},{x},{y},0,{dur})"
        rf"\clip({x1},{y2 - 2},{x2},{y2})"
        + _t(0, dur, D.ACCEL_OUT, rf"\clip({x1},{y1},{x2},{y2})")
        + rf"\fad(0,{D.FADE_OUT_MS})}}"
    )


def _fade_rise(
    x: int,
    y: int,
    *,
    an: int = 7,
    safe_y: int,
    dur: int = D.DUR_BASE,
    drift: int = D.DRIFT_PX,
) -> str:
    return (
        rf"{{\an{an}\move({x},{y + drift},{x},{y},0,{dur})"
        rf"{_clip(safe_y)}\alpha&HFF&"
        + _t(0, int(dur * 0.7), D.ACCEL_SOFT, r"\alpha&H00&")
        + rf"\fad(0,{D.FADE_OUT_MS})}}"
    )


def _settle(
    x: int,
    y: int,
    *,
    an: int = 5,
    safe_y: int,
    dur: int = D.DUR_BASE,
    from_scale: int = 94,
) -> str:
    return (
        rf"{{\an{an}\pos({x},{y}){_clip(safe_y)}"
        rf"\fscx{from_scale}\fscy{from_scale}"
        + _t(0, dur, D.ACCEL_OUT, r"\fscx100\fscy100")
        + r"\alpha&HFF&"
        + _t(0, D.DUR_FAST, D.ACCEL_SOFT, r"\alpha&H00&")
        + rf"\fad(0,{D.FADE_OUT_MS})}}"
    )


def _slide_x(
    x: int,
    y: int,
    *,
    dx: int,
    an: int = 5,
    safe_y: int,
    dur: int = D.DUR_BASE,
) -> str:
    return (
        rf"{{\an{an}\move({x + dx},{y},{x},{y},0,{dur})"
        rf"{_clip(safe_y)}\alpha&HFF&"
        + _t(0, int(dur * 0.7), D.ACCEL_SOFT, r"\alpha&H00&")
        + rf"\fad(0,{D.FADE_OUT_MS})}}"
    )


def _hline(
    lines: list[str],
    s: float,
    e: float,
    x: int,
    y: int,
    w: int,
    *,
    h: int = 2,
    style: str = "GHair",
    dur: int = 360,
    layer: int = 2,
) -> None:
    """A horizontal rule that draws left to right."""
    text = (
        rf"{{\an7\pos({x},{y})\p1"
        rf"\clip({x},{y - 2},{x + 2},{y + h + 2})"
        + _t(0, dur, D.ACCEL_OUT, rf"\clip({x},{y - 2},{x + w},{y + h + 2})")
        + rf"\fad(0,{D.FADE_OUT_MS})}}"
        + rf"m 0 0 l {w} 0 l {w} {h} l 0 {h}"
    )
    lines.append(_dialogue(s, e, style, text, layer))


def _vline(
    lines: list[str],
    s: float,
    e: float,
    x: int,
    y1: int,
    y2: int,
    *,
    grow: str = "down",
    style: str = "GHair",
    dur: int = 360,
    w: int = 2,
    layer: int = 2,
) -> None:
    """A vertical rule that draws from one end."""
    h = y2 - y1
    if grow == "down":
        c0 = rf"\clip({x - 2},{y1},{x + w + 2},{y1 + 2})"
    else:
        c0 = rf"\clip({x - 2},{y2 - 2},{x + w + 2},{y2})"
    c1 = rf"\clip({x - 2},{y1},{x + w + 2},{y2})"
    text = (
        rf"{{\an7\pos({x},{y1})\p1{c0}"
        + _t(0, dur, D.ACCEL_OUT, c1)
        + rf"\fad(0,{D.FADE_OUT_MS})}}"
        + rf"m 0 0 l {w} 0 l {w} {h} l 0 {h}"
    )
    lines.append(_dialogue(s, e, style, text, layer))


def _accent_keywords(text: str, th: D.Theme) -> str:
    """Numbers, percentages, and money in headlines take the accent color.

    Gives hooks like "90% of AI projects fail" an immediate focal point.
    """
    out = []
    for token in text.split():
        safe = _escape_ass(token)
        if any(ch.isdigit() for ch in token) or "%" in token or "$" in token:
            out.append(rf"{{\c{th.accent}&}}{safe}{{\c{th.ink}&}}")
        else:
            out.append(safe)
    return " ".join(out)


def _title_motion(
    beat: GraphicBeat,
    x: int,
    y: int,
    *,
    fs: int,
    width: int,
    safe_y: int,
) -> str:
    if beat.motion == "fade":
        return _fade_rise(x, y, an=7, safe_y=safe_y)
    if beat.motion == "scale_in":
        return _settle(x, y, an=7, safe_y=safe_y, from_scale=96)
    x2 = min(x + width + 16, D.CANVAS_W - 8)
    return _mask_rise(x, y, fs=fs, x1=max(x - 8, 0), x2=x2, safe_y=safe_y)


def _on(beat: GraphicBeat, delay_ms: int) -> float:
    start = beat.start + max(0, delay_ms) / 1000.0
    latest = beat.end - 0.28
    if latest <= beat.start:
        return beat.start
    return min(start, latest)


def _dialogue(start: float, end: float, style: str, text: str, layer: int = 1) -> str:
    return (
        f"Dialogue: {layer},"
        f"{_ass_time(start)},{_ass_time(end)},"
        f"{style},,0,0,0,,{text}"
    )


def _fallback_bullets(beat: GraphicBeat) -> list[GraphicBullet]:
    if beat.kind in {"vs_split", "process"}:
        return []
    if beat.bullets:
        return list(beat.bullets)[:2]
    if beat.subtitle:
        return [GraphicBullet(text=beat.subtitle, delay_ms=0)]
    return []


def _header(
    beat: GraphicBeat,
    lines: list[str],
    *,
    y: int,
    safe_y: int,
    th: D.Theme,
) -> int:
    s, e = beat.start, beat.end
    x = D.MARGIN_X
    if beat.kicker:
        _hline(lines, _on(beat, 0), e, x, y + 14, 40, h=4, style="GRule", dur=280)
        lines.append(
            _dialogue(
                _on(beat, 40),
                e,
                "GKicker",
                _fade_rise(x + 60, y, an=7, safe_y=safe_y, drift=8)
                + rf"{{\fsp{D.TRACK_KICKER}}}"
                + _escape_ass(beat.kicker.upper()),
                3,
            )
        )
        y += 66
    title_lines = _wrap_title(beat.title)
    is_quote = beat.kind == "quote"
    style = "GQuote" if is_quote else "GTitle"
    max_fs = D.FS_QUOTE if is_quote else D.FS_TITLE
    # Short statements go display-size: the hook has to land instantly.
    if not is_quote and len(title_lines) == 1 and len(beat.title) <= 20:
        max_fs = D.FS_DISPLAY
    fs = _fit_fs(max(title_lines, key=len), max_fs)
    line_h = int(fs * D.LINE_HEIGHT_TITLE)
    for i, part in enumerate(title_lines):
        ty = y + i * line_h
        start = _on(beat, 50 + i * D.STAGGER_MS)
        tags = _title_motion(
            beat, x, ty, fs=fs, width=_text_width(part, fs), safe_y=safe_y
        )
        base_fs = D.FS_QUOTE if is_quote else D.FS_TITLE
        size = rf"{{\fs{fs}}}" if fs != base_fs else ""
        body = _escape_ass(part) if is_quote else _accent_keywords(part, th)
        lines.append(_dialogue(start, e, style, tags + size + body, 3))
    return y + len(title_lines) * line_h + 30


def _staggered_bullets(
    beat: GraphicBeat,
    lines: list[str],
    *,
    y0: int,
    safe_y: int,
) -> None:
    bullets = _fallback_bullets(beat)
    if not bullets:
        return
    # Structural hairline separating headline from the body group.
    if y0 + 90 <= safe_y:
        _hline(
            lines,
            _on(beat, 180),
            beat.end,
            D.MARGIN_X,
            y0 - 6,
            D.USABLE_W,
            h=2,
            style="GHair",
            dur=440,
        )
    y0 += 34
    avail = safe_y - y0
    gap = 100
    if len(bullets) == 2 and avail > 230:
        gap = min(132, avail - 80)
    max_n = 0
    for n in range(len(bullets), 0, -1):
        if y0 + (n - 1) * gap + 64 <= safe_y:
            max_n = n
            break
    for i, bullet in enumerate(bullets[:max_n]):
        y = y0 + i * gap
        start = _on(beat, 240 + (bullet.delay_ms if bullet.delay_ms else i * 380))
        lines.append(
            _dialogue(
                start,
                beat.end,
                "GIndex",
                _fade_rise(D.MARGIN_X, y + 16, an=7, safe_y=safe_y) + f"{i + 1:02d}",
                3,
            )
        )
        text = bullet.text
        fs = _fit_fs(text, D.FS_BULLET, usable=D.USABLE_W - 84)
        size = rf"{{\fs{fs}}}" if fs != D.FS_BULLET else ""
        lines.append(
            _dialogue(
                start,
                beat.end,
                "GBullet",
                _fade_rise(D.MARGIN_X + 84, y, an=7, safe_y=safe_y)
                + size
                + _escape_ass(text),
                3,
            )
        )


def _vs_card(
    beat: GraphicBeat, lines: list[str], *, y: int, safe_y: int, full: bool = False
) -> None:
    e = beat.end
    cy = min(y + (220 if full else 130), safe_y - 170)
    lx, rx = 300, 780
    col_w = 420
    left = beat.left or "A"
    right = beat.right or "B"
    lfs = _fit_fs(left, D.FS_VS_LABEL, usable=col_w)
    rfs = _fit_fs(right, D.FS_VS_LABEL, usable=col_w)

    # Center divider draws outward from the VS badge.
    _vline(lines, _on(beat, 60), e, 540, cy - 96, cy - 36, grow="up", dur=300)
    _vline(lines, _on(beat, 60), e, 540, cy + 36, cy + 96, grow="down", dur=300)
    lines.append(
        _dialogue(
            _on(beat, 220),
            e,
            "GVsBadge",
            _fade_rise(540, cy, an=5, safe_y=safe_y, drift=6)
            + rf"{{\fsp{D.TRACK_BADGE}}}VS",
            3,
        )
    )

    for label, fs, x, dx, delay in (
        (left, lfs, lx, -D.SLIDE_PX, 120),
        (right, rfs, rx, D.SLIDE_PX, 120 + D.STAGGER_MS),
    ):
        size = rf"{{\fs{fs}}}" if fs != D.FS_VS_LABEL else ""
        lines.append(
            _dialogue(
                _on(beat, delay),
                e,
                "GVs",
                _slide_x(x, cy, dx=dx, an=5, safe_y=safe_y) + size + _escape_ass(label),
                3,
            )
        )
        uw = max(120, min(_text_width(label, fs), col_w - 60))
        _hline(
            lines,
            _on(beat, delay + 140),
            e,
            x - uw // 2,
            cy + int(fs * 0.72),
            uw,
            h=3,
            style="GRule",
            dur=340,
        )

    if beat.subtitle and cy + 160 + 50 <= safe_y:
        lines.append(
            _dialogue(
                _on(beat, 420),
                e,
                "GSub",
                _fade_rise(540, cy + 160, an=8, safe_y=safe_y) + _escape_ass(beat.subtitle),
                3,
            )
        )


def _process_card(beat: GraphicBeat, lines: list[str], *, y: int, safe_y: int) -> None:
    e = beat.end
    # No real steps -> no fake steps. The headline card stands on its own.
    chips = [c.strip() for c in (beat.chips or []) if c.strip()][:3]
    if len(chips) < 2:
        return
    xs = [216, 540, 864]
    py = min(y + 90, safe_y - 170)
    if py + 130 > safe_y:
        return
    for i, chip in enumerate(chips):
        start = _on(beat, 180 + i * 260)
        if i > 0:
            _hline(
                lines,
                _on(beat, 180 + i * 260 - 130),
                e,
                xs[i - 1] + 120,
                py + 20,
                xs[i] - xs[i - 1] - 240,
                h=2,
                style="GHair",
                dur=280,
            )
        lines.append(
            _dialogue(
                start,
                e,
                "GIndex",
                _fade_rise(xs[i], py, an=8, safe_y=safe_y) + f"{i + 1:02d}",
                3,
            )
        )
        fs = _fit_fs(chip, D.FS_CHIP, usable=300)
        size = rf"{{\fs{fs}}}" if fs != D.FS_CHIP else ""
        lines.append(
            _dialogue(
                start,
                e,
                "GChip",
                _fade_rise(xs[i], py + 64, an=8, safe_y=safe_y) + size + _escape_ass(chip),
                3,
            )
        )


_NUM_RE = re.compile(r"^([^\d]*)(\d[\d,]*\.?\d*)(.*)$")


def _stat_steps(value: float, decimals: int, grouped: bool) -> list[str]:
    steps: list[str] = []
    n = D.STAT_COUNT_STEPS
    for i in range(1, n + 1):
        p = i / n
        val = value * (1 - (1 - p) ** 2.0)
        if decimals:
            text = f"{val:,.{decimals}f}" if grouped else f"{val:.{decimals}f}"
        else:
            text = f"{round(val):,}" if grouped else f"{round(val)}"
        steps.append(text)
    return steps


def _process_vertical(
    beat: GraphicBeat,
    lines: list[str],
    *,
    y: int,
    safe_y: int,
    diagram: bool = False,
) -> None:
    """9:16 stack: numbered nodes with a growing spine and underline."""
    e = beat.end
    chips = [c.strip() for c in (beat.chips or []) if c.strip()][:3]
    if len(chips) < 2:
        chips = [b.text.strip() for b in (beat.bullets or []) if b.text.strip()][:3]
    if len(chips) < 2:
        return
    step = 230 if diagram else 200
    x_idx = D.MARGIN_X
    x_text = D.MARGIN_X + 110
    py0 = y + 36
    for i, chip in enumerate(chips):
        py = py0 + i * step
        if py + 90 > safe_y:
            break
        start = _on(beat, 160 + i * 280)
        if i > 0:
            _vline(
                lines,
                _on(beat, 160 + i * 280 - 140),
                e,
                x_idx + 22,
                py0 + (i - 1) * step + 56,
                py - 8,
                grow="down",
                dur=280,
            )
        lines.append(
            _dialogue(
                start,
                e,
                "GIndex",
                _fade_rise(x_idx, py, an=7, safe_y=safe_y) + f"{i + 1:02d}",
                3,
            )
        )
        fs = _fit_fs(chip, D.FS_CHIP, usable=D.USABLE_W - 130)
        size = rf"{{\fs{fs}}}" if fs != D.FS_CHIP else ""
        lines.append(
            _dialogue(
                start,
                e,
                "GChip",
                _fade_rise(x_text, py + 4, an=7, safe_y=safe_y)
                + size
                + _escape_ass(chip),
                3,
            )
        )
        rule_w = max(160, min(_text_width(chip, fs) + 24, D.USABLE_W - 130))
        _hline(
            lines,
            _on(beat, 160 + i * 280 + 120),
            e,
            x_text,
            py + int(fs * 1.15),
            rule_w,
            h=4 if diagram else 3,
            style="GRule",
            dur=320,
        )


def _stat_card(
    beat: GraphicBeat, lines: list[str], *, safe_y: int, full: bool = False
) -> None:
    e = beat.end
    cx, cy = 540, 700 if full else 320
    kicker_y = 200 if full else 96
    if beat.kicker:
        lines.append(
            _dialogue(
                _on(beat, 0),
                e,
                "GKicker",
                _fade_rise(cx, kicker_y, an=8, safe_y=safe_y, drift=8)
                + rf"{{\fsp{D.TRACK_KICKER}}}"
                + _escape_ass(beat.kicker.upper()),
                3,
            )
        )
    title = beat.title.strip()
    fs = _fit_fs(title, D.FS_STAT, usable=D.USABLE_W)
    size = rf"{{\fs{fs}}}" if fs != D.FS_STAT else ""
    match = _NUM_RE.match(title)
    if match and match.group(2).replace(",", "").replace(".", "").isdigit():
        prefix, num, suffix = match.groups()
        clean = num.replace(",", "")
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        steps = _stat_steps(float(clean), decimals, grouped="," in num)
        t0 = _on(beat, 60)
        dt = D.STAT_COUNT_MS / 1000.0 / len(steps)
        base = rf"{{\an5\pos({cx},{cy}){_clip(safe_y)}}}" + size
        for i, step in enumerate(steps[:-1]):
            entry = rf"{{\fad(90,0)}}" if i == 0 else ""
            lines.append(
                _dialogue(
                    t0 + i * dt,
                    t0 + (i + 1) * dt,
                    "GStat",
                    base + entry + _escape_ass(f"{prefix}{step}{suffix}"),
                    3,
                )
            )
        # Final value lands with a small settle so the number "hits".
        lines.append(
            _dialogue(
                t0 + (len(steps) - 1) * dt,
                e,
                "GStat",
                rf"{{\an5\pos({cx},{cy}){_clip(safe_y)}\fscx104\fscy104"
                + _t(0, 220, D.ACCEL_SOFT, r"\fscx100\fscy100")
                + rf"\fad(0,{D.FADE_OUT_MS})}}"
                + size
                + _escape_ass(f"{prefix}{steps[-1]}{suffix}"),
                3,
            )
        )
    else:
        lines.append(
            _dialogue(
                _on(beat, 120),
                e,
                "GStat",
                _settle(cx, cy, an=5, safe_y=safe_y) + size + _escape_ass(title),
                3,
            )
        )
    _hline(
        lines,
        _on(beat, 480),
        e,
        cx - 110,
        cy + 116,
        220,
        h=4,
        style="GRule",
        dur=360,
    )
    if beat.subtitle:
        lines.append(
            _dialogue(
                _on(beat, 340),
                e,
                "GSub",
                _fade_rise(cx, cy + 152, an=8, safe_y=safe_y) + _escape_ass(beat.subtitle),
                3,
            )
        )


def _rect_outline(
    lines: list[str],
    s: float,
    e: float,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    style: str = "GHair",
    dur: int = 320,
    layer: int = 2,
) -> None:
    _hline(lines, s, e, x, y, w, h=2, style=style, dur=dur, layer=layer)
    _hline(lines, s, e, x, y + h - 2, w, h=2, style=style, dur=dur, layer=layer)
    _vline(lines, s, e, x, y, y + h, style=style, dur=dur, w=2, layer=layer)
    _vline(lines, s, e, x + w - 2, y, y + h, style=style, dur=dur, w=2, layer=layer)


def _dotted_vline(
    lines: list[str],
    s: float,
    e: float,
    x: int,
    y1: int,
    y2: int,
    *,
    style: str = "GHair",
) -> None:
    dash, gap = 8, 10
    if y2 - y1 < 8:
        return
    y = y1
    while y + 4 < y2:
        ye = min(y + dash, y2)
        _vline(lines, s, e, x, y, ye, grow="down", style=style, dur=140, w=2)
        y = ye + gap


def _dotted_hline(
    lines: list[str],
    s: float,
    e: float,
    x: int,
    y: int,
    w: int,
    *,
    style: str = "GHair",
) -> None:
    dash, gap = 8, 10
    cx = x
    end = x + w
    while cx + 4 < end:
        dw = min(dash, end - cx)
        _hline(lines, s, e, cx, y, dw, h=2, style=style, dur=140)
        cx += dw + gap


def _scene_nodes(beat: GraphicBeat) -> list[GraphicNode]:
    if beat.nodes:
        return list(beat.nodes)[:5]
    chips = [c.strip() for c in (beat.chips or []) if c.strip()]
    if len(chips) >= 2:
        return [GraphicNode(label=c) for c in chips[:5]]
    if beat.kind == "vs_split":
        nodes: list[GraphicNode] = []
        if beat.left:
            nodes.append(GraphicNode(label=beat.left, sub=beat.subtitle))
        if beat.right:
            nodes.append(GraphicNode(label=beat.right))
        return nodes
    bullets = list(beat.bullets or [])[:4]
    return [GraphicNode(label=b.text[:20]) for b in bullets if b.text]


def _scene_bullets(beat: GraphicBeat) -> list[GraphicBullet]:
    if beat.bullets:
        return list(beat.bullets)[:5]
    if beat.subtitle:
        return [GraphicBullet(text=beat.subtitle, delay_ms=0)]
    return []


def _full_header(
    beat: GraphicBeat,
    lines: list[str],
    *,
    safe_y: int,
    th: D.Theme,
) -> int:
    e = beat.end
    x = D.FULL_MARGIN_X
    y = 72
    if beat.kicker:
        _hline(lines, _on(beat, 0), e, x, y + 16, 48, h=4, style="GRule", dur=260)
        lines.append(
            _dialogue(
                _on(beat, 40),
                e,
                "GKicker",
                _fade_rise(x + 64, y, an=7, safe_y=safe_y, drift=8)
                + rf"{{\fsp{D.TRACK_KICKER}}}"
                + _escape_ass(beat.kicker.upper()),
                3,
            )
        )
        y += 58
    if beat.kind == "stat":
        return y
    title_lines = _wrap_title(beat.title, limit=18)
    fs = _fit_fs(max(title_lines, key=len), D.FS_TITLE_FULL, usable=D.FULL_USABLE_W)
    line_h = int(fs * 1.08)
    for i, part in enumerate(title_lines[:2]):
        ty = y + i * line_h
        size = rf"{{\fs{fs}}}" if fs != D.FS_TITLE else ""
        lines.append(
            _dialogue(
                _on(beat, 50 + i * D.STAGGER_MS),
                e,
                "GTitle",
                _title_motion(
                    beat, x, ty, fs=fs, width=_text_width(part, fs), safe_y=safe_y
                )
                + size
                + _accent_keywords(part, th),
                3,
            )
        )
    y = y + len(title_lines[:2]) * line_h + 18
    if beat.kind != "stat" and beat.subtitle and y + 36 < safe_y:
        lines.append(
            _dialogue(
                _on(beat, 160),
                e,
                "GSub",
                _fade_rise(x, y, an=7, safe_y=safe_y)
                + rf"{{\fs{D.FS_NODE_SUB}}}"
                + _escape_ass(beat.subtitle),
                3,
            )
        )
        y += 44
    return y


def _draw_node_box(
    beat: GraphicBeat,
    lines: list[str],
    node: GraphicNode,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    index: int,
    safe_y: int,
    delay_ms: int,
) -> None:
    e = beat.end
    start = _on(beat, delay_ms)
    _rect_outline(lines, start, e, x, y, w, h, style="GHair", dur=300)
    _vline(
        lines,
        start,
        e,
        x + 6,
        y + 14,
        y + h - 14,
        style="GRule",
        dur=280,
        w=4,
    )
    lines.append(
        _dialogue(
            start,
            e,
            "GIndex",
            _fade_rise(x + 28, y + 18, an=7, safe_y=safe_y) + f"{index + 1:02d}",
            4,
        )
    )
    label_fs = _fit_fs(node.label, D.FS_NODE, usable=w - 130)
    size = rf"{{\fs{label_fs}}}" if label_fs != D.FS_NODE else ""
    lines.append(
        _dialogue(
            start,
            e,
            "GNode",
            _fade_rise(x + 96, y + 16, an=7, safe_y=safe_y)
            + size
            + _escape_ass(node.label),
            4,
        )
    )
    if node.sub and h >= 108:
        sub_fs = _fit_fs(node.sub, D.FS_NODE_SUB, usable=w - 130)
        lines.append(
            _dialogue(
                _on(beat, delay_ms + 90),
                e,
                "GNodeSub",
                _fade_rise(x + 96, y + 62, an=7, safe_y=safe_y)
                + rf"{{\fs{sub_fs}}}"
                + _escape_ass(node.sub),
                4,
            )
        )


def _draw_flow(
    beat: GraphicBeat,
    lines: list[str],
    nodes: list[GraphicNode],
    *,
    y0: int,
    y1: int,
    safe_y: int,
) -> int:
    if not nodes:
        return y0
    n = len(nodes)
    if n < 1 or y1 <= y0 + 40:
        return y0
    gap = 44 if n <= 4 else 32
    box_h = int((y1 - y0 - gap * (n - 1)) / n)
    box_h = max(96, min(168, box_h))
    x = D.FULL_MARGIN_X
    w = D.FULL_USABLE_W
    y = y0
    for i, node in enumerate(nodes):
        if y + box_h > safe_y:
            break
        _draw_node_box(
            beat,
            lines,
            node,
            x=x,
            y=y,
            w=w,
            h=box_h,
            index=i,
            safe_y=safe_y,
            delay_ms=140 + i * 220,
        )
        if i + 1 < n:
            cy1 = y + box_h
            cy2 = y + box_h + gap
            _dotted_vline(
                lines,
                _on(beat, 140 + i * 220 + 80),
                beat.end,
                x + 36,
                cy1 + 4,
                cy2 - 4,
            )
        y += box_h + gap
    return y - gap + 12


def _draw_vs_pair(
    beat: GraphicBeat,
    lines: list[str],
    nodes: list[GraphicNode],
    *,
    y0: int,
    y1: int,
    safe_y: int,
) -> int:
    e = beat.end
    left = nodes[0] if nodes else GraphicNode(label=beat.left or "A")
    right = nodes[1] if len(nodes) > 1 else GraphicNode(label=beat.right or "B")
    gap = 56
    w = (D.FULL_USABLE_W - gap) // 2
    h = max(140, min(220, y1 - y0 - 20))
    y = y0
    lx = D.FULL_MARGIN_X
    rx = lx + w + gap
    _draw_node_box(
        beat, lines, left, x=lx, y=y, w=w, h=h, index=0, safe_y=safe_y, delay_ms=120
    )
    _draw_node_box(
        beat, lines, right, x=rx, y=y, w=w, h=h, index=1, safe_y=safe_y, delay_ms=220
    )
    mid_y = y + h // 2
    _dotted_hline(
        lines,
        _on(beat, 180),
        e,
        lx + w + 8,
        mid_y,
        gap - 16,
        style="GHair",
    )
    lines.append(
        _dialogue(
            _on(beat, 240),
            e,
            "GVsBadge",
            _fade_rise(lx + w + gap // 2, mid_y, an=5, safe_y=safe_y, drift=4)
            + rf"{{\fsp{D.TRACK_BADGE}}}VS",
            4,
        )
    )
    return y + h + 20


def _draw_support(
    beat: GraphicBeat,
    lines: list[str],
    bullets: list[GraphicBullet],
    *,
    y0: int,
    safe_y: int,
) -> None:
    if not bullets or y0 + 70 > safe_y:
        return
    e = beat.end
    x = D.FULL_MARGIN_X
    avail = safe_y - y0
    n = len(bullets)
    gap = 72
    if n * gap > avail:
        gap = max(58, avail // max(n, 1))
        while n > 1 and y0 + (n - 1) * gap + 50 > safe_y:
            n -= 1
    if y0 + 8 < safe_y:
        _hline(lines, _on(beat, 200), e, x, y0, D.FULL_USABLE_W, h=2, style="GHair", dur=360)
    y = y0 + 20
    for i, bullet in enumerate(bullets[:n]):
        if y + 48 > safe_y:
            break
        start = _on(beat, 280 + (bullet.delay_ms if bullet.delay_ms else i * 320))
        lines.append(
            _dialogue(
                start,
                e,
                "GIndex",
                _fade_rise(x, y + 8, an=7, safe_y=safe_y) + f"{i + 1:02d}",
                3,
            )
        )
        fs = _fit_fs(bullet.text, D.FS_BULLET_FULL, usable=D.FULL_USABLE_W - 90)
        lines.append(
            _dialogue(
                start,
                e,
                "GBullet",
                _fade_rise(x + 78, y, an=7, safe_y=safe_y)
                + rf"{{\fs{fs}}}"
                + _escape_ass(bullet.text),
                3,
            )
        )
        y += gap


def _full_stat_hero(
    beat: GraphicBeat,
    lines: list[str],
    *,
    y: int,
    safe_y: int,
    th: D.Theme,
) -> int:
    e = beat.end
    cx = 540
    title = beat.title.strip()
    fs = _fit_fs(title, min(D.FS_STAT, 140), usable=D.FULL_USABLE_W)
    size = rf"{{\fs{fs}}}" if fs != D.FS_STAT else ""
    cy = y + int(fs * 0.55) + 8
    match = _NUM_RE.match(title)
    if match and match.group(2).replace(",", "").replace(".", "").isdigit():
        prefix, num, suffix = match.groups()
        clean = num.replace(",", "")
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        steps = _stat_steps(float(clean), decimals, grouped="," in num)
        t0 = _on(beat, 60)
        dt = D.STAT_COUNT_MS / 1000.0 / len(steps)
        base = rf"{{\an5\pos({cx},{cy}){_clip(safe_y)}}}" + size
        for i, step in enumerate(steps[:-1]):
            entry = rf"{{\fad(90,0)}}" if i == 0 else ""
            lines.append(
                _dialogue(
                    t0 + i * dt,
                    t0 + (i + 1) * dt,
                    "GStat",
                    base + entry + _escape_ass(f"{prefix}{step}{suffix}"),
                    3,
                )
            )
        lines.append(
            _dialogue(
                t0 + (len(steps) - 1) * dt,
                e,
                "GStat",
                rf"{{\an5\pos({cx},{cy}){_clip(safe_y)}\fscx104\fscy104"
                + _t(0, 220, D.ACCEL_SOFT, r"\fscx100\fscy100")
                + rf"\fad(0,{D.FADE_OUT_MS})}}"
                + size
                + _escape_ass(f"{prefix}{steps[-1]}{suffix}"),
                3,
            )
        )
    else:
        lines.append(
            _dialogue(
                _on(beat, 80),
                e,
                "GStat",
                _settle(cx, cy, an=5, safe_y=safe_y) + size + _escape_ass(title),
                3,
            )
        )
    y = cy + int(fs * 0.55) + 16
    _hline(lines, _on(beat, 400), e, cx - 90, y, 180, h=4, style="GRule", dur=320)
    y += 18
    if beat.subtitle:
        lines.append(
            _dialogue(
                _on(beat, 280),
                e,
                "GSub",
                _fade_rise(cx, y + 8, an=8, safe_y=safe_y) + _escape_ass(beat.subtitle),
                3,
            )
        )
        y += 48
    _ = th
    return y + 12


def _full_scene(beat: GraphicBeat, lines: list[str], th: D.Theme) -> None:
    safe_y = D.FULL_SAFE_Y
    y = _full_header(beat, lines, safe_y=safe_y, th=th)
    if beat.kind == "stat":
        y = _full_stat_hero(beat, lines, y=y, safe_y=safe_y, th=th)
    nodes = _scene_nodes(beat)
    bullets = _scene_bullets(beat)
    # Keep room at the bottom for 3–5 supporting lines.
    support_n = min(5, max(3, len(bullets))) if bullets else 0
    support_h = (28 + support_n * 68) if support_n else 0
    body_end = safe_y - support_h
    if body_end - y < 180:
        body_end = min(safe_y - 80, y + 280)
        support_h = max(0, safe_y - body_end)
    if beat.kind == "vs_split" and len(nodes) >= 2:
        y = _draw_vs_pair(beat, lines, nodes, y0=y, y1=body_end, safe_y=safe_y)
    elif nodes:
        y = _draw_flow(beat, lines, nodes, y0=y, y1=body_end, safe_y=safe_y)
    if bullets:
        _draw_support(beat, lines, bullets, y0=max(y, body_end - 8), safe_y=safe_y)


def _graphic_events(
    beats: list[GraphicBeat], top_h: int, th: D.Theme, *, full: bool = False
) -> list[str]:
    if full:
        lines: list[str] = []
        for beat in beats:
            _full_scene(beat, lines, th)
        return lines
    safe_y = top_h - D.SAFE_BOTTOM_PAD
    lines = []
    for beat in beats:
        if beat.kind == "stat":
            _stat_card(beat, lines, safe_y=safe_y, full=False)
            continue
        y = _header(beat, lines, y=D.TOP_Y, safe_y=safe_y, th=th)
        if beat.kind == "vs_split":
            _vs_card(beat, lines, y=y, safe_y=safe_y, full=False)
            continue
        if beat.kind in {"process", "diagram", "chip_row"}:
            _process_card(beat, lines, y=y, safe_y=safe_y)
            continue
        if beat.kind == "quote":
            if beat.subtitle and y + 70 <= safe_y:
                lines.append(
                    _dialogue(
                        _on(beat, 420),
                        beat.end,
                        "GMeta",
                        _fade_rise(D.MARGIN_X, y + 26, an=7, safe_y=safe_y)
                        + "— "
                        + _escape_ass(beat.subtitle),
                        3,
                    )
                )
            continue
        _staggered_bullets(beat, lines, y0=y, safe_y=safe_y)
    return lines


def _style(
    name: str,
    font: str,
    fs: int,
    color: str,
    *,
    bold: bool = False,
    spacing: float = 0.0,
    outline_color: str = "&H00000000",
    outline: int = 0,
    shadow: int = 0,
    align: int = 7,
    ml: int = 0,
    mr: int = 0,
    mv: int = 0,
) -> str:
    b = -1 if bold else 0
    return (
        f"Style: {name},{font},{fs},"
        f"{color},&H000000FF,{outline_color},&H00000000,"
        f"{b},0,0,0,100,100,{spacing},0,1,{outline},{shadow},"
        f"{align},{ml},{mr},{mv},1"
    )


def _graphic_styles(th: D.Theme) -> list[str]:
    return [
        _style("GKicker", D.FONT_SEMIBOLD, D.FS_KICKER, th.accent),
        _style("GTitle", D.FONT_BOLD, D.FS_TITLE, th.ink, bold=True, spacing=D.TRACK_TITLE),
        _style("GQuote", D.FONT_MEDIUM, D.FS_QUOTE, th.ink, spacing=0.2),
        _style("GStat", D.FONT_BOLD, D.FS_STAT, th.accent, bold=True, spacing=-1),
        _style("GSub", D.FONT_MEDIUM, D.FS_SUBTITLE, th.muted, spacing=0.2),
        _style("GMeta", D.FONT_MEDIUM, D.FS_META, th.faint, spacing=0.4),
        _style("GBullet", D.FONT_SEMIBOLD, D.FS_BULLET, th.ink),
        _style("GIndex", D.FONT_SEMIBOLD, D.FS_INDEX, th.accent, spacing=1.5),
        _style("GVs", D.FONT_BOLD, D.FS_VS_LABEL, th.ink, bold=True, spacing=-0.3, align=5),
        _style("GVsBadge", D.FONT_SEMIBOLD, D.FS_VS_BADGE, th.muted, align=5),
        _style("GChip", D.FONT_SEMIBOLD, D.FS_CHIP, th.ink, spacing=0.4, align=5),
        _style("GNode", D.FONT_SEMIBOLD, D.FS_NODE, th.ink, spacing=0.15),
        _style("GNodeSub", D.FONT_MEDIUM, D.FS_NODE_SUB, th.muted, spacing=0.15),
        _style("GRule", D.FONT_REGULAR, 1, th.accent),
        _style("GHair", D.FONT_REGULAR, 1, th.hair),
        _style("GLine", D.FONT_REGULAR, 10, th.hair, align=5),
    ]


def _styles(*, layout: str, font_name: str, th: D.Theme) -> list[str]:
    if layout == "overlay":
        return [
            f"Style: Default,{font_name},72,"
            f"&H00FFFFFF,&H0000FFFF,&H00101010,&H80000000,"
            f"-1,0,0,0,100,100,1.2,0,1,6,2,8,72,72,64,1"
        ]
    if layout == "full":
        light = th.name in {"paper", "ivory"}
        cap_color = th.ink if light else "&H00FFFFFF"
        cap_outline = "&H00FFFFFF" if light else th.caption_outline
        return [
            _style(
                "Default", D.FONT_BOLD, 72, cap_color,
                bold=True, spacing=0.3, outline_color=cap_outline,
                outline=3 if light else 6, align=8, ml=48, mr=48,
            ),
            *_graphic_styles(th),
        ]
    return [
        _style(
            "Default", D.FONT_BOLD, 80, "&H00FFFFFF",
            bold=True, spacing=0.3, outline_color=th.caption_outline, outline=6,
            align=8, ml=12, mr=12,
        ),
        *_graphic_styles(th),
    ]


def write_ass_file(
    timeline: CaptionTimeline,
    output_path: str,
    *,
    width: int = 1080,
    height: int = 1920,
    font_name: str = "Montserrat",
    graphics: list[GraphicBeat] | None = None,
    split_layout: bool = False,
    layout: str | None = None,
    video_duration: float = 0.0,
    theme: str | D.Theme | None = None,
    head_top: int | None = None,
) -> str:
    th = theme if isinstance(theme, D.Theme) else D.get_theme(theme)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = layout or ("split" if split_layout else "overlay")
    timeline = explode_caption_timeline(timeline)

    lines = [
        "[Script Info]",
        "Title: Split Reel Captions" if mode != "full" else "Title: Audio Motion Reel",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        *_styles(layout=mode, font_name=font_name, th=th),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    end = video_duration if video_duration > 0.2 else max(
        (c.end for c in timeline.captions), default=8.0
    )
    if mode == "split":
        top_h, _ = split_panel_sizes(height)
        lines.append(
            _dialogue(
                0.0,
                end,
                "GLine",
                rf"{{\an5\pos(540,{top_h})\p1\alpha&H96&}}m 0 0 l 1080 0 l 1080 2 l 0 2",
                0,
            )
        )
        lines.extend(_graphic_events(graphics or [], top_h, th))
    elif mode == "full":
        lines.extend(_graphic_events(graphics or [], height, th, full=True))

    uppercase = mode == "overlay"
    seam_y = split_panel_sizes(height)[0] if mode == "split" else 960
    overlay_y = _overlay_caption_y(head_top or int(height * 0.30), height)
    clip_head = head_top or overlay_y + 80
    for cap in _dedupe_overlaps(list(timeline.captions)):
        text = _styled_caption_text(
            cap,
            uppercase=uppercase,
            split=mode == "split",
            layout=mode,
            seam_y=seam_y,
            theme=th,
            overlay_y=overlay_y,
            head_top=clip_head,
        )
        lines.append(
            "Dialogue: 5,"
            f"{_ass_time(cap.start)},{_ass_time(cap.end)},"
            f"Default,,0,0,0,,{text}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)

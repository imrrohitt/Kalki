from __future__ import annotations

from pathlib import Path

from app.captions.models import Caption, CaptionTimeline, CaptionWord
from app.editorial.models import GraphicBeat, GraphicBullet
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


def _lines_from_caption(caption: Caption, *, uppercase: bool, max_chars: int = 22) -> list[str]:
    display = (caption.text or "").replace("\\n", "\n").strip()
    if uppercase:
        display = display.upper()
    if "\n" in display:
        parts = [p.strip() for p in display.split("\n") if p.strip()]
        return parts[:2]
    words = display.split()
    if not words:
        words = [w.text.upper() if uppercase else w.text for w in caption.words]
    joined = " ".join(words)
    if len(joined) <= max_chars or len(words) <= 2:
        return [joined]
    mid = max(1, (len(words) + 1) // 2)
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def _emphasis_set(caption: Caption) -> set[str]:
    return {
        w.text.upper().strip(".,!?")
        for w in caption.words
        if w.emphasis
    }


def _style_line(line: str, hot: set[str], *, uppercase: bool) -> str:
    parts = []
    tokens = line.upper().split() if uppercase else line.split()
    for token in tokens:
        clean = token.strip(".,!?").upper()
        safe = _escape_ass(token)
        if clean in hot:
            parts.append(rf"{{\c&H0000D7FF&\b1}}{safe}{{\c&H00FFFFFF&\b0}}")
        else:
            parts.append(safe)
    return " ".join(parts)


def _caption_pop(uppercase: bool) -> str:
    if uppercase:
        return (
            r"{\fad(80,90)"
            r"\t(0,130,\fscx128\fscy128)"
            r"\t(130,230,\fscx100\fscy100)}"
        )
    return r"{\fad(90,80)}"


def _styled_caption_text(
    caption: Caption, *, uppercase: bool, split: bool, seam_y: int = 720
) -> str:
    hot = _emphasis_set(caption)
    lines = _lines_from_caption(
        caption, uppercase=uppercase, max_chars=14 if split else 22
    )
    body = r"\N".join(_style_line(line, hot, uppercase=uppercase) for line in lines)
    anim = _caption_pop(uppercase)
    if split:
        # Hang just below the graphics canvas so captions never sit in a box on the type.
        return rf"{{\an8\pos(540,{seam_y + 10})}}" + anim + body
    return anim + body


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


def _title_width(text: str, fs: int) -> int:
    return int(0.52 * fs * max(len(text), 1))


def _fit_fs(text: str, max_fs: int, usable: int = 960) -> int:
    fs = max_fs
    while fs > 64 and _title_width(text, fs) > usable:
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
    return rf"\clip(0,8,1080,{safe_y})"


def _enter(
    x: int,
    y: int,
    *,
    an: int = 7,
    tracking: float = 1.2,
    fs: int | None = None,
    safe_y: int = 620,
) -> str:
    size = rf"\fs{fs}" if fs else ""
    return (
        rf"{{\an{an}\move({x},{y + 8},{x},{y},0,420){_clip(safe_y)}"
        rf"{size}\fad(160,140)\fsp{tracking}\t(0,420,\fsp0)}}"
    )


def _punch(x: int, y: int, *, an: int = 7, safe_y: int = 620) -> str:
    return _enter(x, y, an=an, tracking=2.0, safe_y=safe_y)


def _rise(x: int, y: int, *, an: int = 7, dy: int = 10, fade_in: int = 180, safe_y: int = 620) -> str:
    return _enter(x, y, an=an, tracking=3.0, safe_y=safe_y)


def _fade_up(x: int, y: int, *, an: int = 7, safe_y: int = 620) -> str:
    return rf"{{\an{an}\pos({x},{y}){_clip(safe_y)}\fad(200,160)}}"


def _slide(x: int, y: int, *, an: int = 7, dy: int = 10, fade_in: int = 180, safe_y: int = 620) -> str:
    return _enter(x, y, an=an, safe_y=safe_y)


def _scale_in(x: int, y: int, *, an: int = 7, safe_y: int = 620) -> str:
    return _enter(x, y, an=an, tracking=5.0, safe_y=safe_y)


def _motion_tags(
    beat: GraphicBeat,
    x: int,
    y: int,
    *,
    an: int = 7,
    fs: int | None = None,
    safe_y: int = 620,
) -> str:
    tracking = 0.6 if (fs and fs < 96) or beat.kind == "stat" else 1.0
    return _enter(x, y, an=an, tracking=tracking, fs=fs, safe_y=safe_y)


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
) -> int:
    s, e = beat.start, beat.end
    x = 64
    if beat.kicker:
        lines.append(
            _dialogue(
                s,
                e,
                "GKicker",
                _enter(x, y, tracking=2.4, safe_y=safe_y)
                + _escape_ass(beat.kicker.upper()),
                2,
            )
        )
        y += 56
    title_lines = _wrap_title(beat.title)
    style = "GTitle"
    max_fs = 108
    if beat.kind == "stat":
        style = "GStat"
        max_fs = 120
        x = 540
        an = 8
    else:
        an = 7
    fs = _fit_fs(max(title_lines, key=len), max_fs)
    line_h = int(fs * 1.18)
    for i, part in enumerate(title_lines):
        ty = y + i * line_h
        start = _on(beat, 60 + i * 50)
        tags = _motion_tags(beat, x, ty, an=an, fs=fs, safe_y=safe_y)
        lines.append(_dialogue(start, e, style, tags + _escape_ass(part), 3))
    return y + max(line_h, len(title_lines) * line_h) + 36


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
    avail = safe_y - y0
    gap = 108
    if len(bullets) == 2 and avail > 230:
        gap = min(140, avail - 80)
    max_n = 0
    for n in range(len(bullets), 0, -1):
        if y0 + (n - 1) * gap + 72 <= safe_y:
            max_n = n
            break
    for i, bullet in enumerate(bullets[:max_n]):
        y = y0 + i * gap
        start = _on(beat, 280 + (bullet.delay_ms if bullet.delay_ms else i * 380))
        index = f"{i + 1:02d}"
        lines.append(
            _dialogue(
                start,
                beat.end,
                "GStep",
                _enter(56, y + 8, an=7, tracking=2.0, safe_y=safe_y) + index,
                3,
            )
        )
        lines.append(
            _dialogue(
                start,
                beat.end,
                "GBullet",
                _enter(148, y, tracking=1.5, safe_y=safe_y) + _escape_ass(bullet.text),
                3,
            )
        )


def _graphic_events(beats: list[GraphicBeat], top_h: int) -> list[str]:
    safe_y = top_h - 72
    lines: list[str] = []
    for beat in beats:
        e = beat.end
        y = _header(beat, lines, y=36, safe_y=safe_y)

        if beat.kind == "vs_split":
            cy = min(y + 48, safe_y - 90)
            lines.append(
                _dialogue(
                    _on(beat, 180),
                    e,
                    "GVs",
                    _enter(250, cy, an=5, tracking=2.0, safe_y=safe_y)
                    + _escape_ass(beat.left or "A"),
                    3,
                )
            )
            lines.append(
                _dialogue(
                    _on(beat, 280),
                    e,
                    "GAccent",
                    _fade_up(540, cy, an=5, safe_y=safe_y) + "VS",
                    3,
                )
            )
            lines.append(
                _dialogue(
                    _on(beat, 380),
                    e,
                    "GVs",
                    _enter(830, cy, an=5, tracking=2.0, safe_y=safe_y)
                    + _escape_ass(beat.right or "B"),
                    3,
                )
            )
            if beat.subtitle and cy + 78 <= safe_y:
                lines.append(
                    _dialogue(
                        _on(beat, 500),
                        e,
                        "GSub",
                        _fade_up(540, cy + 78, an=5, safe_y=safe_y)
                        + _escape_ass(beat.subtitle),
                        3,
                    )
                )
            continue

        if beat.kind == "process":
            chips = (beat.chips or ["In", "Then", "Out"])[:3]
            xs = [200, 540, 880]
            py = min(y + 28, safe_y - 140)
            if py + 100 <= safe_y:
                for i, chip in enumerate(chips):
                    start = _on(beat, 220 + i * 220)
                    connector = ""
                    lines.append(
                        _dialogue(
                            start,
                            e,
                            "GStep",
                            _enter(xs[i], py, an=5, tracking=2.0, safe_y=safe_y)
                            + f"{i + 1:02d}",
                            3,
                        )
                    )
                    lines.append(
                        _dialogue(
                            start,
                            e,
                            "GChip",
                            _enter(xs[i], py + 72, an=5, tracking=1.2, safe_y=safe_y)
                            + _escape_ass(chip)
                            + connector,
                            3,
                        )
                    )
            continue

        if beat.kind == "stat" and beat.subtitle:
            lines.append(
                _dialogue(
                    _on(beat, 280),
                    e,
                    "GSub",
                    _fade_up(540, y + 8, an=8, safe_y=safe_y) + _escape_ass(beat.subtitle),
                    3,
                )
            )
            continue

        _staggered_bullets(beat, lines, y0=y, safe_y=safe_y)
    return lines


def _styles(*, split: bool, font_name: str) -> list[str]:
    if not split:
        return [
            f"Style: Default,{font_name},96,"
            f"&H00FFFFFF,&H0000FFFF,&H00101010,&H80000000,"
            f"-1,0,0,0,100,100,1.6,0,1,6,2,2,80,80,280,1"
        ]
    ink = "&H0014181C"
    accent = "&H001A5AE8"
    muted = "&H005C646B"
    return [
        f"Style: Default,{font_name},84,"
        f"&H00FFFFFF,&H0000D7FF,&H00101010,&H00000000,"
        f"-1,0,0,0,100,100,0.3,0,1,9,0,8,12,12,0,1",
        f"Style: GKicker,{font_name},48,"
        f"{accent},&H000000FF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,1.4,0,1,0,0,7,40,40,0,1",
        f"Style: GTitle,{font_name},112,"
        f"{ink},&H000000FF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,0,0,7,24,24,0,1",
        f"Style: GTopic,{font_name},112,"
        f"{ink},&H000000FF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,0,0,7,24,24,0,1",
        f"Style: GStat,{font_name},128,"
        f"{accent},&H000000FF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,0,0,8,24,24,0,1",
        f"Style: GSub,{font_name},48,"
        f"{muted},&H000000FF,&H00000000,&H00000000,"
        f"0,0,0,0,100,100,0.2,0,1,0,0,8,24,24,0,1",
        f"Style: GQuote,{font_name},64,"
        f"{ink},&H000000FF,&H00000000,&H00000000,"
        f"0,0,0,0,100,100,0.1,0,1,0,0,7,40,40,0,1",
        f"Style: GChip,{font_name},56,"
        f"{ink},&H000000FF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,0.1,0,1,0,0,5,12,12,0,1",
        f"Style: GMuted,{font_name},40,"
        f"{muted},&H000000FF,&H00000000,&H00000000,"
        f"0,0,0,0,100,100,0.2,0,1,0,0,7,24,24,0,1",
        f"Style: GBullet,{font_name},64,"
        f"{ink},&H000000FF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,0,0,7,24,24,0,1",
        f"Style: GAccent,{font_name},44,"
        f"{accent},&H000000FF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,1.2,0,1,0,0,5,24,24,0,1",
        f"Style: GKickerOn,{font_name},48,"
        f"{accent},&H000000FF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,1.4,0,1,0,0,7,40,40,0,1",
        f"Style: GChipOn,{font_name},56,"
        f"{ink},&H000000FF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,0.1,0,1,0,0,5,12,12,0,1",
        f"Style: GVs,{font_name},92,"
        f"{ink},&H000000FF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,0,0,5,12,12,0,1",
        f"Style: GStep,{font_name},72,"
        f"{accent},&H000000FF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,0,0,5,12,12,0,1",
        f"Style: GBadge,{font_name},36,"
        f"&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1",
        f"Style: GShape,{font_name},1,"
        f"&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1",
        f"Style: GLine,{font_name},10,"
        f"&H66403A34,&H000000FF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1",
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
    video_duration: float = 0.0,
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "[Script Info]",
        "Title: Split Reel Captions",
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
        *_styles(split=split_layout, font_name=font_name),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    if split_layout:
        top_h, _ = split_panel_sizes(height)
        end = video_duration if video_duration > 0.2 else max(
            (c.end for c in timeline.captions), default=8.0
        )
        lines.append(
            _dialogue(
                0.0,
                end,
                "GLine",
                rf"{{\an5\pos(540,{top_h})\p1\alpha&HAA&}}m 0 0 l 980 0 l 980 2 l 0 2",
                0,
            )
        )
        lines.extend(_graphic_events(graphics or [], top_h))

    uppercase = not split_layout
    seam_y = split_panel_sizes(height)[0] if split_layout else 960
    for cap in _dedupe_overlaps(list(timeline.captions)):
        text = _styled_caption_text(
            cap, uppercase=uppercase, split=split_layout, seam_y=seam_y
        )
        lines.append(
            "Dialogue: 5,"
            f"{_ass_time(cap.start)},{_ass_time(cap.end)},"
            f"Default,,0,0,0,,{text}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)

from __future__ import annotations

from pathlib import Path

from app.captions.models import Caption, CaptionTimeline, CaptionWord


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


def _lines_from_caption(caption: Caption) -> list[str]:
    display = (caption.text or "").replace("\\n", "\n").strip()
    if "\n" in display:
        parts = [p.strip() for p in display.split("\n") if p.strip()]
        return parts[:2]
    words = display.upper().split()
    if not words:
        words = [w.text.upper() for w in caption.words]
    if len(" ".join(words)) <= 18 or len(words) <= 3:
        return [" ".join(words)]
    # Balanced two-line wrap for longer phrases.
    mid = max(1, (len(words) + 1) // 2)
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def _emphasis_set(caption: Caption) -> set[str]:
    return {
        w.text.upper().strip(".,!?")
        for w in caption.words
        if w.emphasis
    }


def _style_line(line: str, hot: set[str]) -> str:
    parts = []
    for token in line.upper().split():
        clean = token.strip(".,!?")
        safe = _escape_ass(token)
        if clean in hot:
            parts.append(rf"{{\c&H0000FFFF&\b1}}{safe}{{\c&H00FFFFFF&\b0}}")
        else:
            parts.append(safe)
    return " ".join(parts)


def _styled_caption_text(caption: Caption) -> str:
    hot = _emphasis_set(caption)
    lines = _lines_from_caption(caption)
    body = r"\N".join(_style_line(line, hot) for line in lines)
    anim = (
        r"{\fad(80,90)"
        r"\t(0,130,\fscx128\fscy128)"
        r"\t(130,230,\fscx100\fscy100)}"
    )
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


def write_ass_file(
    timeline: CaptionTimeline,
    output_path: str,
    *,
    width: int = 1080,
    height: int = 1920,
    font_name: str = "Montserrat",
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    style = (
        f"Style: Default,{font_name},96,"
        f"&H00FFFFFF,&H0000FFFF,&H00101010,&H80000000,"
        f"-1,0,0,0,100,100,1.6,0,1,6,2,2,80,80,280,1"
    )

    lines = [
        "[Script Info]",
        "Title: Dynamic Social Captions",
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
        style,
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for cap in _dedupe_overlaps(list(timeline.captions)):
        text = _styled_caption_text(cap)
        lines.append(
            "Dialogue: 0,"
            f"{_ass_time(cap.start)},{_ass_time(cap.end)},"
            f"Default,,0,0,0,,{text}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)

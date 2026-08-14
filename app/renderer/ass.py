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


def _split_two_lines(words: list[str], max_chars: int = 16) -> tuple[list[str], list[str]]:
    line1: list[str] = []
    line2: list[str] = []
    for word in words:
        candidate = (" ".join(line1 + [word])).strip()
        if not line2 and (not line1 or len(candidate) <= max_chars):
            line1.append(word)
        else:
            line2.append(word)
    return line1, line2


def _karaoke_line(words: list[CaptionWord], start_offset: float) -> str:
    """Active word pops yellow; upcoming words stay white."""
    parts: list[str] = []
    for index, word in enumerate(words):
        token = _escape_ass(word.text.upper())
        # Duration of this word highlight in centiseconds.
        dur = max(int(round((word.end - word.start) * 100)), 8)
        # {\k} advances karaoke; {\kf} fill. Use highlight swap via \1c.
        if word.emphasis or index == 0:
            # Emphasized / lead words get stronger treatment.
            parts.append(rf"{{\k{dur}\1c&H0000FFFF&}}{token}{{\1c&H00FFFFFF&}}")
        else:
            parts.append(rf"{{\k{dur}}}{token}")
        if index < len(words) - 1:
            parts.append(" ")
    return "".join(parts)


def _styled_caption_text(caption: Caption) -> str:
    words = list(caption.words or [])
    if words:
        plain = [w.text.upper() for w in words]
        line1_words, line2_words = _split_two_lines(plain, max_chars=16)
        n1 = len(line1_words)
        w1, w2 = words[:n1], words[n1:]
        body = _karaoke_line(w1, caption.start)
        if w2:
            body += r"\N" + _karaoke_line(w2, caption.start)
    else:
        plain = _escape_ass(caption.text.upper()).split()
        line1, line2 = _split_two_lines(plain, max_chars=16)
        body = " ".join(line1)
        if line2:
            body += r"\N" + " ".join(line2)

    # Strong pop + soft fade — visible social-caption motion.
    anim = (
        r"{\fad(70,90)"
        r"\t(0,120,\fscx145\fscy145)"
        r"\t(120,240,\fscx100\fscy100)"
        r"\bord8\shad3}"
    )
    return anim + body


def _dedupe_overlaps(captions: list[Caption]) -> list[Caption]:
    ordered = sorted(captions, key=lambda c: (c.start, c.end))
    fixed: list[Caption] = []
    last_end = -1.0
    for cap in ordered:
        start = max(float(cap.start), last_end + 0.05)
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

    # Outline+shadow style (no harsh opaque box). Broader social look.
    # Fontsize 112, Spacing 3, Outline 7, Shadow 3, Alignment 2 bottom-center.
    # MarginV 300 keeps captions lower-third and away from chin a bit.
    style = (
        f"Style: Default,{font_name},112,"
        f"&H00FFFFFF,&H0000FFFF,&H00000000,&H64000000,"
        f"-1,0,0,0,100,100,3,0,1,7,3,2,90,90,300,1"
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

from __future__ import annotations


# Lanczos when covering a 9:16 frame — bilinear makes the 2× height-upscale look soft.
SCALE_FLAGS = "flags=lanczos+accurate_rnd+full_chroma_int"


def escape_drawtext(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    text = text.replace("%", "%%")
    text = text.replace(",", "\\,")
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")
    return text


def font_path_for_ffmpeg(path: str) -> str:
    # Escape filter special chars and spaces in paths.
    return (
        path.replace("\\", "/")
        .replace(":", "\\:")
        .replace(" ", "\\ ")
        .replace("'", "\\'")
    )


def vertical_fit_pad_filter(width: int, height: int) -> str:
    """Fit the entire source into 9:16. Never crop-zoom the face."""
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:"
        f"force_divisible_by=2:{SCALE_FLAGS},"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
        f"setsar=1"
    )


def vertical_scale_crop_filter(width: int, height: int) -> str:
    return vertical_fit_pad_filter(width, height)


def position_xy(position: str) -> tuple[str, str]:
    if position == "top":
        return ("(w-text_w)/2", "h*0.12")
    if position == "center":
        return ("(w-text_w)/2", "(h-text_h)/2")
    return ("(w-text_w)/2", "h*0.78")


def build_drawtext_filter(
    *,
    text: str,
    start: float,
    end: float,
    font_path: str,
    position: str = "bottom_center",
    animation: str = "pop",
    font_size: int = 72,
    font_color: str = "white",
    border_color: str = "black",
    border_width: int = 4,
    has_emphasis: bool = False,
) -> str:
    x, y = position_xy(position)
    safe_text = escape_drawtext(text.upper() if has_emphasis else text)
    font = font_path_for_ffmpeg(font_path)

    if animation == "pop":
        size_expr = (
            f"if(lt(t-{start:.3f}\\,0.12)\\,"
            f"{int(font_size * 1.18)}\\,{font_size})"
        )
    else:
        size_expr = str(font_size)

    color = "yellow" if has_emphasis else font_color

    return (
        f"drawtext=fontfile={font}"
        f":text='{safe_text}'"
        f":fontsize={size_expr}"
        f":fontcolor={color}"
        f":borderw={border_width}"
        f":bordercolor={border_color}"
        f":x={x}:y={y}"
        f":enable='between(t\\,{start:.3f}\\,{end:.3f})'"
    )


def sanitize_filter_graph(filters: list[str]) -> str:
    return ",".join(f for f in filters if f)

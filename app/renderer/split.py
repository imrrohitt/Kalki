from __future__ import annotations


def split_panel_sizes(height: int) -> tuple[int, int]:
    """Top canvas, then talking-head panel. Head gets more than half so a
    portrait face is not cropped. Both even for yuv420.
    """
    head_h = (height * 5) // 8
    if head_h % 2:
        head_h -= 1
    top_h = height - head_h
    if top_h % 2:
        top_h -= 1
        head_h = height - top_h
    return top_h, head_h


def head_panel_filter(width: int, head_h: int) -> str:
    """Fit the full talking-head frame. Never crop the face.

    Portrait sources get a blurred fill on the sides instead of black bars.
    """
    return (
        f"split=2[hs][hb];"
        f"[hb]scale={width}:{head_h}:force_original_aspect_ratio=increase,"
        f"crop={width}:{head_h},boxblur=22:8[bg];"
        f"[hs]scale={width}:{head_h}:force_original_aspect_ratio=decrease:force_divisible_by=2[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )


def build_split_filtergraph(
    *,
    width: int,
    height: int,
    fps: int,
    ass_escaped: str,
    fonts_dir: str,
    zoom_graph: str | None = None,
    top_color: str = "0xF4EFE6",
) -> str:
    """Talking head in the bottom panel, motion canvas on top, captions on the seam."""
    _ = zoom_graph
    top_h, head_h = split_panel_sizes(height)
    finish = f"fps={fps},format=yuv420p,ass={ass_escaped}:fontsdir={fonts_dir}[vout]"
    color = (
        f"color=c={top_color}:s={width}x{top_h}:d=3600:r={fps},"
        f"format=yuv420p[top]"
    )
    return (
        f"[0:v]{head_panel_filter(width, head_h)}[head];"
        f"{color};"
        f"[top][head]vstack=inputs=2[vbase];"
        f"[vbase]{finish}"
    )

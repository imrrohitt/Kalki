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
    top_color: str | None = None,
    theme: str | None = None,
) -> str:
    """Talking head in the bottom panel, motion canvas on top, captions on the seam."""
    from app.renderer.design import get_theme

    _ = zoom_graph
    th = get_theme(theme)
    top_h, head_h = split_panel_sizes(height)
    finish = f"fps={fps},format=yuv420p,ass={ass_escaped}:fontsdir={fonts_dir}[vout]"
    if top_color:
        canvas = (
            f"color=c={top_color}:s={width}x{top_h}:d=3600:r={fps},"
            f"format=yuv420p[top]"
        )
    else:
        # Near-flat vertical gradient: the canvas gets a hint of depth
        # without competing with the type.
        grid = ""
        if th.grid:
            grid = (
                f",drawgrid=w=90:h=90:t=1:"
                f"color={th.grid_color}@{th.grid_opacity}"
            )
        canvas = (
            f"gradients=s={width}x{top_h}:type=linear:"
            f"x0={width // 2}:y0=0:x1={width // 2}:y1={top_h}:"
            f"c0={th.bg_top}:c1={th.bg_bottom}:"
            f"speed=0.00001:duration=3600:rate={fps}"
            f"{grid},format=yuv420p[top]"
        )
    return (
        f"[0:v]{head_panel_filter(width, head_h)}[head];"
        f"{canvas};"
        f"[top][head]vstack=inputs=2[vbase];"
        f"[vbase]{finish}"
    )

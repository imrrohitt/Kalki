from __future__ import annotations

from app.renderer.design import get_theme


def build_full_canvas_filtergraph(
    *,
    width: int,
    height: int,
    fps: int,
    ass_escaped: str,
    fonts_dir: str,
    theme: str | None = None,
) -> str:
    """Generate a 9:16 motion canvas from lavfi (no source video). Ends at [vout]."""
    th = get_theme(theme)
    grid = ""
    if th.grid:
        grid = (
            f",drawgrid=w=90:h=90:t=1:"
            f"color={th.grid_color}@{th.grid_opacity}"
        )
    return (
        f"gradients=s={width}x{height}:type=linear:"
        f"x0={width // 2}:y0=0:x1={width // 2}:y1={height}:"
        f"c0={th.bg_top}:c1={th.bg_bottom}:"
        f"speed=0.00001:duration=3600:rate={fps}"
        f"{grid},fps={fps},format=yuv420p,"
        f"ass={ass_escaped}:fontsdir={fonts_dir}[vout]"
    )

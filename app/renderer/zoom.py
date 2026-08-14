from __future__ import annotations

from dataclasses import dataclass

from app.editorial.models import ZoomDecision, ZoomEasing, ZoomStyle


def _progress(t0: float, t1: float, var: str, comma: str) -> str:
    dur = max(t1 - t0, 0.04)
    return f"min(max(({var}-{t0:.3f})/{dur:.3f}{comma}0){comma}1)"


def _ease(progress: str, easing: ZoomEasing, comma: str) -> str:
    if easing == "linear":
        return progress
    if easing == "ease_in":
        return f"pow({progress}{comma}3)"
    if easing == "ease_out":
        return f"(1-pow(1-{progress}{comma}3))"
    return (
        f"if(lt({progress}{comma}0.5){comma}"
        f"(4*pow({progress}{comma}3)){comma}"
        f"(1-pow(-2*{progress}+2{comma}3)/2))"
    )


def _punch_scale(decision: ZoomDecision, progress: str, comma: str) -> str:
    eased = _ease(progress, decision.easing, comma)
    delta = decision.target_scale - 1.0
    return f"(1+{delta:.3f}*{eased})"


def _inner_scale(decision: ZoomDecision, var: str, comma: str) -> str:
    t0 = decision.start
    t1 = decision.end
    target = decision.target_scale
    style: ZoomStyle = decision.style
    release = decision.release_duration
    progress = _progress(t0, t1, var, comma)

    if style == "hold":
        ease_in = _progress(t0, min(t0 + 0.14, t0 + 0.35 * (t1 - t0)), var, comma)
        rise = _punch_scale(decision, _ease(ease_in, "ease_out", comma), comma)
        body = f"if(lt({var}{comma}{t0 + 0.14:.3f}){comma}{rise}{comma}{target:.3f})"
    elif style == "punch_release":
        mid = t0 + 0.45 * (t1 - t0)
        p_in = _progress(t0, mid, var, comma)
        p_out = _progress(mid, t1, var, comma)
        up = _punch_scale(decision, _ease(p_in, "ease_out", comma), comma)
        down = f"(1+{target - 1:.3f}*(1-{_ease(p_out, 'ease_in', comma)}))"
        return f"if(lt({var}{comma}{mid:.3f}){comma}{up}{comma}{down})"
    else:
        body = _punch_scale(decision, progress, comma)

    if release <= 0.001:
        return body
    p_rel = _progress(t1, t1 + release, var, comma)
    release_expr = f"(1+{target - 1:.3f}*(1-{p_rel}))"
    return f"if(lt({var}{comma}{t1:.3f}){comma}{body}{comma}{release_expr})"


def _window_end(decision: ZoomDecision) -> float:
    return decision.start + decision.ease_in + decision.hold + decision.ease_out


def build_zoom_scale_expr(
    zooms: list[ZoomDecision],
    *,
    var: str = "t",
    comma: str = "\\,",
) -> str:
    expr = "1"
    for zoom in reversed(sorted(zooms, key=lambda z: z.start)):
        if zoom.style == "none" or zoom.target_scale <= 1.001:
            continue
        inner = _inner_scale(zoom, var, comma)
        t1 = _window_end(zoom)
        expr = (
            f"if(between({var}{comma}{zoom.start:.3f}{comma}{t1:.3f})"
            f"{comma}{inner}{comma}{expr})"
        )
    return expr


def _ease_in_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4.0 * t**3
    return 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


@dataclass(frozen=True)
class ZoomPiece:
    start: float
    end: float
    scale: float
    anchor_x: float
    anchor_y: float


def _even(value: float) -> int:
    return int(value) - (int(value) % 2)


def _crop_xy(
    scale: float,
    width: int,
    height: int,
    anchor_x: float,
    anchor_y: float,
) -> tuple[int, int]:
    sw = width * scale
    sh = height * scale
    x = anchor_x * sw - width / 2.0
    y = anchor_y * sh - height / 2.0
    x = max(0.0, min(x, sw - width))
    y = max(0.0, min(y, sh - height))
    return _even(x), _even(y)


def _sample_ramp(
    t0: float,
    t1: float,
    s0: float,
    s1: float,
    steps: int,
    ax: float,
    ay: float,
) -> list[ZoomPiece]:
    dur = t1 - t0
    if dur <= 0.001 or abs(s1 - s0) < 0.004:
        return [ZoomPiece(t0, t1, s1, ax, ay)]
    n = max(2, steps)
    pieces: list[ZoomPiece] = []
    for i in range(n):
        a = i / n
        b = (i + 1) / n
        eased = _ease_in_out_cubic(b)
        scale = round(s0 + (s1 - s0) * eased, 3)
        pieces.append(ZoomPiece(t0 + a * dur, t0 + b * dur, scale, ax, ay))
    return pieces


def zoom_segments(zooms: list[ZoomDecision]) -> list[ZoomPiece]:
    """Stepped ease-in / hold / ease-out covering the timeline. scale=1.0 is wide."""
    active = sorted(
        [z for z in zooms if z.style != "none" and z.target_scale > 1.001],
        key=lambda z: z.start,
    )
    if not active:
        return []
    pieces: list[ZoomPiece] = []
    cursor = 0.0
    for zoom in active:
        start = max(zoom.start, cursor)
        ease_in = max(zoom.ease_in, 0.12)
        hold = max(zoom.hold, 0.08)
        ease_out = max(zoom.ease_out, 0.12)
        hold_start = start + ease_in
        hold_end = hold_start + hold
        span_end = hold_end + ease_out
        ax = zoom.anchor_x
        ay = zoom.anchor_y
        target = float(zoom.target_scale)

        if start > cursor + 0.04:
            pieces.append(ZoomPiece(cursor, start, 1.0, 0.5, 0.5))
        pieces.extend(_sample_ramp(start, hold_start, 1.0, target, 8, ax, ay))
        pieces.append(ZoomPiece(hold_start, hold_end, target, ax, ay))
        pieces.extend(_sample_ramp(hold_end, span_end, target, 1.0, 6, ax, ay))
        cursor = span_end
    pieces.append(ZoomPiece(cursor, -1.0, 1.0, 0.5, 0.5))
    return _merge_adjacent(pieces)


def _merge_adjacent(pieces: list[ZoomPiece]) -> list[ZoomPiece]:
    if not pieces:
        return []
    merged: list[ZoomPiece] = [pieces[0]]
    for piece in pieces[1:]:
        prev = merged[-1]
        same_scale = round(prev.scale, 3) == round(piece.scale, 3)
        same_anchor = abs(prev.anchor_x - piece.anchor_x) < 0.02 and abs(
            prev.anchor_y - piece.anchor_y
        ) < 0.02
        if same_scale and same_anchor and prev.end >= 0 and piece.start - prev.end < 0.02:
            merged[-1] = ZoomPiece(
                prev.start,
                piece.end,
                piece.scale,
                piece.anchor_x,
                piece.anchor_y,
            )
        else:
            merged.append(piece)
    return merged


def build_zoom_filtergraph(
    zooms: list[ZoomDecision] | None,
    width: int,
    height: int,
    fps: int = 30,
) -> str | None:
    """Return a filter_complex graph, or None if there is nothing to zoom."""
    if not zooms:
        return None
    pieces = zoom_segments(zooms)
    if not pieces:
        return None
    return _concat_graph(pieces, width, height, fps)


def _concat_graph(
    pieces: list[ZoomPiece],
    width: int,
    height: int,
    fps: int,
) -> str:
    n = len(pieces)
    labels = "".join(f"[b{i}]" for i in range(n))
    parts = [
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},split={n}{labels}"
    ]
    concat_in = ""
    for i, piece in enumerate(pieces):
        if piece.end < 0:
            trim = f"trim=start={piece.start:.3f}"
        else:
            trim = f"trim=start={piece.start:.3f}:end={piece.end:.3f}"
        chain = f"[b{i}]{trim},setpts=PTS-STARTPTS"
        if piece.scale > 1.001:
            x, y = _crop_xy(piece.scale, width, height, piece.anchor_x, piece.anchor_y)
            chain += (
                f",scale=trunc(iw*{piece.scale:.3f}/2)*2:trunc(ih*{piece.scale:.3f}/2)*2,"
                f"crop={width}:{height}:{x}:{y}"
            )
        chain += f",setsar=1,fps={fps}[s{i}]"
        parts.append(chain)
        concat_in += f"[s{i}]"
    parts.append(f"{concat_in}concat=n={n}:v=1:a=0[vzoom]")
    return ";".join(parts)

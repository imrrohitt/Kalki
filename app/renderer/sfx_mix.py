from __future__ import annotations

from pathlib import Path

from app.editorial.models import SfxHit
from app.editorial.sfx.library import resolve_clip


def build_sfx_mix(
    hits: list[SfxHit],
    *,
    voice_has_audio: bool,
    video_duration: float = 8.0,
) -> tuple[list[Path], str]:
    """Return extra input paths and an audio filter that ends at [aout]."""
    resolved: list[tuple[SfxHit, Path, float, float]] = []
    for hit in hits:
        found = resolve_clip(hit.kind)
        if found is None:
            continue
        path, clip = found
        resolved.append((hit, path, clip.trim, hit.gain if hit.gain else clip.gain))
    if not resolved:
        return [], ""

    parts: list[str] = []
    dur = max(video_duration, 1.0)
    if voice_has_audio:
        parts.append(
            "[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,volume=1.0[voice]"
        )
    else:
        parts.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{dur:.2f},asetpts=PTS-STARTPTS[voice]")
    mix_labels = ["[voice]"]

    files: list[Path] = []
    for index, (hit, path, trim, gain) in enumerate(resolved, start=1):
        files.append(path)
        delay_ms = max(0, int(round(hit.at * 1000)))
        fade_start = max(0.05, trim - 0.12)
        parts.append(
            f"[{index}:a]atrim=0:{trim:.2f},asetpts=PTS-STARTPTS,"
            f"afade=t=out:st={fade_start:.2f}:d=0.12,"
            f"volume={gain:.2f},"
            f"adelay={delay_ms}|{delay_ms},"
            f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[s{index}]"
        )
        mix_labels.append(f"[s{index}]")

    n = len(mix_labels)
    parts.append(
        "".join(mix_labels)
        + f"amix=inputs={n}:duration=first:dropout_transition=0:normalize=0[aout]"
    )
    return files, ";".join(parts)

"""Soundtrack for full-frame talking-head reels.

Premium reels carry a quiet music bed under the whole voice track and only a
handful of accents: a soft riser on the hook and a sparkle when a hand-drawn
oval or underline lands. Nothing fires per word.

Music comes from `MUSIC_DIR` when the creator has dropped licensed tracks there
(the file whose name best matches the director's mood wins). Otherwise a warm
ambient bed is synthesized for the mood — sustained pads and a soft plucked
arpeggio, no drums — so every reel has a bed without licensing risk.
"""

from __future__ import annotations

import logging
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.captions.models import CaptionTimeline
from app.config import settings
from app.editorial.models import SfxHit
from app.editorial.sfx.library import resolve_clip

logger = logging.getLogger(__name__)

SR = 48000
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}

CHORDS = {
    "maj": (0, 4, 7, 12),
    "min": (0, 3, 7, 12),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "maj9": (0, 4, 7, 11, 14),
    "m9": (0, 3, 7, 10, 14),
    "six": (0, 4, 7, 9),
    "sus2": (0, 2, 7, 12),
}


@dataclass(frozen=True)
class Mood:
    root: int  # MIDI note of the tonic (pad register)
    progression: tuple[tuple[int, str], ...]
    bpm: float
    arp: bool
    brightness: float  # 0..1, harmonic content of the pad
    keywords: tuple[str, ...]


MOODS: dict[str, Mood] = {
    "warm_inspiring": Mood(
        62, ((0, "maj9"), (-3, "m7"), (-7, "maj7"), (-5, "six")), 82, True, 0.55,
        ("warm", "inspir", "uplift", "hope", "motiv"),
    ),
    "focused_tech": Mood(
        57, ((0, "m9"), (-4, "maj7"), (3, "maj"), (-2, "six")), 90, True, 0.45,
        ("tech", "focus", "minimal", "future", "corporate", "lofi"),
    ),
    "calm_reflective": Mood(
        63, ((0, "maj7"), (-3, "m7"), (-7, "maj9"), (-5, "sus2")), 70, False, 0.40,
        ("calm", "reflect", "ambient", "soft", "piano", "chill"),
    ),
    "confident_upbeat": Mood(
        67, ((0, "maj"), (-3, "m7"), (-7, "maj7"), (-5, "maj")), 100, True, 0.60,
        ("upbeat", "confident", "happy", "bright", "energetic", "pop"),
    ),
    "serious_story": Mood(
        62, ((0, "m9"), (-4, "maj7"), (-9, "maj"), (-2, "sus2")), 72, False, 0.35,
        ("serious", "story", "cinematic", "dark", "drama", "emotional"),
    ),
}


def _midi_hz(note: float) -> float:
    return 440.0 * 2 ** ((note - 69) / 12)


def _write_wav(path: Path, stereo: np.ndarray) -> None:
    pcm = np.clip(stereo, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


def _pad_note(freq: float, n: int, brightness: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    out = np.zeros((n, 2), np.float32)
    harmonics = 7
    for detune, pan in ((-7.0, 0.2), (0.0, 0.5), (6.0, 0.8)):
        f = freq * 2 ** (detune / 1200)
        voice = np.zeros(n, np.float32)
        for k in range(1, harmonics + 1):
            amp = (1.0 / k) ** (2.2 - 1.2 * brightness)
            phase = rng.uniform(0, 2 * np.pi)
            voice += amp * np.sin(2 * np.pi * f * k * t + phase).astype(np.float32)
        # Slow shimmer so the pad breathes.
        voice *= 1.0 + 0.06 * np.sin(2 * np.pi * rng.uniform(0.08, 0.2) * t + rng.uniform(0, 6))
        out[:, 0] += voice * math.cos(pan * math.pi / 2)
        out[:, 1] += voice * math.sin(pan * math.pi / 2)
    return out


def _pluck(freq: float, n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    env = np.exp(-t / 0.42) * (1 - np.exp(-t / 0.004))
    tone = np.sin(2 * np.pi * freq * t) + 0.28 * np.sin(4 * np.pi * freq * t) + 0.08 * np.sin(6 * np.pi * freq * t)
    return (tone * env).astype(np.float32)


def synthesize_music_bed(mood_name: str, duration: float, path: Path, *, seed: int = 7) -> Path:
    mood = MOODS.get(mood_name) or MOODS["warm_inspiring"]
    rng = np.random.default_rng(seed)
    total = int((duration + 1.0) * SR)
    mix = np.zeros((total, 2), np.float32)
    beat = 60.0 / mood.bpm
    bar = 4 * beat
    chord_len = 2 * bar
    fade = 1.2
    n_chords = int(math.ceil(duration / chord_len)) + 1
    for ci in range(n_chords):
        offset, quality = mood.progression[ci % len(mood.progression)]
        start = int(ci * chord_len * SR)
        if start >= total:
            break
        length = int((chord_len + fade) * SR)
        end = min(total, start + length)
        n = end - start
        intervals = CHORDS[quality]
        root = mood.root + offset
        seg = np.zeros((n, 2), np.float32)
        for iv in intervals:
            seg += _pad_note(_midi_hz(root + iv), n, mood.brightness, rng) * 0.16
        # Soft sub on the chord root.
        tt = np.arange(n, dtype=np.float32) / SR
        sub = 0.22 * np.sin(2 * np.pi * _midi_hz(root - 12) * tt)
        seg += sub[:, None]
        env = np.ones(n, np.float32)
        a = int(fade * SR)
        env[:a] = np.linspace(0, 1, min(a, n))[: min(a, n)]
        rel = int(fade * SR)
        if n > rel:
            env[-rel:] = np.minimum(env[-rel:], np.linspace(1, 0, rel))
        seg *= env[:, None]
        if mood.arp:
            order = [0, 1, 2, 3, 2, 1, 0, 2]
            step = beat / 2
            for si in range(int(chord_len / step)):
                note = root + 12 + intervals[order[si % len(order)] % len(intervals)]
                s0 = int(si * step * SR)
                pn = min(int(1.2 * SR), n - s0)
                if pn <= 0:
                    break
                p = _pluck(_midi_hz(note), pn) * (0.075 if si % 2 == 0 else 0.055)
                pan = 0.35 if si % 2 == 0 else 0.65
                seg[s0 : s0 + pn, 0] += p * math.cos(pan * math.pi / 2)
                seg[s0 : s0 + pn, 1] += p * math.sin(pan * math.pi / 2)
        mix[start:end] += seg
    mix = mix[: int(duration * SR)]
    # Intro swell and outro fade.
    n = len(mix)
    intro = min(n, int(1.8 * SR))
    outro = min(n, int(2.5 * SR))
    mix[:intro] *= np.linspace(0.0, 1.0, intro)[:, None] ** 1.5
    mix[n - outro :] *= np.linspace(1.0, 0.0, outro)[:, None]
    peak = float(np.max(np.abs(mix))) or 1.0
    mix *= 0.89 / peak
    _write_wav(path, mix)
    return path


def synthesize_shimmer(path: Path) -> Path:
    """A soft three-note sparkle for oval / underline moments."""
    if path.exists():
        return path
    n = int(1.6 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    out = np.zeros((n, 2), np.float32)
    for i, (note, pan) in enumerate(((100, 0.3), (104, 0.7), (107, 0.5))):
        delay = int(i * 0.065 * SR)
        tt = t[: n - delay]
        f = _midi_hz(note)
        env = np.exp(-tt / (0.5 - i * 0.08)) * (1 - np.exp(-tt / 0.003))
        tone = (
            np.sin(2 * np.pi * f * tt)
            + 0.35 * np.sin(2 * np.pi * f * 2.76 * tt)
            + 0.12 * np.sin(2 * np.pi * f * 5.4 * tt)
        ) * env * (0.5 - 0.1 * i)
        out[delay:, 0] += tone * math.cos(pan * math.pi / 2)
        out[delay:, 1] += tone * math.sin(pan * math.pi / 2)
    out *= 0.8 / (float(np.max(np.abs(out))) or 1.0)
    _write_wav(path, out)
    return path


def pick_music_track(mood_name: str) -> Path | None:
    folder = settings.resolve_path(settings.music_dir)
    if not folder.is_dir():
        return None
    tracks = sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
    if not tracks:
        return None
    mood = MOODS.get(mood_name) or MOODS["warm_inspiring"]
    keys = (mood_name.lower(),) + mood.keywords

    def score(p: Path) -> int:
        name = p.stem.lower()
        return sum(1 for k in keys if k in name)

    best = max(tracks, key=score)
    return best


def plan_accents(timeline: CaptionTimeline, *, video_duration: float) -> list[SfxHit]:
    """Hook riser plus sparse sparkles on hand-drawn moments. Never per word."""
    hits: list[SfxHit] = []
    caps = timeline.captions
    if caps:
        first = caps[0]
        hits.append(SfxHit(at=max(0.0, first.start), kind="riser", gain=0.14, reason="hook"))
    last_at = -99.0
    for cap in caps[1:]:
        if cap.treatment not in {"oval", "underline", "tape"}:
            continue
        at = cap.start + (0.30 if cap.treatment == "oval" else 0.2)
        if at - last_at < 12.0 or at > video_duration - 0.5:
            continue
        kind = "swoosh" if cap.treatment == "tape" else "shimmer"
        gain = {"oval": 0.13, "underline": 0.09, "tape": 0.10}[cap.treatment]
        hits.append(SfxHit(at=round(at, 3), kind=kind, gain=gain, reason=cap.treatment))
        last_at = at
    return hits[:10]


def build_soundtrack(
    *,
    voice_input: int,
    first_extra_input: int,
    hits: list[SfxHit],
    music_path: Path | None,
    video_duration: float,
    cache_dir: Path,
) -> tuple[list[Path], str]:
    """Extra input files and a filter graph ending at [aout].

    voice → cleaned and gently compressed; music → ducked under the voice;
    accents mixed on top; final loudness normalized for Reels.
    """
    files: list[Path] = []
    parts: list[str] = []
    dur = max(video_duration, 1.0)
    parts.append(
        f"[{voice_input}:a]aformat=sample_fmts=fltp:sample_rates={SR}:channel_layouts=stereo,"
        "highpass=f=75,acompressor=threshold=-21dB:ratio=2.6:attack=6:release=160:makeup=2dB,"
        "asplit=2[voice][sc]"
    )
    labels = ["[voice]"]
    idx = first_extra_input
    if music_path is not None:
        files.append(music_path)
        gain_db = settings.music_gain_db
        parts.append(
            f"[{idx}:a]aformat=sample_fmts=fltp:sample_rates={SR}:channel_layouts=stereo,"
            f"atrim=0:{dur:.3f},asetpts=PTS-STARTPTS,"
            "highpass=f=60,lowpass=f=7500,"
            "aecho=0.8:0.55:70|140|230:0.22|0.15|0.10,"
            f"volume={gain_db:.1f}dB,"
            f"afade=t=out:st={max(0.0, dur - 2.2):.3f}:d=2.2[bed]"
        )
        parts.append(
            "[bed][sc]sidechaincompress=threshold=0.035:ratio=5:attack=35:release=650:makeup=1[music]"
        )
        labels.append("[music]")
        idx += 1
    else:
        parts.append("[sc]anullsink")

    for hit in hits:
        if hit.kind == "shimmer":
            path = synthesize_shimmer(cache_dir / "shimmer.wav")
            trim = 1.5
        else:
            found = resolve_clip(hit.kind)
            if found is None:
                continue
            path, clip = found
            trim = clip.trim
        files.append(path)
        delay = max(0, int(round(hit.at * 1000)))
        label = f"[fx{idx}]"
        parts.append(
            f"[{idx}:a]aformat=sample_fmts=fltp:sample_rates={SR}:channel_layouts=stereo,"
            f"atrim=0:{trim:.2f},asetpts=PTS-STARTPTS,"
            f"afade=t=out:st={max(0.05, trim - 0.18):.2f}:d=0.18,"
            f"volume={hit.gain:.3f},adelay={delay}|{delay}{label}"
        )
        labels.append(label)
        idx += 1

    parts.append(
        "".join(labels)
        + f"amix=inputs={len(labels)}:duration=first:dropout_transition=0:normalize=0,"
        "loudnorm=I=-15:TP=-1.5:LRA=11,"
        f"aresample={SR}[aout]"
    )
    return files, ";".join(parts)

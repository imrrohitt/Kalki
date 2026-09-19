"""Real vocal-emphasis detection: does the speaker's own voice get louder here?

This is the one signal in the whole caption pipeline that comes from the audio
itself rather than the transcript — the LLM never hears tone or volume, only
words. A genuine loudness spike relative to the speaker's OWN baseline (not an
absolute dB threshold, since mic gain and room noise vary per video) is a real
prosodic cue: the speaker just leaned into a word. That is exactly the moment a
"blink" pop or a colored highlight should land, on top of whatever the LLM's
text-only judgement already decided.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# The pipeline always extracts 16 kHz mono PCM16 (see app/media/audio.py) — no
# decoder dependency needed, Python's stdlib `wave` module reads it directly.
HOP_SECONDS = 0.20


def voice_loudness_track(wav_path: str, *, hop_s: float = HOP_SECONDS) -> tuple[np.ndarray, np.ndarray]:
    """RMS loudness (dBFS) of the voice track in small hops across the whole clip."""
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError(f"expected 16-bit PCM, got sampwidth={sampwidth}")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    hop = max(1, int(round(hop_s * sr)))
    n_hops = len(data) // hop
    if n_hops == 0:
        return np.zeros(1), np.array([-60.0])
    trimmed = data[: n_hops * hop].reshape(n_hops, hop)
    rms = np.sqrt(np.mean(trimmed * trimmed, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-9)
    times = np.arange(n_hops, dtype=np.float32) * hop_s
    return times, db


def span_loudness(times: np.ndarray, db: np.ndarray, start: float, end: float) -> float:
    mask = (times >= start) & (times < max(end, start + 1e-3))
    if mask.any():
        return float(db[mask].mean())
    idx = int(np.argmin(np.abs(times - start)))
    return float(db[idx])


def flag_vocal_emphasis(
    spans: list[tuple[float, float]],
    times: np.ndarray,
    db: np.ndarray,
    *,
    min_gap_s: float = 5.0,
    max_fraction: float = 0.18,
) -> list[bool]:
    """True for the rare span where the voice is genuinely louder than the
    speaker's own typical level — real emphasis, spaced out, capped in count.

    Real speech dynamics are often subtle: a talking-head recording (phone
    mic, outdoor noise floor) rarely swings more than a handful of dB even on
    a genuinely emphasized word, so the threshold reads relative to the
    speaker's own interquartile spread rather than demanding a loud shout.
    """
    n = len(spans)
    if n == 0 or len(times) == 0:
        return [False] * n
    levels = np.array([span_loudness(times, db, s, e) for s, e in spans], dtype=np.float32)
    baseline = float(np.median(levels))
    q1, q3 = np.percentile(levels, [25, 75])
    spread = float(q3 - q1) or 1.0
    threshold = baseline + max(1.2, 0.40 * spread)
    budget = max(1, int(round(n * max_fraction)))
    order = sorted(range(n), key=lambda i: levels[i], reverse=True)
    flags = [False] * n
    last_t = -1e9
    picked = 0
    for i in order:
        if picked >= budget or levels[i] < threshold:
            break
        if spans[i][0] - last_t < min_gap_s:
            continue
        flags[i] = True
        last_t = spans[i][0]
        picked += 1
    return flags


def try_voice_loudness_track(wav_path: str | Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Best-effort wrapper — a bad or missing wav should never fail the reel."""
    try:
        return voice_loudness_track(str(wav_path))
    except Exception as exc:  # noqa: BLE001 - acoustic emphasis is a bonus signal
        logger.warning("voice loudness analysis skipped: %s", exc)
        return None, None

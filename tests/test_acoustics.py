import numpy as np

from app.captions.acoustics import flag_vocal_emphasis, span_loudness, voice_loudness_track


def _write_wav(path, samples: np.ndarray, sr: int = 16000) -> None:
    import wave

    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def test_voice_loudness_track_reads_real_pcm_wav(tmp_path):
    sr = 16000
    t = np.arange(sr * 3, dtype=np.float32) / sr
    # Quiet throughout, one clear loud burst in the middle second.
    quiet = 0.03 * np.sin(2 * np.pi * 200 * t)
    loud_mask = (t >= 1.0) & (t < 2.0)
    samples = quiet.copy()
    samples[loud_mask] = 0.5 * np.sin(2 * np.pi * 200 * t[loud_mask])
    path = tmp_path / "voice.wav"
    _write_wav(path, samples, sr)

    times, db = voice_loudness_track(str(path))
    assert len(times) == len(db)
    loud_level = span_loudness(times, db, 1.2, 1.8)
    quiet_level = span_loudness(times, db, 0.2, 0.8)
    assert loud_level > quiet_level + 10  # a real, unmistakable jump


def test_flag_vocal_emphasis_finds_the_spike_and_only_the_spike():
    times = np.arange(0, 30, 0.2, dtype=np.float32)
    baseline = -32.0
    db = np.full_like(times, baseline) + np.random.default_rng(0).normal(0, 0.6, len(times))
    # One genuine spike around t=15s.
    db[(times >= 14.5) & (times < 15.5)] += 12.0

    spans = [(float(i * 2), float(i * 2 + 1.8)) for i in range(15)]
    flags = flag_vocal_emphasis(spans, times, db, min_gap_s=5.0, max_fraction=0.18)

    assert any(flags)
    flagged_spans = [s for s, f in zip(spans, flags) if f]
    assert any(s[0] <= 15.5 and s[1] >= 14.5 for s in flagged_spans)
    # Not everything gets flagged — this is a rare-accent signal.
    assert sum(flags) <= 3


def test_flag_vocal_emphasis_respects_min_gap_and_budget():
    times = np.arange(0, 60, 0.2, dtype=np.float32)
    rng = np.random.default_rng(1)
    db = -32.0 + rng.normal(0, 0.5, len(times))
    # Several loud bursts, some close together.
    for center in (5.0, 5.5, 20.0, 40.0, 40.3):
        db[(times >= center - 0.3) & (times < center + 0.3)] += 15.0

    spans = [(float(i), float(i + 0.9)) for i in range(60)]
    flags = flag_vocal_emphasis(spans, times, db, min_gap_s=5.0, max_fraction=0.18)
    flagged_starts = sorted(s for (s, _), f in zip(spans, flags) if f)
    for a, b in zip(flagged_starts, flagged_starts[1:]):
        assert b - a >= 5.0
    assert sum(flags) <= round(60 * 0.18)


def test_flag_vocal_emphasis_handles_no_data():
    assert flag_vocal_emphasis([], np.array([]), np.array([])) == []
    assert flag_vocal_emphasis([(0.0, 1.0)], np.array([]), np.array([])) == [False]

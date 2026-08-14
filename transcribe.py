import json
from faster_whisper import WhisperModel

AUDIO = "Video by rohit_k_idea [DaDMiRYNAR-].wav"
OUTPUT = "output/transcript.json"

model = WhisperModel("tiny", device="cpu", compute_type="int8")

segments, info = model.transcribe(
    AUDIO,
    language=None,
    task="transcribe",
    beam_size=5,
    vad_filter=True,
    word_timestamps=True,
)

results = []
for seg in segments:
    words = []
    if seg.words:
        for w in seg.words:
            words.append({
                "word": w.word,
                "start": w.start,
                "end": w.end,
                "probability": w.probability,
            })
    results.append({
        "id": len(results),
        "start": seg.start,
        "end": seg.end,
        "text": seg.text.strip(),
        "words": words,
    })
    print(f"[{seg.start:.2f} -> {seg.end:.2f}] {seg.text.strip()}")

transcript = {
    "language": info.language,
    "language_probability": info.language_probability,
    "segments": results,
}

import os
os.makedirs("output", exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(transcript, f, ensure_ascii=False, indent=2)

print(f"\nLanguage: {info.language} ({info.language_probability:.2f})")
print(f"Segments: {len(results)}")
print(f"Saved: {OUTPUT}")

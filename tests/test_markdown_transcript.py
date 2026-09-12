from pathlib import Path

from app.transcription.markdown import parse_timestamped_markdown, transcript_from_upload


def test_parse_timestamped_english_markdown():
    text = Path(
        "storage/jobs/11e1945a-8e3d-4faf-a33c-48420d670bfe/transcription.md"
    ).read_text(encoding="utf-8")
    transcript = parse_timestamped_markdown(text, duration=119.8)
    assert transcript.language == "en"
    assert len(transcript.segments) == 7
    words = [w.word for seg in transcript.segments for w in seg.words]
    assert "Hi" in words or "hi" in [w.lower() for w in words]
    assert any("RAG" in w or "rag" in w.lower() for w in words)
    assert any("Cursor" in w or "Claude" in w for w in words)
    assert transcript.segments[0].start == 0.0
    assert transcript.segments[-1].end >= 119.0
    assert all(w.end > w.start for seg in transcript.segments for w in seg.words)
    joined = " ".join(seg.text for seg in transcript.segments)
    assert "Devanagari" not in joined
    assert "आपिँ" not in joined


def test_transcript_from_upload_json():
    payload = b'{"language":"en","duration":2,"segments":[{"start":0,"end":1,"text":"hi","words":[{"word":"hi","start":0,"end":1}]}]}'
    transcript = transcript_from_upload(payload, "captions.json")
    assert transcript.segments[0].words[0].word == "hi"

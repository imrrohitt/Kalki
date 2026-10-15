from pathlib import Path

from app.transcription.markdown import parse_timestamped_markdown, transcript_from_upload


def test_parse_timestamped_english_markdown():
    text = "\n\n".join(
        f"**{a} - {b}**\n\n{line}"
        for a, b, line in [
            ("00:00", "00:15", "Hi, today I want to talk about how I use RAG in real projects."),
            ("00:15", "00:30", "First, I set up the retriever with Cursor and Claude."),
            ("00:30", "00:50", "Then I chunk the documents carefully so context is preserved."),
            ("00:50", "01:10", "Next comes the embedding step, which decides recall quality."),
            ("01:10", "01:25", "After that I evaluate answers against a small test set."),
            ("01:25", "01:45", "Finally I ship it behind a cache to keep latency low."),
            ("01:45", "01:59", "That is the whole flow, thank you for watching."),
        ]
    )
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

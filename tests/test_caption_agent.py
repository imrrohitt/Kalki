from app.captions.agent import (
    _captions_from_ids,
    _chunk_words,
    _flatten_words,
    _lite_llm_model,
    _parse_caption_groups,
)
from app.transcription.models import Segment, Transcript, Word


def _words(*tokens: str) -> list[Word]:
    words = []
    t = 0.0
    for token in tokens:
        words.append(Word(word=token, start=t, end=t + 0.2, probability=0.9))
        t += 0.2
    return words


def test_deepseek_openai_prefix_uses_deepseek_provider():
    assert _lite_llm_model("openai/deepseek-v4-flash") == "deepseek/deepseek-v4-flash"
    assert _lite_llm_model("deepseek/deepseek-v4-flash") == "deepseek/deepseek-v4-flash"


def test_parse_caption_groups_object_and_array():
    assert _parse_caption_groups('{"captions": [{"ids": [0]}]}') == [{"ids": [0]}]
    assert _parse_caption_groups([{"ids": [1, 2]}]) == [{"ids": [1, 2]}]


def test_captions_from_ids_groups_and_emphasis():
    words = _words("if", "you", "give", "RAG")
    captions = _captions_from_ids(
        [
            {"ids": [0, 1], "text": "IF YOU", "emphasis_id": None},
            {"ids": [2, 3], "text": "GIVE RAG", "emphasis_id": 3},
        ],
        words,
    )
    assert len(captions) == 2
    assert captions[0]["text"] == "IF YOU"
    assert captions[1]["words"][-1]["emphasis"] is True
    assert captions[0]["words"][0]["emphasis"] is False


def test_captions_rewrite_asr_slips():
    words = _words("the", "RAKA", "system")
    captions = _captions_from_ids(
        [{"ids": [0, 1, 2], "text": "THE RAKA\\nSYSTEM"}],
        words,
    )
    assert "RAG" in captions[0]["text"]
    assert "RAKA" not in captions[0]["text"]
    assert captions[0]["words"][1]["text"] == "RAG"


def test_captions_from_ids_fills_skipped_words():
    words = _words("a", "b", "c")
    captions = _captions_from_ids([{"ids": [0, 2], "text": "A C"}], words)
    assert [w["text"] for w in captions[0]["words"]] == ["A", "C"]
    assert captions[0]["start"] == words[0].start
    assert captions[0]["end"] >= words[2].end


def test_captions_from_hinglish_asr_use_english_text():
    words = _words("aapko", "data", "secure", "karna", "hai")
    captions = _captions_from_ids(
        [
            {
                "ids": [0, 1, 2, 3, 4],
                "text": "You Need To\\nSecure Data",
                "emphasis_id": 1,
            }
        ],
        words,
    )
    assert captions[0]["text"] == "You Need To\nSecure Data"
    shown = [w["text"] for w in captions[0]["words"]]
    assert shown == ["You", "Need", "To", "Secure", "Data"]
    assert "aapko" not in shown
    assert "karna" not in shown
    assert any(w["emphasis"] for w in captions[0]["words"])


def test_captions_skip_devanagari_leftover():
    words = _words("hello", "दुनिया")
    captions = _captions_from_ids(
        [{"ids": [0], "text": "Hello"}],
        words,
    )
    assert len(captions) == 1
    assert captions[0]["text"] == "Hello"


def test_flatten_and_chunk_words():
    transcript = Transcript(
        language="en",
        language_probability=1.0,
        duration=5.0,
        segments=[
            Segment(
                start=0.0,
                end=3.0,
                text="hello world again",
                words=[
                    Word(word="hello", start=0.0, end=0.4, probability=0.9),
                    Word(word="world", start=0.4, end=0.8, probability=0.9),
                    Word(word="again", start=2.5, end=2.8, probability=0.9),
                ],
            )
        ],
    )
    flat = _flatten_words(transcript)
    chunks = _chunk_words(flat, chunk_seconds=1.0)
    assert len(flat) == 3
    assert len(chunks) == 2
    assert [w.word for w in chunks[0][1]] == ["hello", "world"]
    assert [w.word for w in chunks[1][1]] == ["again"]


def test_captions_split_leftover_words_instead_of_one_blob():
    tokens = [f"w{i}" for i in range(12)]
    words = _words(*tokens)
    captions = _captions_from_ids(
        [{"ids": [0, 1], "text": "Hello There"}],
        words,
    )
    assert len(captions) > 3
    leftover = [c for c in captions if c["start"] >= words[2].start]
    assert leftover
    assert all(c["end"] - c["start"] <= 4.0 for c in leftover)
    assert all(len(c["words"]) <= 6 for c in leftover)


def test_captions_explode_huge_last_group():
    tokens = [f"word{i}" for i in range(40)]
    words = _words(*tokens)
    # 0.2s each → 8s span for 40 words; last LLM group swallows the rest.
    captions = _captions_from_ids(
        [
            {"ids": [0, 1], "text": "Intro Line"},
            {"ids": [2, 39], "text": "Deployment"},
        ],
        words,
    )
    assert len(captions) >= 12
    assert all(c["end"] - c["start"] <= 4.0 for c in captions)
    assert all(c["text"].lower() != "deployment" or (c["end"] - c["start"]) <= 4.0 for c in captions)
    last = captions[-1]
    assert last["end"] >= words[-1].end - 0.05
    # Speech at the end still has a caption, not a 8s freeze on "Deployment".
    assert last["start"] >= words[30].start


def test_explode_caption_timeline_splits_stuck_line():
    from app.captions.heuristic import explode_caption_timeline
    from app.captions.models import Caption, CaptionTimeline, CaptionWord

    words = [
        CaptionWord(text=f"w{i}", start=i * 0.3, end=i * 0.3 + 0.28)
        for i in range(80)
    ]
    stuck = Caption(
        start=words[0].start,
        end=words[-1].end,
        text="Deployment",
        words=words,
    )
    timeline = explode_caption_timeline(CaptionTimeline(captions=[stuck]))
    assert len(timeline.captions) >= 25
    assert all(c.end - c.start <= 4.0 for c in timeline.captions)
    assert timeline.captions[-1].end >= words[-1].end - 0.9

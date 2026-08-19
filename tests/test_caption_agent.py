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
    assert [w["text"] for w in captions[0]["words"]] == ["a", "b", "c"]


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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.transcription.faster_whisper_provider import FasterWhisperProvider
from app.transcription.models import Transcript


def test_faster_whisper_provider_maps_words():
    fake_seg = SimpleNamespace(
        start=0.0,
        end=1.0,
        text=" AI is cool",
        words=[
            SimpleNamespace(word=" AI", start=0.0, end=0.4, probability=0.97),
            SimpleNamespace(word=" is", start=0.4, end=0.6, probability=0.99),
            SimpleNamespace(word=" cool", start=0.6, end=1.0, probability=0.95),
        ],
    )
    fake_info = SimpleNamespace(
        language="en",
        language_probability=0.98,
        duration=1.0,
    )

    provider = FasterWhisperProvider(model_size="tiny", device="cpu", compute_type="int8")

    with patch.object(provider, "_get_model") as get_model:
        model = MagicMock()
        model.transcribe.return_value = (iter([fake_seg]), fake_info)
        get_model.return_value = model

        result = provider._transcribe_sync("dummy.wav")

    assert isinstance(result, Transcript)
    assert result.language == "en"
    assert result.duration == 1.0
    assert len(result.segments) == 1
    assert result.segments[0].words[0].word == "AI"
    assert result.segments[0].words[0].start == 0.0
    assert result.segments[0].words[0].end == 0.4
    assert model.transcribe.call_args.kwargs["word_timestamps"] is True

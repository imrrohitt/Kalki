from __future__ import annotations

import asyncio
import logging

from faster_whisper import WhisperModel

from app.config import settings
from app.transcription.models import Segment, Transcript, Word

logger = logging.getLogger(__name__)


class FasterWhisperProvider:
    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        self.model_size = model_size or settings.whisper_model
        self.device = device or settings.whisper_device
        self.compute_type = compute_type or settings.whisper_compute_type
        self._model: WhisperModel | None = None

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            logger.info(
                "loading whisper model=%s device=%s compute=%s",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("whisper model ready")
        return self._model

    def _detect_language(self, model: WhisperModel, audio_path: str) -> str | None:
        try:
            from faster_whisper.audio import decode_audio

            audio = decode_audio(audio_path, sampling_rate=16000)
            language, probability, _ = model.detect_language(
                audio, vad_filter=settings.whisper_vad_enabled
            )
            logger.info("whisper detected language=%s p=%.2f", language, probability)
            return language
        except Exception as exc:  # noqa: BLE001 - detection is advisory only
            logger.warning("whisper language detection failed: %s", exc)
            return None

    def _transcribe_sync(self, audio_path: str) -> Transcript:
        model = self._get_model()
        vad_parameters = None
        if settings.whisper_vad_enabled:
            vad_parameters = {
                "min_silence_duration_ms": settings.whisper_min_silence_ms,
            }

        spoken = self._detect_language(model, audio_path)
        # Hindi / Hinglish through the multilingual decoder comes back as broken
        # Devanagari. Decoding straight to English keeps word timings and gives
        # the caption director real sentences to work with.
        output_language = (settings.whisper_output_language or "").strip() or None
        language = output_language or spoken
        logger.info(
            "whisper transcribe started (spoken=%s, output=%s)", spoken, language
        )
        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            beam_size=settings.whisper_beam_size,
            word_timestamps=True,
            vad_filter=settings.whisper_vad_enabled,
            vad_parameters=vad_parameters,
            initial_prompt=settings.whisper_initial_prompt or None,
            condition_on_previous_text=False,
        )
        segments_list = []
        last_log = 0.0
        for seg in segments_iter:
            segments_list.append(seg)
            if seg.end - last_log >= 15.0:
                logger.info(
                    "whisper progress %.0fs / ~%.0fs (%s segments)",
                    seg.end,
                    getattr(info, "duration", 0.0) or 0.0,
                    len(segments_list),
                )
                last_log = float(seg.end)

        segments: list[Segment] = []
        for seg in segments_list:
            words: list[Word] = []
            if seg.words:
                for w in seg.words:
                    words.append(
                        Word(
                            word=w.word.strip(),
                            start=float(w.start),
                            end=float(w.end),
                            probability=float(w.probability or 0.0),
                        )
                    )
            segments.append(
                Segment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text.strip(),
                    words=words,
                )
            )

        duration = getattr(info, "duration", None)
        if duration is None and segments:
            duration = segments[-1].end

        return Transcript(
            language=getattr(info, "language", None),
            spoken_language=spoken,
            language_probability=getattr(info, "language_probability", None),
            duration=float(duration) if duration is not None else None,
            segments=segments,
        )

    async def transcribe(self, audio_path: str) -> Transcript:
        return await asyncio.to_thread(self._transcribe_sync, audio_path)

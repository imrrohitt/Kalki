from typing import Protocol

from app.transcription.models import Transcript


class TranscriptionProvider(Protocol):
    async def transcribe(self, audio_path: str) -> Transcript:
        ...

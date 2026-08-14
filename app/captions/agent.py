from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from litellm.exceptions import RateLimitError
from pydantic import ValidationError

from app.captions.models import Caption, CaptionTimeline, CaptionWord
from app.captions.validation import (
    CaptionValidationError,
    format_validation_error,
    validate_caption_timeline,
)
from app.config import settings
from app.transcription.models import Transcript, Word


CAPTION_INSTRUCTION = """You are CaptionAgent for short-form vertical social videos.

Convert word-level transcript words into caption groups.

Rules:
- Use ONLY provided word timestamps. Never invent or change timestamps.
- Group 2-5 words per caption when natural.
- Prefer semantic phrase breaks.
- Avoid awkward breaks and captions shorter than 0.2s.
- style=dynamic_social, position=bottom_center, animation=pop
- Mark important nouns/verbs with emphasis=true
- Each word object must use key "text" (not "word")
- word.end must be > word.start
- Return ONLY raw JSON (no markdown, no tools) with shape:
{
  "version": "1.0",
  "style": "dynamic_social",
  "captions": [
    {
      "start": 0.2,
      "end": 1.4,
      "text": "AI IS",
      "position": "bottom_center",
      "animation": "pop",
      "words": [
        {"text": "AI", "start": 0.2, "end": 0.55, "emphasis": true},
        {"text": "IS", "start": 0.55, "end": 0.8, "emphasis": false}
      ]
    }
  ]
}
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    # Groq sometimes wraps tool-like junk; try to recover JSON object.
    if "<function" in text:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _normalize_timeline_dict(data: dict[str, Any]) -> dict[str, Any]:
    captions = []
    for cap in data.get("captions", []):
        words_in = cap.get("words") or []
        words_out = []
        for w in words_in:
            text = str(w.get("text") or w.get("word") or "").strip()
            if not text:
                continue
            start = float(w.get("start", 0))
            end = float(w.get("end", start))
            if end <= start:
                end = start + 0.08
            words_out.append(
                {
                    "text": text,
                    "start": start,
                    "end": end,
                    "emphasis": bool(w.get("emphasis", False)),
                }
            )
        if not words_out:
            continue
        start = float(cap.get("start", words_out[0]["start"]))
        end = float(cap.get("end", words_out[-1]["end"]))
        if end <= start:
            end = start + 0.2
        # Clamp caption bounds to word bounds for consistency.
        start = min(start, words_out[0]["start"])
        end = max(end, words_out[-1]["end"])
        text = str(cap.get("text") or " ".join(x["text"] for x in words_out)).strip()
        captions.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "position": cap.get("position") or "bottom_center",
                "animation": cap.get("animation") or "pop",
                "words": words_out,
            }
        )
    return {
        "version": data.get("version") or "1.0",
        "style": data.get("style") or "dynamic_social",
        "captions": captions,
    }


def _flatten_words(transcript: Transcript) -> list[Word]:
    words: list[Word] = []
    for seg in transcript.segments:
        for w in seg.words:
            if w.word.strip():
                words.append(w)
    return words


def _chunk_words(words: list[Word], chunk_seconds: float = 12.0) -> list[list[Word]]:
    if not words:
        return []
    chunks: list[list[Word]] = []
    current: list[Word] = []
    chunk_start = words[0].start
    for w in words:
        if current and (w.start - chunk_start) >= chunk_seconds:
            chunks.append(current)
            current = [w]
            chunk_start = w.start
        else:
            current.append(w)
    if current:
        chunks.append(current)
    return chunks


def _heuristic_captions(words: list[Word]) -> list[Caption]:
    """Deterministic 2-4 word grouping using Whisper timestamps only."""
    captions: list[Caption] = []
    i = 0
    emphasis_words = {
        "ai",
        "llm",
        "rag",
        "raka",
        "fine",
        "tuning",
        "data",
        "model",
        "train",
        "domain",
    }
    while i < len(words):
        size = 3
        # Prefer ending on punctuation-ish or pause.
        if i + 1 < len(words) and (words[i + 1].start - words[i].end) > 0.35:
            size = 1
        elif i + 2 < len(words) and (words[i + 2].start - words[i + 1].end) > 0.35:
            size = 2
        elif i + 4 <= len(words):
            size = 4 if (words[i + 3].end - words[i].start) < 1.8 else 3
        else:
            size = min(3, len(words) - i)

        group = words[i : i + size]
        cap_words = []
        for w in group:
            start = float(w.start)
            end = float(w.end)
            if end <= start:
                end = start + 0.08
            text = w.word.strip()
            cap_words.append(
                CaptionWord(
                    text=text,
                    start=start,
                    end=end,
                    emphasis=text.lower().strip(".,!?") in emphasis_words,
                )
            )
        start = cap_words[0].start
        end = max(cap_words[-1].end, start + 0.2)
        captions.append(
            Caption(
                start=start,
                end=end,
                text=" ".join(w.text for w in cap_words).upper(),
                position="bottom_center",
                animation="pop",
                words=cap_words,
            )
        )
        i += size
    return captions


class CaptionAgentService:
    def __init__(self) -> None:
        kwargs: dict[str, Any] = {}
        if settings.llm_api_key:
            kwargs["api_key"] = settings.llm_api_key
        if settings.llm_base_url:
            kwargs["api_base"] = settings.llm_base_url

        if settings.llm_api_key and settings.llm_model.startswith("groq/"):
            import os

            os.environ.setdefault("GROQ_API_KEY", settings.llm_api_key)

        # Avoid ADK output_schema/tool calling — small local models + Groq
        # handle plain JSON text more reliably than tool schemas.
        self.agent = LlmAgent(
            name="caption_agent",
            model=LiteLlm(model=settings.llm_model, **kwargs),
            description="Builds dynamic social caption timelines from word timestamps.",
            instruction=CAPTION_INSTRUCTION,
        )
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.agent,
            app_name="reel_editor",
            session_service=self.session_service,
        )

    async def _run_once(self, prompt: str, session_id: str) -> str:
        content = types.Content(role="user", parts=[types.Part(text=prompt)])

        last_error: Exception | None = None
        for attempt in range(5):
            sid = session_id if attempt == 0 else f"{session_id}-r{attempt}"
            await self.session_service.create_session(
                app_name="reel_editor",
                user_id="pipeline",
                session_id=sid,
            )
            try:
                final_text = ""
                async for event in self.runner.run_async(
                    user_id="pipeline",
                    session_id=sid,
                    new_message=content,
                ):
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                final_text += part.text
                if not final_text.strip():
                    raise ValueError("Caption agent returned empty response")
                return final_text
            except RateLimitError as exc:
                last_error = exc
                await asyncio.sleep(8 + attempt * 4)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if "RateLimitError" in msg or "rate_limit_exceeded" in msg:
                    last_error = exc
                    await asyncio.sleep(8 + attempt * 4)
                    continue
                raise

        raise RuntimeError(f"Caption agent rate-limited: {last_error}")

    async def _generate_chunk(
        self,
        words: list[Word],
        video_duration: float,
        language: str | None,
        job_id: str,
        chunk_index: int,
    ) -> list[Caption]:
        prompt = (
            "Create CaptionTimeline JSON for ONLY these words.\n"
            "Keep 2-5 words per caption.\n"
            "Return ONLY JSON. No markdown.\n\n"
            + json.dumps(
                {
                    "duration": video_duration,
                    "language": language,
                    "words": [
                        {"text": w.word, "start": round(w.start, 2), "end": round(w.end, 2)}
                        for w in words
                    ],
                },
                ensure_ascii=False,
            )
        )

        last_error: Exception | None = None
        for attempt in range(1):
            try:
                raw = await self._run_once(
                    prompt,
                    session_id=f"{job_id}-c{chunk_index}-a{attempt}",
                )
                data = _normalize_timeline_dict(_extract_json(raw))
                timeline = validate_caption_timeline(data, video_duration)
                return timeline.captions
            except Exception as exc:  # noqa: BLE001 - fall back for tiny local models
                last_error = exc

        _ = last_error
        return _heuristic_captions(words)

    async def generate(
        self,
        transcript: Transcript,
        video_duration: float,
        job_id: str,
    ) -> CaptionTimeline:
        words = _flatten_words(transcript)
        if not words:
            raise RuntimeError("No words in transcript for caption generation")

        chunks = _chunk_words(words, chunk_seconds=12.0)
        all_captions: list[Caption] = []
        use_local = settings.llm_model.startswith("ollama/")
        for index, chunk in enumerate(chunks):
            if index > 0 and settings.llm_model.startswith("groq/"):
                await asyncio.sleep(6)
            if use_local:
                # gemma:2b is unreliable for full JSON timelines; keep Whisper timing.
                caps = _heuristic_captions(chunk)
            else:
                caps = await self._generate_chunk(
                    words=chunk,
                    video_duration=video_duration,
                    language=transcript.language,
                    job_id=job_id,
                    chunk_index=index,
                )
            all_captions.extend(caps)

        timeline = CaptionTimeline(
            version="1.0",
            style="dynamic_social",
            captions=all_captions,
        )
        return validate_caption_timeline(timeline, video_duration)

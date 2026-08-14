from __future__ import annotations

import json
import os
import re
from typing import Any

from litellm import acompletion
from pydantic import ValidationError

from app.captions.models import CaptionTimeline
from app.captions.validation import (
    CaptionValidationError,
    format_validation_error,
    validate_caption_timeline,
)
from app.config import settings
from app.transcription.models import Transcript, Word


CAPTION_INSTRUCTION = """You are a senior Reels caption editor.

Turn numbered transcript words into short, readable, professional captions.

RULES
- Group by meaning, not by a fixed word count. Usually 2–4 words.
- Keep phrases together: fine tuning, RAG system, AI interviews, domain data, LLM.
- Break after a complete thought, a question, or a pause.
- caption text: UPPERCASE, max 2 lines, use \\n for a line break.
- Each line ~12–20 characters. No emojis, hashtags, or extra punctuation except ?.
- Lightly clean Hinglish grammar in the display text only.
- emphasis: 0 or 1 word index per caption (the payload word: RAG, LLM, AI, TUNING, DATA).
- Return ONLY JSON. No markdown.

OUTPUT
{
  "captions": [
    {
      "ids": [0, 1, 2],
      "text": "IF YOU ARE\\nGIVING",
      "emphasis_id": 2,
      "position": "bottom_center"
    }
  ]
}

ids must be consecutive, cover every word in order, no duplicates, no skips.
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
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


def _flatten_words(transcript: Transcript) -> list[Word]:
    words: list[Word] = []
    for seg in transcript.segments:
        for w in seg.words:
            token = w.word.strip()
            if not token:
                continue
            end = float(w.end) if w.end > w.start else float(w.start) + 0.08
            words.append(Word(word=token, start=float(w.start), end=end, probability=w.probability))
    return words


def _chunk_words(words: list[Word], chunk_seconds: float = 18.0) -> list[tuple[int, list[Word]]]:
    chunks: list[tuple[int, list[Word]]] = []
    start_idx = 0
    current: list[Word] = []
    chunk_t0 = words[0].start if words else 0.0
    for i, w in enumerate(words):
        if current and (w.start - chunk_t0) >= chunk_seconds:
            chunks.append((start_idx, current))
            start_idx = i
            current = [w]
            chunk_t0 = w.start
        else:
            if not current:
                start_idx = i
                chunk_t0 = w.start
            current.append(w)
    if current:
        chunks.append((start_idx, current))
    return chunks


def _captions_from_ids(
    groups: list[dict[str, Any]],
    source_words: list[Word],
    offset: int,
) -> list[dict[str, Any]]:
    captions: list[dict[str, Any]] = []
    cursor = 0
    for group in groups:
        raw_ids = group.get("ids") or []
        ids = []
        for item in raw_ids:
            try:
                ids.append(int(item) + offset)
            except (TypeError, ValueError):
                continue
        if not ids:
            continue
        ids = sorted(set(ids))
        # Fill gaps so no source word is dropped.
        lo, hi = ids[0], ids[-1]
        if lo < cursor:
            lo = cursor
        if lo > cursor:
            lo = cursor
        hi = max(hi, lo)
        used = list(range(lo, hi + 1))
        cursor = hi + 1
        group_words = [source_words[i] for i in used if 0 <= i < len(source_words)]
        if not group_words:
            continue
        emphasis_id = group.get("emphasis_id")
        try:
            emphasis_local = int(emphasis_id) if emphasis_id is not None else None
        except (TypeError, ValueError):
            emphasis_local = None
        text = str(group.get("text") or " ".join(w.word.upper() for w in group_words))
        captions.append(
            {
                "start": group_words[0].start,
                "end": max(group_words[-1].end, group_words[0].start + 0.25),
                "text": text.replace("\\n", "\n").strip(),
                "position": group.get("position") or "bottom_center",
                "animation": "pop",
                "words": [
                    {
                        "text": w.word,
                        "start": w.start,
                        "end": w.end,
                        "emphasis": used[local_i] == emphasis_local
                        if emphasis_local is not None
                        else False,
                    }
                    for local_i, w in enumerate(group_words)
                ],
            }
        )
    if cursor < len(source_words):
        leftover = source_words[cursor:]
        captions.append(
            {
                "start": leftover[0].start,
                "end": leftover[-1].end,
                "text": " ".join(w.word.upper() for w in leftover),
                "position": "bottom_center",
                "animation": "pop",
                "words": [
                    {"text": w.word, "start": w.start, "end": w.end, "emphasis": False}
                    for w in leftover
                ],
            }
        )
    return captions


class CaptionAgentService:
    def __init__(self) -> None:
        self.api_key = settings.llm_api_key.replace("Bearer ", "").strip()
        self.api_base = (settings.llm_base_url or "").rstrip("/") or None
        self.model = settings.llm_model
        if self.api_key:
            os.environ.setdefault("DEEPSEEK_API_KEY", self.api_key)

    async def _complete(self, user_prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": CAPTION_INSTRUCTION},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1200,
            "timeout": 45,
            "drop_params": True,
            # DeepSeek v4 thinks by default; that is why this was so slow.
            "extra_body": {
                "thinking": {"type": "disabled"},
                "reasoning_effort": "low",
            },
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        response = await acompletion(**kwargs)
        text = response.choices[0].message.content or ""
        if not str(text).strip():
            raise ValueError("Caption LLM returned empty response")
        return str(text)

    async def _generate_chunk(
        self,
        words: list[Word],
        offset: int,
        video_duration: float,
    ) -> list[dict[str, Any]]:
        payload = {
            "offset": offset,
            "duration": round(video_duration, 2),
            "words": [
                {
                    "id": i,
                    "text": w.word,
                    "start": round(w.start, 2),
                    "end": round(w.end, 2),
                }
                for i, w in enumerate(words)
            ],
        }
        prompt = (
            "Group these words into captions. ids are 0-based inside THIS list.\n"
            "Return compact JSON only.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        last_error: Exception | None = None
        for _ in range(2):
            try:
                data = _extract_json(await self._complete(prompt))
                groups = data.get("captions") or []
                if not isinstance(groups, list) or not groups:
                    raise ValueError("LLM returned no captions")
                return _captions_from_ids(groups, words, offset=0)
            except (json.JSONDecodeError, ValidationError, ValueError, KeyError) as exc:
                last_error = exc
                prompt += f"\n\nFix this error and return JSON only: {last_error}"
        raise RuntimeError(f"Caption chunk failed: {format_validation_error(last_error)}")

    async def generate(
        self,
        transcript: Transcript,
        video_duration: float,
        job_id: str,
    ) -> CaptionTimeline:
        _ = job_id
        source_words = _flatten_words(transcript)
        if not source_words:
            raise RuntimeError("No words in transcript for caption generation")

        all_captions: list[dict[str, Any]] = []
        for offset, chunk in _chunk_words(source_words, chunk_seconds=18.0):
            # Rebuild captions against the global word list after local grouping.
            local = await self._generate_chunk(chunk, offset, video_duration)
            for cap in local:
                # Map local words back by matching timestamps already on chunk words.
                all_captions.append(cap)

        data = {"version": "1.0", "style": "dynamic_social", "captions": all_captions}
        return validate_caption_timeline(data, video_duration)

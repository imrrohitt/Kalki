from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import ValidationError

from app.asr import known_terms_in, stabilize_copy
from app.captions.heuristic import heuristic_caption_timeline
from app.captions.models import CaptionTimeline
from app.captions.validation import format_validation_error, validate_caption_timeline
from app.config import settings
from app.transcription.models import Transcript, Word


logger = logging.getLogger(__name__)

APP_NAME = "kalki_caption_app"
USER_ID = "kalki_pipeline"

CAPTION_INSTRUCTION = """You are CaptionAgent, an autonomous Reels caption editor.

Complete the task yourself. Do not ask the user questions.

The word list is ASR. It may be misspelled or the speaker may have misspoken. You do NOT put raw model tokens on screen.

Workflow:
1. Read numbered ASR words (timing only).
2. Group them into short captions by meaning (usually 2-4 words).
3. WRITE the on-screen `text` yourself: correct English, correct technical spelling.
4. Call submit_caption_groups once with JSON.
5. If ok=false, fix and submit again.

Display text rules:
- `text` is the published caption, not a join of ASR tokens.
- Fix slips: RAKA→RAG, fine tunning→Fine-tuning, lora→LoRA, destillation→distillation.
- Keep the same meaning as those timed words. Do not invent extra claims.
- Title Case. Max 2 lines, use \\n. Each line ~12–20 characters.
- No emojis, hashtags. ? is OK.
- Keep phrases together: Fine-tuning, RAG, AI interviews, domain data, LLM.
- emphasis_id: 0 or 1 word id per caption (payload words that mean RAG, LLM, AI, Fine-tuning, DATA).
- ids must be consecutive, cover every word in order, no duplicates, no skips.

submit_caption_groups JSON:
{
  "captions": [
    {
      "ids": [0, 1, 2],
      "text": "If You Are\\nGiving",
      "emphasis_id": 2,
      "position": "bottom_center"
    }
  ]
}
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
            token = stabilize_copy(w.word.strip())
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
    offset: int = 0,
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
            emphasis_local = int(emphasis_id) + offset if emphasis_id is not None else None
        except (TypeError, ValueError):
            emphasis_local = None
        text = stabilize_copy(str(group.get("text") or " ".join(w.word for w in group_words)))
        if not text:
            text = stabilize_copy(" ".join(w.word for w in group_words))
        captions.append(
            {
                "start": group_words[0].start,
                "end": max(group_words[-1].end, group_words[0].start + 0.25),
                "text": text,
                "position": group.get("position") or "bottom_center",
                "animation": "pop",
                "words": [
                    {
                        "text": stabilize_copy(w.word) or w.word,
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
                "text": stabilize_copy(" ".join(w.word for w in leftover)),
                "position": "bottom_center",
                "animation": "pop",
                "words": [
                    {
                        "text": stabilize_copy(w.word) or w.word,
                        "start": w.start,
                        "end": w.end,
                        "emphasis": False,
                    }
                    for w in leftover
                ],
            }
        )
    return captions


def _parse_caption_groups(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, str):
        payload = _extract_json(payload)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        groups = payload.get("captions") or []
        if isinstance(groups, list):
            return groups
    raise ValueError("expected captions array or {captions: [...]}")


def _is_deepseek_model(model: str) -> bool:
    return "deepseek" in (model or "").lower()


def _lite_llm_model(model: str) -> str:
    # openai/deepseek-* uses the OpenAI transformer, which drops `thinking`.
    # deepseek/* keeps the DeepSeek transformer so thinking can be disabled.
    if _is_deepseek_model(model) and model.lower().startswith("openai/"):
        return "deepseek/" + model.split("/", 1)[1]
    return model


def _deepseek_no_think_body() -> dict[str, Any]:
    return {"thinking": {"type": "disabled"}}


class CaptionAgentService:
    def __init__(self) -> None:
        self.api_key = settings.llm_api_key.replace("Bearer ", "").strip()
        self.api_base = (settings.llm_base_url or "").rstrip("/") or None
        self.model = settings.llm_model
        if self.api_key:
            os.environ.setdefault("DEEPSEEK_API_KEY", self.api_key)
            os.environ.setdefault("OPENAI_API_KEY", self.api_key)
        lite_kwargs: dict[str, Any] = {
            "model": _lite_llm_model(self.model),
            "api_key": self.api_key or None,
            "api_base": self.api_base,
            "timeout": 90,
            "drop_params": not _is_deepseek_model(self.model),
        }
        if _is_deepseek_model(self.model):
            # V4 thinks by default. Never send reasoning_effort — that turns thinking back on.
            lite_kwargs["thinking"] = {"type": "disabled"}
            lite_kwargs["extra_body"] = _deepseek_no_think_body()
        self._lite_llm = LiteLlm(**lite_kwargs)

    def _build_agent(self, tools: list[Any]) -> LlmAgent:
        http_options = None
        if _is_deepseek_model(self.model):
            http_options = types.HttpOptions(
                extra_body=_deepseek_no_think_body(),
                timeout=90_000,
            )
        return LlmAgent(
            name="caption_agent",
            model=self._lite_llm,
            description="Groups transcript words into short vertical-video captions.",
            instruction=CAPTION_INSTRUCTION,
            tools=tools,
            generate_content_config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=8000,
                http_options=http_options,
            ),
        )

    async def _run_agent(
        self,
        agent: LlmAgent,
        prompt: str,
        session_id: str,
    ) -> str:
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
        )
        runner = Runner(
            agent=agent,
            app_name=APP_NAME,
            session_service=session_service,
        )
        content = types.Content(role="user", parts=[types.Part(text=prompt)])
        texts: list[str] = []
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=content,
        ):
            if not event.is_final_response() or not event.content or not event.content.parts:
                continue
            for part in event.content.parts:
                if part.text and part.text.strip():
                    texts.append(part.text.strip())
        return "\n".join(texts)

    async def _generate_chunk(
        self,
        words: list[Word],
        offset: int,
        video_duration: float,
        job_id: str,
        chunk_index: int,
    ) -> list[dict[str, Any]]:
        state: dict[str, Any] = {"accepted": False, "captions": None}
        payload = {
            "offset": offset,
            "duration": round(video_duration, 2),
            "spell_as": known_terms_in(" ".join(w.word for w in words))
            or ["RAG", "Fine-tuning", "LLM"],
            "words": [
                {
                    "id": i,
                    "asr": w.word,
                    "start": round(w.start, 2),
                    "end": round(w.end, 2),
                }
                for i, w in enumerate(words)
            ],
        }

        def submit_caption_groups(
            captions_json: str,
            tool_context: ToolContext,
        ) -> dict[str, Any]:
            """Submit finished caption groups as a JSON string.

            Expected shape:
            {"captions": [{"ids": [0, 1], "text": "HELLO\\nWORLD", "emphasis_id": 0, "position": "bottom_center"}]}

            ids are 0-based inside THIS word list, consecutive, covering every word.
            """
            try:
                groups = _parse_caption_groups(captions_json)
                if not groups:
                    raise ValueError("no captions in payload")
                captions = _captions_from_ids(groups, words, offset=0)
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                state["accepted"] = False
                return {"ok": False, "error": format_validation_error(exc)}
            state["captions"] = captions
            state["accepted"] = True
            # Stop ADK from making a second "DONE" LLM call after the tool.
            tool_context.actions.skip_summarization = True
            tool_context.get_invocation_context().end_invocation = True
            return {"ok": True, "caption_count": len(captions)}

        agent = self._build_agent([submit_caption_groups])
        prompt = (
            "Write captions for this ASR chunk. Timing from ids. "
            "On-screen text is YOUR rewrite — correct spelling, not raw ASR. "
            "Cover every word. Call submit_caption_groups once.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        final_text = await self._run_agent(
            agent,
            prompt,
            session_id=f"{job_id}-c{chunk_index}",
        )

        if state["accepted"] and state["captions"]:
            return state["captions"]

        if final_text.strip():
            try:
                groups = _parse_caption_groups(final_text)
                if groups:
                    return _captions_from_ids(groups, words, offset=0)
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
                pass

        raise RuntimeError(
            "Caption agent did not submit valid groups"
            + (f": {final_text[:300]}" if final_text else "")
        )

    async def generate(
        self,
        transcript: Transcript,
        video_duration: float,
        job_id: str,
    ) -> CaptionTimeline:
        if settings.caption_heuristic_only:
            timeline = heuristic_caption_timeline(transcript, video_duration)
            logger.info(
                "[%s] captions: heuristic only (%s groups)",
                job_id[:8],
                len(timeline.captions),
            )
            return timeline

        source_words = _flatten_words(transcript)
        if not source_words:
            raise RuntimeError("No words in transcript for caption generation")

        chunk_seconds = settings.caption_chunk_seconds
        if chunk_seconds <= 0:
            chunks = [(0, source_words)]
        else:
            chunks = _chunk_words(source_words, chunk_seconds=chunk_seconds)
        logger.info(
            "[%s] captions LLM: %s words in %s chunk(s)",
            job_id[:8],
            len(source_words),
            len(chunks),
        )
        sem = asyncio.Semaphore(3)

        async def run_one(
            index: int, offset: int, chunk: list[Word]
        ) -> tuple[int, list[dict[str, Any]]]:
            async with sem:
                t0 = time.perf_counter()
                captions = await self._generate_chunk(
                    words=chunk,
                    offset=offset,
                    video_duration=video_duration,
                    job_id=job_id,
                    chunk_index=index,
                )
                logger.info(
                    "[%s] caption chunk %s/%s: %s words in %.1fs",
                    job_id[:8],
                    index + 1,
                    len(chunks),
                    len(chunk),
                    time.perf_counter() - t0,
                )
                return index, captions

        try:
            parts = await asyncio.gather(
                *[run_one(i, offset, chunk) for i, (offset, chunk) in enumerate(chunks)]
            )
            all_captions: list[dict[str, Any]] = []
            for _, captions in sorted(parts, key=lambda item: item[0]):
                all_captions.extend(captions)

            data = {"version": "1.0", "style": "dynamic_social", "captions": all_captions}
            return validate_caption_timeline(data, video_duration)
        except Exception as exc:
            logger.warning(
                "[%s] caption LLM failed (%s); using heuristic grouping",
                job_id[:8],
                exc,
            )
            return heuristic_caption_timeline(transcript, video_duration)

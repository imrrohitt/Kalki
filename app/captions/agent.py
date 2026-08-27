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
from app.captions.heuristic import (
    explode_caption_timeline,
    heuristic_caption_timeline,
    packs_for_words,
)
from app.captions.models import CaptionTimeline
from app.captions.validation import format_validation_error, validate_caption_timeline
from app.config import settings
from app.transcription.models import Transcript, Word


logger = logging.getLogger(__name__)

APP_NAME = "kalki_caption_app"
USER_ID = "kalki_pipeline"

CAPTION_INSTRUCTION = """You are CaptionAgent, an autonomous Reels caption editor.

Complete the task yourself. Do not ask the user questions.

The word list is ASR. It may be English, Hindi, Hinglish (Hindi in Latin letters),
or mixed. Tokens may be misspelled. You do NOT put raw ASR on screen.

Language law (non-negotiable):
- Every on-screen `text` value MUST be fluent English. Title Case.
- If ASR is Hindi or Hinglish, TRANSLATE the meaning into natural English captions.
  "aapko data secure karna hai" → "You Need To\\nSecure Data"
  "pehla step yeh hai" → "The First Step"
- Never print Devanagari. Never print Hinglish filler (hai, hain, karna, aapko, toh, yeh, woh, ka, ki, ke, mein, se).
- Keep technical terms in English (RAG, LLM, GDPR, Fine-tuning, LoRA).
- Reuse the speaker's intent and examples. Do not invent claims they did not make.

Workflow:
1. Read numbered ASR words (timing only — they may be any language).
2. Group them into short captions by meaning (usually 2-4 English words).
3. WRITE the on-screen `text` yourself in English.
4. Call submit_caption_groups once with JSON.
5. If ok=false, fix and submit again.

Duration law (non-negotiable):
- Each caption is 2–4 English words and typically 1–3 seconds on screen.
- NEVER dump remaining speech into one caption. If many words remain, emit many short groups.
- A caption must not cover more than ~6 spoken words or ~4 seconds.
- Cover the whole chunk with a stream of short groups so new lines keep appearing.

Display text rules:
- `text` is the published English caption, not a join of ASR tokens.
- Fix slips: RAKA→RAG, fine tunning→Fine-tuning, lora→LoRA, destillation→distillation.
- Keep the same meaning as those timed words.
- Title Case. Max 2 lines, use \\n. Each line ~12–20 characters.
- No emojis, hashtags. ? is OK.
- Keep phrases together: Fine-tuning, RAG, AI interviews, domain data, LLM.
- emphasis_id: 0 or 1 word id per caption (the payload word that names RAG, LLM, AI, Fine-tuning, DATA).
- ids must be consecutive, cover every ASR word in order, no duplicates, no skips.

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


def _has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in text)


def _display_tokens(text: str) -> list[str]:
    cleaned = stabilize_copy(text)
    return [tok for tok in cleaned.replace("\n", " ").split() if tok]


def _timed_english_words(
    text: str,
    group_words: list[Word],
    *,
    hot_asr: str | None,
) -> list[dict[str, Any]]:
    """Map English caption tokens onto the spoken time span (ASR may be Hindi)."""
    tokens = _display_tokens(text)
    if not tokens:
        tokens = [stabilize_copy(w.word) or w.word for w in group_words]
        tokens = [t for t in tokens if t]
    if not tokens:
        return []
    t0 = float(group_words[0].start)
    t1 = max(float(group_words[-1].end), t0 + 0.25)
    n = len(tokens)
    span = max(t1 - t0, 0.08 * n)
    hot = (hot_asr or "").lower().strip(".,!?")
    out: list[dict[str, Any]] = []
    for i, tok in enumerate(tokens):
        start = t0 + span * i / n
        end = t0 + span * (i + 1) / n
        needle = tok.lower().strip(".,!?")
        emphasis = bool(hot) and (
            needle == hot or hot in needle or needle in hot
        )
        out.append(
            {
                "text": tok,
                "start": start,
                "end": max(end, start + 0.05),
                "emphasis": emphasis,
            }
        )
    if hot and not any(w["emphasis"] for w in out):
        # Translated line: punch the last content word if ASR hot-term did not match.
        out[-1]["emphasis"] = True
    return out


def _english_covers_span(text: str, n_asr: int) -> bool:
    tokens = _display_tokens(text)
    if n_asr <= 6:
        return True
    return len(tokens) >= max(2, n_asr // 4)


def _allocate_token_counts(n_tokens: int, weights: list[int]) -> list[int]:
    total = sum(weights) or 1
    raw = [n_tokens * w / total for w in weights]
    counts = [int(x) for x in raw]
    remainder = n_tokens - sum(counts)
    order = sorted(range(len(weights)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in range(max(0, remainder)):
        counts[order[i % len(weights)]] += 1
    return counts


def _caption_dict(
    pack: list[Word],
    text: str,
    *,
    position: str,
    hot_asr: str | None,
) -> dict[str, Any]:
    return {
        "start": pack[0].start,
        "end": max(pack[-1].end, pack[0].start + 0.25),
        "text": text,
        "position": position or "bottom_center",
        "animation": "pop",
        "words": _timed_english_words(text, pack, hot_asr=hot_asr),
    }


def _hot_asr_in_pack(pack: list[Word], hot_asr: str | None) -> str | None:
    if not hot_asr:
        return None
    needle = hot_asr.lower().strip(".,!?")
    for src in pack:
        token = src.word.lower().strip(".,!?")
        if token == needle or needle in token or token in needle:
            return hot_asr
    return None


def _captions_from_pack_splits(
    packs: list[list[Word]],
    text: str,
    *,
    position: str,
    hot_asr: str | None,
) -> list[dict[str, Any]]:
    if len(packs) == 1:
        return [
            _caption_dict(packs[0], text, position=position, hot_asr=hot_asr)
        ]
    tokens = _display_tokens(text) if _english_covers_span(text, sum(len(p) for p in packs)) else []
    captions: list[dict[str, Any]] = []
    if tokens:
        counts = _allocate_token_counts(len(tokens), [len(p) for p in packs])
        idx = 0
        for pack, count in zip(packs, counts):
            piece = tokens[idx : idx + count]
            idx += count
            pack_text = (
                " ".join(piece)
                if piece
                else stabilize_copy(" ".join(w.word for w in pack))
            )
            if not pack_text or _has_devanagari(pack_text):
                continue
            captions.append(
                _caption_dict(
                    pack,
                    pack_text,
                    position=position,
                    hot_asr=_hot_asr_in_pack(pack, hot_asr),
                )
            )
        return captions
    for pack in packs:
        pack_text = stabilize_copy(" ".join(w.word for w in pack))
        if not pack_text or _has_devanagari(pack_text):
            continue
        captions.append(
            _caption_dict(
                pack,
                pack_text,
                position=position,
                hot_asr=_hot_asr_in_pack(pack, hot_asr),
            )
        )
    return captions


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
        hot_asr = None
        if emphasis_local is not None:
            for local_i, src in enumerate(group_words):
                if used[local_i] == emphasis_local:
                    hot_asr = stabilize_copy(src.word) or src.word
                    break
        position = group.get("position") or "bottom_center"
        captions.extend(
            _captions_from_pack_splits(
                packs_for_words(group_words),
                text,
                position=position,
                hot_asr=hot_asr,
            )
        )
    if cursor < len(source_words):
        leftover = source_words[cursor:]
        leftover_text = stabilize_copy(" ".join(w.word for w in leftover))
        # Do not put leftover Hindi/Devanagari on screen; English groups already cover meaning.
        if leftover_text and not _has_devanagari(leftover_text):
            captions.extend(
                _captions_from_pack_splits(
                    packs_for_words(leftover),
                    leftover_text,
                    position="bottom_center",
                    hot_asr=None,
                )
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
            description="Turns ASR (any language) into short English vertical-video captions.",
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
            "Write English captions for this ASR chunk. Timing from ids. "
            "ASR may be Hindi, Hinglish, or English — on-screen `text` is always English. "
            "Translate meaning; keep technical terms (RAG, LLM, GDPR). "
            "Never print Devanagari or Hinglish filler. Cover every id. "
            "Each caption 2-4 words, 1-3 seconds. Never one caption for the rest of the talk. "
            "Call submit_caption_groups once.\n\n"
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
            timeline = validate_caption_timeline(data, video_duration)
            return explode_caption_timeline(timeline)
        except Exception as exc:
            logger.warning(
                "[%s] caption LLM failed (%s); using heuristic grouping",
                job_id[:8],
                exc,
            )
            return heuristic_caption_timeline(transcript, video_duration)

from __future__ import annotations

import json
import logging
import os
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import ValidationError

from app.captions.agent import (
    _deepseek_no_think_body,
    _extract_json,
    _is_deepseek_model,
    _lite_llm_model,
)
from app.config import settings
from app.editorial.transcript.repair import apply_segment_repairs, heuristic_repair
from app.transcription.models import Transcript


logger = logging.getLogger(__name__)

APP_NAME = "kalki_transcript_app"
USER_ID = "kalki_pipeline"

REPAIR_INSTRUCTION = """You are the transcript editor for an educational Instagram Reel.

Whisper ASR is noisy. Your job is to recover what the speaker actually said.

Rules:
- Keep the same meaning, order, and number of sentences.
- Fix spelling, grammar, and technical terms (RAKA→RAG, fine tunning→fine-tuning).
- Do not invent facts, names, numbers, or claims that are not in the ASR.
- Do not drop sentences. Do not merge two segments into one.
- Return one cleaned string per segment_id.

Call submit_transcript_repair ONCE with JSON:
{"segments":[{"id":0,"text":"Why do we use RAG instead of fine-tuning?"}]}
"""


def _repairs_from_payload(raw: Any, segment_count: int) -> dict[int, str]:
    payload = _extract_json(raw) if isinstance(raw, str) else raw
    items = payload.get("segments") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("expected segments array")
    repairs: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= segment_count:
            continue
        text = str(item.get("text") or "").strip()
        if text:
            repairs[idx] = text
    if not repairs:
        raise ValueError("no repaired segments")
    return repairs


class TranscriptRepairAgent:
    def __init__(self) -> None:
        self.api_key = settings.llm_api_key.replace("Bearer ", "").strip()
        self.api_base = (settings.llm_base_url or "").rstrip("/") or None
        self.model = settings.llm_model
        self.use_llm = bool(settings.transcript_repair_llm_enabled and self.api_key)
        self._lite_llm: LiteLlm | None = None
        if self.use_llm:
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
                lite_kwargs["thinking"] = {"type": "disabled"}
                lite_kwargs["extra_body"] = _deepseek_no_think_body()
            self._lite_llm = LiteLlm(**lite_kwargs)

    def _build_agent(self, tools: list[Any]) -> LlmAgent:
        assert self._lite_llm is not None
        http_options = None
        if _is_deepseek_model(self.model):
            http_options = types.HttpOptions(
                extra_body=_deepseek_no_think_body(),
                timeout=90_000,
            )
        return LlmAgent(
            name="transcript_repair_agent",
            model=self._lite_llm,
            description="Cleans Whisper ASR into readable spoken English.",
            instruction=REPAIR_INSTRUCTION,
            tools=tools,
            generate_content_config=types.GenerateContentConfig(
                temperature=0.15,
                max_output_tokens=8000,
                http_options=http_options,
            ),
        )

    async def _run_agent(self, agent: LlmAgent, prompt: str, session_id: str) -> str:
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
        )
        runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
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

    async def _repair_chunk(
        self,
        chunk: list[dict[str, Any]],
        segment_count: int,
        job_id: str,
        chunk_index: int,
    ) -> dict[int, str]:
        state: dict[str, Any] = {"accepted": False, "repairs": None}

        def submit_transcript_repair(
            repair_json: str, tool_context: ToolContext
        ) -> dict[str, Any]:
            """Submit cleaned transcript segments as JSON."""
            try:
                repairs = _repairs_from_payload(repair_json, segment_count)
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                state["accepted"] = False
                return {"ok": False, "error": str(exc)}
            state["repairs"] = repairs
            state["accepted"] = True
            tool_context.actions.skip_summarization = True
            tool_context.get_invocation_context().end_invocation = True
            return {"ok": True, "segment_count": len(repairs)}

        agent = self._build_agent([submit_transcript_repair])
        prompt = (
            "Clean these ASR segments. Keep ids. Do not invent facts. "
            "Call submit_transcript_repair once.\n\n"
            + json.dumps({"segments": chunk}, ensure_ascii=False)
        )
        final_text = await self._run_agent(
            agent, prompt, session_id=f"{job_id}-tr{chunk_index}"
        )
        if state["accepted"] and state["repairs"]:
            return state["repairs"]
        if final_text.strip():
            return _repairs_from_payload(final_text, segment_count)
        raise RuntimeError("Transcript repair agent did not submit valid segments")

    async def repair(self, transcript: Transcript, *, job_id: str = "repair") -> Transcript:
        fallback = heuristic_repair(transcript)
        if not self.use_llm or not transcript.segments:
            logger.info(
                "[%s] transcript repair: heuristic (%s segments)",
                job_id[:8],
                len(fallback.segments),
            )
            return fallback
        payload = [
            {
                "id": i,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text or " ".join(w.word for w in seg.words),
            }
            for i, seg in enumerate(transcript.segments)
        ]
        chunk_size = 12
        chunks = [
            payload[i : i + chunk_size] for i in range(0, len(payload), chunk_size)
        ]
        try:
            merged: dict[int, str] = {}
            logger.info(
                "[%s] transcript repair LLM: %s segments in %s chunk(s)",
                job_id[:8],
                len(payload),
                len(chunks),
            )
            for i, chunk in enumerate(chunks):
                try:
                    merged.update(
                        await self._repair_chunk(
                            chunk, len(transcript.segments), job_id, i
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "[%s] transcript repair chunk %s failed (%s); skipping",
                        job_id[:8],
                        i + 1,
                        exc,
                    )
            if not merged:
                return fallback
            repaired = apply_segment_repairs(transcript, merged)
            logger.info("[%s] transcript repair: cleaned %s segments", job_id[:8], len(merged))
            return repaired
        except Exception as exc:
            logger.warning(
                "[%s] transcript repair LLM failed (%s); using heuristic",
                job_id[:8],
                exc,
            )
            return fallback

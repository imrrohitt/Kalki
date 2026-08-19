from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import ValidationError

from app.captions.agent import _deepseek_no_think_body, _is_deepseek_model, _lite_llm_model
from app.config import settings
from app.editorial.context import build_sentence_windows
from app.editorial.framing import default_visual
from app.editorial.heuristic import detect_story_patterns, heuristic_analyze
from app.editorial.models import (
    EditorialAnalysis,
    EditorialSentence,
    LlmAnalysisPayload,
    SentenceWindow,
    VisualContext,
)
from app.transcription.models import Transcript


logger = logging.getLogger(__name__)

APP_NAME = "kalki_editorial_app"
USER_ID = "kalki_pipeline"

EDITORIAL_INSTRUCTION = """You are EditorialAnalyzer, the rhetorical intelligence layer of a talking-head video editor.

Complete the task yourself. Do not ask questions.

You are NOT detecting important keywords. You are answering:
"What is the speaker doing rhetorically at this moment — and how should the camera move?"

Each item already includes previous / current / next sentences. Use that window.
A line that looks ordinary in isolation can be a reversal or reveal in context.

You also own CAMERA TIMING. The renderer eases your numbers; it must never snap.
Do not mention FFmpeg. Do not output crop math. Intensity is mapped to zoom amount
using the subject bounding box (max_safe_scale). Never ask for more zoom than that.

Roles (pick one per sentence):
emphasis, surprise, contrast, strong_opinion, key_insight, important_number,
question, answer, story_climax, reveal, warning, hook, cta, emotional_peak,
humor, transition, generic, assumption, contradiction.

Signals are 0-1 scores: emphasis, surprise, contrast, emotion, humor, question,
reveal, warning, cta.

Story patterns you MUST look for across sentences — they change CAMERA, not just labels:
- setup → reversal → reveal: setup stays WIDE (apply=false). Reversal is a modest,
  slower push. Reveal is the strongest move, delayed so it lands on the payoff.
- question → answer: question is a gentle, long ease-in. Answer is stronger.
- hook → payoff: hook can push in; payoff is stronger and a bit slower.

Framing / how much zoom (from the prompt's subject_bbox + max_safe_scale):
- intensity 0 = no zoom. intensity 1 = zoom to max_safe_scale. Never beyond.
- If the bbox is already large (w or h > 0.72) the face is close-up: apply=false
  or intensity ≤ 0.22. Punching in further clips the head.
- Tight talking-head default still needs modest intensity (0.35–0.75), not 1.0
  on every beat.

Timing (milliseconds). This is the feel of the edit — never a hard cut:
- delay_ms: wait after the line starts (0–400). Reveals often 150–250.
- ease_in_ms: 450–900. NEVER below 450. Instant zoom is forbidden.
- hold_ms: 350–900 at peak scale.
- ease_out_ms: 380–700 back to wide.
- Fit timing to duration_ms of the sentence. Shrink hold first. If the line is
  too short to ease (≥450ms in), set apply=false rather than snapping.

When NOT to zoom:
- generic / CTA / "today we will talk about" / assumption (setup) → apply=false.
- Do not mark everything as emphasis. Most sentences should be apply=false.
- Prefer the role that matches the speaker's move, not a word in the sentence.
- story_position: setup | development | climax | resolution | none
- After a valid submit, stop.

Call submit_editorial_analysis once with JSON:
{
  "sentences": [
    {
      "sentence_id": 17,
      "editorial_role": "reveal",
      "signals": {
        "emphasis": 0.4, "surprise": 0.2, "contrast": 0.94, "emotion": 0.1,
        "humor": 0.0, "question": 0.0, "reveal": 0.86, "warning": 0.0, "cta": 0.0
      },
      "visual_interest": 0.91,
      "story_position": "climax",
      "confidence": 0.88,
      "zoom": {
        "apply": true,
        "intensity": 0.72,
        "delay_ms": 180,
        "ease_in_ms": 560,
        "hold_ms": 700,
        "ease_out_ms": 480
      }
    }
  ],
  "story_patterns": [
    {"pattern": "setup → reversal → reveal", "sentence_ids": [15, 16, 17], "confidence": 0.9}
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


def _chunk_windows(
    windows: list[SentenceWindow],
    size: int = 12,
    overlap: int = 2,
) -> list[list[SentenceWindow]]:
    if size <= 0 or len(windows) <= size:
        return [windows]
    chunks: list[list[SentenceWindow]] = []
    start = 0
    while start < len(windows):
        end = min(len(windows), start + size)
        chunks.append(windows[start:end])
        if end >= len(windows):
            break
        start = max(end - overlap, start + 1)
    return chunks


def merge_annotations(
    windows: list[SentenceWindow],
    payload: LlmAnalysisPayload,
) -> EditorialAnalysis:
    by_id = {item.sentence_id: item for item in payload.sentences}
    fallback = heuristic_analyze(windows)
    fallback_by_id = {s.sentence_id: s for s in fallback.sentences}
    sentences: list[EditorialSentence] = []
    for window in windows:
        annotated = by_id.get(window.sentence_id)
        if annotated is None:
            sentences.append(fallback_by_id[window.sentence_id])
            continue
        sentences.append(
            EditorialSentence(
                sentence_id=window.sentence_id,
                start=window.start,
                end=window.end,
                text=window.text,
                word_ids=window.word_ids,
                context=window.context,
                signals=annotated.signals,
                editorial_role=annotated.editorial_role,
                visual_interest=annotated.visual_interest,
                story_position=annotated.story_position,
                confidence=annotated.confidence,
                prosody=window.prosody,
                zoom=(
                    annotated.zoom
                    if annotated.zoom is not None
                    else fallback_by_id[window.sentence_id].zoom
                ),
            )
        )
    patterns = list(payload.story_patterns)
    if not patterns:
        patterns = detect_story_patterns(sentences)
    return EditorialAnalysis(sentences=sentences, story_patterns=patterns)


class EditorialAnalyzer:
    def __init__(self, use_llm: bool | None = None) -> None:
        self.api_key = settings.llm_api_key.replace("Bearer ", "").strip()
        self.api_base = (settings.llm_base_url or "").rstrip("/") or None
        self.model = settings.llm_model
        enabled = settings.editorial_llm_enabled if use_llm is None else use_llm
        self.use_llm = bool(enabled and self.api_key)
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
            name="editorial_analyzer",
            model=self._lite_llm,
            description="Labels rhetorical intent for each transcript sentence.",
            instruction=EDITORIAL_INSTRUCTION,
            tools=tools,
            generate_content_config=types.GenerateContentConfig(
                temperature=0.2,
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

    async def _analyze_chunk(
        self,
        windows: list[SentenceWindow],
        job_id: str,
        chunk_index: int,
        visual: VisualContext,
    ) -> LlmAnalysisPayload:
        state: dict[str, Any] = {"accepted": False, "payload": None}
        items = [
            {
                "sentence_id": w.sentence_id,
                "start": round(w.start, 2),
                "end": round(w.end, 2),
                "duration_ms": int(max(0.0, w.end - w.start) * 1000),
                "previous": w.context.previous,
                "current": w.context.current,
                "next": w.context.next,
                "pause_before": w.prosody.pause_before,
                "speaking_rate": w.prosody.speaking_rate,
            }
            for w in windows
        ]

        def submit_editorial_analysis(
            analysis_json: str,
            tool_context: ToolContext,
        ) -> dict[str, Any]:
            """Submit rhetorical analysis JSON for this sentence window."""
            try:
                raw = analysis_json if not isinstance(analysis_json, str) else _extract_json(analysis_json)
                payload = LlmAnalysisPayload.model_validate(raw)
                if not payload.sentences:
                    raise ValueError("no sentences in payload")
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                state["accepted"] = False
                return {"ok": False, "error": str(exc)}
            state["payload"] = payload
            state["accepted"] = True
            tool_context.actions.skip_summarization = True
            tool_context.get_invocation_context().end_invocation = True
            return {"ok": True, "sentence_count": len(payload.sentences)}

        agent = self._build_agent([submit_editorial_analysis])
        box = visual.bbox
        prompt = (
            "Analyze rhetorical intent AND camera motion for each sentence. "
            "Use previous/current/next. Intensity maps to zoom amount using "
            f"max_safe_scale={visual.max_safe_scale:.3f}. "
            "Never snap (ease_in_ms >= 450). Most lines apply=false. "
            "Call submit_editorial_analysis once with JSON.\n\n"
            + json.dumps(
                {
                    "framing": {
                        "subject_bbox": {
                            "x": round(box.x, 3),
                            "y": round(box.y, 3),
                            "w": round(box.w, 3),
                            "h": round(box.h, 3),
                        },
                        "max_safe_scale": round(visual.max_safe_scale, 3),
                        "note": (
                            "scale = 1 + intensity * (max_safe_scale - 1). "
                            "Large bbox (close-up) → tiny intensity or apply=false."
                        ),
                    },
                    "sentences": items,
                },
                ensure_ascii=False,
            )
        )
        final_text = await self._run_agent(
            agent,
            prompt,
            session_id=f"{job_id}-e{chunk_index}",
        )
        if state["accepted"] and state["payload"]:
            return state["payload"]
        if final_text.strip():
            try:
                return LlmAnalysisPayload.model_validate(_extract_json(final_text))
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
                pass
        raise RuntimeError(
            "Editorial analyzer did not submit valid analysis"
            + (f": {final_text[:300]}" if final_text else "")
        )

    async def analyze(
        self,
        transcript: Transcript,
        video_duration: float,
        job_id: str,
        visual: VisualContext | None = None,
    ) -> EditorialAnalysis:
        _ = video_duration
        visual = visual or default_visual()
        windows = build_sentence_windows(transcript)
        if not windows:
            logger.info("[%s] editorial: no sentences", job_id[:8])
            return EditorialAnalysis()
        if not self.use_llm:
            analysis = heuristic_analyze(windows)
            logger.info(
                "[%s] editorial: heuristic (LLM off), %s sentences",
                job_id[:8],
                len(analysis.sentences),
            )
            return analysis

        chunks = _chunk_windows(windows)
        logger.info(
            "[%s] editorial LLM: %s sentences in %s chunk(s)",
            job_id[:8],
            len(windows),
            len(chunks),
        )
        try:
            sem = asyncio.Semaphore(3)

            async def run_one(index: int, chunk: list[SentenceWindow]) -> LlmAnalysisPayload:
                async with sem:
                    return await self._analyze_chunk(chunk, job_id, index, visual)

            parts = await asyncio.gather(
                *[run_one(i, chunk) for i, chunk in enumerate(chunks)]
            )
            merged_annotations = []
            merged_patterns = []
            seen_ids: set[int] = set()
            for payload in parts:
                for item in payload.sentences:
                    if item.sentence_id in seen_ids:
                        continue
                    seen_ids.add(item.sentence_id)
                    merged_annotations.append(item)
                merged_patterns.extend(payload.story_patterns)
            combined = LlmAnalysisPayload(
                sentences=merged_annotations,
                story_patterns=merged_patterns,
            )
            return merge_annotations(windows, combined)
        except Exception as exc:
            logger.warning(
                "[%s] editorial LLM failed (%s); using heuristic fallback",
                job_id[:8],
                exc,
            )
            return heuristic_analyze(windows)

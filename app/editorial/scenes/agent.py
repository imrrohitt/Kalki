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
    _is_deepseek_model,
    _lite_llm_model,
)
from app.config import settings
from app.asr import fix_asr_text
from app.editorial.graphics.agent import _beats_from_payload, _chunk, _glossary, _stabilize_beat
from app.editorial.graphics.planner import cover_duration
from app.editorial.models import EditorialAnalysis, GraphicBeat
from app.editorial.scenes.planner import enrich_scene, plan_scenes


logger = logging.getLogger(__name__)

APP_NAME = "kalki_scene_app"
USER_ID = "kalki_pipeline"

SCENE_INSTRUCTION = """You are the scene director for a 9:16 Instagram Reel built from AUDIO ONLY.

There is NO talking head. The entire 1080×1920 frame is a cinematic motion-graphics scene.
Captions sit at the very bottom (y≈1748). Graphics must occupy TOP + MIDDLE + BOTTOM of the frame.
Do not leave the lower half empty. Do not design a thin title card.

You complete the job yourself. Do not ask questions.

Language law (non-negotiable):
- Every headline, kicker, bullet, chip, and node label MUST be English.
- ASR may be Hindi, Hinglish, or mixed. Translate the idea into English scene copy.
- Never print Devanagari or Hinglish filler. Keep technical terms in English (RAG, LLM, GDPR).

## What a SCENE is
A scene is not a caption. It is a diagram of the idea being spoken in that window:
- TOP: kicker + a short headline (what this moment is about).
- MIDDLE: 3–5 BOXED nodes with short labels. Dotted connectors between them.
  This is the system / flow / comparison the speaker is explaining.
- BOTTOM: 3–5 supporting lines (not two). Each line is a consequence, example, or rule
  that belongs to THIS spoken window.

If the speaker describes a process, draw the process.
If they compare two things, draw two boxed sides plus why each exists.
If they hit a number, the number is the hero AND the boxes explain what it is attached to.
If they ask a question, the boxes are the answer-in-waiting.

## Hard visual rules
- Prefer kind = diagram, process, vs_split, or stat. Use topic/bullets only when the line has no structure.
- nodes: 3–5 items. label ≤ 18 characters. Optional sub ≤ 36 characters (the clause inside the box).
- chips: same as node labels if you also send chips.
- bullets: 3–5 lines, each ≤ 44 characters. NEVER stop at two.
- vs_split: left/right 1–3 words AND 3–5 bullets that explain the tradeoff AND 2 boxed nodes.
- Title: 3–8 words, Title Case, ≤ 44 characters. One thought, not a transcript paste.
- kicker: HOOK | INSIGHT | COMPARE | STEPS | NUMBER | FLOW | CTA
- Cover the whole duration. One scene at a time, usually 5–9 seconds.
- Correct ASR slips (RAKA→RAG, fine tunning→Fine-tuning). Never invent numbers or claims.
- You MAY rephrase, structure, and add connective labels that are implied by the speech
  (Query → Retrieve → Generate is allowed if they describe that flow).
- Do not write hashtags, emojis, or ALL-CAPS titles.

## Kind guide
- stat: a spoken number is the point. Still include nodes + 3–5 bullets under it.
- vs_split: two named sides. Boxes + dotted connector through VS + bullets.
- process: ordered steps. 3–5 nodes in a vertical flow.
- diagram: a system, architecture, or “how it works”. 3–5 named boxes.
- quote: a spoken question as the headline, boxes as the stakes.
- topic / bullets: headline plus a real diagram of the idea, not two lonely lines.

## Motion
- stat: scale_in
- everything else: slide_up

Call submit_scenes ONCE with JSON like:
{
  "graphics": [
    {
      "start": 0.0,
      "end": 6.4,
      "kind": "diagram",
      "kicker": "HOOK",
      "title": "Why RAG Beats Fine-Tuning",
      "subtitle": "update docs, not weights",
      "motion": "slide_up",
      "nodes": [
        {"label": "Question", "sub": "user asks the model"},
        {"label": "Retrieve", "sub": "pull fresh chunks"},
        {"label": "Generate", "sub": "answer from those docs"}
      ],
      "chips": ["Question", "Retrieve", "Generate"],
      "bullets": [
        "Re-index when files change",
        "No GPU retrain on every update",
        "Fine-tune only if knowledge is stable",
        "Same LLM, two update paths"
      ]
    }
  ]
}
"""


class SceneDirectorAgent:
    def __init__(self) -> None:
        self.api_key = settings.llm_api_key.replace("Bearer ", "").strip()
        self.api_base = (settings.llm_base_url or "").rstrip("/") or None
        self.model = settings.llm_model
        self.use_llm = bool(settings.scenes_llm_enabled and self.api_key)
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
            name="scene_director_agent",
            model=self._lite_llm,
            description="Directs full-canvas motion-graphic scenes for an audio reel.",
            instruction=SCENE_INSTRUCTION,
            tools=tools,
            generate_content_config=types.GenerateContentConfig(
                temperature=0.45,
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

    async def _generate_chunk(
        self,
        sentences: list[dict[str, Any]],
        duration: float,
        job_id: str,
        chunk_index: int,
        glossary: list[str] | None = None,
    ) -> list[GraphicBeat]:
        state: dict[str, Any] = {"accepted": False, "beats": None}

        def submit_scenes(scenes_json: str, tool_context: ToolContext) -> dict[str, Any]:
            """Submit full-canvas reel scenes as JSON."""
            try:
                beats = _beats_from_payload(scenes_json, duration, canvas="full")
                if not beats:
                    raise ValueError("no scenes in payload")
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                state["accepted"] = False
                return {"ok": False, "error": str(exc)}
            state["beats"] = beats
            state["accepted"] = True
            tool_context.actions.skip_summarization = True
            tool_context.get_invocation_context().end_invocation = True
            return {"ok": True, "scene_count": len(beats)}

        agent = self._build_agent([submit_scenes])
        prompt = (
            "Build FULL-FRAME scenes for this window of an audio-only Reel. "
            "ASR may be Hindi or Hinglish — every headline, node, and line is English. "
            "Every scene must fill 1080x1920: headline, 3-5 boxed nodes with "
            "dotted connectors, and 3-5 supporting lines. Never two bullets only. "
            "First window opens with a HOOK. Structure what they said — do not paste ASR. "
            f"Duration={duration:.2f}s. Call submit_scenes once.\n\n"
            + json.dumps(
                {
                    "duration": round(duration, 2),
                    "canvas": "1080x1920 full frame",
                    "fill": ["top headline", "middle boxed flow", "bottom 3-5 lines"],
                    "spell_as": glossary or ["RAG", "Fine-tuning", "LLM"],
                    "sentences": sentences,
                },
                ensure_ascii=False,
            )
        )
        final_text = await self._run_agent(
            agent, prompt, session_id=f"{job_id}-sc{chunk_index}"
        )
        if state["accepted"] and state["beats"]:
            return state["beats"]
        if final_text.strip():
            try:
                return _beats_from_payload(final_text, duration, canvas="full")
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
                pass
        raise RuntimeError("Scene director did not submit valid scenes")

    async def plan(
        self,
        analysis: EditorialAnalysis,
        *,
        video_duration: float,
        job_id: str = "scenes",
    ) -> list[GraphicBeat]:
        fallback = [
            _stabilize_beat(b, canvas="full")
            for b in plan_scenes(analysis, video_duration=video_duration)
        ]
        if not self.use_llm:
            logger.info("[%s] scenes: heuristic %s cards", job_id[:8], len(fallback))
            return fallback
        payload = [
            {
                "sentence_id": s.sentence_id,
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "role": s.editorial_role,
                "text": fix_asr_text(s.text),
            }
            for s in analysis.sentences
        ]
        glossary = _glossary(payload)
        try:
            chunks = _chunk(payload, size=8)
            logger.info(
                "[%s] scenes LLM: %s sentences in %s chunk(s)",
                job_id[:8],
                len(payload),
                len(chunks),
            )
            merged: list[GraphicBeat] = []
            for i, chunk in enumerate(chunks):
                try:
                    merged.extend(
                        await self._generate_chunk(
                            chunk, video_duration, job_id, i, glossary=glossary
                        )
                    )
                except Exception as chunk_exc:
                    logger.warning(
                        "[%s] scenes chunk %s failed (%s); retrying once",
                        job_id[:8],
                        i + 1,
                        chunk_exc,
                    )
                    try:
                        merged.extend(
                            await self._generate_chunk(
                                chunk, video_duration, job_id, i + 100, glossary=glossary
                            )
                        )
                    except Exception as retry_exc:
                        logger.warning(
                            "[%s] scenes chunk %s failed twice (%s); skipping",
                            job_id[:8],
                            i + 1,
                            retry_exc,
                        )
            if not merged:
                logger.warning("[%s] scenes LLM returned nothing; using heuristic", job_id[:8])
                return fallback
            merged.sort(key=lambda b: b.start)
            cleaned: list[GraphicBeat] = []
            last = -1.0
            for beat in merged:
                start = max(beat.start, last + 0.12)
                end = min(beat.end, video_duration)
                if end - start < 1.4:
                    continue
                cleaned.append(
                    enrich_scene(
                        _stabilize_beat(
                            beat.model_copy(
                                update={"start": round(start, 3), "end": round(end, 3)}
                            ),
                            canvas="full",
                        )
                    )
                )
                last = end
            if cleaned and cleaned[0].kicker.upper() != "HOOK":
                cleaned[0] = cleaned[0].model_copy(update={"kicker": "HOOK"})
            return cover_duration(cleaned, video_duration) or fallback
        except Exception as exc:
            logger.warning("[%s] scenes LLM failed (%s); using heuristic", job_id[:8], exc)
            return fallback

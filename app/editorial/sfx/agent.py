from __future__ import annotations

import json
import logging
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
from app.editorial.models import EditorialAnalysis, GraphicBeat, SfxHit
from app.editorial.sfx.planner import _dedupe, plan_sfx


logger = logging.getLogger(__name__)

APP_NAME = "kalki_sfx_app"
USER_ID = "kalki_pipeline"

SFX_INSTRUCTION = """You are the sound editor for a 9:16 educational Reel.

Place SHORT whoosh / impact hits under the voice. Sparse. Professional. Never wallpaper.

Kinds (only these):
- impact: hook, hard reveal, first frame
- hit: a number, punchline, answer
- swoosh: VS / contrast
- whoosh: process steps, CTA, card change when nothing stronger fits

Rules:
- 6–14 hits for a 1–3 minute talk. One hit about every 8–12 seconds, plus the hook.
- Hit the HOOK in the first 1.5 seconds.
- Align `at` to sentence.start or graphic.start (not mid-word).
- Do not stack two hits closer than 1.6s.
- Skip glass, beeps, comedy, and long rumbles.

Call submit_sfx ONCE:
{"sfx":[{"at":0.12,"kind":"impact","reason":"hook"},{"at":7.1,"kind":"swoosh","reason":"compare"}]}
"""


def _hits_from_payload(raw: str | dict[str, Any], duration: float) -> list[SfxHit]:
    payload = _extract_json(raw) if isinstance(raw, str) else raw
    items = payload.get("sfx") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError("sfx must be a list")
    hits: list[SfxHit] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        if kind not in {"whoosh", "swoosh", "impact", "hit"}:
            continue
        hits.append(
            SfxHit(
                at=float(item.get("at") or 0),
                kind=kind,  # type: ignore[arg-type]
                reason=str(item.get("reason") or "")[:40],
            )
        )
    return _dedupe(hits, duration)


class SfxAgent:
    def __init__(self) -> None:
        self.use_llm = bool(settings.sfx_llm_enabled and settings.llm_api_key)
        self.model = settings.llm_model
        self.api_key = settings.llm_api_key
        self.api_base = settings.llm_base_url or None
        self._lite_llm: LiteLlm | None = None
        if self.use_llm:
            lite_kwargs: dict[str, Any] = {
                "model": _lite_llm_model(self.model),
                "api_key": self.api_key or None,
                "api_base": self.api_base,
                "timeout": 60,
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
                timeout=60_000,
            )
        return LlmAgent(
            name="sfx_agent",
            model=self._lite_llm,
            description="Places timed whoosh and impact hits on a Reel.",
            instruction=SFX_INSTRUCTION,
            tools=tools,
            generate_content_config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=2500,
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

    async def plan(
        self,
        analysis: EditorialAnalysis,
        graphics: list[GraphicBeat],
        *,
        video_duration: float,
        job_id: str = "sfx",
    ) -> list[SfxHit]:
        fallback = plan_sfx(analysis, graphics, video_duration=video_duration)
        if not self.use_llm:
            logger.info("[%s] sfx: heuristic %s hits", job_id[:8], len(fallback))
            return fallback
        state: dict[str, Any] = {"accepted": False, "hits": None}

        def submit_sfx(sfx_json: str, tool_context: ToolContext) -> dict[str, Any]:
            """Submit timed sound effects as JSON."""
            try:
                hits = _hits_from_payload(sfx_json, video_duration)
                if not hits:
                    raise ValueError("no sfx in payload")
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                state["accepted"] = False
                return {"ok": False, "error": str(exc)}
            state["hits"] = hits
            state["accepted"] = True
            tool_context.actions.skip_summarization = True
            tool_context.get_invocation_context().end_invocation = True
            return {"ok": True, "sfx_count": len(hits)}

        prompt = (
            "Place SFX on this talk. Hook first. Sparse. Call submit_sfx once.\n\n"
            + json.dumps(
                {
                    "duration": round(video_duration, 2),
                    "sentences": [
                        {
                            "start": round(s.start, 2),
                            "role": s.editorial_role,
                            "text": s.text[:80],
                        }
                        for s in analysis.sentences[:40]
                    ],
                    "graphics": [
                        {
                            "start": round(g.start, 2),
                            "kind": g.kind,
                            "kicker": g.kicker,
                            "title": g.title[:40],
                        }
                        for g in graphics[:24]
                    ],
                },
                ensure_ascii=False,
            )
        )
        try:
            agent = self._build_agent([submit_sfx])
            final_text = await self._run_agent(agent, prompt, session_id=f"{job_id}-sfx")
            if state["accepted"] and state["hits"]:
                logger.info("[%s] sfx: LLM %s hits", job_id[:8], len(state["hits"]))
                return state["hits"]
            if final_text.strip():
                hits = _hits_from_payload(final_text, video_duration)
                if hits:
                    logger.info("[%s] sfx: LLM parse %s hits", job_id[:8], len(hits))
                    return hits
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] sfx LLM failed (%s); using heuristic", job_id[:8], exc)
        logger.info("[%s] sfx: heuristic %s hits", job_id[:8], len(fallback))
        return fallback

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
from app.asr import fix_asr_text, stabilize_copy
from app.editorial.graphics.glyphs import resolve_glyph
from app.editorial.graphics.planner import cover_duration, plan_graphics
from app.editorial.graphics.terms import extract_terms
from app.editorial.models import EditorialAnalysis, GraphicBeat, GraphicBullet, GraphicNode


logger = logging.getLogger(__name__)

APP_NAME = "kalki_graphics_app"
USER_ID = "kalki_pipeline"

GRAPHICS_INSTRUCTION = """You are the motion-graphics director for a 9:16 educational Reel.

The TOP 1080×720 px is YOUR infographic. The bottom is a talking head. Captions sit on y=720. Stay above y=620.

You complete the job yourself. Do not ask questions.

Language law (non-negotiable):
- Every title, kicker, bullet, chip, node label, and subtitle MUST be English.
- ASR may be Hindi, Hinglish, or mixed. Translate the meaning into English slide copy.
- Never print Devanagari or Hinglish filler (hai, karna, aapko, toh, yeh).
- Keep technical terms in English (RAG, LLM, GDPR, Fine-tuning).

## What the cards must do
- Each card explains WHAT THE SPEAKER IS SAYING in that time window. Not a generic AI poster.
- Rewrite as slide copy (headline + 1–2 bullets). Do not paste the transcript.
- If ASR or the speaker misspeaks (RAKA, fine tunning, destillation), print the intended term correctly (RAG, Fine-tuning, distillation).
- Never invent a system, number, or claim that is not in that window.
- If they ask a question, the card is the question. If they define a term, the card is the definition. If they compare, use vs_split.

## Workflow
1. Read timestamped sentences (ASR, may contain slips).
2. Design cards locked to those timestamps.
3. Call submit_graphics ONCE with JSON.
4. If ok=false, fix and submit again.

## Canvas
- One card at a time. No overlaps.
- Each card 5–8 seconds. Cover first sentence to last.
- Title: 3–7 words, Title Case, ONE line, ≤40 characters. Prefer one line.
- kicker: 1–2 words UPPERCASE (HOOK, INSIGHT, COMPARE, STEPS, NUMBER).
- bullets: MAX TWO, each ≤32 characters.
- vs_split: left/right 1–3 words. NO bullets.
- process: 3 chips, 1–2 words each.
- stat: a number or short metric + one subtitle.

## Kind variety (important)
Do NOT make every card "bullets". Pick the kind that matches the sentence:
- A number is the point ("1000 tables", "73%", "3x") → stat (we animate a count-up).
- A pipeline, flow, or sequence of steps → process. process MUST include
  exactly 3 chips (1–2 words each, e.g. "Question","Query","Answer");
  if you cannot name 3 concrete steps, use bullets instead.
- Two things compared or contrasted → vs_split.
- A strong spoken question or one-liner → topic with a short display title.
Aim for at least one stat / process / vs_split when the content supports it.

## Glyphs
search | brain | scale | gear | chart | bolt | docs | loop | target | spark | shield | clock

## Motion
- bullets/topic: slide_up
- vs_split/process: slide_up
- stat: scale_in

## Spelling (always)
RAG, Fine-tuning, LoRA, PEFT, LLM, distillation, quantization, embeddings.
No hashtags. No ALL CAPS titles. No third bullet.
"""


def _chunk(items: list[dict[str, Any]], size: int = 10) -> list[list[dict[str, Any]]]:
    if size <= 0 or len(items) <= size:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _clip_words(text: str, max_chars: int) -> str:
    text = stabilize_copy(text)
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars + 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.strip(" ,.;:") or text[:max_chars]


def _glossary(sentences: list[dict[str, Any]]) -> list[str]:
    blob = " ".join(str(s.get("text") or "") for s in sentences)
    return [name for name, _sub in extract_terms(blob)]


def _stabilize_beat(beat: GraphicBeat, *, canvas: str = "split") -> GraphicBeat:
    full = canvas == "full"
    title_n = 44 if full else 40
    sub_n = 42 if full else 28
    bullet_n = 44 if full else 32
    chip_n = 18 if full else 12
    side_n = 18 if full else 16
    nodes = [
        GraphicNode(
            label=_clip_words(n.label, 20 if full else 12),
            sub=_clip_words(n.sub, 36 if full else 20),
        )
        for n in (beat.nodes or [])[: (6 if full else 3)]
        if n.label.strip()
    ]
    bullets = list(beat.bullets)[: (5 if full else 2)]
    return beat.model_copy(
        update={
            "title": _clip_words(beat.title, title_n),
            "subtitle": _clip_words(beat.subtitle, sub_n),
            "left": _clip_words(beat.left, side_n),
            "right": _clip_words(beat.right, side_n),
            "chips": [_clip_words(c, chip_n) for c in beat.chips[: (6 if full else 3)]],
            "nodes": nodes,
            "bullets": [
                b.model_copy(update={"text": _clip_words(b.text, bullet_n)})
                for b in bullets
            ],
        }
    )


def _beats_from_payload(
    raw: Any, duration: float, *, canvas: str = "split"
) -> list[GraphicBeat]:
    if isinstance(raw, str):
        raw = _extract_json(raw)
    items = raw.get("graphics") if isinstance(raw, dict) else raw
    if isinstance(raw, dict) and not isinstance(items, list):
        items = raw.get("scenes")
    if not isinstance(items, list):
        raise ValueError("expected graphics array")
    full = canvas == "full"
    beats: list[GraphicBeat] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "bullets")
        if kind not in {
            "term_card",
            "vs_split",
            "stat",
            "chip_row",
            "process",
            "diagram",
            "quote",
            "topic",
            "bullets",
        }:
            kind = "bullets"
        if full:
            max_bullets = 5
        elif kind == "vs_split":
            max_bullets = 0
        elif kind in {"process", "diagram", "stat"}:
            max_bullets = 1
        else:
            max_bullets = 2
        bullet_n = 44 if full else 32
        bullets_raw = item.get("bullets") or []
        bullets: list[GraphicBullet] = []
        for i, b in enumerate(bullets_raw[:max_bullets]):
            if isinstance(b, str):
                text = _clip_words(b, bullet_n)
                if text:
                    bullets.append(GraphicBullet(text=text, delay_ms=i * 420))
                continue
            if not isinstance(b, dict):
                continue
            text = _clip_words(str(b.get("text") or ""), bullet_n)
            if not text:
                continue
            bullets.append(
                GraphicBullet(
                    text=text,
                    icon=str(b.get("icon") or "")[:8],
                    delay_ms=int(b.get("delay_ms") if b.get("delay_ms") is not None else i * 420),
                )
            )
        start = float(item.get("start") or 0)
        end = float(item.get("end") or (start + 6))
        end = min(end, duration)
        if end - start < 1.2:
            continue
        title = _clip_words(str(item.get("title") or "").strip(), 44 if full else 40)
        if not title:
            continue
        glyph = resolve_glyph(
            str(item.get("icon") or ""),
            kind,
            str(item.get("glyph") or ""),
        )
        motion = str(item.get("motion") or "")
        if motion not in {"fade", "slide_up", "scale_in"}:
            motion = "scale_in" if kind == "stat" else "slide_up"
        chip_n = 18 if full else 12
        chips = [
            _clip_words(str(c), chip_n)
            for c in (item.get("chips") or [])[: (6 if full else 3)]
            if str(c).strip()
        ]
        nodes: list[GraphicNode] = []
        for n in (item.get("nodes") or [])[:6]:
            if isinstance(n, str):
                label = _clip_words(n, 20)
                if label:
                    nodes.append(GraphicNode(label=label))
                continue
            if not isinstance(n, dict):
                continue
            label = _clip_words(str(n.get("label") or n.get("text") or ""), 20)
            if not label:
                continue
            nodes.append(
                GraphicNode(
                    label=label,
                    sub=_clip_words(str(n.get("sub") or n.get("detail") or ""), 36),
                )
            )
        beats.append(
            GraphicBeat(
                start=round(start, 3),
                end=round(end, 3),
                kind=kind,  # type: ignore[arg-type]
                title=title,
                subtitle=_clip_words(str(item.get("subtitle") or ""), 42 if full else 28),
                kicker=_clip_words(str(item.get("kicker") or "").upper(), 14),
                icon=str(item.get("icon") or "")[:8],
                glyph=glyph,
                chips=chips,
                nodes=nodes,
                bullets=bullets,
                left=_clip_words(str(item.get("left") or ""), 18 if full else 16),
                right=_clip_words(str(item.get("right") or ""), 18 if full else 16),
                motion=motion,  # type: ignore[arg-type]
                confidence=0.82,
            )
        )
    return [_stabilize_beat(b, canvas=canvas) for b in beats]


class GraphicsAgent:
    def __init__(self) -> None:
        self.api_key = settings.llm_api_key.replace("Bearer ", "").strip()
        self.api_base = (settings.llm_base_url or "").rstrip("/") or None
        self.model = settings.llm_model
        self.use_llm = bool(settings.graphics_llm_enabled and self.api_key)
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
            name="graphics_agent",
            model=self._lite_llm,
            description="Writes context-aware motion graphics for the top half of a split reel.",
            instruction=GRAPHICS_INSTRUCTION,
            tools=tools,
            generate_content_config=types.GenerateContentConfig(
                temperature=0.35,
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

        def submit_graphics(graphics_json: str, tool_context: ToolContext) -> dict[str, Any]:
            """Submit motion-graphic cards as JSON."""
            try:
                beats = _beats_from_payload(graphics_json, duration)
                if not beats:
                    raise ValueError("no graphics in payload")
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                state["accepted"] = False
                return {"ok": False, "error": str(exc)}
            state["beats"] = beats
            state["accepted"] = True
            tool_context.actions.skip_summarization = True
            tool_context.get_invocation_context().end_invocation = True
            return {"ok": True, "graphic_count": len(beats)}

        agent = self._build_agent([submit_graphics])
        prompt = (
            "Write infographic cards for THIS time window only. "
            "ASR may be Hindi or Hinglish — all card copy must be English. "
            "Match what is said at each timestamp. Correct ASR/misspeaks. "
            "Do not invent facts. Max 2 bullets. vs_split has no bullets. "
            f"Video duration={duration:.2f}s. Call submit_graphics once.\n\n"
            + json.dumps(
                {
                    "duration": round(duration, 2),
                    "canvas": "1080x720",
                    "spell_as": glossary or ["RAG", "Fine-tuning", "LLM"],
                    "sentences": sentences,
                },
                ensure_ascii=False,
            )
        )
        final_text = await self._run_agent(agent, prompt, session_id=f"{job_id}-g{chunk_index}")
        if state["accepted"] and state["beats"]:
            return state["beats"]
        if final_text.strip():
            try:
                return _beats_from_payload(final_text, duration)
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
                pass
        raise RuntimeError("Graphics agent did not submit valid cards")

    async def plan(
        self,
        analysis: EditorialAnalysis,
        *,
        video_duration: float,
        job_id: str = "graphics",
    ) -> list[GraphicBeat]:
        fallback = [_stabilize_beat(b) for b in plan_graphics(analysis, video_duration=video_duration)]
        if not self.use_llm:
            logger.info(
                "[%s] graphics: heuristic (LLM off), %s cards",
                job_id[:8],
                len(fallback),
            )
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
            chunks = _chunk(payload, size=12)
            logger.info(
                "[%s] graphics LLM: %s sentences in %s chunk(s)",
                job_id[:8],
                len(payload),
                len(chunks),
            )
            merged: list[GraphicBeat] = []
            for i, chunk in enumerate(chunks):
                logger.info("[%s] graphics chunk %s/%s", job_id[:8], i + 1, len(chunks))
                # A failed chunk must not sink the whole plan: retry once,
                # then skip it (cover_duration stretches neighbors over gaps).
                try:
                    merged.extend(
                        await self._generate_chunk(
                            chunk, video_duration, job_id, i, glossary=glossary
                        )
                    )
                except Exception as chunk_exc:
                    logger.warning(
                        "[%s] graphics chunk %s failed (%s); retrying once",
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
                            "[%s] graphics chunk %s failed twice (%s); skipping",
                            job_id[:8],
                            i + 1,
                            retry_exc,
                        )
            if not merged:
                logger.warning(
                    "[%s] graphics LLM returned nothing; using heuristic",
                    job_id[:8],
                )
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
                    _stabilize_beat(
                        beat.model_copy(
                            update={"start": round(start, 3), "end": round(end, 3)}
                        )
                    )
                )
                last = end
            return cover_duration(cleaned, video_duration) or fallback
        except Exception as exc:
            logger.warning(
                "[%s] graphics LLM failed (%s); using heuristic cards",
                job_id[:8],
                exc,
            )
            return fallback

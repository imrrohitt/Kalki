from __future__ import annotations

from app.captions.models import CaptionTreatment

_ACRONYMS = {
    "AI": "AI",
    "RAG": "RAG",
    "LLM": "LLM",
    "LLMS": "LLMs",
    "GDPR": "GDPR",
    "LORA": "LoRA",
    "PEFT": "PEFT",
    "API": "API",
    "GPU": "GPU",
    "CPU": "CPU",
    "SQL": "SQL",
    "CTA": "CTA",
}

_PRONOUN_I = {"I", "I'M", "I'LL", "I'VE", "I'D"}

_TREATMENTS = {
    "plain",
    "mix",
    "serif",
    "quote",
    "oval",
    "underline",
    "blob",
    "stack",
    "tape",
    "chip",
}


def parse_treatment(value: object) -> CaptionTreatment:
    token = str(value or "plain").strip().lower()
    if token in _TREATMENTS:
        return token  # type: ignore[return-value]
    return "plain"


def _case_token(token: str) -> str:
    lead = ""
    trail = ""
    body = token
    while body and body[0] in "\"“”'":
        lead += body[0]
        body = body[1:]
    while body and body[-1] in "\"“”',.!?":
        trail = body[-1] + trail
        body = body[:-1]
    if not body:
        return token
    key = body.upper()
    if key in _ACRONYMS:
        return lead + _ACRONYMS[key] + trail
    if key in _PRONOUN_I:
        if "'" in body:
            return lead + "I" + body[1:].lower() + trail
        return lead + "I" + trail
    return lead + body.lower() + trail


def _cap_first(line: str) -> str:
    for i, ch in enumerate(line):
        if ch.isalpha():
            return line[:i] + ch.upper() + line[i + 1 :]
    return line


def caption_spoken_case(text: str, *, treatment: CaptionTreatment = "plain") -> str:
    """Lowercase spoken captions. Keep I / acronyms. Hooks may take a capital."""
    lines: list[str] = []
    for li, line in enumerate((text or "").replace("\\n", "\n").split("\n")):
        tokens = [_case_token(tok) for tok in line.split() if tok]
        if not tokens:
            continue
        built = " ".join(tokens)
        if treatment in {"serif", "quote", "oval", "underline", "blob", "stack"} and li == 0:
            built = _cap_first(built)
        lines.append(built)
    return "\n".join(lines)

from __future__ import annotations

import re

from app.asr import fix_asr_text
from app.editorial.models import EditorialSentence

# (pattern, canonical, subtitle)
_TERM_SPECS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\b(rag|raka)\s+system\b", re.I), "RAG", "retrieval over your documents"),
    (re.compile(r"\b(rag|raka)\b", re.I), "RAG", "retrieve, then generate"),
    (re.compile(r"\bfine[-\s]?tun(?:e|ing)\b", re.I), "Fine-tuning", "train the model on your data"),
    (re.compile(r"\blora\b", re.I), "LoRA", "low-rank adaptation"),
    (re.compile(r"\bpeft\b|\bparameter[-\s]?efficien", re.I), "PEFT", "train adapters, not the full net"),
    (re.compile(r"\bllm\b", re.I), "LLM", "the base model"),
    (re.compile(r"\bembedd", re.I), "Embeddings", "vectors for retrieval"),
    (re.compile(r"\bvector\s+(db|database|store)\b", re.I), "Vector DB", "where chunks live"),
    (re.compile(r"\bre-?index", re.I), "Re-index", "refresh the document store"),
    (re.compile(r"\bstatic data\b", re.I), "Static data", "stable enough to train on"),
    (re.compile(r"\b(latest|changing|frequently changed)\b", re.I), "Latest data", "updates without retraining"),
    (re.compile(r"\bdomain\b", re.I), "Domain data", "your company's knowledge"),
    (re.compile(r"\bdocuments?\b", re.I), "Documents", "source of truth for RAG"),
    (re.compile(r"\bcost", re.I), "Cost", "GPU time vs retrieval"),
    (re.compile(r"\b(ai interviews?|llm)\b", re.I), "AI interviews", "core LLM concepts"),
]

_VS = re.compile(
    r"\b(over the|vs\.?|versus|instead of|rather than)\b",
    re.I,
)
_PROCESS = re.compile(
    r"\b(first|second|then|next|starts at|breakdown)\b",
    re.I,
)
_NUMBER = re.compile(r"\b(\d+(\.\d+)?\s*(%|x|k|million)?)\b", re.I)


def extract_terms(text: str) -> list[tuple[str, str]]:
    """Unique (canonical, subtitle) hits, RAG/fine-tune first."""
    cleaned = fix_asr_text(text)
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pattern, canonical, subtitle in _TERM_SPECS:
        if pattern.search(cleaned) and canonical not in seen:
            seen.add(canonical)
            found.append((canonical, subtitle))
    priority = {"RAG": 0, "Fine-tuning": 1, "LoRA": 2, "PEFT": 3, "LLM": 4}
    found.sort(key=lambda item: (priority.get(item[0], 9), item[0]))
    return found


def is_contrast(sentence: EditorialSentence) -> bool:
    text = fix_asr_text(sentence.text)
    if sentence.editorial_role in {"contrast", "contradiction"}:
        return True
    return bool(_VS.search(text))


def is_process(text: str) -> bool:
    return bool(_PROCESS.search(fix_asr_text(text)))


def extract_number(text: str) -> str | None:
    match = _NUMBER.search(fix_asr_text(text))
    return match.group(1).strip() if match else None

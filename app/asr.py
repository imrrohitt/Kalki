from __future__ import annotations

import re

# ASR slips and common misspeaks → on-screen canonical form.
ASR_FIXES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bRAKA\b", re.I), "RAG"),
    (re.compile(r"\braka\b", re.I), "RAG"),
    (re.compile(r"\brag system\b", re.I), "RAG system"),
    (re.compile(r"\bretrieval[\s-]*augmented[\s-]*generat\w*\b", re.I), "RAG"),
    (re.compile(r"\bfine[\s-]*tunn+ings?\b", re.I), "Fine-tuning"),
    (re.compile(r"\bfine[\s-]*tuneings?\b", re.I), "Fine-tuning"),
    (re.compile(r"\bfine[\s-]*tunings?\b", re.I), "Fine-tuning"),
    (re.compile(r"\bfune[\s-]*tunings?\b", re.I), "Fine-tuning"),
    (re.compile(r"\bfine[\s-]*tunes?\b", re.I), "Fine-tune"),
    (re.compile(r"\blora\b", re.I), "LoRA"),
    (re.compile(r"\bpeft\b", re.I), "PEFT"),
    (re.compile(r"\bParameterficion\b", re.I), "PEFT"),
    (re.compile(r"\bparameterficion\b", re.I), "PEFT"),
    (re.compile(r"\bparameter[\s-]*efficient\b", re.I), "PEFT"),
    (re.compile(r"\bllms\b", re.I), "LLMs"),
    (re.compile(r"\bllm\b", re.I), "LLM"),
    (re.compile(r"\bdestillat\w*\b", re.I), "distillation"),
    (re.compile(r"\bdistilation\b", re.I), "distillation"),
    (re.compile(r"\bquantisation\b", re.I), "quantization"),
    (re.compile(r"\bqunatization\b", re.I), "quantization"),
    (re.compile(r"\bstateic\b", re.I), "static"),
    (re.compile(r"\bcompetition power\b", re.I), "compute power"),
    (re.compile(r"\bcompution\b", re.I), "compute"),
    (re.compile(r"\bembedings?\b", re.I), "embeddings"),
    (re.compile(r"\bvector\s+d\.?b\.?\b", re.I), "vector DB"),
]


def fix_asr_text(text: str) -> str:
    out = text
    for pattern, repl in ASR_FIXES:
        out = pattern.sub(repl, out)
    return out


def known_terms_in(text: str) -> list[str]:
    cleaned = fix_asr_text(text)
    names = [
        "RAG",
        "Fine-tuning",
        "LoRA",
        "PEFT",
        "LLM",
        "distillation",
        "quantization",
        "embeddings",
    ]
    found: list[str] = []
    for name in names:
        if re.search(rf"\b{re.escape(name)}\b", cleaned, re.I):
            found.append(name)
    return found


def stabilize_copy(text: str) -> str:
    """Clean ASR/speaker slips for on-screen type. Keeps line breaks."""
    raw = (text or "").replace("\\n", "\n")
    lines: list[str] = []
    for line in raw.split("\n"):
        cleaned = " ".join(fix_asr_text(line).split()).strip(" ,.;:")
        lines.append(cleaned)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)

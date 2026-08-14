from __future__ import annotations

from app.editorial.models import EditorialRole, ZoomEasing, ZoomStyle


class ZoomRecipe:
    __slots__ = ("style", "target_scale", "easing", "start_offset", "release")

    def __init__(
        self,
        style: ZoomStyle,
        target_scale: float,
        easing: ZoomEasing,
        start_offset: float = 0.0,
        release: float = 0.22,
    ) -> None:
        self.style = style
        self.target_scale = target_scale
        self.easing = easing
        self.start_offset = start_offset
        self.release = release


# LLM owns intensity + timing. These leftover recipes are style labels only.
INTENT_RECIPES: dict[EditorialRole, ZoomRecipe] = {
    "emphasis": ZoomRecipe("fast_punch", 1.22, "ease_out"),
    "surprise": ZoomRecipe("fast_punch", 1.32, "ease_out"),
    "contrast": ZoomRecipe("slow_punch", 1.22, "ease_in_out"),
    "strong_opinion": ZoomRecipe("fast_punch", 1.30, "ease_out"),
    "key_insight": ZoomRecipe("slow_punch", 1.24, "ease_in_out"),
    "important_number": ZoomRecipe("hold", 1.26, "linear", release=0.18),
    "question": ZoomRecipe("slow_punch", 1.16, "ease_out"),
    "answer": ZoomRecipe("fast_punch", 1.28, "ease_out"),
    "story_climax": ZoomRecipe("slow_punch", 1.28, "ease_in_out"),
    "reveal": ZoomRecipe("fast_punch", 1.32, "ease_out", start_offset=0.18),
    "warning": ZoomRecipe("fast_punch", 1.32, "ease_out"),
    "hook": ZoomRecipe("fast_punch", 1.26, "ease_out", start_offset=-0.15),
    "cta": ZoomRecipe("none", 1.0, "linear", release=0.0),
    "emotional_peak": ZoomRecipe("slow_punch", 1.24, "ease_in_out"),
    "humor": ZoomRecipe("punch_release", 1.30, "ease_in_out", release=0.0),
    "transition": ZoomRecipe("slow_punch", 1.16, "ease_in_out"),
    "generic": ZoomRecipe("none", 1.0, "linear", release=0.0),
    "assumption": ZoomRecipe("none", 1.0, "linear", release=0.0),
    "contradiction": ZoomRecipe("slow_punch", 1.20, "ease_out"),
}

ROLE_SEMANTIC_WEIGHT: dict[EditorialRole, float] = {
    "reveal": 0.95,
    "story_climax": 0.93,
    "warning": 0.90,
    "hook": 0.88,
    "strong_opinion": 0.86,
    "important_number": 0.84,
    "surprise": 0.84,
    "answer": 0.82,
    "key_insight": 0.80,
    "emphasis": 0.78,
    "emotional_peak": 0.78,
    "contrast": 0.76,
    "humor": 0.74,
    "contradiction": 0.70,
    "question": 0.62,
    "transition": 0.55,
    "assumption": 0.32,
    "cta": 0.12,
    "generic": 0.05,
}

SKIP_ROLES = {"cta", "generic", "assumption"}
HIGH_PRIORITY_ROLES = {
    "reveal",
    "story_climax",
    "warning",
    "hook",
    "strong_opinion",
}

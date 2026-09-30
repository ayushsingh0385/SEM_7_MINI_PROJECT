"""The single result object every modality detector returns."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import DECISION_THRESHOLD, UNCERTAIN_MARGIN


@dataclass
class DetectionResult:
    """Outcome of one detection run.

    Attributes:
        ai_probability: P(content is AI-generated), in [0, 1].
        modality: "text" | "image" | "audio" | "video".
        model_name: which model produced this.
        explanation: per-modality evidence shown in the UI (feature values,
            patch votes, ...). Free-form, display-only.
        meta: anything else worth surfacing (word count, image size, ...).
    """

    ai_probability: float
    modality: str
    model_name: str = "unknown"
    explanation: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def human_probability(self) -> float:
        return 1.0 - self.ai_probability

    @property
    def is_ai(self) -> bool:
        return self.ai_probability >= DECISION_THRESHOLD

    @property
    def confidence(self) -> float:
        """How far the call sits from a coin flip, rescaled to [0, 1]."""
        return abs(self.ai_probability - 0.5) * 2.0

    @property
    def is_uncertain(self) -> bool:
        """True when the probability sits too close to the threshold to commit."""
        return abs(self.ai_probability - DECISION_THRESHOLD) < UNCERTAIN_MARGIN

    @property
    def verdict(self) -> str:
        if self.is_uncertain:
            return "Uncertain"
        return "AI-generated" if self.is_ai else "Human / Real"

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "ai_probability": round(self.ai_probability, 4),
            "human_probability": round(self.human_probability, 4),
            "confidence": round(self.confidence, 4),
            "uncertain": self.is_uncertain,
            "modality": self.modality,
            "model": self.model_name,
            "explanation": self.explanation,
            "meta": self.meta,
        }

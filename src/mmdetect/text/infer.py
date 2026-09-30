"""Serving-time wrapper around the trained text detector.

Loads the joblib pipeline once and answers `predict(text)` with a
:class:`DetectionResult`. Because every feature transform lives inside the
pipeline, there is no preprocessing here that could drift from training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import TEXT_METRICS_PATH, TEXT_MODEL_PATH
from ..result import DetectionResult
from ..utils import get_logger, load_json
from .features import FEATURE_NAMES, describe

log = get_logger("text.infer")

# Below this, style statistics are too noisy to trust; the UI warns the user.
MIN_RELIABLE_WORDS = 50

# Plain-English labels for the stylometric features shown in the UI.
_READABLE = {
    "avg_word_len": "Average word length",
    "avg_sent_len": "Average sentence length",
    "std_sent_len": "Sentence-length variation (burstiness)",
    "sent_len_cv": "Sentence-length variation, pace-adjusted",
    "max_sent_len": "Longest sentence",
    "type_token_ratio": "Vocabulary richness",
    "hapax_ratio": "Share of words used only once",
    "repeat_bigram_ratio": "Repeated word pairs",
    "stopword_ratio": "Common-word ratio",
    "long_word_ratio": "Share of long words",
    "comma_rate": "Comma usage",
    "period_rate": "Full-stop usage",
    "semicolon_rate": "Semicolon usage",
    "colon_rate": "Colon usage",
    "exclaim_rate": "Exclamation marks",
    "question_rate": "Question marks",
    "quote_rate": "Quotation marks",
    "dash_rate": "Dash usage",
    "paren_rate": "Parentheses",
    "apostrophe_rate": "Apostrophes / contractions",
    "uppercase_ratio": "Capital letters",
    "digit_ratio": "Digits",
    "whitespace_anomaly": "Irregular spacing",
    "paragraph_rate": "Paragraph breaks",
}


class TextDetector:
    """Lazy-loading wrapper around the trained text pipeline."""

    modality = "text"

    def __init__(self, model_path: Path | None = None):
        self.model_path = Path(model_path or TEXT_MODEL_PATH)
        self._pipeline = None
        self._stylo_coefs: np.ndarray | None = None
        self._scaler = None

    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        return self.model_path.exists()

    def load(self):
        """Load the pipeline from disk (idempotent)."""
        if self._pipeline is not None:
            return self._pipeline
        if not self.is_available:
            raise FileNotFoundError(
                f"No trained text model at {self.model_path}.\n"
                "Train one with:  python scripts/train_text.py"
            )
        import joblib

        log.info("loading text model from %s", self.model_path.name)
        self._pipeline = joblib.load(self.model_path)
        self._cache_stylo_parts()
        return self._pipeline

    def _cache_stylo_parts(self) -> None:
        """Grab the stylometric coefficients so verdicts can be explained.

        The FeatureUnion concatenates [word tfidf | char tfidf | stylometry],
        so the stylometric weights are the final ``len(FEATURE_NAMES)`` entries
        of the logistic-regression coefficient vector.
        """
        try:
            coefs = self._pipeline.named_steps["clf"].coef_[0]
            self._stylo_coefs = coefs[-len(FEATURE_NAMES):]
            stylo_pipe = dict(
                self._pipeline.named_steps["features"].transformer_list
            )["stylo"]
            self._scaler = stylo_pipe.named_steps["scale"]
        except Exception as exc:
            log.warning("could not extract stylometric weights: %s", exc)
            self._stylo_coefs = None

    # ------------------------------------------------------------------
    def _explain(self, text: str) -> dict[str, Any]:
        """Per-feature contributions toward the AI / human decision.

        For a linear model the contribution of feature i is
        ``coefficient_i * scaled_value_i``; positive pushes toward AI.
        """
        raw = describe(text)
        out: dict[str, Any] = {"raw_features": raw}

        if self._stylo_coefs is None or self._scaler is None:
            return out

        try:
            vals = np.asarray([[raw[n] for n in FEATURE_NAMES]], dtype=np.float64)
            scaled = self._scaler.transform(vals)[0]
            contrib = self._stylo_coefs * scaled

            order = np.argsort(np.abs(contrib))[::-1][:6]
            out["signals"] = [
                {
                    "feature": FEATURE_NAMES[i],
                    "label": _READABLE.get(FEATURE_NAMES[i],
                                           FEATURE_NAMES[i].replace("_", " ").title()),
                    "value": float(raw[FEATURE_NAMES[i]]),
                    "contribution": float(contrib[i]),
                    "points_to": "AI" if contrib[i] > 0 else "Human",
                }
                for i in order
            ]
        except Exception as exc:
            log.warning("explanation failed: %s", exc)
        return out

    @staticmethod
    def _prepare(text: str) -> str:
        """Apply the exact preprocessing the training corpus received.

        The training data is passed through ``normalise_text`` to strip the
        detokenisation spacing out of HC3's human answers. Serving has to do the
        same thing or the model sees a distribution it was never fitted on -
        the classic train/serve skew. It is a no-op on normally punctuated text,
        which is what most input will be.
        """
        from ..data.text_data import normalise_text

        return normalise_text((text or "").strip())

    def predict(self, text: str) -> DetectionResult:
        """Classify one document."""
        pipeline = self.load()
        text = self._prepare(text)
        if not text:
            raise ValueError("Empty text: nothing to analyse.")

        proba = float(pipeline.predict_proba([text])[0, 1])
        n_words = len(text.split())

        return DetectionResult(
            ai_probability=proba,
            modality="text",
            model_name="TF-IDF + stylometry -> Logistic Regression",
            explanation=self._explain(text),
            meta={
                "characters": len(text),
                "words": n_words,
                "too_short": n_words < MIN_RELIABLE_WORDS,
                "min_reliable_words": MIN_RELIABLE_WORDS,
            },
        )

    def predict_batch(self, texts: list[str]) -> list[DetectionResult]:
        """Classify many documents in one pass (much faster than looping).

        Runs the same :meth:`_prepare` step as :meth:`predict`, so the two paths
        cannot disagree about a document.
        """
        pipeline = self.load()
        prepared = [self._prepare(t) for t in texts]
        probas = pipeline.predict_proba(prepared)[:, 1]
        return [
            DetectionResult(
                ai_probability=float(p),
                modality="text",
                model_name="TF-IDF + stylometry -> Logistic Regression",
                meta={"words": len(t.split())},
            )
            for p, t in zip(probas, prepared)
        ]

    # ------------------------------------------------------------------
    def training_metrics(self) -> dict[str, Any] | None:
        """Held-out metrics recorded at training time, for the UI."""
        if TEXT_METRICS_PATH.exists():
            try:
                return load_json(TEXT_METRICS_PATH)
            except Exception:
                return None
        return None


_DETECTOR: TextDetector | None = None


def get_text_detector() -> TextDetector:
    """Process-wide singleton, so Streamlit reruns do not reload the model."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = TextDetector()
    return _DETECTOR


__all__ = ["TextDetector", "get_text_detector", "MIN_RELIABLE_WORDS"]

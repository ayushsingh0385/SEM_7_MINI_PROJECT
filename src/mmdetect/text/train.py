"""Week 2: train the AI-vs-human text detector.

Model = a single sklearn Pipeline with three parallel feature views:

    word TF-IDF    (1-2 grams)   -> vocabulary and phrasing
    char TF-IDF    (3-5 grams)   -> punctuation/morphology, robust to new words
    stylometry     (24 features) -> burstiness, richness, punctuation habits

fed into a LogisticRegression. Logistic regression is deliberate: it emits
calibrated probabilities (the UI shows a percentage, not just a label) and its
coefficients are readable, so the report can show *which* features drove the
decision.

Everything lives in one Pipeline object, so inference is `pipeline.predict_proba`
on raw strings with no feature-engineering code duplicated at serving time.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import (
    RANDOM_SEED,
    TEXT_METRICS_PATH,
    TEXT_MODEL_PATH,
    TEXT_REPORT_DIR,
)
from ..utils import get_logger, save_json, set_seed
from .features import StylometricFeatures

log = get_logger("text.train")

MODEL_NAME = "tfidf+stylometry-logreg"


def build_pipeline(
    word_features: int = 100_000,
    char_features: int = 50_000,
    C: float = 4.0,
) -> Pipeline:
    """Assemble the full feature-extraction + classifier pipeline."""
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=word_features,
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    # char_wb keeps n-grams inside word boundaries, which captures punctuation
    # and suffix habits without exploding on cross-word noise.
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=char_features,
        min_df=2,
        sublinear_tf=True,
        lowercase=False,
    )
    stylo = Pipeline([
        ("extract", StylometricFeatures()),
        # Stylometric columns span very different ranges (n_chars in thousands,
        # ratios in [0,1]); without scaling the regulariser would crush them.
        ("scale", StandardScaler()),
    ])

    features = FeatureUnion(
        [("word", word_tfidf), ("char", char_tfidf), ("stylo", stylo)],
        n_jobs=None,  # threads fight over the GIL here; serial is faster
    )

    clf = LogisticRegression(
        C=C,
        solver="liblinear",   # fastest for sparse binary problems this size
        max_iter=2000,
        random_state=RANDOM_SEED,
    )
    return Pipeline([("features", features), ("clf", clf)])


def evaluate(
    pipeline: Pipeline, x_test, y_test
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Score the pipeline; returns (metrics, ai_probabilities, predictions)."""
    proba = pipeline.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()

    metrics: dict[str, Any] = {
        "model": MODEL_NAME,
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": {
            "true_human": {"pred_human": int(tn), "pred_ai": int(fp)},
            "true_ai": {"pred_human": int(fn), "pred_ai": int(tp)},
        },
        # False positives matter most here: wrongly calling a student's own
        # essay "AI-generated" is the costly error, so it is reported up front.
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "report": classification_report(
            y_test, pred, target_names=["Human", "AI"], output_dict=True, zero_division=0
        ),
    }
    return metrics, proba, pred


def _save_plots(y_test, proba, pred, pipeline: Pipeline) -> None:
    """Write confusion-matrix, ROC and top-coefficient figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    TEXT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --- confusion matrix -------------------------------------------------
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Human", "AI"])
    ax.set_yticks([0, 1], ["Human", "AI"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Text detector - confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(TEXT_REPORT_DIR / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    # --- ROC curve --------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Text detector - ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(TEXT_REPORT_DIR / "roc_curve.png", dpi=140)
    plt.close(fig)

    # --- most influential features ---------------------------------------
    try:
        names = pipeline.named_steps["features"].get_feature_names_out()
        coefs = pipeline.named_steps["clf"].coef_[0]
        order = np.argsort(coefs)
        top_ai = order[-15:][::-1]
        top_human = order[:15]
        idx = np.concatenate([top_ai, top_human])
        fig, ax = plt.subplots(figsize=(7.5, 8))
        colors = ["#c0392b" if coefs[i] > 0 else "#2471a3" for i in idx]
        ax.barh(range(len(idx)), coefs[idx], color=colors)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([str(names[i])[:38] for i in idx], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Logistic-regression coefficient  (+ pushes toward AI)")
        ax.set_title("Top signals for each class")
        fig.tight_layout()
        fig.savefig(TEXT_REPORT_DIR / "top_features.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # plotting must never fail the training run
        log.warning("skipped feature-importance plot: %s", exc)


def train_text_model(
    word_features: int = 100_000,
    char_features: int = 50_000,
    C: float = 4.0,
    save: bool = True,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train, evaluate and persist the text detector."""
    from ..data.text_data import load_text_dataset

    set_seed()
    train_df, test_df = load_text_dataset()
    log.info("train=%d  test=%d", len(train_df), len(test_df))
    log.info("train balance: %s", train_df["label"].value_counts().to_dict())

    x_train = train_df["text"].tolist()
    y_train = train_df["label"].to_numpy()
    x_test = test_df["text"].tolist()
    y_test = test_df["label"].to_numpy()

    pipeline = build_pipeline(word_features, char_features, C)

    log.info("fitting (this takes a few minutes: char n-grams are the slow part)")
    t0 = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_secs = time.perf_counter() - t0
    log.info("fitted in %.1fs", train_secs)

    metrics, proba, pred = evaluate(pipeline, x_test, y_test)
    metrics["train_seconds"] = round(train_secs, 1)
    metrics["n_train"] = int(len(y_train))
    metrics["hyperparams"] = {
        "word_features": word_features,
        "char_features": char_features,
        "C": C,
    }
    try:
        n_feat = len(pipeline.named_steps["features"].get_feature_names_out())
        metrics["n_features"] = int(n_feat)
    except Exception:
        pass

    log.info(
        "accuracy=%.4f  f1=%.4f  roc_auc=%.4f  FPR=%.4f",
        metrics["accuracy"], metrics["f1"], metrics["roc_auc"],
        metrics["false_positive_rate"],
    )

    if save:
        TEXT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, TEXT_MODEL_PATH, compress=3)
        size_mb = TEXT_MODEL_PATH.stat().st_size / 1e6
        log.info("saved model -> %s (%.1f MB)", TEXT_MODEL_PATH, size_mb)

        save_json(metrics, TEXT_METRICS_PATH)
        _save_plots(y_test, proba, pred, pipeline)
        log.info("saved metrics + figures -> %s", TEXT_REPORT_DIR)

    return pipeline, metrics


__all__ = ["build_pipeline", "train_text_model", "evaluate", "MODEL_NAME"]

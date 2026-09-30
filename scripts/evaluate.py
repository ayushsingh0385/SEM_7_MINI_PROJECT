"""Re-evaluate every trained detector and print one combined report.

    python scripts/evaluate.py            # all trained modalities
    python scripts/evaluate.py --only text

Reads the saved models and scores them against the held-out test splits again,
rather than trusting the metrics file. Useful as the results table in a
write-up, and as a check that a saved model still loads and behaves.
"""
import argparse

import _bootstrap  # noqa: F401

import numpy as np

from mmdetect.config import DECISION_THRESHOLD
from mmdetect.registry import MODALITIES


def _bar(value: float, width: int = 28) -> str:
    filled = int(round(value * width))
    return "#" * filled + "." * (width - filled)


def evaluate_text() -> dict | None:
    from mmdetect.data.text_data import load_text_dataset
    from mmdetect.text.infer import get_text_detector
    from mmdetect.text.train import evaluate as eval_text

    det = get_text_detector()
    if not det.is_available:
        print("  text : no trained model, skipping")
        return None

    pipeline = det.load()
    _, test_df = load_text_dataset()
    metrics, _, _ = eval_text(pipeline, test_df["text"].tolist(),
                              test_df["label"].to_numpy())
    return metrics


def evaluate_image() -> dict | None:
    from mmdetect.data.image_data import load_image_dataset
    from mmdetect.image.infer import get_image_detector
    from mmdetect.image.train import compute_metrics

    det = get_image_detector()
    if not det.is_available:
        print("  image : no trained model, skipping")
        return None

    det.load()
    (_, _), (x_test, y_test) = load_image_dataset()

    # Score the 32x32 test split directly - the multi-view path in the app is
    # for arbitrary uploads and would not match these native-resolution images.
    proba = det.score_batch(x_test)
    metrics = compute_metrics(y_test, proba)
    metrics["n_test"] = int(len(y_test))
    return metrics


def _print_block(name: str, metrics: dict) -> None:
    print(f"\n  {name.upper()}")
    print("  " + "-" * 50)
    if "n_test" in metrics:
        print(f"  {'samples':<18} {metrics['n_test']:,}")
    for key, label in [
        ("accuracy", "accuracy"),
        ("precision", "precision (AI)"),
        ("recall", "recall (AI)"),
        ("f1", "F1 (AI)"),
        ("roc_auc", "ROC-AUC"),
    ]:
        v = metrics[key]
        print(f"  {label:<18} {v:.4f}  {_bar(v)}")
    print(f"  {'false positives':<18} {metrics['false_positive_rate']:.4f}"
          "   (human wrongly flagged as AI)")
    print(f"  {'false negatives':<18} {metrics['false_negative_rate']:.4f}"
          "   (AI missed)")

    cm = metrics["confusion_matrix"]
    print("\n                     predicted human   predicted AI")
    print(f"  actually human   {cm['true_human']['pred_human']:>13,} "
          f"{cm['true_human']['pred_ai']:>14,}")
    print(f"  actually AI      {cm['true_ai']['pred_human']:>13,} "
          f"{cm['true_ai']['pred_ai']:>14,}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=["text", "image"], default=None,
                   help="evaluate a single modality")
    args = p.parse_args()

    print("=" * 56)
    print("  MULTIMEDIA AI CONTENT DETECTION - evaluation")
    print(f"  decision threshold: {DECISION_THRESHOLD}")
    print("=" * 56)

    targets = [args.only] if args.only else ["text", "image"]
    results = {}

    for key in targets:
        info = MODALITIES[key]
        if not info.is_ready:
            continue
        metrics = evaluate_text() if key == "text" else evaluate_image()
        if metrics:
            results[key] = metrics
            _print_block(key, metrics)

    if len(results) > 1:
        print("\n" + "=" * 56)
        print("  SUMMARY")
        print("=" * 56)
        print(f"  {'modality':<10}{'accuracy':>10}{'F1':>10}{'ROC-AUC':>10}{'FPR':>10}")
        for key, m in results.items():
            print(f"  {key:<10}{m['accuracy']:>10.4f}{m['f1']:>10.4f}"
                  f"{m['roc_auc']:>10.4f}{m['false_positive_rate']:>10.4f}")

    if not results:
        print("\n  Nothing to evaluate. Train a model first:")
        print("    python scripts/train_text.py")
        print("    python scripts/train_image.py")


if __name__ == "__main__":
    main()

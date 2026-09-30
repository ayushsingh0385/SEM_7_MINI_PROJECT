"""Train the AI-vs-human text detector (Week 2).

    python scripts/prepare_text_data.py     # run this first
    python scripts/train_text.py
    python scripts/train_text.py --C 2.0 --word-features 150000

Writes:
    models/text/text_detector.joblib
    reports/text/metrics.json
    reports/text/{confusion_matrix,roc_curve,top_features}.png
"""
import argparse

import _bootstrap  # noqa: F401

from mmdetect.config import ensure_dirs
from mmdetect.text.train import train_text_model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--word-features", type=int, default=100_000,
                   help="max word-level TF-IDF vocabulary size")
    p.add_argument("--char-features", type=int, default=50_000,
                   help="max char-level TF-IDF vocabulary size")
    p.add_argument("--C", type=float, default=4.0,
                   help="inverse regularisation strength for logistic regression")
    p.add_argument("--no-save", action="store_true",
                   help="evaluate without writing the model to disk")
    args = p.parse_args()

    ensure_dirs()
    _, metrics = train_text_model(
        word_features=args.word_features,
        char_features=args.char_features,
        C=args.C,
        save=not args.no_save,
    )

    print("\n" + "=" * 52)
    print("  TEXT DETECTOR - held-out test results")
    print("=" * 52)
    print(f"  test samples     : {metrics['n_test']:,}")
    print(f"  accuracy         : {metrics['accuracy']:.4f}")
    print(f"  precision (AI)   : {metrics['precision']:.4f}")
    print(f"  recall    (AI)   : {metrics['recall']:.4f}")
    print(f"  F1        (AI)   : {metrics['f1']:.4f}")
    print(f"  ROC-AUC          : {metrics['roc_auc']:.4f}")
    print(f"  false-positive   : {metrics['false_positive_rate']:.4f}"
          "   (human text wrongly flagged as AI)")
    print("=" * 52)
    print("\nNext:  python scripts/train_image.py")


if __name__ == "__main__":
    main()

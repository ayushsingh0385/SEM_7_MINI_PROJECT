"""Train the real-vs-AI image detector on CIFAKE (Week 3).

    python scripts/prepare_image_data.py    # run this first
    python scripts/train_image.py
    python scripts/train_image.py --epochs 25 --batch-size 128

Writes:
    models/image/image_detector.pt
    reports/image/metrics.json
    reports/image/{confusion_matrix,roc_curve,training_curves}.png
"""
import argparse

import _bootstrap  # noqa: F401

from mmdetect.config import ensure_dirs
from mmdetect.image.train import train_image_model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3, help="OneCycle peak learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers; keep 0 on Windows")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    ensure_dirs()
    _, metrics = train_image_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        save=not args.no_save,
    )

    print("\n" + "=" * 52)
    print("  IMAGE DETECTOR - held-out test results")
    print("=" * 52)
    print(f"  test samples     : {metrics['n_test']:,}")
    print(f"  accuracy         : {metrics['accuracy']:.4f}")
    print(f"  precision (AI)   : {metrics['precision']:.4f}")
    print(f"  recall    (AI)   : {metrics['recall']:.4f}")
    print(f"  F1        (AI)   : {metrics['f1']:.4f}")
    print(f"  ROC-AUC          : {metrics['roc_auc']:.4f}")
    print(f"  false-positive   : {metrics['false_positive_rate']:.4f}"
          "   (real photo wrongly flagged as AI)")
    print(f"  parameters       : {metrics['parameters']:,}")
    print(f"  trained on       : {metrics['device']}")
    print("=" * 52)
    print("\nNext:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()

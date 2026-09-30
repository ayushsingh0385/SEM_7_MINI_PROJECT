"""Download and cache the CIFAKE real-vs-AI image dataset (Week 3).

    python scripts/prepare_image_data.py                 # 60k train / 10k test
    python scripts/prepare_image_data.py --max-train 0   # all 100k
    python scripts/prepare_image_data.py --force         # rebuild the cache
"""
import argparse

import _bootstrap  # noqa: F401

from mmdetect.config import ensure_dirs
from mmdetect.data.image_data import prepare_image_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-train", type=int, default=60_000,
                   help="cap on balanced training images; 0 means all 100k")
    p.add_argument("--max-test", type=int, default=10_000,
                   help="cap on balanced test images; 0 means all 20k")
    p.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = p.parse_args()

    ensure_dirs()
    prepare_image_dataset(
        max_train_rows=args.max_train or None,
        max_test_rows=args.max_test or None,
        force=args.force,
    )
    print("\nImage data ready. Next:  python scripts/train_image.py")


if __name__ == "__main__":
    main()

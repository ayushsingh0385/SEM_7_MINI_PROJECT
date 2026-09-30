"""Download and cache the human-vs-AI text dataset (Week 2).

    python scripts/prepare_text_data.py                  # HC3, normalised
    python scripts/prepare_text_data.py --force          # rebuild the cache

Two sources, and the choice matters:

    --source hc3      (default) HC3 Human-ChatGPT Comparison Corpus. Real text
                      with real punctuation and casing.
    --source essays   "AI Vs Human Text". Lowercased and stripped of ALL
                      punctuation in both classes, so a model trained on it
                      scores ~99.5% and then fails on real input. Available for
                      the comparison in the report, not for a usable model.

    --no-normalise    Skip the detokenisation fix. HC3's human answers are
                      detokenised ("like this , here .") in 88% of cases against
                      0.3% of the ChatGPT answers, so skipping the fix lets the
                      model learn that shortcut instead of writing style. Useful
                      only to demonstrate the inflated number.
"""
import argparse

import _bootstrap  # noqa: F401

from mmdetect.config import ensure_dirs
from mmdetect.data.text_data import prepare_text_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["hc3", "essays"], default="hc3")
    p.add_argument("--max-train", type=int, default=40_000,
                   help="cap on balanced training rows; 0 means no cap")
    p.add_argument("--max-test", type=int, default=8_000,
                   help="cap on balanced test rows; 0 means no cap")
    p.add_argument("--no-normalise", action="store_true",
                   help="skip the detokenisation fix (inflates accuracy)")
    p.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = p.parse_args()

    ensure_dirs()
    prepare_text_dataset(
        source=args.source,
        max_train_rows=args.max_train or None,
        max_test_rows=args.max_test or None,
        normalise=not args.no_normalise,
        force=args.force,
    )
    print("\nText data ready. Next:  python scripts/train_text.py")


if __name__ == "__main__":
    main()

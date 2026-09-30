"""Central configuration: paths, label convention and dataset sources.

Everything in the project imports paths from here so that no module ever
hard-codes a directory. Import this before anything else.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# config.py -> mmdetect -> src -> project root
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT / "models"
TEXT_MODEL_DIR = MODELS_DIR / "text"
IMAGE_MODEL_DIR = MODELS_DIR / "image"

REPORTS_DIR = ROOT / "reports"
TEXT_REPORT_DIR = REPORTS_DIR / "text"
IMAGE_REPORT_DIR = REPORTS_DIR / "image"

# Saved artifacts
TEXT_MODEL_PATH = TEXT_MODEL_DIR / "text_detector.joblib"
TEXT_METRICS_PATH = TEXT_REPORT_DIR / "metrics.json"

IMAGE_MODEL_PATH = IMAGE_MODEL_DIR / "image_detector.pt"
IMAGE_METRICS_PATH = IMAGE_REPORT_DIR / "metrics.json"

# Processed dataset files
TEXT_TRAIN_PARQUET = PROCESSED_DIR / "text_train.parquet"
TEXT_TEST_PARQUET = PROCESSED_DIR / "text_test.parquet"
IMAGE_TRAIN_NPZ = PROCESSED_DIR / "image_train.npz"
IMAGE_TEST_NPZ = PROCESSED_DIR / "image_test.npz"


def ensure_dirs() -> None:
    """Create every directory the pipeline writes into."""
    for d in (
        RAW_DIR, PROCESSED_DIR,
        TEXT_MODEL_DIR, IMAGE_MODEL_DIR,
        TEXT_REPORT_DIR, IMAGE_REPORT_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Label convention  --  THE single rule for the whole project
# --------------------------------------------------------------------------
# 1 = AI-generated / synthetic
# 0 = human-written / real photograph
#
# Source datasets do NOT agree with each other, so each loader normalises to
# this convention exactly once, at load time:
#
#   * AI-human-text  ships `generated`: 1 = AI      -> already correct, kept
#   * CIFAKE         ships `label`:     0 = FAKE,
#                                       1 = REAL    -> INVERTED, must be flipped
#
# Getting this wrong silently trains a perfectly accurate *backwards* model,
# so the flip lives in one place (data/image_data.py) and is asserted in tests.
LABEL_AI = 1
LABEL_HUMAN = 0
LABEL_NAMES = {LABEL_HUMAN: "Human / Real", LABEL_AI: "AI-generated"}

# Probability above which we call something AI-generated.
DECISION_THRESHOLD = 0.5
# Band around the threshold where the model is not confident enough to commit.
UNCERTAIN_MARGIN = 0.15

RANDOM_SEED = 42


# --------------------------------------------------------------------------
# Dataset sources
# --------------------------------------------------------------------------
# Text, ALTERNATIVE source only. The default is HC3, configured in
# data/text_data.py; see the long note there for why.
#
# This is the HuggingFace mirror of the Kaggle "AI Vs Human Text" dataset
# (Kaggle equivalent: shanegerami/ai-vs-human-text). Every document in it, in
# both classes, is lowercased and stripped of punctuation, so a model trained
# on it does not work on real input. Reachable via --source essays, kept for
# the comparison written up in the README.
TEXT_HF_REPO = "andythetechnerd03/AI-human-text"
TEXT_HF_TRAIN_SHARDS = [
    "data/train-00000-of-00003.parquet",
    "data/train-00001-of-00003.parquet",
    "data/train-00002-of-00003.parquet",
]
TEXT_HF_TEST_FILE = "data/test-00000-of-00001.parquet"
TEXT_KAGGLE_DATASET = "shanegerami/ai-vs-human-text"

# Image: HuggingFace mirror of the Kaggle CIFAKE dataset
# (60k real CIFAR-10 photos + 60k Stable-Diffusion-generated equivalents).
# Kaggle equivalent: birdy654/cifake-real-and-ai-generated-synthetic-images
IMAGE_HF_REPO = "dragonintelligence/CIFAKE-image-dataset"
IMAGE_HF_TRAIN_FILE = "data/train-00000-of-00001.parquet"
IMAGE_HF_TEST_FILE = "data/test-00000-of-00001.parquet"
IMAGE_KAGGLE_DATASET = "birdy654/cifake-real-and-ai-generated-synthetic-images"

IMAGE_SIZE = 32  # CIFAKE is natively 32x32

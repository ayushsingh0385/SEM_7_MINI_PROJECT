"""Small shared helpers: logging, JSON IO, seeding, HuggingFace downloads."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from .config import RANDOM_SEED

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a console logger, configured once per process."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed python / numpy / torch so training runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def download_hf_file(repo_id: str, filename: str, dest_dir: Path) -> Path:
    """Download one file from a HuggingFace dataset repo into ``dest_dir``.

    Uses the local cache, so re-running the prepare scripts is cheap. Public
    datasets need no token; a private mirror picks one up from the standard
    HF_TOKEN environment variable.
    """
    from huggingface_hub import hf_hub_download

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("download")
    log.info("fetching %s : %s", repo_id, filename)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(dest_dir),
        token=os.environ.get("HF_TOKEN"),
    )
    return Path(path)


def download_kaggle_dataset(dataset: str, dest_dir: Path) -> Path:
    """Download a Kaggle dataset (requires ~/.kaggle/kaggle.json credentials).

    Offered as an alternative source; the HuggingFace mirrors are the default
    because they need no credentials.
    """
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "kagglehub is not installed. Run `pip install kagglehub`, or use the "
            "default HuggingFace source which needs no credentials."
        ) from exc

    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists() and not os.environ.get("KAGGLE_USERNAME"):
        raise RuntimeError(
            "No Kaggle credentials found. Create an API token at "
            "https://www.kaggle.com/settings and save it to "
            f"{cred}, or use the default HuggingFace source."
        )

    get_logger("download").info("fetching kaggle dataset %s", dataset)
    return Path(kagglehub.dataset_download(dataset))

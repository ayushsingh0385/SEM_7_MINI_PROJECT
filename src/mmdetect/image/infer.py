"""Serving-time wrapper around the trained CIFAKE image detector.

The honest problem this file solves
-----------------------------------
The model is trained on 32x32 CIFAKE images, but people upload 4000x3000
photos. Downscaling a big photo straight to 32x32 throws away most of the
high-frequency evidence the network learned, so a single resize is a weak
prediction on real uploads.

So inference uses several *views* of the same image and combines them:

  global  - the whole image resized to 32x32. This is the closest match to how
            CIFAKE itself was built (CIFAR-10 is downsampled photographs), so
            it carries the most weight.
  tiles   - the image resized so its short side is 96px, then cut into 32x32
            tiles. These see local texture at a scale nearer to training than a
            raw native-resolution crop would be.

The per-view probabilities and their spread are returned in the explanation, so
the UI can show *how much the views agreed* rather than hiding the uncertainty
behind one number. Disagreement between views is reported as low confidence.

This is a genuine domain shift and it is documented in the README: accuracy on
arbitrary web images is materially lower than the 32x32 test-set number.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from ..config import IMAGE_METRICS_PATH, IMAGE_MODEL_PATH, IMAGE_SIZE
from ..result import DetectionResult
from ..utils import get_logger, load_json
from .model import CifakeCNN, get_device, normalize_batch

log = get_logger("image.infer")

TILE_SHORT_SIDE = 96      # resize target before tiling
MAX_TILES = 16            # cap the work on huge images
GLOBAL_WEIGHT = 0.5       # global view vs. the averaged tile view


def load_image(source) -> Image.Image:
    """Open anything Streamlit hands us as an upright RGB PIL image."""
    img = source if isinstance(source, Image.Image) else Image.open(source)
    # Honour EXIF orientation, otherwise phone photos arrive rotated.
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _global_view(img: Image.Image) -> np.ndarray:
    arr = np.asarray(
        img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC), dtype=np.uint8
    )
    return arr[None, ...]


def _tile_views(img: Image.Image) -> np.ndarray:
    """Resize so the short side is TILE_SHORT_SIDE, then cut 32x32 tiles."""
    w, h = img.size
    if min(w, h) < IMAGE_SIZE:
        return np.empty((0, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    scale = TILE_SHORT_SIDE / min(w, h)
    # Never upscale a small image: that invents detail the model would read
    # as smoothing, which biases it toward "AI".
    scale = min(scale, 1.0)
    new_w, new_h = max(IMAGE_SIZE, int(w * scale)), max(IMAGE_SIZE, int(h * scale))
    resized = np.asarray(img.resize((new_w, new_h), Image.BICUBIC), dtype=np.uint8)

    tiles = []
    for y in range(0, new_h - IMAGE_SIZE + 1, IMAGE_SIZE):
        for x in range(0, new_w - IMAGE_SIZE + 1, IMAGE_SIZE):
            tiles.append(resized[y:y + IMAGE_SIZE, x:x + IMAGE_SIZE])
            if len(tiles) >= MAX_TILES:
                break
        if len(tiles) >= MAX_TILES:
            break

    if not tiles:
        return np.empty((0, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    return np.stack(tiles)


class ImageDetector:
    """Lazy-loading wrapper around the trained CNN."""

    modality = "image"

    def __init__(self, model_path: Path | None = None):
        self.model_path = Path(model_path or IMAGE_MODEL_PATH)
        self._model: CifakeCNN | None = None
        self._device = None
        self._checkpoint_meta: dict[str, Any] = {}

    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        return self.model_path.exists()

    def load(self) -> CifakeCNN:
        """Load weights onto the best available device (idempotent)."""
        if self._model is not None:
            return self._model
        if not self.is_available:
            raise FileNotFoundError(
                f"No trained image model at {self.model_path}.\n"
                "Train one with:  python scripts/train_image.py"
            )

        self._device = get_device()
        log.info("loading image model from %s onto %s",
                 self.model_path.name, self._device)
        ckpt = torch.load(self.model_path, map_location=self._device, weights_only=False)

        model = CifakeCNN()
        model.load_state_dict(ckpt["state_dict"])
        model.to(self._device).eval()

        self._model = model
        self._checkpoint_meta = {
            k: ckpt.get(k) for k in ("model_name", "image_size", "metrics",
                                     "label_convention")
        }
        return model

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _probs_for(self, batch: np.ndarray) -> np.ndarray:
        """P(AI) for a batch of (N, 32, 32, 3) uint8 images."""
        if len(batch) == 0:
            return np.empty(0, dtype=np.float32)
        model = self.load()
        x = normalize_batch(batch).to(self._device)
        logits = model(x)
        return torch.sigmoid(logits).float().cpu().numpy()

    def score_batch(self, images: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """P(AI) for an array of (N, 32, 32, 3) uint8 images, chunked.

        Used for bulk evaluation against the native-resolution test split,
        where the multi-view machinery in :meth:`predict` does not apply.
        """
        out = [
            self._probs_for(images[i:i + batch_size])
            for i in range(0, len(images), batch_size)
        ]
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)

    def predict(self, source) -> DetectionResult:
        """Classify one image (a path, file-like object or PIL image)."""
        self.load()
        img = load_image(source)
        width, height = img.size

        global_p = float(self._probs_for(_global_view(img))[0])

        tiles = _tile_views(img)
        tile_probs = self._probs_for(tiles)

        if len(tile_probs):
            tile_mean = float(tile_probs.mean())
            combined = GLOBAL_WEIGHT * global_p + (1 - GLOBAL_WEIGHT) * tile_mean
            spread = float(tile_probs.std())
        else:
            # Image too small to tile: the global view is all we have.
            tile_mean, spread = float("nan"), 0.0
            combined = global_p

        # An image already at training resolution needs none of this machinery.
        native = width == height == IMAGE_SIZE
        if native:
            combined = global_p

        explanation = {
            "global_probability": global_p,
            "tile_probability": None if np.isnan(tile_mean) else tile_mean,
            "n_tiles": int(len(tile_probs)),
            "tile_spread": spread,
            "tile_probabilities": [float(p) for p in tile_probs[:MAX_TILES]],
            # Views disagreeing is the clearest sign the answer is shaky.
            "views_agree": bool(spread < 0.2) if len(tile_probs) else True,
            "native_resolution": native,
        }

        return DetectionResult(
            ai_probability=float(combined),
            modality="image",
            model_name="CIFAKE CNN (multi-view)",
            explanation=explanation,
            meta={
                "width": width,
                "height": height,
                "megapixels": round(width * height / 1e6, 2),
                "upscaled_from_small": min(width, height) < IMAGE_SIZE,
                # Flagged so the UI can warn about the 32x32 domain shift.
                "far_from_training_resolution": min(width, height) > 256,
            },
        )

    # ------------------------------------------------------------------
    def training_metrics(self) -> dict[str, Any] | None:
        if IMAGE_METRICS_PATH.exists():
            try:
                return load_json(IMAGE_METRICS_PATH)
            except Exception:
                return None
        return None


_DETECTOR: ImageDetector | None = None


def get_image_detector() -> ImageDetector:
    """Process-wide singleton, so Streamlit reruns do not reload the weights."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = ImageDetector()
    return _DETECTOR


__all__ = ["ImageDetector", "get_image_detector", "load_image"]

"""One place that knows about every modality, built or not yet built.

The Streamlit app renders itself entirely from this registry, so adding the
audio detector in week 4 means implementing a detector and flipping its
``status`` here - no UI surgery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .config import IMAGE_MODEL_PATH, TEXT_MODEL_PATH


@dataclass(frozen=True)
class ModalityInfo:
    """Everything the UI needs to know about one content type."""

    key: str
    label: str
    icon: str
    week: int
    status: str                      # "ready" | "planned"
    extensions: list[str]
    description: str
    dataset: str = ""
    approach: str = ""
    planned_approach: str = ""
    _loader: Callable[[], Any] | None = field(default=None, repr=False)

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"

    @property
    def is_trained(self) -> bool:
        """True when a trained artifact actually exists on disk."""
        if self.key == "text":
            return TEXT_MODEL_PATH.exists()
        if self.key == "image":
            return IMAGE_MODEL_PATH.exists()
        return False

    def detector(self):
        """Return the detector instance, or explain why it is unavailable."""
        if self._loader is None:
            raise NotImplementedError(
                f"{self.label} detection is scheduled for week {self.week} "
                "and is not implemented yet."
            )
        return self._loader()


def _text_loader():
    from .text.infer import get_text_detector

    return get_text_detector()


def _image_loader():
    from .image.infer import get_image_detector

    return get_image_detector()


MODALITIES: dict[str, ModalityInfo] = {
    "text": ModalityInfo(
        key="text",
        label="Text",
        icon="\U0001F4DD",  # memo
        week=2,
        status="ready",
        extensions=["txt", "md", "csv"],
        description="Detect whether an essay, article or answer was written by a human or produced by a language model.",
        dataset="HC3 Human-ChatGPT Comparison Corpus - 40k balanced answers, detokenisation artefacts normalised out.",
        approach="Word + character TF-IDF combined with 24 stylometric features, classified by logistic regression.",
        _loader=_text_loader,
    ),
    "image": ModalityInfo(
        key="image",
        label="Image",
        icon="\U0001F5BC",  # framed picture
        week=3,
        status="ready",
        extensions=["png", "jpg", "jpeg", "bmp", "webp"],
        description="Detect whether a photograph is real or was produced by an image generator such as Stable Diffusion.",
        dataset="CIFAKE - 60k real CIFAR-10 photos vs 60k Stable Diffusion images",
        approach="A compact VGG-style CNN over 32x32 inputs, applied to several views of the upload and averaged.",
        _loader=_image_loader,
    ),
    "audio": ModalityInfo(
        key="audio",
        label="Audio",
        icon="\U0001F3B5",  # musical note
        week=4,
        status="planned",
        extensions=["wav", "mp3", "flac", "ogg"],
        description="Detect synthetic speech and voice clones.",
        planned_approach="Mel-spectrogram CNN trained on the ASVspoof / WaveFake corpora, reusing this same detector interface.",
    ),
    "video": ModalityInfo(
        key="video",
        label="Video",
        icon="\U0001F3AC",  # clapper board
        week=5,
        status="planned",
        extensions=["mp4", "avi", "mov", "mkv"],
        description="Detect deepfaked and fully generated video.",
        planned_approach="Per-frame scoring with the image detector, plus a temporal consistency model over frame embeddings, trained on FaceForensics++ / DFDC.",
    ),
}


def get_modality(key: str) -> ModalityInfo:
    if key not in MODALITIES:
        raise KeyError(f"Unknown modality '{key}'. Known: {list(MODALITIES)}")
    return MODALITIES[key]


def get_detector(key: str):
    """Convenience: registry lookup plus detector construction."""
    return get_modality(key).detector()


def ready_modalities() -> list[ModalityInfo]:
    return [m for m in MODALITIES.values() if m.is_ready]


def project_status() -> dict[str, Any]:
    """Week-by-week status, used by the app's sidebar."""
    return {
        key: {
            "week": m.week,
            "status": m.status,
            "model_trained": m.is_trained,
        }
        for key, m in MODALITIES.items()
    }


__all__ = [
    "ModalityInfo", "MODALITIES", "get_modality", "get_detector",
    "ready_modalities", "project_status",
]

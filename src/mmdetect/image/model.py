"""Week 3: CNN architecture for real-vs-AI image detection.

A compact VGG-style network sized for CIFAKE's 32x32 inputs. Design notes:

  * Small 3x3 kernels stacked two-deep per block. Diffusion models leave
    high-frequency fingerprints (upsampling artefacts, unnatural local noise),
    and small early kernels at full resolution are what pick those up.
  * BatchNorm after every conv so the net trains in ~15 epochs on a laptop GPU.
  * Global average pooling instead of a big flatten -> ~296k parameters rather
    than several million, which keeps it from memorising 60k images.
  * A single logit output with BCEWithLogitsLoss; the sigmoid of that logit is
    P(AI-generated) and feeds the UI directly.

Normalisation uses CIFAR-10 statistics, since CIFAKE's real half *is* CIFAR-10
and the synthetic half was generated to match it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# CIFAR-10 channel statistics (the dataset CIFAKE is built on).
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two 3x3 convolutions at one resolution, then halve the spatial size."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CifakeCNN(nn.Module):
    """32x32 RGB -> one logit, where sigmoid(logit) = P(AI-generated)."""

    def __init__(self, dropout: float = 0.3, width: int = 32) -> None:
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _block(3, w),          # 32 -> 16
            _block(w, w * 2),      # 16 -> 8
            _block(w * 2, w * 4),  # 8  -> 4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(w * 4, w * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(w * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x).squeeze(1)  # (B,) logits

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def normalize_batch(x_uint8: np.ndarray | torch.Tensor) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> normalised (N, 3, H, W) float tensor.

    The single shared conversion used by training, evaluation and inference, so
    serving can never drift from training preprocessing.
    """
    if isinstance(x_uint8, np.ndarray):
        arr = np.ascontiguousarray(x_uint8)
        # Arrays sliced out of a memory-mapped .npz are read-only, and
        # torch.from_numpy warns on those; copy only in that case.
        if not arr.flags.writeable:
            arr = arr.copy()
        x = torch.from_numpy(arr)
    else:
        x = x_uint8
    if x.ndim == 3:            # a single image
        x = x.unsqueeze(0)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


__all__ = ["CifakeCNN", "normalize_batch", "get_device", "MEAN", "STD"]

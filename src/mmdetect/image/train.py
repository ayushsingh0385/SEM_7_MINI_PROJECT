"""Week 3: train the real-vs-AI image detector on CIFAKE.

Training choices worth knowing:

  * Augmentation is deliberately mild - horizontal flip and a 4px shift only.
    Blur, JPEG re-encoding or resizing would destroy the high-frequency
    generator fingerprints that the model is supposed to key on, so anything
    that touches the frequency content is left out on purpose.
  * OneCycle LR + AdamW converges in ~15 epochs, a few minutes on a laptop GPU.
  * Mixed precision when CUDA is present, plain float32 on CPU.
  * A 10% validation split carved out of train picks the best epoch; the test
    split is touched exactly once, at the end.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

from ..config import (
    IMAGE_METRICS_PATH,
    IMAGE_MODEL_PATH,
    IMAGE_REPORT_DIR,
    RANDOM_SEED,
)
from ..utils import get_logger, save_json, set_seed
from .model import MEAN, STD, CifakeCNN, get_device, normalize_batch

log = get_logger("image.train")

MODEL_NAME = "cifake-cnn"


class CifakeDataset(Dataset):
    """Wraps the cached uint8 arrays, with optional light augmentation."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.images = images
        self.labels = labels.astype(np.float32)
        self.augment = augment
        self.rng = np.random.default_rng(RANDOM_SEED)

    def __len__(self) -> int:
        return len(self.labels)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() < 0.5:
            img = img[:, ::-1]                      # horizontal flip
        # 4px reflect-pad then random crop back to 32x32 (a small translation)
        pad = 4
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        y = int(self.rng.integers(0, 2 * pad))
        x = int(self.rng.integers(0, 2 * pad))
        h, w = img.shape[:2]
        return padded[y:y + h, x:x + w]

    def __getitem__(self, idx: int):
        img = self.images[idx]
        if self.augment:
            img = self._augment(img)
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), torch.tensor(self.labels[idx])


def _to_device_batch(imgs: torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 (B,H,W,C) on CPU -> normalised (B,C,H,W) float on ``device``."""
    return normalize_batch(imgs.to(device, non_blocking=True))


@torch.no_grad()
def _predict_proba(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, targets = [], []
    for imgs, ys in loader:
        x = _to_device_batch(imgs, device)
        logits = model(x)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        targets.append(ys.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def _metrics_from(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = (proba >= 0.5).astype(int)
    y_true = y_true.astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "confusion_matrix": {
            "true_human": {"pred_human": int(tn), "pred_ai": int(fp)},
            "true_ai": {"pred_human": int(fn), "pred_ai": int(tp)},
        },
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "report": classification_report(
            y_true, pred, target_names=["Real", "AI"], output_dict=True, zero_division=0
        ),
    }


def _save_plots(y_true, proba, history) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    IMAGE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pred = (proba >= 0.5).astype(int)

    cm = confusion_matrix(y_true.astype(int), pred)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks([0, 1], ["Real", "AI"])
    ax.set_yticks([0, 1], ["Real", "AI"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Image detector - confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(IMAGE_REPORT_DIR / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true.astype(int), proba)
    auc = roc_auc_score(y_true.astype(int), proba)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, lw=2, color="#6c3483", label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Image detector - ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(IMAGE_REPORT_DIR / "roc_curve.png", dpi=140)
    plt.close(fig)

    if history:
        ep = [h["epoch"] for h in history]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
        a1.plot(ep, [h["train_loss"] for h in history], label="train")
        a1.plot(ep, [h["val_loss"] for h in history], label="val")
        a1.set_xlabel("epoch"); a1.set_ylabel("BCE loss"); a1.legend(); a1.set_title("Loss")
        a2.plot(ep, [h["val_acc"] for h in history], color="#2e8b57")
        a2.set_xlabel("epoch"); a2.set_ylabel("accuracy"); a2.set_title("Validation accuracy")
        fig.tight_layout()
        fig.savefig(IMAGE_REPORT_DIR / "training_curves.png", dpi=140)
        plt.close(fig)


def train_image_model(
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.1,
    num_workers: int = 0,
    save: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train, evaluate and persist the CIFAKE image detector."""
    from ..data.image_data import load_image_dataset

    set_seed()
    (x_train, y_train), (x_test, y_test) = load_image_dataset()
    log.info("train=%s  test=%s", x_train.shape, x_test.shape)
    log.info("train balance: AI=%d real=%d",
             int((y_train == 1).sum()), int((y_train == 0).sum()))

    # Stratified validation split carved out of train.
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(y_train))
    n_val = int(len(idx) * val_fraction)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    train_ds = CifakeDataset(x_train[tr_idx], y_train[tr_idx], augment=True)
    val_ds = CifakeDataset(x_train[val_idx], y_train[val_idx], augment=False)
    test_ds = CifakeDataset(x_test, y_test, augment=False)

    dl = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **dl)
    val_dl = DataLoader(val_ds, shuffle=False, **dl)
    test_dl = DataLoader(test_ds, shuffle=False, **dl)

    device = get_device()
    model = CifakeCNN().to(device)
    log.info("device=%s  parameters=%s", device, f"{model.n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_dl)
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, float]] = []
    best_val_acc, best_state = 0.0, None
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for imgs, ys in train_dl:
            x = _to_device_batch(imgs, device)
            y = ys.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item() * len(y)
            seen += len(y)

        train_loss = running / max(seen, 1)

        val_proba, val_y = _predict_proba(model, val_dl, device)
        val_loss = float(nn.functional.binary_cross_entropy(
            torch.from_numpy(val_proba).clamp(1e-6, 1 - 1e-6), torch.from_numpy(val_y)
        ))
        val_acc = float(accuracy_score(val_y.astype(int), (val_proba >= 0.5).astype(int)))

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_acc": val_acc})
        log.info("epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
                 epoch, epochs, train_loss, val_loss, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_secs = time.perf_counter() - t0
    log.info("trained in %.1fs, best val acc %.4f", train_secs, best_val_acc)

    if best_state is not None:
        model.load_state_dict(best_state)

    # The test split is evaluated exactly once, here.
    test_proba, test_y = _predict_proba(model, test_dl, device)
    metrics = _metrics_from(test_y, test_proba)
    metrics.update({
        "model": MODEL_NAME,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_y)),
        "best_val_accuracy": best_val_acc,
        "train_seconds": round(train_secs, 1),
        "parameters": model.n_params,
        "device": str(device),
        "hyperparams": {"epochs": epochs, "batch_size": batch_size,
                        "lr": lr, "weight_decay": weight_decay},
        "history": history,
    })

    log.info("TEST accuracy=%.4f  f1=%.4f  roc_auc=%.4f  FPR=%.4f",
             metrics["accuracy"], metrics["f1"], metrics["roc_auc"],
             metrics["false_positive_rate"])

    if save:
        IMAGE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "arch": "CifakeCNN",
            "model_name": MODEL_NAME,
            "image_size": x_train.shape[1],
            "mean": MEAN,
            "std": STD,
            # Recorded so inference can never disagree about polarity.
            "label_convention": {"1": "AI-generated", "0": "real"},
            "metrics": {k: metrics[k] for k in
                        ("accuracy", "f1", "roc_auc", "false_positive_rate")},
        }, IMAGE_MODEL_PATH)
        log.info("saved model -> %s (%.1f MB)", IMAGE_MODEL_PATH,
                 IMAGE_MODEL_PATH.stat().st_size / 1e6)

        save_json(metrics, IMAGE_METRICS_PATH)
        _save_plots(test_y, test_proba, history)
        log.info("saved metrics + figures -> %s", IMAGE_REPORT_DIR)

    return model, metrics


# Public alias: other modules (scripts/evaluate.py) score predictions with the
# exact same metric definitions used during training.
compute_metrics = _metrics_from

__all__ = ["train_image_model", "CifakeDataset", "compute_metrics", "MODEL_NAME"]

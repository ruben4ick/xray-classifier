"""
Evaluation and visualization for the xray-classifier experiment.

Responsibilities:
- Load best checkpoints and run inference on the test set
- Compute metrics: accuracy, precision, recall, F1 (macro), AUC-ROC
- Generate confusion matrices, learning curves, Grad-CAM visualizations
- Produce a side-by-side comparison of pretrained vs scratch models
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from src.data import CLASSES, IMAGENET_MEAN, IMAGENET_STD, get_loaders
from src.models import get_model

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"

MODEL_NAMES = ("pretrained", "scratch")
COLORS = {"pretrained": "steelblue", "scratch": "coral"}


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device) -> tuple:
    """Return (labels, preds, probs) numpy arrays for the whole loader.

    probs: probability of the PNEUMONIA class (index 1), shape (N,).
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_labels.append(labels.numpy())
        all_preds.append(preds)
        all_probs.append(probs)
    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    """Return dict with accuracy, precision, recall, f1 (macro), auc_roc."""
    fpr, tpr, _ = roc_curve(labels, probs)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall":    recall_score(labels, preds, average="macro", zero_division=0),
        "f1":        f1_score(labels, preds, average="macro", zero_division=0),
        "auc_roc":   auc(fpr, tpr),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(results: dict) -> None:
    """1×2 grid of confusion matrices for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, data) in zip(axes, results.items()):
        cm = confusion_matrix(data["labels"], data["preds"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            ax=ax,
        )
        acc = data["metrics"]["accuracy"]
        ax.set_title(f"{name}  (acc={acc:.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    _save(fig, "confusion_matrices.png")


def plot_learning_curves() -> None:
    """Loss and accuracy curves for both models on the same figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for name, color in COLORS.items():
        history_path = CHECKPOINTS_DIR / f"{name}_history.json"
        if not history_path.exists():
            print(f"[eval] history not found: {history_path} — skipping")
            continue
        with open(history_path) as f:
            h = json.load(f)
        epochs = range(1, len(h["train_loss"]) + 1)

        axes[0].plot(epochs, h["train_loss"], color=color, linestyle="--",
                     alpha=0.7, label=f"{name} train")
        axes[0].plot(epochs, h["val_loss"],   color=color, linestyle="-",
                     label=f"{name} val")
        axes[1].plot(epochs, h["val_acc"],    color=color, linestyle="-",
                     label=f"{name} val")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()

    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    _save(fig, "learning_curves.png")


def plot_roc_curves(results: dict) -> None:
    """ROC curves for both models on the same axes."""
    fig, ax = plt.subplots(figsize=(6, 6))

    for name, data in results.items():
        fpr, tpr, _ = roc_curve(data["labels"], data["probs"])
        roc_auc = data["metrics"]["auc_roc"]
        ax.plot(fpr, tpr, color=COLORS[name], lw=2,
                label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right")

    plt.tight_layout()
    _save(fig, "roc_curves.png")


def print_comparison_table(results: dict) -> None:
    """Print formatted metric table to stdout."""
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    col_w = 14
    header = f"{'Metric':<12}" + "".join(f"{n:>{col_w}}" for n in results)
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for m in metrics:
        row = f"{m:<12}" + "".join(
            f"{data['metrics'][m]:>{col_w}.4f}" for data in results.values()
        )
        print(row)
    print(sep + "\n")


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """Grad-CAM for ResNet18: hooks into layer4[-1]."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        target = model.layer4[-1]
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output):
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, image_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Return Grad-CAM heatmap (H×W, values in [0,1]).

        Args:
            image_tensor: shape (1, C, H, W), on the correct device.
            class_idx:    target class index; defaults to argmax.
        """
        self.model.eval()
        self.model.zero_grad()

        # torch.enable_grad() ensures backward works even inside no_grad scopes
        # and even for models with frozen parameters (gradients still flow through).
        with torch.enable_grad():
            img = image_tensor.clone().requires_grad_(True)
            logits = self.model(img)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            logits[0, class_idx].backward()

        # Global average pool the gradients → weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1).squeeze()    # (H, W)
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()


def plot_gradcam(models_dict: dict, loader, device: torch.device, n_images: int = 4) -> None:
    """Grid of Grad-CAM overlays: rows = models, cols = test images."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    images, labels = next(iter(loader))
    images = images[:n_images]
    labels = labels[:n_images]

    model_names = list(models_dict.keys())
    n_rows = len(model_names)

    fig, axes = plt.subplots(n_rows, n_images, figsize=(n_images * 3, n_rows * 3 + 0.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, name in enumerate(model_names):
        cam_extractor = GradCAM(models_dict[name])

        for col in range(n_images):
            img_tensor = images[col].unsqueeze(0).to(device)
            cam = cam_extractor(img_tensor)

            # Unnormalize for display
            raw = (images[col] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

            # Upsample cam to image resolution
            cam_up = F.interpolate(
                torch.tensor(cam).unsqueeze(0).unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()

            ax = axes[row, col]
            ax.imshow(raw)
            ax.imshow(cam_up, cmap="jet", alpha=0.45)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(name, fontsize=11, labelpad=6)
            if row == 0:
                ax.set_title(CLASSES[labels[col].item()], fontsize=10)

    plt.suptitle("Grad-CAM — Test Set", fontsize=13)
    plt.tight_layout()
    _save(fig, "gradcam.png")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] saved {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def evaluate_all(batch_size: int = 16) -> dict:
    """Load both best checkpoints, evaluate on test set, save all plots.

    Returns:
        dict mapping model name → {"labels", "preds", "probs", "metrics"}.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device: {device}")

    loaders = get_loaders(batch_size=batch_size)
    test_loader = loaders["test"]

    results: dict = {}
    models_dict: dict = {}

    for name in MODEL_NAMES:
        ckpt = CHECKPOINTS_DIR / f"{name}_best.pt"
        if not ckpt.exists():
            print(f"[eval] checkpoint not found: {ckpt} — skipping {name}")
            continue

        model = get_model(name, device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()

        labels, preds, probs = predict(model, test_loader, device)
        metrics = compute_metrics(labels, preds, probs)
        results[name] = {"labels": labels, "preds": preds, "probs": probs, "metrics": metrics}
        models_dict[name] = model

        print(f"[eval] {name}: " +
              "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    if not results:
        print("[eval] no checkpoints found — run train.py first")
        return results

    plot_confusion_matrices(results)
    plot_learning_curves()
    plot_roc_curves(results)
    print_comparison_table(results)
    plot_gradcam(models_dict, test_loader, device)

    return results


if __name__ == "__main__":
    evaluate_all()

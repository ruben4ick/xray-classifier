"""
Evaluation, metrics, and visualization for the xray-classifier experiment.

Functions:
  load_model            — restore model from best checkpoint
  get_predictions       — run inference, collect y_true / y_pred / y_prob
  compute_metrics       — accuracy, precision, recall, F1 (macro), AUC-ROC
  plot_confusion_matrix — save confusion matrix PNG
  plot_learning_curves  — save loss + accuracy curves PNG from saved history
  grad_cam              — Grad-CAM heatmap for a single image (numpy array)
  plot_grad_cam         — save Grad-CAM overlay grid PNG
  evaluate_model        — full pipeline for one model variant
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.data import RAW_DIR, eval_transform, get_loaders
from src.models import NUM_CLASSES, get_model

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"

CLASSES = ["NORMAL", "PNEUMONIA"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


# model loading
def load_model(name: str, device: torch.device | str = "cpu") -> nn.Module:
    """Load model architecture and restore best-checkpoint weights."""
    ckpt = CHECKPOINTS_DIR / f"{name}_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint found: {ckpt}")
    model = get_model(name, device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


# inference
@torch.no_grad()
def get_predictions(
    model: nn.Module, loader, device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_pred, y_prob) over the full loader.

    y_prob is the probability of the positive class (PNEUMONIA, index 1).
    """
    model.eval()
    y_true_list, y_pred_list, y_prob_list = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs  = F.softmax(logits, dim=1)
        y_true_list.extend(labels.numpy())
        y_pred_list.extend(logits.argmax(dim=1).cpu().numpy())
        y_prob_list.extend(probs[:, 1].cpu().numpy())
    return np.array(y_true_list), np.array(y_pred_list), np.array(y_prob_list)


# metrics
def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict[str, float]:
    """Return accuracy, precision, recall, F1 (macro), and AUC-ROC."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "auc_roc":   float(roc_auc_score(y_true, y_prob)),
    }


# plots
def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, name: str
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASSES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {name}")
    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    _save(fig, f"{name}_confusion_matrix.png")


def plot_learning_curves(name: str) -> None:
    history_path = CHECKPOINTS_DIR / f"{name}_history.json"
    with open(history_path) as f:
        h = json.load(f)

    epochs = range(1, len(h["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, h["train_loss"], label="train")
    ax1.plot(epochs, h["val_loss"],   label="val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss — {name}"); ax1.legend()

    best_acc = max(h["val_acc"])
    ax2.plot(epochs, h["val_acc"])
    ax2.axhline(best_acc, ls="--", color="gray", lw=0.8,
                label=f"best: {best_acc:.4f}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Val Accuracy — {name}"); ax2.legend()

    _save(fig, f"{name}_learning_curves.png")


# Grad-CAM
def grad_cam(
    model: nn.Module,
    image_tensor: torch.Tensor,  # (1, 3, 224, 224), on model device
    target_class: int,
) -> np.ndarray:
    """Return a (224, 224) Grad-CAM heatmap, values in [0, 1]."""
    saved: list = []  # will hold the layer4 output tensor

    def fwd_hook(_, __, output):
        # retain_grad() keeps the gradient on this non-leaf tensor after
        # backward(), even when backbone parameters are frozen.
        saved.append(output)
        output.retain_grad()

    h_fwd = model.layer4.register_forward_hook(fwd_hook)

    model.eval()
    model.zero_grad()
    with torch.enable_grad():
        inp = image_tensor.clone().requires_grad_(True)
        logits = model(inp)
        logits[0, target_class].backward()

    h_fwd.remove()

    act_t = saved[0]
    act  = act_t.detach().squeeze(0)  # (512, 7, 7)
    grad = act_t.grad.squeeze(0)       # (512, 7, 7)
    weights = grad.mean(dim=(1, 2))    # (512,)

    cam = F.relu((weights[:, None, None] * act).sum(dim=0))  # (7, 7)

    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()

    return (cam / cam.max()) if cam.max() > 0 else cam


def plot_grad_cam(model: nn.Module, name: str, device, n_samples: int = 4) -> None:
    """Save a grid of original + Grad-CAM overlay images from the test set."""
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(str(RAW_DIR / "test"), transform=eval_transform)

    # pick n_samples // NUM_CLASSES examples per class
    per_class = n_samples // NUM_CLASSES
    indices = []
    for cls_idx in range(NUM_CLASSES):
        cls_samples = [i for i, (_, l) in enumerate(dataset.samples) if l == cls_idx]
        indices.extend(cls_samples[:per_class])

    n = len(indices)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

    for col, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        inp = img_tensor.unsqueeze(0).to(device)
        cam = grad_cam(model, inp, target_class=label)

        img_display = (
            (img_tensor * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None])
            .permute(1, 2, 0)
            .numpy()
            .clip(0, 1)
        )

        axes[0, col].imshow(img_display)
        axes[0, col].set_title(CLASSES[label], fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(img_display)
        axes[1, col].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, col].set_title("Grad-CAM", fontsize=9)
        axes[1, col].axis("off")

    fig.suptitle(f"Grad-CAM — {name}", fontsize=13)
    _save(fig, f"{name}_grad_cam.png")


# full pipeline

def evaluate_model(name: str) -> dict[str, float]:
    """Run full evaluation for one model variant. Returns test metrics dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    model   = load_model(name, device)
    loaders = get_loaders()
    y_true, y_pred, y_prob = get_predictions(model, loaders["test"], device)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    print(f"\n[{name}] test metrics:")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    plot_confusion_matrix(y_true, y_pred, name)
    plot_learning_curves(name)
    plot_grad_cam(model, name, device)

    metrics_path = OUTPUTS_DIR / f"{name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: round(v, 6) for k, v in metrics.items()}, f, indent=2)
    print(f"[eval] saved {metrics_path}")

    return metrics


def _save(fig: plt.Figure, filename: str) -> None:
    path = OUTPUTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eval] saved {path}")


if __name__ == "__main__":
    results = {}
    for name in ("pretrained", "scratch"):
        results[name] = evaluate_model(name)

    # side-by-side comparison table
    print("\n=== Side-by-side comparison ===")
    header = f"{'metric':12s}  {'pretrained':>12s}  {'scratch':>10s}"
    print(header)
    print("-" * len(header))
    for metric in results["pretrained"]:
        p = results["pretrained"][metric]
        s = results["scratch"][metric]
        print(f"{metric:12s}  {p:>12.4f}  {s:>10.4f}")

"""
Evaluation, metrics, and visualization for the xray-classifier experiment.

Per-model outputs go to outputs/<model_name>/.
Combined comparison plots go to outputs/comparison/.

Functions:
  load_model            — restore model from best checkpoint
  get_predictions       — run inference, collect y_true / y_pred / y_prob
  compute_metrics       — accuracy, precision, recall, F1 (macro), AUC-ROC
  compute_val_stats     — mean, median, std, stability of val accuracy
  plot_confusion_matrix — save confusion matrix PNG (per-model)
  plot_learning_curves  — save loss + accuracy curves PNG (per-model)
  plot_grad_cam         — save Grad-CAM overlay grid PNG (per-model)
  plot_combined_curves  — overlay both models' curves (comparison)
  plot_roc_curves       — ROC curves for both models (comparison)
  plot_comparison_table — side-by-side metrics table image (comparison)
  grad_cam              — Grad-CAM heatmap for a single image (numpy array)
  evaluate_model        — full pipeline for one model variant
"""

from __future__ import annotations

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
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from src.data import RAW_DIR, eval_transform, get_loaders
from src.models import NUM_CLASSES, get_model

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"

CLASSES = ["NORMAL", "PNEUMONIA"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

sns.set_style("whitegrid")


def _model_dir(name: str) -> Path:
    """Return outputs/<name>/, creating it if needed."""
    d = OUTPUTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


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


def compute_val_stats(name: str) -> dict[str, float]:
    """Compute summary statistics for val accuracy across all epochs.

    Returns mean, median, std, min, max — more honest than reporting
    only the best epoch (which on a 16-image val set can be a fluke).
    """
    path = CHECKPOINTS_DIR / f"{name}_history.json"
    with open(path) as f:
        h = json.load(f)
    acc = np.array(h["val_acc"])
    median = float(np.median(acc))
    std = float(np.std(acc))
    # Stability score: how many std above chance (0.5) the median sits.
    # Like a Sharpe ratio — rewards high accuracy and penalizes volatility.
    stability = (median - 0.5) / std if std > 0 else 0.0
    return {
        "mean":      float(np.mean(acc)),
        "median":    median,
        "std":       std,
        "stability": round(stability, 4),
        "min":       float(np.min(acc)),
        "max":       float(np.max(acc)),
    }


def _moving_average(values: list[float], window: int = 5) -> np.ndarray:
    """Simple moving average for smoothing noisy curves."""
    arr = np.array(values)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="same")
    half = window // 2
    smoothed[:half] = arr[:half]
    smoothed[-half:] = arr[-half:]
    return smoothed


# plots
def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, name: str
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name} (Test Set, n={len(y_true)})")
    _save(fig, f"{name}/confusion_matrix.png")


def plot_learning_curves(name: str) -> None:
    """Per-model learning curves with moving-average trend and summary stats."""
    history_path = CHECKPOINTS_DIR / f"{name}_history.json"
    with open(history_path) as f:
        h = json.load(f)

    stats = compute_val_stats(name)
    epochs = range(1, len(h["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, h["train_loss"], label="train")
    ax1.plot(epochs, h["val_loss"],   label="val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss — {name}"); ax1.legend()

    val_acc = h["val_acc"]
    smoothed = _moving_average(val_acc, window=5)
    ax2.plot(epochs, val_acc, alpha=0.25, color="steelblue")
    ax2.plot(epochs, smoothed, color="steelblue", linewidth=2, label="val acc")
    ax2.axhline(stats["median"], ls="--", color="gray", lw=1,
                label=f"median: {stats['median']:.0%}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.set_title(f"Val Accuracy — {name}"); ax2.legend()

    _save(fig, f"{name}/learning_curves.png")


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


def plot_grad_cam(model: nn.Module, name: str, device, n_samples: int = 6) -> None:
    """Save a grid: top row = original X-rays, bottom row = Grad-CAM overlay."""
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

        pred = model(inp).argmax(dim=1).item()
        prob = F.softmax(model(inp), dim=1).max().item()
        cam_map = grad_cam(model, inp, target_class=label)

        img_display = (
            (img_tensor * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None])
            .permute(1, 2, 0)
            .numpy()
            .clip(0, 1)
        )

        axes[0, col].imshow(img_display)
        axes[0, col].set_title(f"True: {CLASSES[label]}", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(img_display)
        axes[1, col].imshow(cam_map, cmap="jet", alpha=0.45)
        axes[1, col].set_title(f"Pred: {CLASSES[pred]} ({prob:.0%})", fontsize=8)
        axes[1, col].axis("off")

    fig.suptitle(f"Grad-CAM — {name}", fontsize=13)
    _save(fig, f"{name}/gradcam.png")


def plot_combined_curves(histories: dict[str, dict]) -> None:
    """Overlay both models on one figure for direct comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = {"pretrained": "steelblue", "scratch": "coral"}

    for name, h in histories.items():
        epochs = range(1, len(h["train_loss"]) + 1)
        c = colors[name]
        stats = compute_val_stats(name)

        # loss
        ax1.plot(epochs, h["val_loss"], color=c, label=name)

        # accuracy — smoothed trend + median
        smoothed = _moving_average(h["val_acc"], window=5)
        ax2.plot(epochs, h["val_acc"], alpha=0.2, color=c)
        ax2.plot(epochs, smoothed, color=c, linewidth=2, label=name)
        ax2.axhline(stats["median"], ls="--", color=c, lw=0.8, alpha=0.6)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Val Loss")
    ax1.legend()

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Val Accuracy (dashed = median)")
    ax2.legend()

    _save(fig, "comparison/learning_curves.png")


def plot_roc_curves(all_results: dict[str, dict]) -> None:
    """ROC curves for both models on one plot. Kept simple."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for name, res in all_results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        auc = res["metrics"]["auc_roc"]
        ax.plot(fpr, tpr, label=f"{name} ({auc:.3f})")

    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(title="AUC")
    _save(fig, "comparison/roc_curve.png")


def plot_comparison_table(all_results: dict[str, dict]) -> None:
    """Render a side-by-side metrics table as an image, using robust stats."""
    # test set metrics
    metrics_order = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    labels = ["Accuracy", "Precision", "Recall", "F1 (macro)", "AUC-ROC"]

    rows = []
    for label, key in zip(labels, metrics_order):
        p = all_results["pretrained"]["metrics"][key]
        s = all_results["scratch"]["metrics"][key]
        winner = "Pretrained" if p > s else "Scratch" if s > p else "Tie"
        rows.append([label, f"{p:.4f}", f"{s:.4f}", winner])

    # val accuracy summary stats
    for name_label, stat_key in [("Val Acc (mean)", "mean"),
                                  ("Val Acc (median)", "median"),
                                  ("Stability", "stability")]:
        p = all_results["pretrained"]["val_stats"][stat_key]
        s = all_results["scratch"]["val_stats"][stat_key]
        winner = "Pretrained" if p > s else "Scratch" if s > p else "Tie"
        rows.append([name_label, f"{p:.4f}", f"{s:.4f}", winner])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Pretrained", "Scratch", "Winner"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    for i, row in enumerate(rows):
        cell = table[i + 1, 3]
        if row[3] == "Pretrained":
            cell.set_facecolor("#d4edda")
        elif row[3] == "Scratch":
            cell.set_facecolor("#f8d7da")

    ax.set_title("Model Comparison — Test Set", fontsize=13, pad=20)
    _save(fig, "comparison/metrics_table.png")


# full pipeline
def evaluate_model(name: str) -> dict:
    """Run full evaluation for one model variant. Returns results dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = _model_dir(name)

    model   = load_model(name, device)
    loaders = get_loaders()
    y_true, y_pred, y_prob = get_predictions(model, loaders["test"], device)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    val_stats = compute_val_stats(name)

    print(f"\n[{name}] test metrics:")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print(f"  val accuracy — mean: {val_stats['mean']:.4f}  "
          f"median: {val_stats['median']:.4f}  "
          f"std: {val_stats['std']:.4f}  "
          f"range: [{val_stats['min']:.4f}, {val_stats['max']:.4f}]")

    plot_confusion_matrix(y_true, y_pred, name)
    plot_learning_curves(name)
    plot_grad_cam(model, name, device)

    # save per-model metrics
    out = {k: round(v, 6) for k, v in metrics.items()}
    out["val_stats"] = {k: round(v, 6) for k, v in val_stats.items()}
    metrics_path = model_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] saved {metrics_path}")

    return {
        "metrics": metrics,
        "val_stats": val_stats,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def _save(fig: plt.Figure, filename: str) -> None:
    path = OUTPUTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] saved {path}")


if __name__ == "__main__":
    all_results = {}
    for name in ("pretrained", "scratch"):
        all_results[name] = evaluate_model(name)

    # combined visualizations
    histories = {}
    for name in ("pretrained", "scratch"):
        with open(CHECKPOINTS_DIR / f"{name}_history.json") as f:
            histories[name] = json.load(f)

    plot_combined_curves(histories)
    plot_roc_curves(all_results)
    plot_comparison_table(all_results)

    # side-by-side console output
    print("\n=== Side-by-side comparison ===")
    header = f"{'metric':12s}  {'pretrained':>12s}  {'scratch':>10s}"
    print(header)
    print("-" * len(header))
    for metric in all_results["pretrained"]["metrics"]:
        p = all_results["pretrained"]["metrics"][metric]
        s = all_results["scratch"]["metrics"][metric]
        print(f"{metric:12s}  {p:>12.4f}  {s:>10.4f}")
    print()
    for stat in ("mean", "median", "std", "stability"):
        p = all_results["pretrained"]["val_stats"][stat]
        s = all_results["scratch"]["val_stats"][stat]
        print(f"val_{stat:10s}  {p:>12.4f}  {s:>10.4f}")

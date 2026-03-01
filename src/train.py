"""
Training loop for the xray-classifier experiment.

Trains both ResNet18 variants (pretrained, scratch) with identical hyperparameters.
Tracks: train loss, val loss, val accuracy per epoch.
Saves best checkpoint (by val accuracy) to checkpoints/<name>_best.pt.
Saves per-epoch history to checkpoints/<name>_history.json.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from src.data import get_loaders
from src.models import get_model, count_params

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"


LR         = 1e-3
EPOCHS     = 30
BATCH_SIZE = 16


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return (avg_loss, accuracy) on *loader*."""
    model.eval()
    total_loss = 0.0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * len(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def train(
    model_name: str,
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Train one model variant and return its history dict.

    Args:
        model_name: "pretrained" or "scratch".
        epochs:     Number of training epochs.
        lr:         Learning rate for Adam.
        batch_size: Batch size for all loaders.

    Returns:
        dict with keys "train_loss", "val_loss", "val_acc" (lists, one entry per epoch).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  model: {model_name}  |  device: {device}  |  epochs: {epochs}  |  lr: {lr}")
    print(f"{sep}")

    loaders = get_loaders(batch_size=batch_size)

    model = get_model(model_name, device)
    counts = count_params(model)
    print(f"  params — total: {counts['total']:,}  trainable: {counts['trainable']:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINTS_DIR / f"{model_name}_best.pt"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            marker = "  [best]"

        print(
            f"  epoch {epoch:3d}/{epochs}"
            f"  train_loss: {train_loss:.4f}"
            f"  val_loss: {val_loss:.4f}"
            f"  val_acc: {val_acc:.4f}{marker}"
        )

    print(f"\n  best val_acc: {best_val_acc:.4f}  ->  {ckpt_path}")

    history_path = CHECKPOINTS_DIR / f"{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  history saved -> {history_path}")

    return history


if __name__ == "__main__":
    for name in ("pretrained", "scratch"):
        train(name)

"""
Data preparation and loading for the xray-classifier experiment.

Three data splits:
  - train (scarce): 50 images (25 NORMAL + 25 PNEUMONIA) sampled from the full
    Kaggle train set. This simulates a data-scarce medical imaging scenario.
  - val: 16 images (8+8) from the original Kaggle validation split. Used for
    checkpoint selection during training.
  - test: 624 images (234 NORMAL + 390 PNEUMONIA) from the original Kaggle test
    split. Used once for final evaluation.

Responsibilities:
- Sample the scarce train set from data/raw/ → data/scarce/
- Build PyTorch DataLoaders for all three splits
- Visualize class distributions and augmented samples
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "chest_xray"
SCARCE_DIR = ROOT / "data" / "scarce"

CLASSES = ["NORMAL", "PNEUMONIA"]
SAMPLES_PER_CLASS = 25  # 50 total scarce train images

# Transforms
# ImageNet stats — correct for transfer learning; also used for from-scratch
# to keep the comparison fair (identical preprocessing for both models).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# Sampling
def sample_scarce(seed: int = 42, overwrite: bool = False) -> None:
    """Copy 25 NORMAL + 25 PNEUMONIA images from raw train → data/scarce/train/.

    Args:
        seed: Random seed for reproducible sampling.
        overwrite: If True, delete and re-create scarce/train before sampling.
    """
    scarce_train = SCARCE_DIR / "train"

    if scarce_train.exists() and any(scarce_train.iterdir()):
        if not overwrite:
            print(f"[data] scarce/train already exists — skipping sampling. "
                  f"Pass overwrite=True to redo.")
            return
        shutil.rmtree(scarce_train)

    rng = random.Random(seed)
    raw_train = RAW_DIR / "train"

    if not raw_train.exists():
        print(
            f"[data] ERROR: raw dataset not found at {raw_train}\n"
            f"  Download it first:\n"
            f"    kaggle datasets download -d paultimothymooney/chest-xray-pneumonia\n"
            f"    unzip -q chest-xray-pneumonia.zip -d data/raw/",
            file=sys.stderr,
        )
        sys.exit(1)

    for cls in CLASSES:
        src_dir = raw_train / cls
        dst_dir = scarce_train / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(src_dir.glob("*.jpeg")) + sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.png"))
        chosen = rng.sample(images, SAMPLES_PER_CLASS)

        for img_path in chosen:
            shutil.copy(img_path, dst_dir / img_path.name)

        print(f"[data] sampled {SAMPLES_PER_CLASS} {cls} images -> {dst_dir}")


def verify_scarce() -> bool:
    """Check that scarce/train has the expected number of images per class."""
    ok = True
    for cls in CLASSES:
        cls_dir = SCARCE_DIR / "train" / cls
        count = len(list(cls_dir.glob("*"))) if cls_dir.exists() else 0
        status = "OK" if count == SAMPLES_PER_CLASS else f"EXPECTED {SAMPLES_PER_CLASS}, GOT {count}"
        print(f"[data] scarce/train/{cls}: {count} images — {status}")
        if count != SAMPLES_PER_CLASS:
            ok = False
    return ok


# DataLoaders
def get_loaders(batch_size: int = 16, num_workers: int = 0) -> dict[str, DataLoader]:
    """Return DataLoaders for train (scarce), val, and test splits.

    Args:
        batch_size: Batch size for all loaders.
        num_workers: Worker processes for DataLoader (0 = main process).

    Returns:
        dict with keys "train", "val", "test".
    """
    splits = {
        "train": (SCARCE_DIR / "train",   train_transform, True),
        "val":   (RAW_DIR / "val",        eval_transform,  False),
        "test":  (RAW_DIR / "test",       eval_transform,  False),
    }

    if not RAW_DIR.exists():
        print(
            f"[data] ERROR: raw dataset not found at {RAW_DIR}\n"
            f"  Download it first:\n"
            f"    kaggle datasets download -d paultimothymooney/chest-xray-pneumonia\n"
            f"    unzip -q chest-xray-pneumonia.zip -d data/raw/",
            file=sys.stderr,
        )
        sys.exit(1)

    loaders = {}
    for name, (path, tfm, shuffle) in splits.items():
        dataset = ImageFolder(root=str(path), transform=tfm)
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"[data] {name:5s}: {len(dataset):4d} images, "
              f"{len(loaders[name])} batches "
              f"(class_to_idx: {dataset.class_to_idx})")

    return loaders


OUTPUTS_DIR = ROOT / "outputs"


def plot_class_distribution() -> None:
    """Bar chart of class counts across train (scarce), val, and test splits."""
    splits = {
        "Train (Scarce)": SCARCE_DIR / "train",
        "Val": RAW_DIR / "val",
        "Test": RAW_DIR / "test",
    }

    counts = {name: {} for name in splits}
    for name, path in splits.items():
        for cls in CLASSES:
            cls_dir = path / cls
            counts[name][cls] = len(list(cls_dir.glob("*"))) if cls_dir.exists() else 0

    x = np.arange(len(splits))
    width = 0.35
    normal = [counts[s]["NORMAL"] for s in splits]
    pneumonia = [counts[s]["PNEUMONIA"] for s in splits]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, normal, width, label="NORMAL", color="steelblue")
    bars2 = ax.bar(x + width / 2, pneumonia, width, label="PNEUMONIA", color="coral")

    ax.bar_label(bars1, padding=3)
    ax.bar_label(bars2, padding=3)

    ax.set_ylabel("Number of Images")
    ax.set_title("Dataset Split Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(list(splits.keys()))
    ax.legend()

    plt.tight_layout()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "class_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[data] saved {out_path}")


def plot_augmented_samples(n: int = 8) -> None:
    """Grid of augmented train images to sanity-check transforms."""
    dataset = ImageFolder(root=str(SCARCE_DIR / "train"), transform=train_transform)
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 1.5, 6))
    axes = axes.flat

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    # Pick equal samples from both classes (first half NORMAL, second half PNEUMONIA)
    indices = list(range(n // 2)) + list(range(25, 25 + n // 2))
    for ax, idx in zip(axes, indices):
        img, label = dataset[idx]
        img = img * std + mean  # unnormalize
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(dataset.classes[label], fontsize=10)
        ax.axis("off")

    plt.suptitle("Augmented Training Samples", fontsize=14)
    plt.tight_layout()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "augmented_samples.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[data] saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for xray-classifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--overwrite", action="store_true", help="Re-create scarce set even if it exists")
    args = parser.parse_args()

    print("=== Sampling scarce train set ===")
    sample_scarce(seed=args.seed, overwrite=args.overwrite)

    print("\n=== Verifying scarce set ===")
    verify_scarce()

    print("\n=== Building DataLoaders ===")
    loaders = get_loaders(batch_size=16)

    print("\n=== Batch shape check ===")
    images, labels = next(iter(loaders["train"]))
    print(f"images: {images.shape}  labels: {labels.shape}")
    print(f"pixel range after norm: [{images.min():.2f}, {images.max():.2f}]")

    print("\n=== RGB channel verification ===")
    from PIL import Image
    test_img = next(iter((SCARCE_DIR / "train" / "NORMAL").glob("*")))
    raw_mode = Image.open(test_img).mode
    print(f"raw image mode: {raw_mode} -> torchvision converts to 3-channel RGB automatically")
    assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"
    print("all images are 3-channel RGB after transform: OK")

    print("\n=== Generating visualizations ===")
    plot_class_distribution()
    plot_augmented_samples()

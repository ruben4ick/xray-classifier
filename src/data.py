"""
Data preparation and loading for the xray-classifier experiment.

Responsibilities:
- Sample 50 images (25 NORMAL + 25 PNEUMONIA) from the raw train set → data/scarce/
- Build PyTorch DataLoaders for train (scarce), val, and test splits
"""

import random
import shutil
from pathlib import Path

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
              f"(classes: {dataset.classes})")

    return loaders


# Quick smoke-test

if __name__ == "__main__":
    print("=== Sampling scarce train set ===")
    sample_scarce()

    print("\n=== Verifying scarce set ===")
    verify_scarce()

    print("\n=== Building DataLoaders ===")
    loaders = get_loaders(batch_size=16)

    print("\n=== Batch shape check ===")
    images, labels = next(iter(loaders["train"]))
    print(f"images: {images.shape}  labels: {labels.shape}")
    print(f"pixel range after norm: [{images.min():.2f}, {images.max():.2f}]")

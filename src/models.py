"""
Model definitions for the xray-classifier experiment.

Two ResNet18 variants:
  pretrained — ImageNet weights, backbone frozen, only the FC head is trained.
  scratch    — random initialization, all layers trainable.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

NUM_CLASSES = 2  # NORMAL, PNEUMONIA


def build_pretrained() -> nn.Module:
    """ResNet18 with ImageNet weights; backbone frozen, head trainable."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # new layer → trainable
    return model


def build_scratch() -> nn.Module:
    """ResNet18 with random initialization; all layers trainable."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


_BUILDERS = {
    "pretrained": build_pretrained,
    "scratch": build_scratch,
}


def get_model(name: str, device: torch.device | str = "cpu") -> nn.Module:
    """Return the requested model moved to *device*.

    Args:
        name:   "pretrained" or "scratch".
        device: torch device or string.

    Returns:
        nn.Module ready for training.
    """
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model {name!r}. Choose from {list(_BUILDERS)}")
    return _BUILDERS[name]().to(device)


def count_params(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    for name in ("pretrained", "scratch"):
        model = get_model(name, device)
        counts = count_params(model)
        print(f"[{name}]")
        print(f"  total params:     {counts['total']:>10,}")
        print(f"  trainable params: {counts['trainable']:>10,}")
        dummy = torch.randn(2, 3, 224, 224, device=device)
        out = model(dummy)
        print(f"  output shape:     {tuple(out.shape)}\n")

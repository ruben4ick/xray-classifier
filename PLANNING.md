# PLANNING.md — Few-Shot X-Ray Classifier

---

## 1. Project Overview

### Problem Statement

Labeled medical images are scarce and expensive to produce — each image requires annotation by a trained radiologist. This project investigates whether ImageNet-pretrained weights on a ResNet18 backbone can achieve clinically useful accuracy on a pneumonia detection task with only **50 training images** (25 per class), compared to training the same architecture from random initialization.

### Hypothesis

| Model | Weights | Expected Accuracy (n=50) |
|-------|---------|--------------------------|
| Transfer Learning (ResNet18) | ImageNet pre-trained, frozen backbone | **>80%** |
| From Scratch (ResNet18) | Random initialization | **~50%** (chance-level) |

**Why this matters:** If pre-trained weights achieve clinical-grade accuracy on just 50 images, it validates few-shot transfer learning as a practical strategy for data-scarce healthcare settings.

### Success Criteria

| Tier | Criterion | Threshold |
|------|-----------|-----------|
| **Primary** | Pretrained test accuracy | >80% |
| **Primary** | Scratch test accuracy | <65% |
| **Secondary** | Pretrained AUC-ROC | >0.85 |
| **Tertiary** | Grad-CAM (qualitative) | Pretrained attends to lung regions; scratch shows scattered/random attention |

### Scope

This is a **controlled experiment**, not a clinical deployment. No HIPAA considerations apply. The goal is to demonstrate the transfer learning effect under data scarcity, not to build a production diagnostic tool.

---

## 2. Dataset

### Source

- **Name:** Chest X-Ray Images (Pneumonia)
- **Author:** Paul Mooney (sourced from Kermany et al., 2018)
- **URL:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Download:**
  ```bash
  kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
  unzip -q chest-xray-pneumonia.zip -d data/raw/
  ```

### Original Split Distribution

| Split | NORMAL | PNEUMONIA | Total | PNEUMONIA % |
|-------|--------|-----------|-------|-------------|
| train | 1,341 | 3,875 | 5,216 | 74.3% |
| val | 8 | 8 | 16 | 50.0% |
| test | 234 | 390 | 624 | 62.5% |
| **Total** | **1,583** | **4,273** | **5,856** | **73.0%** |

**Class imbalance warning:** The original train set is ~74% PNEUMONIA. The test set is ~62.5% PNEUMONIA. This means a naive model predicting all-PNEUMONIA would score 62.5% accuracy on the test set. This is why F1 (macro) and AUC-ROC are essential metrics alongside accuracy.

### Image Properties

| Property | Value |
|----------|-------|
| File format | `.jpeg` (all 5,856 files) |
| Color mode | Mixed — ~5,573 grayscale (PIL mode `L`), ~283 RGB (PIL mode `RGB`) |
| Resolution | Variable (e.g., 2090×1858, 1422×1152, 1152×760). No standard size. |
| Content | Anterior-posterior chest radiographs |

### Grayscale → RGB Conversion

`torchvision.datasets.ImageFolder` uses `pil_loader` internally, which calls `img.convert('RGB')`. This replicates the single grayscale channel across all 3 RGB channels automatically. No explicit conversion is needed in our transforms pipeline, but this implicit behavior is a critical dependency — without it, grayscale images would produce `(1, H, W)` tensors and ResNet18 (which expects 3-channel input) would crash.

### PNEUMONIA Subtypes

File naming reveals subtypes: `person1000_bacteria_2931.jpeg` vs `person1000_virus_1681.jpeg`. These are **collapsed into a single PNEUMONIA class**. This is a deliberate simplification — our experiment tests transfer learning effectiveness, not bacterial vs viral differentiation.

---

## 3. Data Pipeline

### 3.1 Downsampling Strategy

**Function:** `sample_scarce(seed=42, overwrite=False)` in `src/data.py`

**Algorithm:**
1. For each class in `["NORMAL", "PNEUMONIA"]`:
   - Glob `*.jpeg` + `*.jpg` + `*.png` from `data/raw/chest_xray/train/{class}/`
   - **Sort** the file list (ensures deterministic ordering across OS/filesystem differences)
   - `random.Random(42).sample(sorted_images, 25)` — uses a dedicated RNG instance, isolated from global state
   - Copy selected files to `data/scarce/train/{class}/`
2. Verification: `verify_scarce()` checks exactly 25 files per class directory

**Key design decisions:**
- Uses `random.Random(seed)` instance, not `random.seed()` global — isolated from any other random state in the process
- Sort-then-sample ensures identical results regardless of filesystem enumeration order
- `overwrite=False` by default — won't destroy an existing scarce set accidentally

### 3.2 Transform Pipeline

#### Train Transforms (in order)

| # | Transform | Parameters | Justification |
|---|-----------|------------|---------------|
| 1 | `Resize` | `(224, 224)` | ResNet18 expects 224×224. Uses bilinear interpolation (PIL default). Distorts aspect ratio — acceptable for this experiment. |
| 2 | `RandomHorizontalFlip` | `p=0.5` | X-rays are roughly left-right symmetric. Effectively doubles dataset to ~100 virtual images. |
| 3 | `RandomRotation` | `degrees=10` | Simulates slight patient positioning differences. 10° is conservative to avoid anatomically unrealistic views. |
| 4 | `ColorJitter` | `brightness=0.2, contrast=0.2` | Simulates exposure variation across X-ray machines. Saturation and hue are 0 (default) — appropriate for grayscale-origin images. |
| 5 | `ToTensor` | — | Converts PIL Image → float32 tensor, scales [0,255] → [0.0,1.0]. Output: `(3, 224, 224)`. |
| 6 | `Normalize` | `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]` | ImageNet channel statistics. Used for **both** models for fair comparison. For pretrained: matches the distribution the backbone was trained on. For scratch: arbitrary but ensures identical input distribution. |

#### Eval Transforms (val and test)

| # | Transform | Parameters |
|---|-----------|------------|
| 1 | `Resize` | `(224, 224)` |
| 2 | `ToTensor` | — |
| 3 | `Normalize` | `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]` |

No augmentation on eval — deterministic evaluation.

### 3.3 DataLoader Configuration

**Function:** `get_loaders(batch_size=16, num_workers=0)` in `src/data.py`

| Split | Source Path | Transform | Shuffle | Images | Batches |
|-------|-------------|-----------|---------|--------|---------|
| train | `data/scarce/train/` | `train_transform` | `True` | 50 | ceil(50/16) = **4** |
| val | `data/raw/chest_xray/val/` | `eval_transform` | `False` | 16 | ceil(16/16) = **1** |
| test | `data/raw/chest_xray/test/` | `eval_transform` | `False` | 624 | ceil(624/16) = **39** |

**Configuration details:**
- `pin_memory=True` — pre-allocates CUDA pinned memory for faster GPU transfer; harmless on CPU
- `num_workers=0` — main process loading. Safe default for macOS/Windows where multiprocessing with CUDA can cause issues. Acceptable performance given our tiny train set (50 images).
- Returns `dict[str, DataLoader]` with keys `"train"`, `"val"`, `"test"`

---

## 4. Model Architecture

### 4.1 Why ResNet18

1. **Lightweight** (~11.2M params) — fast iteration, runs on CPU or single GPU
2. **Strong ImageNet pre-trained weights** available via `torchvision` (`ResNet18_Weights.IMAGENET1K_V1`)
3. **Same architecture for both variants** — isolates the effect of weight initialization only
4. **Well-documented** — widely used as a baseline in medical imaging literature (Raghu et al., 2019)

### 4.2 Architecture Breakdown

ResNet18 consists of an initial convolution + 4 layer groups + a classifier head:

| Layer Group | Structure | Output Size | Parameters |
|-------------|-----------|-------------|------------|
| `conv1` | Conv2d(3→64, 7×7, stride=2, padding=3), BN, ReLU | 112×112 | 9,408 + 128 |
| `maxpool` | MaxPool2d(3×3, stride=2, padding=1) | 56×56 | 0 |
| `layer1` | 2 × BasicBlock(64→64) | 56×56 | 147,968 |
| `layer2` | 2 × BasicBlock(64→128, stride=2) | 28×28 | 525,568 |
| `layer3` | 2 × BasicBlock(128→256, stride=2) | 14×14 | 2,099,712 |
| `layer4` | 2 × BasicBlock(256→512, stride=2) | 7×7 | 8,393,728 |
| `avgpool` | AdaptiveAvgPool2d(1×1) | 1×1 | 0 |
| `fc` | Linear(512→2) | 2 | **1,026** |
| **Total** | | | **~11,177,538** |

Each **BasicBlock** contains: `Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm + skip connection → ReLU`

### 4.3 Transfer Learning Variant

**Builder:** `build_pretrained()` in `src/models.py`

```python
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False          # freeze entire backbone
model.fc = nn.Linear(512, 2)            # new head → requires_grad=True by default
```

| Metric | Value |
|--------|-------|
| Total parameters | ~11,177,538 |
| Trainable parameters | **1,026** (fc only: 512×2 weights + 2 biases) |
| Frozen parameters | ~11,176,512 |

**Why frozen backbone isolates transfer learning:** By freezing all convolutional layers, the pretrained model can only learn a linear decision boundary on top of ImageNet features. If this achieves >80% accuracy, it proves that ImageNet features (edges, textures, shapes learned from natural images) are **transferable** to the X-ray domain. The model cannot memorize the training data because it only has 1,026 learnable parameters for 50 images.

### 4.4 From-Scratch Variant

**Builder:** `build_scratch()` in `src/models.py`

```python
model = resnet18(weights=None)           # Kaiming initialization (PyTorch default)
model.fc = nn.Linear(512, 2)
```

| Metric | Value |
|--------|-------|
| Total parameters | ~11,177,538 |
| Trainable parameters | **~11,177,538** (all layers) |

**Expected failure mode:** 11.2M trainable parameters trained on 50 images → catastrophic overfitting. The model will memorize training data (train accuracy → 100%) but fail to generalize (test accuracy ≈ chance level ~50%).

### 4.5 Model Factory API

```python
get_model(name: str, device: torch.device | str = "cpu") -> nn.Module
```
- Valid names: `"pretrained"`, `"scratch"` (dispatched via `_BUILDERS` dict)
- Raises `ValueError` for unknown names

```python
count_params(model: nn.Module) -> dict[str, int]
```
- Returns `{"total": int, "trainable": int}`

---

## 5. Training Protocol

### 5.1 Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Optimizer | `Adam` | Adaptive per-parameter learning rates; works well with small datasets and few trainable params. Default betas `(0.9, 0.999)`. |
| Learning Rate | `1e-3` | Adam default. Standard starting point. Sufficient for 1,026-param FC head (pretrained). |
| Epochs | `30` | Enough for pretrained to converge (typically within 5-10 epochs). Enough to observe overfitting in scratch model. |
| Batch Size | `16` | Yields ~4 batches/epoch for 50 images. Small batches → gradient noise acts as mild regularization. |
| Loss | `nn.CrossEntropyLoss()` | Combines log-softmax + NLL. No class weights — the scarce train set is balanced (25+25). |
| Checkpoint | Best `val_acc` (strict `>`) | Saved to `checkpoints/{model_name}_best.pt` |

### 5.2 Design Decisions

**No learning rate schedule:** With only 1,026 trainable parameters (pretrained) and 30 epochs, the optimization landscape is simple enough that a fixed LR suffices. A scheduler would add complexity without measurable benefit.

**No early stopping:** The checkpoint strategy saves the best model by `val_acc`, so overfitting past the optimal point is handled automatically. Training always runs for the full 30 epochs to capture the complete learning curve for visualization.

**No weight decay / dropout:** The pretrained model has only 1,026 params — regularization is unnecessary. The scratch model is expected to overfit, and preventing that would weaken our hypothesis demonstration.

### 5.3 Optimizer Configuration

```python
optimizer = Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
)
```

The `filter()` is critical for the pretrained model: without it, Adam would allocate momentum and variance buffers for 11M frozen parameters — wasting memory and potentially causing subtle issues.

### 5.4 Training Loop Flow (per epoch)

```
1. model.train()               ← sets BN to training mode, enables dropout
2. For each batch (4 batches/epoch):
   a. images, labels = batch.to(device)
   b. optimizer.zero_grad()
   c. logits = model(images)
   d. loss = CrossEntropyLoss(logits, labels)
   e. loss.backward()
   f. optimizer.step()
3. train_loss = Σ(loss × batch_size) / 50
4. model.eval() + @torch.no_grad()
5. val_loss, val_acc = evaluate(model, val_loader)
6. If val_acc > best_val_acc → save checkpoint
7. Append metrics to history dict
```

### 5.5 Checkpoint & History Format

**Checkpoint:** `torch.save(model.state_dict(), path)` — saves only weights, not optimizer state.

**History JSON** (`checkpoints/{model_name}_history.json`):
```json
{
  "train_loss": [0.693147, 0.654321, ...],   // 30 entries, rounded to 6 decimals
  "val_loss":   [0.712345, 0.689012, ...],   // 30 entries
  "val_acc":    [0.500000, 0.562500, ...]    // 30 entries
}
```

### 5.6 BatchNorm Behavior (Pretrained Model)

`model.train()` is called each epoch, which puts BatchNorm layers into training mode (they compute batch statistics). However, since the backbone is frozen (`requires_grad=False`), the BatchNorm **running mean and variance are NOT updated** during the optimizer step. The `train()` call changes inference behavior (using batch stats instead of running stats), but this discrepancy is small and standard practice for frozen-backbone transfer learning.

### 5.7 Reproducibility

**Current state:** Partially reproducible.
- Data sampling: fully deterministic via `random.Random(42)` instance
- Pretrained weights: fixed ImageNet checkpoint
- Scratch initialization: Kaiming init (PyTorch default) — **NOT seeded** in current code

**Known limitation:** `torch.manual_seed()` is not called before model creation or training. Results may vary slightly across runs. For full reproducibility, the following should be added:
```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
This is tracked in GitHub Issue #7.

---

## 6. Evaluation Framework

### 6.1 Module: `src/evaluate.py` [TODO]

#### Planned Function Signatures

```python
def load_best_model(model_name: str, device: torch.device) -> nn.Module:
    """Load best checkpoint for model_name. Returns model in eval mode."""

def predict_all(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on entire loader.
    Returns:
        y_true: ground truth labels, shape (N,)
        y_pred: predicted labels (argmax), shape (N,)
        y_prob: softmax probability of class 1 (PNEUMONIA), shape (N,) — for AUC-ROC
    """

def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict:
    """Compute all metrics. Returns dict with keys:
    accuracy, precision_normal, precision_pneumonia, recall_normal, recall_pneumonia,
    f1_macro, auc_roc, confusion_matrix (2x2 ndarray)
    """

def generate_gradcam(
    model: nn.Module, image_tensor: torch.Tensor, target_layer: nn.Module
) -> np.ndarray:
    """Generate Grad-CAM heatmap for a single image.
    Returns 2D array (224, 224) with values in [0, 1].
    """

def compare_models() -> None:
    """Orchestrator: evaluate both models, generate all plots, save metrics JSON."""
```

### 6.2 Metrics — Formulas & Medical Interpretation

#### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
**Caveat:** The test set is 62.5% PNEUMONIA. A model predicting **all PNEUMONIA** scores 62.5% accuracy. This means accuracy alone is misleading — it must be paired with F1 and AUC-ROC.

#### Precision (per class)
```
Precision = TP / (TP + FP)
```
For PNEUMONIA: of all images predicted as PNEUMONIA, how many truly are.

#### Recall (per class)
```
Recall = TP / (TP + FN)
```
For PNEUMONIA: of all actual PNEUMONIA cases, how many were detected.

**Medical significance:** PNEUMONIA recall is the most critical metric. A **false negative** (FN) means a patient with pneumonia is sent home without treatment — this is dangerous. A **false positive** (FP) triggers unnecessary follow-up but the patient is safe.

#### F1 Score (macro)
```
F1_class = 2 × Precision × Recall / (Precision + Recall)
F1_macro = mean(F1_NORMAL, F1_PNEUMONIA)
```
Implementation: `sklearn.metrics.f1_score(y_true, y_pred, average='macro')`

Robust to class imbalance — treats both classes equally regardless of sample count.

#### AUC-ROC
Area Under the Receiver Operating Characteristic curve. Measures the model's ability to distinguish classes across **all possible classification thresholds**.

- Range: [0, 1]. Score of 0.5 = random. Score of 1.0 = perfect separation.
- Implementation: `sklearn.metrics.roc_auc_score(y_true, y_prob)`
- `y_prob` = softmax probability of class 1 (PNEUMONIA)

#### Confusion Matrix

```
                  Predicted NORMAL    Predicted PNEUMONIA
Actual NORMAL          TN                    FP
Actual PNEUMONIA       FN (DANGEROUS)        TP
```

Implementation: `sklearn.metrics.confusion_matrix(y_true, y_pred)`

- **FN (bottom-left):** Missed pneumonia — patient sent home untreated. Clinically the worst outcome.
- **FP (top-right):** False alarm — unnecessary follow-up, but patient is safe.

### 6.3 Grad-CAM Implementation

**Target layer:** `model.layer4[-1]` — the last BasicBlock of ResNet18. Standard choice because it has the richest semantic features while retaining spatial resolution (7×7 feature maps at this stage).

**Algorithm (9 steps):**
1. Register a **forward hook** on `model.layer4[-1]` → capture activations `A` (shape: `[1, 512, 7, 7]`)
2. Register a **backward hook** on the same layer → capture gradients `∂L/∂A` (shape: `[1, 512, 7, 7]`)
3. Forward pass: `logits = model(image)` (image shape: `[1, 3, 224, 224]`)
4. Select score for the **predicted class** (or a specific target class)
5. Backward pass: `score.backward()` — fills the gradient hook
6. Compute importance weights: `αₖ = GlobalAvgPool(∂L/∂Aₖ)` → shape `[512]`
7. Weighted combination: `cam = ReLU(Σₖ αₖ · Aₖ)` → shape `[7, 7]`
8. Upsample to input size: `F.interpolate(cam, (224, 224), mode='bilinear')` → shape `[224, 224]`
9. Normalize to [0, 1]: `cam = (cam - cam.min()) / (cam.max() - cam.min() + ε)`

**Image selection strategy:**
- Pick 3 NORMAL + 3 PNEUMONIA from the test set (6 total)
- Use the **same 6 images** for both models for direct visual comparison
- Prefer images where one model is correct and the other is wrong — these are the most revealing

---

## 7. Visualization Specifications

All plots saved to `outputs/`. Global settings: `seaborn` style, `dpi=150`.

### 7.1 `learning_curves.png`

**Layout:** 1×2 subplot figure, `figsize=(12, 5)`

| Left panel: Loss curves | Right panel: Accuracy curves |
|---|---|
| X-axis: Epoch (1–30) | X-axis: Epoch (1–30) |
| Y-axis: Loss | Y-axis: Accuracy (0.0–1.0) |
| 4 lines: pretrained train_loss (solid blue), pretrained val_loss (solid orange), scratch train_loss (dashed blue), scratch val_loss (dashed orange) | 2 lines: pretrained val_acc (solid green), scratch val_acc (dashed red) |
| Legend: upper-right | Horizontal dashed gray line at y=0.5 labeled "chance" |

`plt.tight_layout()`, `plt.savefig('outputs/learning_curves.png', dpi=150, bbox_inches='tight')`

### 7.2 `confusion_pretrained.png` / `confusion_scratch.png`

**Layout:** Single heatmap, `figsize=(6, 5)`

- `seaborn.heatmap(cm, annot=True, fmt='d', cmap='Blues')`
- X-axis label: "Predicted"
- Y-axis label: "Actual"
- Tick labels: `['NORMAL', 'PNEUMONIA']`
- Title: `"Confusion Matrix — {Model Name} (Test Set, n=624)"`

### 7.3 `comparison_table.png`

**Layout:** Rendered table as image, `figsize=(8, 3)`

- Use `matplotlib.pyplot.table()` or `ax.table()`
- Columns: `Metric`, `Pretrained`, `Scratch`, `Winner`
- Rows: Accuracy, F1 (macro), AUC-ROC, Precision (PNEUMONIA), Recall (PNEUMONIA), Trainable Params
- Highlight the winning model per row with bold or background color
- `ax.axis('off')` — no visible axes

### 7.4 `gradcam_pretrained.png` / `gradcam_scratch.png`

**Layout:** 2×3 grid, `figsize=(15, 10)`

- Top row: 3 NORMAL test images
- Bottom row: 3 PNEUMONIA test images
- Each cell: original X-ray with Grad-CAM heatmap overlay (`alpha=0.4`, colormap `jet`)
- Title per cell: `"True: {label} | Pred: {pred} | Conf: {prob:.1%}"`
- Shared colorbar for heatmap intensity

### 7.5 `class_distribution.png`

**Layout:** Grouped bar chart, `figsize=(8, 5)`

- 3 groups on X-axis: `Train (Scarce)`, `Val`, `Test`
- 2 bars per group: NORMAL (steelblue), PNEUMONIA (coral)
- Annotate exact count on top of each bar
- Title: `"Dataset Split Distribution"`
- Y-axis: `"Number of Images"`

### 7.6 `roc_curve.png` (recommended addition)

**Layout:** Single plot, `figsize=(7, 6)`

- ROC curves for both models on the same axes
- Diagonal dashed gray line for random baseline (AUC=0.5)
- Legend: `"Pretrained (AUC = 0.XX)"`, `"Scratch (AUC = 0.XX)"`
- X-axis: `"False Positive Rate"`, Y-axis: `"True Positive Rate"`

---

## 8. Project Structure (Detailed)

```
xray-classifier/
├── .gitignore              # data/, checkpoints/, __pycache__/, *.pt, *.pyc, .ipynb_checkpoints/
├── PLAN.md                 # original high-level plan (reference)
├── PLANNING.md             # THIS FILE — detailed implementation spec
├── README.md               # [TODO] project showcase with embedded results
├── requirements.txt        # pinned: torch 2.3.1, torchvision 0.18.1, numpy, Pillow,
│                           #         scikit-learn, matplotlib, seaborn, kaggle, jupyter, tqdm
│
├── data/
│   ├── raw/                # [git-ignored] Full Kaggle dataset
│   │   └── chest_xray/
│   │       ├── train/      # 5,216 images (1,341 NORMAL + 3,875 PNEUMONIA)
│   │       ├── val/        # 16 images (8 NORMAL + 8 PNEUMONIA)
│   │       └── test/       # 624 images (234 NORMAL + 390 PNEUMONIA)
│   └── scarce/             # [git-ignored] 50-image stratified subset
│       └── train/
│           ├── NORMAL/     # 25 images (seed=42)
│           └── PNEUMONIA/  # 25 images (seed=42)
│
├── src/
│   ├── data.py             # [DONE] Data pipeline
│   │   ├── sample_scarce(seed=42, overwrite=False) -> None
│   │   ├── verify_scarce() -> bool
│   │   ├── get_loaders(batch_size=16, num_workers=0) -> dict[str, DataLoader]
│   │   ├── train_transform  (Compose: Resize, HFlip, Rotate, ColorJitter, ToTensor, Normalize)
│   │   ├── eval_transform   (Compose: Resize, ToTensor, Normalize)
│   │   └── Constants: ROOT, RAW_DIR, SCARCE_DIR, CLASSES, SAMPLES_PER_CLASS,
│   │                  IMAGENET_MEAN, IMAGENET_STD
│   │
│   ├── models.py           # [DONE] Model definitions
│   │   ├── build_pretrained() -> nn.Module     (frozen backbone, trainable FC)
│   │   ├── build_scratch() -> nn.Module        (all trainable)
│   │   ├── get_model(name, device) -> nn.Module
│   │   ├── count_params(model) -> dict[str, int]
│   │   └── Constants: NUM_CLASSES=2, _BUILDERS dict
│   │
│   ├── train.py            # [DONE] Training loop
│   │   ├── train_one_epoch(model, loader, criterion, optimizer, device) -> float
│   │   ├── evaluate(model, loader, criterion, device) -> tuple[float, float]
│   │   ├── train(model_name, epochs=30, lr=1e-3, batch_size=16) -> dict
│   │   └── Constants: LR=1e-3, EPOCHS=30, BATCH_SIZE=16, CHECKPOINTS_DIR
│   │
│   └── evaluate.py         # [TODO] Evaluation & visualization
│       ├── load_best_model(model_name, device) -> nn.Module
│       ├── predict_all(model, loader, device) -> (y_true, y_pred, y_prob)
│       ├── compute_metrics(y_true, y_pred, y_prob) -> dict
│       ├── generate_gradcam(model, image_tensor, target_layer) -> np.ndarray
│       ├── plot_learning_curves(histories) -> None
│       ├── plot_confusion_matrix(y_true, y_pred, model_name) -> None
│       ├── plot_comparison_table(metrics_pretrained, metrics_scratch) -> None
│       ├── plot_gradcam_grid(model, loader, model_name, n=6) -> None
│       ├── plot_class_distribution() -> None
│       ├── plot_roc_curves(results) -> None
│       └── compare_models() -> None   (orchestrator)
│
├── checkpoints/            # [git-ignored]
│   ├── pretrained_best.pt
│   ├── pretrained_history.json
│   ├── scratch_best.pt
│   └── scratch_history.json
│
├── outputs/                # Committed to git (referenced by README)
│   ├── learning_curves.png
│   ├── confusion_pretrained.png
│   ├── confusion_scratch.png
│   ├── comparison_table.png
│   ├── gradcam_pretrained.png
│   ├── gradcam_scratch.png
│   ├── class_distribution.png
│   ├── roc_curve.png
│   └── metrics.json        # machine-readable results
│
└── notebook.ipynb          # [TODO] Final Jupyter report
    ├── Cell 1: Title + introduction + hypothesis
    ├── Cell 2: Dataset exploration (class distribution, sample images)
    ├── Cell 3: Model architecture comparison (param counts table)
    ├── Cell 4: Training results (load histories, show learning curves)
    ├── Cell 5: Test metrics (load metrics.json, display comparison)
    ├── Cell 6: Confusion matrices
    ├── Cell 7: Grad-CAM analysis with interpretation
    └── Cell 8: Conclusion — hypothesis confirmed/rejected
```

### Import Dependency Graph

```
data.py  ←──────────┐
  (independent)      │ get_loaders()
                     │
models.py ←──────────┤ get_model(), count_params()
  (independent)      │
                     ▼
               train.py
                     │
                     ▼
              evaluate.py  (imports from data.py, models.py, reads checkpoints/)
```

---

## 9. Implementation Roadmap

| Step | Task | Status | Dependencies | Owner |
|------|------|--------|-------------|-------|
| 1 | Environment setup, .gitignore | TODO | — | Issue #1 |
| 2 | Directory scaffolding | TODO | — | Issue #2 |
| 3 | Data sampling & verification | **DONE** | Step 1 | `src/data.py` |
| 4 | DataLoader hardening | TODO | Step 3 | Issue #4 |
| 5 | Model definitions | **DONE** | — | `src/models.py` |
| 6 | Training loop | **DONE** | Steps 3, 5 | `src/train.py` |
| 7 | Train loop CLI & logging | TODO | Step 6 | Issue #7 |
| 8 | Run baseline (scratch) | TODO | Steps 1, 3 | Issue #5 |
| 9 | Run transfer learning | TODO | Steps 1, 3 | Issue #6 |
| 10 | Implement `evaluate.py` | TODO | Steps 8, 9 | Issue #8 |
| 11 | Grad-CAM visualizations | TODO | Step 10 | Issue #9 |
| 12 | Final notebook & README | TODO | Steps 10, 11 | Issue #10 |

**Parallelization:** Steps 8 and 9 can run simultaneously. Steps 1, 2, 4, 7 can be done in parallel with each other.

---

## 10. Risk Analysis

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Test set class imbalance** (62.5% PNEUMONIA) | Accuracy misleading — all-PNEUMONIA baseline = 62.5% | HIGH | Use F1 (macro) and AUC-ROC as primary metrics. Report per-class precision/recall. |
| **Val set too small** (16 images) | Checkpoint selection is noisy. 1 image difference = ±6.25% accuracy. | MEDIUM | Acknowledge in report. Best checkpoint may not be globally optimal. Use test set for final comparison only. |
| **No reproducibility seeds in training** | Results vary across runs | MEDIUM | Add `torch.manual_seed(42)` (Issue #7). Or document as known limitation with variance analysis. |
| **Scratch model overfitting** | Train acc → 100%, test acc → ~50% | VERY HIGH | **Expected outcome.** This is the control condition — document as a feature, not a bug. |
| **BatchNorm train/eval gap** (pretrained) | Slight performance difference between `model.train()` and `model.eval()` | LOW | Gap is typically small. Could set `model.eval()` for backbone only, but adds complexity. Document behavior. |
| **Grayscale/RGB mismatch** | If any image isn't converted to 3-channel, ResNet18 crashes | LOW | `ImageFolder.pil_loader` calls `.convert('RGB')` by default. Add a defensive assertion in evaluate.py. |
| **PNEUMONIA subtype bias** | Sampled images might over-represent bacterial or viral subtype | LOW | The `random.sample()` is unbiased; with 25 images from a pool of 3,875, subtype distribution is approximately preserved. |

---

## 11. References

1. **Dataset:** Kermany, D.S., Goldbaum, M., et al. (2018). "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." *Cell*, 172(5), 1122–1131.
   - Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

2. **ResNet:** He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR 2016*. arXiv:1512.03385

3. **Transfer Learning:** Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). "How transferable are features in deep neural networks?" *NeurIPS 2014*. arXiv:1411.1792

4. **Grad-CAM:** Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV 2017*. arXiv:1610.02391

5. **Transfer Learning in Medical Imaging:** Raghu, M., Zhang, C., Kleinberg, J., & Bengio, S. (2019). "Transfusion: Understanding Transfer Learning for Medical Imaging." *NeurIPS 2019*. arXiv:1902.07208

# Transfer Learning vs. Training from Scratch (Data Scarcity)

## Hypothesis
- **Transfer Learning** (pre-trained ResNet18): >80% accuracy with only 50 training images
- **From Scratch** (random weights): ~50% accuracy or severe overfitting

---

## Phase 1: Data Preparation
- Download Kaggle dataset (`chest_xray`, NORMAL vs PNEUMONIA)
- Sample **50 images** from train set (25 per class) → "scarce" train set
- Use original validation/test sets for evaluation
- Verify class balance and image quality

## Phase 2: Model Implementation
Two models, identical architecture (ResNet18 backbone):

| Model            | Weights             | Training strategy                     |
|------------------|---------------------|---------------------------------------|
| Transfer Learning | ImageNet pre-trained | Frozen backbone + trainable head      |
| From Scratch      | Random init          | All layers trained from random weights |

## Phase 3: Training
- Same hyperparameters for both (LR, epochs, batch size, augmentation)
- Track: train loss, val loss, val accuracy per epoch
- Save best checkpoint per model

## Phase 4: Evaluation & Visualization
- Test accuracy, Precision, Recall, F1, AUC-ROC
- Confusion matrices
- **Learning curves** (loss/accuracy vs epoch) — key visual
- Grad-CAM visualizations (what each model "looks at")
- Side-by-side comparison table

## Phase 5: Report / Notebook
- Jupyter notebook with narrative
- All plots embedded
- Clear conclusion validating/rejecting hypothesis

---

## Tech Stack
| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| PyTorch + torchvision | Models, training, pre-trained weights |
| Kaggle API | Dataset download |
| scikit-learn | Metrics (F1, AUC, confusion matrix) |
| matplotlib / seaborn | Plots |
| Jupyter | Final report notebook |

---

## Project Structure
```
xray-classifier/
├── PLAN.md
├── requirements.txt
├── data/
│   ├── raw/              # original Kaggle dataset
│   └── scarce/           # sampled 50-image train set
├── src/
│   ├── data.py           # dataset loading & sampling
│   ├── models.py         # ResNet18 (transfer) + ResNet18 (scratch)
│   ├── train.py          # training loop
│   └── evaluate.py       # metrics, plots, Grad-CAM
├── checkpoints/          # saved model weights
├── outputs/              # plots, confusion matrices
└── notebook.ipynb        # final report
```

---

## Key Design Decisions
1. **ResNet18** — lightweight, fast, well-understood
2. **Frozen backbone** for transfer learning → experiment isolates *knowledge transfer*, not capacity
3. **Identical augmentation** for both models → fair comparison
4. **50 images total** (25 NORMAL + 25 PNEUMONIA) → small enough to expose the from-scratch failure mode
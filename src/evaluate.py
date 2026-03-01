"""
Evaluation and visualization for the xray-classifier experiment.

Responsibilities:
- Load best checkpoints and run inference on the test set
- Compute metrics: accuracy, precision, recall, F1 (macro), AUC-ROC
- Generate confusion matrices, learning curves, Grad-CAM visualizations
- Produce a side-by-side comparison of pretrained vs scratch models
"""

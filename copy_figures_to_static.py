"""
copy_figures_to_static.py
Run ONCE (from project root) to copy the key report figures into
app/static/figures/, where Flask can serve them to the dashboard.

Usage: python copy_figures_to_static.py
"""

import shutil
import os

SOURCE_DIR = "reports/figures"
DEST_DIR = "app/static/figures"

FILES_TO_COPY = [
    "model_comparison_bars.png",
    "confusion_matrices_all_models.png",
    "roc_curves_best_model.png",
    "precision_recall_curves_best_model.png",
    "feature_importance.png",
    "class_distribution.png",
    "shap/shap_global_bar_all_classes.png",
]

os.makedirs(DEST_DIR, exist_ok=True)

for filename in FILES_TO_COPY:
    src = os.path.join(SOURCE_DIR, filename)
    # flatten subfolder paths (e.g. shap/xxx.png) into a single filename
    dest_filename = filename.replace("/", "_").replace("\\", "_")
    dest = os.path.join(DEST_DIR, dest_filename)

    if os.path.exists(src):
        shutil.copy2(src, dest)
        print(f"Copied {filename} -> {dest}")
    else:
        print(f"WARNING: {src} not found, skipping")

print("\nDone. Figures are now available in app/static/figures/")
"""
generate_evaluation_plots.py

Generates all evaluation visualizations for the report/dashboard:
  1. Bar charts comparing all 7 models across metrics
  2. Confusion matrix heatmap grid (all 7 models)
  3. ROC curves (one-vs-rest, per class) for the best model
  4. Precision-Recall curves (per class) for the best model

Outputs saved to reports/figures/
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

MODELS_DIR = "saved_models"
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
TEST_PATH = "data/processed/test_untouched.csv"
COMPARISON_CSV = os.path.join(REPORTS_DIR, "model_comparison.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)
sns.set_style("whitegrid")


def plot_metric_comparisons(comparison_df):
    """Bar charts for accuracy, F1-macro, and training/prediction time."""
    comparison_df = comparison_df.sort_values("f1_macro", ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics_to_plot = [
        ("accuracy", "Accuracy", axes[0, 0]),
        ("f1_macro", "F1-Score (macro)", axes[0, 1]),
        ("training_time_seconds", "Training Time (s)", axes[1, 0]),
        ("predict_time_seconds", "Prediction Time (s)", axes[1, 1]),
    ]

    for col, title, ax in metrics_to_plot:
        colors = sns.color_palette("viridis", len(comparison_df))
        ax.barh(comparison_df["model_name"], comparison_df[col], color=colors)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.invert_yaxis()  # highest value at top

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "model_comparison_bars.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_confusion_matrices(model_names, class_names):
    """Grid of confusion matrix heatmaps for all models."""
    n_models = len(model_names)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i, name in enumerate(model_names):
        key = name.replace(" ", "_").lower()
        cm_path = os.path.join(REPORTS_DIR, f"confusion_matrix_{key}.npy")
        if not os.path.exists(cm_path):
            continue
        cm = np.load(cm_path)

        # Normalize by row (true class) so values are interpretable as
        # "% of this true class predicted as each label" - raw counts
        # would be dominated visually by the huge Benign class.
        cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_normalized, annot=False, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[i], cbar=True, vmin=0, vmax=1,
        )
        axes[i].set_title(name, fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].tick_params(axis="y", rotation=0)

    # hide unused subplots
    for j in range(len(model_names), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "confusion_matrices_all_models.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_roc_and_pr_curves(best_model_name, X_test, y_test, class_names):
    """One-vs-rest ROC and Precision-Recall curves for the best model."""
    key = best_model_name.replace(" ", "_").lower()
    model = joblib.load(os.path.join(MODELS_DIR, f"{key}.pkl"))

    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    y_proba = model.predict_proba(X_test)

    # --- ROC curves ---
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (One-vs-Rest) - {best_model_name}")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "roc_curves_best_model.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    # --- Precision-Recall curves ---
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
        plt.plot(recall, precision, label=class_name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves - {best_model_name}")
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "precision_recall_curves_best_model.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def main():
    comparison_df = pd.read_csv(COMPARISON_CSV)
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    class_names = list(le.classes_)

    print("Generating model comparison bar charts...")
    plot_metric_comparisons(comparison_df)

    print("\nGenerating confusion matrix grid...")
    all_model_names = comparison_df["model_name"].tolist()
    plot_confusion_matrices(all_model_names, class_names)

    print("\nGenerating ROC and Precision-Recall curves for best model...")
    test_df = pd.read_csv(TEST_PATH, low_memory=False)
    X_test = test_df.drop(columns=["Label"])
    y_test = le.transform(test_df["Label"])

    best_model_name = comparison_df.sort_values("f1_macro", ascending=False).iloc[0]["model_name"]
    print(f"Best model: {best_model_name}")
    plot_roc_and_pr_curves(best_model_name, X_test, y_test, class_names)

    print(f"\nAll evaluation plots saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
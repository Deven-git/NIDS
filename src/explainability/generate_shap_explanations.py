"""
generate_shap_explanations.py  (v2 - CatBoost-native SHAP)

The external `shap` package's TreeExplainer crashed with an access
violation (0xC0000005) when computing SHAP values for our CatBoost
model on this machine - a known Windows-specific fragility in the
shap<->catboost interaction for certain version combinations.

FIX: compute SHAP values using CatBoost's OWN built-in SHAP engine
(get_feature_importance(type="ShapValues")), which is implemented in
CatBoost's own C++ backend and does not depend on the external shap
package's internals at all. We then build all plots ourselves with
matplotlib, so the `shap` package isn't used anywhere in this script.

Outputs saved to reports/figures/shap/
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from catboost import Pool

MODELS_DIR = "saved_models"
REPORTS_DIR = "reports"
SHAP_DIR = os.path.join(REPORTS_DIR, "figures", "shap")
TEST_PATH = "data/processed/test_untouched.csv"

GLOBAL_SAMPLE_SIZE = 3000
RANDOM_STATE = 42

os.makedirs(SHAP_DIR, exist_ok=True)


def load_production_model():
    model = joblib.load(os.path.join(MODELS_DIR, "catboost.pkl"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    return model, le


def get_stratified_sample(X, y, n):
    df = X.copy()
    df["__label__"] = y
    parts = []
    for label_val in df["__label__"].unique():
        group = df[df["__label__"] == label_val]
        take = max(1, int(n * len(group) / len(df)))
        parts.append(group.sample(min(take, len(group)), random_state=RANDOM_STATE))
    sample_df = pd.concat(parts, ignore_index=True)
    y_sample = sample_df.pop("__label__").values
    return sample_df, y_sample


def compute_catboost_shap(model, X, y):
    """
    Returns shap_values with shape (n_samples, n_features, n_classes)
    and base_values with shape (n_classes,).
    CatBoost's native output is (n_samples, n_features+1, n_classes)
    where the last feature-row is the base (expected) value.
    """
    pool = Pool(X, label=y)
    raw = model.get_feature_importance(pool, type="ShapValues")
    # raw shape: (n_samples, n_classes, n_features+1) for multiclass
    print(f"Raw CatBoost SHAP output shape: {raw.shape}")

    shap_values = raw[:, :, :-1]      # drop last column (base value)
    base_values = raw[0, :, -1]        # base value per class (same for all rows)

    # rearrange to (n_samples, n_features, n_classes) for easier per-class slicing
    shap_values = np.transpose(shap_values, (0, 2, 1))
    return shap_values, base_values


def plot_global_bar(shap_values, feature_names, class_names):
    """Mean |SHAP value| per feature, averaged across all classes."""
    mean_abs = np.abs(shap_values).mean(axis=(0, 2))  # avg over samples and classes
    order = np.argsort(mean_abs)[::-1][:20]

    plt.figure(figsize=(9, 8))
    plt.barh(
        [feature_names[i] for i in order][::-1],
        mean_abs[order][::-1],
        color="teal",
    )
    plt.xlabel("Mean |SHAP value| (across all classes)")
    plt.title("SHAP Global Feature Importance - CatBoost")
    plt.tight_layout()
    path = os.path.join(SHAP_DIR, "shap_global_bar_all_classes.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_per_class_bar(shap_values, feature_names, class_names):
    """Top features per individual class."""
    for class_idx, class_name in enumerate(class_names):
        class_shap = shap_values[:, :, class_idx]
        mean_abs = np.abs(class_shap).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:15]

        plt.figure(figsize=(8, 6))
        plt.barh(
            [feature_names[i] for i in order][::-1],
            mean_abs[order][::-1],
            color="darkorange",
        )
        plt.xlabel("Mean |SHAP value|")
        plt.title(f"Top Features Driving: {class_name}")
        plt.tight_layout()
        safe_name = class_name.replace(" ", "_").lower()
        path = os.path.join(SHAP_DIR, f"shap_bar_{safe_name}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")


def plot_local_waterfall(row_shap, base_value, feature_values, feature_names,
                          class_name, save_path, max_display=12):
    """
    Custom waterfall plot: shows how each feature's SHAP value pushes
    the prediction away from the base (expected) value toward the
    final prediction for one specific sample.
    """
    order = np.argsort(np.abs(row_shap))[::-1][:max_display]
    order = order[::-1]  # smallest contribution at top for waterfall reading order

    labels = [f"{feature_names[i]} = {feature_values[i]:.2f}" for i in order]
    values = row_shap[order]

    cumulative = base_value + np.concatenate([[0], np.cumsum(values)[:-1]])
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in values]

    plt.figure(figsize=(9, 6))
    plt.barh(labels, values, left=cumulative, color=colors)
    plt.axvline(base_value, color="gray", linestyle="--", linewidth=1,
                label=f"Base value = {base_value:.3f}")
    plt.xlabel("SHAP value (impact on prediction)")
    plt.title(f"Why the model predicted: {class_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("Loading production model (CatBoost) and data...")
    model, le = load_production_model()
    class_names = list(le.classes_)

    test_df = pd.read_csv(TEST_PATH, low_memory=False)
    X_test = test_df.drop(columns=["Label"])
    y_test = le.transform(test_df["Label"])
    feature_names = X_test.columns.tolist()

    print("Building stratified sample for global SHAP computation...")
    X_sample, y_sample = get_stratified_sample(X_test, y_test, GLOBAL_SAMPLE_SIZE)
    print(f"Sample shape: {X_sample.shape}")

    print("Computing SHAP values via CatBoost's native engine...")
    shap_values, base_values = compute_catboost_shap(model, X_sample, y_sample)
    print(f"shap_values shape: {shap_values.shape} (samples, features, classes)")

    print("\nGenerating global plots...")
    plot_global_bar(shap_values, feature_names, class_names)
    plot_per_class_bar(shap_values, feature_names, class_names)

    print("\nGenerating local (per-prediction) waterfall explanations...")
    y_pred = model.predict(X_test).flatten()
    for class_idx, class_name in enumerate(class_names):
        matches = np.where((y_test == class_idx) & (y_pred == class_idx))[0]
        if len(matches) == 0:
            print(f"  Skipping {class_name} - no correctly predicted example found")
            continue

        row_idx = matches[0]
        X_row = X_test.iloc[[row_idx]]
        row_shap_all, row_base_all = compute_catboost_shap(
            model, X_row, np.array([class_idx])
        )
        row_shap = row_shap_all[0, :, class_idx]
        base_value = row_base_all[class_idx]

        safe_name = class_name.replace(" ", "_").lower()
        save_path = os.path.join(SHAP_DIR, f"shap_local_{safe_name}.png")
        plot_local_waterfall(
            row_shap, base_value, X_row.values[0], feature_names,
            class_name, save_path,
        )
        print(f"  Saved shap_local_{safe_name}.png (test row {row_idx})")

    print(f"\nAll SHAP visualizations saved to {SHAP_DIR}/")


if __name__ == "__main__":
    main()
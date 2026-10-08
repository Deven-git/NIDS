"""
select_features.py

Two-stage feature selection on the cleaned CICIDS2017 dataset:
  Stage 1: Correlation filter - drop one feature from every pair
           with correlation > 0.95 (removes redundant duplicates)
  Stage 2: Embedded method - train a quick Random Forest and rank
           remaining features by importance; keep top N

Outputs:
  - data/processed/selected_features.csv   (final feature-selected dataset)
  - reports/figures/feature_importance.png (importance bar chart)
  - reports/feature_selection_summary.json (record of what was dropped/kept)
"""

import pandas as pd
import numpy as np
import json
import os
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

INPUT_PATH = "data/processed/cleaned_data.csv"
OUTPUT_PATH = "data/processed/selected_features.csv"
SUMMARY_PATH = "reports/feature_selection_summary.json"
IMPORTANCE_PLOT_PATH = "reports/figures/feature_importance.png"

CORRELATION_THRESHOLD = 0.95

# How many top features to keep after importance ranking.
# We start with a reasonable default; Module 5 will let you
# adjust this after seeing the importance plot.
TOP_N_FEATURES = 30

# For the quick Random Forest used ONLY for importance ranking,
# we subsample rows so this runs in a few minutes, not hours.
# This is standard practice - feature importance from a well-chosen
# sample is stable and doesn't require the full 2.5M rows.
SAMPLE_SIZE = 300_000
RANDOM_STATE = 42


def correlation_filter(df, feature_cols, threshold=CORRELATION_THRESHOLD):
    """
    Drops one feature from every pair with correlation above `threshold`.
    Returns the list of features to KEEP.
    """
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = set()
    for col in upper.columns:
        # if this column has any correlation above threshold with a
        # column we haven't already decided to drop, drop THIS one
        high_corr = upper.index[upper[col] > threshold].tolist()
        for other in high_corr:
            if other not in to_drop:
                to_drop.add(col)
                break

    kept = [c for c in feature_cols if c not in to_drop]
    return kept, sorted(to_drop)


def rank_by_importance(df, feature_cols, label_col="Label"):
    """
    Trains a Random Forest on a stratified sample and returns
    features ranked by importance (descending).
    """
    # Stratified sample: keeps class proportions similar to the
    # full dataset, so rare classes like Heartbleed aren't lost
    # entirely from the importance calculation.
    sample_parts = []
    for label_value, group in df.groupby(label_col):
        n = min(len(group), int(SAMPLE_SIZE * len(group) / len(df)) + 50)
        sample_parts.append(group.sample(n, random_state=RANDOM_STATE))
    sample_df = pd.concat(sample_parts, ignore_index=True)

    X = sample_df[feature_cols]
    y = LabelEncoder().fit_transform(sample_df[label_col])

    print(f"Training importance-ranking Random Forest on {len(X):,} sampled rows...")
    start = time.time()

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",  # important: dataset is imbalanced, even in the sample
    )
    rf.fit(X, y)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f} seconds")

    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    return importances


def plot_importance(importances, top_n=TOP_N_FEATURES):
    plt.figure(figsize=(10, 12))
    top_features = importances.head(40)  # show more than we keep, for context
    plt.barh(top_features.index[::-1], top_features.values[::-1], color="teal")
    plt.axhline(
        y=len(top_features) - top_n - 0.5, color="red", linestyle="--",
        label=f"Cutoff at top {top_n}"
    )
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importance (top 40 shown)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(IMPORTANCE_PLOT_PATH), exist_ok=True)
    plt.savefig(IMPORTANCE_PLOT_PATH, dpi=150)
    print(f"Saved importance plot to {IMPORTANCE_PLOT_PATH}")


def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    feature_cols = [c for c in df.columns if c != "Label"]
    print(f"Starting features: {len(feature_cols)}")

    # --- Stage 1: correlation filter ---
    kept_after_corr, dropped_by_corr = correlation_filter(df, feature_cols)
    print(f"\nStage 1 (correlation filter): dropped {len(dropped_by_corr)} features")
    print(f"Remaining: {len(kept_after_corr)} features")

    # --- Stage 2: importance ranking on the survivors ---
    importances = rank_by_importance(df, kept_after_corr)

    print("\nTop 15 features by importance:")
    print(importances.head(15))

    plot_importance(importances)

    final_features = importances.head(TOP_N_FEATURES).index.tolist()
    print(f"\nFinal selected features ({len(final_features)}):")
    print(final_features)

    # --- Save the feature-selected dataset ---
    final_df = df[final_features + ["Label"]]
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved feature-selected dataset to {OUTPUT_PATH}")

    # --- Save a summary for your documentation/methodology chapter ---
    summary = {
        "starting_feature_count": len(feature_cols),
        "dropped_by_correlation_filter": dropped_by_corr,
        "features_after_correlation_filter": len(kept_after_corr),
        "top_n_kept": TOP_N_FEATURES,
        "final_selected_features": final_features,
        "full_importance_ranking": importances.to_dict(),
    }
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved feature selection summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
"""
split_and_balance.py

1. Loads the feature-selected dataset
2. Performs a STRATIFIED train/test split (80/20) - preserves class
   proportions so rare classes (Heartbleed, Infiltration) appear in
   both sets
3. Applies SMOTE to the TRAINING set only, to balance minority classes
4. Saves train (balanced) and test (untouched, real-world) sets separately

CRITICAL: SMOTE is applied AFTER splitting, and ONLY to the training
data. The test set must remain untouched and reflect real-world class
imbalance, or evaluation metrics become artificially inflated.
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

INPUT_PATH = "data/processed/selected_features.csv"
TRAIN_OUTPUT_PATH = "data/processed/train_balanced.csv"
TEST_OUTPUT_PATH = "data/processed/test_untouched.csv"
SUMMARY_PATH = "reports/split_balance_summary.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# SMOTE needs at least k_neighbors+1 samples of a class to work.
# Default k_neighbors=5 means a class needs >= 6 samples.
MIN_SAMPLES_FOR_SMOTE = 6

# HYBRID RESAMPLING STRATEGY:
# Forcing every minority class up to match Benign (~1.67M) both
# exhausts memory AND is statistically unsound - it means generating
# e.g. 1.67M synthetic Heartbleed samples by interpolating between
# just 9 real ones, which mostly creates near-duplicate noise rather
# than real signal.
#
# Instead: oversample small classes up to a capped ceiling, and
# moderately undersample the majority class. This is a standard,
# published technique (SMOTE + RandomUnderSampler hybrid).
SMOTE_TARGET_CEILING = 5000       # minority classes oversampled up to this many
MAJORITY_UNDERSAMPLE_TARGET = 200000  # Benign reduced to this many


def stratified_split(df, label_col="Label"):
    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # this is what guarantees proportional class representation
    )
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    print("\nClass distribution BEFORE resampling (training set):")
    class_counts = y_train.value_counts()
    print(class_counts)

    # --- Build a capped SMOTE target: oversample classes below the
    # ceiling UP TO the ceiling, but never reduce a class that's
    # already larger than the ceiling (SMOTE only oversamples,
    # it never removes samples).
    smote_targets = {}
    for label, count in class_counts.items():
        if count < SMOTE_TARGET_CEILING:
            smote_targets[label] = SMOTE_TARGET_CEILING
        # else: leave this class alone during the SMOTE step

    too_small = class_counts[class_counts < MIN_SAMPLES_FOR_SMOTE]
    if len(too_small) > 0:
        print(
            f"\nNOTE: these classes have fewer than {MIN_SAMPLES_FOR_SMOTE} "
            f"samples - k_neighbors will be reduced automatically for them:"
        )
        print(too_small)

    # k_neighbors must be smaller than the smallest class size minus 1.
    smallest_class_size = class_counts.min()
    safe_k_neighbors = max(1, min(5, smallest_class_size - 1))
    print(f"\nUsing k_neighbors={safe_k_neighbors} for SMOTE "
          f"(smallest class has {smallest_class_size} samples)")
    print(f"SMOTE will oversample classes below {SMOTE_TARGET_CEILING} "
          f"up to {SMOTE_TARGET_CEILING}")

    smote = SMOTE(
        sampling_strategy=smote_targets,
        random_state=RANDOM_STATE,
        k_neighbors=safe_k_neighbors,
    )
    X_over, y_over = smote.fit_resample(X_train, y_train)

    print("\nClass distribution AFTER SMOTE oversampling (before undersampling):")
    print(pd.Series(y_over).value_counts())

    # --- Undersample the majority class (Benign) down to a reasonable cap ---
    over_counts = pd.Series(y_over).value_counts()
    undersample_targets = {}
    for label, count in over_counts.items():
        if count > MAJORITY_UNDERSAMPLE_TARGET:
            undersample_targets[label] = MAJORITY_UNDERSAMPLE_TARGET
        else:
            undersample_targets[label] = count  # leave as-is

    rus = RandomUnderSampler(
        sampling_strategy=undersample_targets,
        random_state=RANDOM_STATE,
    )
    X_resampled, y_resampled = rus.fit_resample(X_over, y_over)

    print("\nFinal class distribution AFTER hybrid resampling "
          "(SMOTE + undersampling):")
    print(pd.Series(y_resampled).value_counts())

    return X_resampled, y_resampled


def main():
    print("Loading feature-selected dataset...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"Total rows: {len(df):,}")

    print("\nPerforming stratified train/test split...")
    X_train, X_test, y_train, y_test = stratified_split(df)
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")

    print("\nApplying SMOTE to training set only...")
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # --- Save outputs ---
    train_df = X_train_balanced.copy()
    train_df["Label"] = y_train_balanced
    os.makedirs(os.path.dirname(TRAIN_OUTPUT_PATH), exist_ok=True)
    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    print(f"\nSaved balanced training set to {TRAIN_OUTPUT_PATH}")

    test_df = X_test.copy()
    test_df["Label"] = y_test
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)
    print(f"Saved untouched test set to {TEST_OUTPUT_PATH}")

    # --- Save a summary for documentation ---
    summary = {
        "total_rows": len(df),
        "train_rows_before_smote": len(X_train),
        "train_rows_after_smote": len(X_train_balanced),
        "test_rows": len(X_test),
        "class_distribution_before_smote": y_train.value_counts().to_dict(),
        "class_distribution_after_smote": pd.Series(y_train_balanced).value_counts().to_dict(),
        "class_distribution_test_set": y_test.value_counts().to_dict(),
    }
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved split/balance summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
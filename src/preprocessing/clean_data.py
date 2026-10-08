"""
clean_data.py
Cleans the combined CICIDS2017 dataset:
- Drops duplicate/constant columns
- Fixes infinite values
- Drops rows with physically impossible negative values
- Repairs corrupted Web Attack labels
- Maps raw labels to 9 target categories
- Removes duplicate rows
"""

import pandas as pd
import numpy as np
import os

INPUT_PATH = "data/interim/combined_raw.csv"
OUTPUT_PATH = "data/processed/cleaned_data.csv"

# Features that are constant (zero variance) across the entire dataset —
# confirmed via EDA. They carry no information and some algorithms
# (e.g. SVM) can error out on zero-variance columns.
CONSTANT_FEATURES = [
    "Bwd PSH Flags", "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]

# Columns that must be physically non-negative — negative values
# indicate corrupted CICFlowMeter measurements, not real traffic
NON_NEGATIVE_COLS = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s"]

# Maps every raw label variant to one of our 9 target categories.
# We match using "startswith" logic for the corrupted Web Attack
# labels so we don't depend on the exact garbled character.
LABEL_MAP = {
    "BENIGN": "Benign",
    "DDoS": "DDoS",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "PortScan": "Port Scan",
    "Bot": "Botnet",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}


def map_label(raw_label):
    """
    Maps a raw label string to one of the 9 target categories.
    Handles the corrupted 'Web Attack ...' labels separately
    since their exact bytes are unreliable.
    """
    raw_label = raw_label.strip()

    if raw_label.startswith("Web Attack"):
        return "Web Attack"

    return LABEL_MAP.get(raw_label, "Unknown")


def clean_data(df):
    print(f"Starting rows: {len(df):,}")

    # --- Step A: Drop the duplicate column ---
    if "Fwd Header Length.1" in df.columns:
        df = df.drop(columns=["Fwd Header Length.1"])
        print("Dropped duplicate column: Fwd Header Length.1")

    # --- Step A2: Drop constant (zero-variance) features ---
    existing_constants = [c for c in CONSTANT_FEATURES if c in df.columns]
    df = df.drop(columns=existing_constants)
    print(f"Dropped {len(existing_constants)} constant features: {existing_constants}")

    # --- Step B: Replace infinite values with NaN, then drop those rows ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df[numeric_cols]).any(axis=1)
    print(f"Rows with infinite values: {inf_mask.sum():,}")

    df = df.replace([np.inf, -np.inf], np.nan)

    # --- Step C: Drop rows with any remaining NaN ---
    # (covers both the original NaN rows AND the ones we just
    # created by converting inf -> NaN)
    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df):,} rows containing NaN/Infinity")

    # --- Step C2: Drop rows with physically impossible negative values ---
    before = len(df)
    for col in NON_NEGATIVE_COLS:
        if col in df.columns:
            df = df[df[col] >= 0]
    print(f"Dropped {before - len(df):,} rows with negative Duration/Bytes-per-sec/Packets-per-sec")

    # --- Step D: Remove exact duplicate rows ---
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df):,} duplicate rows")

    # --- Step E: Map labels to target categories ---
    df["Label"] = df["Label"].apply(map_label)

    unknown_count = (df["Label"] == "Unknown").sum()
    if unknown_count > 0:
        print(f"WARNING: {unknown_count} rows mapped to 'Unknown' — inspect these!")

    print(f"\nFinal rows: {len(df):,}")
    print(f"Final columns: {df.shape[1]}")
    print("\nFinal class distribution:")
    print(df["Label"].value_counts())

    return df


if __name__ == "__main__":
    print("Loading combined dataset...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    df_clean = clean_data(df)

    os.makedirs("data/processed", exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned dataset to {OUTPUT_PATH}")
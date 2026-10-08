"""
filter_classes.py

Filters the cleaned dataset down to 5 classes for the simplified
project scope:
  - Benign
  - DoS
  - DDoS
  - Port Scan
  - Brute Force

Drops: Botnet, Web Attack, Infiltration, Heartbleed (the smallest,
most problematic classes for balancing/tuning).

The original 9-class cleaned file is backed up before being
overwritten, in case you want to reference or revert to it later.
"""

import pandas as pd
import shutil
import os

INPUT_PATH = "data/processed/cleaned_data.csv"
BACKUP_PATH = "data/processed/cleaned_data_9class_backup.csv"

KEEP_CLASSES = ["Benign", "DoS", "DDoS", "Port Scan", "Brute Force"]


def main():
    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(INPUT_PATH, BACKUP_PATH)
        print(f"Backed up original 9-class dataset to {BACKUP_PATH}")
    else:
        print(f"Backup already exists at {BACKUP_PATH}, skipping backup step.")

    print("Loading cleaned dataset...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"Original rows: {len(df):,}")
    print("Original class distribution:")
    print(df["Label"].value_counts())

    df_filtered = df[df["Label"].isin(KEEP_CLASSES)].reset_index(drop=True)

    print(f"\nFiltered rows: {len(df_filtered):,}")
    print("Filtered class distribution:")
    print(df_filtered["Label"].value_counts())

    df_filtered.to_csv(INPUT_PATH, index=False)
    print(f"\nSaved filtered (5-class) dataset to {INPUT_PATH}")
    print("This file will now be used by all downstream scripts "
          "(feature selection, training, etc.) automatically.")


if __name__ == "__main__":
    main()
"""
load_data.py
Loads and combines all CICIDS2017 CSV files, and reports
basic diagnostics: shape, dtypes, missing values, class distribution.
"""

import pandas as pd
import numpy as np
import os
import glob

# Path to the folder containing the raw CICIDS2017 CSVs
RAW_DATA_DIR = "data/raw"


def load_all_csvs(raw_dir=RAW_DATA_DIR):
    """
    Reads every CSV file in raw_dir and concatenates them into
    a single DataFrame.
    """
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. "
            "Make sure the CICIDS2017 CSVs are placed there."
        )

    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")

    df_list = []
    for file in csv_files:
        # low_memory=False avoids dtype-guessing warnings on large files
        df = pd.read_csv(file, low_memory=False)

        # Strip whitespace from column names — this fixes the
        # ' Label' vs 'Label' inconsistency across files
        df.columns = df.columns.str.strip()

        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


def inspect_dataframe(df):
    """
    Prints diagnostic information about the combined dataset.
    """
    print("\n" + "=" * 60)
    print("DATASET SHAPE")
    print("=" * 60)
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\n" + "=" * 60)
    print("COLUMN NAMES")
    print("=" * 60)
    print(list(df.columns))

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION (Label column)")
    print("=" * 60)
    print(df["Label"].value_counts())

    print("\n" + "=" * 60)
    print("MISSING VALUES (top 10 columns with most NaN)")
    print("=" * 60)
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing[missing > 0].head(10))

    print("\n" + "=" * 60)
    print("COLUMNS CONTAINING THE STRING 'Infinity'")
    print("=" * 60)
    inf_cols = []
    for col in df.select_dtypes(include="object").columns:
        if col == "Label":
            continue
        if df[col].astype(str).str.contains("Infinity").any():
            inf_cols.append(col)
    print(inf_cols if inf_cols else "None found")

    print("\n" + "=" * 60)
    print("DATA TYPES SUMMARY")
    print("=" * 60)
    print(df.dtypes.value_counts())


if __name__ == "__main__":
    df = load_all_csvs()
    inspect_dataframe(df)

    # Save the raw combined file to interim/ for the next module
    # (we do NOT clean it yet — this is just a merged snapshot)
    os.makedirs("data/interim", exist_ok=True)
    df.to_csv("data/interim/combined_raw.csv", index=False)
    print("\nSaved combined dataset to data/interim/combined_raw.csv")
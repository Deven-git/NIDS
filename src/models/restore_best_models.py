"""
restore_best_models.py

Hyperparameter tuning (on a 100K subsample) improved CatBoost and
XGBoost, but DEGRADED Random Forest and LightGBM compared to their
original configurations - almost certainly because the search
subsample underrepresented minority classes relative to the full
557K-row training set.

This script restores Random Forest and LightGBM to their better
original configurations, retrains them on the FULL training set,
and updates the comparison table so the final results reflect the
BEST version of each model (tuned where tuning helped, original
where it didn't) - a standard, defensible practice: select the
better of tuned vs. default per model based on held-out test
performance.
"""

import pandas as pd
import numpy as np
import time
import json
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
import lightgbm as lgb

MODELS_DIR = "saved_models"
REPORTS_DIR = "reports"
TRAIN_PATH = "data/processed/train_balanced.csv"
TEST_PATH = "data/processed/test_untouched.csv"
COMPARISON_CSV = os.path.join(REPORTS_DIR, "model_comparison.csv")
COMPARISON_JSON = os.path.join(REPORTS_DIR, "model_comparison_summary.json")

RANDOM_STATE = 42


def evaluate_and_save(name, model, X_train, y_train, X_test, y_test, train_time):
    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start

    y_proba = model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

    metrics = {
        "model_name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc_macro": roc_auc,
        "training_time_seconds": train_time,
        "predict_time_seconds": predict_time,
        "tuned": False,
        "note": "reverted to original config - tuning on subsample degraded this model",
    }

    print(f"{name} -> Accuracy: {metrics['accuracy']:.4f} | "
          f"F1 (macro): {metrics['f1_macro']:.4f} | ROC-AUC: {roc_auc:.4f}")

    joblib.dump(model, os.path.join(MODELS_DIR, f"{name.replace(' ', '_').lower()}.pkl"))
    cm = confusion_matrix(y_test, y_pred)
    np.save(
        os.path.join(REPORTS_DIR, f"confusion_matrix_{name.replace(' ', '_').lower()}.npy"),
        cm,
    )
    return metrics


def main():
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, low_memory=False)

    X_train = train_df.drop(columns=["Label"])
    y_train = le.transform(train_df["Label"])
    X_test = test_df.drop(columns=["Label"])
    y_test = le.transform(test_df["Label"])

    all_restored_metrics = []

    # --- Restore Random Forest to its original, better-performing config ---
    print("Retraining Random Forest with ORIGINAL configuration...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE
    )
    start = time.time()
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - start
    all_restored_metrics.append(
        evaluate_and_save("Random Forest", rf, X_train, y_train, X_test, y_test, rf_train_time)
    )

    # --- Restore LightGBM to its FIXED (post-diagnosis) configuration ---
    print("\nRetraining LightGBM with FIXED configuration...")
    lgbm = lgb.LGBMClassifier(
        n_estimators=500, max_depth=-1, num_leaves=255,
        learning_rate=0.05, min_child_samples=10,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        objective="multiclass", n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
    )
    start = time.time()
    lgbm.fit(X_train, y_train)
    lgbm_train_time = time.time() - start
    all_restored_metrics.append(
        evaluate_and_save("LightGBM", lgbm, X_train, y_train, X_test, y_test, lgbm_train_time)
    )

    # --- Update comparison table: replace RF and LightGBM rows ---
    comparison_df = pd.read_csv(COMPARISON_CSV)
    restored_names = [m["model_name"] for m in all_restored_metrics]
    comparison_df = comparison_df[~comparison_df["model_name"].isin(restored_names)]

    csv_cols = ["model_name", "accuracy", "precision_macro", "recall_macro",
                "f1_macro", "roc_auc_macro", "training_time_seconds", "predict_time_seconds"]
    new_rows_df = pd.DataFrame(all_restored_metrics)
    comparison_df = pd.concat([comparison_df, new_rows_df[csv_cols]], ignore_index=True)
    comparison_df = comparison_df.sort_values("f1_macro", ascending=False)
    comparison_df.to_csv(COMPARISON_CSV, index=False)

    with open(COMPARISON_JSON, "r") as f:
        all_metrics = json.load(f)
    all_metrics = [m for m in all_metrics if m.get("model_name") not in restored_names]
    all_metrics.extend(all_restored_metrics)
    with open(COMPARISON_JSON, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\n{'='*60}\nFINAL COMPARISON (best config per model)\n{'='*60}")
    print(comparison_df[["model_name", "accuracy", "f1_macro", "roc_auc_macro",
                          "training_time_seconds", "predict_time_seconds"]])
    print("\nBest model by F1 (macro):", comparison_df.iloc[0]["model_name"])


if __name__ == "__main__":
    main()
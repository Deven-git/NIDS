"""
retrain_lightgbm.py

Retrains ONLY LightGBM with the corrected num_leaves parameter
(the earlier run underfit because num_leaves was left at its
default of 31 while max_depth was set to 8).

Updates:
  - saved_models/lightgbm.pkl              (overwritten with fixed model)
  - reports/confusion_matrix_lightgbm.npy   (overwritten)
  - reports/model_comparison.csv            (LightGBM row replaced)
  - reports/model_comparison_summary.json   (LightGBM entry replaced)
"""

import pandas as pd
import numpy as np
import time
import json
import os
import joblib

import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

MODELS_DIR = "saved_models"
REPORTS_DIR = "reports"
TRAIN_PATH = "data/processed/train_balanced.csv"
TEST_PATH = "data/processed/test_untouched.csv"
COMPARISON_CSV = os.path.join(REPORTS_DIR, "model_comparison.csv")
COMPARISON_JSON = os.path.join(REPORTS_DIR, "model_comparison_summary.json")

RANDOM_STATE = 42


def main():
    print("Loading data and existing label encoder...")
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, low_memory=False)

    X_train = train_df.drop(columns=["Label"])
    y_train = le.transform(train_df["Label"])
    X_test = test_df.drop(columns=["Label"])
    y_test = le.transform(test_df["Label"])

    print("Training LightGBM with corrected configuration "
          "(no depth cap, more rounds, regularization)...")
    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=-1,            # unlimited - let num_leaves control complexity
        num_leaves=255,
        learning_rate=0.05,
        min_child_samples=10,
        feature_fraction=0.8,    # use 80% of features per tree - regularization
        bagging_fraction=0.8,    # use 80% of rows per iteration - regularization
        bagging_freq=5,
        objective="multiclass",
        n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
    )

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"Training time: {train_time:.2f} seconds")

    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start

    y_proba = model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

    metrics = {
        "model_name": "LightGBM",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc_macro": roc_auc,
        "training_time_seconds": train_time,
        "predict_time_seconds": predict_time,
    }

    print(f"\nAccuracy: {metrics['accuracy']:.4f} | "
          f"F1 (macro): {metrics['f1_macro']:.4f} | "
          f"ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")

    # --- Save the fixed model and confusion matrix ---
    joblib.dump(model, os.path.join(MODELS_DIR, "lightgbm.pkl"))
    cm = confusion_matrix(y_test, y_pred)
    np.save(os.path.join(REPORTS_DIR, "confusion_matrix_lightgbm.npy"), cm)
    print("Saved updated model and confusion matrix.")

    # --- Update the comparison CSV: replace the old LightGBM row ---
    comparison_df = pd.read_csv(COMPARISON_CSV)
    comparison_df = comparison_df[comparison_df["model_name"] != "LightGBM"]
    comparison_df = pd.concat([comparison_df, pd.DataFrame([metrics])], ignore_index=True)
    comparison_df = comparison_df.sort_values("f1_macro", ascending=False)
    comparison_df.to_csv(COMPARISON_CSV, index=False)

    # --- Update the JSON summary too ---
    with open(COMPARISON_JSON, "r") as f:
        all_metrics = json.load(f)
    all_metrics = [m for m in all_metrics if m.get("model_name") != "LightGBM"]
    all_metrics.append(metrics)
    with open(COMPARISON_JSON, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\nUpdated comparison table saved to {COMPARISON_CSV}")
    print("\n=== FULL UPDATED COMPARISON ===")
    print(comparison_df[["model_name", "accuracy", "f1_macro", "roc_auc_macro",
                          "training_time_seconds", "predict_time_seconds"]])


if __name__ == "__main__":
    main()
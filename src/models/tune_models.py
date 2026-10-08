"""
tune_models.py

Hyperparameter tuning for the 4 top-performing models:
CatBoost, Random Forest, XGBoost, LightGBM

Strategy:
  1. RandomizedSearchCV on a 100K stratified subsample (fast search)
  2. Refit the WINNING parameter set on the FULL balanced training set
  3. Evaluate on the untouched test set
  4. Overwrite the corresponding saved model + update comparison table

This avoids running full-scale training for every candidate combination
(which would take many hours) while still tuning on real, representative
data and finalizing on the complete training set.
"""

import pandas as pd
import numpy as np
import time
import json
import os
import joblib

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, make_scorer,
)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

MODELS_DIR = "saved_models"
REPORTS_DIR = "reports"
TRAIN_PATH = "data/processed/train_balanced.csv"
TEST_PATH = "data/processed/test_untouched.csv"
COMPARISON_CSV = os.path.join(REPORTS_DIR, "model_comparison.csv")
COMPARISON_JSON = os.path.join(REPORTS_DIR, "model_comparison_summary.json")
TUNING_LOG_PATH = os.path.join(REPORTS_DIR, "hyperparameter_tuning_log.json")

RANDOM_STATE = 42
SEARCH_SAMPLE_SIZE = 100000
N_ITER = 8    # number of random parameter combinations to try per model
CV_FOLDS = 3

f1_macro_scorer = make_scorer(f1_score, average="macro", zero_division=0)

PARAM_DISTRIBUTIONS = {
    "Random Forest": {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [10, 15, 20, 25, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators": [200, 300, 500],
        "num_leaves": [63, 127, 255],
        "learning_rate": [0.01, 0.05, 0.1],
        "feature_fraction": [0.7, 0.8, 1.0],
        "bagging_fraction": [0.7, 0.8, 1.0],
    },
    "CatBoost": {
        "iterations": [100, 200, 300],
        "depth": [6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5, 7],
    },
}


def get_base_estimator(name, num_classes):
    """Base estimator with n_jobs=1 during search to avoid nested
    parallelism fighting with RandomizedSearchCV's own n_jobs=-1."""
    if name == "Random Forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
    if name == "XGBoost":
        return xgb.XGBClassifier(
            objective="multi:softprob", num_class=num_classes,
            random_state=RANDOM_STATE, n_jobs=1, eval_metric="mlogloss",
        )
    if name == "LightGBM":
        return lgb.LGBMClassifier(
            objective="multiclass", random_state=RANDOM_STATE,
            n_jobs=1, verbose=-1,
        )
    if name == "CatBoost":
        return CatBoostClassifier(
            loss_function="MultiClass", random_state=RANDOM_STATE,
            verbose=False, thread_count=-1,  # let CatBoost use its own multithreading
        )
    raise ValueError(name)


def search_best_params(name, X_search, y_search, num_classes):
    print(f"\n{'='*60}\nSearching hyperparameters: {name}\n{'='*60}")
    base_model = get_base_estimator(name, num_classes)

    # CatBoost has its own internal multithreading (thread_count=-1).
    # Combining that with sklearn's own multiprocessing (n_jobs=-1) can
    # deadlock on Windows. So for CatBoost specifically, we run the
    # search single-process and let CatBoost parallelize internally.
    search_n_jobs = 1 if name == "CatBoost" else -1

    search = RandomizedSearchCV(
        base_model,
        param_distributions=PARAM_DISTRIBUTIONS[name],
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring=f1_macro_scorer,
        n_jobs=search_n_jobs,
        random_state=RANDOM_STATE,
        verbose=2,
    )

    start = time.time()
    search.fit(X_search, y_search)
    elapsed = time.time() - start

    print(f"Search completed in {elapsed:.1f}s")
    print(f"Best CV F1-macro: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    return search.best_params_, search.best_score_


def refit_on_full_data(name, best_params, X_train, y_train, X_test, y_test, num_classes):
    print(f"\nRefitting {name} on FULL training set with best params...")

    if name == "Random Forest":
        model = RandomForestClassifier(
            **best_params, random_state=RANDOM_STATE, n_jobs=-1
        )
    elif name == "XGBoost":
        model = xgb.XGBClassifier(
            **best_params, objective="multi:softprob", num_class=num_classes,
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric="mlogloss",
        )
    elif name == "LightGBM":
        model = lgb.LGBMClassifier(
            **best_params, objective="multiclass",
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
    elif name == "CatBoost":
        model = CatBoostClassifier(
            **best_params, loss_function="MultiClass",
            random_state=RANDOM_STATE, verbose=False,
        )

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

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
        "tuned": True,
        "best_params": best_params,
    }

    print(f"Accuracy: {metrics['accuracy']:.4f} | "
          f"F1 (macro): {metrics['f1_macro']:.4f} | "
          f"ROC-AUC: {metrics['roc_auc_macro']:.4f}")

    joblib.dump(model, os.path.join(MODELS_DIR, f"{name.replace(' ', '_').lower()}.pkl"))
    cm = confusion_matrix(y_test, y_pred)
    np.save(
        os.path.join(REPORTS_DIR, f"confusion_matrix_{name.replace(' ', '_').lower()}.npy"),
        cm,
    )

    return metrics


def main():
    print("Loading data...")
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    num_classes = len(le.classes_)

    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, low_memory=False)

    X_train = train_df.drop(columns=["Label"])
    y_train = le.transform(train_df["Label"])
    X_test = test_df.drop(columns=["Label"])
    y_test = le.transform(test_df["Label"])

    # Stratified subsample for the search phase only
    X_search, _, y_search, _ = train_test_split(
        X_train, y_train,
        train_size=SEARCH_SAMPLE_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    print(f"Search subsample: {X_search.shape}")

    tuning_log = {}
    all_new_metrics = []

    for name in ["CatBoost", "Random Forest", "XGBoost", "LightGBM"]:
        best_params, best_cv_score = search_best_params(
            name, X_search, y_search, num_classes
        )
        tuning_log[name] = {
            "best_params": best_params,
            "best_cv_f1_macro": best_cv_score,
        }

        metrics = refit_on_full_data(
            name, best_params, X_train, y_train, X_test, y_test, num_classes
        )
        all_new_metrics.append(metrics)

    # --- Save tuning log for documentation ---
    with open(TUNING_LOG_PATH, "w") as f:
        json.dump(tuning_log, f, indent=2, default=str)
    print(f"\nSaved tuning log to {TUNING_LOG_PATH}")

    # --- Update comparison table: replace tuned models' rows ---
    comparison_df = pd.read_csv(COMPARISON_CSV)
    tuned_names = [m["model_name"] for m in all_new_metrics]
    comparison_df = comparison_df[~comparison_df["model_name"].isin(tuned_names)]

    new_rows_df = pd.DataFrame(all_new_metrics)
    # keep only the shared columns for the CSV (best_params is nested, save separately)
    csv_cols = ["model_name", "accuracy", "precision_macro", "recall_macro",
                "f1_macro", "roc_auc_macro", "training_time_seconds", "predict_time_seconds"]
    comparison_df = pd.concat(
        [comparison_df, new_rows_df[csv_cols]], ignore_index=True
    )
    comparison_df = comparison_df.sort_values("f1_macro", ascending=False)
    comparison_df.to_csv(COMPARISON_CSV, index=False)

    with open(COMPARISON_JSON, "r") as f:
        all_metrics = json.load(f)
    all_metrics = [m for m in all_metrics if m.get("model_name") not in tuned_names]
    all_metrics.extend(all_new_metrics)
    with open(COMPARISON_JSON, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\n{'='*60}\nFINAL TUNED COMPARISON\n{'='*60}")
    print(comparison_df[["model_name", "accuracy", "f1_macro", "roc_auc_macro",
                          "training_time_seconds", "predict_time_seconds"]])
    print("\nBest model by F1 (macro):", comparison_df.iloc[0]["model_name"])


if __name__ == "__main__":
    main()
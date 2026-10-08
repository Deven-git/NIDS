"""
train_models.py

Trains all 7 models on the SMOTE-balanced training set and evaluates
each on the untouched test set:
  - Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost
    -> trained on the FULL balanced training set (~556K rows)
  - SVM, KNN
    -> trained on a 30K stratified subsample of the balanced training
       set, for computational feasibility (documented methodology
       choice - see README notes)

All models are evaluated on the SAME full, untouched test set, so
accuracy/precision/recall/F1/ROC-AUC comparisons remain fair across
all 7 models regardless of training set size.

Outputs:
  - saved_models/<model_name>.pkl       (each trained model)
  - saved_models/label_encoder.pkl      (shared label encoder)
  - saved_models/scaler.pkl             (StandardScaler, for SVM/KNN)
  - reports/model_comparison.csv        (metrics table)
  - reports/model_comparison_summary.json
"""

import pandas as pd
import numpy as np
import time
import json
import os
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

TRAIN_PATH = "data/processed/train_balanced.csv"
TEST_PATH = "data/processed/test_untouched.csv"
MODELS_DIR = "saved_models"
REPORTS_DIR = "reports"

RANDOM_STATE = 42
DISTANCE_MODEL_SAMPLE_SIZE = 30000  # for SVM and KNN only

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_data():
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, low_memory=False)

    X_train = train_df.drop(columns=["Label"])
    y_train_raw = train_df["Label"]

    X_test = test_df.drop(columns=["Label"])
    y_test_raw = test_df["Label"]

    # Fit label encoder on TRAIN labels (which include all 9 classes,
    # since SMOTE guaranteed every class is present)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print(f"Label classes: {list(le.classes_)}")

    return X_train, y_train, X_test, y_test, le


def get_distance_model_subsample(X_train, y_train):
    """
    Stratified 30K subsample for SVM/KNN, plus a fitted StandardScaler.
    """
    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train,
        train_size=DISTANCE_MODEL_SAMPLE_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_sub_scaled = scaler.fit_transform(X_sub)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    return X_sub_scaled, y_sub, scaler


def evaluate_model(model, X_test, y_test, num_classes):
    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start

    # ROC-AUC needs probability scores; not all models expose them
    # the same way, so we handle it defensively.
    try:
        y_proba = model.predict_proba(X_test)
        roc_auc = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro"
        )
    except Exception as e:
        print(f"  (ROC-AUC skipped: {e})")
        roc_auc = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc_macro": roc_auc,
        "predict_time_seconds": predict_time,
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm


def train_and_evaluate(name, model, X_train, y_train, X_test, y_test, num_classes):
    print(f"\n{'='*60}\nTraining: {name}\n{'='*60}")

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"Training time: {train_time:.2f} seconds")

    metrics, cm = evaluate_model(model, X_test, y_test, num_classes)
    metrics["training_time_seconds"] = train_time
    metrics["model_name"] = name

    print(f"Accuracy: {metrics['accuracy']:.4f} | "
          f"F1 (macro): {metrics['f1_macro']:.4f} | "
          f"ROC-AUC (macro): {metrics['roc_auc_macro']}")

    model_path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_').lower()}.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    np.save(
        os.path.join(REPORTS_DIR, f"confusion_matrix_{name.replace(' ', '_').lower()}.npy"),
        cm,
    )

    return metrics


def main():
    print("Loading data...")
    X_train, y_train, X_test, y_test, le = load_data()
    num_classes = len(le.classes_)
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    print("\nBuilding stratified subsample for SVM/KNN...")
    X_train_sub, y_train_sub, scaler = get_distance_model_subsample(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Distance-model training subsample: {X_train_sub.shape}")

    all_metrics = []

    # --- Tree-based models: full balanced training set, no scaling needed ---
    tree_models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=20
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            objective="multi:softprob", num_class=num_classes,
            n_jobs=-1, random_state=RANDOM_STATE,
            eval_metric="mlogloss",
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            objective="multiclass", num_class=num_classes,
            n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200, depth=8, learning_rate=0.1,
            loss_function="MultiClass", random_state=RANDOM_STATE,
            verbose=False,
        ),
    }

    for name, model in tree_models.items():
        metrics = train_and_evaluate(
            name, model, X_train, y_train, X_test, y_test, num_classes
        )
        all_metrics.append(metrics)

    # --- Distance-based models: 30K subsample, scaled features ---
    distance_models = {
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }

    for name, model in distance_models.items():
        metrics = train_and_evaluate(
            name, model, X_train_sub, y_train_sub, X_test_scaled, y_test, num_classes
        )
        metrics["note"] = f"trained on {DISTANCE_MODEL_SAMPLE_SIZE}-row stratified subsample"
        all_metrics.append(metrics)

    # --- Save comparison table ---
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df = comparison_df.sort_values("f1_macro", ascending=False)
    comparison_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)

    print(f"\n{'='*60}\nFINAL COMPARISON\n{'='*60}")
    print(comparison_df[["model_name", "accuracy", "f1_macro", "roc_auc_macro",
                          "training_time_seconds", "predict_time_seconds"]])

    with open(os.path.join(REPORTS_DIR, "model_comparison_summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\nSaved comparison table to {REPORTS_DIR}/model_comparison.csv")
    print("\nBest model by F1 (macro):", comparison_df.iloc[0]["model_name"])


if __name__ == "__main__":
    main()
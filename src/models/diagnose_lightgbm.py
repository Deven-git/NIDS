"""
diagnose_lightgbm.py
Quick diagnostic to understand why LightGBM underperformed compared
to the other tree-based models trained on the same data.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report

MODELS_DIR = "saved_models"
TEST_PATH = "data/processed/test_untouched.csv"

# Load the trained LightGBM model and label encoder
model = joblib.load(f"{MODELS_DIR}/lightgbm.pkl")
le = joblib.load(f"{MODELS_DIR}/label_encoder.pkl")

test_df = pd.read_csv(TEST_PATH, low_memory=False)
X_test = test_df.drop(columns=["Label"])
y_test = le.transform(test_df["Label"])

y_pred = model.predict(X_test)

print("Predicted class distribution:")
pred_labels = le.inverse_transform(y_pred)
print(pd.Series(pred_labels).value_counts())

print("\nActual class distribution (test set):")
print(test_df["Label"].value_counts())

print("\nFull classification report:")
print(classification_report(
    y_test, y_pred, target_names=le.classes_, zero_division=0
))

print("\nModel's num_class / objective info:")
print(f"n_classes_: {model.n_classes_}")
print(f"classes_: {model.classes_}")
print(f"Booster params: {model.get_params()}")
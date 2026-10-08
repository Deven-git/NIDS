"""
save_feature_list.py
Run ONCE (from project root) to save the exact list of features the
production model (CatBoost) expects, as JSON. The Flask upload route
uses this to validate uploaded CSVs before running predictions.

Usage: python save_feature_list.py
"""

import joblib
import json
import os

MODELS_DIR = "saved_models"

model = joblib.load(os.path.join(MODELS_DIR, "catboost.pkl"))
feature_names = model.feature_names_

with open(os.path.join(MODELS_DIR, "feature_list.json"), "w") as f:
    json.dump(feature_names, f, indent=2)

print(f"Saved {len(feature_names)} required features to saved_models/feature_list.json")
print(feature_names)
"""
app/routes/model_comparison.py
Displays the full comparison table of all 7 trained models, read
directly from reports/model_comparison.csv.
"""

import os
import pandas as pd
from flask import Blueprint, render_template, current_app
from flask_login import login_required

model_comparison_bp = Blueprint("model_comparison", __name__)


@model_comparison_bp.route("/model-comparison")
@login_required
def model_comparison():
    csv_path = os.path.join(current_app.config["REPORTS_DIR"], "model_comparison.csv")
    df = pd.read_csv(csv_path)
    df = df.sort_values("f1_macro", ascending=False)

    # Round for cleaner display
    numeric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (df[col] * 100).round(2)  # show as percentages

    for col in ["training_time_seconds", "predict_time_seconds"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    models = df.to_dict(orient="records")
    best_model = models[0]["model_name"] if models else None

    return render_template("model_comparison.html", models=models, best_model=best_model)
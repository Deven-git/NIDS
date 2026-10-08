"""
app/routes/dashboard.py
Shows the live attack-distribution chart from the most recent prediction
run (using Final_Label, which combines ML prediction + rate-based
heuristic), plus the pre-generated model performance figures.
"""

import os
import json
import pandas as pd
from flask import Blueprint, render_template, session, current_app
from flask_login import login_required

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/dashboard")
@login_required
def dashboard():
    chart_data = None
    predictions_path = session.get("predictions_path")

    if predictions_path and os.path.exists(predictions_path):
        df = pd.read_csv(predictions_path, low_memory=False)
        label_col = "Final_Label" if "Final_Label" in df.columns else "Predicted_Label"
        class_counts = df[label_col].value_counts()
        chart_data = {
            "labels": class_counts.index.tolist(),
            "values": class_counts.values.tolist(),
            "avg_confidence": round(float(df["Confidence"].mean()) * 100, 2),
        }

    return render_template(
        "dashboard.html",
        chart_data_json=json.dumps(chart_data) if chart_data else "null",
        has_predictions=chart_data is not None,
    )
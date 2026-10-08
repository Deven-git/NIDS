"""
app/routes/live_capture_route.py
Configure and run a live capture between two IP addresses, then run
the same extraction -> prediction -> rate-heuristic pipeline used for
uploaded pcap files.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from flask import (
    Blueprint, render_template, request, redirect, url_for, flash,
    session, current_app,
)
from flask_login import login_required

from app.utils.live_capture import list_interfaces, capture_between_ips
from app.utils.flow_extractor import extract_features_from_pcap
from app.utils.rate_heuristic import apply_rate_heuristic

live_bp = Blueprint("live_capture", __name__)


@live_bp.route("/live-capture", methods=["GET", "POST"])
@login_required
def live_capture():
    try:
        interfaces = list_interfaces()
    except Exception as e:
        flash(f"Could not list network interfaces: {e}", "danger")
        interfaces = []

    if request.method == "GET":
        return render_template("live_capture.html", interfaces=interfaces, result=None)

    # --- POST: run a capture ---
    interface_index = request.form.get("interface")
    ip1 = request.form.get("ip1", "").strip()
    ip2 = request.form.get("ip2", "").strip()
    duration = int(request.form.get("duration", 15))
    auto_repeat = request.form.get("auto_repeat") == "on"

    if not ip1 or not ip2:
        flash("Both IP addresses are required.", "danger")
        return render_template("live_capture.html", interfaces=interfaces, result=None)

    capture_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "live_captures")
    os.makedirs(capture_dir, exist_ok=True)
    pcap_path = os.path.join(capture_dir, f"live_{int(time.time())}.pcap")

    try:
        capture_between_ips(interface_index, ip1, ip2, duration, pcap_path)
    except Exception as e:
        flash(f"Capture failed: {e}", "danger")
        return render_template("live_capture.html", interfaces=interfaces, result=None)

    df = extract_features_from_pcap(pcap_path)

    if df.empty:
        result = {
            "no_traffic": True,
            "ip1": ip1, "ip2": ip2, "duration": duration,
            "interface": interface_index, "auto_repeat": auto_repeat,
        }
        return render_template("live_capture.html", interfaces=interfaces, result=result)

    # --- Run through the same model + rate heuristic pipeline ---
    models_dir = current_app.config["MODELS_DIR"]
    model = joblib.load(os.path.join(models_dir, "catboost.pkl"))
    le = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))
    with open(os.path.join(models_dir, "feature_list.json")) as f:
        required_features = json.load(f)

    X = df[required_features].replace([np.inf, -np.inf], np.nan)
    valid_mask = ~X.isna().any(axis=1)
    X_clean = X[valid_mask].reset_index(drop=True)
    df_clean = df[valid_mask].reset_index(drop=True)

    if len(X_clean) == 0:
        result = {"no_traffic": True, "ip1": ip1, "ip2": ip2, "duration": duration,
                   "interface": interface_index, "auto_repeat": auto_repeat}
        return render_template("live_capture.html", interfaces=interfaces, result=result)

    y_pred_encoded = model.predict(X_clean).flatten()
    y_proba = model.predict_proba(X_clean)
    predicted_labels = le.inverse_transform(y_pred_encoded)
    confidence = y_proba.max(axis=1)

    results_df = df_clean.copy()
    results_df["Predicted_Label"] = predicted_labels
    results_df["Confidence"] = confidence
    results_df, rate_triggered = apply_rate_heuristic(
        results_df, flow_count_threshold=8  # lower threshold - short window, only 2 known hosts
    )

    # Save so Dashboard/Report reflect this session too
    reports_dir = current_app.config["REPORTS_DIR"]
    predictions_dir = os.path.join(reports_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    results_path = os.path.join(predictions_dir, f"live_predictions_{int(time.time())}.csv")
    results_df.to_csv(results_path, index=False)
    session["predictions_path"] = results_path
    session["uploaded_original_name"] = f"Live capture: {ip1} <-> {ip2}"

    class_counts = results_df["Final_Label"].value_counts().to_dict()
    total_flows = len(results_df)
    benign_count = class_counts.get("Benign", 0)
    attack_count = total_flows - benign_count

    result = {
        "no_traffic": False,
        "ip1": ip1, "ip2": ip2, "duration": duration,
        "interface": interface_index, "auto_repeat": auto_repeat,
        "total_flows": total_flows,
        "benign_count": benign_count,
        "attack_count": attack_count,
        "class_counts": class_counts,
        "rate_triggered": rate_triggered,
        "avg_confidence": round(float(confidence.mean()) * 100, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return render_template("live_capture.html", interfaces=interfaces, result=result)
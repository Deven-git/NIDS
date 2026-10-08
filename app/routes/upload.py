"""
app/routes/upload.py
Handles CSV and PCAP/PCAPNG upload:
  - .csv: validated directly against required feature columns
  - .pcap/.pcapng: converted to flow features using our own
    driver-free extractor (app/utils/flow_extractor.py), which
    computes the exact 30 required features directly - no external
    tool or column-mapping guesswork needed.
"""

import os
import json
import time
import pandas as pd
from flask import (
    Blueprint, render_template, request, redirect, url_for, flash,
    session, current_app,
)
from flask_login import login_required
from werkzeug.utils import secure_filename

from app.utils.flow_extractor import extract_features_from_pcap

upload_bp = Blueprint("upload", __name__)

PCAP_EXTENSIONS = {"pcap", "pcapng"}


def allowed_file(filename):
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in current_app.config["ALLOWED_EXTENSIONS"] or ext in PCAP_EXTENSIONS


def get_required_features():
    feature_list_path = os.path.join(current_app.config["MODELS_DIR"], "feature_list.json")
    with open(feature_list_path, "r") as f:
        return json.load(f)


@upload_bp.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "GET":
        return render_template("upload.html", preview=None)

    if "file" not in request.files:
        flash("No file part in the request.", "danger")
        return redirect(url_for("upload.upload"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected.", "danger")
        return redirect(url_for("upload.upload"))

    if not allowed_file(file.filename):
        flash("Only .csv, .pcap, or .pcapng files are allowed.", "danger")
        return redirect(url_for("upload.upload"))

    original_name = secure_filename(file.filename)
    unique_name = f"{int(time.time())}_{original_name}"
    save_path = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    file_ext = original_name.rsplit(".", 1)[1].lower()
    is_pcap = file_ext in PCAP_EXTENSIONS

    # --- If a pcap was uploaded, extract flow features directly ---
    if is_pcap:
        flash("PCAP file detected - extracting flow features...", "info")
        try:
            df = extract_features_from_pcap(save_path)
        except Exception as e:
            flash(f"Failed to process PCAP file: {e}", "danger")
            os.remove(save_path)
            return redirect(url_for("upload.upload"))

        if df.empty:
            flash("No valid flows could be extracted from this pcap file.", "danger")
            os.remove(save_path)
            return redirect(url_for("upload.upload"))

        csv_path = save_path + "_extracted.csv"
        df.to_csv(csv_path, index=False)
        save_path = csv_path
        flash(f"Extracted {len(df)} flow(s) from the pcap file.", "success")

    # --- Validate content (shared path for CSV and extracted PCAP flows) ---
    try:
        df = pd.read_csv(save_path, low_memory=False)
    except Exception as e:
        flash(f"Could not read the file: {e}", "danger")
        return redirect(url_for("upload.upload"))

    df.columns = df.columns.str.strip()

    required_features = get_required_features()
    missing_columns = [c for c in required_features if c not in df.columns]

    if missing_columns:
        flash(
            f"Uploaded file is missing {len(missing_columns)} required column(s): "
            f"{', '.join(missing_columns[:5])}"
            + ("..." if len(missing_columns) > 5 else ""),
            "danger",
        )
        return redirect(url_for("upload.upload"))

    session["uploaded_file_path"] = save_path
    session["uploaded_row_count"] = len(df)
    session["uploaded_original_name"] = original_name

    flash(f"File uploaded and validated successfully: {len(df):,} rows found.", "success")

    preview = {
        "filename": original_name,
        "row_count": len(df),
        "columns": list(df.columns),
        "sample_rows": df.head(5).to_dict(orient="records"),
    }
    return render_template("upload.html", preview=preview)
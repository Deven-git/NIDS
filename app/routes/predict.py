"""
app/routes/predict.py

Loads the file the user uploaded (path stored in session), runs the
production model (CatBoost), applies the behavioral detection layer
for PCAP-derived flows, and shows a results summary.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from flask import (
    Blueprint,
    render_template,
    redirect,
    url_for,
    flash,
    session,
    current_app,
)

from flask_login import login_required

from app.utils.rate_heuristic import apply_rate_heuristic


predict_bp = Blueprint("predict", __name__)


def load_model_and_encoder():
    """Load production model, label encoder, and required features."""

    models_dir = current_app.config["MODELS_DIR"]

    model = joblib.load(
        os.path.join(models_dir, "catboost.pkl")
    )

    le = joblib.load(
        os.path.join(models_dir, "label_encoder.pkl")
    )

    with open(
        os.path.join(models_dir, "feature_list.json"),
        encoding="utf-8",
    ) as f:
        required_features = json.load(f)

    return model, le, required_features


@predict_bp.route("/predict")
@login_required
def predict():

    # --------------------------------------------------
    # 1. Get uploaded file
    # --------------------------------------------------

    file_path = session.get("uploaded_file_path")

    if not file_path or not os.path.exists(file_path):
        flash(
            "Please upload a file first.",
            "warning",
        )
        return redirect(
            url_for("upload.upload")
        )

    # --------------------------------------------------
    # 2. Load model
    # --------------------------------------------------

    model, le, required_features = (
        load_model_and_encoder()
    )

    # --------------------------------------------------
    # 3. Read uploaded CSV
    # --------------------------------------------------

    try:
        df = pd.read_csv(
            file_path,
            low_memory=False,
        )

        df.columns = df.columns.str.strip()

    except Exception as e:
        flash(
            f"Unable to read uploaded file: {e}",
            "danger",
        )

        return redirect(
            url_for("upload.upload")
        )

    # --------------------------------------------------
    # 4. Validate required model features
    # --------------------------------------------------

    missing_features = [
        feature
        for feature in required_features
        if feature not in df.columns
    ]

    if missing_features:

        flash(
            "The uploaded file is missing required "
            "model features: "
            + ", ".join(missing_features),
            "danger",
        )

        return redirect(
            url_for("upload.upload")
        )

    # --------------------------------------------------
    # 5. Check ground truth
    # --------------------------------------------------

    has_ground_truth = "Label" in df.columns

    # --------------------------------------------------
    # 6. Prepare ML input
    # --------------------------------------------------

    X = df[required_features].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid_mask = ~X.isna().any(axis=1)

    dropped_count = int(
        (~valid_mask).sum()
    )

    X_clean = X.loc[
        valid_mask
    ].reset_index(drop=True)

    df_clean = df.loc[
        valid_mask
    ].reset_index(drop=True)

    if len(X_clean) == 0:

        flash(
            "No valid rows remained after cleaning "
            "(all rows had missing/infinite values).",
            "danger",
        )

        return redirect(
            url_for("upload.upload")
        )

    # --------------------------------------------------
    # 7. CatBoost prediction
    # --------------------------------------------------

    start = time.time()

    y_pred_encoded = (
        model.predict(X_clean)
        .flatten()
    )

    y_proba = model.predict_proba(
        X_clean
    )

    predict_time = (
        time.time() - start
    )

    # --------------------------------------------------
    # 8. Convert prediction to label
    # --------------------------------------------------

    predicted_labels = (
        le.inverse_transform(
            y_pred_encoded
        )
    )

    confidence = (
        y_proba.max(axis=1)
    )

    # --------------------------------------------------
    # 9. Build result dataframe
    # --------------------------------------------------

    results_df = df_clean.copy()

    results_df[
        "Predicted_Label"
    ] = predicted_labels

    results_df[
        "Confidence"
    ] = confidence

    # --------------------------------------------------
    # 10. Behavioral detection layer
    #
    #     Port Scan
    #     DoS
    #     DDoS
    # --------------------------------------------------

    results_df, heuristic_triggered = (
        apply_rate_heuristic(
            results_df
        )
    )

    # --------------------------------------------------
    # 11. Save prediction results
    # --------------------------------------------------

    reports_dir = (
        current_app.config[
            "REPORTS_DIR"
        ]
    )

    predictions_dir = os.path.join(
        reports_dir,
        "predictions",
    )

    os.makedirs(
        predictions_dir,
        exist_ok=True,
    )

    results_path = os.path.join(
        predictions_dir,
        (
            f"predictions_"
            f"{int(time.time())}.csv"
        ),
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    session[
        "predictions_path"
    ] = results_path

    # --------------------------------------------------
    # 12. Summary statistics
    # --------------------------------------------------

    total_rows = len(
        results_df
    )

    class_counts = (
        results_df[
            "Final_Label"
        ]
        .value_counts()
        .to_dict()
    )

    benign_count = int(
        class_counts.get(
            "Benign",
            0,
        )
    )

    attack_count = (
        total_rows - benign_count
    )

    attack_percentage = (
        round(
            (
                attack_count
                / total_rows
            )
            * 100,
            2,
        )
        if total_rows > 0
        else 0
    )

    heuristic_counts = (
        results_df[
            "Heuristic_Detection"
        ]
        .value_counts()
        .to_dict()
        if "Heuristic_Detection"
        in results_df.columns
        else {}
    )

    # --------------------------------------------------
    # 13. Final summary object
    # --------------------------------------------------

    summary = {
        "total_rows": total_rows,

        "dropped_rows": dropped_count,

        "benign_count": benign_count,

        "attack_count": attack_count,

        "attack_percentage": attack_percentage,

        "class_counts": class_counts,

        "heuristic_counts": (
            heuristic_counts
        ),

        "avg_confidence": round(
            float(
                confidence.mean()
            )
            * 100,
            2,
        ),

        "predict_time_seconds": round(
            predict_time,
            3,
        ),

        "rate_heuristic_triggered": (
            bool(
                heuristic_triggered
            )
        ),
    }

    # --------------------------------------------------
    # 14. Ground-truth accuracy
    # --------------------------------------------------

    if has_ground_truth:

        true_labels = (
            df_clean[
                "Label"
            ].values
        )

        correct = (
            true_labels
            == predicted_labels
        ).sum()

        summary[
            "accuracy"
        ] = round(
            (
                correct
                / total_rows
            )
            * 100,
            2,
        )

        summary[
            "has_ground_truth"
        ] = True

    else:

        summary[
            "has_ground_truth"
        ] = False

    # --------------------------------------------------
    # 15. Render results page
    # --------------------------------------------------

    return render_template(
        "predict.html",
        summary=summary,
    )
"""
app/routes/live_capture_route.py

Configure and run a live capture between two IP addresses, then run
the same extraction -> prediction -> behavioral detection pipeline
used for uploaded PCAP files.
"""

import os
import json
import time
import joblib

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    current_app,
)

from flask_login import login_required

from app.utils.live_capture import (
    list_interfaces,
    capture_between_ips,
)

from app.utils.flow_extractor import (
    extract_features_from_pcap,
)

from app.utils.rate_heuristic import (
    apply_rate_heuristic,
)


live_bp = Blueprint(
    "live_capture",
    __name__,
)


@live_bp.route(
    "/live-capture",
    methods=["GET", "POST"],
)
@login_required
def live_capture():

    # --------------------------------------------------
    # 1. List network interfaces
    # --------------------------------------------------

    try:
        interfaces = list_interfaces()

    except Exception as e:
        flash(
            f"Could not list network interfaces: {e}",
            "danger",
        )
        interfaces = []

    # --------------------------------------------------
    # 2. GET request
    # --------------------------------------------------

    if request.method == "GET":
        return render_template(
            "live_capture.html",
            interfaces=interfaces,
            result=None,
        )

    # --------------------------------------------------
    # 3. Read POST parameters
    # --------------------------------------------------

    interface_index = request.form.get(
        "interface"
    )

    ip1 = request.form.get(
        "ip1",
        "",
    ).strip()

    ip2 = request.form.get(
        "ip2",
        "",
    ).strip()

    try:
        duration = int(
            request.form.get(
                "duration",
                15,
            )
        )
    except (TypeError, ValueError):
        duration = 15

    auto_repeat = (
        request.form.get(
            "auto_repeat"
        )
        == "on"
    )

    # --------------------------------------------------
    # 4. Validate IP addresses
    # --------------------------------------------------

    if not ip1 or not ip2:
        flash(
            "Both IP addresses are required.",
            "danger",
        )

        return render_template(
            "live_capture.html",
            interfaces=interfaces,
            result=None,
        )

    # --------------------------------------------------
    # 5. Prepare capture directory
    # --------------------------------------------------

    capture_dir = os.path.join(
        current_app.config["UPLOAD_FOLDER"],
        "live_captures",
    )

    os.makedirs(
        capture_dir,
        exist_ok=True,
    )

    pcap_path = os.path.join(
        capture_dir,
        f"live_{int(time.time())}.pcap",
    )

    # --------------------------------------------------
    # 6. Capture traffic
    # --------------------------------------------------

    try:

        capture_between_ips(
            interface_index,
            ip1,
            ip2,
            duration,
            pcap_path,
        )

    except Exception as e:

        flash(
            f"Capture failed: {e}",
            "danger",
        )

        return render_template(
            "live_capture.html",
            interfaces=interfaces,
            result=None,
        )

    # --------------------------------------------------
    # 7. Extract network flows
    # --------------------------------------------------

    try:

        df = extract_features_from_pcap(
            pcap_path
        )

    except Exception as e:

        flash(
            f"PCAP processing failed: {e}",
            "danger",
        )

        return render_template(
            "live_capture.html",
            interfaces=interfaces,
            result=None,
        )

    # --------------------------------------------------
    # 8. No traffic
    # --------------------------------------------------

    if df.empty:

        result = {
            "no_traffic": True,
            "ip1": ip1,
            "ip2": ip2,
            "duration": duration,
            "interface": interface_index,
            "auto_repeat": auto_repeat,
        }

        return render_template(
            "live_capture.html",
            interfaces=interfaces,
            result=result,
        )

    # --------------------------------------------------
    # 9. Load production model
    # --------------------------------------------------

    try:

        models_dir = (
            current_app.config[
                "MODELS_DIR"
            ]
        )

        model = joblib.load(
            os.path.join(
                models_dir,
                "catboost.pkl",
            )
        )

        le = joblib.load(
            os.path.join(
                models_dir,
                "label_encoder.pkl",
            )
        )

        with open(
            os.path.join(
                models_dir,
                "feature_list.json",
            ),
            encoding="utf-8",
        ) as f:

            required_features = json.load(
                f
            )

    except Exception as e:

        flash(
            f"Model loading failed: {e}",
            "danger",
        )

        return render_template(
            "live_capture.html",
            interfaces=interfaces,
            result=None,
        )

    # --------------------------------------------------
    # 10. Validate model features
    # --------------------------------------------------

    missing_features = [
        feature
        for feature in required_features
        if feature not in df.columns
    ]

    if missing_features:

        flash(
            "Captured traffic is missing required "
            "model features: "
            + ", ".join(missing_features),
            "danger",
        )

        return render_template(
            "live_capture.html",
            interfaces=interfaces,
            result=None,
        )

    # --------------------------------------------------
    # 11. Prepare ML input
    # --------------------------------------------------

    X = df[
        required_features
    ].replace(
        [float("inf"), float("-inf")],
        float("nan"),
    )

    valid_mask = ~X.isna().any(
        axis=1
    )

    X_clean = X.loc[
        valid_mask
    ].reset_index(
        drop=True
    )

    df_clean = df.loc[
        valid_mask
    ].reset_index(
        drop=True
    )

    # --------------------------------------------------
    # 12. No valid flows
    # --------------------------------------------------

    if len(X_clean) == 0:

        result = {
            "no_traffic": True,
            "ip1": ip1,
            "ip2": ip2,
            "duration": duration,
            "interface": interface_index,
            "auto_repeat": auto_repeat,
        }

        return render_template(
            "live_capture.html",
            interfaces=interfaces,
            result=result,
        )

    # --------------------------------------------------
    # 13. CatBoost prediction
    # --------------------------------------------------

    try:

        y_pred_encoded = (
            model.predict(
                X_clean
            ).flatten()
        )

        y_proba = model.predict_proba(
            X_clean
        )

        predicted_labels = (
            le.inverse_transform(
                y_pred_encoded
            )
        )

        confidence = y_proba.max(
            axis=1
        )

    except Exception as e:

        flash(
            f"Prediction failed: {e}",
            "danger",
        )

        return render_template(
            "live_capture.html",
            interfaces=interfaces,
            result=None,
        )

    # --------------------------------------------------
    # 14. Create prediction dataframe
    # --------------------------------------------------

    results_df = df_clean.copy()

    results_df[
        "Predicted_Label"
    ] = predicted_labels

    results_df[
        "Confidence"
    ] = confidence

    # --------------------------------------------------
    # 15. Behavioral detection
    #
    #     Port Scan
    #     Brute Force
    #     DoS
    #     DDoS
    # --------------------------------------------------

    scaled_threshold = max(
        30,
        int(duration * 2.5),
    )

    try:

        results_df, heuristic_triggered = (
            apply_rate_heuristic(
                results_df,

                flow_count_threshold=(
                    scaled_threshold
                ),

                # Port Scan
                port_scan_unique_port_threshold=5,

                port_scan_min_flow_threshold=5,

                # Brute Force
                brute_force_flow_threshold=5,
            )
        )

    except Exception as e:

        flash(
            f"Behavioral detection failed: {e}",
            "danger",
        )

        return render_template(
            "live_capture.html",
            interfaces=interfaces,
            result=None,
        )

    # --------------------------------------------------
    # 16. Save results
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
            f"live_predictions_"
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

    session[
        "uploaded_original_name"
    ] = (
        f"Live capture: "
        f"{ip1} <-> {ip2}"
    )

    # --------------------------------------------------
    # 17. Calculate class counts
    # --------------------------------------------------

    class_counts = (
        results_df[
            "Final_Label"
        ]
        .value_counts()
        .to_dict()
    )

    total_flows = len(
        results_df
    )

    benign_count = int(
        class_counts.get(
            "Benign",
            0,
        )
    )

    attack_count = (
        total_flows
        - benign_count
    )

    # --------------------------------------------------
    # 18. Build result
    # --------------------------------------------------

    result = {
        "no_traffic": False,

        "ip1": ip1,

        "ip2": ip2,

        "duration": duration,

        "interface": interface_index,

        "auto_repeat": auto_repeat,

        "total_flows": total_flows,

        "benign_count": benign_count,

        "attack_count": attack_count,

        "class_counts": class_counts,

        "rate_triggered": bool(
            heuristic_triggered
        ),

        "avg_confidence": round(
            float(
                confidence.mean()
            )
            * 100,
            2,
        ),

        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    # --------------------------------------------------
    # 19. Render page
    # --------------------------------------------------

    return render_template(
        "live_capture.html",
        interfaces=interfaces,
        result=result,
    )
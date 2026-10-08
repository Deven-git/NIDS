"""
app/routes/shap_explain.py
Lets the user browse SHAP explainability visuals per class:
  - Per-class feature importance (which features drive this class)
  - Local waterfall explanation (why one real example got this label)

Images are served directly from reports/figures/shap/ via
send_from_directory, rather than duplicating them into static/.
"""

import os
import joblib
from flask import Blueprint, render_template, send_from_directory, current_app, abort
from flask_login import login_required

shap_bp = Blueprint("shap_explain", __name__)


def get_class_names():
    le = joblib.load(os.path.join(current_app.config["MODELS_DIR"], "label_encoder.pkl"))
    return list(le.classes_)


@shap_bp.route("/shap-explanation")
@shap_bp.route("/shap-explanation/<class_name>")
@login_required
def shap_explanation(class_name=None):
    class_names = get_class_names()

    if class_name is None:
        class_name = class_names[0]
    elif class_name not in class_names:
        abort(404)

    safe_name = class_name.replace(" ", "_").lower()

    return render_template(
        "shap_explanation.html",
        class_names=class_names,
        selected_class=class_name,
        bar_image=f"shap_bar_{safe_name}.png",
        local_image=f"shap_local_{safe_name}.png",
    )


@shap_bp.route("/shap-image/<filename>")
@login_required
def shap_image(filename):
    shap_dir = os.path.join(current_app.config["REPORTS_DIR"], "figures", "shap")
    return send_from_directory(shap_dir, filename)
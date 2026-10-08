"""
app/routes/main.py
Core pages that require login. Right now just a placeholder home page -
Upload, Predict, Dashboard, etc. get added as their own routes/blueprints
in later steps.
"""

from flask import Blueprint, render_template
from flask_login import login_required, current_user

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
@login_required
def home():
    return render_template("home.html", username=current_user.username)
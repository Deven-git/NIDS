"""
app/routes/auth.py
Login/logout routes, grouped as a Blueprint.
"""

from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required

from app.database.models import User

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()

        # We check `user is not None` FIRST before calling check_password,
        # to avoid an error if the username doesn't exist at all.
        if user is not None and user.check_password(password):
            login_user(user)
            flash("Logged in successfully.", "success")
            return redirect(url_for("main.home"))

        # Deliberately vague error message - never reveal whether it was
        # the username or password that was wrong, as a basic security
        # practice (avoids helping an attacker enumerate valid usernames).
        flash("Invalid username or password.", "danger")

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))
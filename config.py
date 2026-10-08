"""
config.py
Central configuration for the Flask app.

IMPORTANT: this file lives at the PROJECT ROOT (nids-project/config.py),
so its own directory already IS the project root - no need to go up
another level.
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


class Config:
    # SECRET_KEY signs session cookies - in production this should come
    # from an environment variable, never hardcoded. For development,
    # a fallback is provided so the app still runs without extra setup.
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

    # SQLite database file, stored in app/database/
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(
        PROJECT_ROOT, "app", "database", "nids.db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Where uploaded CSV files get saved temporarily
    UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "data", "uploads_temp")
    ALLOWED_EXTENSIONS = {"csv"}
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB max upload size

    # Paths to the trained model artifacts (from earlier modules)
    MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
    REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
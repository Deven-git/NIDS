"""
app/__init__.py
The application factory.
"""

import os
from flask import Flask
from flask_login import LoginManager

from config import Config
from app.database.models import db, User


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(os.path.dirname(app.config["SQLALCHEMY_DATABASE_URI"].replace("sqlite:///", "")), exist_ok=True)

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    from app.routes.auth import auth_bp
    from app.routes.main import main_bp
    from app.routes.upload import upload_bp
    from app.routes.predict import predict_bp
    from app.routes.dashboard import dashboard_bp
    from app.routes.model_comparison import model_comparison_bp
    from app.routes.shap_explain import shap_bp
    from app.routes.report import report_bp
    from app.routes.live_capture_route import live_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(model_comparison_bp)
    app.register_blueprint(shap_bp)
    app.register_blueprint(report_bp)
    app.register_blueprint(live_bp)

    return app
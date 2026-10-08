"""
app/database/models.py
Database models. Currently just the User model for authentication.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """
    UserMixin (from Flask-Login) provides default implementations of
    the properties Flask-Login needs: is_authenticated, is_active,
    is_anonymous, and get_id(). We just define the actual data fields.
    """
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    def set_password(self, password):
        """Hashes and stores the password - we NEVER store plain text."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Verifies a plain-text password against the stored hash."""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.username}>"
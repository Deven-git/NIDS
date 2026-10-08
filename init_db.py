"""
init_db.py
Run ONCE to create the database tables and a default admin user.

Usage: python init_db.py
"""

from app import create_app
from app.database.models import db, User

app = create_app()

with app.app_context():
    db.create_all()
    print("Database tables created.")

    # Create a default admin user if one doesn't already exist
    existing = User.query.filter_by(username="admin").first()
    if existing is None:
        admin = User(username="admin")
        admin.set_password("admin123")  # CHANGE this after first login in a real deployment
        db.session.add(admin)
        db.session.commit()
        print("Default admin user created:")
        print("  username: admin")
        print("  password: admin123")
        print("  (change this password before any real deployment)")
    else:
        print("Admin user already exists - skipping creation.")
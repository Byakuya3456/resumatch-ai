# api package — registers all blueprints
from flask import Flask
from .routes.upload import upload_bp
from .routes.generate import generate_bp
from .routes.health import health_bp


def register_blueprints(app: Flask) -> None:
    """Attach all route blueprints to the Flask app."""
    app.register_blueprint(upload_bp)
    app.register_blueprint(generate_bp)
    app.register_blueprint(health_bp)

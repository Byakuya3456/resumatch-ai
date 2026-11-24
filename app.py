"""
app.py — Application entry point (minimal).

All logic lives in src/. This file only:
  - Creates the Flask app
  - Wires up blueprints
  - Serves the frontend HTML
  - Starts the dev server
"""
import logging
import sys
from pathlib import Path

from flask import Flask, send_from_directory
from flask_cors import CORS

# ── Add project root to sys.path so `config` and `src` are importable ──────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import config
from src.api import register_blueprints
from src.utils.serializers import JSONEncoder

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── App factory ────────────────────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    app.secret_key = config.SECRET_KEY
    app.json_encoder = JSONEncoder  # type: ignore[attr-defined]
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

    CORS(app)

    # Register all API blueprints
    register_blueprints(app)

    # Serve the frontend HTML from the project root
    @app.route("/")
    def serve_frontend():
        return send_from_directory(str(ROOT), "index.html")

    return app


# ── Dev server entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("🚀  AI Job Matcher  v2.0  (Modular + Upgraded Parsers)")
    print("=" * 55)
    print(f"   Embedding model : {config.EMBEDDING_MODEL}")
    print(f"   LLM model       : {config.OLLAMA_MODEL}")
    print(f"   MongoDB         : {config.MONGODB_URI}")
    print(f"   Debug mode      : {config.DEBUG}")
    print("=" * 55)

    application = create_app()
    application.run(
        debug=config.DEBUG,
        port=config.PORT,
        use_reloader=False,
    )

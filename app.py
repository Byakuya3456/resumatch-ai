"""
app.py — Headless Production JSON API Gateway.

Zero frontend dependencies. Exposes pure REST endpoints:
  - GET  /         -> API Root Index & Discovery catalog
  - POST /upload   -> Resume ingestion & parsing
  - POST /generate -> Hybrid RAG retrieval + LLM matching analysis
  - GET  /health   -> Comprehensive service healthcheck
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import config
from src.api import register_blueprints
from src.utils.serializers import JSONEncoder

logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api_gateway")


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = config.SECRET_KEY
    app.json_encoder = JSONEncoder  # type: ignore[attr-defined]
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

    # Enable cross-origin resource sharing for external frontend clients
    CORS(app)

    # Attach core route blueprints
    register_blueprints(app)

    # Root API Discovery & Capability Catalog
    @app.route("/", methods=["GET"])
    def api_root():
        return jsonify({
            "service": "AI Job Matcher API Gateway",
            "version": "2.0.0",
            "status": "online",
            "architecture": "Headless Modular Microservice",
            "endpoints": {
                "health": {
                    "method": "GET",
                    "path": "/health",
                    "description": "Liveness & readiness probe for MongoDB, PostgreSQL, and Ollama",
                },
                "upload_resume": {
                    "method": "POST",
                    "path": "/upload",
                    "consumes": "multipart/form-data",
                    "fields": {"file": "PDF or DOCX document (max 15MB)"},
                    "description": "Multi-strategy parsing (pymupdf4llm, pdfplumber, python-docx) + GLiNER zero-shot NER",
                },
                "generate_matches": {
                    "method": "POST",
                    "path": "/generate",
                    "consumes": "application/json",
                    "body": {"user_id": "Candidate 8-character ID returned from /upload"},
                    "description": "Hybrid search (BM25 + BGE-base Dense + RRF) + Ollama reasoning & skill gap analysis",
                },
            },
            "configuration": {
                "embedding_model": config.EMBEDDING_MODEL,
                "llm_provider": config.LLM_PROVIDER,
                "llm_model": config.ACTIVE_LLM_MODEL,
                "hybrid_search_enabled": config.HYBRID_SEARCH_ENABLED,
            },
        }), 200

    return app


if __name__ == "__main__":
    print("=" * 60)
    print("🚀  AI Job Matcher — Headless API Server")
    print("=" * 60)
    print(f"   API Listening   : http://localhost:{config.PORT}")
    print(f"   Embedding Model : {config.EMBEDDING_MODEL}")
    print(f"   LLM Provider    : {config.LLM_PROVIDER}")
    print(f"   LLM Model       : {config.ACTIVE_LLM_MODEL}")
    print(f"   Debug Mode      : {config.DEBUG}")
    print("=" * 60)

    application = create_app()
    application.run(
        debug=config.DEBUG,
        port=config.PORT,
        use_reloader=False,
    )

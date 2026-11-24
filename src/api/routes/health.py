"""
routes/health.py — GET /health

Liveness + readiness check endpoint.
Reports status of MongoDB and Ollama connections.
"""
from __future__ import annotations

import logging

import requests
from flask import Blueprint, jsonify

from config import config
from src.db.mongo import mongo_client, jobs_collection

logger = logging.getLogger(__name__)
health_bp = Blueprint("health", __name__)


def _check_mongodb() -> dict:
    try:
        mongo_client.server_info()
        job_count = jobs_collection.count_documents({})
        return {"status": "ok", "job_count": job_count}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def _check_ollama() -> dict:
    try:
        resp = requests.get(
            f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5
        )
        models = [m["name"] for m in resp.json().get("models", [])]
        active = config.OLLAMA_MODEL in models
        return {
            "status": "ok" if active else "model_missing",
            "configured_model": config.OLLAMA_MODEL,
            "available_models": models,
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@health_bp.route("/health", methods=["GET"])
def health_check():
    """
    GET /health
    Returns the health of all external dependencies.
    """
    mongo_status = _check_mongodb()
    ollama_status = _check_ollama()

    all_ok = (
        mongo_status["status"] == "ok"
        and ollama_status["status"] == "ok"
    )

    return jsonify({
        "status": "healthy" if all_ok else "degraded",
        "version": "2.0.0",
        "services": {
            "mongodb": mongo_status,
            "ollama": ollama_status,
        },
        "config": {
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_model": config.OLLAMA_MODEL,
            "rag_top_k": config.RAG_TOP_K,
        },
    }), 200 if all_ok else 503

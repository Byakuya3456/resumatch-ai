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


def _check_llm() -> dict:
    # 1. External Cloud LLM Provider (OpenAI, OpenRouter, Groq, DeepSeek, Custom)
    if config.LLM_PROVIDER in ("openai", "openrouter", "groq", "deepseek", "custom") or (
        bool(config.LLM_API_KEY) and config.LLM_PROVIDER != "ollama"
    ):
        has_key = bool(config.LLM_API_KEY)
        return {
            "status": "ok" if (has_key or config.LLM_PROVIDER == "custom") else "missing_api_key",
            "provider": config.LLM_PROVIDER,
            "configured_model": config.ACTIVE_LLM_MODEL,
            "base_url": config.LLM_BASE_URL,
        }

    # 2. Local Ollama Service
    try:
        resp = requests.get(
            f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5
        )
        models = [m["name"] for m in resp.json().get("models", [])]
        active = config.OLLAMA_MODEL in models
        return {
            "status": "ok" if active else "model_missing",
            "provider": "ollama",
            "configured_model": config.OLLAMA_MODEL,
            "available_models": models,
        }
    except Exception as exc:
        return {"status": "error", "provider": "ollama", "detail": str(exc)}


@health_bp.route("/health", methods=["GET"])
def health_check():
    """
    GET /health
    Returns the health of all external dependencies.
    """
    mongo_status = _check_mongodb()
    llm_status = _check_llm()

    all_ok = (
        mongo_status["status"] == "ok"
        and llm_status["status"] == "ok"
    )

    return jsonify({
        "status": "healthy" if all_ok else "degraded",
        "version": "2.0.0",
        "services": {
            "mongodb": mongo_status,
            "llm": llm_status,
            "ollama": llm_status,  # Backwards compatibility
        },
        "config": {
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_provider": config.LLM_PROVIDER,
            "llm_model": config.ACTIVE_LLM_MODEL,
            "rag_top_k": config.RAG_TOP_K,
        },
    }), 200 if all_ok else 503

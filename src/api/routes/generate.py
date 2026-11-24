"""
routes/generate.py — POST /generate

Runs the full RAG + LLM pipeline:
  1. Fetch resume from MongoDB by user_id
  2. Retrieve top-K relevant jobs via embedding RAG
  3. Build prompt and call Ollama
  4. Parse and return structured match results
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from config import config
from src.db.mongo import resumes_collection
from src.llm.ollama_client import call_ollama, generate_matches_prompt, parse_llm_response
from src.rag.simple_rag import get_relevant_jobs
from src.utils.serializers import serialize_document

logger = logging.getLogger(__name__)
generate_bp = Blueprint("generate", __name__)


@generate_bp.route("/generate", methods=["POST"])
def generate_matches():
    """
    POST /generate
    Body: { "user_id": "<8-char id>" }

    Returns:
        200 — { matches, rejection_analysis }
        400 — missing / bad user_id
        404 — resume or jobs not found
        500 — LLM or unexpected error
    """
    body = request.get_json(force=True, silent=True) or {}
    user_id: str = body.get("user_id", "").strip()

    if not user_id:
        return jsonify({"error": "user_id is required in the request body"}), 400

    # ── STEP 1: Load resume ────────────────────────────────────────────
    resume_doc = resumes_collection.find_one({"user_id": user_id})
    if not resume_doc:
        return jsonify({
            "error": f"No resume found for user_id: {user_id}",
            "message": "Please upload your resume first via POST /upload",
        }), 404

    resume_data = serialize_document(resume_doc)
    logger.info(
        "Processing resume for user %s — skills: %s",
        user_id, resume_data.get("skills", []),
    )

    # ── STEP 2: RAG — retrieve relevant jobs ──────────────────────────
    relevant_jobs = get_relevant_jobs(resume_data, top_k=config.RAG_TOP_K)
    if not relevant_jobs:
        return jsonify({
            "error": "No jobs found in database",
            "message": (
                "Ensure jobs are loaded into the "
                f"{config.JOBS_DB}.{config.JOBS_COLLECTION} collection."
            ),
        }), 404

    logger.info("RAG returned %d candidate jobs", len(relevant_jobs))

    # ── STEP 3: Build prompt ───────────────────────────────────────────
    prompt = generate_matches_prompt(
        resume_data=resume_data,
        relevant_jobs=relevant_jobs,
        top_n=config.LLM_TOP_K,
    )

    # ── STEP 4: Call LLM ──────────────────────────────────────────────
    logger.info("Calling Ollama model: %s", config.OLLAMA_MODEL)
    raw_response = call_ollama(prompt)

    if raw_response is None:
        logger.warning("Ollama unavailable — returning RAG fallback")
        result = parse_llm_response("", relevant_jobs, top_n=config.LLM_TOP_K)
        result["_warning"] = "LLM unavailable; results are RAG-ranked only."
        return jsonify(result), 200

    # ── STEP 5: Parse and return ───────────────────────────────────────
    result = parse_llm_response(raw_response, relevant_jobs, top_n=config.LLM_TOP_K)
    logger.info("Returning %d matches", len(result.get("matches", [])))
    return jsonify(result), 200

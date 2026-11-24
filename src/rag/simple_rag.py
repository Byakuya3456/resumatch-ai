"""
simple_rag.py — Embedding-based job retrieval (Phase 3 upgrade target).

Current: loads all jobs into RAM, encodes with sentence-transformers,
         ranks by cosine similarity.

Phase 3 target: replace with Qdrant (dense) + BM25 (sparse) + RRF fusion.
                Only this module needs to change — all callers are unaffected.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from config import config
from src.db.mongo import jobs_collection
from src.utils.serializers import serialize_document

logger = logging.getLogger(__name__)


# ── Load embedding model ───────────────────────────────────────────────────

def _load_embedding_model():
    """
    Load the preferred embedding model, falling back to MiniLM if needed.
    Upgrade: BAAI/bge-base-en-v1.5 (768-dim, MTEB 63.4) replaces
             all-MiniLM-L6-v2 (384-dim, MTEB 56.3).
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("✓ Embedding model loaded: %s", config.EMBEDDING_MODEL)
        return model
    except Exception as exc:
        logger.warning(
            "Failed to load %s (%s) — trying fallback %s",
            config.EMBEDDING_MODEL, exc, config.EMBEDDING_FALLBACK,
        )
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(config.EMBEDDING_FALLBACK)
            logger.info("✓ Fallback embedding model loaded: %s", config.EMBEDDING_FALLBACK)
            return model
        except Exception as exc2:
            logger.error("Both embedding models failed: %s", exc2)
            return None


_embedding_model = _load_embedding_model()


# ── Helper ─────────────────────────────────────────────────────────────────

def _build_job_text(job: dict) -> str:
    """Combine job fields into a single string for embedding."""
    return (
        f"Company: {job.get('company_name', '')}\n"
        f"Role: {job.get('job_role', job.get('job_title', ''))}\n"
        f"Description: {job.get('job_description', '')}\n"
        f"Skills: {', '.join(job.get('required_skills', []))}\n"
        f"Experience: {job.get('experience_level', '')}"
    )


def _build_resume_query(resume_data: dict) -> str:
    """Build a query string from the resume for embedding lookup."""
    # Use the full raw_text now (no 1000-char truncation)
    return (
        f"Skills: {', '.join(resume_data.get('skills', []))}\n"
        f"Experience: {', '.join(resume_data.get('experience', []))}\n"
        f"Background: {resume_data.get('raw_text', '')}"
    )


# ── Main retrieval function ────────────────────────────────────────────────

def get_relevant_jobs(
    resume_data: dict[str, Any],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve the top-K most relevant jobs for a given resume.

    Args:
        resume_data: Structured resume dict (from extract_resume_data).
        top_k:       Number of jobs to return. Defaults to config.RAG_TOP_K.

    Returns:
        List of serialized job dicts, ranked by cosine similarity.
    """
    top_k = top_k or config.RAG_TOP_K

    # ── Fallback: return recent jobs if model unavailable ──────────────
    if not _embedding_model:
        logger.warning("Embedding model unavailable — returning random jobs")
        return [
            serialize_document(j)
            for j in jobs_collection.find().limit(top_k)
        ]

    try:
        all_jobs = list(jobs_collection.find())
        if not all_jobs:
            logger.warning("No jobs found in database")
            return []

        # Build text representations
        job_texts = [_build_job_text(j) for j in all_jobs]
        resume_query = _build_resume_query(resume_data)

        # Encode in one batch (more efficient than per-job calls)
        logger.debug("Encoding %d job texts…", len(job_texts))
        job_embeddings = _embedding_model.encode(
            job_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        )
        resume_embedding = _embedding_model.encode(
            [resume_query], normalize_embeddings=True
        )[0]

        # Cosine similarity (vectors already L2-normalised → dot product)
        similarities = np.dot(job_embeddings, resume_embedding)

        # Top-K indices (highest similarity first)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        relevant_jobs = [all_jobs[i] for i in top_indices]

        logger.info("RAG returned %d relevant jobs (top score: %.3f)",
                    len(relevant_jobs), float(similarities[top_indices[0]]))

        return [serialize_document(j) for j in relevant_jobs]

    except Exception as exc:
        logger.error("RAG retrieval failed: %s — falling back to recent jobs", exc)
        return [
            serialize_document(j)
            for j in jobs_collection.find().limit(top_k)
        ]

"""
simple_rag.py — Hybrid Retrieval-Augmented Generation for jobs.

Combines:
  1. Dense Semantic Embeddings (BAAI/bge-base-en-v1.5 or all-MiniLM-L6-v2)
  2. Sparse Lexical Search (BM25Okapi)
  3. Reciprocal Rank Fusion (RRF) for production-grade retrieval
"""
from __future__ import annotations

import logging
from typing import Any
import numpy as np

from config import config
from src.db.mongo import jobs_collection
from src.utils.serializers import serialize_document
from .hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)


# ── Load embedding model ───────────────────────────────────────────────────

def _load_embedding_model():
    """
    Load the preferred embedding model, falling back to MiniLM if needed.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("✓ Embedding model loaded: %s", config.EMBEDDING_MODEL)
        return model
    except Exception as exc:
        logger.warning(
            "Failed to load %s (%s) — trying fallback %s",
            config.EMBEDDING_MODEL,
            exc,
            config.EMBEDDING_FALLBACK,
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
_hybrid_engine = HybridSearchEngine(rrf_k=config.RRF_K)
_indexed_job_count = -1


def _ensure_corpus_indexed():
    """Index jobs for hybrid search if not yet indexed or if count changed."""
    global _indexed_job_count
    try:
        current_count = jobs_collection.count_documents({})
        if current_count != _indexed_job_count or _indexed_job_count == -1:
            all_jobs = list(jobs_collection.find())
            _hybrid_engine.index_jobs(all_jobs, _embedding_model)
            _indexed_job_count = len(all_jobs)
    except Exception as exc:
        logger.warning("Could not auto-index jobs for hybrid search: %s", exc)


def _build_resume_query(resume_data: dict) -> str:
    """Build a search query from the resume for both sparse and dense retrieval."""
    skills_part = " ".join(resume_data.get("skills", []))
    exp_part = " ".join(resume_data.get("experience", []))
    raw_snippet = resume_data.get("raw_text", "")[:1200]
    return f"{skills_part} {exp_part} {raw_snippet}".strip()


# ── Main retrieval function ────────────────────────────────────────────────

def get_relevant_jobs(
    resume_data: dict[str, Any],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve top-K most relevant jobs for a given resume using Hybrid Search.
    """
    top_k = top_k or config.RAG_TOP_K

    try:
        # Check if database has any jobs
        all_jobs_count = jobs_collection.count_documents({})
        if all_jobs_count == 0:
            logger.warning("No jobs found in database")
            return []

        if config.HYBRID_SEARCH_ENABLED:
            _ensure_corpus_indexed()

            query_text = _build_resume_query(resume_data)
            query_vector: np.ndarray | None = None

            if _embedding_model:
                query_vector = _embedding_model.encode(
                    [query_text], normalize_embeddings=True
                )[0]

            ranked_results = _hybrid_engine.search(
                query_text=query_text,
                query_vector=query_vector,
                top_k=top_k,
                dense_weight=config.DENSE_SEARCH_WEIGHT,
            )

            relevant_jobs = [job for job, _score in ranked_results]
            logger.info(
                "✓ Hybrid Search (Dense+BM25+RRF) retrieved %d jobs", len(relevant_jobs)
            )
            return [serialize_document(j) for j in relevant_jobs]

        # Fallback if hybrid search disabled
        all_jobs = list(jobs_collection.find().limit(top_k))
        return [serialize_document(j) for j in all_jobs]

    except Exception as exc:
        logger.error("RAG retrieval failed: %s — falling back to limit query", exc)
        return [
            serialize_document(j)
            for j in jobs_collection.find().limit(top_k)
        ]

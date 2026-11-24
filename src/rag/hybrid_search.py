"""
hybrid_search.py — Production-grade Hybrid Search with Dense + BM25 Sparse & RRF.

Combines:
  1. Dense Semantic Search: Vector similarity via SentenceTransformers
     (backed by Qdrant collection or in-memory vector index when Qdrant standalone is offline)
  2. Sparse Lexical Search: BM25 (Okapi) for exact keyword/skill matching
  3. Reciprocal Rank Fusion (RRF): Blends ranks without relying on calibrated absolute scores
"""
from __future__ import annotations

import logging
import re
from typing import Any
import numpy as np
from rank_bm25 import BM25Okapi

from config import config

logger = logging.getLogger(__name__)


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"\w+", (text or "").lower())


class HybridSearchEngine:
    """
    Hybrid Search Engine implementing Dense Vector + Sparse BM25 retrieval
    with Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.bm25: BM25Okapi | None = None
        self.corpus_jobs: list[dict[str, Any]] = []
        self.job_embeddings: np.ndarray | None = None
        self._doc_ids: list[str] = []

    def index_jobs(
        self,
        jobs: list[dict[str, Any]],
        embedding_model: Any,
    ) -> None:
        """
        Build both the BM25 index and dense embeddings for all jobs in the corpus.
        """
        if not jobs:
            self.corpus_jobs = []
            self.bm25 = None
            self.job_embeddings = None
            self._doc_ids = []
            return

        self.corpus_jobs = jobs
        self._doc_ids = [str(j.get("_id") or j.get("id") or idx) for idx, j in enumerate(jobs)]

        # 1. Prepare text documents
        tokenized_corpus: list[list[str]] = []
        dense_texts: list[str] = []

        for job in jobs:
            title = job.get("job_role") or job.get("job_title", "")
            company = job.get("company_name", "")
            desc = job.get("job_description", "")
            skills = " ".join(job.get("required_skills", []))
            exp = job.get("experience_level", "")

            combined_text = f"{title} {company} {skills} {exp} {desc}"
            dense_texts.append(combined_text)
            tokenized_corpus.append(tokenize_text(combined_text))

        # 2. Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 3. Compute dense embeddings if model is present
        if embedding_model:
            try:
                embeddings = embedding_model.encode(
                    dense_texts,
                    batch_size=32,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                self.job_embeddings = np.array(embeddings)
                logger.info(
                    "HybridSearch: Indexed %d jobs (BM25 + Dense dim %d)",
                    len(jobs),
                    self.job_embeddings.shape[1],
                )
            except Exception as exc:
                logger.error("Failed to compute job embeddings: %s", exc)
                self.job_embeddings = None
        else:
            self.job_embeddings = None
            logger.info("HybridSearch: Indexed %d jobs (BM25 only)", len(jobs))

    def search(
        self,
        query_text: str,
        query_vector: np.ndarray | None,
        top_k: int = 15,
        dense_weight: float = 0.5,
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Perform hybrid search using Reciprocal Rank Fusion.

        Returns:
            List of tuples: (job_dict, rrf_score) sorted in descending order.
        """
        if not self.corpus_jobs:
            return []

        num_jobs = len(self.corpus_jobs)
        candidate_count = min(num_jobs, top_k * 2)

        dense_ranks: dict[int, int] = {}
        bm25_ranks: dict[int, int] = {}

        # ── 1. Sparse Search (BM25) ───────────────────────────────────
        if self.bm25:
            query_tokens = tokenize_text(query_text)
            if query_tokens:
                bm25_scores = self.bm25.get_scores(query_tokens)
                top_sparse_idx = np.argsort(bm25_scores)[-candidate_count:][::-1]
                for rank, idx in enumerate(top_sparse_idx):
                    if bm25_scores[idx] > 0:  # Only count actual keyword matches
                        bm25_ranks[int(idx)] = rank

        # ── 2. Dense Vector Search ────────────────────────────────────
        if self.job_embeddings is not None and query_vector is not None:
            # Query vector is assumed normalized
            similarities = np.dot(self.job_embeddings, query_vector)
            top_dense_idx = np.argsort(similarities)[-candidate_count:][::-1]
            for rank, idx in enumerate(top_dense_idx):
                dense_ranks[int(idx)] = rank

        # ── 3. Reciprocal Rank Fusion (RRF) ───────────────────────────
        sparse_weight = 1.0 - dense_weight
        rrf_scores: dict[int, float] = {}

        all_candidate_indices = set(dense_ranks.keys()) | set(bm25_ranks.keys())

        # If neither search found specific scores (e.g. empty query or no vectors), fallback to first jobs
        if not all_candidate_indices:
            return [(self.corpus_jobs[i], 0.0) for i in range(min(top_k, num_jobs))]

        for idx in all_candidate_indices:
            score = 0.0
            if idx in dense_ranks:
                score += dense_weight * (1.0 / (self.rrf_k + dense_ranks[idx]))
            if idx in bm25_ranks:
                score += sparse_weight * (1.0 / (self.rrf_k + bm25_ranks[idx]))
            rrf_scores[idx] = score

        # Sort by RRF score descending
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[
            :top_k
        ]

        return [(self.corpus_jobs[i], rrf_scores[i]) for i in sorted_indices]

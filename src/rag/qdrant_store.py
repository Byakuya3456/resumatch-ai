"""
qdrant_store.py — Optional Vector Store connector for Qdrant.

Supports:
  - In-memory client for testing or standalone execution without external services
  - Remote client connecting via QDRANT_URL for production clustering
"""
from __future__ import annotations

import logging
from typing import Any
from config import config

logger = logging.getLogger(__name__)

COLLECTION_NAME = "job_descriptions"


def get_qdrant_client():
    """
    Returns an active QdrantClient instance.
    Uses QDRANT_URL if specified in config, otherwise in-memory mode.
    """
    try:
        from qdrant_client import QdrantClient

        url = getattr(config, "QDRANT_URL", None)
        if url:
            logger.info("Connecting to Qdrant cluster at: %s", url)
            return QdrantClient(url=url)
        else:
            logger.info("Using embedded in-memory Qdrant instance")
            return QdrantClient(":memory:")
    except Exception as exc:
        logger.warning("Could not initialize QdrantClient: %s", exc)
        return None

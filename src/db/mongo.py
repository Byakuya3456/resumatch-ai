"""
mongo.py — MongoDB connection and collection handles.

Phase 2 upgrade target: replace with PostgreSQL + pgvector.
For now this module isolates all MongoDB concerns so the swap
only touches this single file.
"""
import sys
from pymongo import MongoClient
from pymongo.collection import Collection
from config import config


def _connect() -> tuple[MongoClient, Collection, Collection]:
    """
    Establish a MongoDB connection and return the client + collection handles.
    Exits the process if the connection cannot be established.
    """
    try:
        client = MongoClient(
            config.MONGODB_URI,
            serverSelectionTimeoutMS=config.MONGO_TIMEOUT_MS,
        )
        # Trigger actual network call to verify connectivity
        client.server_info()

        resumes_col = client[config.RESUME_DB][config.RESUME_COLLECTION]
        jobs_col = client[config.JOBS_DB][config.JOBS_COLLECTION]

        job_count = jobs_col.count_documents({})
        print(f"✓ MongoDB connected — jobs in DB: {job_count}")

        return client, resumes_col, jobs_col

    except Exception as exc:
        print(f"✗ MongoDB connection failed: {exc}")
        print("  → Ensure MongoDB is running or set MONGODB_URI in .env")
        sys.exit(1)


mongo_client, resumes_collection, jobs_collection = _connect()

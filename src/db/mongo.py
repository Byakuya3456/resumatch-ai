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


def _connect():
    """
    Establish a MongoDB connection and return the client + collection handles.
    Soft-fails gracefully if offline so offline tasks/tests don't crash the process immediately.
    """
    try:
        client = MongoClient(
            config.MONGODB_URI,
            serverSelectionTimeoutMS=config.MONGO_TIMEOUT_MS,
        )
        client.server_info()

        resumes_col = client[config.RESUME_DB][config.RESUME_COLLECTION]
        jobs_col = client[config.JOBS_DB][config.JOBS_COLLECTION]

        job_count = jobs_col.count_documents({})
        print(f"✓ MongoDB connected — jobs in DB: {job_count}")
        return client, resumes_col, jobs_col

    except Exception as exc:
        print(f"⚠️ MongoDB connection failed: {exc}")
        print("  → App will run with lazy/fallback connection. Ensure MongoDB is active when serving requests.")
        client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=1000)
        return client, client[config.RESUME_DB][config.RESUME_COLLECTION], client[config.JOBS_DB][config.JOBS_COLLECTION]


mongo_client, resumes_collection, jobs_collection = _connect()

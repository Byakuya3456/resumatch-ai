# db package
from .mongo import resumes_collection, jobs_collection, mongo_client

__all__ = ["resumes_collection", "jobs_collection", "mongo_client"]

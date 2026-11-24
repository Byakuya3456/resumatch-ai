"""
serializers.py — JSON serialization helpers for MongoDB documents.
"""
import json
from datetime import datetime
from bson import ObjectId


class JSONEncoder(json.JSONEncoder):
    """Custom encoder that handles MongoDB ObjectId and datetime objects."""

    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


def serialize_document(doc: dict | None) -> dict | None:
    """
    Recursively convert a MongoDB document to a JSON-safe plain dict.
    Handles nested ObjectId values as well.
    """
    if doc is None:
        return None

    result = {}
    for key, value in doc.items():
        if isinstance(value, ObjectId):
            result[key] = str(value)
        elif isinstance(value, datetime):
            result[key] = value.isoformat()
        elif isinstance(value, dict):
            result[key] = serialize_document(value)
        elif isinstance(value, list):
            result[key] = [
                serialize_document(item) if isinstance(item, dict) else
                str(item) if isinstance(item, ObjectId) else item
                for item in value
            ]
        else:
            result[key] = value

    return result

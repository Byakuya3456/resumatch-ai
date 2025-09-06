# db/mongo.py

from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from bson.objectid import ObjectId
from bson import Binary
import numpy as np
import pickle
import zlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from config import MONGODB_URI, MONGODB_DB, MONGODB_TIMEOUT_MS, EMBEDDING_DIMENSION
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class AdvancedMongoClient:
    """Advanced MongoDB client with embedding optimization and caching."""
    
    def __init__(self):
        self.client = None
        self.db = None
        self._initialize_connection()
        self._ensure_indexes()
    
    def _initialize_connection(self):
        """Initialize MongoDB connection with optimized settings."""
        try:
            self.client = MongoClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=MONGODB_TIMEOUT_MS,
                connectTimeoutMS=MONGODB_TIMEOUT_MS,
                socketTimeoutMS=MONGODB_TIMEOUT_MS,
                maxPoolSize=100,
                minPoolSize=10
            )
            self.db = self.client[MONGODB_DB]
            # Test connection
            self.client.admin.command('ping')
            logger.info("MongoDB connection established successfully")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
    
    def _ensure_indexes(self):
        """Create necessary indexes for optimal performance."""
        try:
            # Candidate indexes
            self.db.candidates.create_index([("embedding", "2dsphere")])
            self.db.candidates.create_index([("skills", ASCENDING)])
            self.db.candidates.create_index([("experience_level", ASCENDING)])
            self.db.candidates.create_index([("created_at", DESCENDING)])
            
            # Job indexes
            self.db.jobs.create_index([("embedding", "2dsphere")])
            self.db.jobs.create_index([("required_skills", ASCENDING)])
            self.db.jobs.create_index([("title", TEXT)])
            self.db.jobs.create_index([("company", ASCENDING)])
            self.db.jobs.create_index([("created_at", DESCENDING)])
            
            # Match indexes
            self.db.matches.create_index([("candidate_id", ASCENDING), ("job_id", ASCENDING)])
            self.db.matches.create_index([("score", DESCENDING)])
            self.db.matches.create_index([("created_at", DESCENDING)])
            
            logger.info("Database indexes ensured")
        except Exception as e:
            logger.warning(f"Index creation failed: {e}")
    
    def _compress_embedding(self, embedding: List[float]) -> bytes:
        """Compress embedding for storage efficiency."""
        if not embedding:
            return b''
        try:
            array = np.array(embedding, dtype=np.float32)
            compressed = zlib.compress(array.tobytes(), level=6)
            return compressed
        except Exception as e:
            logger.error(f"Embedding compression failed: {e}")
            return b''
    
    def _decompress_embedding(self, compressed_data: bytes) -> List[float]:
        """Decompress embedding from storage."""
        if not compressed_data:
            return []
        try:
            decompressed = zlib.decompress(compressed_data)
            array = np.frombuffer(decompressed, dtype=np.float32)
            return array.tolist()
        except Exception as e:
            logger.error(f"Embedding decompression failed: {e}")
            return []
    
    # ----------- Candidate Collection Methods ------------
    # In db/mongo.py - Add safe document conversion

    def safe_doc_to_dict(doc):
        """Safely convert MongoDB document to dict, handling missing _id."""
        if not doc:
            return None
        
        try:
            return doc_to_dict(doc)
        except KeyError as e:
            if '_id' in str(e):
                # Create a safe version without _id handling
                safe_doc = doc.copy()
                if '_id' in safe_doc:
                    safe_doc['id'] = str(safe_doc['_id'])
                    del safe_doc['_id']
                return safe_doc
            raise
    def insert_candidate(self, candidate_dict: Dict) -> str:
        """Insert candidate with optimized embedding storage."""
        try:
            candidate_dict = candidate_dict.copy()
            candidate_dict['created_at'] = datetime.utcnow()
            candidate_dict['updated_at'] = datetime.utcnow()
            
            # Compress embedding for storage
            if 'embedding' in candidate_dict:
                candidate_dict['embedding_compressed'] = self._compress_embedding(
                    candidate_dict.pop('embedding')
                )
            
            # Extract experience level
            candidate_dict['experience_level'] = self._extract_experience_level(
                candidate_dict.get('experience', '')
            )
            
            result = self.db.candidates.insert_one(candidate_dict)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to insert candidate: {e}")
            raise
    
    def get_candidate(self, candidate_id: str) -> Optional[Dict]:
        """Get candidate with decompressed embedding."""
        try:
            doc = self.db.candidates.find_one({"_id": ObjectId(candidate_id)})
            if doc:
                return self._prepare_document(doc)
            return None
        except Exception as e:
            logger.error(f"Failed to get candidate {candidate_id}: {e}")
            return None
    
    def find_candidates(self, query: Dict = None, limit: int = 10, skip: int = 0) -> List[Dict]:
        """Find candidates with pagination and embedding decompression."""
        query = query or {}
        try:
            cursor = self.db.candidates.find(query).skip(skip).limit(limit)
            return [self._prepare_document(doc) for doc in cursor]
        except Exception as e:
            logger.error(f"Failed to find candidates: {e}")
            return []
    
    def find_candidates_by_skills(self, skills: List[str], min_match: int = 1) -> List[Dict]:
        """Find candidates matching specific skills."""
        try:
            query = {"skills": {"$in": skills}}
            cursor = self.db.candidates.find(query)
            candidates = [self._prepare_document(doc) for doc in cursor]
            
            # Filter by minimum match count
            return [
                cand for cand in candidates
                if len(set(cand.get('skills', [])).intersection(set(skills))) >= min_match
            ]
        except Exception as e:
            logger.error(f"Failed to find candidates by skills: {e}")
            return []
    
    def find_similar_candidates(self, embedding: List[float], threshold: float = 0.7, limit: int = 10) -> List[Dict]:
        """Find similar candidates using vector similarity."""
        try:
            # MongoDB vector search query
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "embedding_index",
                        "path": "embedding",
                        "queryVector": embedding,
                        "numCandidates": 50,
                        "limit": limit
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "full_name": 1,
                        "skills": 1,
                        "experience": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$match": {
                        "score": {"$gte": threshold}
                    }
                }
            ]
            
            results = list(self.db.candidates.aggregate(pipeline))
            return [self._prepare_document(doc) for doc in results]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return self._fallback_similar_search(embedding, limit)
    
    def _fallback_similar_search(self, embedding: List[float], limit: int) -> List[Dict]:
        """Fallback similarity search for when vector search is unavailable."""
        try:
            all_candidates = self.find_candidates(limit=100)  # Get recent candidates
            from services.llm_wrapper import llm_wrapper
            
            scored_candidates = []
            for candidate in all_candidates:
                if candidate.get('embedding'):
                    score = llm_wrapper._cosine_similarity(embedding, candidate['embedding'])
                    scored_candidates.append((candidate, score))
            
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return [cand for cand, score in scored_candidates[:limit]]
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    # ------------- Job Collection Methods ----------------
    
    def insert_job(self, job_dict: Dict) -> str:
        """Insert job with optimized embedding storage."""
        try:
            job_dict = job_dict.copy()
            job_dict['created_at'] = datetime.utcnow()
            job_dict['updated_at'] = datetime.utcnow()
            job_dict['is_active'] = True
            
            # Compress embedding
            if 'embedding' in job_dict:
                job_dict['embedding_compressed'] = self._compress_embedding(
                    job_dict.pop('embedding')
                )
            
            result = self.db.jobs.insert_one(job_dict)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to insert job: {e}")
            raise
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job with decompressed embedding."""
        try:
            doc = self.db.jobs.find_one({"_id": ObjectId(job_id)})
            if doc:
                return self._prepare_document(doc)
            return None
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None
    
    def find_jobs(self, query: Dict = None, limit: int = 10, skip: int = 0) -> List[Dict]:
        """Find jobs with pagination."""
        query = query or {}
        query['is_active'] = True  # Only active jobs by default
        
        try:
            cursor = self.db.jobs.find(query).skip(skip).limit(limit)
            return [self._prepare_document(doc) for doc in cursor]
        except Exception as e:
            logger.error(f"Failed to find jobs: {e}")
            return []
    
    def find_jobs_by_skills(self, skills: List[str], min_match: int = 2) -> List[Dict]:
        """Find jobs requiring specific skills."""
        try:
            query = {
                "required_skills": {"$in": skills},
                "is_active": True
            }
            cursor = self.db.jobs.find(query)
            jobs = [self._prepare_document(doc) for doc in cursor]
            
            # Filter by minimum match count
            return [
                job for job in jobs
                if len(set(job.get('required_skills', [])).intersection(set(skills))) >= min_match
            ]
        except Exception as e:
            logger.error(f"Failed to find jobs by skills: {e}")
            return []
    
    # ------------- Match Collection Methods --------------
    
    def insert_match(self, match_dict: Dict) -> str:
        """Insert match result with additional analytics."""
        try:
            match_dict = match_dict.copy()
            match_dict['created_at'] = datetime.utcnow()
            match_dict['match_date'] = datetime.utcnow().date()
            
            result = self.db.matches.insert_one(match_dict)
            
            # Update candidate match count
            self.db.candidates.update_one(
                {"_id": ObjectId(match_dict['candidate_id'])},
                {"$inc": {"match_count": 1}, "$set": {"last_matched": datetime.utcnow()}}
            )
            
            # Update job match count
            self.db.jobs.update_one(
                {"_id": ObjectId(match_dict['job_id'])},
                {"$inc": {"match_count": 1}, "$set": {"last_matched": datetime.utcnow()}}
            )
            
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to insert match: {e}")
            raise
    
    def get_match(self, match_id: str) -> Optional[Dict]:
        """Get match result."""
        try:
            doc = self.db.matches.find_one({"_id": ObjectId(match_id)})
            return self._prepare_document(doc) if doc else None
        except Exception as e:
            logger.error(f"Failed to get match {match_id}: {e}")
            return None
    
    def find_matches(self, query: Dict = None, limit: int = 10, skip: int = 0) -> List[Dict]:
        """Find matches with pagination."""
        query = query or {}
        try:
            cursor = self.db.matches.find(query).skip(skip).limit(limit)
            return [self._prepare_document(doc) for doc in cursor]
        except Exception as e:
            logger.error(f"Failed to find matches: {e}")
            return []
    
    def get_match_stats(self, days: int = 30) -> Dict:
        """Get matching statistics for dashboard."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            pipeline = [
                {
                    "$match": {
                        "created_at": {"$gte": start_date}
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "total_matches": {"$sum": 1},
                        "avg_score": {"$avg": "$score"},
                        "high_quality_matches": {
                            "$sum": {
                                "$cond": [{"$gte": ["$score", 0.7]}, 1, 0]
                            }
                        }
                    }
                }
            ]
            
            result = list(self.db.matches.aggregate(pipeline))
            if result:
                stats = result[0]
                return {
                    "total_matches": stats.get('total_matches', 0),
                    "average_score": round(stats.get('avg_score', 0), 3),
                    "high_quality_matches": stats.get('high_quality_matches', 0),
                    "time_period_days": days
                }
            return {
                "total_matches": 0,
                "average_score": 0,
                "high_quality_matches": 0,
                "time_period_days": days
            }
        except Exception as e:
            logger.error(f"Failed to get match stats: {e}")
            return {}
    
    # ------------- Utility Methods ----------------------
    
    # In db/mongo.py - Update the _prepare_document method

    def _prepare_document(self, doc: Dict) -> Dict:
        """Prepare MongoDB document for API response with error handling."""
        if not doc:
            return None
        
        doc = doc.copy()
        
        # Handle _id field safely
        if '_id' in doc:
            doc["id"] = str(doc["_id"])
            del doc["_id"]
        else:
            # If no _id, generate a placeholder or use existing id
            doc.setdefault("id", "unknown_id")
        
        # Decompress embedding if exists
        if 'embedding_compressed' in doc:
            doc['embedding'] = self._decompress_embedding(doc.pop('embedding_compressed'))
        
        # Convert ObjectId fields to strings
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                doc[key] = str(value)
            elif isinstance(value, datetime):
                doc[key] = value.isoformat()
        
        return doc
    
    def _extract_experience_level(self, experience_text: str) -> str:
        """Extract experience level from text."""
        if not experience_text:
            return "unknown"
        
        text_lower = experience_text.lower()
        
        if any(word in text_lower for word in ['senior', 'lead', 'principal', 'manager', 'director']):
            return "senior"
        elif any(word in text_lower for word in ['mid', 'mid-level', 'experienced']):
            return "mid"
        elif any(word in text_lower for word in ['junior', 'entry', 'graduate', 'fresh']):
            return "junior"
        else:
            # Try to extract from years
            import re
            year_match = re.search(r'(\d+)\s*years?', text_lower)
            if year_match:
                years = int(year_match.group(1))
                if years >= 5:
                    return "senior"
                elif years >= 2:
                    return "mid"
                else:
                    return "junior"
            return "unknown"
    
    def close_connection(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

# Singleton instance
mongo_client = AdvancedMongoClient()

# Backward compatibility functions
def insert_candidate(candidate_dict):
    return mongo_client.insert_candidate(candidate_dict)

def get_candidate(candidate_id):
    return mongo_client.get_candidate(candidate_id)

def find_candidates(query=None, limit=10):
    return mongo_client.find_candidates(query, limit)

def insert_job(job_dict):
    return mongo_client.insert_job(job_dict)

def get_job(job_id):
    return mongo_client.get_job(job_id)

def find_jobs(query=None, limit=10):
    return mongo_client.find_jobs(query, limit)

def insert_match(match_dict):
    return mongo_client.insert_match(match_dict)

def get_match(match_id):
    return mongo_client.get_match(match_id)

def find_matches(query=None, limit=10):
    return mongo_client.find_matches(query, limit)

def doc_to_dict(doc):
    return mongo_client._prepare_document(doc) if doc else None
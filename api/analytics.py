# api/analytics.py

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
from datetime import datetime, timedelta
from db.mongo import mongo_client
import logging

router = APIRouter(prefix="/analytics", tags=["Analytics"])
logger = logging.getLogger(__name__)

@router.get("/dashboard")
async def get_dashboard_analytics(days: int = 30):
    """Get comprehensive dashboard analytics."""
    try:
        # Match statistics
        match_stats = mongo_client.get_match_stats(days)
        
        # Candidate statistics
        candidate_stats = await _get_candidate_stats(days)
        
        # Job statistics
        job_stats = await _get_job_stats(days)
        
        # Skill analytics
        skill_analytics = await _get_skill_analytics()
        
        return {
            "time_period": f"last_{days}_days",
            "match_analytics": match_stats,
            "candidate_analytics": candidate_stats,
            "job_analytics": job_stats,
            "skill_analytics": skill_analytics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Dashboard analytics failed: {e}")
        raise HTTPException(status_code=500, detail="Analytics generation failed")

@router.get("/skills/trending")
async def get_trending_skills(limit: int = 10):
    """Get trending skills in the market."""
    try:
        pipeline = [
            {
                "$unwind": "$required_skills"
            },
            {
                "$group": {
                    "_id": "$required_skills",
                    "count": {"$sum": 1},
                    "avg_salary": {"$avg": "$salary_range.min"},
                    "companies": {"$addToSet": "$company"}
                }
            },
            {
                "$sort": {"count": -1}
            },
            {
                "$limit": limit
            }
        ]
        
        results = list(mongo_client.db.jobs.aggregate(pipeline))
        return [
            {
                "skill": doc["_id"],
                "demand_count": doc["count"],
                "average_salary": doc.get("avg_salary"),
                "companies": list(doc.get("companies", []))[:5]
            }
            for doc in results
        ]
    except Exception as e:
        logger.error(f"Trending skills analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Skills analysis failed")

@router.get("/matches/quality")
async def get_match_quality_metrics(days: int = 30):
    """Get match quality metrics."""
    try:
        pipeline = [
            {
                "$match": {
                    "created_at": {
                        "$gte": datetime.utcnow() - timedelta(days=days)
                    }
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_matches": {"$sum": 1},
                    "avg_score": {"$avg": "$overall_score"},
                    "high_quality_matches": {
                        "$sum": {"$cond": [{"$gte": ["$overall_score", 0.7]}, 1, 0]}
                    },
                    "medium_quality_matches": {
                        "$sum": {"$cond": [
                            {"$and": [
                                {"$gte": ["$overall_score", 0.4]},
                                {"$lt": ["$overall_score", 0.7]}
                            ]}, 1, 0]
                        }
                    },
                    "low_quality_matches": {
                        "$sum": {"$cond": [{"$lt": ["$overall_score", 0.4]}, 1, 0]}
                    }
                }
            }
        ]
        
        results = list(mongo_client.db.matches.aggregate(pipeline))
        if results:
            return results[0]
        return {}
    except Exception as e:
        logger.error(f"Match quality analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Match quality analysis failed")

async def _get_candidate_stats(days: int) -> Dict[str, Any]:
    """Get candidate statistics."""
    try:
        pipeline = [
            {
                "$match": {
                    "created_at": {
                        "$gte": datetime.utcnow() - timedelta(days=days)
                    }
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_candidates": {"$sum": 1},
                    "avg_experience": {"$avg": "$total_experience_years"},
                    "experience_levels": {
                        "$push": "$experience_level"
                    }
                }
            }
        ]
        
        results = list(mongo_client.db.candidates.aggregate(pipeline))
        if results:
            result = results[0]
            return {
                "total_candidates": result["total_candidates"],
                "average_experience": round(result.get("avg_experience", 0), 1),
                "experience_distribution": _count_values(result.get("experience_levels", []))
            }
        return {}
    except Exception as e:
        logger.error(f"Candidate stats failed: {e}")
        return {}

async def _get_job_stats(days: int) -> Dict[str, Any]:
    """Get job statistics."""
    try:
        pipeline = [
            {
                "$match": {
                    "created_at": {
                        "$gte": datetime.utcnow() - timedelta(days=days)
                    },
                    "is_active": True
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_jobs": {"$sum": 1},
                    "companies": {"$addToSet": "$company"},
                    "experience_levels": {
                        "$push": "$experience_level"
                    }
                }
            }
        ]
        
        results = list(mongo_client.db.jobs.aggregate(pipeline))
        if results:
            result = results[0]
            return {
                "total_jobs": result["total_jobs"],
                "unique_companies": len(result.get("companies", [])),
                "experience_distribution": _count_values(result.get("experience_levels", []))
            }
        return {}
    except Exception as e:
        logger.error(f"Job stats failed: {e}")
        return {}

async def _get_skill_analytics() -> Dict[str, Any]:
    """Get skill analytics."""
    try:
        # Most demanded skills in jobs
        job_skills_pipeline = [
            {"$unwind": "$required_skills"},
            {"$group": {"_id": "$required_skills", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 15}
        ]
        
        job_skills = list(mongo_client.db.jobs.aggregate(job_skills_pipeline))
        
        # Most common candidate skills
        candidate_skills_pipeline = [
            {"$unwind": "$skills"},
            {"$group": {"_id": "$skills", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 15}
        ]
        
        candidate_skills = list(mongo_client.db.candidates.aggregate(candidate_skills_pipeline))
        
        return {
            "most_demanded_skills": [
                {"skill": doc["_id"], "job_count": doc["count"]}
                for doc in job_skills
            ],
            "most_common_skills": [
                {"skill": doc["_id"], "candidate_count": doc["count"]}
                for doc in candidate_skills
            ]
        }
    except Exception as e:
        logger.error(f"Skill analytics failed: {e}")
        return {}

def _count_values(values: List) -> Dict:
    """Count occurrences of values in a list."""
    from collections import Counter
    return dict(Counter(values))
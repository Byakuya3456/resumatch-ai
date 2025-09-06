# api/recruiter.py

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List
from db.mongo import insert_job, find_jobs, find_candidates, doc_to_dict, get_job
from services.llm_wrapper import AdvancedLMWrapper as LMWrapper
from services.matcher import AdvancedMatcher as Matcher
from db.mongo import AdvancedMongoClient as mongo_client
from models.job import Job
from config import MAX_CANDIDATE_RESULTS
import logging
import datetime
from datetime import datetime, timedelta  # Add this import at the top
import time
router = APIRouter(prefix="/recruiter", tags=["Recruiter"])

llm = LMWrapper()
matcher = Matcher()

logger = logging.getLogger(__name__)

@router.get("/test", summary="Test endpoint")
async def test_endpoint():
    return {"message": "Recruiter API is working!", "status": "success"}

# api/recruiter.py - Add this to relax validation temporarily

# In api/recruiter.py - Fix job insertion

@router.post("/post_job", summary="Post a new job")
async def post_job(job_data: dict = Body(...)):
    """
    Post a new job listing with proper ID handling.
    """
    try:
        # Basic validation
        if not job_data.get('title') or not job_data.get('description'):
            raise HTTPException(status_code=400, detail="Title and description are required")
        
        # Remove any existing ID to let MongoDB generate it
        job_data.pop('id', None)
        job_data.pop('_id', None)
        
        # Set default values
        job_data.setdefault('required_skills', [])
        job_data.setdefault('preferred_skills', [])
        job_data.setdefault('department', 'General')
        job_data.setdefault('is_active', True)
        job_data.setdefault('created_at', datetime.now())
        
        # Generate embedding
        llm = LMWrapper()
        combined_text = (f"{job_data.get('title', '')} at {job_data.get('department', '')}. "
                     f"Required Skills: {', '.join(job_data.get('required_skills', []))}. "
                     f"Description: {job_data.get('description', '')}")
        embedding = llm.get_embeddings(combined_text)
        job_data['embedding'] = embedding

        job_id = insert_job(job_data)
        logger.info(f"Posted new job with id: {job_id}")

        return {
            "jobId": job_id,
            "title": job_data.get('title'),
            "department": job_data.get('department'),
            "requiredSkills": job_data.get('required_skills', []),
            "message": "Job posted successfully",
            "embeddingGenerated": bool(embedding)
        }
        
    except Exception as e:
        logger.error(f"Failed to post job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create job listing")

# In api/recruiter.py - Update jobs_overview endpoint

@router.get("/jobs_overview", summary="Get recruiter dashboard overview")
async def jobs_overview(days: int = 30):
    """
    Get comprehensive overview for recruiter dashboard with error handling.
    """
    try:
        # Get jobs with error handling
        jobs = find_jobs(limit=20)
        safe_jobs = []
        
        for job in jobs:
            try:
                safe_jobs.append(doc_to_dict(job))
            except Exception as e:
                logger.warning(f"Skipping invalid job document: {e}")
                continue
        
        active_jobs = len([job for job in safe_jobs if job.get('is_active', True)])
        
        # Get match statistics with error handling
        try:
            match_stats = mongo_client.get_match_stats(days)
        except Exception:
            match_stats = {"total_matches": 0, "average_score": 0}
        
        # Filter jobs created within the last 7 days
        new_jobs_this_week = [
            j for j in safe_jobs
            if j.get('created_at') and
            datetime.fromisoformat(j['created_at'].replace('Z', '+00:00')) > datetime.now() - timedelta(days=7)
        ]

        return {
            "timePeriod": f"last_{days}_days",
            "jobStatistics": {
                "activeJobs": active_jobs,
                "totalJobs": len(safe_jobs),
                "newJobsThisWeek": len(new_jobs_this_week)
            },
            "matchStatistics": match_stats,
            "jobs": safe_jobs[:5]  # Only return safe documents
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate dashboard overview")

@router.get("/candidate_matches/{job_id}", summary="Find candidate matches for a job")
async def candidate_matches(job_id: str):
    job_doc = get_job(job_id)
    if not job_doc:
        raise HTTPException(status_code=404, detail="Job not found")

    job_dict = doc_to_dict(job_doc)
    candidates = find_candidates()
    candidate_dicts = [doc_to_dict(c) for c in candidates]

    matches = matcher.rank_job_to_candidates(job_dict, candidate_dicts, MAX_CANDIDATE_RESULTS)
    return {"jobId": job_id, "matches": matches}

# Additional recruiter endpoints can be added similarly if needed

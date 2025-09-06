# api/candidate.py

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List
from db.mongo import insert_candidate, find_jobs, doc_to_dict
from services.file_parser import FileParser
from services.llm_wrapper import AdvancedLMWrapper as LMWrapper
from services.matcher import AdvancedMatcher as Matcher
from models.candidate import Candidate
from config import ALLOWED_CANDIDATE_EXTENSIONS, MAX_JOB_RESULTS
import logging
import traceback
router = APIRouter(prefix="/candidate", tags=["Candidate"])

llm = LMWrapper()
matcher = Matcher()

logger = logging.getLogger(__name__)

# api/candidate.py - Fix the matcher usage
@router.get("/test", summary="Test endpoint")
async def test_endpoint():
    return {"message": "Candidate API is working!", "status": "success"}

@router.post("/upload_resume", summary="Upload and process candidate resume")
async def upload_resume(file: UploadFile = File(...)):
    try:
        logger.info(f"Received resume upload: {file.filename}")
        
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
            
        ext = f".{file.filename.lower().rsplit('.', 1)[-1]}" if '.' in file.filename else ""
        if ext not in ALLOWED_CANDIDATE_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Only {ALLOWED_CANDIDATE_EXTENSIONS} files allowed")

        # Read file content
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Extract text
        text = FileParser.extract_text(file_bytes, ext)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Failed to extract text from resume")

        logger.info(f"Successfully extracted text from resume: {len(text)} characters")

        # Basic data extraction
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        full_name = lines[0] if lines else "Unknown Candidate"
        
        # Initialize services
        llm = LMWrapper()
        matcher = Matcher()  # This should create an AdvancedMatcher instance
        
        # Use the correct method name based on your matcher class
        skills = matcher.extract_skills(text)  # This should work now
        experience = "Experience details from resume"
        education = "Education details from resume"

        # Generate embedding
        combined_text = f"{full_name}. Skills: {', '.join(skills)}. Experience: {experience}. Education: {education}"
        embedding = llm.get_embeddings(combined_text)
        
        logger.info(f"Generated embedding: {len(embedding) if embedding else 0} dimensions")

        # Create candidate profile - use simple dict for compatibility
        candidate_data = {
            "full_name": full_name,
            "skills": skills,
            "experience": experience,
            "education": education,
            "embedding": embedding,
            "resume_filename": file.filename
        }

        # Store candidate
        candidate_id = insert_candidate(candidate_data)
        logger.info(f"Stored candidate profile with id: {candidate_id}")

        # Find matches
        jobs = find_jobs(limit=MAX_JOB_RESULTS)
        job_dicts = [doc_to_dict(job) for job in jobs]

        matches = matcher.rank_candidate_to_jobs(candidate_data, job_dicts)

        return JSONResponse({
            "candidateId": candidate_id,
            "fullName": full_name,
            "skills": skills,
            "experience": experience,
            "education": education,
            "topJobMatches": matches,
            "message": "Resume processed successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume upload failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error during resume processing")
# You can add more candidate-related endpoints as needed, e.g., retrieve profile, update, etc.

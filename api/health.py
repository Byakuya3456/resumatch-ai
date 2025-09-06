# api/health.py

from fastapi import APIRouter, HTTPException
from datetime import datetime
import psutil
import logging
from db.mongo import mongo_client
from services.llm_wrapper import llm_wrapper
from typing import Dict

router = APIRouter(prefix="/health", tags=["Health"])
logger = logging.getLogger(__name__)

@router.get("/system")
async def system_health():
    """Get detailed system health information."""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Database health
        db_health = await _check_database_health()
        
        # LLM health
        llm_health = await _check_llm_health()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_total": memory.total / 1024 / 1024,  # MB
                "memory_used": memory.used / 1024 / 1024,    # MB
                "memory_percent": memory.percent,
                "disk_total": disk.total / 1024 / 1024 / 1024,  # GB
                "disk_used": disk.used / 1024 / 1024 / 1024,    # GB
                "disk_percent": disk.percent
            },
            "process": {
                "memory_used_mb": round(process_memory, 2),
                "cpu_percent": process.cpu_percent(),
                "threads": process.num_threads()
            },
            "services": {
                "database": db_health,
                "llm": llm_health
            },
            "status": "healthy" if all([
                db_health["status"] == "healthy",
                llm_health["status"] == "healthy",
                cpu_percent < 90,
                memory.percent < 85
            ]) else "degraded"
        }
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@router.get("/database")
async def database_health():
    """Check database health specifically."""
    return await _check_database_health()

@router.get("/llm")
async def llm_health():
    """Check LLM service health."""
    return await _check_llm_health()

async def _check_database_health() -> Dict:
    """Check database connection and performance."""
    try:
        start_time = datetime.utcnow()
        
        # Test connection and basic query
        result = mongo_client.db.command("ping")
        ping_time = (datetime.utcnow() - start_time).total_seconds() * 1000  # ms
        
        # Check collection counts
        candidate_count = mongo_client.db.candidates.count_documents({})
        job_count = mongo_client.db.jobs.count_documents({})
        match_count = mongo_client.db.matches.count_documents({})
        
        return {
            "status": "healthy",
            "ping_time_ms": round(ping_time, 2),
            "collections": {
                "candidates": candidate_count,
                "jobs": job_count,
                "matches": match_count
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

async def _check_llm_health() -> Dict:
    """Check LLM service health."""
    try:
        start_time = datetime.utcnow()
        
        # Test with a simple embedding request
        test_text = "test health check"
        embedding = llm_wrapper.get_embeddings(test_text)
        
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000  # ms
        
        return {
            "status": "healthy",
            "response_time_ms": round(response_time, 2),
            "embedding_length": len(embedding) if embedding else 0,
            "model": llm_wrapper.model_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
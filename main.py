# main.py

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any
import time
import os
from pathlib import Path

# Create necessary directories
Path("logs").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

# Configure logging first
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = f"logs/resumatch_{time.strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Define middleware classes here since they're causing import issues
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {response.status_code} "
            f"Time: {process_time:.3f}s "
            f"Size: {response.headers.get('content-length', '0')} bytes"
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

class RateLimitingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 100, time_window: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                t for t in self.requests[client_ip] 
                if current_time - t < self.time_window
            ]
        else:
            self.requests[client_ip] = []
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": str(self.time_window)}
            )
        
        # Add current request
        self.requests[client_ip].append(current_time)
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        remaining = self.max_requests - len(self.requests[client_ip])
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.time_window))
        
        return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting ResuMatch AI Server...")
    logger.info(f"Log level: {LOG_LEVEL}")
    
    # Health check initialization
    app.state.startup_time = time.time()
    app.state.healthy = True
    
    yield
    
    # Shutdown
    logger.info("Shutting down ResuMatch AI Server...")
    try:
        # Import here to avoid circular imports
        from db.mongo import mongo_client
        mongo_client.close_connection()
        logger.info("MongoDB connection closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

app = FastAPI(
    title="ResuMatch AI - Advanced Resume Parsing and Job Matching",
    description="AI-powered job matching system with advanced ML and LLM integration",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Middleware - Comment out custom middleware for now to test
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Comment out custom middleware until basic functionality works
# app.add_middleware(LoggingMiddleware)
# app.add_middleware(RateLimitingMiddleware, max_requests=100, time_window=60)

# Exception handlers
# main.py - Fix the validation exception handler

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors gracefully."""
    # Convert ValueError objects to strings for JSON serialization
    cleaned_errors = []
    for error in exc.errors():
        cleaned_error = error.copy()
        # Convert any ValueError objects to strings
        if 'ctx' in cleaned_error and 'error' in cleaned_error['ctx']:
            if isinstance(cleaned_error['ctx']['error'], ValueError):
                cleaned_error['ctx']['error'] = str(cleaned_error['ctx']['error'])
        cleaned_errors.append(cleaned_error)
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": cleaned_errors,
        },
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Comprehensive health check."""
    status = {
        "status": "healthy",
        "version": "2.0.0",
        "uptime": round(time.time() - app.state.startup_time, 2),
        "timestamp": time.time(),
        "services": {
            "database": "unknown",
            "llm": "unknown",
            "ner": "unknown"
        }
    }
    return status

# Include API routers - import here to avoid circular imports
try:
    from api import candidate, recruiter
    app.include_router(candidate.router, prefix="/api/v1/candidate", tags=["Candidate"])
    app.include_router(recruiter.router, prefix="/api/v1/recruiter", tags=["Recruiter"])
    logger.info("Candidate and Recruiter routers loaded successfully")
except ImportError as e:
    logger.error(f"Failed to load API routers: {e}")

# Try to load optional routers
try:
    from api import analytics, health
    app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])
    app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])
    logger.info("Analytics and Health routers loaded successfully")
except ImportError:
    logger.warning("Analytics and Health routers not available")

# Serve static files (for resumes, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ResuMatch AI API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "candidate": "/api/v1/candidate",
            "recruiter": "/api/v1/recruiter"
        }
    }

# API information endpoint
@app.get("/api/info", tags=["API"])
async def api_info():
    """Get detailed API information."""
    return {
        "name": "ResuMatch AI",
        "version": "2.0.0",
        "description": "Advanced AI-powered job matching system",
        "features": [
            "Resume parsing with AI extraction",
            "Job matching with ML+LLM approach",
            "Real-time analytics",
            "Skill gap analysis"
        ]
    }

if __name__ == "__main__":
    # Development configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level=LOG_LEVEL.lower(),
        reload=True  # Enable auto-reload for development
    )
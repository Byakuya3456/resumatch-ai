
# config.py

import os
from datetime import datetime
from enum import Enum

# Database Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB = os.getenv("MONGODB_DB", "resumatch_ai")
MONGODB_TIMEOUT_MS = 5000

# LM Studio Configuration
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "StableLM-Zephyr-3B-Q4_K_M")
LLM_TIMEOUT = 45
LLM_MAX_TOKENS = 512

# ML Model Configuration - UPDATED with fallback
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "./models/xgboost_job_matcher.json")
USE_ML_MODEL = os.getenv("USE_ML_MODEL", "false").lower() == "true"  # Disable by default

# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
MAX_EMBEDDING_TEXT_LENGTH = 1000

# File Processing
ALLOWED_CANDIDATE_EXTENSIONS = [".pdf", ".docx", ".txt"]
MAX_FILE_SIZE_MB = 10
MAX_RESUME_PAGES = 10

# Matching Thresholds
class MatchThreshold(float, Enum):
    EXCELLENT = 0.8
    GOOD = 0.6
    FAIR = 0.4
    POOR = 0.2

TOP_N_RESULTS = 5
MIN_MATCH_SCORE = 0.3

# Cache Configuration - UPDATED with fallback
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
USE_REDIS = os.getenv("USE_REDIS", "false").lower() == "true"  # Disable by default
CACHE_TTL = 3600  # 1 hour

# Performance Optimization
BATCH_SIZE = 4
MAX_CONCURRENT_REQUESTS = 10

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = f"logs/resumatch_{datetime.now().strftime('%Y%m%d')}.log"

# Feature Flags - UPDATED with safe defaults
ENABLE_ADVANCED_NER = os.getenv("ENABLE_ADVANCED_NER", "false").lower() == "true"
ENABLE_SALARY_PREDICTION = False
ENABLE_SKILL_GAP_ANALYSIS = True

# Email validation fallback
ENABLE_EMAIL_VALIDATION = os.getenv("ENABLE_EMAIL_VALIDATION", "false").lower() == "true"
# Add these lines at the end of your config.py file:

# Matching limits
MAX_JOB_RESULTS = 10    # Top N jobs to return for candidates
MAX_CANDIDATE_RESULTS = 10  # Top N candidates to return for recruiters
"""
config.py — Centralised application configuration.
All settings loaded from environment variables with safe defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent / ".env")


class Config:
    # ── Paths ──────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).parent
    UPLOAD_FOLDER: Path = BASE_DIR / os.getenv("UPLOAD_FOLDER", "uploads")
    ALLOWED_EXTENSIONS: set = {"pdf", "docx"}
    MAX_CONTENT_LENGTH: int = 10 * 1024 * 1024  # 10 MB upload cap

    # ── Database ───────────────────────────────────────────
    # Primary: PostgreSQL + pgvector (Phase 2 target)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/jobmatcher"
    )

    # Secondary / Legacy: MongoDB
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    MONGO_TIMEOUT_MS: int = 5000
    RESUME_DB: str = "user_resumes"
    RESUME_COLLECTION: str = "resumes"
    JOBS_DB: str = "job_descriptions"
    JOBS_COLLECTION: str = "jobs"

    # ── Embedding model ────────────────────────────────────
    # Upgraded from all-MiniLM-L6-v2 (384-dim, MTEB 56.3)
    # to BAAI/bge-base-en-v1.5 (768-dim, MTEB 63.4)
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"
    )
    EMBEDDING_FALLBACK: str = "all-MiniLM-L6-v2"

    # ── LLM Configuration (Local Ollama vs External Cloud API) ────
    # Supported providers: "ollama", "openai", "openrouter", "groq", "deepseek", "custom"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower().strip()
    LLM_API_KEY: str | None = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
    )

    # Provider-specific default base URLs for OpenAI-compatible endpoints
    _PROVIDER_URLS = {
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "groq": "https://api.groq.com/openai/v1",
        "deepseek": "https://api.deepseek.com/v1",
    }

    LLM_BASE_URL: str = os.getenv(
        "LLM_BASE_URL",
        _PROVIDER_URLS.get(LLM_PROVIDER, "https://api.openai.com/v1"),
    )

    # Ollama local settings (when LLM_PROVIDER == "ollama")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_GENERATE_ENDPOINT: str = f"{OLLAMA_BASE_URL}/api/generate"
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))

    # Unified active model name and request timeout
    ACTIVE_LLM_MODEL: str = os.getenv("LLM_MODEL") or OLLAMA_MODEL
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", str(OLLAMA_TIMEOUT)))

    # ── RAG & Hybrid Search ────────────────────────────────
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "15"))
    LLM_TOP_K: int = int(os.getenv("LLM_TOP_K", "5"))
    HYBRID_SEARCH_ENABLED: bool = os.getenv("HYBRID_SEARCH_ENABLED", "1") == "1"
    RRF_K: int = int(os.getenv("RRF_K", "60"))
    DENSE_SEARCH_WEIGHT: float = float(os.getenv("DENSE_SEARCH_WEIGHT", "0.6"))
    QDRANT_URL: str | None = os.getenv("QDRANT_URL", None)

    # ── Parsers ────────────────────────────────────────────
    # Maximum raw_text characters stored. 0 = unlimited (recommended).
    # Old code had 1000 — raised to 0 to fix embedding quality.
    RAW_TEXT_MAX_CHARS: int = int(os.getenv("RAW_TEXT_MAX_CHARS", "0"))
    # Minimum text length for a PDF to skip OCR fallback
    OCR_FALLBACK_THRESHOLD: int = 100

    # ── Flask ──────────────────────────────────────────────
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production")
    DEBUG: bool = os.getenv("FLASK_DEBUG", "0") == "1"
    PORT: int = int(os.getenv("PORT", "5000"))

    # ── spaCy ──────────────────────────────────────────────
    SPACY_MODEL: str = os.getenv("SPACY_MODEL", "en_core_web_sm")


config = Config()

"""
resume_extractor.py — Structured data extraction from raw resume text.

Uses spaCy NER for named entities + an expanded regex/keyword layer.
Phase 4 upgrade target: replace with GLiNER zero-shot NER.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from config import config
from .gliner_extractor import extract_entities_zero_shot

logger = logging.getLogger(__name__)

# ── Skill taxonomy (expanded from original 18 → 80+ terms) ────────────────
_SKILLS_TAXONOMY: list[str] = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "c",
    "ruby", "go", "golang", "rust", "scala", "kotlin", "swift",
    "php", "r", "matlab", "julia", "perl", "bash", "shell",
    # Frontend
    "react", "angular", "vue", "next.js", "nuxt", "svelte",
    "html", "css", "sass", "tailwind", "bootstrap", "webpack",
    # Backend / API
    "node", "node.js", "express", "django", "flask", "fastapi",
    "spring", "spring boot", "asp.net", "graphql", "rest", "grpc",
    # Data / AI / ML
    "machine learning", "deep learning", "nlp", "computer vision",
    "data science", "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "matplotlib", "seaborn", "hugging face",
    "langchain", "llm", "rag", "openai", "bert", "transformers",
    # Databases
    "sql", "postgresql", "mysql", "sqlite", "mongodb", "redis",
    "elasticsearch", "cassandra", "dynamodb", "firebase", "supabase",
    "pgvector", "qdrant", "pinecone", "chroma",
    # Cloud / DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "terraform", "ansible", "jenkins", "github actions", "ci/cd",
    "linux", "nginx", "apache",
    # Tools / Methods
    "git", "agile", "scrum", "jira", "confluence", "postman",
    "figma", "tableau", "power bi", "excel", "spark", "kafka",
]

# Pre-compile the skill matcher pattern for efficiency
_SKILL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _SKILLS_TAXONOMY) + r")\b",
    re.IGNORECASE,
)

# ── Regex patterns ─────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
_PHONE_RE = re.compile(
    r"(?:\+?\d[\d\s\-\.\(\)]{7,}\d)"
)
_LINKEDIN_RE = re.compile(
    r"linkedin\.com/in/([A-Za-z0-9\-_%]+)", re.IGNORECASE
)
_GITHUB_RE = re.compile(
    r"github\.com/([A-Za-z0-9\-_%]+)", re.IGNORECASE
)
_YEARS_EXP_RE = re.compile(
    r"(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)",
    re.IGNORECASE,
)


def _load_spacy():
    """Lazy-load spaCy model with download fallback."""
    try:
        import spacy
        try:
            return spacy.load(config.SPACY_MODEL)
        except OSError:
            logger.info("Downloading spaCy model: %s", config.SPACY_MODEL)
            from spacy.cli import download
            download(config.SPACY_MODEL)
            return spacy.load(config.SPACY_MODEL)
    except Exception as exc:
        logger.warning("spaCy unavailable: %s — NER disabled", exc)
        return None


_nlp = _load_spacy()
logger.info("spaCy model: %s", "loaded" if _nlp else "unavailable")


# ── Main extraction function ───────────────────────────────────────────────

def extract_resume_data(text: str) -> dict[str, Any]:
    """
    Extract structured fields from raw resume text.

    Returns a dict with:
        name, email, phone, linkedin, github,
        skills, experience, education, certifications,
        years_experience, raw_text, extracted_date
    """
    if not text:
        return _empty_resume()

    # ── NER via spaCy ─────────────────────────────────────────────────
    persons: list[str] = []
    organizations: list[str] = []
    dates: list[str] = []

    if _nlp:
        doc = _nlp(text[:100_000])  # spaCy limits; use first 100k chars
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                persons.append(ent.text.strip())
            elif ent.label_ == "ORG":
                organizations.append(ent.text.strip())
            elif ent.label_ == "DATE":
                dates.append(ent.text.strip())

    # ── Regex extractions ─────────────────────────────────────────────
    emails = _EMAIL_RE.findall(text)
    phones = _PHONE_RE.findall(text)
    linkedin_matches = _LINKEDIN_RE.findall(text)
    github_matches = _GITHUB_RE.findall(text)
    years_matches = _YEARS_EXP_RE.findall(text)

    # ── Skill extraction via expanded taxonomy ─────────────────────────
    found_skills = list(
        {m.lower().strip() for m in _SKILL_PATTERN.findall(text)}
    )
    # Normalise to title-case for display
    found_skills = sorted(s.title() for s in found_skills)

    # ── Zero-Shot Extraction via GLiNER (Degrees, Certs, Dynamic Skills) ──
    gliner_res = extract_entities_zero_shot(text)
    if gliner_res["skills"]:
        # Merge taxonomy skills with dynamic zero-shot skills
        found_skills = sorted(list(set(found_skills + gliner_res["skills"])))

    # ── Determine raw_text slice ───────────────────────────────────────
    max_chars = config.RAW_TEXT_MAX_CHARS
    raw_text_stored = text if max_chars == 0 else text[:max_chars]

    return {
        "name": persons[0] if persons else _extract_name_heuristic(text),
        "email": emails[0] if emails else "",
        "phone": phones[0].strip() if phones else "",
        "linkedin": f"linkedin.com/in/{linkedin_matches[0]}" if linkedin_matches else "",
        "github": f"github.com/{github_matches[0]}" if github_matches else "",
        "skills": found_skills,
        "experience": list(dict.fromkeys(organizations[:8] + gliner_res.get("companies", [])[:4])),
        "education": gliner_res.get("education", []),
        "certifications": gliner_res.get("certifications", []),
        "years_experience": int(years_matches[0]) if years_matches else None,
        "raw_text": raw_text_stored,
        "extracted_date": datetime.now().isoformat(),
    }


def _extract_name_heuristic(text: str) -> str:
    """
    Fallback name extractor: assume the name is on the first non-empty line
    if it looks like a proper name (2-4 words, all capitalized, no digits).
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w) \
                and not any(c.isdigit() for c in line):
            return line
        break  # Only check first non-empty line
    return "Unknown"


def _empty_resume() -> dict[str, Any]:
    return {
        "name": "Unknown",
        "email": "",
        "phone": "",
        "linkedin": "",
        "github": "",
        "skills": [],
        "experience": [],
        "education": [],
        "certifications": [],
        "years_experience": None,
        "raw_text": "",
        "extracted_date": datetime.now().isoformat(),
    }

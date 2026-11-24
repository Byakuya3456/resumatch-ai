"""
gliner_extractor.py — Zero-Shot Named Entity Recognition using GLiNER.

Extracts dynamic entities without hardcoded keyword dictionaries:
  - Technical skills, languages, frameworks, developer tools
  - Universities, degrees, educational institutions
  - Certifications & credentials
  - Company names & job titles
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Primary resume entity labels for GLiNER zero-shot detection
RESUME_LABELS = [
    "person name",
    "email address",
    "phone number",
    "technical skill",
    "programming language",
    "framework",
    "software tool",
    "cloud platform",
    "database",
    "company name",
    "job title",
    "university",
    "degree",
    "certification",
]

_gliner_model = None


def _get_gliner_model():
    """Lazy load GLiNER small/medium model."""
    global _gliner_model
    if _gliner_model is None:
        try:
            from gliner import GLiNER
            # Lightweight, high-accuracy multi-task model
            _gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
            logger.info("✓ GLiNER zero-shot NER model initialized (gliner_small-v2.1)")
        except Exception as exc:
            logger.warning("Could not initialize GLiNER model: %s", exc)
            _gliner_model = None
    return _gliner_model


def extract_entities_zero_shot(text: str, threshold: float = 0.45) -> dict[str, Any]:
    """
    Predict zero-shot entities using GLiNER.
    Returns:
        {
            "skills": [...],
            "education": [...],
            "certifications": [...],
            "companies": [...],
            "roles": [...]
        }
    """
    model = _get_gliner_model()
    if not model or not text:
        return {
            "skills": [],
            "education": [],
            "certifications": [],
            "companies": [],
            "roles": [],
        }

    try:
        # GLiNER works best on chunks up to ~2000 words
        snippet = text[:4000]
        entities = model.predict_entities(snippet, RESUME_LABELS, threshold=threshold)

        skills: set[str] = set()
        education: set[str] = set()
        certifications: set[str] = set()
        companies: set[str] = set()
        roles: set[str] = set()

        for ent in entities:
            label = ent.get("label", "")
            val = ent.get("text", "").strip()
            if not val or len(val) < 2:
                continue

            if label in ("technical skill", "programming language", "framework", "software tool", "cloud platform", "database"):
                skills.add(val.title())
            elif label in ("university", "degree"):
                education.add(val)
            elif label == "certification":
                certifications.add(val)
            elif label == "company name":
                companies.add(val)
            elif label == "job title":
                roles.add(val)

        return {
            "skills": sorted(list(skills)),
            "education": sorted(list(education)),
            "certifications": sorted(list(certifications)),
            "companies": sorted(list(companies)),
            "roles": sorted(list(roles)),
        }

    except Exception as exc:
        logger.warning("GLiNER extraction error: %s", exc)
        return {
            "skills": [],
            "education": [],
            "certifications": [],
            "companies": [],
            "roles": [],
        }

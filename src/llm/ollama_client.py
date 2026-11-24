"""
ollama_client.py — Ollama API wrapper + prompt construction.

Phase 5 upgrade target: replace with LangChain + Pydantic structured output.
For now: clean separation so callers don't depend on raw requests logic.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import requests

from config import config

logger = logging.getLogger(__name__)


# ── Low-level API call ─────────────────────────────────────────────────────

def call_ollama(prompt: str) -> str | None:
    """
    Send a prompt to Ollama and return the raw text response.

    Returns:
        Response string, or None if the call fails.
    """
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,   # Lower → more deterministic JSON
            "top_p": 0.9,
            "top_k": 40,
        },
    }
    try:
        resp = requests.post(
            config.OLLAMA_GENERATE_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=config.OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        logger.error(
            "Cannot reach Ollama at %s — is it running?",
            config.OLLAMA_BASE_URL,
        )
    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out after %ds", config.OLLAMA_TIMEOUT)
    except requests.exceptions.HTTPError as exc:
        logger.error("Ollama HTTP error: %s", exc)
    except Exception as exc:
        logger.error("Unexpected Ollama error: %s", exc)
    return None


# ── Prompt builder ─────────────────────────────────────────────────────────

def generate_matches_prompt(
    resume_data: dict[str, Any],
    relevant_jobs: list[dict[str, Any]],
    top_n: int | None = None,
) -> str:
    """
    Build the job-matching prompt to send to the LLM.

    Keeps prompt construction in one place so it can be unit-tested
    and updated independently of the API route.
    """
    top_n = top_n or config.LLM_TOP_K
    jobs_json = json.dumps(relevant_jobs, indent=2)

    return f"""
TASK: Analyze the resume against the provided job listings and return the top {top_n} matches.

RESUME DATA:
- Name: {resume_data.get('name', 'Unknown')}
- Skills: {resume_data.get('skills', [])}
- Experience (Organisations): {resume_data.get('experience', [])}
- Years of experience: {resume_data.get('years_experience', 'Unknown')}
- LinkedIn: {resume_data.get('linkedin', 'N/A')}
- GitHub: {resume_data.get('github', 'N/A')}
- Background (excerpt):
{resume_data.get('raw_text', '')[:800]}

AVAILABLE JOBS (from database — {len(relevant_jobs)} candidates):
{jobs_json}

INSTRUCTIONS:
1. Analyse each job against the resume carefully.
2. Select the top {top_n} most suitable matches only from the jobs listed above.
3. For every match provide: company_name, job_title, match_reason, match_score (0-100 integer).
4. Identify skill gaps and give improvement suggestions.
5. Only use job data provided — do not invent companies or roles.

RETURN ONLY VALID JSON in exactly this format (no markdown, no extra text):
{{
  "matches": [
    {{
      "company_name": "Exact Company Name",
      "job_title": "Exact Job Title",
      "match_reason": "Specific reason based on skills and experience",
      "match_score": 85
    }}
  ],
  "rejection_analysis": {{
    "potential_rejections": [
      {{
        "company": "Company Name",
        "reason": "Specific rejection reason",
        "skill_gaps": ["Skill1", "Skill2"]
      }}
    ],
    "skill_development": {{
      "critical_missing_skills": ["Skill1", "Skill2"],
      "improvement_suggestions": ["Suggestion1", "Suggestion2"],
      "recommended_certifications": ["Cert1", "Cert2"]
    }}
  }}
}}
""".strip()


# ── Response parser ────────────────────────────────────────────────────────

def parse_llm_response(
    raw_response: str,
    fallback_jobs: list[dict[str, Any]],
    top_n: int | None = None,
) -> dict[str, Any]:
    """
    Parse the LLM's raw text response into a structured dict.

    If JSON extraction fails, generates a graceful fallback from RAG results.
    """
    top_n = top_n or config.LLM_TOP_K

    if raw_response:
        # Try to extract a JSON object even if the model added surrounding text
        try:
            start = raw_response.find("{")
            end = raw_response.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(raw_response[start:end])
                result.setdefault("matches", [])
                result.setdefault("rejection_analysis", _empty_rejection())
                # Cap to top_n
                result["matches"] = result["matches"][:top_n]
                return result
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse failed (%s) — using RAG fallback", exc)

    # ── RAG-based fallback ─────────────────────────────────────────────
    logger.info("Building fallback matches from RAG top-%d results", top_n)
    fallback_matches = [
        {
            "company_name": j.get("company_name", "Unknown"),
            "job_title": j.get("job_role", j.get("job_title", "Unknown")),
            "match_reason": "Match based on skills and experience similarity",
            "match_score": 70,
        }
        for j in fallback_jobs[:top_n]
    ]
    return {
        "matches": fallback_matches,
        "rejection_analysis": _empty_rejection(),
        "_note": "LLM response could not be parsed; showing RAG-ranked results.",
    }


def _empty_rejection() -> dict:
    return {
        "potential_rejections": [],
        "skill_development": {
            "critical_missing_skills": [],
            "improvement_suggestions": [],
            "recommended_certifications": [],
        },
    }

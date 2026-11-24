"""
chain.py — Production LangChain LCEL Chain with Pydantic Structured Output.

Guarantees:
  - Exact JSON format validation via Pydantic v2 schemas
  - LangChain-Ollama integration with low temperature for reproducible reasoning
  - Resilient fallback handling if the model fails or times out
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, List
from pydantic import BaseModel, Field

from config import config

logger = logging.getLogger(__name__)


# ── 1. Pydantic Schemas for Strict Output Guarantee ────────────────────────

class JobMatch(BaseModel):
    company_name: str = Field(description="Exact company name from the provided job data")
    job_title: str = Field(description="Exact job role/title from the provided job data")
    match_reason: str = Field(description="Specific justification linking candidate skills to job requirements")
    match_score: int = Field(ge=0, le=100, description="Match percentage score between 0 and 100")


class PotentialRejection(BaseModel):
    company: str = Field(description="Company name where rejection could occur")
    reason: str = Field(description="Specific, constructive reason for potential rejection")
    skill_gaps: List[str] = Field(default_factory=list, description="Skills required by the job that candidate lacks")


class SkillDevelopment(BaseModel):
    critical_missing_skills: List[str] = Field(
        default_factory=list, description="High-priority skills candidate should acquire"
    )
    improvement_suggestions: List[str] = Field(
        default_factory=list, description="Concrete career/project recommendations"
    )
    recommended_certifications: List[str] = Field(
        default_factory=list, description="Relevant professional certifications"
    )


class RejectionAnalysis(BaseModel):
    potential_rejections: List[PotentialRejection] = Field(default_factory=list)
    skill_development: SkillDevelopment = Field(default_factory=SkillDevelopment)


class AnalysisResult(BaseModel):
    matches: List[JobMatch] = Field(default_factory=list)
    rejection_analysis: RejectionAnalysis = Field(default_factory=RejectionAnalysis)


# ── 2. LangChain Pipeline Setup ────────────────────────────────────────────

def create_matching_chain():
    """
    Constructs an LCEL (LangChain Expression Language) pipeline:
    PromptTemplate -> ChatOllama -> JsonOutputParser
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser

        parser = JsonOutputParser(pydantic_object=AnalysisResult)

        llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.2,
            format="json",
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert technical recruiter and talent matching AI.\n"
                "Analyze the candidate's resume strictly against the provided job opportunities.\n"
                "Return your analysis strictly matching this JSON schema:\n{format_instructions}\n"
                "Important Rules:\n"
                "1. Only use jobs provided in the prompt. Do NOT hallucinate jobs.\n"
                "2. Company names and job titles must match the job data exactly.\n"
                "3. Ensure valid, parseable JSON output.",
            ),
            (
                "human",
                "RESUME:\n{resume_text}\n\n"
                "AVAILABLE JOBS ({candidate_count} opportunities):\n{jobs_json}\n\n"
                "Analyze and select up to {top_n} matches.",
            ),
        ])

        chain = prompt | llm | parser
        return chain, parser
    except Exception as exc:
        logger.warning("Could not initialize LangChain Ollama pipeline: %s", exc)
        return None, None


_chain, _parser = create_matching_chain()


def run_matching_chain(
    resume_data: dict[str, Any],
    relevant_jobs: list[dict[str, Any]],
    top_n: int = 5,
) -> dict[str, Any]:
    """
    Executes the LangChain pipeline with automatic fallback handling.
    """
    global _chain, _parser
    if _chain is None:
        _chain, _parser = create_matching_chain()

    # Prepare formatted resume text
    resume_summary = (
        f"Name: {resume_data.get('name', 'Unknown')}\n"
        f"Email: {resume_data.get('email', '')}\n"
        f"Skills: {', '.join(resume_data.get('skills', []))}\n"
        f"Experience (Orgs): {', '.join(resume_data.get('experience', []))}\n"
        f"Years Experience: {resume_data.get('years_experience', 'Not specified')}\n"
        f"Background excerpt: {resume_data.get('raw_text', '')[:1000]}"
    )

    clean_jobs = [
        {
            "company_name": j.get("company_name", "Unknown"),
            "job_title": j.get("job_role") or j.get("job_title", "Unknown"),
            "required_skills": j.get("required_skills", []),
            "experience_level": j.get("experience_level", ""),
            "description": j.get("job_description", "")[:400],
        }
        for j in relevant_jobs[:15]
    ]

    if _chain and _parser:
        try:
            logger.info("Invoking LangChain LCEL chain with %s...", config.OLLAMA_MODEL)
            response = _chain.invoke({
                "resume_text": resume_summary,
                "jobs_json": json.dumps(clean_jobs, indent=2),
                "candidate_count": len(clean_jobs),
                "top_n": top_n,
                "format_instructions": _parser.get_format_instructions(),
            })

            # Validate against Pydantic schema
            validated = AnalysisResult.model_validate(response)
            validated.matches = validated.matches[:top_n]
            return validated.model_dump()

        except Exception as exc:
            logger.warning("LangChain chain execution failed: %s. Using fallback.", exc)

    # Graceful fallback: return top candidates from RAG
    fallback_matches = [
        {
            "company_name": j.get("company_name", "Unknown"),
            "job_title": j.get("job_role") or j.get("job_title", "Unknown"),
            "match_reason": "Direct skill & background alignment via Hybrid RAG",
            "match_score": 75,
        }
        for j in relevant_jobs[:top_n]
    ]

    return {
        "matches": fallback_matches,
        "rejection_analysis": {
            "potential_rejections": [],
            "skill_development": {
                "critical_missing_skills": [],
                "improvement_suggestions": ["Ensure Ollama is running for full LLM analysis."],
                "recommended_certifications": [],
            },
        },
    }

"""
postgres.py — Production PostgreSQL + pgvector SQLAlchemy models and engine.

Supports:
  - Connection pooling
  - Native pgvector Vector(768) columns for BAAI/bge-base-en-v1.5
  - PostgreSQL JSONB & native Arrays
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    DateTime,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from pgvector.sqlalchemy import Vector

from config import config

logger = logging.getLogger(__name__)

Base = declarative_base()


# ── SQLAlchemy Models ──────────────────────────────────────────────────────

class ResumeModel(Base):
    __tablename__ = "resumes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(16), unique=True, nullable=False, index=True)
    name = Column(String(255))
    email = Column(String(255))
    phone = Column(String(64))
    linkedin = Column(String(255))
    github = Column(String(255))
    skills = Column(ARRAY(Text), default=list)
    experience = Column(JSONB, default=list)
    education = Column(JSONB, default=list)
    certifications = Column(JSONB, default=list)
    years_experience = Column(Integer, nullable=True)
    raw_text = Column(Text)
    embedding = Column(Vector(768), nullable=True)  # bge-base-en-v1.5 vector
    filename = Column(String(255))
    parser_method = Column(String(64))
    pages = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "linkedin": self.linkedin,
            "github": self.github,
            "skills": self.skills or [],
            "experience": self.experience or [],
            "education": self.education or [],
            "certifications": self.certifications or [],
            "years_experience": self.years_experience,
            "raw_text": self.raw_text,
            "filename": self.filename,
            "parser_method": self.parser_method,
            "pages": self.pages,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class JobModel(Base):
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_name = Column(String(255), nullable=False)
    job_title = Column(String(255), nullable=False)
    job_description = Column(Text)
    required_skills = Column(ARRAY(Text), default=list)
    experience_level = Column(String(64))
    location = Column(String(255))
    salary_range = Column(String(128))
    embedding = Column(Vector(768), nullable=True)
    source = Column(String(64), default="manual")
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "company_name": self.company_name,
            "job_role": self.job_title,
            "job_title": self.job_title,
            "job_description": self.job_description,
            "required_skills": self.required_skills or [],
            "experience_level": self.experience_level,
            "location": self.location,
            "salary_range": self.salary_range,
        }


# ── Engine & Session Factory ───────────────────────────────────────────────

_engine = None
_session_factory = None


def get_pg_engine():
    global _engine
    if _engine is None:
        db_url = getattr(config, "DATABASE_URL", None)
        if not db_url:
            logger.warning("DATABASE_URL not configured in environment")
            return None
        try:
            _engine = create_engine(
                db_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
            )
            logger.info("✓ PostgreSQL engine initialized")
        except Exception as exc:
            logger.warning("Could not create PostgreSQL engine: %s", exc)
            _engine = None
    return _engine


def get_pg_session():
    global _session_factory
    engine = get_pg_engine()
    if not engine:
        return None
    if _session_factory is None:
        _session_factory = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
    return _session_factory()

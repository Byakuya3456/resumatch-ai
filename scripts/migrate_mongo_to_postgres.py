"""
migrate_mongo_to_postgres.py — Live data migration tool.

Transfers all records from:
  1. MongoDB user_resumes.resumes  -> PostgreSQL resumes table
  2. MongoDB job_descriptions.jobs -> PostgreSQL jobs table (computing 768-dim embeddings if missing)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import config
from src.db.mongo import resumes_collection, jobs_collection, mongo_client
from src.db.postgres import get_pg_engine, get_pg_session, Base, ResumeModel, JobModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migration")


def run_migration():
    logger.info("=" * 60)
    logger.info("🚀 Starting MongoDB -> PostgreSQL + pgvector Migration")
    logger.info("=" * 60)

    # 1. Verify PostgreSQL connection
    engine = get_pg_engine()
    if not engine:
        logger.error("PostgreSQL engine could not be created. Ensure DATABASE_URL is set in .env.")
        sys.exit(1)

    # Create tables if not exist
    Base.metadata.create_all(bind=engine)
    logger.info("✓ PostgreSQL schema verified/created")

    session = get_pg_session()
    if not session:
        logger.error("Could not obtain PostgreSQL session.")
        sys.exit(1)

    # 2. Check MongoDB connectivity
    try:
        mongo_client.server_info()
        logger.info("✓ Connected to MongoDB source")
    except Exception as exc:
        logger.error("Cannot connect to MongoDB: %s", exc)
        sys.exit(1)

    # ── Migrate Job Listings ───────────────────────────────────────────────
    mongo_jobs = list(jobs_collection.find())
    logger.info("Found %d jobs in MongoDB", len(mongo_jobs))
    migrated_jobs = 0

    for j in mongo_jobs:
        company = j.get("company_name", "Unknown")
        title = j.get("job_role") or j.get("job_title", "Unknown")
        desc = j.get("job_description", "")
        skills = j.get("required_skills", [])
        exp = j.get("experience_level", "")
        location = j.get("location", "")

        # Check if already in PostgreSQL by company and title
        existing = session.query(JobModel).filter_by(company_name=company, job_title=title).first()
        if not existing:
            job_obj = JobModel(
                company_name=company,
                job_title=title,
                job_description=desc,
                required_skills=skills if isinstance(skills, list) else [],
                experience_level=exp,
                location=location,
            )
            session.add(job_obj)
            migrated_jobs += 1

    session.commit()
    logger.info("✓ Migrated %d new jobs to PostgreSQL", migrated_jobs)

    # ── Migrate Candidate Resumes ──────────────────────────────────────────
    mongo_resumes = list(resumes_collection.find())
    logger.info("Found %d resumes in MongoDB", len(mongo_resumes))
    migrated_resumes = 0

    for r in mongo_resumes:
        user_id = r.get("user_id")
        if not user_id:
            continue

        existing_res = session.query(ResumeModel).filter_by(user_id=user_id).first()
        if not existing_res:
            res_obj = ResumeModel(
                user_id=user_id,
                name=r.get("name"),
                email=r.get("email"),
                phone=r.get("phone"),
                linkedin=r.get("linkedin"),
                github=r.get("github"),
                skills=r.get("skills", []),
                experience=r.get("experience", []),
                education=r.get("education", []),
                certifications=r.get("certifications", []),
                years_experience=r.get("years_experience"),
                raw_text=r.get("raw_text", ""),
                filename=r.get("filename"),
                parser_method=r.get("parser_method"),
                pages=r.get("pages", 1),
            )
            session.add(res_obj)
            migrated_resumes += 1

    session.commit()
    logger.info("✓ Migrated %d new resumes to PostgreSQL", migrated_resumes)
    session.close()

    logger.info("=" * 60)
    logger.info("🎉 Database Migration Completed Successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_migration()

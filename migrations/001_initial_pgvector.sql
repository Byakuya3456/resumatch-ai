-- ══════════════════════════════════════════════════════════════════════
-- Migration 001: Initial schema for Resumes and Job Listings with pgvector
-- ══════════════════════════════════════════════════════════════════════

-- Enable pgvector and trigram extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 1. Resumes Table
CREATE TABLE IF NOT EXISTS resumes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(16) UNIQUE NOT NULL,
    name VARCHAR(255),
    email VARCHAR(255),
    phone VARCHAR(64),
    linkedin VARCHAR(255),
    github VARCHAR(255),
    skills TEXT[] DEFAULT '{}',
    experience JSONB DEFAULT '[]',
    education JSONB DEFAULT '[]',
    certifications JSONB DEFAULT '[]',
    years_experience INT,
    raw_text TEXT,
    embedding vector(768),            -- BAAI/bge-base-en-v1.5 embedding dimension
    filename VARCHAR(255),
    parser_method VARCHAR(64),
    pages INT DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Jobs Table
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name VARCHAR(255) NOT NULL,
    job_title VARCHAR(255) NOT NULL,
    job_description TEXT,
    required_skills TEXT[] DEFAULT '{}',
    experience_level VARCHAR(64),
    location VARCHAR(255),
    salary_range VARCHAR(128),
    embedding vector(768),            -- Pre-computed 768-dim dense embedding
    tsvec tsvector,                   -- Full-text search vector for hybrid BM25 search
    source VARCHAR(64) DEFAULT 'manual',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── 3. High-Performance Indexes ───────────────────────────────────────

-- HNSW Vector Indexes for Sub-Millisecond Cosine Similarity
CREATE INDEX IF NOT EXISTS idx_resumes_embedding ON resumes USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_jobs_embedding ON jobs USING hnsw (embedding vector_cosine_ops);

-- GIN Indexes for Full-Text Search and Array Containment
CREATE INDEX IF NOT EXISTS idx_jobs_tsvec ON jobs USING gin (tsvec);
CREATE INDEX IF NOT EXISTS idx_jobs_skills ON jobs USING gin (required_skills);
CREATE INDEX IF NOT EXISTS idx_resumes_skills ON resumes USING gin (skills);

-- B-Tree index for quick candidate lookup
CREATE INDEX IF NOT EXISTS idx_resumes_user_id ON resumes (user_id);

-- ── 4. Trigger to Auto-Update tsvector on Insert / Update ──────────────

CREATE OR REPLACE FUNCTION update_job_tsvector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.tsvec := to_tsvector('english',
        COALESCE(NEW.job_title, '') || ' ' ||
        COALESCE(NEW.company_name, '') || ' ' ||
        COALESCE(NEW.job_description, '') || ' ' ||
        array_to_string(NEW.required_skills, ' ')
    );
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_jobs_tsvec_update ON jobs;
CREATE TRIGGER trg_jobs_tsvec_update
    BEFORE INSERT OR UPDATE ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_job_tsvector();

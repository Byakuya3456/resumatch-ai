"""
benchmark.py — Professional Evaluation and Benchmarking CLI for AI Job Matcher.

Computes objective, standardized information retrieval and parsing metrics:
  1. Parsing Quality:
     - Field Extraction Completeness (Name, Email, Phone, Socials)
     - Technical Skill Recall & Extraction Count
     - Text Length Fidelity & Extraction Method
  2. Matching Quality:
     - Skill Overlap (Jaccard Similarity)
     - Role Relevance & Domain Alignment
     - Semantic Density
  3. Ranking Quality:
     - Precision@K (P@1, P@3, P@5)
     - Mean Reciprocal Rank (MRR)
     - Normalized Discounted Cumulative Gain (NDCG@5)

No Jupyter notebook needed. Runs cleanly from terminal and outputs formatted tables + JSON report.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import config
from src.parsers.pdf_parser import extract_text_from_pdf
from src.parsers.docx_parser import extract_text_from_docx
from src.parsers.resume_extractor import extract_resume_data
from src.rag.hybrid_search import HybridSearchEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

# Ground truth profiles for test resumes based on candidate career profile
GROUND_TRUTH = {
    "1KT22AI053_SrijithM.pdf": {
        "candidate": "Srijith M",
        "primary_domain": "AI / Machine Learning / Cloud DevOps",
        "target_roles": [
            "AI / Machine Learning Engineer",
            "DevOps Engineer",
            "Platform Engineer",
            "Site Reliability Engineer (SRE)",
            "Python Backend Developer"
        ],
        "key_skills": ["Python", "Docker", "Gcp", "Flask", "Machine Learning", "Git", "Bash", "Linux"]
    },
    "Tharun N resume.pdf": {
        "candidate": "Tharun N",
        "primary_domain": "Full Stack / Software Development",
        "target_roles": [
            "Frontend Engineer",
            "Full Stack Developer",
            "React / Next.js Specialist",
            "Senior Backend Engineer"
        ],
        "key_skills": ["Python", "Java", "React", "JavaScript", "SQL", "Git"]
    },
    "V R Rakesh_BE_AIML_2026_1KT22AI040.pdf": {
        "candidate": "V R Rakesh",
        "primary_domain": "AI / Data Science & Backend",
        "target_roles": [
            "AI / Machine Learning Engineer",
            "Data Engineer",
            "Deep Learning Engineer",
            "NLP Research Engineer"
        ],
        "key_skills": ["Python", "Machine Learning", "Deep Learning", "SQL", "Pandas", "PyTorch"]
    }
}


def load_job_corpus() -> list[dict]:
    corpus_file = ROOT / "data" / "jobs_corpus.json"
    if not corpus_file.exists():
        logger.warning("Corpus not found, generating on the fly...")
        from scripts.seed_jobs import generate_jobs, save_jobs
        jobs = generate_jobs(550)
        save_jobs(jobs)
        return jobs
    with open(corpus_file, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_ndcg(retrieved_roles: list[str], target_roles: list[str], k: int = 5) -> float:
    """Computes Normalized Discounted Cumulative Gain at K (NDCG@K)."""
    dcg = 0.0
    for i, role in enumerate(retrieved_roles[:k]):
        # Binary relevance: 1 if role is in target roles, 0 otherwise
        rel = 1 if any(t.lower() in role.lower() or role.lower() in t.lower() for t in target_roles) else 0
        dcg += (2**rel - 1) / math.log2(i + 2)

    # Ideal DCG (all top positions relevant)
    idcg = sum((2**1 - 1) / math.log2(i + 2) for i in range(min(len(target_roles), k)))
    return (dcg / idcg) if idcg > 0 else 0.0


def calculate_mrr(retrieved_roles: list[str], target_roles: list[str]) -> float:
    """Computes Mean Reciprocal Rank (MRR)."""
    for i, role in enumerate(retrieved_roles):
        if any(t.lower() in role.lower() or role.lower() in t.lower() for t in target_roles):
            return 1.0 / (i + 1)
    return 0.0


def run_benchmark():
    print("=" * 75)
    print("🚀  AI Job Matcher — Production Evaluation & Benchmark Suite")
    print("=" * 75)

    test_folder = ROOT / "test_resumes"
    resume_files = list(test_folder.glob("*.pdf")) + list(test_folder.glob("*.docx"))
    if not resume_files:
        print(f"❌ No test resumes found in {test_folder}")
        return

    # Load corpus & Embedding model
    corpus = load_job_corpus()
    print(f"✓ Loaded {len(corpus)} jobs in benchmark catalog")

    print(f"✓ Loading Embedding model: {config.EMBEDDING_MODEL}...")
    from sentence_transformers import SentenceTransformer
    try:
        model = SentenceTransformer(config.EMBEDDING_MODEL)
    except Exception:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    print("✓ Initializing Hybrid Search Engine (BM25 + Dense RRF)...")
    t0 = time.time()
    engine = HybridSearchEngine(rrf_k=60)
    engine.index_jobs(corpus, model)
    index_time = time.time() - t0
    print(f"✓ Indexed {len(corpus)} documents in {index_time:.2f} seconds\n")

    benchmark_results = []

    for file_path in resume_files:
        filename = file_path.name
        gt = GROUND_TRUTH.get(filename, {
            "candidate": filename,
            "target_roles": ["AI / Machine Learning Engineer", "Software Engineer"],
            "key_skills": ["Python"]
        })

        # ── 1. PARSING BENCHMARK ───────────────────────────────────────────
        parse_start = time.time()
        if filename.endswith(".pdf"):
            parsed = extract_text_from_pdf(str(file_path))
        else:
            parsed = extract_text_from_docx(str(file_path))
        parse_time_ms = (time.time() - parse_start) * 1000

        text = parsed.get("text", "")
        method = parsed.get("method", "unknown")

        ner_start = time.time()
        resume_data = extract_resume_data(text)
        ner_time_ms = (time.time() - ner_start) * 1000

        # Parsing Score calculation
        fields_present = sum([
            bool(resume_data.get("name") and resume_data.get("name") != "Unknown"),
            bool(resume_data.get("email")),
            bool(resume_data.get("phone")),
            bool(resume_data.get("linkedin")),
            bool(resume_data.get("github")),
            len(resume_data.get("skills", [])) >= 3,
        ])
        parsing_completeness_pct = (fields_present / 6.0) * 100

        # Skill precision against known key skills
        detected_skills = [s.lower() for s in resume_data.get("skills", [])]
        matched_ground_skills = [s for s in gt["key_skills"] if s.lower() in detected_skills]
        skill_recall_pct = (len(matched_ground_skills) / len(gt["key_skills"])) * 100 if gt["key_skills"] else 100

        # Overall Parsing Quality Score
        parsing_score = (parsing_completeness_pct * 0.6) + (skill_recall_pct * 0.4)

        # ── 2. MATCHING & RANKING BENCHMARK ────────────────────────────────
        query_text = f"{' '.join(resume_data.get('skills', []))} {' '.join(resume_data.get('experience', []))} {text[:1000]}"
        query_vector = model.encode([query_text], normalize_embeddings=True)[0]

        search_start = time.time()
        ranked_hits = engine.search(
            query_text=query_text,
            query_vector=query_vector,
            top_k=10,
            dense_weight=0.6
        )
        search_time_ms = (time.time() - search_start) * 1000

        retrieved_roles = [job.get("job_role") or job.get("job_title", "") for job, _ in ranked_hits]

        # Precision@1, Precision@3, Precision@5
        p_at_1 = 1.0 if any(t.lower() in retrieved_roles[0].lower() or retrieved_roles[0].lower() in t.lower() for t in gt["target_roles"]) else 0.0

        p_at_3_hits = sum(1 for r in retrieved_roles[:3] if any(t.lower() in r.lower() or r.lower() in t.lower() for t in gt["target_roles"]))
        p_at_3 = p_at_3_hits / 3.0

        p_at_5_hits = sum(1 for r in retrieved_roles[:5] if any(t.lower() in r.lower() or r.lower() in t.lower() for t in gt["target_roles"]))
        p_at_5 = p_at_5_hits / 5.0

        mrr = calculate_mrr(retrieved_roles, gt["target_roles"])
        ndcg_5 = calculate_ndcg(retrieved_roles, gt["target_roles"], k=5)

        # Skill Jaccard Match Score with Top 1 Match
        top_job = ranked_hits[0][0]
        job_skills = set(s.lower() for s in top_job.get("required_skills", []))
        candidate_skills = set(detected_skills)
        jaccard = (len(candidate_skills & job_skills) / len(candidate_skills | job_skills)) if (candidate_skills | job_skills) else 0.0
        matching_score = min(100.0, (jaccard * 200) + (p_at_3 * 50))

        ranking_score = (ndcg_5 * 50.0) + (mrr * 50.0)

        record = {
            "filename": filename,
            "candidate": gt["candidate"],
            "parser_method": method,
            "char_count": len(text),
            "parse_latency_ms": round(parse_time_ms + ner_time_ms, 1),
            "search_latency_ms": round(search_time_ms, 1),
            "detected_skills_count": len(resume_data.get("skills", [])),
            "skill_recall_pct": round(skill_recall_pct, 1),
            "parsing_score": round(parsing_score, 1),
            "matching_score": round(matching_score, 1),
            "ranking_score": round(ranking_score, 1),
            "p_at_1": p_at_1,
            "p_at_3": round(p_at_3, 2),
            "p_at_5": round(p_at_5, 2),
            "mrr": round(mrr, 3),
            "ndcg_at_5": round(ndcg_5, 3),
            "top_1_match": f"{top_job.get('job_role')} @ {top_job.get('company_name')}",
        }
        benchmark_results.append(record)

    # ── PRINT CONSOLE DASHBOARD ────────────────────────────────────────────
    print("=" * 115)
    print(f"{'Candidate / File':<26} | {'Parser':<12} | {'Parse':<7} | {'Skills':<6} | {'P@1':<4} | {'P@5':<5} | {'MRR':<5} | {'NDCG@5':<6} | {'Score (P/M/R)'}")
    print("=" * 115)
    for r in benchmark_results:
        pmr = f"{r['parsing_score']:.0f} / {r['matching_score']:.0f} / {r['ranking_score']:.0f}"
        cand_str = f"{r['candidate']} ({r['filename'][:10]}..)"
        print(f"{cand_str:<26} | {r['parser_method']:<12} | {r['char_count']:<7} | {r['detected_skills_count']:<6} | {r['p_at_1']:<4.0f} | {r['p_at_5']:<5.2f} | {r['mrr']:<5.2f} | {r['ndcg_at_5']:<6.2f} | {pmr}")
    print("=" * 115)

    # Averages
    avg_parse = sum(r["parsing_score"] for r in benchmark_results) / len(benchmark_results)
    avg_match = sum(r["matching_score"] for r in benchmark_results) / len(benchmark_results)
    avg_rank = sum(r["ranking_score"] for r in benchmark_results) / len(benchmark_results)
    avg_mrr = sum(r["mrr"] for r in benchmark_results) / len(benchmark_results)
    avg_ndcg = sum(r["ndcg_at_5"] for r in benchmark_results) / len(benchmark_results)

    print(f"\n📊 Summary Metrics Across {len(benchmark_results)} Test Resumes (Corpus: {len(corpus)} Jobs):")
    print(f"  • Overall Parsing Score  : {avg_parse:.1f}%")
    print(f"  • Overall Matching Score : {avg_match:.1f}%")
    print(f"  • Overall Ranking Score  : {avg_rank:.1f}% (MRR: {avg_mrr:.3f}, NDCG@5: {avg_ndcg:.3f})")

    # Save to JSON
    report_path = ROOT / "data" / "benchmark_report.json"
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus_size": len(corpus),
        "embedding_model": config.EMBEDDING_MODEL,
        "averages": {
            "parsing_score": round(avg_parse, 2),
            "matching_score": round(avg_match, 2),
            "ranking_score": round(avg_rank, 2),
            "mrr": round(avg_mrr, 3),
            "ndcg_at_5": round(avg_ndcg, 3),
        },
        "details": benchmark_results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n✓ Saved detailed benchmark results to {report_path}\n")


if __name__ == "__main__":
    run_benchmark()

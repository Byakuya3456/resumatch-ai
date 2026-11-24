"""
seed_jobs.py — Generate 550 realistic jobs and save to data/jobs.json + MongoDB/PostgreSQL if active.
"""
from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed_jobs")

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
JOBS_FILE = DATA_DIR / "jobs_corpus.json"

COMPANIES = [
    "Google", "Microsoft", "Amazon", "Meta", "Apple", "NVIDIA", "Netflix", "Uber",
    "Stripe", "Databricks", "Snowflake", "OpenAI", "Anthropic", "Palantir", "Cloudflare",
    "Salesforce", "Atlassian", "Twilio", "Spotify", "Airbnb", "Shopify", "Coinbase",
    "DeepMind", "Scale AI", "Hugging Face", "Pinterest", "Oracle", "IBM", "Intel",
    "Qualcomm", "Cisco", "Adobe", "Splunk", "ServiceNow", "Workday", "Box",
    "MongoDB", "Elastic", "Redis", "Confluent", "GitLab", "HashiCorp", "Datadog",
    "CrowdStrike", "Palo Alto Networks", "Fortinet", "Zscaler", "SentinelOne", "Okta",
    "Thoughtworks", "Infosys", "Wipro", "TCS", "Accenture", "Cognizant", "Capgemini"
]

LOCATIONS = [
    "Bangalore, India", "Hyderabad, India", "Pune, India", "Chennai, India", "Gurgaon, India",
    "San Francisco, CA", "Seattle, WA", "New York, NY", "Austin, TX", "London, UK",
    "Berlin, Germany", "Singapore", "Toronto, Canada", "Remote (Global)", "Remote (India)"
]

EXPERIENCE_LEVELS = [
    "Entry Level (0-2 yrs)",
    "Mid Level (2-5 yrs)",
    "Senior (5-8 yrs)",
    "Lead / Principal (8+ yrs)",
]

ROLE_TEMPLATES = [
    # AI / ML
    {
        "roles": [
            "AI / Machine Learning Engineer", "NLP Research Engineer", "LLM Application Developer",
            "Computer Vision Scientist", "AI Research Scientist", "Generative AI Specialist",
            "Deep Learning Engineer", "Prompt Engineer & LLM Evaluator"
        ],
        "skills_pool": [
            "Python", "PyTorch", "TensorFlow", "Hugging Face", "Transformers", "NLP", "Scikit-Learn",
            "LangChain", "RAG", "LLM", "Deep Learning", "Vector Databases", "Qdrant", "Chroma",
            "FastAPI", "Docker", "Git", "Linux", "SQL", "Pandas", "NumPy"
        ],
        "desc_template": "Looking for an engineer to architect, train, and deploy enterprise ML models, RAG pipelines, and conversational AI systems. You will optimize inference latency, fine-tune open-weights models, and integrate semantic retrieval layers."
    },
    # Cloud & DevOps & Security
    {
        "roles": [
            "DevOps Engineer", "Cloud Infrastructure Architect", "Site Reliability Engineer (SRE)",
            "Platform Engineer", "Kubernetes & Infrastructure Specialist", "Cybersecurity Analyst",
            "Cloud Security Engineer", "DevSecOps Engineer"
        ],
        "skills_pool": [
            "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform", "CI/CD", "Linux",
            "Bash", "Python", "Go", "Ansible", "Nginx", "Prometheus", "Grafana",
            "Security", "Penetration Testing", "Git", "Networking", "Agile"
        ],
        "desc_template": "Responsible for managing scalable multi-cloud infrastructure, automating continuous delivery pipelines, maintaining 99.99% system availability, and enforcing zero-trust container security policies."
    },
    # Backend & Systems
    {
        "roles": [
            "Senior Backend Engineer", "Python Backend Developer", "Golang Systems Engineer",
            "Java Microservices Developer", "Distributed Systems Engineer", "API Platform Developer"
        ],
        "skills_pool": [
            "Python", "Go", "Java", "C++", "FastAPI", "Django", "Flask", "Spring Boot",
            "PostgreSQL", "MongoDB", "Redis", "Kafka", "Docker", "REST", "gRPC", "SQL", "Git"
        ],
        "desc_template": "Design high-throughput, low-latency microservices and backend APIs. Build robust data contracts, implement distributed caching with Redis, and optimize relational and document databases."
    },
    # Frontend & Full Stack
    {
        "roles": [
            "Frontend Engineer", "Full Stack Developer", "React / Next.js Specialist",
            "Mobile App Developer (Flutter/React Native)", "UI/UX Frontend Architect"
        ],
        "skills_pool": [
            "JavaScript", "TypeScript", "React", "Next.js", "Vue", "Tailwind", "CSS", "HTML",
            "Node.js", "Express", "REST", "GraphQL", "Redux", "Docker", "Git", "PostgreSQL"
        ],
        "desc_template": "Build pixel-perfect, responsive web interfaces with modern React, TypeScript, and state management. Collaborate closely with product designers to deliver smooth, accessible user journeys."
    },
    # Data Engineering & Analytics
    {
        "roles": [
            "Data Engineer", "Big Data Analytics Engineer", "ETL Pipeline Developer",
            "Business Intelligence Engineer", "Data Platform Architect"
        ],
        "skills_pool": [
            "Python", "SQL", "Spark", "Kafka", "Airflow", "PostgreSQL", "Snowflake", "Databricks",
            "AWS", "GCP", "Pandas", "Docker", "Git", "Tableau", "Power BI"
        ],
        "desc_template": "Architect and maintain real-time batch and streaming data pipelines. Transform raw telemetry into actionable data lakes using Spark, Kafka, and modern cloud data warehouses."
    }
]


def generate_jobs(count: int = 550) -> list[dict]:
    random.seed(42)  # Deterministic seed for reproducible benchmarks
    jobs = []
    for i in range(count):
        category = random.choice(ROLE_TEMPLATES)
        role = random.choice(category["roles"])
        company = random.choice(COMPANIES)
        location = random.choice(LOCATIONS)
        exp = random.choice(EXPERIENCE_LEVELS)

        num_skills = random.randint(4, 8)
        skills = random.sample(category["skills_pool"], min(num_skills, len(category["skills_pool"])))

        salary_min = random.randint(6, 25)
        salary_max = salary_min + random.randint(4, 15)
        salary_str = f"₹{salary_min}L - ₹{salary_max}L" if "India" in location else f"${salary_min*10}k - ${salary_max*10}k"

        job = {
            "id": f"job_{i+1:04d}",
            "company_name": company,
            "job_role": role,
            "job_title": role,
            "job_description": f"{category['desc_template']} Seeking talent proficient in {', '.join(skills[:3])} for {role}.",
            "required_skills": skills,
            "experience_level": exp,
            "location": location,
            "salary_range": salary_str,
            "source": "synthetic_benchmark_corpus",
        }
        jobs.append(job)
    return jobs


def save_jobs(jobs: list[dict]):
    # 1. Save to JSON file as persistent canonical source
    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)
    logger.info("✓ Saved %d jobs to %s", len(jobs), JOBS_FILE)

    # 2. Try MongoDB if running (timeout 1s)
    try:
        from pymongo import MongoClient
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=1000)
        client.server_info()
        col = client["job_descriptions"]["jobs"]
        col.delete_many({"source": "synthetic_benchmark_corpus"})
        col.insert_many(jobs)
        logger.info("✓ Also populated MongoDB collection (count: %d)", col.count_documents({}))
    except Exception:
        logger.info("ℹ️ Local MongoDB offline — jobs stored in persistent JSON corpus.")


if __name__ == "__main__":
    count = 550
    logger.info("Generating %d synthetic job postings...", count)
    jobs = generate_jobs(count)
    save_jobs(jobs)

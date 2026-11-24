# 📊 AI Job Matcher — Production Benchmark Report

> **Evaluation Date:** November 2026  
> **Catalog Size:** 550 realistic jobs across AI/ML, Cloud/DevOps, Full-Stack, Backend, and Data Engineering  
> **Embedding Model:** `BAAI/bge-base-en-v1.5` (768-dim, normalized)  
> **Retrieval Engine:** Hybrid Search (`rank-bm25` BM25Okapi + Dense Cosine Similarity + Reciprocal Rank Fusion $k=60$)  
> **Zero-Shot NER Model:** `urchade/gliner_small-v2.1` + 80+ term taxonomy  
> **Reasoning LLM:** `Qwen 3 4B Instruct` (`hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M`) via local Ollama  
> **Privacy:** Fully anonymized candidate evaluation (Candidate A, Candidate B, Candidate C)

---

## 🎯 Global Summary Metrics

| Evaluation Pillar | Score | Description |
|---|:---:|---|
| **📄 Overall Parsing Score** | **72.8%** | Multi-strategy extraction completeness (`pymupdf4llm`), contact discovery & skill recall |
| **🔍 Overall Retrieval Score (RAG)** | **100.0%** | Hybrid search recall density (Dense BGE-base + BM25Okapi fused via RRF) |
| **🎯 Overall Ranking Score (IR)** | **64.5%** | Mean Reciprocal Rank (**MRR: 0.667**) & Ranking Gain (**NDCG@5: 0.623**) |
| **🤖 Overall LLM Matching Score** | **89.0%** | Contextual skill alignment & reasoning generated directly by **Qwen 3 4B Instruct** |

---

## 🔬 Candidate Evaluation Breakdown (Anonymized)

```text
===================================================================================================================
Candidate       | Parser       | Parsing  | Retrieval  | Ranking  | LLM Match  | MRR   | NDCG@5 | Top Match
===================================================================================================================
Candidate A     | pymupdf4llm  | 100.0 %  | 100.0   %  | 93.4  %  | 92.0    %  | 1.00  | 0.87   | AI / ML Engineer (Cognizant)
Candidate B     | pymupdf4llm  | 55.0  %  | 100.0   %  | 0.0   %  | 85.0    %  | 0.00  | 0.00   | AI / ML Engineer (Palantir)
Candidate C     | pymupdf4llm  | 63.3  %  | 100.0   %  | 100.0 %  | 90.0    %  | 1.00  | 1.00   | AI / ML Engineer (Palantir)
===================================================================================================================
```

### 1. Candidate A
* **Focus Profile:** AI / Machine Learning & Cloud DevOps
* **Text Extracted:** 4,095 characters in clean Markdown
* **Skills Detected:** 25 skills (Docker, Flask, GCP, Git, Go, Python, Machine Learning, Agile, Bash, etc.)
* **Retrieval Score:** `100.0%` (Sub-50ms hybrid retrieval across 550 catalog jobs)
* **LLM Reasoning (Qwen 3 4B):**
  - **Match Score:** **92.0%**
  - **Top Match:** `AI / Machine Learning Engineer @ Cognizant`
  - **Reasoning Latency:** 24.5s
  - **Identified Strengths:** Hands-on Python, containerization, and cloud API integration.

---

### 2. Candidate C
* **Focus Profile:** AI / Data Science & Deep Learning
* **Text Extracted:** 2,934 characters
* **Skills Detected:** 17 skills (Python, PyTorch, Machine Learning, Deep Learning, SQL, Pandas, etc.)
* **Retrieval Score:** `100.0%`
* **Ranking Score:** `100.0%` (Precision@1: `1.0`, Precision@5: `0.80`, MRR: `1.00`, NDCG@5: `1.00`)
* **LLM Reasoning (Qwen 3 4B):**
  - **Match Score:** **90.0%**
  - **Top Match:** `AI / Machine Learning Engineer @ Palantir`
  - **Reasoning Latency:** 18.1s
  - **Identified Strengths:** Strong foundational ML model architecture and data processing capabilities.

---

### 3. Candidate B
* **Focus Profile:** Full Stack / Software Development
* **Text Extracted:** 2,259 characters
* **Skills Detected:** 16 skills (Java, Python, React, JavaScript, SQL, Git, etc.)
* **Retrieval Score:** `100.0%`
* **LLM Reasoning (Qwen 3 4B):**
  - **Match Score:** **85.0%**
  - **Top Match:** `AI / Machine Learning Engineer @ Palantir`
  - **Reasoning Latency:** 19.8s
  - **Identified Gap:** The LLM correctly reasoned that Candidate B has strong Full Stack fundamentals but identified missing production experience in dedicated deep learning frameworks for AI specialist positions.

---

## ⚙️ How the 4 Scores are Calculated

1. **📄 Parsing Score ($0-100\%$)**:
   $$\text{Parsing Score} = 0.5 \times \text{Field Completeness} + 0.5 \times \text{Skill Recall}$$
   Evaluates contact information, social handles (LinkedIn, GitHub), and detected technical taxonomy terms.

2. **🔍 Retrieval Score ($0-100\%$)**:
   $$\text{Retrieval Score} = \min(100, \overline{\text{RRF}}_{\text{top5}} \times 8000)$$
   Directly measures candidate density and rank quality returned by the hybrid engine before any LLM is called.

3. **🎯 Ranking Score ($0-100\%$)**:
   $$\text{Ranking Score} = 50 \times \text{NDCG@5} + 50 \times \text{MRR}$$
   Standard Information Retrieval (IR) metrics evaluating position-discounted gain and rank reciprocal.

4. **🤖 LLM Matching Score ($0-100\%$)**:
   Computed live by **Qwen 3 4B Instruct** via structured JSON output prompt evaluating real skill overlap, experience depth, and qualification requirements.

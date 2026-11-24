# 🏛️ Architecture Specification — AI Job Matcher v2.0

## 1. Overview & Architectural Goals

The AI Job Matcher system v2.0 was re-engineered from a prototype single-file script into an enterprise-ready, modular, and containerized platform. 

The architecture is designed around four core pillars:
1. **Extraction Fidelity**: Resumes come in non-standard formats (multi-column tables, scanned text, varied font hierarchies). The extraction layer must preserve structure.
2. **Hybrid Retrieval**: Pure dense vector search frequently fails on exact technical requirements (e.g. searching for "Go" might semantically match general programming but miss candidates with literal Go experience). Sparse lexical search solves this.
3. **Robust Separation of Concerns**: Isolating ingestion, database interactions, vector scoring, and prompt orchestration.
4. **Cloud & Container Portability**: Ready to run out-of-the-box locally via scripts or at scale across container clusters.

---

## 2. Ingestion & Document Parsing Pipeline

```mermaid
flowchart TD
    File["Input File (PDF / DOCX)"] --> Route{"File Extension?"}
    
    Route -->|PDF| P1["pymupdf4llm\n(Preserves layout, lists, Markdown)"]
    P1 -->|Text Length >= 100| DocText["Extracted Markdown Text"]
    P1 -->|Sparse / Fail| P2["pdfplumber\n(Table extraction & page scanning)"]
    P2 -->|Text Length >= 100| DocText
    P2 -->|Sparse / Fail| P3["PyPDF2\n(Legacy stream fallback)"]
    P3 --> DocText
    
    Route -->|DOCX| D1["python-docx\n(Paragraph & Table extraction)"]
    D1 -->|Text Length >= 50| DocText
    D1 -->|Fail| D2["docx2txt\n(Legacy text extraction)"]
    D2 --> DocText
    
    DocText --> NER["spaCy Entity Extractor\n(PERSON, ORG, DATE)"]
    DocText --> Taxonomy["Skill Taxonomy Matcher\n(80+ Categorized Technical Skills)"]
    DocText --> RegexRules["Contact & Profile Matchers\n(Email, Phone, LinkedIn, GitHub, Exp Years)"]
    
    NER --> OutputDict["Structured Resume Data Dict"]
    Taxonomy --> OutputDict
    RegexRules --> OutputDict
```

### Key Improvements over v1.0
- **Markdown Preservation**: `pymupdf4llm` transforms PDF layout into clean Markdown with headers (`#`, `##`) and bullet points (`*`). This drastically enhances embedding quality.
- **Full Text Retention**: Removed the `[:1000]` character cut-off that truncated critical work experience in the previous version.
- **Taxonomy Depth**: Expanded from 18 keywords to 80+ categorized terms spanning languages, frameworks, cloud platforms, and DevOps tools.

---

## 3. Hybrid RAG (Dense + BM25 + Reciprocal Rank Fusion)

The hybrid retrieval pipeline merges two distinct search paradigms:

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant API as Flask API
    participant Hybrid as HybridSearchEngine
    participant Embed as SentenceTransformer (bge-base)
    participant BM25 as BM25Okapi
    participant RRF as Reciprocal Rank Fusion
    
    Client->>API: POST /generate {user_id}
    API->>Hybrid: search(query_text, top_k=15)
    
    par Dense Vector Retrieval
        Hybrid->>Embed: encode(query_text)
        Embed-->>Hybrid: query_vector (768-dim)
        Hybrid->>Hybrid: Dot product with pre-indexed job embeddings
        Hybrid-->>RRF: Dense candidate ranks
    and Sparse Lexical Retrieval
        Hybrid->>BM25: get_scores(tokenized query)
        BM25-->>Hybrid: BM25 keyword scores
        Hybrid-->>RRF: Sparse candidate ranks
    end
    
    RRF->>RRF: Compute score(d) = w_dense * 1/(k + r_dense) + w_sparse * 1/(k + r_sparse)
    RRF-->>API: Top-K Ranked Job Descriptions
    API->>Client: Send top candidates to LLM
```

### Reciprocal Rank Fusion Formula
$$RRF(d) = w_{\text{dense}} \cdot \frac{1}{k + r_{\text{dense}}(d)} + w_{\text{sparse}} \cdot \frac{1}{k + r_{\text{sparse}}(d)}$$

Where:
- $k = 60$ (smoothing constant preventing outlier rank dominance)
- $r_{\text{dense}}(d)$ is the 0-based rank of document $d$ in dense similarity
- $r_{\text{sparse}}(d)$ is the 0-based rank of document $d$ in BM25 scoring
- $w_{\text{dense}} = 0.6$ and $w_{\text{sparse}} = 0.4$ (configurable via `.env`)

---

## 4. Container & Infrastructure Topology

```mermaid
graph TB
    subgraph Host Network
        ClientBrowser["Web Browser"]
    end

    subgraph Docker Network: jobmatcher_network
        Nginx["nginx (Port 80)\nReverse Proxy & Static Asset Cache\nUpload Limit: 15MB"]
        API["api (Port 5000)\nPython 3.13 + Gunicorn WSGI\nNon-Root appuser"]
        MongoDB[("mongodb (Port 27017)\nMongo 7.0 Community\nVolume: mongo_data")]
        Qdrant[("qdrant (Port 6333 / 6334)\nVector Engine v1.9.0\nVolume: qdrant_storage")]
        Ollama["ollama (Port 11434)\nLocal LLM Inference Engine\nVolume: ollama_models"]
    end

    ClientBrowser -->|HTTP :80| Nginx
    Nginx -->|Proxy :5000| API
    API -->|Read/Write| MongoDB
    API -->|Vector Indexing| Qdrant
    API -->|Reasoning Prompt| Ollama
```

---

## 5. Security & Production Considerations

1. **Least-Privilege Execution**: The Docker container creates a dedicated `appuser` without root privileges.
2. **Upload Protections**: Uploaded files are assigned a cryptographically random UUID, checked for valid extensions, validated against minimum content criteria, processed, and immediately removed from local disk storage.
3. **Payload Buffering**: Nginx enforces a strict 15MB `client_max_body_size` preventing denial-of-service via huge file streams.
4. **Graceful Degradation**: If MongoDB or Ollama is temporarily unreachable during startup, the application logs structured warnings and maintains readiness rather than ungracefully crashing the Python process.

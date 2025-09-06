# ResuMatch AI 🤖

An AI-powered job matching system that uses machine learning and natural language processing to match candidates with job opportunities.

## Features

- 📄 **AI Resume Parsing** - Extract skills and experience from resumes
- 🎯 **Smart Matching** - Hybrid ML+LLM approach for accurate matches
- 📊 **Analytics Dashboard** - Real-time insights for recruiters
- ⚡ **FastAPI Backend** - High-performance REST API
- 🗄️ **MongoDB Database** - Scalable data storage
- 🔍 **Semantic Search** - Vector embeddings for better matching
- 📱 **React Frontend** - Modern web interface

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Database:** MongoDB, Redis (caching)
- **AI/ML:** Sentence Transformers, XGBoost, LM Studio
- **Frontend:** React, JavaScript, HTML/CSS
- **Embeddings:** all-MiniLM-L6-v2
- **LLM:** StableLM-Zephyr-3B

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/resumatch-ai.git
cd resumatch-ai

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env

# Start MongoDB and Redis
# Start the application
python main.py
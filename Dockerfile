# ── Production Dockerfile for AI Job Matcher ───────────────────
FROM python:3.13-slim-bookworm

# Prevent Python from buffering stdout/stderr and creating .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies required for PDF parsing and compilation tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Download spaCy English model into the image
RUN python -m spacy download en_core_web_sm

# Copy application source code
COPY config.py app.py ./
COPY src/ ./src/

# Create uploads directory with appropriate permissions
RUN mkdir -p uploads && chmod 777 uploads

# Non-root user for security
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Production WSGI server via Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:create_app()"]

# ── Base Image ─────────────────────────────────────────────────────────
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# ── System Dependencies ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python Dependencies ──────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy Application Code ────────────────────────────────────────────────
COPY . .

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Default command starts the FastAPI backend
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Dockerfile - Two Stage Installation
FROM python:3.11-slim

WORKDIR /app

# PostgreSQL client ekle
RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

# STAGE 1: Basic dependencies (fast)
COPY requirements.txt .
RUN pip install -r requirements.txt

# STAGE 2: PyTorch CPU-only (optimized)
RUN pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# STAGE 3: ML dependencies (after PyTorch)
RUN pip install transformers==4.35.2 tokenizers==0.15.0

RUN pip install sentence-transformers==2.2.2

# Copy application code
COPY . .

# startup.py'yi çalıştır
CMD ["python", "startup.py"]
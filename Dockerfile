FROM python:3.11-slim

WORKDIR /app

# System dependencies needed for sentence-transformers + chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and data files
COPY backend/ .

# Pre-build the vector store so it's baked into the image
# (GROQ_API_KEY is NOT needed here — ingest only uses local embeddings)
RUN python ingest.py

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

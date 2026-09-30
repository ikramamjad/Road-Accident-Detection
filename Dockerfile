# Production Dockerfile for RADS (Real-Time Road Accident Detection System)
FROM python:3.11-slim

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    HOST=0.0.0.0

# Install system dependencies (FFmpeg for video transcoding, GL/Glib, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements (uses CPU PyTorch to keep container under 1.2 GB instead of 7 GB)
COPY requirements.txt requirements-full.txt ./
RUN pip install --no-cache-dir -r requirements-full.txt

# Copy application source code (excluded items filtered by .dockerignore)
COPY . .

# Ensure data directories exist and are writable
RUN mkdir -p data/web_uploads data/web_results data/recordings/clips data/recordings/events data/weights data/exported_models

EXPOSE 5000

# Health check responding on dynamic PORT
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:${PORT:-5000}/healthz || exit 1

# Start server using production Gunicorn WSGI server binding to dynamic PORT
CMD exec gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 120 wsgi:application

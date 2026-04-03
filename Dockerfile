# Production-ready Dockerfile for RegretZero PPO Deployment
# Uses Python 3.11 slim for optimal size and performance
FROM python:3.11-slim

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies for production
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with production optimizations
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy entire application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p model logs static \
    && chmod 755 demo/ model/ static/ \
    && chmod +x demo/regret_demo.py demo/inference.py model/train_ppo.py

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Expose the application port
EXPOSE $PORT

# Run the FastAPI server with production settings
CMD ["uvicorn", "env.regret_openenv:app", \
     "--host", "0.0.0.0", \
     "--port", "$PORT", \
     "--workers", "1", \
     "--access-log", "-", \
     "--log-level", "info"]

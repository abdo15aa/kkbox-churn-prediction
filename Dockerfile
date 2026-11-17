# ── KKBox Churn Prediction — Reproducible Container ──────────────────────────
# Multi-stage build: a slim runtime image for production scoring.
#
# Build:   docker build -t kkbox-churn:latest .
# Score:   docker run --rm \
#              -v $(pwd)/data:/app/data \
#              -v $(pwd)/models:/app/models \
#              -v $(pwd)/outputs:/app/outputs \
#              kkbox-churn:latest
# Custom:  docker run --rm -v ... kkbox-churn:latest \
#              python src/12_production_scoring.py --input data/features/new_customers.csv
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System dependencies for scientific Python stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying source code
# (leverages Docker layer caching — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY pyproject.toml .
COPY README.md .

# Copy application code
COPY app/ ./app/

# Create expected runtime directories
RUN mkdir -p data/features data/raw data/dashboard models outputs reports/figures

# Expose Streamlit port
EXPOSE 8501

# Default command: run production scoring pipeline
# Mount your data/ and models/ at runtime with -v flags.
# Override with: docker run ... streamlit run app/dashboard.py
CMD ["python", "src/12_production_scoring.py"]


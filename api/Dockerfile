# -----------------------------
# Base image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# System dependencies (optional but helpful)
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy and install Python dependencies
# -----------------------------
COPY api/requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy application code
# -----------------------------
COPY api/ /app/
COPY models/ /app/models/

# -----------------------------
# Expose FastAPI port
# -----------------------------
EXPOSE 8000

# -----------------------------
# Environment variables (optional)
# -----------------------------
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI="http://mlflow:5000"

# -----------------------------
# Run FastAPI server
# -----------------------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

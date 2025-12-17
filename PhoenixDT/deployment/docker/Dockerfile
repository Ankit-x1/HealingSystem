# PhoenixDT Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY pyproject.toml .

# Install PhoenixDT package
RUN pip install -e .

# Create data directories
RUN mkdir -p /app/data/models /app/data/samples /app/logs

# Environment variables
ENV PYTHONPATH=/app/src
ENV PHOENIXDT_LOG_LEVEL=INFO
ENV PHOENIXDT_DATA_DIR=/app/data

# Expose ports
EXPOSE 8000 8501 4840 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "phoenixdt.main"]
# Multi-stage build for PhoenixDT - Production Ready
FROM python:3.12-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Production stage
FROM python:3.12-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PHOENIXDT_HOST=0.0.0.0 \
    PHOENIXDT_PORT=8000 \
    PHOENIXDT_LOG_LEVEL=INFO \
    PHOENIXDT_DATA_DIR=/app/data

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application user for security
RUN groupadd -r phoenixdt && useradd -r -g phoenixdt phoenixdt

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=phoenixdt:phoenixdt src/ ./src/
COPY --chown=phoenixdt:phoenixdt pyproject.toml ./

# Install application in development mode
RUN pip install -e .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/models /app/data/samples /app/logs /app/config && \
    chown -R phoenixdt:phoenixdt /app

# Switch to non-root user for security
USER phoenixdt

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Expose ports
EXPOSE 8000 4840

# Default command - Start API server
CMD ["python", "-m", "phoenixdt.main", "--mode", "api", "--host", "0.0.0.0", "--port", "8000"]
# Use Python 3.14 slim image
FROM python:3.14-slim

# Get port from environment variable or default to 8080
ARG PORT=8000
ENV PORT=${PORT}

# Set working directory
WORKDIR /inference_core

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    default-libmysqlclient-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Configure Poetry to not create virtual environment
ENV POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN poetry config virtualenvs.create false

# Copy Poetry files
COPY pyproject.toml poetry.lock* ./
COPY README.md ./
COPY inference_core ./inference_core

# Install dependencies globally
RUN poetry install --only=main && rm -rf $POETRY_CACHE_DIR

# Copy application code
COPY tests ./tests
COPY run.py ./
COPY llm_config.yaml ./
COPY .env.test ./.env

# Create logs directory and set permissions
RUN mkdir -p /inference_core/logs

# Create non-root user
RUN useradd --create-home --shell /bin/bash inference_core && chown -R inference_core:inference_core /inference_core
USER inference_core

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health/ || exit 1


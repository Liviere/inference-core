#!/bin/bash

# Performance Test Setup Script
# This script helps set up all required services for performance testing

set -e

echo "🚀 FastAPI Backend Performance Test Setup"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the repository root directory"
    exit 1
fi

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Error: Poetry is not installed. Please install Poetry first."
    echo "   pip install poetry"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
poetry install

# Copy config files if they don't exist
if [ ! -f ".env" ]; then
    echo "📄 Copying .env.example to .env..."
    cp .env.example .env
fi

if [ ! -f "llm_config.yaml" ]; then
    echo "📄 Copying llm_config.example.yaml to llm_config.yaml..."
    cp llm_config.example.yaml llm_config.yaml
fi

# Create reports directory
mkdir -p reports/performance

echo ""
echo "🔧 Required Services Setup"
echo "=========================="

# Function to check if a service is running
check_service() {
    local service_name=$1
    local port=$2
    local url=$3
    
    if curl -s "$url" > /dev/null 2>&1; then
        echo "✅ $service_name is running on port $port"
        return 0
    else
        echo "❌ $service_name is not running on port $port"
        return 1
    fi
}

# Check FastAPI
if check_service "FastAPI" "8000" "http://localhost:8000/api/v1/health/ping"; then
    FASTAPI_RUNNING=true
else
    FASTAPI_RUNNING=false
fi

# Check Redis
if check_service "Redis" "6379" "http://localhost:6379"; then
    REDIS_RUNNING=true
else
    REDIS_RUNNING=false
    # Alternative check for Redis
    if command -v redis-cli &> /dev/null && redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis is running (detected via redis-cli)"
        REDIS_RUNNING=true
    fi
fi

echo ""
echo "📋 Service Status Summary"
echo "========================"
echo "FastAPI Application: $([ "$FASTAPI_RUNNING" = true ] && echo "✅ Running" || echo "❌ Not Running")"
echo "Redis Server:        $([ "$REDIS_RUNNING" = true ] && echo "✅ Running" || echo "❌ Not Running")"

echo ""
echo "🛠️  Setup Instructions"
echo "====================="

if [ "$FASTAPI_RUNNING" = false ]; then
    echo ""
    echo "To start FastAPI (choose one):"
    echo "  # Development mode (recommended for testing):"
    echo "  poetry run fastapi dev --host 0.0.0.0 --port 8000"
    echo ""
    echo "  # Production mode:"
    echo "  poetry run fastapi run --host 0.0.0.0 --port 8000"
    echo ""
    echo "  # Alternative with uvicorn directly:"
    echo "  poetry run uvicorn inference_core.main_factory:create_application --factory --host 0.0.0.0 --port 8000 --reload"
fi

if [ "$REDIS_RUNNING" = false ]; then
    echo ""
    echo "To start Redis (choose one):"
    echo "  # Using Docker (recommended):"
    echo "  docker run -d --name redis-test -p 6379:6379 redis:7-alpine"
    echo ""
    echo "  # Using local Redis installation:"
    echo "  redis-server"
    echo ""
    echo "  # Using Docker Compose (if available):"
    echo "  docker-compose up redis -d"
fi

echo ""
echo "To start Celery worker (optional, for task monitoring endpoints):"
echo "  poetry run celery -A inference_core.celery.celery_main:celery_app worker --loglevel=info --queues=default"

echo ""
echo "🧪 Running Performance Tests"
echo "============================"
echo ""
echo "Quick test (light profile):"
echo "  cd tests/performance"
echo "  poetry run locust -f locustfile.py --host http://localhost:8000 --headless -u 3 -r 1 -t 30s --html ../../reports/performance/quick_test.html"
echo ""
echo "Interactive mode (web UI):"
echo "  cd tests/performance"
echo "  poetry run locust -f locustfile.py --host http://localhost:8000"
echo "  # Then open http://localhost:8089"
echo ""
echo "All available profiles:"
echo "  LOAD_PROFILE=light    # 5 users, 1 minute"
echo "  LOAD_PROFILE=medium   # 20 users, 5 minutes"
echo "  LOAD_PROFILE=heavy    # 50 users, 10 minutes"
echo "  LOAD_PROFILE=spike    # 100 users, 3 minutes"
echo "  LOAD_PROFILE=endurance # 25 users, 30 minutes"

echo ""
echo "📊 Reports"
echo "========="
echo "HTML reports will be saved to: reports/performance/"
echo "View reports in your browser after running tests."

if [ "$FASTAPI_RUNNING" = true ] && [ "$REDIS_RUNNING" = true ]; then
    echo ""
    echo "🎉 All services are running! You can start performance testing immediately."
elif [ "$FASTAPI_RUNNING" = true ]; then
    echo ""
    echo "⚠️  FastAPI is running, but Redis is not. Task monitoring endpoints will fail."
    echo "   You can still test health and auth endpoints."
else
    echo ""
    echo "⚠️  Please start the required services before running performance tests."
fi

echo ""
echo "For detailed instructions, see: tests/performance/README.md"
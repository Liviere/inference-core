#!/bin/bash
# Validation script for isolated test Docker environment
# This script validates the compose files and environment configuration

set -e  # Exit on any error

echo "=== Isolated Test Docker Environment Validation ==="
echo ""

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    echo "Error: Please run this script from the repository root"
    exit 1
fi

echo "1. Checking directory structure..."
if [[ -d "docker/tests" ]]; then
    echo "✓ docker/tests/ directory exists"
else
    echo "✗ docker/tests/ directory missing"
    exit 1
fi

echo ""
echo "2. Checking required files..."
required_files=(
    "docker/tests/.env.test.example"
    "docker/tests/docker-compose.test.sqlite.yml"
    "docker/tests/docker-compose.test.postgres.yml"
    "docker/tests/docker-compose.test.mysql.yml"
    "docs/testing-docker.md"
)

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✓ $file exists"
    else
        echo "✗ $file missing"
        exit 1
    fi
done

echo ""
echo "3. Validating Docker Compose file syntax..."
compose_files=(
    "docker/tests/docker-compose.test.sqlite.yml"
    "docker/tests/docker-compose.test.postgres.yml"  
    "docker/tests/docker-compose.test.mysql.yml"
)

for file in "${compose_files[@]}"; do
    if docker compose -f "$file" config --quiet > /dev/null 2>&1; then
        echo "✓ $file syntax valid"
    else
        echo "✗ $file syntax invalid"
        exit 1
    fi
done

echo ""
echo "4. Checking environment file structure..."
env_file="docker/tests/.env.test.example"

# Check for required environment variables
required_vars=(
    "ENVIRONMENT=testing"
    "PORT=8100"
    "REDIS_PORT=6380"
    "DATABASE_PORT_POSTGRES=55432"
    "DATABASE_PORT_MYSQL=33306"
    "SECRET_KEY=test-secret-key-do-not-use-in-production"
)

for var in "${required_vars[@]}"; do
    if grep -q "^$var" "$env_file"; then
        echo "✓ Found $var"
    else
        echo "✗ Missing $var"
        exit 1
    fi
done

echo ""
echo "5. Checking port mappings..."
# Verify ports are correctly mapped in compose files
ports_check=(
    "docker/tests/docker-compose.test.sqlite.yml:8100"
    "docker/tests/docker-compose.test.sqlite.yml:6380"
    "docker/tests/docker-compose.test.postgres.yml:55432"
    "docker/tests/docker-compose.test.mysql.yml:33306"
)

for check in "${ports_check[@]}"; do
    file="${check%:*}"
    port="${check#*:}"
    if grep -q "$port" "$file"; then
        echo "✓ Port $port found in $(basename "$file")"
    else
        echo "✗ Port $port missing in $(basename "$file")"
        exit 1
    fi
done

echo ""
echo "6. Checking service names..."
# Verify all services have -test suffix
service_names=(
    "app-test"
    "celery-worker-test"
    "redis-test"
    "postgres-test"
    "mysql-test"
)

for service in "${service_names[@]}"; do
    found=false
    for file in "${compose_files[@]}"; do
        if grep -q "^  $service:" "$file"; then
            found=true
            break
        fi
    done
    if $found; then
        echo "✓ Service $service found"
    else
        echo "✗ Service $service missing"
        exit 1
    fi
done

echo ""
echo "7. Checking tmpfs configurations..."
# Verify ephemeral storage configurations
if grep -q "tmpfs:" docker/tests/docker-compose.test.postgres.yml && \
   grep -q "tmpfs:" docker/tests/docker-compose.test.mysql.yml && \
   grep -q "tmpfs:" docker/tests/docker-compose.test.sqlite.yml; then
    echo "✓ tmpfs configurations found"
else
    echo "✗ tmpfs configurations missing"
    exit 1
fi

echo ""
echo "8. Checking documentation..."
if [[ -f "docs/testing-docker.md" ]] && grep -q "Port Mapping" "docs/testing-docker.md"; then
    echo "✓ Documentation exists and contains port mapping table"
else
    echo "✗ Documentation incomplete"
    exit 1
fi

if grep -q "Isolated Test Docker Environment" "README.md"; then
    echo "✓ README.md updated with test environment section"
else
    echo "✗ README.md not updated"
    exit 1
fi

echo ""
echo "=== ✓ All validations passed! ==="
echo ""
echo "The isolated test Docker environment is properly configured with:"
echo "  • Three database backends (SQLite, PostgreSQL, MySQL)"
echo "  • Alternative ports to avoid conflicts"
echo "  • Ephemeral storage (no persistence)"
echo "  • Service names with -test suffix"
echo "  • Comprehensive documentation"
echo ""
echo "To use the test environment:"
echo "  1. cp docker/tests/.env.test.example docker/tests/.env.test"
echo "  2. docker compose -f docker/tests/docker-compose.test.postgres.yml --env-file docker/tests/.env.test up -d"
echo "  3. curl http://localhost:8100/api/v1/health/ping"
echo "  4. docker compose -f docker/tests/docker-compose.test.postgres.yml down -v"
echo ""
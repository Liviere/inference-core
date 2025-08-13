# Isolated Test Docker Environment

This document describes the isolated Docker environment for running automated tests in a stateless, non-conflicting manner.

## Overview & Rationale

The isolated test Docker environment provides:

1. **Stateless Operation**: No persistent volumes or named volumes - all data is ephemeral
2. **Non-Conflicting**: Uses alternative ports and service names to avoid conflicts with development/production stacks
3. **Parallel Execution**: Can run alongside development environments without port conflicts
4. **Easy Cleanup**: Single command removes all containers and ephemeral data
5. **Database Coverage**: Supports SQLite, PostgreSQL, and MySQL backends for comprehensive testing

This environment is designed for:
- Automated test execution (unit, integration, future performance tests)
- CI/CD pipelines
- Isolated testing without affecting development data
- Clean, reproducible test runs

## File Structure

```
docker/tests/
├── .env.test.example          # Environment variables template for testing
├── docker-compose.test.sqlite.yml    # SQLite test environment
├── docker-compose.test.postgres.yml  # PostgreSQL test environment
└── docker-compose.test.mysql.yml     # MySQL test environment
```

## Port Mapping

The test environment uses alternative ports to avoid conflicts with development stacks:

| Service | Development Port | Test Port | Purpose |
|---------|------------------|-----------|---------|
| FastAPI App | 8000 | 8100 | Main application |
| Redis | 6379 | 6380 | Celery broker/sessions |
| PostgreSQL | 5432 | 55432 | Database (postgres only) |
| MySQL | 3306 | 33306 | Database (mysql only) |
| Flower | 5555 | 5556 | Celery monitoring (optional) |

### Redis Database Indices

The test environment uses distinct Redis database indices:
- DB 0: Celery broker
- DB 1: Celery result backend  
- DB 10: App refresh sessions

## Quick Start

### 1. Prepare Environment

```bash
# Copy the test environment template
cp docker/tests/.env.test.example docker/tests/.env.test

# Optionally customize the .env.test file for your needs
```

### 2. Start Test Environment

Choose your database backend:

**SQLite (Fastest startup, in-memory/ephemeral file):**
```bash
docker compose -f docker/tests/docker-compose.test.sqlite.yml --env-file docker/tests/.env.test up -d
```

**PostgreSQL (Full database testing):**
```bash
docker compose -f docker/tests/docker-compose.test.postgres.yml --env-file docker/tests/.env.test up -d
```

**MySQL (Alternative database testing):**
```bash
docker compose -f docker/tests/docker-compose.test.mysql.yml --env-file docker/tests/.env.test up -d
```

### 3. Verify Health

```bash
# Check application health
curl http://localhost:8100/api/v1/health/ping

# Expected response: {"message": "pong", "timestamp": "...", "status": "ok"}
```

### 4. Run Tests

Once the environment is up, you can run tests from inside containers or from your host system pointing to the test ports.

## Cleanup Instructions

**Important**: Always use the `-v` flag to remove ephemeral volumes.

```bash
# For SQLite
docker compose -f docker/tests/docker-compose.test.sqlite.yml down -v

# For PostgreSQL  
docker compose -f docker/tests/docker-compose.test.postgres.yml down -v

# For MySQL
docker compose -f docker/tests/docker-compose.test.mysql.yml down -v
```

**Verify cleanup:**
```bash
# Check no test containers remain
docker ps -a | grep backend-template-test

# Check no test networks remain
docker network ls | grep test

# Check no new volumes were created (should show empty or only pre-existing volumes)
docker volume ls
```

## Configuration Details

### Environment Variables

Key test environment variables in `.env.test.example`:

- `ENVIRONMENT=testing` - Sets application to testing mode
- `SECRET_KEY=test-secret-key-do-not-use-in-production` - Non-production secret
- `DEBUG=false` - Disable debug mode for cleaner logs
- `RUN_LLM_REAL_TESTS=0` - Disable real LLM API calls by default
- Alternative ports for all services to prevent conflicts

### Service Configuration

**App & Celery Worker:**
- `ENVIRONMENT=testing` for test-specific behavior
- Reduced timeouts and worker limits for faster testing
- Ephemeral SQLite database (`/tmp/app_test.db`) or tmpfs-backed database storage
- Test-specific Redis database indices

**Databases:**
- **PostgreSQL**: Uses tmpfs mount (`/var/lib/postgresql/data`) for ephemeral storage
- **MySQL**: Uses tmpfs mount (`/var/lib/mysql`) with optimized test settings
- **SQLite**: Uses ephemeral file in container tmp directory

**Redis:**
- Uses tmpfs mount (`/data`) for ephemeral storage
- `--appendonly no` and `--save ""` to prevent persistence
- Healthchecks with shorter intervals for faster startup detection

### Healthchecks

All services include healthchecks for reliable dependency orchestration:
- **App**: HTTP ping endpoint check
- **Databases**: Native health commands (`pg_isready`, `mysqladmin ping`)
- **Redis**: Redis ping command

## Tips for CI Usage

### GitHub Actions Example

```yaml
- name: Start Test Environment
  run: |
    cp docker/tests/.env.test.example docker/tests/.env.test
    docker compose -f docker/tests/docker-compose.test.postgres.yml --env-file docker/tests/.env.test up -d

- name: Wait for Services
  run: |
    timeout 60 bash -c 'until curl -f http://localhost:8100/api/v1/health/ping; do sleep 2; done'

- name: Run Tests
  run: |
    # Your test commands here, pointing to localhost:8100

- name: Cleanup
  run: |
    docker compose -f docker/tests/docker-compose.test.postgres.yml down -v
```

### Performance Considerations

- **SQLite**: Fastest startup, suitable for unit tests and basic integration
- **PostgreSQL**: Moderate startup time, best for full database feature testing
- **MySQL**: Slower startup due to initialization, use for MySQL-specific testing

### Resource Limits

The tmpfs mounts include size limits:
- Redis: 100MB (sufficient for test data)
- PostgreSQL: 500MB (handles moderate test datasets)
- MySQL: 500MB (includes InnoDB buffer pool sizing)

## Switching Between Databases

To switch between database backends:

1. Stop current environment: `docker compose -f docker/tests/docker-compose.test.{current}.yml down -v`
2. Start new environment: `docker compose -f docker/tests/docker-compose.test.{new}.yml --env-file docker/tests/.env.test up -d`

No data is preserved between switches due to the ephemeral nature.

## Internal-Only Networking (Alternative)

To run tests with no published ports (maximum isolation):

1. Remove the `ports:` sections from the compose files
2. Run tests from within containers or using Docker network access
3. Access services using internal hostnames (`app-test:8100`, `redis-test:6380`, etc.)

## Troubleshooting

**Port conflicts:**
- Ensure development environments are stopped or use different ports
- Check `docker ps` and `netstat -tulpn | grep :8100` to identify conflicts

**Container startup issues:**
- Check logs: `docker compose -f docker/tests/docker-compose.test.{db}.yml logs`
- Verify Docker has sufficient resources (especially for MySQL)

**Test connectivity:**
- Use healthcheck endpoints: `curl http://localhost:8100/api/v1/health/`
- Check Redis: `docker exec backend-template-test-redis redis-cli -p 6380 ping`

**Memory issues:**
- MySQL requires more memory - ensure Docker has at least 2GB allocated
- tmpfs sizes can be adjusted in compose files if needed

## Integration with Performance Testing

This environment can be consumed by performance testing tools like Locust:

```bash
# Start test environment
docker compose -f docker/tests/docker-compose.test.postgres.yml --env-file docker/tests/.env.test up -d

# Run Locust against test environment
poetry run locust -f tests/performance/locustfile.py --host http://localhost:8100
```

## Security Notes

- **Never use test secrets in production** - the `.env.test.example` contains obviously non-production values
- Test environment is designed for isolated testing only
- No data persistence means no security risks from leftover test data
- Consider network isolation in shared environments
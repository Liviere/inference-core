# Performance Testing Guide

This directory contains comprehensive performance tests for the FastAPI Backend Template using Locust. Default profiles avoid provider-backed LLM traffic, while the `llm_mock` profile exercises agent, embedding, and vector workflows only against a no-cost emulated environment.

## Overview

The performance test suite includes:

- **Health Check Tests**: Root endpoint, health checks, database health, ping
- **Health Check (Full) Tests**: A heavier variant that aggregates multiple health calls; split into a separate user class to control load
- **Authentication Flow Tests**: Registration, login, profile management, password changes, token refresh, logout
- **Task System Tests**: Task health monitoring, worker statistics, active tasks
- **Database Health Tests**: Focused database performance testing
- **No-Cost LLM Mock Tests**: Agent instances, emulated agent runs, fake embeddings, and vector search workflows

## Prerequisites

### Required Services

Before running performance tests, ensure these services are running:

1. **FastAPI Application** (development mode recommended):

   ```bash
   poetry run fastapi dev --host 0.0.0.0 --port 8000
   ```

2. **Redis** (required for Celery and refresh token sessions):

   ```bash
   # Using Docker (recommended)
   docker run -d --name redis-test -p 6379:6379 redis:7-alpine

   # Or install locally and run:
   # redis-server
   ```

3. **Celery Worker** (required for task monitoring endpoints):
   ```bash
   poetry run celery -A inference_core.celery.celery_main:celery_app worker --loglevel=info --queues=default
   ```

### Environment Setup

1. Ensure dependencies are installed:

   ```bash
   poetry install
   ```

2. Copy configuration files if not already done:

   ```bash
   cp .env.example .env
   cp llm_config.example.yaml llm_config.yaml
   ```

3. For the no-cost LLM mock profile, configure the API process with explicit guardrails:

   ```bash
   LLM_EMULATION_ENABLED=true
   EMBEDDING_BACKEND=fake
   VECTOR_BACKEND=memory
   LLM_API_ACCESS_MODE=user
   AGENT_TOOL_ENVIRONMENT=strict_test
   AGENT_REQUIRE_TEST_DOUBLES=true
   AGENT_TOOL_DOUBLE_STRATEGY=replace
   LLM_TOOL_EMULATION_MODE=external
   ```

   `LOAD_PROFILE=llm_mock` also checks the local Locust environment for `LLM_EMULATION_ENABLED=true` and `EMBEDDING_BACKEND=fake` before it starts. If Locust targets a separately managed test server where those variables are set only on the server, set `LOCUST_ALLOW_UNSAFE_LLM_TRAFFIC=true` only after verifying that server-side no-cost configuration.

## Load Profiles

The test suite includes predefined load profiles for different testing scenarios:

### Light Profile (`light`)

- **Purpose**: Smoke testing and development
- **Users**: 10
- **Duration**: 1 minute
- **Spawn Rate**: 1 user/second
- **Use Case**: Quick validation, CI/CD pipelines

### Medium Profile (`medium`)

- **Purpose**: Regular performance testing
- **Users**: 20
- **Duration**: 5 minutes
- **Spawn Rate**: 2 users/second
- **Use Case**: Development performance validation

### Heavy Profile (`heavy`)

- **Purpose**: Stress testing
- **Users**: 50
- **Duration**: 10 minutes
- **Spawn Rate**: 5 users/second
- **Use Case**: Pre-production stress testing

### Spike Profile (`spike`)

- **Purpose**: Spike testing with rapid ramp-up
- **Users**: 100
- **Duration**: 3 minutes
- **Spawn Rate**: 10 users/second
- **Use Case**: Testing system resilience to traffic spikes

### Endurance Profile (`endurance`)

- **Purpose**: Long-running stability testing
- **Users**: 50
- **Duration**: 30 minutes
- **Spawn Rate**: 1 user/second
- **Use Case**: Memory leak detection, long-term stability

### LLM Mock Profile (`llm_mock`)

- **Purpose**: Realistic user-workspace traffic without paid provider calls
- **Users**: 25
- **Duration**: 5 minutes
- **Spawn Rate**: 2.5 users/second
- **Use Case**: E2E/performance validation of agent instances, emulated agent runs, fake embeddings, and vector search
- **Requires**: `LLM_EMULATION_ENABLED=true`, `EMBEDDING_BACKEND=fake`, and preferably `VECTOR_BACKEND=memory`

## Running Performance Tests

### Basic Usage

1. Start all required services (see Prerequisites above)

2. Run with default light profile:

   ```bash
   cd tests/performance
   poetry run locust -f locustfile.py --host http://localhost:8000
   ```

3. Open the Locust web UI at: http://localhost:8089

4. Configure users and spawn rate in the web UI, or use predefined profiles

### Using Predefined Profiles

Run with specific load profiles using environment variables (note: explicit -u/-r/-t override profile defaults):

```bash
# Light load (default)
LOAD_PROFILE=light poetry run locust -f tests/performance/locustfile.py --host http://localhost:8000 --headless -u 10 -r 1 -t 1m --html reports/performance/light_load_report.html

# Medium load
LOAD_PROFILE=medium poetry run locust -f tests/performance/locustfile.py --host http://localhost:8000 --headless -u 20 -r 2 -t 5m --html reports/performance/medium_load_report.html

# Heavy load
LOAD_PROFILE=heavy poetry run locust -f tests/performance/locustfile.py --host http://localhost:8000 --headless -u 50 -r 5 -t 10m --html reports/performance/heavy_load_report.html

# Spike test
LOAD_PROFILE=spike poetry run locust -f tests/performance/locustfile.py --host http://localhost:8000 --headless -u 100 -r 10 -t 3m --html reports/performance/spike_load_report.html

# Endurance test
LOAD_PROFILE=endurance poetry run locust -f tests/performance/locustfile.py --host http://localhost:8000 --headless -u 50 -r 1 -t 30m --html reports/performance/endurance_load_report.html

# No-cost LLM mock traffic
LLM_EMULATION_ENABLED=true EMBEDDING_BACKEND=fake LOAD_PROFILE=llm_mock \
  poetry run locust -f tests/performance/locustfile.py \
  --host http://localhost:8000 \
  --headless \
  -u 25 \
  -r 2.5 \
  -t 5m \
  --html reports/performance/llm_mock_load_report.html
```

### Headless Mode (CI/CD)

For automated testing without the web UI:

```bash
poetry run locust -f locustfile.py \
  --host http://localhost:8000 \
  --headless \
  --users 10 \
  --spawn-rate 2 \
  --run-time 2m \
  --html reports/performance/ci_report.html \
  --csv reports/performance/ci_results
```

### Custom Configuration

Override target host:

```bash
poetry run locust -f locustfile.py --host http://staging.yourapi.com
```

Set custom profile with environment variables:

```bash
TARGET_HOST=http://production.yourapi.com \
LOAD_PROFILE=medium \
poetry run locust -f locustfile.py --headless -u 20 -r 2 -t 5m
```

## Test Scenarios

### HealthCheckUser

Tests basic system health endpoints:

- `GET /` - Root endpoint
- `GET /api/v1/health/` - Overall health check
- `GET /api/v1/health/database` - Database health
- `GET /api/v1/health/ping` - Simple ping
- `GET /docs` - API documentation (development only)

### HealthCheckFullUser

A heavier health-checking user separated out to avoid overloading the API unintentionally. Use this to apply controlled pressure on combined or more expensive health routes.

- Typically aggregates multiple health endpoints into one flow
- Weight is configurable per profile (see `config.py`), defaulting to a low weight

### AuthUserFlow

Tests complete authentication workflows:

- User registration with unique credentials
- Login and token acquisition
- Profile retrieval and updates
- Password changes
- Token refresh operations
- Logout and session cleanup
- Password reset requests

### TasksMonitoringUser

Tests task system monitoring:

- `GET /api/v1/tasks/health` - Task system health
- `GET /api/v1/tasks/workers/stats` - Worker statistics
- `GET /api/v1/tasks/active` - Active tasks information

### LLMMockWorkspaceUser

Tests a realistic no-cost user session. Each simulated user registers, logs in, creates one agent instance from the configured templates, seeds a small vector collection, then repeats read/write workflows:

- `GET /api/v1/agent-instances/templates` - Browse available agent templates
- `POST /api/v1/agent-instances` - Create a user-owned agent instance
- `GET /api/v1/agent-instances` and `GET /api/v1/agent-instances/{id}` - Refresh workspace state
- `PATCH /api/v1/agent-instances/{id}` - Update harmless agent metadata
- `POST /api/v1/agent-instances/{id}/run` - Run the agent through the emulated LLM path
- `POST /api/v1/embeddings/generate` - Generate deterministic fake embeddings and fail if the backend is not `fake`
- `GET /api/v1/vector/health` - Check vector readiness
- `POST /api/v1/vector/ingest` with `async_mode=false` - Seed and update a small knowledge base without Celery ingestion
- `POST /api/v1/vector/query` and `POST /api/v1/vector/list` - Exercise search and listing
- `GET /api/v1/vector/collections/{collection}/stats` - Refresh collection statistics

## Endpoint and Mock Readiness

| Area                           | Locust status                   | No-cost readiness                                                                                                   |
| ------------------------------ | ------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Health/auth/tasks              | Covered by existing profiles    | Safe; no paid providers involved                                                                                    |
| Agent instances and agent runs | Covered by `llm_mock`           | Safe when `LLM_EMULATION_ENABLED=true` and tool exposure uses `strict_test` or emulated tools                       |
| Embeddings                     | Covered by `llm_mock`           | Safe only with `EMBEDDING_BACKEND=fake`; the scenario fails responses that report another backend                   |
| Vector store                   | Covered by `llm_mock`           | Safe with fake embeddings; `VECTOR_BACKEND=memory` is recommended for isolated load runs                            |
| Batch jobs                     | Not included in default traffic | Full batch lifecycle still needs a fake batch provider before it is safe to load-test provider submission/execution |

## Excluded or Deferred Endpoints

The following endpoints are excluded or deferred to avoid costs and complexity:

- **Provider-backed LLM traffic**: real model/provider calls are not included in default profiles; use `llm_mock` only with emulation enabled
- **Full batch lifecycle**: create/submit/provider execution is deferred until a fake batch provider is implemented
- **Task result endpoints**: `/api/v1/tasks/{task_id}/status`, `/api/v1/tasks/{task_id}/result`
- **Task cancellation**: `DELETE /api/v1/tasks/{task_id}`
- **Email delivery**: Password reset email sending

## Output and Reports

### HTML Reports

Reports are generated in `reports/performance/` with naming pattern:

- `{profile_name}_load_report.html`

Example: `light_load_report.html`, `heavy_load_report.html`

### CSV Data

For detailed analysis, generate CSV files:

```bash
poetry run locust -f locustfile.py --host http://localhost:8000 --headless -u 10 -r 2 -t 2m --csv reports/performance/detailed_results
```

This creates:

- `detailed_results_stats.csv` - Request statistics
- `detailed_results_stats_history.csv` - Timeline data
- `detailed_results_failures.csv` - Failure details

## Performance Thresholds

The test suite includes performance thresholds for regression detection:

```python
PERFORMANCE_THRESHOLDS = {
    "health_p95_ms": 100,      # Health endpoints should be fast
    "auth_p95_ms": 500,        # Auth operations can be slower
    "tasks_p95_ms": 200,       # Task monitoring should be responsive
    "agent_run_p95_ms": 2500,  # Emulated agent runs exercise LangChain setup
    "embedding_p95_ms": 300,   # Fake embeddings should be lightweight
    "vector_p95_ms": 800,      # Vector workflows include storage/search
    "overall_failure_rate": 0.01,  # Less than 1% failure rate
}
```

## Troubleshooting

### Common Issues

1. **Connection Refused**

   ```
   ConnectionError: [Errno 111] Connection refused
   ```

   - Ensure FastAPI app is running on the correct host/port
   - Check firewall settings
   - Verify host URL in command

2. **Authentication Failures**

   ```
   401 Unauthorized responses
   ```

   - Ensure database is properly initialized
   - Check if user registration is working
   - Verify JWT configuration in .env

3. **Task Endpoints Failing**

   ```
   500 Internal Server Error on /api/v1/tasks/*
   ```

   - Ensure Redis is running and accessible
   - Start Celery worker process
   - Check Celery configuration in .env

4. **High Error Rates**
   - Reduce user count or spawn rate
   - Check system resources (CPU, memory, database connections)
   - Monitor application logs for errors

### Debugging

Enable verbose logging:

```bash
poetry run locust -f locustfile.py --host http://localhost:8000 --loglevel DEBUG
```

Test individual endpoints:

```bash
curl -v http://localhost:8000/api/v1/health/
curl -v http://localhost:8000/api/v1/health/ping
```

## CI/CD Integration

### Light Smoke Test

For CI/CD pipelines, use a very light smoke test:

```yaml
# .github/workflows/performance.yml
name: Performance Smoke Test
on:
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  performance-smoke:
    if: contains(github.event.pull_request.labels.*.name, 'performance-test')
    runs-on: ubuntu-latest

    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.14'

      - name: Install dependencies
        run: |
          pip install poetry
          poetry install

      - name: Setup environment
        run: |
          cp .env.example .env
          cp llm_config.example.yaml llm_config.yaml

      - name: Start application
        run: |
          poetry run fastapi run --host 0.0.0.0 --port 8000 &
          sleep 10

      - name: Run smoke test
        run: |
          cd tests/performance
          LOAD_PROFILE=light poetry run locust -f locustfile.py \
            --host http://localhost:8000 \
            --headless \
            --users 2 \
            --spawn-rate 1 \
            --run-time 30s \
            --html reports/performance/ci_smoke_report.html

      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: performance-smoke-report
          path: reports/performance/ci_smoke_report.html
```

## Best Practices

1. **Start Small**: Begin with light profile, then scale up
2. **Monitor Resources**: Watch CPU, memory, and database connections
3. **Realistic Data**: Use varied test data to simulate real usage
4. **Gradual Ramp**: Use appropriate spawn rates to avoid overwhelming the system
5. **Environment Isolation**: Run performance tests in isolated environments
6. **Baseline Establishment**: Establish performance baselines for comparison
7. **Regular Testing**: Include performance tests in CI/CD for regression detection

## Extending the Tests

### Adding New Endpoints

1. Create a new user class or extend existing ones:

   ```python
   class MyNewUser(BaseUser):
       weight = 2

       @task(1)
       def test_my_endpoint(self):
           with self.client.get("/api/v1/my/endpoint") as response:
               if response.status_code == 200:
                   response.success()
               else:
                   response.failure(f"Failed: {response.status_code}")
   ```

2. Add the user class to profile weights in `config.py`

3. Update the `get_user_classes()` function in `locustfile.py`

### Custom Load Profiles

Add new profiles to `LOAD_PROFILES` in `config.py`:

```python
"custom": LoadProfile(
    name="custom",
    description="Custom load profile",
    users=30,
    spawn_rate=3.0,
    run_time="7m",
    weight_config={
        "HealthCheckUser": 10,
        "HealthCheckFullUser": 1,
        "AuthUserFlow": 15,
        "TasksMonitoringUser": 1,
    }
```

)

## Support

For issues with performance testing:

1. Check application logs for errors
2. Verify all prerequisites are running
3. Test individual endpoints manually first
4. Start with the light profile and scale up gradually
5. Monitor system resources during tests

For questions about specific endpoints or expected behavior, refer to the main application documentation.

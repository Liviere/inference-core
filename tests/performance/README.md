# Performance Testing Guide

This directory contains comprehensive performance tests for the FastAPI Backend Template using Locust. Default profiles avoid provider-backed LLM traffic, while the `llm_mock` profile exercises agent, embedding, and vector workflows only against a no-cost emulated environment.

## Overview

The performance test suite includes:

- **Health Check Tests**: Root endpoint, health checks, database health, ping
- **Health Check (Full) Tests**: A heavier variant that aggregates multiple health calls; split into a separate user class to control load
- **Authentication Flow Tests**: Registration, login, profile management, password changes, token refresh, logout
- **Task System Tests**: Task health monitoring, worker statistics, active tasks
- **Database Health Tests**: Focused database performance testing
- **No-Cost LLM Mock Tests**: Agent instances, emulated agent runs, fake or local embeddings, and vector search workflows

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

3. Initialize the database schema before any auth-dependent profile (`light`, `medium`, `heavy`, `spike`, `endurance`, `llm_mock`):

For a fresh local test database, use the bootstrap helper first:

```bash
ENVIRONMENT=testing poetry run python scripts/bootstrap_test_db.py
```

For an already initialized database that only needs pending delta migrations, use:

```bash
ENVIRONMENT=testing poetry run alembic upgrade head
```

4. For the no-cost LLM mock profile:

Wrapper scripts in `scripts/perf_tests/run_perf_*.py` automatically load root `.env.test` when that file exists. This means they can derive the default target host from `TARGET_HOST` or, when absent, from `HOST` plus `PORT`, so local test runs usually do not need an explicit `--host` flag.

The shared performance launcher also rotates report artifacts by day under `reports/performance/YYYY/MM/DD` and reuses the same suffix for HTML and CSV outputs when you run the same profile multiple times. Use `--html`, `--csv-prefix`, `--output-dir`, or `--name-suffix` when you need a custom artifact layout.

When the API process starts with `ENVIRONMENT=testing`, the application now defaults to safe no-cost runtime guardrails unless you override them explicitly:

```bash
LLM_EMULATION_ENABLED=true
EMBEDDING_BACKEND=fake
AGENT_TOOL_ENVIRONMENT=strict_test
AGENT_REQUIRE_TEST_DOUBLES=true
AGENT_TOOL_DOUBLE_STRATEGY=replace
LLM_TOOL_EMULATION_MODE=external
```

If you run the API outside `ENVIRONMENT=testing`, set the guardrails explicitly:

```bash
LLM_EMULATION_ENABLED=true
EMBEDDING_BACKEND=fake  # or local
VECTOR_BACKEND=memory
LLM_API_ACCESS_MODE=user
AGENT_TOOL_ENVIRONMENT=strict_test
AGENT_REQUIRE_TEST_DOUBLES=true
AGENT_TOOL_DOUBLE_STRATEGY=replace
LLM_TOOL_EMULATION_MODE=external
```

When you explicitly choose `EMBEDDING_BACKEND=local`, also start the dedicated embeddings worker so `/api/v1/embeddings/generate` and vector ingestion can resolve SentenceTransformer requests:

```bash
poetry run celery -A inference_core.celery.celery_main:celery_app worker -n embeddings@%h --queues=embeddings --pool=solo --loglevel=info
```

The `run_perf_llm_mock.py` wrapper layers a performance-oriented emulation profile on top of those guardrails by default:

- `LLM_EMULATION_LATENCY_MS=3000`
- `LLM_EMULATION_LATENCY_JITTER_MS=500`
- `LLM_EMULATION_SESSION_SCALE_MIN=0.9`
- `LLM_EMULATION_SESSION_SCALE_MAX=5.0`
- `LLM_EMULATION_STEP_LATENCY_GROWTH=0.35`
- `LLM_EMULATION_STREAM_FIRST_CHUNK_RATIO=0.25`

Those values make repeated agent sessions feel more like real traffic while staying fully no-cost. Override them if you want a lighter smoke run or a heavier stress profile.

`LOAD_PROFILE=llm_mock` also checks the local Locust environment for `LLM_EMULATION_ENABLED=true` and `EMBEDDING_BACKEND` set to either `fake` or `local` before it starts. If Locust targets a separately managed test server where those variables are set only on the server, set `LOCUST_ALLOW_UNSAFE_LLM_TRAFFIC=true` only after verifying that server-side no-cost configuration.

Auth-dependent profiles also run a startup preflight against `/api/v1/auth/register` and `/api/v1/auth/login`. If the target is missing schema bootstrap, the `users` table, or a usable auth configuration, Locust stops immediately with an actionable error instead of spending the whole run reporting secondary `401` and `500` failures. Set `LOCUST_SKIP_AUTH_PREFLIGHT=true` only when you intentionally want to bypass that safety check.

## Session Model

Authenticated Locust scenarios are stateful on a per-user basis:

- Each `HttpUser` instance keeps its own credentials, access token metadata, and scenario-owned resources in Python instance attributes.
- The Locust HTTP client cookie jar carries the refresh token cookie, so login, refresh, and logout behave like a browser session rather than a stateless API script.
- Protected flows use a shared auth helper that refreshes the access token proactively before expiry and falls back to re-login when the refresh session was revoked or expired.
- Logout is exercised both at user shutdown and during selected tasks, so the suite covers session teardown and recovery, not only initial login.

Because refresh-token revocation is backed by Redis, authenticated performance tests should be treated as incomplete when Redis is unavailable.

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
- **Use Case**: E2E/performance validation of agent instances, emulated agent runs, fake or local embeddings, vector search, and Celery-backed vector ingestion
- **Requires**: `LLM_EMULATION_ENABLED=true`, `EMBEDDING_BACKEND=fake|local`, and preferably `VECTOR_BACKEND=memory`

`llm_mock` now submits vector ingestion through `/api/v1/vector/ingest` with `async_mode=true` and polls `/api/v1/tasks/{task_id}/status` plus `/api/v1/tasks/{task_id}/result` until the Celery worker completes the job. This means the main worker should now show `vector.ingest_documents` activity during the profile run.

The embeddings worker is still only involved when the API process runs with `EMBEDDING_BACKEND=local`. With the default testing-safe `EMBEDDING_BACKEND=fake`, `/api/v1/embeddings/generate` stays in-process by design, so you should expect activity on the main worker but not on the dedicated `embeddings` worker.

## Running Performance Tests

### Basic Usage

1. Start all required services (see Prerequisites above)

2. Run the default light profile through the wrapper script:

   ```bash
   poetry run python scripts/perf_tests/run_perf_light.py
   ```

3. To use the Locust web UI instead of headless mode:

   ```bash
   poetry run python scripts/perf_tests/run_perf_light.py --web-ui
   ```

4. Open the Locust web UI at: http://localhost:8089

5. Configure users and spawn rate in the web UI, or use predefined profiles

### Using Predefined Profiles

Wrapper scripts are now the recommended entrypoint. They set `LOAD_PROFILE` for you, load root `.env.test` automatically when present, apply profile defaults, generate the standard HTML report by default, and still let you override the important Locust knobs. They also use a shared launcher that prints the effective command, applies daily report rotation, and keeps HTML/CSV artifact naming aligned across runs.

```bash
# Light load (default)
poetry run python scripts/perf_tests/run_perf_light.py

# Medium load
poetry run python scripts/perf_tests/run_perf_medium.py

# Heavy load
poetry run python scripts/perf_tests/run_perf_heavy.py

# Spike test
poetry run python scripts/perf_tests/run_perf_spike.py

# Endurance test
poetry run python scripts/perf_tests/run_perf_endurance.py

# No-cost LLM mock traffic with the default fake embeddings backend
poetry run python scripts/perf_tests/run_perf_llm_mock.py

# Same profile with a local SentenceTransformer backend (requires embeddings worker)
poetry run python scripts/perf_tests/run_perf_llm_mock.py \
  --embedding-backend local \
  --name-suffix local-embeddings
```

Common wrapper flags:

- `--users`, `--spawn-rate`, `--duration` override the profile defaults
- `--host` still overrides the target explicitly when you want to bypass the value loaded from `.env.test`
- `--web-ui` switches from headless mode to the Locust UI
- `--csv` adds the standard Locust CSV outputs next to the HTML report
- `--html`, `--csv-prefix`, `--output-dir`, and `--name-suffix` customize artifact paths
- `--no-html` disables the default HTML report for one run
- `--skip-auth-preflight` bypasses the auth startup safety check when you really need it
- `scripts/perf_tests/run_perf_llm_mock.py` also supports `--embedding-backend fake|local` and `--allow-unsafe-llm-traffic`

### Headless Mode (CI/CD)

For automated testing without the web UI:

```bash
poetry run python scripts/perf_tests/run_perf_light.py \
  --users 10 \
  --spawn-rate 2 \
  --duration 2m \
  --html reports/performance/ci_report.html \
  --csv
```

### Custom Configuration

Override target host:

```bash
poetry run python scripts/perf_tests/run_perf_medium.py --host http://staging.yourapi.com
```

Run with a custom report suffix so repeated runs do not overwrite each other:

```bash
poetry run python scripts/perf_tests/run_perf_medium.py \
  --host http://production.yourapi.com \
  --name-suffix baseline \
  --csv
```

### Advanced: Raw Locust Commands

Use raw Locust commands only when you need debugging flags or an execution shape not covered by the wrappers. The wrappers are thin orchestration around the same `tests/performance/locustfile.py` entrypoint.

```bash
# Equivalent manual light profile run
LOAD_PROFILE=light poetry run locust \
  -f tests/performance/locustfile.py \
  --host http://localhost:8100 \
  --headless \
  --users 10 \
  --spawn-rate 1 \
  --run-time 1m \
  --html reports/performance/light_load_report.html

# Equivalent manual llm_mock run with local embeddings
LLM_EMULATION_ENABLED=true EMBEDDING_BACKEND=local LOAD_PROFILE=llm_mock \
  poetry run locust -f tests/performance/locustfile.py \
  --host http://localhost:8100 \
  --headless \
  --users 25 \
  --spawn-rate 2.5 \
  --run-time 5m \
  --html reports/performance/llm_mock_local_embeddings_load_report.html
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
- Login and token acquisition with per-user session state
- Profile retrieval and updates under a reusable authenticated session
- Password changes
- Explicit token refresh operations
- Logout and re-login within the lifetime of the same simulated user
- Revoked refresh-token validation after logout
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
- `POST /api/v1/embeddings/generate` - Generate embeddings and fail if the backend is neither `fake` nor `local`
- `GET /api/v1/vector/health` - Check vector readiness
- `POST /api/v1/vector/ingest` with `async_mode=true` - Push seed/update ingestion through Celery-backed task execution
- `GET /api/v1/tasks/{task_id}/status` and `GET /api/v1/tasks/{task_id}/result` - Poll task completion for vector ingestion
- `POST /api/v1/vector/query` and `POST /api/v1/vector/list` - Exercise search and listing
- `GET /api/v1/vector/collections/{collection}/stats` - Refresh collection statistics

The shared session helper now retries one time after a `401` by refreshing or rebuilding the session, so long-running agent/vector/embedding workloads do not fail only because an access token expired mid-run.

## Endpoint and Mock Readiness

| Area                           | Locust status                   | No-cost readiness                                                                                                   |
| ------------------------------ | ------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Health/auth/tasks              | Covered by existing profiles    | Safe; no paid providers involved                                                                                    |
| Agent instances and agent runs | Covered by `llm_mock`           | Safe when `LLM_EMULATION_ENABLED=true` and tool exposure uses `strict_test` or emulated tools                       |
| Embeddings                     | Covered by `llm_mock`           | Safe with `EMBEDDING_BACKEND=fake` or `local`; the scenario fails responses that report another backend             |
| Vector store                   | Covered by `llm_mock`           | Safe with fake or local embeddings; `VECTOR_BACKEND=memory` is recommended for isolated load runs                   |
| Batch jobs                     | Not included in default traffic | Full batch lifecycle still needs a fake batch provider before it is safe to load-test provider submission/execution |

## Excluded or Deferred Endpoints

The following endpoints are excluded or deferred to avoid costs and complexity:

- **Provider-backed LLM traffic**: real model/provider calls are not included in default profiles; use `llm_mock` only with emulation enabled
- **Full batch lifecycle**: create/submit/provider execution is deferred until a fake batch provider is implemented
- **Task cancellation**: `DELETE /api/v1/tasks/{task_id}`
- **Email delivery**: Password reset email sending

## Output and Reports

### HTML Reports

Wrapper scripts generate an HTML report by default in a dated directory under `reports/performance/YYYY/MM/DD/`.

- First run of a given profile on that day: `{profile_name}_load_report.html`
- Next run of the same profile on that day: `{profile_name}_load_report_02.html`
- Further runs continue rotating as `_03`, `_04`, and so on

Use `--name-suffix` when you want separate artifact families inside the same dated directory, for example `llm_mock_local-embeddings_load_report.html`. Pass `--no-html` when you intentionally want to skip HTML generation.

### CSV Data

For detailed analysis, add `--csv` to any wrapper command or call raw Locust directly. CSV files go to the same dated directory as the HTML report and reuse the same rotation suffix for that run.

```bash
poetry run python scripts/perf_tests/run_perf_light.py \
  --host http://localhost:8000 \
  --users 10 \
  --spawn-rate 2 \
  --duration 2m \
  --csv \
  --name-suffix detailed
```

This creates:

- `light_detailed_results_stats.csv` - Request statistics
- `light_detailed_results_stats_history.csv` - Timeline data
- `light_detailed_results_failures.csv` - Failure details

## Performance Thresholds

The test suite includes performance thresholds for regression detection:

```python
PERFORMANCE_THRESHOLDS = {
    "health_p95_ms": 100,      # Health endpoints should be fast
    "auth_p95_ms": 500,        # Auth operations can be slower
    "tasks_p95_ms": 200,       # Task monitoring should be responsive
    "agent_run_p95_ms": 2500,  # Emulated agent runs exercise LangChain setup
    "embedding_p95_ms": 2000,  # No-cost embeddings can include Celery + local inference
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

- Run `poetry run alembic upgrade head` against the target database
- Check if user registration is working
- Verify JWT configuration in .env
- Ensure Redis is running; refresh-session revocation and rotation rely on it

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
          poetry run python scripts/perf_tests/run_perf_light.py \
            --host http://localhost:8000 \
            --users 2 \
            --spawn-rate 1 \
            --duration 30s \
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

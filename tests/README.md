## Test Suite Documentation

### Test Structure

The test suite is organized as follows:

- `tests/conftest.py` - Test configuration and shared fixtures (async engine/session, HTTP client)
- `tests/integration/` - Integration tests for API endpoints, DB operations, and LLM tasks
- `tests/unit/` - Fast, isolated unit tests (no network/db) covering core config, security, Redis client, database connection helpers, and services

Unit test layout:

- `tests/unit/core/` — config parsing/validation, security (hashing/JWT), logging config, Redis client
- `tests/unit/database/` — engine/session creation, event listeners, health and masking helpers
- `tests/unit/services/` — AuthService (CRUD/auth flows with mocks), TaskService (Celery orchestration with mocks), RefreshSessionStore, LLMService (mocked chains/models)

### Running Tests

To run all tests:

```bash
poetry run pytest
```

More specific test commands can be obtained by running:

```bash
poetry run pytest --help
```

Common examples:

```bash
# Only integration tests (uses the registered 'integration' marker)
poetry run pytest -m integration

# Only unit tests
poetry run pytest tests/unit

# Only core unit tests (subset)
poetry run pytest tests/unit/core

# Only LLM task tests (mocked chains)
poetry run pytest tests/integration/test_llm_tasks.py

# Real-chain LLM tests (opt-in). Requires valid API keys and testing models in llm_config.yaml
RUN_LLM_REAL_TESTS=1 poetry run pytest tests/integration/test_llm_tasks_real.py -q -m integration
```

### Test Fixtures

The project provides several useful fixtures:

#### Database Fixtures

- **`test_settings`** - Provides application settings configured for testing (sets `ENVIRONMENT=testing`)
- **`async_engine`** - Creates a temporary database engine with proper cleanup
- **`async_session_with_engine`** - Provides a database session with automatic table creation and cleanup
- **`async_test_client`** - HTTP test client with database dependency injection

#### Automatic Table Creation

Fixtures create an async engine and sessions, but do not automatically create tables. Create/drop tables explicitly in tests when needed (see examples below).

### Example Test Usage

```python
import pytest
from sqlalchemy import text

@pytest.mark.asyncio
async def test_database_operations(async_session_with_engine):
    """Test database operations with automatic table creation."""
    session, engine = async_session_with_engine

    # Tables are NOT auto-created; create them if your test requires models
    result = await session.execute(text("SELECT 1"))
    assert result.scalar() == 1

@pytest.mark.asyncio
async def test_api_endpoint(async_test_client):
    """Test API endpoints with test client."""
    response = await async_test_client.get("/api/v1/health/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

### Test Database Configuration

Tests use the same database configuration as the main application but with:

- Environment set to "testing"
- Proper connection cleanup to prevent resource leaks
- Isolated database sessions for each test
- Explicit table create/drop within tests that need schema

Notes:

- By default, tests run against SQLite (aiosqlite) unless you override DATABASE_URL/SERVICE in your env/.env.
- If you point tests to PostgreSQL/MySQL, ensure the server is reachable and credentials match your env.

### LLM Integration Tests

There are two layers of LLM task tests:

- Mocked chains: `tests/integration/test_llm_tasks.py` — fast, no external calls. Chain factories are monkeypatched.
- Real chains: `tests/integration/test_llm_tasks_real.py` — use authentic chains and models from `llm_config.yaml` under `tasks.<task>.testing`.

Notes for real-chain tests:

- Opt-in via environment variable: set `RUN_LLM_REAL_TESTS=1` to enable.
- Tests are skipped when the selected model isn't available (e.g., missing API keys).
- In case of empty provider responses (e.g., network restrictions or invalid keys), tests are marked as xfail to avoid false negatives.
- Ensure provider API keys are configured via environment variables described in `llm_config.yaml` providers section.

### Markers

The `integration` marker is registered in `pytest.ini` and is used across integration tests.

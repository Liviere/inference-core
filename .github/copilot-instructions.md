# FastAPI Backend Template

FastAPI backend template with Celery background tasks, JWT authentication, LLM integration via LangChain, and multi-database support (SQLite/PostgreSQL/MySQL). Built with Poetry for dependency management and includes comprehensive Docker deployment options.

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Acquiring up-to-date third‑party library context – MCP Context7

To minimize hallucinations and apply the latest recommended best practices, EVERY task that depends on behavior / API / configuration of an external library SHOULD begin by fetching fresh docs via the MCP "context7" mechanism.

#### When to use it

Use MCP Context7 whenever the task involves (non‑exhaustive list):

- FastAPI / Starlette (routing, dependency injection, background tasks, lifespan, streaming)
- Pydantic / pydantic-settings (validation, BaseModel, ConfigDict)
- SQLAlchemy (models, async engine, session patterns, migrations)
- Celery / Redis (broker/result backend config, retry, acks_late, task routing)
- LangChain / langchain-openai / langchain-community (chains, tools, structured output, rate limiting, async patterns)
- Sentry SDK (performance, traces_sample_rate, FastAPI / Celery integrations)
- Uvicorn (server params, workers vs reload)
- Locust (performance testing, task definitions)
- JWT / python-jose / security (algorithms, rotating secrets)
- Password hashing: passlib, bcrypt (rounds / gensalt configuration)
- Any newly introduced or less‑familiar library in the project

If you are NOT 100% sure about the current API or there may have been breaking changes in the pinned version range in `pyproject.toml` – fetch context first.

#### Minimal Context7 workflow

1. Identify the library + topic (e.g. "fastapi dependency overrides", "sqlalchemy async session best practices", "celery retry exponential backoff").
2. Call resolve-library-id (mcp_context7_resolve-library-id) with the library name.
3. Fetch focused docs (mcp_context7_get-library-docs) with a `topic`; raise token limit if you need broader scope (e.g. 12000).
4. Summarize key points (specific classes / functions / params), list pitfalls, THEN implement.
5. Add brief inline source references (e.g. `# Source: FastAPI docs – dependency overrides, 2025-08 snapshot`).

#### Agent response convention

- If context was fetched: include a "Context7 Sources" section listing library → topics.
- If the user explicitly wants a quick trivial answer (e.g. a simple BaseModel), you MAY skip fetching, BUT mention that extended context can be pulled on request.
- If intuition conflicts with docs: prefer docs and flag the discrepancy.

#### Example (condensed)

Task: add an endpoint that streams LLM output.

1. resolve-library-id: fastapi, langchain-openai
2. get-library-docs topics: "streaming responses", "async openai streaming"
3. Summary: FastAPI – use `StreamingResponse`; LangChain – attach token streaming callback handler.
4. Implement + reference sources in comments.

> In short: WHEN IN DOUBT → FETCH Context7 FIRST, THEN CODE.

---

### Bootstrap, Build, and Test the Repository

**Prerequisites:**

- Python 3.12+ (verified working with Python 3.12.3)
- Poetry for dependency management (install via pip if not available)

**Setup Commands (validated and timed):**

```bash
# Install Poetry if not available
pip install poetry

# Copy configuration files
cp .env.example .env
cp llm_config.example.yaml llm_config.yaml

# Install dependencies - takes 36 seconds. NEVER CANCEL. Set timeout to 120+ seconds.
poetry install

# Run tests - takes 16 seconds. NEVER CANCEL. Set timeout to 60+ seconds.
poetry run pytest
```

**Expected Test Results:**

- 206 tests total: 204 pass, 2 skip (LLM real tests)
- No failures in the default local setup (SQLite or running Postgres). Skips are for optional real LLM tests.
- Core functionality (API, database, authentication, Celery orchestration) works correctly

### Run the Application

**Development Mode:**

```bash
# Start development server with auto-reload
poetry run fastapi dev --host 0.0.0.0 --port 8000
```

**Production Mode:**

```bash
# Start production server
poetry run fastapi run --host 0.0.0.0 --port 8000
```

**Application URLs:**

- API: http://localhost:8000
- Health Check: http://localhost:8000/api/v1/health/
- API Documentation: http://localhost:8000/docs (development mode only)

### Celery Background Tasks

**Start Redis (required for Celery):**

```bash
# Using Docker (recommended)
docker run -d --name redis-test -p 6379:6379 redis:7-alpine
```

**Start Celery Worker:**

```bash
# In a separate terminal
poetry run celery -A app.celery.celery_main:celery_app worker --loglevel=info --queues=default
```

**Start Flower Monitoring (optional):**

```bash
# Monitor Celery tasks at http://localhost:5555
poetry run celery -A app.celery.celery_main:celery_app flower --port=5555
```

## Validation

### Always Test Core Functionality After Changes

**Health Check Validation:**

```bash
curl http://localhost:8000/api/v1/health/
# Should return healthy status for database, application, and tasks (if Redis running)
```

**Authentication Flow Validation:**

```bash
# Register user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "email": "test@example.com", "password": "SecurePass123!"}'

# Login and get tokens
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "SecurePass123!"}'
```

**ALWAYS manually validate any new code by:**

1. Starting the application (`poetry run fastapi dev`)
2. Testing API endpoints with curl or browser
3. Verifying authentication flows work
4. If modifying Celery tasks, ensure Redis is running and test task execution
5. Running the test suite to catch regressions

### Test Suite Quick Reference

- Run everything:
  - `poetry run pytest`
- Only unit tests (fast, isolated):
  - `poetry run pytest tests/unit`
- Only integration tests:
  - `poetry run pytest -m integration`
- Real-chain LLM tests (opt-in; require API keys):
  - `RUN_LLM_REAL_TESTS=1 poetry run pytest tests/integration/test_llm_tasks_real.py -q -m integration`

### Unit Tests Coverage Snapshot

- Core: settings parsing/validation, JWT/token helpers, logging, Redis client
- Database: async engine/session creation, event listeners, health info masking
- Services: AuthService, TaskService (Celery interactions mocked), RefreshSessionStore, LLMService (mocked chains/models)

### Runtime Services for Integration Tests

- Database: SQLite works out of the box. If using PostgreSQL/MySQL, ensure the server is running and reachable (defaults: Postgres on 127.0.0.1:5432).
- Redis: Required for Celery workers; integration tests that patch Celery do not require a running worker/broker unless explicitly testing them.

### Docker Deployment (Known Issues)

**Current Status:** Docker build configuration has timeout and YAML formatting issues.

**Working Alternative:** Use local Poetry-based development as documented above.

**If Docker is needed:** Fix the following known issues in docker-compose files:

- Build timeouts during Poetry install step
- YAML indentation errors in docker-compose.base.yml (lines 27-28, 51-52, 75)
- Path mapping issues in docker/docker-compose.sqlite.yml

## Common Tasks

### Repository Structure

```
.
├── README.md                 # Main documentation
├── pyproject.toml           # Poetry dependencies and config
├── poetry.lock              # Locked dependency versions
├── pytest.ini              # Test configuration
├── .env.example             # Environment variables template
├── llm_config.example.yaml  # LLM configuration template
├── run.py                   # Alternative startup script
├── app/                     # Main application code
│   ├── main.py             # FastAPI app factory
│   ├── api/v1/routes/      # API endpoints
│   ├── celery/             # Background task definitions
│   ├── core/               # Core configuration
│   ├── database/           # Database models and connections
│   ├── llm/                # LLM integration
│   ├── schemas/            # Pydantic schemas
│   └── services/           # Business logic services
├── tests/                   # Test suite
│   ├── integration/        # Integration tests
│   └── unit/               # Unit tests (fast, isolated)
├── logs/                   # Application logs (local runs)
├── scripts/                # Helper scripts (if any)
└── docker/                 # Docker configuration (has issues)
    ├── Dockerfile
    ├── docker-compose.sqlite.yml
    ├── docker-compose.mysql.yml
    └── docker-compose.postgres.yml
```

### Key Configuration Files

**Environment Variables (.env):**

- Database: SQLite by default, supports PostgreSQL/MySQL
- Redis: Required for Celery tasks (localhost:6379)
- JWT: Configure SECRET_KEY for production
- LLM: Set API keys for OpenAI or other providers

**LLM Configuration (llm_config.yaml):**

- Providers: OpenAI, custom OpenAI-compatible endpoints
- Models: Configurable per task (explain, conversation)
- Testing: Uses different models for testing vs production

### API Endpoints Summary

**Health & Info:**

- `GET /` - Application info
- `GET /api/v1/health/` - Overall health check
- `GET /api/v1/health/database` - Database health
- `GET /api/v1/health/ping` - Simple ping

**Authentication (JWT):**

- `POST /api/v1/auth/register` - Create account
- `POST /api/v1/auth/login` - Get access/refresh tokens
- `POST /api/v1/auth/refresh` - Rotate tokens
- `POST /api/v1/auth/logout` - Revoke refresh token
- `GET /api/v1/auth/me` - Current user profile
- `PUT /api/v1/auth/me` - Update profile

**Tasks:**

- `GET /api/v1/tasks/health` - Celery worker health
- `GET /api/v1/tasks/{task_id}/status` - Task status
- `GET /api/v1/tasks/{task_id}/result` - Task result

**LLM (if configured):**

- LLM endpoints for explain and conversation tasks via Celery

## Critical Reminders

- **NEVER CANCEL** Poetry install (36 seconds) or tests (16 seconds) - they complete quickly
- **ALWAYS** ensure Redis is running before starting Celery workers
- **ALWAYS** copy .env.example to .env and llm_config.example.yaml to llm_config.yaml before first run
- **ALWAYS** test authentication flow after auth-related changes
- **DO NOT** rely on Docker deployment until configuration issues are resolved
- **ALWAYS** run `poetry run pytest` before committing changes to catch regressions
- Application works perfectly in local development mode with Poetry

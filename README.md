# Inference Core

A production-ready FastAPI backend template with Celery-powered background tasks and first-class LLM integration using LangChain.

## Requirements

- Python 3.12 or higher
- [Poetry](https://python-poetry.org/) for dependency management
- [Sentry account](https://sentry.io/) for error monitoring (optional, for production)

## Getting Started

This project uses [Poetry](https://python-poetry.org/) for dependency management.
Make sure you have it installed before proceeding.

### Install dependencies

```bash
poetry install
```

### Development

This template now exposes only an application factory (`create_application`) in `inference_core/main_factory.py`.
You can run the API using Uvicorn (recommended) with the `--factory` flag so the callable is invoked
to create a fresh FastAPI instance.

Development (auto-reload):

```bash
poetry run uvicorn inference_core.main_factory:create_application --factory --reload --host 0.0.0.0 --port 8000
```

### Production Run

```bash
poetry run uvicorn inference_core.main_factory:create_application --factory --host 0.0.0.0 --port 8000
```

If you still prefer a single-file entrypoint locally, you can create (and keep untracked)
a small `inference_core/main.py` like:

```python
from inference_core.main_factory import create_application

app = create_application()
```

Because `main.py` is now in `.gitignore`, it will not be committed—ideal for local experiments.

### Using as a Dependency (Embedding the Template)

If another project depends on this template package, it can create an application instance:

```python
from inference_core.main_factory import create_application

api = create_application()
```

Then mount it (for example) inside a larger ASGI app or serve directly with Uvicorn as shown above.

## Docker Deployment

This application includes Docker support for containerized deployment with SQLite, MySQL, and PostgreSQL database options. The Docker configuration is organized in the `docker/` directory with separate compose files for each database type.

For detailed instructions on how to deploy the application using Docker, see [docker/README.md](docker/README.md).

## Asynchronous tasks with Celery

This project includes Celery for background task processing with Redis as the default broker and result backend.

For detailed instructions on how to set up and use Celery, see [inference_core/celery/README.md](inference_core/celery/README.md).

## Testing

The project includes a test suite with fixtures for database testing and API integration testing.

For comprehensive testing documentation, environment setup, and troubleshooting, see [tests/README.md](tests/README.md).

### Isolated Test Docker Environment

For running tests in an isolated, stateless Docker environment that doesn't conflict with development setups:

```bash
# Quick start with PostgreSQL test environment
cp docker/tests/.env.test.example .env.test
docker compose -f docker/tests/docker-compose.test.postgres.yml --env-file .env.test up -d

# Verify health
curl http://localhost:8100/api/v1/health/ping

# Cleanup when done
docker compose -f docker/tests/docker-compose.test.postgres.yml down -v
```

Alternative test environments are available for SQLite and MySQL. For detailed instructions, configuration options, and CI integration, see [docs/testing-docker.md](docs/testing-docker.md).

## API Endpoints (v1)

This project also includes an API dedicated to working with LLM models. For details on LLM endpoints, configuration, and usage, see [inference_core/llm/README.md](inference_core/llm/README.md).

### Health Check

- `GET /api/v1/health/` - Overall application health check
- `GET /api/v1/health/database` - Check database connection health
- `GET /api/v1/health/ping` - Simple ping endpoint for basic health checking

### Authentication (JWT)

Built-in authentication supports JWT access tokens and stateful refresh tokens stored in Redis for rotation and logout.

- Auth flow

  - `POST /api/v1/auth/register` — create account
  - `POST /api/v1/auth/login` — returns `access_token` and `refresh_token`
  - `POST /api/v1/auth/refresh` — exchange refresh for new tokens (rotates refresh)
  - `POST /api/v1/auth/logout` — revoke provided refresh token (best-effort)

- User endpoints
  - `GET /api/v1/auth/me` — current profile (requires access token)
  - `PUT /api/v1/auth/me` — update profile (requires access token)
  - `POST /api/v1/auth/change-password` — change password (requires access token)
  - `POST /api/v1/auth/forgot-password` — request reset (always returns success)
  - `POST /api/v1/auth/reset-password` — set new password with reset token

Notes

- Access tokens are short-lived and must include type `access`.
- Refresh tokens are stored/validated in Redis and rotated on `/auth/refresh`.
- On `/auth/logout`, the provided refresh token is revoked in Redis when available.

## Configuration

The application uses environment variables for configuration. You can set them directly or create a `.env` file in the project root.

To get started quickly, copy the example configuration file:

```bash
cp .env.example .env
```

Then edit the `.env` file with your specific settings.

## Database Configuration

The application can be configured to use different databases through the `DATABASE_URL` environment variable.
The application automatically uses the appropriate asynchronous database driver (`asyncpg` for PostgreSQL, `aiomysql` for MySQL, `aiosqlite` for SQLite) based on the `DATABASE_URL`.

### SQLite (Default)

For local development, the application uses SQLite by default. The database will be created in a file named `inference_core.db` in the project root.

### PostgreSQL (Recommended for Production)

For production environments, PostgreSQL is recommended.

### MySQL

You can also use MySQL.

## Monitoring with Sentry

This application includes [Sentry](https://sentry.io/) integration for error monitoring and performance tracking. Sentry helps you:

- **Error Tracking**: Automatically capture and report unhandled exceptions
- **Performance Monitoring**: Track request performance and identify bottlenecks
- **Release Tracking**: Monitor deployments and track issues across releases
- **Session Tracking**: Monitor user sessions and application health

### Sentry Setup

1. Create a Sentry account at [sentry.io](https://sentry.io/)
2. Create a new project for your application
3. Copy the DSN from your Sentry project settings
4. Set the `SENTRY_DSN` environment variable in your `.env` file

The Sentry SDK is automatically installed with the project dependencies. Sentry monitoring is automatically enabled when:

- The environment is set to `production`
- A valid `SENTRY_DSN` is provided

For production environments, consider adjusting the sample rates to reduce overhead:

- `SENTRY_TRACES_SAMPLE_RATE=0.1` (10% of transactions)
- `SENTRY_PROFILES_SAMPLE_RATE=0.1` (10% of transactions)

### Environment Variables

| Variable                      | Description                                                      | Default                                                      | Used in             |
| ----------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------ | ------------------- |
| `APP_NAME`                    | Application name                                                 | "Inference Core API"                                         | API, Docker         |
| `APP_TITLE`                   | Application title                                                | "Inference Core API"                                         | API                 |
| `APP_DESCRIPTION`             | Application description                                          | "A production-ready Inference Core API with LLM integration" | API                 |
| `APP_VERSION`                 | Application version                                              | "0.1.0"                                                      | API                 |
| `ENVIRONMENT`                 | Application environment (development/staging/production/testing) | "development"                                                | API, Docker         |
| `DEBUG`                       | Debug mode                                                       | `True`                                                       | API                 |
| `HOST`                        | Server host                                                      | "0.0.0.0"                                                    | API, Docker         |
| `PORT`                        | Server port                                                      | 8000                                                         | API, Docker         |
| `CORS_METHODS`                | Allowed HTTP methods (comma-separated or \*)                     | \*                                                           | API                 |
| `CORS_ORIGINS`                | Allowed origins (comma-separated or \*)                          | \*                                                           | API                 |
| `CORS_HEADERS`                | Allowed headers (comma-separated or \*)                          | \*                                                           | API                 |
| `SENTRY_DSN`                  | Sentry Data Source Name for error monitoring                     | None                                                         | API                 |
| `SENTRY_TRACES_SAMPLE_RATE`   | Sentry performance monitoring sample rate (0.0 to 1.0)           | 1.0                                                          | API                 |
| `SENTRY_PROFILES_SAMPLE_RATE` | Sentry profiling sample rate (0.0 to 1.0)                        | 1.0                                                          | API                 |
| `DATABASE_URL`                | Database connection URL                                          | `sqlite+aiosqlite:///./inference_core.db`                    | API                 |
| `DATABASE_ECHO`               | Echo SQL queries (development only)                              | `False`                                                      | API                 |
| `DATABASE_POOL_SIZE`          | Database connection pool size                                    | `20`                                                         | API                 |
| `DATABASE_MAX_OVERFLOW`       | Maximum database connection overflow                             | `30`                                                         | API                 |
| `DATABASE_POOL_TIMEOUT`       | Pool connection timeout in seconds                               | `30`                                                         | API                 |
| `DATABASE_POOL_RECYCLE`       | Connection recycle time in seconds                               | `3600`                                                       | API                 |
| `DATABASE_MYSQL_CHARSET`      | MySQL character set                                              | `utf8mb4`                                                    | API, Docker         |
| `DATABASE_MYSQL_COLLATION`    | MySQL collation                                                  | `utf8mb4_unicode_ci`                                         | Docker              |
| `DATABASE_NAME`               | Database name                                                    | `app_db`                                                     | Docker              |
| `DATABASE_USER`               | Database user                                                    | `db_user`                                                    | Docker              |
| `DATABASE_PASSWORD`           | Database password                                                | `your_password`                                              | Docker              |
| `DATABASE_ROOT_PASSWORD`      | Database root password (MySQL only)                              | `your_root_password`                                         | Docker              |
| `DATABASE_PORT`               | Database port                                                    | `3306` (MySQL) / `5432` (PostgreSQL)                         | Docker              |
| `DATABASE_HOST`               | Database host                                                    | `localhost` (or service name in Docker)                      | API, Docker         |
| `DATABASE_SERVICE`            | Database backend/driver (async)                                  | `sqlite+aiosqlite`                                           | API, Docker         |
| `SECRET_KEY`                  | Secret used to sign JWT tokens                                   | `change-me-in-production`                                    | Auth                |
| `ALGORITHM`                   | JWT signing algorithm                                            | `HS256`                                                      | Auth                |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Access token lifetime (minutes)                                  | `30`                                                         | Auth                |
| `REFRESH_TOKEN_EXPIRE_DAYS`   | Refresh token lifetime (days)                                    | `7`                                                          | Auth                |
| `REDIS_URL`                   | Redis URL for app sessions/locks (refresh sessions)              | `redis://localhost:6379/10`                                  | API/Auth            |
| `REDIS_REFRESH_PREFIX`        | Key prefix for refresh sessions                                  | `auth:refresh:`                                              | Auth                |
| `CELERY_BROKER_URL`           | Celery broker URL                                                | `redis://localhost:6379/0`                                   | API, Celery, Docker |
| `CELERY_RESULT_BACKEND`       | Celery result backend URL                                        | `redis://localhost:6379/1`                                   | Celery, Docker      |
| `DEBUG_CELERY`                | Enable debugpy for Celery worker (1 to enable)                   | `0`                                                          | Celery, Docker      |
| `REDIS_PORT`                  | Redis port (used for both host and container)                    | `6379`                                                       | Docker              |
| `FLOWER_PORT`                 | Flower port (used for both host and container)                   | `5555`                                                       | Docker              |
| `OPENAI_API_KEY`              | API key for OpenAI provider                                      | None                                                         | LLM                 |
| `GOOGLE_API_KEY`              | API key for Google Gemini models                                 | None                                                         | LLM                 |
| `ANTHROPIC_API_KEY`           | API key for Anthropic Claude models                              | None                                                         | LLM                 |
| `LLM_EXPLAIN_MODEL`           | Override model for the 'explain' task                            | None                                                         | LLM                 |
| `LLM_CONVERSATION_MODEL`      | Override model for the 'conversation' task                       | None                                                         | LLM                 |
| `LLM_ENABLE_CACHING`          | Enable in-process LLM response caching (fallback mode)           | `true`                                                       | LLM                 |
| `LLM_CACHE_TTL`               | Cache TTL in seconds (fallback mode)                             | `3600`                                                       | LLM                 |
| `LLM_MAX_CONCURRENT`          | Max concurrent LLM requests (fallback mode)                      | `5`                                                          | LLM                 |
| `LLM_ENABLE_MONITORING`       | Enable basic LLM monitoring hooks (fallback mode)                | `true`                                                       | LLM                 |
| `RUN_LLM_REAL_TESTS`          | Opt-in to run real-chain tests hitting providers in CI/local     | `0`                                                          | Tests               |

## Features

### Error Monitoring & Performance Tracking

The application includes comprehensive monitoring capabilities through Sentry integration:

- **Automatic Error Capture**: Unhandled exceptions are automatically sent to Sentry
- **Performance Monitoring**: HTTP requests, database queries, and other operations are tracked
- **Release Tracking**: Deployments are tracked with version information
- **Environment Separation**: Different environments (dev/staging/prod) are tracked separately
- **Custom Context**: User information, request data, and custom tags can be attached to events

### Example: Manual Error Reporting

```python
import sentry_sdk

# Capture a custom message
sentry_sdk.capture_message("Something important happened", level="info")

# Capture an exception with additional context
try:
    risky_operation()
except Exception as e:
    sentry_sdk.set_tag("operation", "risky")
    sentry_sdk.set_extra("user_id", user_id)
    sentry_sdk.capture_exception(e)
```

## Logging

The application is configured to log information to both the console and a rotating file.

- **Console Logging**: Provides real-time output during development.
- **File Logging**: Logs are saved in JSON format to the `logs/inference_core.log` file. The log file is rotated daily, and backups are kept for 30 days.

This setup is handled by the configuration in `inference_core/core/logging_config.py`.

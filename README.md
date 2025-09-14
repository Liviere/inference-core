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

## Email Delivery Service

The application includes a production-grade email delivery service built on Python's `smtplib` with Celery for asynchronous processing. It supports:

- **Multiple SMTP providers** with fallback support (Gmail, Office 365, Mailgun, custom)
- **Secure transport** (SSL/STARTTLS) with hostname verification
- **HTML and text email** with Jinja2 templating
- **File attachments** with configurable size limits
- **Celery task queuing** with retry/backoff on transient failures
- **Environment-based secrets** (no passwords in configuration files)

### Quick Setup

1. **Copy email configuration template:**

   ```bash
   cp email_config.example.yaml email_config.yaml
   ```

2. **Configure SMTP credentials in `.env`:**

   ```bash
   # Primary SMTP (e.g., Gmail)
   SMTP_PRIMARY_USERNAME=your-email@gmail.com
   SMTP_PRIMARY_PASSWORD=your-app-password

   # Public URL for email links
   APP_PUBLIC_URL=https://yourdomain.com
   ```

3. **Start mail queue worker:**
   ```bash
   # In a separate terminal
   poetry run celery -A inference_core.celery.celery_main:celery_app worker --loglevel=info --queues=mail
   ```

### Email Configuration

Edit `email_config.yaml` to configure SMTP hosts. The configuration supports:

- **Multi-host setup** with primary/backup providers
- **Environment variable resolution** for usernames: `${SMTP_USERNAME}`
- **Secure password handling** via environment variables only
- **Per-host settings** (timeouts, rate limits, attachment limits)

Example minimal configuration:

```yaml
email:
  default_host: primary
  hosts:
    primary:
      host: smtp.gmail.com
      port: 465
      use_ssl: true
      username: ${SMTP_PRIMARY_USERNAME}
      password_env: SMTP_PRIMARY_PASSWORD
      from_email: no-reply@example.com
      from_name: Your App Name
```

### Usage Examples

**Programmatic email sending:**

```python
from inference_core.services.email_service import get_email_service

email_service = get_email_service()
if email_service:
    message_id = email_service.send_email(
        to="user@example.com",
        subject="Welcome!",
        text="Welcome to our service.",
        html="<p>Welcome to our service.</p>"
    )
```

**Asynchronous via Celery:**

```python
from inference_core.celery.tasks.email_tasks import send_email_async

task = send_email_async(
    to="user@example.com",
    subject="Password Reset",
    text="Click here to reset your password: ...",
    html="<p>Click <a href='...'>here</a> to reset your password.</p>"
)
```

**Password reset emails** are automatically sent when users request password resets via the `/auth/reset-password` endpoint.

### Monitoring

- **Structured logging** with message IDs, recipient counts, and timing
- **Celery task monitoring** via Flower: http://localhost:5555
- **Error tracking** with automatic retry on transient failures
- **Sentry integration** for error reporting and performance monitoring

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

- Email verification endpoints
  - `POST /api/v1/auth/verify-email/request` — request verification email (always returns success)
  - `POST /api/v1/auth/verify-email` — verify email with token

#### Email Verification Configuration

User activation and email verification behavior is configurable via environment variables:

| Variable                                    | Description                                     | Default | Effect                                                          |
| ------------------------------------------- | ----------------------------------------------- | ------- | --------------------------------------------------------------- |
| `AUTH_REGISTER_DEFAULT_ACTIVE`              | Whether new users are active by default         | `true`  | If `false`, users are inactive until manually activated         |
| `AUTH_SEND_VERIFICATION_EMAIL_ON_REGISTER`  | Send verification email on registration         | `false` | If `true`, sends email with verification link                   |
| `AUTH_LOGIN_REQUIRE_ACTIVE`                 | Require active user for login                   | `true`  | If `false`, inactive users can login                            |
| `AUTH_LOGIN_REQUIRE_VERIFIED`               | Require verified email for login                | `false` | If `true`, unverified users cannot login                        |
| 'AUTH_EMAIL_VERIFICATION_MAKES_ACTIVE'      | Whether verifying email also activates the user | `true`  | If `true`, verifying email sets `is_active = true`              |
| `AUTH_EMAIL_VERIFICATION_TOKEN_TTL_MINUTES` | Verification token lifetime                     | `60`    | Token expires after this many minutes                           |
| `AUTH_EMAIL_VERIFICATION_URL_BASE`          | Base URL for verification links                 | `null`  | If set, creates frontend links; otherwise uses backend endpoint |

**Example verification flow:**

1. User registers with `AUTH_SEND_VERIFICATION_EMAIL_ON_REGISTER=true`
2. System sends email with verification link/token
3. User clicks link or posts token to `/auth/verify-email`
4. User's `is_verified` status becomes `true`
5. User can login if `AUTH_LOGIN_REQUIRE_VERIFIED=true`

Notes

- Access tokens are short-lived and must include type `access`.
- Refresh tokens are stored/validated in Redis and rotated on `/auth/refresh`.
- On `/auth/logout`, the provided refresh token is revoked in Redis when available.

### LLM API Access Control

All LLM endpoints under `/api/v1/llm/*` (including batch processing at `/api/v1/llm/batch/*`) support configurable access control via the `LLM_API_ACCESS_MODE` environment variable.

**Access Modes:**

- **`superuser`** (default): Only superusers can access LLM endpoints. Returns 403 for non-superuser or unauthenticated requests.
- **`user`**: Any authenticated active user can access LLM endpoints. Returns 401 for unauthenticated requests.
- **`public`**: No authentication required. All LLM endpoints are publicly accessible.

**Security Considerations:**

⚠️ **Production Warning**: Public mode exposes LLM endpoints without authentication, which can lead to:
- Unauthorized usage and increased costs
- Potential data exposure through conversation history
- Resource abuse and service degradation

**Recommended settings:**
- **Production**: `LLM_API_ACCESS_MODE=superuser` (default)
- **Internal staging**: `LLM_API_ACCESS_MODE=user`
- **Local development/demos**: `LLM_API_ACCESS_MODE=public` (with compensating controls)

**Examples:**

```bash
# Production (default)
LLM_API_ACCESS_MODE=superuser

# Internal staging
LLM_API_ACCESS_MODE=user

# Local development only
LLM_API_ACCESS_MODE=public
```

**Protected Endpoints:**
- All `/api/v1/llm/*` endpoints (explain, conversation, models, health, stats, streaming)
- All `/api/v1/llm/batch/*` endpoints (create, get, list, cancel)

**Unaffected Endpoints:**
- General health check (`/api/v1/health/`)
- Authentication endpoints (`/api/v1/auth/*`)
- Root endpoint (`/`)

Changes take effect immediately upon application restart.

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

| Variable                                    | Description                                                      | Default                                                      | Used in             |
| ------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------ | ------------------- |
| `APP_NAME`                                  | Application name                                                 | "Inference Core API"                                         | API, Docker         |
| `APP_TITLE`                                 | Application title                                                | "Inference Core API"                                         | API                 |
| `APP_DESCRIPTION`                           | Application description                                          | "A production-ready Inference Core API with LLM integration" | API                 |
| `APP_VERSION`                               | Application version                                              | "0.1.0"                                                      | API                 |
| `ENVIRONMENT`                               | Application environment (development/staging/production/testing) | "development"                                                | API, Docker         |
| `DEBUG`                                     | Debug mode                                                       | `True`                                                       | API                 |
| `HOST`                                      | Server host                                                      | "0.0.0.0"                                                    | API, Docker         |
| `PORT`                                      | Server port                                                      | 8000                                                         | API, Docker         |
| `CORS_METHODS`                              | Allowed HTTP methods (comma-separated or \*)                     | \*                                                           | API                 |
| `CORS_ORIGINS`                              | Allowed origins (comma-separated or \*)                          | \*                                                           | API                 |
| `CORS_HEADERS`                              | Allowed headers (comma-separated or \*)                          | \*                                                           | API                 |
| `ALLOWED_HOSTS`                             | Trusted hosts for TrustedHostMiddleware (comma-separated or \*)   | Derived from CORS_ORIGINS                                    | API                 |
| `SENTRY_DSN`                                | Sentry Data Source Name for error monitoring                     | None                                                         | API                 |
| `SENTRY_TRACES_SAMPLE_RATE`                 | Sentry performance monitoring sample rate (0.0 to 1.0)           | 1.0                                                          | API                 |
| `SENTRY_PROFILES_SAMPLE_RATE`               | Sentry profiling sample rate (0.0 to 1.0)                        | 1.0                                                          | API                 |
| `DATABASE_URL`                              | Database connection URL                                          | `sqlite+aiosqlite:///./inference_core.db`                    | API                 |
| `DATABASE_ECHO`                             | Echo SQL queries (development only)                              | `False`                                                      | API                 |
| `DATABASE_POOL_SIZE`                        | Database connection pool size                                    | `20`                                                         | API                 |
| `DATABASE_MAX_OVERFLOW`                     | Maximum database connection overflow                             | `30`                                                         | API                 |
| `DATABASE_POOL_TIMEOUT`                     | Pool connection timeout in seconds                               | `30`                                                         | API                 |
| `DATABASE_POOL_RECYCLE`                     | Connection recycle time in seconds                               | `3600`                                                       | API                 |
| `DATABASE_MYSQL_CHARSET`                    | MySQL character set                                              | `utf8mb4`                                                    | API, Docker         |
| `DATABASE_MYSQL_COLLATION`                  | MySQL collation                                                  | `utf8mb4_unicode_ci`                                         | Docker              |
| `DATABASE_NAME`                             | Database name                                                    | `app_db`                                                     | Docker              |
| `DATABASE_USER`                             | Database user                                                    | `db_user`                                                    | Docker              |
| `DATABASE_PASSWORD`                         | Database password                                                | `your_password`                                              | Docker              |
| `DATABASE_ROOT_PASSWORD`                    | Database root password (MySQL only)                              | `your_root_password`                                         | Docker              |
| `DATABASE_PORT`                             | Database port                                                    | `3306` (MySQL) / `5432` (PostgreSQL)                         | Docker              |
| `DATABASE_HOST`                             | Database host                                                    | `localhost` (or service name in Docker)                      | API, Docker         |
| `DATABASE_SERVICE`                          | Database backend/driver (async)                                  | `sqlite+aiosqlite`                                           | API, Docker         |
| `SECRET_KEY`                                | Secret used to sign JWT tokens                                   | `change-me-in-production`                                    | Auth                |
| `ALGORITHM`                                 | JWT signing algorithm                                            | `HS256`                                                      | Auth                |
| `ACCESS_TOKEN_EXPIRE_MINUTES`               | Access token lifetime (minutes)                                  | `30`                                                         | Auth                |
| `REFRESH_TOKEN_EXPIRE_DAYS`                 | Refresh token lifetime (days)                                    | `7`                                                          | Auth                |
| `AUTH_REGISTER_DEFAULT_ACTIVE`              | Whether new users are active by default                          | `true`                                                       | Auth                |
| `AUTH_SEND_VERIFICATION_EMAIL_ON_REGISTER`  | Send verification email on registration                          | `false`                                                      | Auth                |
| `AUTH_LOGIN_REQUIRE_ACTIVE`                 | Require active user for login                                    | `true`                                                       | Auth                |
| `AUTH_LOGIN_REQUIRE_VERIFIED`               | Require verified email for login                                 | `false`                                                      | Auth                |
| `AUTH_EMAIL_VERIFICATION_TOKEN_TTL_MINUTES` | Verification token lifetime (minutes)                            | `60`                                                         | Auth                |
| `AUTH_EMAIL_VERIFICATION_URL_BASE`          | Base URL for verification links                                  | `null`                                                       | Auth                |
| `LLM_API_ACCESS_MODE`                       | Access control mode for LLM endpoints (public/user/superuser)    | `superuser`                                                  | Auth/LLM            |
| `REDIS_URL`                                 | Redis URL for app sessions/locks (refresh sessions)              | `redis://localhost:6379/10`                                  | API/Auth            |
| `REDIS_REFRESH_PREFIX`                      | Key prefix for refresh sessions                                  | `auth:refresh:`                                              | Auth                |
| `CELERY_BROKER_URL`                         | Celery broker URL                                                | `redis://localhost:6379/0`                                   | API, Celery, Docker |
| `CELERY_RESULT_BACKEND`                     | Celery result backend URL                                        | `redis://localhost:6379/1`                                   | Celery, Docker      |
| `DEBUG_CELERY`                              | Enable debugpy for Celery worker (1 to enable)                   | `0`                                                          | Celery, Docker      |
| `REDIS_PORT`                                | Redis port (used for both host and container)                    | `6379`                                                       | Docker              |
| `FLOWER_PORT`                               | Flower port (used for both host and container)                   | `5555`                                                       | Docker              |
| `OPENAI_API_KEY`                            | API key for OpenAI provider                                      | None                                                         | LLM                 |
| `GOOGLE_API_KEY`                            | API key for Google Gemini models                                 | None                                                         | LLM                 |
| `ANTHROPIC_API_KEY`                         | API key for Anthropic Claude models                              | None                                                         | LLM                 |
| `LLM_EXPLAIN_MODEL`                         | Override model for the 'explain' task                            | None                                                         | LLM                 |
| `LLM_CONVERSATION_MODEL`                    | Override model for the 'conversation' task                       | None                                                         | LLM                 |
| `LLM_ENABLE_CACHING`                        | Enable in-process LLM response caching (fallback mode)           | `true`                                                       | LLM                 |
| `LLM_CACHE_TTL`                             | Cache TTL in seconds (fallback mode)                             | `3600`                                                       | LLM                 |
| `LLM_MAX_CONCURRENT`                        | Max concurrent LLM requests (fallback mode)                      | `5`                                                          | LLM                 |
| `LLM_ENABLE_MONITORING`                     | Enable basic LLM monitoring hooks (fallback mode)                | `true`                                                       | LLM                 |
| `RUN_LLM_REAL_TESTS`                        | Opt-in to run real-chain tests hitting providers in CI/local     | `0`                                                          | Tests               |
| `APP_PUBLIC_URL`                            | Public URL for email links (password reset, etc.)                | `http://localhost:8000`                                      | Email               |
| `EMAIL_CONFIG_PATH`                         | Path to email configuration YAML file                            | `email_config.yaml`                                          | Email               |
| `SMTP_PRIMARY_USERNAME`                     | Primary SMTP username (supports ${} env var syntax)              | None                                                         | Email               |
| `SMTP_PRIMARY_PASSWORD`                     | Primary SMTP password                                            | None                                                         | Email               |
| `SMTP_BACKUP_USERNAME`                      | Backup SMTP username                                             | None                                                         | Email               |
| `SMTP_BACKUP_PASSWORD`                      | Backup SMTP password                                             | None                                                         | Email               |
| `SMTP_O365_USERNAME`                        | Office 365 SMTP username                                         | None                                                         | Email               |
| `SMTP_O365_PASSWORD`                        | Office 365 SMTP password                                         | None                                                         | Email               |

### CORS vs Trusted Hosts Configuration

The application separates browser CORS policy from server host header validation for better security and Kubernetes compatibility:

#### CORS (Cross-Origin Resource Sharing)
- **Purpose**: Controls which browser origins can make cross-origin requests to your API
- **Configuration**: `CORS_ORIGINS`, `CORS_METHODS`, `CORS_HEADERS`
- **Example**: `CORS_ORIGINS=https://app.example.com,https://admin.example.com`
- **Security**: Should be strict in production (avoid `*` wildcards)

#### Trusted Hosts (TrustedHostMiddleware)
- **Purpose**: Validates the `Host` header to prevent Host header injection attacks
- **Configuration**: `ALLOWED_HOSTS` (optional, falls back to normalized `CORS_ORIGINS`)
- **Example**: `ALLOWED_HOSTS=app.example.com,api.example.com,127.0.0.1`
- **Kubernetes**: Automatically includes `localhost` and `127.0.0.1` in non-production for health probes

#### Why Separate Them?

**Problem**: Kubernetes health probes and internal services may not send proper browser-like `Host` headers, causing 400 errors when hosts are restricted.

**Solution**: 
- Keep CORS strict for browser security: `CORS_ORIGINS=https://app.example.com`
- Allow necessary hosts for infrastructure: `ALLOWED_HOSTS=app.example.com,127.0.0.1,localhost`

#### Configuration Examples

**Development (default)**:
```bash
# CORS and hosts allow everything - convenient for local development
CORS_ORIGINS=*
# ALLOWED_HOSTS automatically derived, includes localhost/127.0.0.1
```

**Production (recommended)**:
```bash
# Strict CORS for browser security
CORS_ORIGINS=https://app.example.com,https://admin.example.com

# Trusted hosts for server validation (includes health probe addresses)
ALLOWED_HOSTS=app.example.com,admin.example.com,127.0.0.1,localhost
```

**If not set**: `ALLOWED_HOSTS` will be automatically derived from `CORS_ORIGINS` by extracting hostnames (stripping schemes, ports, and paths).

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

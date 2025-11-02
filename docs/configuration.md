# Configuration Reference

Environment variables grouped by functional domain.

> Tip: Keep secrets out of VCS. Use `.env` locally and real secret stores in production.

## Core Application

| Variable          | Default                                                    | Description                                               |
| ----------------- | ---------------------------------------------------------- | --------------------------------------------------------- |
| `APP_NAME`        | Inference Core API                                         | Application name                                          |
| `APP_TITLE`       | Inference Core API                                         | Title exposed in OpenAPI                                  |
| `APP_DESCRIPTION` | A production-ready Inference Core API with LLM integration | API description                                           |
| `APP_VERSION`     | 0.1.0                                                      | Semantic version / release tag                            |
| `ENVIRONMENT`     | development                                                | Environment: development / staging / production / testing |
| `DEBUG`           | True                                                       | Enables verbose debug features (disable in prod)          |
| `HOST`            | 0.0.0.0                                                    | Bind host                                                 |
| `PORT`            | 8000                                                       | Bind port                                                 |

## HTTP / CORS / Hosts

| Variable        | Default                   | Description                                 |
| --------------- | ------------------------- | ------------------------------------------- |
| `CORS_METHODS`  | \*                        | Allowed HTTP methods                        |
| `CORS_ORIGINS`  | \*                        | Allowed origins (set explicit list in prod) |
| `CORS_HEADERS`  | \*                        | Allowed headers                             |
| `ALLOWED_HOSTS` | Derived from CORS_ORIGINS | TrustedHostMiddleware host allow-list       |

## Observability (Sentry)

| Variable                      | Default | Description                                     |
| ----------------------------- | ------- | ----------------------------------------------- |
| `SENTRY_DSN`                  | (none)  | Sentry DSN enables error & performance tracking |
| `SENTRY_TRACES_SAMPLE_RATE`   | 1.0     | Fraction of transactions for APM                |
| `SENTRY_PROFILES_SAMPLE_RATE` | 1.0     | Fraction of transactions for profiling          |

## Database & ORM

| Variable                   | Default                                 | Description                    |
| -------------------------- | --------------------------------------- | ------------------------------ |
| `DATABASE_URL`             | sqlite+aiosqlite:///./inference_core.db | SQLAlchemy async URL           |
| `DATABASE_ECHO`            | False                                   | Echo SQL statements (dev only) |
| `DATABASE_POOL_SIZE`       | 20                                      | Core pool size                 |
| `DATABASE_MAX_OVERFLOW`    | 30                                      | Overflow connections           |
| `DATABASE_POOL_TIMEOUT`    | 30                                      | Acquire timeout (s)            |
| `DATABASE_POOL_RECYCLE`    | 3600                                    | Connection recycle (s)         |
| `DATABASE_MYSQL_CHARSET`   | utf8mb4                                 | Charset for MySQL              |
| `DATABASE_MYSQL_COLLATION` | utf8mb4_unicode_ci                      | Collation for MySQL            |
| `DATABASE_NAME`            | app_db                                  | Name (compose / container use) |
| `DATABASE_USER`            | db_user                                 | User (compose)                 |
| `DATABASE_PASSWORD`        | your_password                           | Password (compose)             |
| `DATABASE_ROOT_PASSWORD`   | your_root_password                      | MySQL root password            |
| `DATABASE_PORT`            | 3306 / 5432                             | Port per engine                |
| `DATABASE_HOST`            | localhost                               | Host or service name           |
| `DATABASE_SERVICE`         | sqlite+aiosqlite                        | Default async driver           |

## Authentication & Tokens

| Variable                                    | Default                 | Description                           |
| ------------------------------------------- | ----------------------- | ------------------------------------- |
| `SECRET_KEY`                                | change-me-in-production | JWT signing secret (rotate in prod)   |
| `ALGORITHM`                                 | HS256                   | JWT signing algorithm                 |
| `ACCESS_TOKEN_EXPIRE_MINUTES`               | 30                      | Access token TTL                      |
| `REFRESH_TOKEN_EXPIRE_DAYS`                 | 7                       | Refresh token TTL                     |
| `AUTH_REGISTER_DEFAULT_ACTIVE`              | true                    | New users active by default           |
| `AUTH_SEND_VERIFICATION_EMAIL_ON_REGISTER`  | false                   | Auto send verification mail           |
| `AUTH_LOGIN_REQUIRE_ACTIVE`                 | true                    | Require active for login              |
| `AUTH_LOGIN_REQUIRE_VERIFIED`               | false                   | Require verified email for login      |
| `AUTH_EMAIL_VERIFICATION_TOKEN_TTL_MINUTES` | 60                      | Verification token TTL                |
| `AUTH_EMAIL_VERIFICATION_URL_BASE`          | null                    | Base URL for email verification links |

## Redis / Celery

| Variable                | Default                  | Description                        |
| ----------------------- | ------------------------ | ---------------------------------- |
| `REDIS_URL`             | redis://localhost:6379/0 | Redis for refresh sessions / locks |
| `REDIS_REFRESH_PREFIX`  | auth:refresh:            | Key prefix for refresh sessions    |
| `CELERY_BROKER_URL`     | redis://localhost:6379/0 | Celery broker                      |
| `CELERY_RESULT_BACKEND` | redis://localhost:6379/1 | Celery result backend              |
| `DEBUG_CELERY`          | 0                        | Enable debugpy attach (1=on)       |
| `REDIS_PORT`            | 6379                     | Exposed port (compose)             |
| `FLOWER_PORT`           | 5555                     | Flower UI port                     |

## LLM & Access Control

| Variable                    | Default   | Description                            |
| --------------------------- | --------- | -------------------------------------- |
| `LLM_API_ACCESS_MODE`       | superuser | Access mode: public / user / superuser |
| `OPENAI_API_KEY`            | (none)    | OpenAI API key                         |
| `GOOGLE_API_KEY`            | (none)    | Gemini key                             |
| `ANTHROPIC_API_KEY`         | (none)    | Claude key                             |
| `LLM_COMPLETION_MODEL`      | (none)    | Override model for completion task     |
| `LLM_CHAT_MODEL`            | (none)    | Override model for chat task           |
| `LLM_ENABLE_CACHING`        | true      | Enable in-process fallback cache       |
| `LLM_CACHE_TTL`             | 3600      | Cache TTL (seconds)                    |
| `LLM_MAX_CONCURRENT`        | 5         | Max concurrent LLM requests (fallback) |
| `LLM_ENABLE_MONITORING`     | true      | Basic monitoring hooks                 |
| `LLM_USAGE_LOGGING_ENABLED` | true      | Persist token/cost usage metadata      |
| `LLM_USAGE_FAIL_OPEN`       | true      | Ignore logging errors if true          |
| `RUN_LLM_REAL_TESTS`        | 0         | Enable real provider test suite        |

## Vector Store

| Variable                       | Default                                | Description                             |
| ------------------------------ | -------------------------------------- | --------------------------------------- |
| `VECTOR_BACKEND`               | (none)                                 | `qdrant`, `memory`, or blank (disabled) |
| `VECTOR_COLLECTION_DEFAULT`    | default_documents                      | Default collection name                 |
| `QDRANT_URL`                   | http://localhost:6333                  | Qdrant endpoint                         |
| `QDRANT_API_KEY`               | (none)                                 | Qdrant API key                          |
| `VECTOR_DISTANCE`              | cosine                                 | Distance metric                         |
| `VECTOR_EMBEDDING_MODEL`       | sentence-transformers/all-MiniLM-L6-v2 | Embedding model                         |
| `VECTOR_DIM`                   | 384                                    | Embedding dimension                     |
| `VECTOR_INGEST_MAX_BATCH_SIZE` | 1000                                   | Max ingestion batch size                |

## Email / Notifications

| Variable                | Default               | Description             |
| ----------------------- | --------------------- | ----------------------- |
| `APP_PUBLIC_URL`        | http://localhost:8000 | Base URL used in emails |
| `EMAIL_CONFIG_PATH`     | email_config.yaml     | Config file path        |
| `SMTP_PRIMARY_USERNAME` | (none)                | Primary SMTP username   |
| `SMTP_PRIMARY_PASSWORD` | (none)                | Primary SMTP password   |
| `SMTP_BACKUP_USERNAME`  | (none)                | Backup SMTP username    |
| `SMTP_BACKUP_PASSWORD`  | (none)                | Backup SMTP password    |
| `SMTP_O365_USERNAME`    | (none)                | Office 365 username     |
| `SMTP_O365_PASSWORD`    | (none)                | Office 365 password     |

## Notes

- Restrict `CORS_ORIGINS` & set explicit `ALLOWED_HOSTS` in production.
- Rotate `SECRET_KEY`; avoid storing in repo or image layers.
- Reduce Sentry sample rates for high throughput environments.
- For Qdrant production add persistence volume & consider auth.

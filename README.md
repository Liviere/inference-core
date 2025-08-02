# backend-template

A ready-to-use API template based on FastAPI.

## Requirements

- Python 3.12 or higher

## Getting Started

This project uses [Poetry](https://python-poetry.org/) for dependency management.
Make sure you have it installed before proceeding.

### Install dependencies

```bash
poetry install
```

### Development

To start the development server, run:

```bash
poetry run fastapi dev
```

### Serve the application

```bash
poetry run fastapi run
```

## API Endpoints (v1)

### Health Check

- `GET /api/v1/health/` - Overall application health check
- `GET /api/v1/health/ping - Simple ping endpoint for basic health checking

## Configuration

The application uses environment variables for configuration. You can set them directly or create a `.env` file in the project root.

To get started quickly, copy the example configuration file:

```bash
cp .env.example .env
```

Then edit the `.env` file with your specific settings.

### Environment Variables

| Variable          | Description                                                      | Default                                                            | Required |
| ----------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------ | -------- |
| `APP_NAME`        | Application name                                                 | "Backend Template API"                                             | No       |
| `APP_TITLE`       | Application title                                                | "Backend Template API"                                             | No       |
| `APP_DESCRIPTION` | Application description                                          | "A production-ready FastAPI backend template with LLM integration" | No       |
| `APP_VERSION`     | Application version                                              | "0.1.0"                                                            | No       |
| `ENVIRONMENT`     | Application environment (development/staging/production/testing) | "development"                                                      | No       |
| `DEBUG`           | Debug mode                                                       | `True`                                                             | No       |
| `HOST`            | Server host                                                      | "0.0.0.0"                                                          | No       |
| `PORT`            | Server port                                                      | 8000                                                               | No       |
| `CORS_METHODS`    | Allowed HTTP methods (comma-separated or \*)                     | \*                                                                 | No       |
| `CORS_ORIGINS`    | Allowed origins (comma-separated or \*)                          | \*                                                                 | No       |
| `CORS_HEADERS`    | Allowed headers (comma-separated or \*)                          | \*                                                                 | No       |

## Logging

The application is configured to log information to both the console and a rotating file.

- **Console Logging**: Provides real-time output during development.
- **File Logging**: Logs are saved in JSON format to the `logs/app.log` file. The log file is rotated daily, and backups are kept for 30 days.

This setup is handled by the configuration in `app/core/logging_config.py`.

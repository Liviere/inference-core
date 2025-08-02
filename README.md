# backend-template

A ready-to-use API template based on FastAPI.

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

| Variable                      | Description                                                      | Default                                                            | Required |
| ----------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------ | -------- |
| `APP_NAME`                    | Application name                                                 | "Backend Template API"                                             | No       |
| `APP_TITLE`                   | Application title                                                | "Backend Template API"                                             | No       |
| `APP_DESCRIPTION`             | Application description                                          | "A production-ready FastAPI backend template with LLM integration" | No       |
| `APP_VERSION`                 | Application version                                              | "0.1.0"                                                            | No       |
| `ENVIRONMENT`                 | Application environment (development/staging/production/testing) | "development"                                                      | No       |
| `DEBUG`                       | Debug mode                                                       | `True`                                                             | No       |
| `HOST`                        | Server host                                                      | "0.0.0.0"                                                          | No       |
| `PORT`                        | Server port                                                      | 8000                                                               | No       |
| `CORS_METHODS`                | Allowed HTTP methods (comma-separated or \*)                     | \*                                                                 | No       |
| `CORS_ORIGINS`                | Allowed origins (comma-separated or \*)                          | \*                                                                 | No       |
| `CORS_HEADERS`                | Allowed headers (comma-separated or \*)                          | \*                                                                 | No       |
| `SENTRY_DSN`                  | Sentry Data Source Name for error monitoring                     | None                                                               | No       |
| `SENTRY_TRACES_SAMPLE_RATE`   | Sentry performance monitoring sample rate (0.0 to 1.0)           | 1.0                                                                | No       |
| `SENTRY_PROFILES_SAMPLE_RATE` | Sentry profiling sample rate (0.0 to 1.0)                        | 1.0                                                                | No       |

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
- **File Logging**: Logs are saved in JSON format to the `logs/app.log` file. The log file is rotated daily, and backups are kept for 30 days.

This setup is handled by the configuration in `app/core/logging_config.py`.

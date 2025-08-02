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

### Run the application

```bash
cd app
poetry run fastapi run
```

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

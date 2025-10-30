# Repository Structure Snapshot

Snapshot date: 2025-10-02 (Phase 1 documentation refactor)

```text
inference_core/
├── docker-compose.base.yml
├── llm_config.example.yaml
├── llm_config.yaml
├── poetry.lock
├── prometheus.yml
├── pyproject.toml
├── pytest.ini
├── README.md
├── run.py
├── test_observability.py
├── inference_core/
│   ├── __init__.py
│   ├── main_factory.py
│   ├── api/
│   │   └── v1/
│   ├── celery/
│   │   ├── __init__.py
│   │   ├── celery_main.py
│   │   ├── config.py
│   │   ├── README.md
│   │   └── tasks/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── dependecies.py  # NOTE: filename contains a typo; planned rename to dependencies.py
│   │   ├── logging_config.py
│   │   ├── redis_client.py
│   │   └── security.py
│   ├── database/
│   │   └── sql/
│   ├── frontend/
│   │   └── stream.html
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── chains.py
│   │   ├── config.py
│   │   ├── models.py
│   │   ├── param_policy.py
│   │   ├── prompts.py
│   │   ├── streaming.py
│   │   └── batch/
│   ├── custom_prompts/
│   │   ├── completion/       # .j2/.jinja2 templates for completion (expects {prompt})
│   │   └── chat/             # .j2/.jinja2 system prompts for chat (e.g., tutor.system.j2)
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── metrics.py
│   │   └── sentry.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── batch.py
│   │   ├── common.py
│   │   └── tasks_responses.py
│   └── services/
│       ├── auth_service.py
│       ├── batch_service.py
│       ├── llm_service.py
│       ├── refresh_session_store.py
│       └── task_service.py
├── docker/
│   ├── docker-compose.mysql.yml
│   ├── docker-compose.postgres.yml
│   ├── docker-compose.sqlite.yml
│   ├── Dockerfile
│   └── README.md
├── docs/
│   ├── batch-providers-gemini-claude.md
│   ├── batch-tasks-testing.md
│   ├── testing-docker.md
│   ├── issues/
│   └── providers/
├── logs/
│   ├── app.log
│   ├── app.log.YYYY-MM-DD (archiwalne)
├── observability/
│   ├── README.md
│   ├── grafana/
│   └── prometheus/
├── reports/
│   └── performance/
├── scripts/
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── README.md
│   ├── frontend/
│   ├── integration/
│   ├── performance/
│   └── unit/
└── __pycache__/ (artefakty Pythona)
```

## Legend (key directories)

- `inference_core/` – application core (API, business logic, integrations)
- `inference_core/core/` – configuration, DI, security, logging
- `inference_core/celery/` – Celery app & task definitions
- `inference_core/llm/` – LLM integration (LangChain, parameter policies, prompts, batch)
- `inference_core/observability/` – metrics, structured logging, Sentry integration
- `inference_core/schemas/` – Pydantic schemas (request/response/data models)
- `inference_core/services/` – service layer abstractions
- `inference_core/vectorstores/` – vector store backends & logic
- `docs/` – technical & design documentation
- `docker/` – container & compose definitions
- `tests/` – unit, integration, performance tests
- `logs/` – runtime logs (local dev)
- `observability/` – Prometheus & Grafana infra configs
- `reports/` – performance or analytics reports
- `scripts/` – helper scripts (future)

## Notes

1. Some directories may be omitted if empty.
2. Runtime artifacts: `__pycache__/`, `*.pyc`, Celery beat schedule files, local SQLite DB.
3. This snapshot is currently maintained manually. Goal: auto-generate via a script in Phase 2.

### Planned automation (example command)

```bash
# Example (to be scripted) – generate tree excluding transient paths
tree -I '__pycache__|*.pyc|*.log|logs|*.db|celerybeat-schedule*' -a -F > docs/structure.md
```

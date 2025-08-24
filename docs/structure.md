# Struktura katalogów projektu

Poniżej znajduje się aktualny schemat plików i katalogów (snapshot). Wygenerowano: 2025-08-24.

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
│   ├── main.py
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
│   │   ├── dependecies.py
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

## Legenda (kluczowe katalogi)

- `app/` – główny kod aplikacji (API, logika biznesowa, integracje)
- `app/core/` – konfiguracja, DI, bezpieczeństwo, logowanie
- `app/celery/` – definicje i konfiguracja zadań asynchronicznych (Celery)
- `app/llm/` – integracje z modelami językowymi (LangChain, polityki parametrów, prompt engineering)
- `app/observability/` – metryki, logowanie, Sentry
- `app/schemas/` – schematy Pydantic (walidacja/request/response)
- `app/services/` – warstwa usługowa (biznesowa) nad modelami/infrastrukturą
- `docs/` – dokumentacja projektowa i techniczna
- `docker/` – pliki uruchomienia w kontenerach / compose
- `tests/` – testy (unit, integration, performance)
- `logs/` – logi runtime (lokalne)
- `observability/` – konfiguracje monitoringu (Prometheus, Grafana)
- `reports/` – raporty np. wydajnościowe
- `scripts/` – (miejsce na) skrypty pomocnicze

## Notatki

1. Katalogi puste (bez plików) mogą nie być pokazane w niektórych listowaniach.
2. Pliki tymczasowe (`*.pyc`, `__pycache__/`, walidacje Celery beat) są artefaktami runtime.
3. Aktualizacja tego pliku jest manualna – rozważ wygenerowanie automatyczne (np. skrypt `tree` z filtrem) jeśli struktura często się zmienia.

### Propozycja automatycznej aktualizacji (opcjonalnie)

Możesz dodać do README fragment komendy:

```bash
# Linux/macOS – wygeneruj aktualne drzewo (ignorując wybrane katalogi)\n\n tree -I '__pycache__|*.pyc|*.log|logs|app.db*|celerybeat-schedule*' -a -F > docs/structure.md
```

Jeśli chcesz inną formę (JSON, tylko katalogi, filtrowanie), daj znać.

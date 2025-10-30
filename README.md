<p align="center">
  <a href="https://github.com/Liviere/backend-template" target="_blank" rel="noopener">
    <img src="assets/inference-core-logo.png" alt="Inference Core" height="140" />
  </a>
</p>
<p align="center">
  <em>Production‑ready FastAPI + Celery backbone for LLM apps: providers, chains, batches, monitoring, vectors & auth.</em>
</p>
<p align="center">
  <a href="#" target="_blank"><img alt="Build" src="https://img.shields.io/badge/build-ci--pending-lightgrey" /></a>
  <a href="#" target="_blank"><img alt="Python" src="https://img.shields.io/badge/python-3.12+-3776AB" /></a>
  <a href="#" target="_blank"><img alt="License" src="https://img.shields.io/badge/license-MIT-green" /></a>
  <a href="#configuration" target="_blank"><img alt="Config" src="https://img.shields.io/badge/config-.env%20%2B%20yaml-blue" /></a>
  <a href="docs/README.md" target="_blank"><img alt="Docs" src="https://img.shields.io/badge/docs-index-informational" /></a>
</p>

---

**Documentation Index** → [`docs/README.md`](docs/README.md) • **Configuration Reference** → [`docs/configuration.md`](docs/configuration.md) • **Docker Deploy Guide** → [`docker/README.md`](docker/README.md)

---

# Inference Core

Inference Core is a modular backend scaffold for Large Language Model–driven platforms. It focuses on:

- Fast provider integration (OpenAI / Gemini / Claude – easily extensible)
- Clean chain layer (Completion → `explain`, Chat → `converse`)
- Multiple request modes: synchronous, streaming, provider‑native batch (cost reduction)
- Deep observability: usage logging, cost estimation, metrics, tracing
- Built‑in user system + JWT auth (access + rotated refresh in Redis)
- Asynchronous background execution with Celery
- Pluggable relational DB (SQLite / PostgreSQL / MySQL) & vector backend (Qdrant)
- Unified operational monitoring (Sentry, Prometheus, Grafana)
- Comprehensive test layers (unit, integration, performance baseline)

> Goal: Collapse the time between an idea and a production‑grade inference service with cost & reliability controls baked in.

---

## ✨ Core Features

**LLM Provider Abstraction** – Central config (YAML + ENV) with overridable model mapping & parameter policies.

**Chains Layer** – Opinionated minimal surface for completion (explain) & chat (converse) tasks; easy to extend with new task types.

**Request Modes** – Same logical interface for:

- Sync (immediate response)
- Streaming (token/event feed)
- Native Batch (provider discounted large runs; orchestration via Celery)

**Cost & Usage Tracking** – Token counts & pricing snapshots, failure‑tolerant logging, Prometheus metrics, Sentry traces. **Generic helpers for custom LLM tasks** – Reusable abstraction to add usage/cost logging to any custom task (extraction, summarization, etc.) without duplicating boilerplate. [Learn more →](docs/custom-task-usage-logging.md)

**Authentication & Users** – Registration, login, refresh rotation, email verification hooks (pluggable email delivery).

**Celery Orchestration** – Background tasks for batch polling, ingestion, email, vector operations with resilience & retry hooks.

**Vector Store Integration** – Qdrant (production) or in‑memory (dev); ingestion + similarity search + async batch flows.

**Observability** – Structured JSON logging, metrics (Prometheus), tracing & error tracking (Sentry), future dashboards (Grafana).

**Testing Layers** – Unit tests (pure logic), integration tests (API/services), performance scaffolding (Locust planned), optional real‑provider tests behind flags.

---

## 🚀 Quick Start (Local Dev)

```bash
poetry install
cp .env.example .env
cp llm_config.example.yaml llm_config.yaml

# Run API (auto-reload)
poetry run uvicorn inference_core.main_factory:create_application --factory --reload --port 8000

# (Optional) Start Redis & Celery for background + batch
docker run -d --name redis -p 6379:6379 redis:7-alpine
poetry run celery -A inference_core.celery.celery_main:celery_app worker --pool=gevent --autoscale=200,10 --loglevel=info
```

Visit: http://localhost:8000/docs (dev only)  
Health: `GET /api/v1/health/`

Docker Deployment: For containerized deployment (SQLite/MySQL/Postgres) and compose examples, see [`docker/README.md`](docker/README.md).

---

## 🔌 Minimal Usage (Programmatic)

```python
from inference_core.main_factory import create_application

app = create_application()  # Standard FastAPI instance
```

Simple LLM call (HTTP):

```bash
curl -X POST http://localhost:8000/api/v1/llm/explain \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <ACCESS_TOKEN>' \
  -d '{"prompt": "Explain vector embeddings simply"}'
```

Streaming (Server-Sent Tokens style endpoint may vary):

```bash
curl -N http://localhost:8000/api/v1/llm/stream/explain?prompt=Hello
```

Batch (conceptual – provider-native):

```bash
curl -X POST http://localhost:8000/api/v1/llm/batch/jobs \
  -H 'Authorization: Bearer <ACCESS_TOKEN>' \
  -H 'Content-Type: application/json' \
  -d '{"items": [{"prompt": "Explain RAG"},{"prompt":"Explain embeddings"}]}'
```

Vector ingest example:

```bash
curl -X POST http://localhost:8000/api/v1/vector/ingest \
  -H 'Authorization: Bearer <ACCESS_TOKEN>' \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Python is great","Embeddings map text to vectors"],"async_mode":false}'
```

---

## 🧩 Integrating Into Your Own Platform

There are two common integration modes:

1. Standalone microservice (run as its own deployment – interact via HTTP).
2. Embedded module (submodule/vendor) merged into your wider application.

### Recommended Embedded Layout

```text
your-project/
├── app/
├── main.py                 # Your FastAPI entrypoint (aggregator)
├── api/
│   └── v1/
│       └── custom.py       # Custom domain router (example)
├── services/
│   └── custom_service.py   # Wrapper extending LLMService
├── core/                   # (git submodule) → inference-core
│   ├── llm_config.yaml     # Model/provider config (use llm_config.example.yaml as template)
│   └── .env                # Core .env overrides (use .env.example as template)
├── frontend/               # (optional) web assets / SPA (or any other external service)
├── pyproject.toml
└── README.md
```

Add the core as a submodule (example):

```bash
git submodule add https://github.com/Liviere/backend-template core
```

### Application Factory (Aggregator)

Minimal pattern to mount the core app and add domain-specific routers:

```python
# app/main.py
from dotenv import load_dotenv
from fastapi import APIRouter
from inference_core.main_factory import create_application

from .api.v1.custom import router as custom_router

load_dotenv()
load_dotenv("core/.env", override=True)


simple_router = APIRouter(tags=["Simple"])


@simple_router.get("/")
def read_root():
    return {"message": "Welcome to your custom API"}


app = create_application(
    external_routers={
        "/api/v1/simple": simple_router,
        "/api/v1/custom": custom_router,
    }
)

```

### Custom Router Using Core Dependencies

```python
# app/api/v1/turns.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from inference_core.core.dependecies import (
  get_current_active_user,
  get_db,
)

from app.services.custom_service import custom_llm_service

router = APIRouter()

@router.get("/", summary="Some main custom route")
async def get_items(
  user = Depends(get_current_active_user),
  db: AsyncSession = Depends(get_db),
):
  # Example: query your own domain tables using db
  return {"user_id": user["id"], "items": []}

@router.post("/hint", summary="Generate a study hint")
async def generate_hint(
  phrase: str,
  level: str = "B1",
  user = Depends(get_current_active_user),
):
  hint = await custom_llm_service.generate_hint(phrase, level)
  return {"hint": hint, "user": user["id"]}
```

### Extending the LLM Service

Instead of calling the lower-level chain constructors directly, wrap `LLMService` to express domain semantics.

```python
# app/services/turn_llm_service.py
from inference_core.services.llm_service import LLMService

class CustomLLMService:
  def __init__(self):
    self._llm = LLMService()

  async def generate_hint(self, phrase: str, cefr_level: str) -> str:
    prompt = f"Explain the phrase '{phrase}' in simple terms for CEFR level {cefr_level}."
    response = await self._llm.explain(question=prompt)
    return response.result["answer"]

# Singleton-style instance (simple pattern)
custom_llm_service = CustomLLMService()
```

### Direct Low-Level Usage (Optional)

```python
from inference_core.services.llm_service import LLMService

llm = LLMService()
answer = await llm.explain("What are embeddings?")
chat_turn = await llm.converse(session_id="demo", user_input="Hello!")
```

---

## ⚙️ Configuration (Where To Look)

Central references:

- Environment variables: [`docs/configuration.md`](docs/configuration.md)
- LLM provider & model mapping: `llm_config.yaml`
- Vector settings: `VECTOR_*` ENV + [`docs/vector-store.md`](docs/vector-store.md)
- Access control: `LLM_API_ACCESS_MODE` (`public`, `user`, `superuser`)

Production tips:

- Pin exact model versions (avoid silent quality regressions).
- Restrict `CORS_ORIGINS` and set explicit `ALLOWED_HOSTS`.
- Reduce Sentry sample rates for high traffic.
- Monitor DB pool saturation & Celery queue latency.

---

## 🔍 Observability Snapshot

You get structured logs, Prometheus metrics (vector operations, LLM usage), and Sentry traces out of the box. Cost tracking stores token counts & pricing metadata (no raw prompts persisted by default).

See: [`docs/observability/llm-usage-logging.md`](docs/observability/llm-usage-logging.md)

---

## 🧪 Testing Strategy

Run everything:

```bash
poetry run pytest
```

Selective layers:

```bash
poetry run pytest tests/unit
poetry run pytest tests/integration -m integration
```

Optional real provider tests (require API keys):

```bash
RUN_LLM_REAL_TESTS=1 poetry run pytest -m integration -k llm
```

---

## 📄 License

MIT – adapt freely with attribution. (Add a `LICENSE` file if not already present.)

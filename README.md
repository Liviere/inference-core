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

**Documentation Index** → [`docs/README.md`](docs/README.md) • **Configuration Reference** → [`docs/configuration.md`](docs/configuration.md) • **Custom Prompts** → [`docs/custom-prompts.md`](docs/custom-prompts.md) • **Docker Deploy Guide** → [`docker/README.md`](docker/README.md)

---

# Inference Core

Inference Core is a modular backend scaffold for Large Language Model–driven platforms. It focuses on:

- Fast provider integration (OpenAI / Gemini / Claude – easily extensible)
- Clean chain layer (Completion and Chat)
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

**Chains Layer** – Opinionated minimal surface for completion & chat tasks; easy to extend with new task types.

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
poetry run celery -A inference_core.celery.celery_main:celery_app worker --loglevel=info
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
curl -X POST http://localhost:8000/api/v1/llm/completion \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <ACCESS_TOKEN>' \
  -d '{"prompt": "Explain vector embeddings simply"}'
```

Streaming (Server-Sent Tokens style endpoint may vary):

```bash
curl -X POST -N http://localhost:8000/api/v1/llm/completion/stream \\
  -H 'Content-Type: application/json' \\
  -d '{"prompt": "Hello"}'
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

Prefer inheriting from `LLMService` and providing default attributes (e.g. system prompt, models, params) or overriding factory hooks.

```python
# app/services/custom_llm_service.py
from inference_core.services.llm_service import LLMService


class CustomLLMService(LLMService):
  """Example domain specialization with its own system prompt and parameters."""

  def __init__(self) -> None:
    super().__init__(
      default_models={
        # tasks: "completion" | "chat"
        "chat": "gpt-4o-mini",
      },
      default_model_params={
        "chat": {"temperature": 0.3},
        "completion": {"temperature": 0.2},
      },
      default_prompt_names={
        "chat": "chat",           # or point to a file in custom_prompts/chat/<name>.j2
        "completion": "completion",
      },
      default_chat_system_prompt=(
        "You are a domain tutor. Be concise, prioritize clarity,"
        " and include one short example if helpful."
      ),
    )

  async def generate_hint(self, phrase: str, cefr_level: str) -> str:
    prompt = (
      f"Explain the phrase '{phrase}' in simple terms for CEFR level {cefr_level}."
    )
    # You can also use per-call overrides, e.g. prompt_name
    response = await self.completion(prompt=prompt, prompt_name="completion")
    return response.result["answer"]

  # Optional: precisely override the hook that builds the chat chain
  # (e.g., to always enforce a specific system prompt or template name)
  # def _build_chat_chain(self, *, model_name, model_params, prompt_name, system_prompt):
  #     return super()._build_chat_chain(
  #         model_name=model_name,
  #         model_params=model_params,
  #         prompt_name=prompt_name or "chat",
  #         system_prompt=system_prompt or self._default_chat_system_prompt,
  #     )


# Simple singleton for injection
custom_llm_service = CustomLLMService()
```

Quick "copy" of a task configuration (e.g. different system prompt) without creating a subclass:

```python
from inference_core.services.llm_service import LLMService

base_llm = LLMService()
coach_llm = base_llm.copy_with(default_chat_system_prompt="Coach tone, ask guiding questions.")

# coach_llm.chat(...)
```

### Direct Low-Level Usage (Optional)

```python
from inference_core.services.llm_service import LLMService

llm = LLMService()
answer = await llm.completion("What are embeddings?")
chat_turn = await llm.chat(session_id="demo", user_input="Hello!")

# Multiple variables with input_vars
# Your Jinja2 templates can reference extra fields.
answer2 = await llm.completion(
  input_vars={
    "prompt": "Describe photosynthesis",
    "topic": "biology",
    "tone": "simplified",
  },
  prompt_name="simple_explainer",
)

chat_turn2 = await llm.chat(
  session_id="demo",
  user_input="How does JWT work?",  # used for history
  input_vars={"context": "FastAPI", "audience": "junior dev"},
  prompt_name="tutor",
)
```

### Custom task types (task_type)

You can route requests through a custom logical task name to pick model defaults and fallbacks from `llm_config.yaml` (tasks section), and optionally select default prompts per task via `LLMService` defaults.

1. Define a custom task in `llm_config.yaml`:

```yaml
tasks:
  summarization:
    primary: 'claude-3-5-haiku-latest'
    fallback: ['gpt-5-nano']
    testing: ['gpt-5-nano']
    description: 'Fast summaries'
```

2. Call the API with `task_type`:

```bash
# Completion (sync)
curl -X POST http://localhost:8000/api/v1/llm/completion \
  -H 'Content-Type: application/json' \
  -d '{
    "task_type": "summarization",
    "prompt": "Summarize the following article..."
  }'

# Chat (stream)
curl -X POST -N http://localhost:8000/api/v1/llm/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{
    "task_type": "summarization",
    "session_id": "demo",
    "user_input": "Summarize the main points.",
    "input_vars": {"audience": "executive"}
  }'
```

3. Optionally set default prompt templates per task when constructing or cloning the service:

```python
from inference_core.services.llm_service import LLMService

llm = LLMService().copy_with(
  default_prompt_names={
    "summarization": "summary_short",  # resolves to custom_prompts/completion/summary_short.j2 for completion
    "chat": "tutor"
  }
)

resp = await llm.completion(task_type="summarization", input_vars={"prompt": "..."})
```

Notes:

- `task_type` is optional. If omitted, the built-in mapping uses "completion" or "chat" as before.
- `default_prompt_names` can be keyed by any task name (built-in or custom). For chat, you can also override `system_prompt` per-call.
- All programmatic methods and streaming variants accept `task_type`.

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

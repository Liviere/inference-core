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

| Variable                        | Default                  | Description                                                                             |
| ------------------------------- | ------------------------ | --------------------------------------------------------------------------------------- |
| `REDIS_URL`                     | redis://localhost:6379/0 | Redis for refresh sessions / locks                                                      |
| `REDIS_REFRESH_PREFIX`          | auth:refresh:            | Key prefix for refresh sessions                                                         |
| `CELERY_BROKER_URL`             | redis://localhost:6379/0 | Celery broker                                                                           |
| `CELERY_RESULT_BACKEND`         | redis://localhost:6379/1 | Celery result backend                                                                   |
| `CELERY_THREADS_CONCURRENCY`    | 20                       | Docker threads worker concurrency for `default`, `llm_tasks`, `mail`, and `batch_tasks` |
| `CELERY_EMBEDDINGS_CONCURRENCY` | 2                        | Docker prefork worker concurrency for the `embeddings` queue                            |
| `DEBUG_CELERY`                  | 0                        | Enable debugpy attach (1=on)                                                            |
| `REDIS_PORT`                    | 6379                     | Exposed port (compose)                                                                  |
| `FLOWER_PORT`                   | 5555                     | Flower UI port                                                                          |

Docker compose uses a split worker topology by default: the threads worker handles I/O-bound queues, while the prefork embeddings worker isolates local SentenceTransformer jobs.

## LLM & Access Control

| Variable                    | Default   | Description                            |
| --------------------------- | --------- | -------------------------------------- |
| `LLM_API_ACCESS_MODE`       | superuser | Access mode: public / user / superuser |
| `OPENAI_API_KEY`            | (none)    | OpenAI API key                         |
| `GOOGLE_API_KEY`            | (none)    | Gemini key                             |
| `ANTHROPIC_API_KEY`         | (none)    | Claude key                             |
| `FIREWORKS_API_KEY`         | (none)    | Fireworks key                          |
| `TAVILY_API_KEY`            | (none)    | Tavily key for internet search tools   |
| `OPEN_WEATHER_API_KEY`      | (none)    | OpenWeatherMap key for weather tools   |
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

## Embedding Backend

Controls how embeddings are generated for vector operations, agent memory, and the embeddings API endpoint.

| Variable                  | Default                                  | Description                                                      |
| ------------------------- | ---------------------------------------- | ---------------------------------------------------------------- |
| `EMBEDDING_BACKEND`       | `local`                                  | Embedding backend: `local` or `remote`                           |
| `EMBEDDING_LOCAL_MODEL`   | `sentence-transformers/all-MiniLM-L6-v2` | SentenceTransformer model used by the dedicated embedding worker |
| `EMBEDDING_LOCAL_TIMEOUT` | `60`                                     | Timeout in seconds while waiting for the Celery embedding task   |

Backend behavior:

- `local`: `EmbeddingService` sends `embeddings.generate` tasks to the dedicated `embeddings` queue. Run a separate Celery worker with `--queues=embeddings --pool=prefork`.
- `remote`: `EmbeddingService` instantiates a LangChain embedding provider from the `embeddings:` section in `llm_config.yaml` (OpenAI, Gemini, DeepInfra, Fireworks, Ollama).

Example YAML for `EMBEDDING_BACKEND=remote`:

```yaml
embeddings:
  default:
    provider: openai
    model: text-embedding-3-small
    # dimensions: 1536
```

## Agent Memory

Long-term memory for LangChain v1 agents. Requires `VECTOR_BACKEND` to be configured.

| Variable                                | Default      | Description                                                          |
| --------------------------------------- | ------------ | -------------------------------------------------------------------- |
| `AGENT_MEMORY_ENABLED`                  | false        | Enable long-term memory for agents                                   |
| `AGENT_MEMORY_COLLECTION`               | agent_memory | Collection name for memory storage (separate from RAG)               |
| `AGENT_MEMORY_MAX_RESULTS`              | 5            | Max memories to retrieve during recall                               |
| `AGENT_MEMORY_AUTO_RECALL`              | true         | Auto-recall relevant memories in middleware before_agent             |
| `AGENT_MEMORY_POSTRUN_ANALYSIS_ENABLED` | true         | Run best-effort post-run tool-call analysis after each agent session |
| `AGENT_MEMORY_POSTRUN_ANALYSIS_MODEL`   | (none)       | Optional override model for post-run memory tool-calling             |

Per-agent memory behavior can also be tuned in `llm_config.yaml` inside each `agents:` entry:

- `memory_tools` controls which memory tools are exposed to the model. Set it to `[]` to disable all model-facing memory tools while keeping `after_agent` analysis active.
- `memory_session_context_enabled` controls whether recalled memories are injected into the session prompt during model calls.
- `memory_tool_instructions_enabled` controls whether CoALA memory usage instructions are appended to the system prompt.
- All three fields default to `null`, which preserves the existing global behavior for backward compatibility.

## Capability-Aware Tool Routing

Agents can expose tools that need capabilities the primary model may not have,
such as vision tools that return image inputs. This is configured in
`llm_config.yaml` across both `models:` and `agents:`.

- Set `multimodal: true` on a model when it can accept multimodal inputs.
- Tools opt in by declaring `requires_multimodal = True` on the tool class.
- `on_missing_capability: skip` filters those tools out when the agent's primary model is not multimodal.
- `on_missing_capability: delegate` keeps those tools visible and auto-routes their execution through `multimodal_support_model` using the tool-model switch middleware.
- `multimodal_support_model` must point to a model defined under `models:` and that model should also declare `multimodal: true`.

Example `llm_config.yaml`:

```yaml
models:
  gpt-5-mini:
    provider: 'openai'
    multimodal: true

  deepseek-ai/DeepSeek-V3-0324:
    provider: 'deepinfra'
    multimodal: false

agents:
  vision_aware_agent:
    primary: 'deepseek-ai/DeepSeek-V3-0324'
    local_tool_providers: ['browser_tools']
    on_missing_capability: 'delegate' # 'skip' | 'delegate'
    multimodal_support_model: 'gpt-5-mini'
```

Use `skip` when you want a strict tool set that matches the primary model.
Use `delegate` when the primary model should remain text-only but selected tool
calls need a multimodal fallback.

## Agent Skills & Subagents (DeepAgent)

Specialized capabilities and delegation for `DeepAgentService`. Configured in `llm_config.yaml` under `agents:`.

- **Skills**: Paths to `SKILL.md` directories containing instructions the agent reads on-demand.
- **Subagents**: List of other agents (from `agents:` section) that the main agent can delegate tasks to using the `task()` tool.

Example `llm_config.yaml`:

```yaml
agents:
  research_agent:
    primary: 'gpt-5-mini'
    skills: ['./skills/research/']
    subagents: ['web_searcher']
  web_searcher:
    primary: 'gpt-5-mini'
    local_tool_providers: ['assistant_tools']
```

## Agent Tool-Call Limits

Per-agent tool call caps are also configured in `llm_config.yaml` under each
`agents:` entry:

```yaml
agents:
  research_agent:
    primary: 'gpt-5-mini'
    tool_call_limits:
      global_limit:
        run_limit: 30
        thread_limit: 120
        exit_behavior: continue
      per_tool:
        - tool_name: 'fetch_url'
          run_limit: 5
```

- `run_limit` resets for each user invocation.
- `thread_limit` persists for the whole conversation thread.
- `exit_behavior: continue` blocks the tool call and lets the model finish the
  response with the context it already has.
- `exit_behavior: error` raises when the limit is hit.

When `tool_call_limits` is present, the runtime also appends policy
instructions to the system prompt so the model can summarize progress and stop
retrying blocked tools in the same response.

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
| `SMTP_OAUTH_TOKEN`      | (none)                | OAuth2 Access Token     |

## Dynamic Configuration Overrides

In addition to environment variables and YAML files, this project supports dynamic overrides stored in the database.

- **Admin Overrides**: Global or scoped system changes managed by administrators.
- **User Preferences**: Per-user configuration for models and tasks (subject to allowlist constraints).

For detailed information on how these layers interact and how to manage them via API, see [Dynamic LLM Configuration](dynamic-configuration.md).

## Notes

- Restrict `CORS_ORIGINS` & set explicit `ALLOWED_HOSTS` in production.
- Rotate `SECRET_KEY`; avoid storing in repo or image layers.
- Reduce Sentry sample rates for high throughput environments.
- For Qdrant production add persistence volume & consider auth.

## Model Parameters (YAML)

The system supports arbitrary parameters in `llm_config.yaml` within model definitions. These can be simple values or nested structures (dictionaries).

To use an extra parameter, you must also allow it in the `param_policies` section:

```yaml
models:
  gpt-5-nano:
    provider: 'openai'
    logit_bias:
      '50256': -100

param_policies:
  providers:
    openai:
      patch:
        allowed: ['logit_bias']
```

## Playwright MCP / Docker timeouts

You can override MCP CLI timeouts via environment variables. The compose files include these variables and provide sensible defaults.

| Variable                        | Default | Description                                                 |
| ------------------------------- | ------- | ----------------------------------------------------------- |
| `PLAYWRIGHT_TIMEOUT_ACTION`     | 5000    | Action timeout in milliseconds (`--timeout-action`)         |
| `PLAYWRIGHT_TIMEOUT_NAVIGATION` | 60000   | Navigation timeout in milliseconds (`--timeout-navigation`) |

These are injected into the MCP CLI command by the Docker compose files (see `docker/docker-compose.playwright-mcp*.yml`). If you don't set them, compose will fall back to the defaults shown above.

## LangGraph Agent Server

Controls remote agent execution via the LangGraph Platform. When enabled, agents with `execution_mode: 'remote'` in `llm_config.yaml` delegate runs to the Agent Server instead of executing locally.

| Variable                  | Default | Description                                                                                                                                                  |
| ------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `AGENT_SERVER_ENABLED`    | false   | Master switch — when True, remote-mode agents delegate to Agent Server                                                                                       |
| `AGENT_SERVER_URL`        | (none)  | Base URL (`http://localhost:2024` for `langgraph dev`, `:8123` for `langgraph up`); also returned by the run-bundle endpoint for direct frontend connections |
| `AGENT_SERVER_API_KEY`    | (none)  | API key for authenticating with the Agent Server                                                                                                             |
| `AGENT_SERVER_TIMEOUT`    | 300     | HTTP timeout in seconds for Agent Server requests (10–3600)                                                                                                  |
| `LANGGRAPH_AUTH_DISABLED` | false   | Local-dev escape hatch for `langgraph dev` without a bearer token; must stay false/unset outside development                                                 |

Development workflow:

- **`langgraph dev`** — lightweight server (port 2024), no Docker, hot reload, in-memory state. Primary tool for daily development.
- **`langgraph up`** — Docker-based stack (port 8123), PostgreSQL + Redis, production-like. Use for pre-deployment validation.

### Direct Frontend Connections, Auth, and CORS

If the frontend connects to the LangGraph Agent Server directly, the Agent
Server must validate the same JWT access tokens issued by the FastAPI backend.
The repository's `langgraph.example.json` demonstrates the required setup:

- `auth.path` points to `langgraph_auth.py:auth`, which verifies backend JWTs
  and applies single-owner resource scoping for threads, assistants, runs, and
  crons.
- `http.cors.allow_origins` must include the frontend origin that will open the
  streaming connection.

Operational notes:

- Use `LANGGRAPH_AUTH_DISABLED=true` only for local `langgraph dev` sessions
  where Studio or a developer tool is connecting without a token.
- Keep `LANGGRAPH_AUTH_DISABLED` unset or `false` in every deployed
  environment.
- Update `langgraph.json` / `langgraph.example.json` CORS allow-lists when the
  frontend is deployed to a non-localhost origin.

Per-agent routing is controlled by `execution_mode` in `llm_config.yaml`:

```yaml
agents:
  my_agent:
    primary: 'gpt-5-mini'
    execution_mode: 'remote' # 'local' | 'remote'
```

### YAML-driven Agent Server Graphs

The Agent Server bootstrap path can now be driven entirely from `llm_config.yaml`.

- `tool_providers:` declaratively registers LangChain tool-provider classes for the Agent Server using `class_path: 'module.path:ClassName'`.
- `enabled: false` skips a provider without deleting its YAML entry.
- `kwargs:` passes constructor arguments into the provider instance.
- `server_graph:` controls whether an agent is exposed by `agent_graphs.py`.
  Omit it to use the default rule: build only when `execution_mode: 'remote'`.
- `use_memory:` controls whether the compiled Agent Server graph includes
  memory middleware. Omit it to auto-detect from the agent's `memory_*`
  hints together with `AGENT_MEMORY_ENABLED`.

Example:

```yaml
tool_providers:
  weather_agent_tools:
    class_path: 'inference_core.agents.tools.weather_provider:WeatherToolsProvider'
  default_agent_tools:
    class_path: 'inference_core.agents.tools.weather_provider:DefaultAgentToolsProvider'

agents:
  weather_agent:
    primary: 'gpt-5-mini'
    execution_mode: 'remote'
    local_tool_providers: ['weather_agent_tools']
    server_graph: true
    use_memory: false
```

Operational note: after changing which agents should be exposed remotely, run
`python scripts/sync_langgraph_json.py` to rewrite the `graphs` section in
`langgraph.json`, or `python scripts/sync_langgraph_json.py --check` in CI.

Provider-specific note: the example weather provider requires
`OPEN_WEATHER_API_KEY`; the bundled `default_agent_tools` provider also uses
`TAVILY_API_KEY` for `internet_search`.

Examples:

- Temporary in shell:

```bash
PLAYWRIGHT_TIMEOUT_ACTION=8000 PLAYWRIGHT_TIMEOUT_NAVIGATION=120000 docker compose -f docker/docker-compose.playwright-mcp.headful.yml --profile headful up
```

- Persist in `.env` (recommended):

```
PLAYWRIGHT_TIMEOUT_ACTION=8000
PLAYWRIGHT_TIMEOUT_NAVIGATION=120000
```

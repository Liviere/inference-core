# LLM Configuration — Detailed Summary

**Document purpose:** A concise, practical summary of how `llm_config.yaml` affects the application, mapping YAML sections → configuration fields in code and listing locations that read those values.

**Main load points**

- **YAML loader:** `inference_core/llm/config.py` → `LLMConfig._load_config()` (path: `repo_root/llm_config.yaml`).
- **Singleton:** `get_llm_config()` creates a global `LLMConfig` object (module-level `llm_config`) and returns it to consumers.
- **Fallback:** when the file is missing or YAML parsing fails, `_load_fallback_config()` is used to set default models and settings.

**YAML sections → what they configure**

- **`providers`**: raw provider definitions (e.g. `openai`, `custom_openai_compatible`, `gemini`, `claude`). Mapped to `LLMConfig.providers` and validated via `ProviderConfig` when needed. Used for:

  - resolving env vars for API keys (`api_key_env`),
  - supplying `base_url`,
  - indicating whether the provider requires an API key (`requires_api_key`).

- **`models`**: each named entry becomes a `ModelConfig` in `LLMConfig.models`.

  - fields include `provider`, `max_tokens`, `temperature`, optional `pricing`, etc.
  - directly affects instance creation in `inference_core/llm/models.py`.
  - missing API key / base_url affects `is_model_available()` results.

- **`tasks`**: assignments for task types (e.g. `completion`, `chat`)

  - short mapping `task_models` (primary) and full `task_configs` (type `TaskConfig`).
  - `TaskConfig.mcp_profile` points to an MCP profile used to load tools.
  - can be overridden by environment variables listed under `settings.env_overrides`.

- **`agents`**: assignments for named agents (agent-specific model choices)

  - short mapping `agent_models` (primary) and full `agent_configs` (type `AgentConfig`).
  - `AgentConfig` is a dedicated Pydantic model (separate from `TaskConfig`) used to represent agent-specific settings and can be extended independently of tasks.
  - Agent primary model can be overridden via environment variables defined in `settings.env_overrides` (same mechanism as for tasks).
  - `LLMConfig.get_agent_model()` and `LLMConfig.get_agent_model_with_fallback()` resolve the effective model for an agent, including fallback logic and availability checks (API key / base_url).
  - Agents are intentionally a separate namespace from `tasks` (no implicit name collision); code first resolves agents from `agents:` and does not mutate existing `tasks` mapping.

- **`settings`**: global behavioral settings (e.g. `enable_caching`, `cache_ttl_seconds`, `default_timeout`, `retry_attempts`, `env_overrides`, `usage_logging`, etc.)

  - mapped directly to fields on `LLMConfig` and consumed by the model factory, usage logger, retry logic, and more.

- **`batch`**: batch processing configuration → `LLMConfig.batch_config` (type `BatchConfig`).

  - used by batch tasks, Celery scheduling (`inference_core/celery/config.py`), and batch logic (`get_provider_models`, `get_model_config`).

- **`mcp`**: MCP configuration (profiles + servers) → `LLMConfig.mcp_config`.

  - server entries support env-var expansion in headers (e.g. `${MCP_PLAYWRIGHT_TOKEN:-}`),
  - in test mode (`ENVIRONMENT==testing` or `PYTEST_CURRENT_TEST`) `_load_mcp_config()` adjusts defaults to avoid relying on local servers (e.g. may disable MCP by default or normalize the Playwright URL).

- **`param_policies`**: declarative overrides for parameter policies for providers and models.

  - read dynamically by `inference_core/llm/param_policy.py` and affects `normalize_params()` — allows permitting, renaming, or dropping parameters and setting passthrough prefixes.

- **`pricing`** (inside `models.<model>.pricing`): mapped to `PricingConfig` and items like `DimensionPrice`, `ContextTier` — used by usage/pricing logic (e.g. `inference_core/llm/usage_logging.py`, if implemented).

**Additional files to review — agent support (merged)**

- `inference_core/llm/config.py`

  - extends the loader with a new Pydantic model `AgentConfig` and adds fields `LLMConfig.agent_models` and `LLMConfig.agent_configs`.
  - responsible for parsing the `agents:` section in `llm_config.yaml`, validating it, and mapping it to runtime structures (including env-overrides).
  - contains logic to load test/production default values for agents (similar to `tasks`).

- `inference_core/llm/models.py`

  - `LLMModelFactory` uses `LLMConfig.models` to create model instances for tasks and — as an extension — exposes `get_model_for_agent()` (or a similar method) to obtain the effective model for a given agent, respecting fallback logic and availability checks (API key / `base_url`).
  - respects global settings (e.g., `config.enable_caching`) and parameter policies when instantiating models.

- `inference_core/services/agents_service.py`

  - uses the factory (`LLMModelFactory.get_model_for_agent()`) to construct model/agent instances; this centralizes model-selection logic and keeps it consistent with task model selection.
  - responsible for initializing agents, applying runtime configuration, and integrating with MCP/tools where applicable.

- `inference_core/llm/tools.py` and `inference_core/llm/usage_logging.py`

  - consume `providers`, `pricing`, and `models` (including agent configs) to load tools and billing/usage logic; when changing agents, review these modules for any impact on tool loading and usage logging.

- tests and CI / Docker
  - ensure unit and integration tests account for new agent behavior (e.g., parsing `agents:` in `tests/unit/llm/test_mcp_config.py` or by adding dedicated agent tests).
  - note that `docker/Dockerfile` and `docker/tests/Dockerfile` copy `llm_config.yaml` into the image — changes to `agents:` may require rebuilding the image or restarting containers if the YAML is baked in.

In short: agent configuration is loaded and validated in `config.py`; model selection for agents is implemented by the factory in `models.py`; and `services/agents_service.py` uses the factory to build and initialize agents. Also review tools, usage logging, and tests/Docker when introducing agent-related changes to keep behavior consistent.

**Notable behaviors / edge cases**

- **API keys**: if `ProviderConfig.requires_api_key` is `true`, `LLMConfig._load_config()` will attempt to read the value from the env var specified by `api_key_env`; missing key → models for that provider are treated as unavailable by `is_model_available()`.
- **Env overrides for tasks**: `settings.env_overrides` allows overriding `tasks.<task>.primary` via environment variables — useful for quick switches without editing the YAML.
- **Test mode**: `_load_mcp_config()` contains logic to change defaults in test environments to avoid external services (e.g. disable MCP, normalize Playwright URL), making local tests easier.
- **Fallback**: when `llm_config.yaml` is absent or unparsable, the app will fall back to default models (e.g. `gpt-5-mini`) and baseline settings — the app remains usable but limited.
- **Param policy**: `param_policies` lets you introduce new rules without code changes: add `patch`/`replace` for providers or models and the system applies them dynamically in `param_policy`.

**Key files for quick review**

- `inference_core/llm/config.py` — main YAML loader and structures.
- `inference_core/services/llm_service.py` — configuration integration with LLM runtime.
- `inference_core/llm/models.py` — model instantiation.
- `inference_core/llm/param_policy.py` — parameter policies and override mechanism.
- `inference_core/celery/config.py` — `batch` usage for Celery beat schedule.
- `llm_config.yaml` — current repository configuration (edit carefully in production).
- `docker/Dockerfile`, `docker/tests/Dockerfile` — indicate whether the YAML is baked into the image.

**Operational notes**

- Changes to `llm_config.yaml` only take effect after restarting the process / re-instantiating the `LLMConfig` singleton (i.e. restarting the server or reloading the module).
- In production, ensure all `api_key_env` variables are set in container/host environment.

Additional agent-specific notes:

- `AgentConfig` is a dedicated configuration model for agents and is intentionally separated from `TaskConfig` — this keeps agent settings independent and easier to extend in the future (e.g., checkpointing, subagents, agent-specific runtime flags).
- Agents live in their own namespace under `agents:` in the YAML; the loader populates `LLMConfig.agent_models` and `LLMConfig.agent_configs`. Agent resolution uses `LLMConfig.get_agent_model()` and `get_agent_model_with_fallback()` and does not override or mutate `tasks` mappings.
- Remember to restart the application when you change `agents:` in `llm_config.yaml` (same restart requirement as for `tasks`).

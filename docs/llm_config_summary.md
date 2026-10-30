# LLM Configuration — Detailed Summary

**Document purpose:** A concise, practical summary of how `llm_config.yaml` affects the application, mapping YAML sections → configuration fields in code and listing locations that read those values.

**Main load points**

- **YAML loader:** `inference_core/llm/config.py` → `LLMConfig._load_config()` (path: `repo_root/llm_config.yaml`).
- **Singleton:** `get_llm_config()` creates a global `LLMConfig` object (module-level `llm_config`) and returns it to consumers.
- **Dynamic Resolution:** `LLMConfigService` merges YAML config with database-backed overrides (Admin/User). See [Dynamic Configuration](dynamic-configuration.md) for details.
- **Fallback:** when the file is missing or YAML parsing fails, `_load_fallback_config()` is used to set default models and settings.

**Regional endpoints**

Some providers process a request in a given region only on a regional endpoint, and price it differently. For Mistral, `https://api.eu.mistral.ai/v1` (EU/EFTA) and `https://api.us.mistral.ai/v1` cost 1.1× the list price and serve only the models hosted in the region, while the global `https://api.mistral.ai/v1` does not commit to a location. To use one:

- `base_url` under the provider sends every call of its models there: chat models and the model catalog's listing alike.
- `price_multiplier: 1.1` under the provider keeps the costs right while `pricing` stays the list price.
- `base_url` and `price_multiplier` on a model entry do the same for one model, next to the provider's global endpoint.
- Regional endpoints may lack APIs the global one has (for Mistral: Batch, Files and Agents). inference-core's batch providers are OpenAI, Claude and Gemini, and it calls no files or agents API of a model provider.

**YAML sections → what they configure**

- **`providers`**: raw provider definitions (e.g. `openai`, `custom_openai_compatible`, `gemini`, `claude`). Mapped to `LLMConfig.providers` and validated via `ProviderConfig` when needed. Used for:
  - resolving env vars for API keys (`api_key_env`),
  - supplying `base_url`,
  - indicating whether the provider requires an API key (`requires_api_key`).
  - `price_multiplier` (default 1.0): what the provider charges against the list prices written in its models' `pricing`, for example 1.1 for an endpoint that costs 10% more. A model's pricing is scaled by it when the config is loaded, so every cost computed from it (usage logs, cost tracking, pricing snapshots) includes it.
  - `request_options` (default empty): fields added to the body of every request to the provider, for options outside the OpenAI schema such as a gateway's routing rules. Read by `cortecs` (chat models, a user's own key on them, and embeddings); other providers do not read it.
  - `locked_request_options` (default empty): keys of `request_options` that a model or embeddings entry may not change. An entry that sets one of them to another value than the provider's is refused when the config is loaded (a model entry fails the load like an invalid `price_multiplier`; an embeddings entry is skipped with a warning), and the error names the entry and the key, never a value. For routing rules that must hold for every request, such as `[eu_native, allow_zero_data_retention]` on a gateway.
- **`cortecs`** is the Cortecs gateway (OpenAI's Chat Completions and `/embeddings`, models hosted by European clouds). Its models are `ChatOpenAI` clients on the provider's `base_url` (required: there is no default address) and key, with usage asked for in streamed responses. The catalog lists its models with their kind, context window and capabilities; its prices are in EUR and are not kept.

- **`models`**: each named entry becomes a `ModelConfig` in `LLMConfig.models`.
  - fields include `provider`, `max_tokens`, `temperature`, optional `pricing`, etc.
  - `base_url` on a model entry overrides its provider's, so one provider can serve a model from another endpoint (for example a regional one). The entry's key is still the model's name at the provider, so a model served from two endpoints needs two names for it (a dated version and an alias, say).
  - `price_multiplier` on a model entry overrides its provider's, for a model on an endpoint that is priced differently from the provider's.
  - `request_options` (unset by default) on a model entry: fields added on top of its provider's `request_options`, winning over them key by key (a list the entry names replaces the provider's). For a rule that holds for this model only, such as `{allowed_providers: [ionos]}` to pin the gateway's host for it. Keys the provider locks cannot be changed. Read by `cortecs`; never passed to the model's client as a parameter.
  - `context_window` (unset by default) is how many tokens of input the model takes, as its provider states it. It is never sent to the provider: it is for an application that shows how much of the window a conversation fills or keeps it within the window.
  - `provider_model_id` (unset by default) is the id the provider knows the model by when it is not the entry's key. A gateway serves other providers' models under their names, so a model configured both directly and through the gateway needs two keys (`cortecs-mistral-small-2603` with `provider_model_id: mistral-small-2603`). Read by `cortecs` and by the model catalog, which looks the model up under it; the key stays the model's name everywhere else (agents, usage logs).
  - `multimodal: true|false` declares whether the model can accept multimodal inputs. Tool loading uses this flag to decide whether tools that declare `requires_multimodal = True` are exposed directly, skipped, or retained for delegation.
  - **Extra & Nested Parameters:** supports arbitrary extra fields from YAML (including nested dictionaries). These are preserved and forwarded to LLM providers via `kwargs`. Useful for provider-specific parameters like `logit_bias`, `response_format`, or Ollama `options`.
  - `reasoning_config` can store provider-specific nested kwargs for reasoning / thinking models (for example OpenAI reasoning, Claude thinking, or Gemini thought settings).
  - Claude thinking config is normalized in `LLMModelFactory`: `temperature` is removed when `thinking` is enabled, and `max_tokens` is increased if it is not greater than the thinking budget.
  - when an agent enables reasoning output, the factory merges `reasoning_config` into the provider kwargs before model construction.
  - `thinking_block_binding: drop_block | error` (Claude only, unset by default) says what the API does with a thinking block whose conversation changed since the block was produced.
    - Claude Fable 5.1 and Opus, Sonnet and Haiku 5.5 bind each thinking block to everything before it: the system prompt, the tools and every earlier message. A request that replays a block after one of them changed is refused with a 400 on accounts created on or after 2026-08-31, and on older accounts only when the request sets the field.
    - A summary of the history, cleared tool results or a system prompt rebuilt for each turn change it.
    - `drop_block` has the API drop the first such block and every later one, unbilled, and answer. `error` asks for the 400 on any account, which is how to find such changes on an older account.
    - The factory sends it as `thinking.block_binding.prefix_mismatch_behavior` with the `thinking-binding-controls-2026-08-01` beta, on every call of the model, with or without reasoning output. Without reasoning output the thinking sent is `{type: adaptive}`, which these models use when `thinking` is left out. A thinking type that does not take it (`between_tools`) is left alone with a warning.
    - Set it only on models that take adaptive thinking: on a model that does not (Haiku 4.5, say), `{type: adaptive}` is refused.
  - `prompt_cache: 5m | 1h` (Claude only, unset by default) turns on Anthropic's automatic prompt caching: every request of the model carries a top-level `cache_control` (`{type: ephemeral, ttl: …}`, through the model's `model_kwargs`), and the API caches the prompt up to its last block. The next request reads the unchanged part at the cache-read price; a write costs 1.25× input for `5m`, 2× for `1h`. Price the writes and reads with `extras` and `key_aliases` (see the Claude entries of `llm_config.example.yaml`). The cache is a prefix match: anything that changes earlier in the prompt (the system prompt, the tools, an earlier message) writes it again from there.
  - `context_management` (Claude only, unset by default) is Anthropic's context editing, sent as the request's `context_management` with the `context-management-2025-06-27` beta on every call of the model, e.g. `{edits: [{type: clear_tool_uses_20250919, trigger: {type: input_tokens, value: 30000}, keep: {type: tool_uses, value: 3}, clear_at_least: {type: input_tokens, value: 5000}}]}`.
    - The API clears old tool results from what the model reads; the conversation sent stays as it was. So the prefix a thinking block is bound to does not change (server-side edits are not edits of the history), where the same clearing done before the request does.
    - Each clearing still writes the cached prompt again from the cleared result on: `clear_at_least` keeps clearings few and large.
    - The model object carries it (`ChatAnthropic.context_management`), so an application that also clears tool results itself can leave such a model's requests alone.
    - A streamed response says how many tokens it wrote to the cache but not for how long (`cache_creation_tokens`), and the factory streams every call: with `1h`, map that key to the hour's price (`cache_creation_tokens: cache_write_1h_tokens`), or the writes are logged at 1.25× input. A model priced otherwise is named in a warning when the file is read (see `observability/llm-usage-logging.md`).
    - One model can be built with another choice than its entry's: `LLMModelFactory.create_model(name, prompt_cache=False)` builds it without the cache, `"5m"` or `"1h"` with that lifetime; left out (or `None`), the entry decides. The factory keeps the two apart (another key in its cache of models). `without_prompt_cache(model)` gives a copy of a model already built that sends no `cache_control`, the same model otherwise (its notes, `metadata`, included).
    - A call whose prompt is different every time (an analysis of a run, a summary, an extraction) only writes the cache and never reads it, and a write costs more than plain input. Such a call is better made without it.
  - Another provider's model with `prompt_cache` or `context_management` is built without them, with a warning.
  - `prompt_cache_routing: true` (unset by default; OpenAI, Mistral, Fireworks, DeepInfra, xAI) has an agent's requests of one conversation carry a key that sends them to the server holding the provider's automatic prompt cache.
    - The key is `prompt_cache_key` in the body, or the `x-grok-conv-id` header for xAI's Chat Completions. It is a hash of the agent's thread and the model's name (`routing_key`), so nothing of the conversation or the person goes with it.
    - `PromptCacheRoutingMiddleware` (`inference_core/agents/middleware/prompt_cache_routing.py`) puts it on each model call. `AgentService` adds it only when a model of the config asks for it, after the fallback middleware and before cost tracking, in the agent's list and in each subagent's: it sees the model each attempt is made on, so a fallback of another provider gets its own key or none.
    - Calls outside an agent (a side job) and calls outside a thread get none. A model of another provider (Claude, Gemini, Cortecs) gets none, with one warning.
    - Measured on two calls of a 5,000-token prompt: xAI read 5,120 cached tokens with the key and 128 without, Fireworks 5,227 and 123. OpenAI, Mistral and DeepInfra read theirs either way; OpenAI asks for the key from GPT-5.6 on.
    - Price what the provider reports with `extras.cache_read_tokens`: LangChain names OpenAI-style `prompt_tokens_details.cached_tokens` `cache_read`. langchain-mistralai passes no cached tokens on, so a Mistral entry is priced as if nothing was read from the cache.
  - An inference-core version that does not know these two fields passes them on to the model as extra parameters (see below), and the request is then refused: add them to the file only once the code that reads them runs.
  - directly affects instance creation in `inference_core/llm/models.py`.
  - missing API key / base_url affects `is_model_available()` results.

- **`agents`**: assignments for named agents (agent-specific model choices)
  - short mapping `agent_models` (primary) and full `agent_configs` (type `AgentConfig`).
  - `AgentConfig` is the dedicated Pydantic model used to represent agent-specific settings.
  - `user_selectable: false` marks an agent as internal-only; the runtime keeps it in the resolved registry, but user-facing template lists and instance creation/update validation will reject it.
  - `reasoning_output: true` enables reasoning output for that agent and forwards the model's `reasoning_config` into the instantiated provider kwargs.
  - `on_missing_capability` controls what happens when a tool requires a capability the primary model lacks. `skip` filters the tool out; `delegate` keeps it exposed so runtime middleware can swap to a support model for that tool call.
  - `multimodal_support_model` names the model used for delegated multimodal tool calls. It must resolve to an entry in `models:` and should declare `multimodal: true`.
  - `LLMConfig.get_agent_model()` and `LLMConfig.get_agent_model_with_fallback()` resolve the effective model for an agent, including fallback logic and availability checks (API key / base_url).
  - New fields `skills` (list of paths) and `subagents` (list of agent names) enable delegating tasks and using specialized workflows.
  - `tool_call_limits` configures `ToolCallLimitMiddleware` instances for the agent: `global_limit` applies to all tools, while `per_tool` adds stricter limits for named tools.
  - `run_limit` resets for each invocation; `thread_limit` persists across the conversation thread.
  - When `tool_call_limits` is present, both local and remote agent builders append matching policy instructions to the system prompt so the model can recover gracefully after a blocked tool call.

- **`settings`**: global behavioral settings (e.g. `enable_caching`, `cache_ttl_seconds`, `default_timeout`, `retry_attempts`, `usage_logging`, etc.)
  - mapped directly to fields on `LLMConfig` and consumed by the model factory, usage logger, retry logic, and more.

- **`batch`**: batch processing configuration → `LLMConfig.batch_config` (type `BatchConfig`).
  - used by batch tasks, Celery scheduling (`inference_core/celery/config.py`), and batch logic (`get_provider_models`, `get_model_config`).

- **`mcp`**: MCP configuration (profiles + servers) → `LLMConfig.mcp_config`.
  - server entries support env-var expansion in headers (e.g. `${MCP_PLAYWRIGHT_TOKEN:-}`),
  - in test mode (`ENVIRONMENT==testing` or `PYTEST_CURRENT_TEST`) `_load_mcp_config()` adjusts defaults to avoid relying on local servers (e.g. may disable MCP by default or normalize the Playwright URL).

- **`param_policies`**: declarative overrides for parameter policies for providers and models.
  - read dynamically by `inference_core/llm/param_policy.py` and affects `normalize_params()` — allows permitting, renaming, or dropping parameters and setting passthrough prefixes.

- **`pricing`** (inside `models.<model>.pricing`): mapped to `PricingConfig` and items like `DimensionPrice`, `ContextTier` — used by usage/pricing logic (e.g. `inference_core/llm/usage_logging.py`, if implemented). `DimensionPrice` accepts either `cost_per_1m` or the legacy `cost_per_1k`; the loader normalizes both to an internal per-1K representation.

**Additional files to review — agent support (merged)**

- `inference_core/llm/config.py`
  - extends the loader with a new Pydantic model `AgentConfig` and adds fields `LLMConfig.agent_models` and `LLMConfig.agent_configs`.
  - responsible for parsing the `agents:` section in `llm_config.yaml`, validating it, and mapping it to runtime structures.
  - contains logic to load test/production default values for agents.

- `inference_core/llm/models.py`
  - `LLMModelFactory` uses `LLMConfig.models` to create model instances and exposes `get_model_for_agent()` to obtain the effective model for a given agent, respecting fallback logic and availability checks (API key / `base_url`).
  - respects global settings (e.g., `config.enable_caching`) and parameter policies when instantiating models.

- `inference_core/services/agents_service.py`
  - uses the factory (`LLMModelFactory.get_model_for_agent()`) to construct model/agent instances; this centralizes model-selection logic for the v1 runtime.
  - responsible for initializing agents, applying runtime configuration, and integrating with MCP/tools where applicable.
  - synthesizes tool-model overrides for multimodal tools retained under `on_missing_capability: delegate`, while preserving explicit YAML overrides when both target the same tool name.
  - appends `tool_call_limits` prompt instructions and injects tool-call limit middleware in local agent and deep-agent execution paths.

- `inference_core/agents/middleware/tool_call_limits.py`
  - shared factory for building `ToolCallLimitMiddleware` instances from YAML config and for generating the companion system-prompt instructions shown to the model.

- `inference_core/llm/tools.py` and `inference_core/llm/usage_logging.py`
  - consume `providers`, `pricing`, and `models` (including agent configs) to load tools and billing/usage logic; the tool loader also filters or retains multimodal-only tools based on the active model capability and delegation strategy.

- tests and CI / Docker
  - ensure unit and integration tests account for new agent behavior (e.g., parsing `agents:` in `tests/unit/llm/test_mcp_config.py` or by adding dedicated agent tests).
  - note that `docker/Dockerfile` and `docker/tests/Dockerfile` copy `llm_config.yaml` into the image — changes to `agents:` may require rebuilding the image or restarting containers if the YAML is baked in.

In short: agent configuration is loaded and validated in `config.py`; model selection for agents is implemented by the factory in `models.py`; and `services/agents_service.py` uses the factory to build and initialize agents. Also review tools, usage logging, and tests/Docker when introducing agent-related changes to keep behavior consistent.

**Notable behaviors / edge cases**

- **API keys**: if `ProviderConfig.requires_api_key` is `true`, `LLMConfig._load_config()` will attempt to read the value from the env var specified by `api_key_env`; missing key → models for that provider are treated as unavailable by `is_model_available()`.
- **Test mode**: `_load_mcp_config()` contains logic to change defaults in test environments to avoid external services (e.g. disable MCP, normalize Playwright URL), making local tests easier.
- **Fallback**: when `llm_config.yaml` is absent or unparsable, the app will fall back to default models (e.g. `gpt-5-mini`) and baseline settings — the app remains usable but limited.
- **Param policy**: `param_policies` lets you introduce new rules without code changes: add `patch`/`replace` for providers or models and the system applies them dynamically in `param_policy`.
- **Capability-aware delegation**: when `on_missing_capability: delegate` is set, the loader keeps multimodal-only tools visible even if the primary model is text-only, and `AgentService` injects synthetic tool-model overrides that route those calls to `multimodal_support_model`.
- **Startup validation is soft**: if `multimodal_support_model` is missing from `models:` or is not marked `multimodal: true`, config loading emits warnings rather than failing startup.

**Key files for quick review**

- `inference_core/llm/config.py` — main YAML loader and structures.
- `inference_core/agents/middleware/tool_call_limits.py` — shared tool-call limit builder and prompt policy generator.
- `inference_core/services/llm_service.py` — configuration integration with LLM runtime.
- `inference_core/llm/models.py` — model instantiation.
- `inference_core/llm/chat_deepinfra.py` — `ChatDeepInfraReasoning`, a drop-in
  replacement for the upstream `ChatDeepInfra` that fixes streaming tool-call
  parsing (the upstream adapter loses tool-call arguments when streaming) and
  forwards `reasoning_effort` from `reasoning_config` in the YAML. Also
  extracts inline ` thinking…  response` reasoning blocks from model content
  into `additional_kwargs["reasoning_content"]` for proper UI rendering.
- `inference_core/llm/think_tags.py` — `ThinkTagStreamRouter`, an incremental
  state machine that splits streamed token content into reasoning and visible
  text across chunk boundaries, used by `ChatDeepInfraReasoning` on the
  streaming path.
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
- `tool_call_limits` is enforced in both the local `AgentService` path and the remote Agent Server graph builder; deep-agent subagents also apply their own configured limits.
- Remember to restart the application when you change `agents:` in `llm_config.yaml` (same restart requirement as for `tasks`).

# LLM Module

This module provides a dedicated API and service layer for working with Large Language Models (LLMs). It is built on top of LangChain and integrates with the app via FastAPI and Celery.

## Components

- `prompts.py`: Prompt templates for various tasks (e.g., explain, conversation)
- `chains.py`: LangChain chains (includes multi-turn conversation with message history)
- `models.py`: Model factory (OpenAI and OpenAI-compatible providers)
- `config.py`: Loads `llm_config.yaml` with providers/models/tasks mapping
- `param_policy.py`: Centralized parameter normalization and validation for all providers
- `../services/llm_service.py`: High-level service interface for LLM operations

## Features

- Explain task: generates explanations for questions
- Conversation task: multi-turn chat with session-based message history
- RunnableWithMessageHistory pattern for robust conversation state
- SQL-backed chat history by default using SQLChatMessageHistory (can be swapped for Redis/other DB backends)
- **Centralized Parameter Normalization**: Automatic parameter filtering and mapping for all providers
- **Real-Time Token Streaming**: SSE endpoints for conversation & explain using LangChain `astream_events` with graceful fallback

## Streaming Architecture

File: `inference_core/llm/streaming.py`

### Pipeline:

1. API endpoint (`inference_core/api/v1/routes/llm.py`) receives POST and constructs an async generator.
2. Model is created with `streaming=True` and callback handler.
3. Preferred streaming path uses `model.astream_events(..., version="v1")` (LangChain 0.3.x) to capture granular events (`on_chat_model_stream`).
4. Each token (content delta) is pushed into an asyncio queue as `StreamChunk(type="token")`.
5. The generator emits SSE frames (`data: {...}\n\n`).
6. Usage metadata (if available) emitted as a `usage` event before `end`.
7. Conversation: final assistant message persisted to SQL history.

### Event JSON structure:

```
{"event":"start","model":"<model>","session_id":"<id?>"}
{"event":"token","content":"partial"}
{"event":"usage","usage":{"input_tokens":N,"output_tokens":M,"total_tokens":T}}
{"event":"end"}
```

### Fallback: If `astream_events` unsupported or errors, code falls back to `model.astream()` and manually extracts chunk content.

Manual Tester (DEV only): Served at `/static/stream.html` when `DEBUG=True`. Provides a unified UI with mode switch (conversation / explain), live output, abort, and event log.

### Client Integration Tips:

- Use `fetch` with ReadableStream (POST body) instead of `EventSource`.
- Split stream buffer on double newline `\n\n`; parse lines starting with `data:`.
- Treat `end` as authoritative completion; network close alone might be premature.
- Accumulate only `event == "token"` for final text body.

### Error Handling:

- `error` event includes `message`.
- Generator ensures an `end` event after an `error`.
- If client disconnects mid-stream, task is cancelled; no history persistence for partial reply.

### Local Test Page

When `DEBUG=True` a minimal manual QA page is served at: `http://localhost:8000/static/stream.html`.

### cURL Examples

Conversation streaming:

```bash
curl -N -H 'Content-Type: application/json' \
  -d '{"user_input":"Hello!","session_id":"demo-1"}' \
  http://localhost:8000/api/v1/llm/conversation/stream
```

Explain streaming:

```bash
curl -N -H 'Content-Type: application/json' \
  -d '{"question":"Explain FastAPI in one sentence"}' \
  http://localhost:8000/api/v1/llm/explain/stream
```

## Parameter Normalization

The LLM module includes a centralized parameter normalization system that prevents runtime errors from provider-specific parameter incompatibilities.

### How It Works

The `param_policy.py` module defines parameter policies for each provider:

- **Allowed parameters**: Parameters the provider accepts
- **Renamed parameters**: Parameter mappings (e.g., `max_tokens` → `max_output_tokens` for Gemini)
- **Dropped parameters**: Parameters that should be silently removed (e.g., `frequency_penalty` for Claude)

### Provider-Specific Behavior

| Provider      | Allowed Parameters                                                                               | Parameter Mappings                 | Dropped Parameters                                         |
| ------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------- | ---------------------------------------------------------- |
| OpenAI        | `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, `request_timeout` | None                               | None                                                       |
| Custom OpenAI | Same as OpenAI                                                                                   | None                               | None                                                       |
| Gemini        | `temperature`, `max_output_tokens`, `top_p`                                                      | `max_tokens` → `max_output_tokens` | `frequency_penalty`, `presence_penalty`, `request_timeout` |
| Claude        | `temperature`, `max_tokens`, `top_p`, `timeout`                                                  | `request_timeout` → `timeout`      | `frequency_penalty`, `presence_penalty`                    |

### Debug Logging

When parameters are renamed or dropped, debug-level log messages are emitted:

```
DEBUG: Parameter renamed for gemini: max_tokens -> max_output_tokens
DEBUG: Parameter dropped for claude: frequency_penalty (value: 0.1)
```

### Usage

Parameter normalization happens automatically in the `LLMModelFactory`. No changes are needed in your application code - the system will:

1. Accept all standard LLM parameters from your service calls
2. Automatically filter/map them based on the target provider
3. Log any transformations at debug level
4. Pass only compatible parameters to the underlying SDK

This prevents errors like `TypeError: unsupported parameter 'frequency_penalty'` when using Claude or similar provider-specific incompatibilities.

## Configuration

Use `llm_config.yaml` (see `llm_config.example.yaml` as a reference) to:

- Define providers and their `api_key_env`
- List available models and default params
- Map tasks (e.g., `explain`, `conversation`) to preferred models and fallbacks
- Control caching, timeouts, and monitoring

You can override task model selection via environment variables:

- `LLM_EXPLAIN_MODEL`
- `LLM_CONVERSATION_MODEL`

Provider API keys (examples, depending on your providers):

- `OPENAI_API_KEY`
- `CUSTOM_LLM_API_KEY`
- `DEEPINFRA_API_KEY`

### LLM Parameter Policies & Dynamic Parameters

The LLM layer includes a centralized, configurable parameter normalization system allowing you to:

- Maintain per-provider base policies (allowed / renamed / dropped parameters)
- Apply YAML-driven overrides per provider or per model
- Introduce entirely new parameters without code changes (allowed or via experimental prefixes)
- Perform hard replaces (replace) or additive (patch) modifications
- Enforce deprecation of legacy parameters (e.g., GPT‑5 family no longer accepts temperature/top_p/etc.)

Configuration lives in `llm_config.yaml` under the `param_policies` section:

```yaml
param_policies:
  settings:
    # Any parameter starting with these prefixes is passed through even if unknown
    passthrough_prefixes: ['x_', 'ext_']
  providers:
    openai:
      patch:
        allowed: ['logit_bias']
  models:
    gpt-5:
      replace: # Fully replace base policy for this model
        allowed: ['reasoning_effort', 'verbosity']
        dropped:
          [
            'temperature',
            'top_p',
            'frequency_penalty',
            'presence_penalty',
            'max_tokens',
            'request_timeout',
          ]
```

Semantics:

- `patch`: merge into existing sets/maps (additive)
- `replace`: overwrite the entire collection(s) provided
- `allowed`: parameters forwarded as-is (after rename mapping if present)
- `renamed`: old_name -> new_name mapping (handled before allowed check)
- `dropped`: always removed silently
- `passthrough_prefixes`: wildcard allow-list for experimental parameters (e.g. `x_reasoning_graph_depth`)

Model-Level Overrides:
The system merges (in order): base provider policy → provider overrides → model override. A model override using `replace` can completely discard legacy parameters.

GPT‑5 Breaking Change Example:
The GPT‑5 models in the example config remove classic sampling parameters and introduce `reasoning_effort` + `verbosity`. Requests supplying deprecated parameters for a `gpt-5*` model result in a validation error at service layer.

HTTP Request Example (Explain):

```json
POST /api/v1/llm/explain
{
  "question": "Explain attention in transformers",
  "model_name": "gpt-5",
  "reasoning_effort": "high",
  "verbosity": "high"
}
```

Experimental Parameter Example:

```json
{
	"model_name": "gpt-5",
	"x_trace_id": "123e4567",
	"x_reasoning_graph_depth": 4
}
```

If the prefixes are listed in `passthrough_prefixes`, both parameters are forwarded to the underlying model SDK (subject to provider acceptance).

Debugging Policies:
When `DEBUG=True`, you can inspect effective policies:

```
GET /api/v1/llm/param-policy/openai              # Provider policy
GET /api/v1/llm/param-policy/openai?model=gpt-5  # Effective merged model policy
```

Migration Tips:

1. Start new parameters behind a passthrough prefix.
2. Once stable, move them into `allowed` (remove prefix usage).
3. When deprecating old params: add them to `dropped` OR use `replace` to exclude them entirely.
4. Add tests asserting normalization if behavior is critical.

Error Handling:

- Unknown parameter without an allowed prefix → dropped with a warning log.
- Deprecated legacy param explicitly sent to a GPT‑5 model → raises `ValueError` (mapped to 500 by default; you can adjust to 400 in the router if desired).

Where Logic Lives:

- Policy merging / normalization: `inference_core/llm/param_policy.py`
- Factory applying normalization: `inference_core/llm/models.py`
- Service-level deprecation guard (GPT‑5 legacy sampling): `inference_core/services/llm_service.py`

To extend further (ideas):

- Add value constraints (ranges/enums) in YAML for validation
- Add policy introspection endpoint listing effective differences vs. base
- Introduce `defaults` section to auto-inject values when caller omits them

## Usage (API)

- Submit explain requests via the dedicated API endpoint (async via Celery).
- Submit conversation turns via the conversation endpoint, passing a `session_id` to continue a chat or omitting it to create a new session.

Refer to the app-level API router under `inference_core/api/v1/routes/llm.py` for the exact route paths and request/response schemas.

## Notes

- Conversation memory is persisted in your configured SQL database via SQLChatMessageHistory (persists across processes).
- If you use Postgres/MySQL, ensure the sync DB drivers are installed (psycopg for Postgres, PyMySQL for MySQL); these are included in pyproject.toml.
- Ensure the appropriate provider API keys are configured in your environment.
- For local development, use the example configuration files and `.env.example` as a starting point.

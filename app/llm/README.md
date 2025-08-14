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

## Parameter Normalization

The LLM module includes a centralized parameter normalization system that prevents runtime errors from provider-specific parameter incompatibilities.

### How It Works

The `param_policy.py` module defines parameter policies for each provider:

- **Allowed parameters**: Parameters the provider accepts
- **Renamed parameters**: Parameter mappings (e.g., `max_tokens` → `max_output_tokens` for Gemini)
- **Dropped parameters**: Parameters that should be silently removed (e.g., `frequency_penalty` for Claude)

### Provider-Specific Behavior

| Provider | Allowed Parameters | Parameter Mappings | Dropped Parameters |
|----------|-------------------|-------------------|-------------------|
| OpenAI | `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, `request_timeout` | None | None |
| Custom OpenAI | Same as OpenAI | None | None |
| Gemini | `temperature`, `max_output_tokens`, `top_p` | `max_tokens` → `max_output_tokens` | `frequency_penalty`, `presence_penalty`, `request_timeout` |
| Claude | `temperature`, `max_tokens`, `top_p`, `timeout` | `request_timeout` → `timeout` | `frequency_penalty`, `presence_penalty` |

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

## Usage (API)

- Submit explain requests via the dedicated API endpoint (async via Celery).
- Submit conversation turns via the conversation endpoint, passing a `session_id` to continue a chat or omitting it to create a new session.

Refer to the app-level API router under `app/api/v1/routes/llm.py` for the exact route paths and request/response schemas.

## Notes

- Conversation memory is persisted in your configured SQL database via SQLChatMessageHistory (persists across processes).
- If you use Postgres/MySQL, ensure the sync DB drivers are installed (psycopg for Postgres, PyMySQL for MySQL); these are included in pyproject.toml.
- Ensure the appropriate provider API keys are configured in your environment.
- For local development, use the example configuration files and `.env.example` as a starting point.

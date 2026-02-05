# Dynamic LLM Configuration

Inference Core supports a layered, dynamic configuration system that combines static YAML definitions, administrative database overrides, and user-specific preferences.

## Configuration Resolution Order

When a request is made, the effective configuration is resolved in the following order (later steps override earlier ones):

1.  **YAML Base (`llm_config.yaml`)**: The default configuration for all models, tasks, and providers.
2.  **Environment Variables**: Temporary overrides via `LLM_COMPLETION_MODEL`, etc.
3.  **Admin Overrides (`llm_config_overrides` table)**: Global, model-specific, task-specific, or agent-specific overrides set by administrators.
4.  **User Preferences (`user_llm_preferences` table)**: Individual settings chosen by the user (subject to allowlist constraints), including preferred models for tasks or tool access for agents.
5.  **Runtime Request Parameters**: Parameters passed directly in the API request (e.g., `temperature` in the request body).

## Key Components

### 1. Database models

- **`LLMConfigOverride`**: Global or scoped overrides managed by admins. Supported scopes: `global`, `model`, `task`, and `agent`.
- **`UserLLMPreference`**: User-specific settings (e.g., "I want my 'research-agent' to only use web-search tools").
- **`AllowedUserOverride`**: An allowlist of which configuration keys users are permitted to change, including validation constraints (min/max for numbers, regex for strings, or select options).

### 2. Resolution logic & Caching

The `LLMConfigService` manages the resolution process. To maintain high performance, resolved configurations are cached in **Redis** with a short TTL (default: 60 seconds). Cache is automatically invalidated when a user updates their preferences or an admin modifies overrides.

## API Usage

### For Users

Registered users can manage their preferences via `/api/v1/config/preferences`.

- **List Preferences**: `GET /api/v1/config/preferences`
- **Set Preference**: `POST /api/v1/config/preferences`
- **Bulk Update**: `PUT /api/v1/config/preferences/bulk`
- **View Effective Config**: `GET /api/v1/config/resolved`
- **Available Options**: `GET /api/v1/config/available-options` (Shows what you are allowed to change, matching available models, tasks, and agents).

### For Administrators

Admins manage the system-wide overrides and the user allowlist via `/api/v1/config/admin`.

- **Global Overrides**: `POST /api/v1/config/admin/overrides` (e.g., temporarily switch all `gpt-4` tasks to `gpt-4o` during a provider outage).
- **Manage Allowlist**: `POST /api/v1/config/admin/allowed-overrides` (Add a new parameter that users can customize).

## Dynamic Agents

Agents now fully participate in the dynamic configuration system. Administrators can override the following agent properties at runtime:

- **`primary_model`**: Change the default LLM for a specific agent.
- **`allowed_tools`**: Enable or disable specific tools (e.g., "google_search") for an agent without redeploying.
- **`mcp_profile`**: Switch the Model Context Protocol profile used by an agent to grant access to different data sources.

Users can also customize these properties (if allowed in the allowlist) using the `agent_params` preference type with a key format: `agent_name.parameter_name` (e.g., `writing_assistant.allowed_tools`).

## Constraints & Validation

Constraints are enforced when a user attempts to set a preference. They use a discriminated union in the API to ensure type safety:

- **Number**: Includes `min`, `max`, and `step`.
- **String**: Includes `pattern` (regex) and `max_length`.
- **Select**: Requires `allowed_values`.
- **Boolean**: Simple toggle.

## Setup & Migration

To initialize the database tables and seed the default allowlist (temperature, max_tokens, default_model), run:

```bash
poetry run python scripts/create_llm_config_tables.py
```

## Internal Implementation Notes

- **`LLMConfig.with_overrides()`**: A helper method on the main config object that returns a deep copy with merged overrides.
- **`LLMConfigService`**: The core logic for merging layers and managing the Redis cache.
- **Dependency Injection**: Use `get_user_resolved_config` or `get_effective_model_params` in your FastAPI endpoints to automatically receive the user's customized configuration.

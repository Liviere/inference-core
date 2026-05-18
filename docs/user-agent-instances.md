# User Agent Instances

User Agent Instances allow users to create personalized configurations of base agents defined in `agents_config.yaml`. This enables users to customize behavior (e.g., system prompts, models, tools) without modifying the global configuration.

## Overview

Each user can create multiple "instances" based on the available templates. An instance acts as a personalized preset that can be selected when starting a chat session.

Key features:

- **Base Template**: Every instance starts from a base agent (e.g., `assistant_agent`) defined by administrators.
- **Custom Overrides**: Users can override the primary model, replace or append to the system prompt, and tweak other parameters.
- **User-defined Skills**: Instances can include private skill instructions that deep agents load on demand.
- **Runtime Model Fallbacks**: Users can override the template fallback chain used when the primary model call fails.
- **Default Instance**: Users can mark one instance as their default for new interactions.
- **Isolation**: Instances are private to the user who created them.
- **Template Visibility**: Only agents marked `user_selectable: true` in `llm_config.yaml` appear in the template list and can be used as `base_agent_name`.

## Configuration Options

When creating or updating an instance, the following fields are available:

| Field | Description |
|Args | |
| `instance_name` | Unique slug-like identifier (e.g., `my-coding-assistant`). Must be unique per user. |
| `display_name` | Human-readable name shown in the UI. |
| `base_agent_name` | The name of the base agent from `agents_config.yaml` to inherit from. The selected agent must be user-selectable. |
| `primary_model` | (Optional) Override the LLM model used by this agent. Must be in the allowed models list. |
| `system_prompt_override` | (Optional) Completely replace the base agent's system prompt. A consuming application may configure a runtime max length, but core defaults to no limit. |
| `system_prompt_append` | (Optional) Append text to the end of the base agent's system prompt. A consuming application may configure a runtime max length, but core defaults to no limit. |
| `skills` | (Optional) List of private skill definitions with `name`, `description`, and `content`. |
| `config_overrides` | (Optional) JSON object for advanced overrides (e.g., `temperature`, `max_tokens`, `allowed_tools`, `fallback`). |
| `is_default` | (Boolean) If true, this becomes the user's default agent. Any previous default is unset. |

If a consuming application configures prompt-length limits, the same limits are enforced across `POST /api/v1/agent-instances`, `PATCH /api/v1/agent-instances/{id}`, direct `UserAgentInstanceService` calls, and any builder/deploy flow built on top of those schemas and services. Frontends can read the current runtime values from `GET /api/v1/config/resolved` under `agent_prompt_limits`; `null` means the core runtime is currently unbounded for that field.

`inference_core` can also derive global defaults from environment variables without any runtime configuration call:

- `INFERENCE_CORE_AGENT_SYSTEM_PROMPT_OVERRIDE_MAX_LENGTH`
- `INFERENCE_CORE_AGENT_SYSTEM_PROMPT_APPEND_MAX_LENGTH`

Unset or empty values keep the corresponding field unbounded.

`config_overrides.fallback` is the canonical list of fallback model names.
The API also accepts `config_overrides.fallback_models` as a compatibility alias
and stores both names. If the field is omitted, the instance inherits the YAML
template fallback chain. If it is an empty list, fallback is explicitly disabled
for that instance. All fallback models must exist in the user's resolved
available model list.

## User-defined Skills

Instances can carry a `skills` array for deep-agent workflows. Each entry is stored as a private `SKILL.md` document for that instance and is exposed to the runtime through the dedicated `read_skill_file` tool rather than arbitrary filesystem access.

Each skill entry must provide:

- `name`: stable slug used as the skill directory name.
- `description`: short summary shown in the skill list.
- `content`: full markdown body written to the skill's `SKILL.md` file.

Validation rules enforced by the API:

- At most 20 skills per instance.
- Skill names must be unique.
- Skill names may contain lowercase letters, digits, and single hyphens only, with no leading or trailing hyphen.
- `description` is limited to 1024 characters.
- `content` is limited to 100000 characters.

Example payload fragment:

```json
{
	"skills": [
		{
			"name": "monitor-currency-rates",
			"description": "Watch public exchange-rate feeds and summarize material changes.",
			"content": "---\nname: monitor-currency-rates\ndescription: Monitor currency rates\n---\n\n# Workflow\n1. Check the configured feeds.\n2. Compare against the previous snapshot.\n3. Report only meaningful changes.\n"
		}
	]
}
```

## API Usage

### Managing Instances

- **List Templates**: `GET /api/v1/agent-instances/templates`  
  View available base agents and models to build upon.

- **List Instances**: `GET /api/v1/agent-instances`  
  View your created instances.

- **Create Instance**: `POST /api/v1/agent-instances`  
  Create a new personalized agent.

- **Update Instance**: `PATCH /api/v1/agent-instances/{id}`  
  Modify an existing instance.

- **Delete Instance**: `DELETE /api/v1/agent-instances/{id}`  
  Soft-delete an instance.

- **Get Run Bundle**: `GET /api/v1/agent-instances/{id}/run-bundle`  
  Returns the frontend handshake payload for a direct LangGraph Agent Server
  connection. Accepts optional `session_id` query param for resumable threads.

## Direct Frontend Connection

When the chat UI talks to the LangGraph Agent Server directly (for example via
`@langchain/react` `useStream`), it should first call the run-bundle endpoint.
The backend resolves the user's instance overrides and returns:

- `assistant_id` — which graph to invoke on the Agent Server.
- `agent_server_url` — base URL for the LangGraph runtime.
- `access_token` — the same bearer token received by FastAPI, echoed back so
  the frontend can reuse it for the Agent Server request.
- `config.configurable` — resolved runtime overrides such as `primary_model`,
  `fallback_models`, `system_prompt_override`, `system_prompt_append`,
  `user_id`, `session_id`, `instance_id`, and subagent settings.
- `is_remote` — whether the selected instance is currently configured to run
  through the Agent Server.

This keeps the frontend unaware of the YAML schema and guarantees that the
same override resolution path is used for both backend-triggered runs and
direct frontend streaming sessions.

## Example Workflow

1.  **Check Templates**:
    User requests `/api/v1/agent-instances/templates` to see that `research_agent` is available.

2.  **Create Custom Researcher**:
    User creates an instance named `deep-researcher` based on `research_agent` but appends specialized instructions to the system prompt:

    ```json
    {
    	"instance_name": "deep-researcher",
    	"display_name": "Deep Researcher",
    	"base_agent_name": "research_agent",
    	"system_prompt_append": "Always cite sources in APA format.",
    	"primary_model": "gpt-4"
    }
    ```

3.  **Request Run Bundle**:
    The chat UI calls `/api/v1/agent-instances/{id}/run-bundle` and receives
    the Agent Server URL, assistant identifier, echoed bearer token, and the
    resolved `config.configurable` payload for this instance.

4.  **Use in Chat**:
    The frontend opens a direct streaming session to the Agent Server using the
    bundle data. The same per-instance overrides are honoured by the remote
    middleware stack without the UI needing to reconstruct them.

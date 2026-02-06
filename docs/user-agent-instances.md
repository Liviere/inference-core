# User Agent Instances

User Agent Instances allow users to create personalized configurations of base agents defined in `agents_config.yaml`. This enables users to customize behavior (e.g., system prompts, models, tools) without modifying the global configuration.

## Overview

Each user can create multiple "instances" based on the available templates. An instance acts as a personalized preset that can be selected when starting a chat session.

Key features:

- **Base Template**: Every instance starts from a base agent (e.g., `assistant_agent`) defined by administrators.
- **Custom Overrides**: Users can override the primary model, replace or append to the system prompt, and tweak other parameters.
- **Default Instance**: Users can mark one instance as their default for new interactions.
- **Isolation**: Instances are private to the user who created them.

## Configuration Options

When creating or updating an instance, the following fields are available:

| Field | Description |
|Args | |
| `instance_name` | Unique slug-like identifier (e.g., `my-coding-assistant`). Must be unique per user. |
| `display_name` | Human-readable name shown in the UI. |
| `base_agent_name` | The name of the base agent from `agents_config.yaml` to inherit from. |
| `primary_model` | (Optional) Override the LLM model used by this agent. Must be in the allowed models list. |
| `system_prompt_override` | (Optional) Completely replace the base agent's system prompt. |
| `system_prompt_append` | (Optional) Append text to the end of the base agent's system prompt. |
| `config_overrides` | (Optional) JSON object for advanced overrides (e.g., `temperature`, `max_tokens`, `allowed_tools`). |
| `is_default` | (Boolean) If true, this becomes the user's default agent. Any previous default is unset. |

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

3.  **Use in Chat**:
    (Future Integration) When starting a chat, the user provides `agent_instance_id` instead of a generic `agent_config_name`. The system loads the base configuration and applies the user's overrides at runtime.

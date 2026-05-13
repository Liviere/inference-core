# Agent Prompts

Prompt customization now belongs to the AgentService and Agent Instances flow. The old file-based completion/chat Jinja loader was removed together with the legacy `LLMService`, chain helpers, and `/api/v1/llm/completion` endpoints.

## Current Prompt Entry Points

- YAML agent templates in `llm_config.yaml` define the base agent description and optional `system_prompt`.
- `AgentService.create_agent(system_prompt=...)` can supply a prompt for a direct programmatic run.
- User Agent Instances can store `system_prompt_override` or `system_prompt_append` in the database.
- Remote Agent Server runs receive prompt overrides through `config.configurable` so `InstanceConfigMiddleware` and `SubagentConfigMiddleware` can apply them at runtime.

## YAML Example

```yaml
agents:
  support_agent:
    primary: gpt-5-mini
    description: Support assistant for product questions.
    system_prompt: |
      You are a support assistant. Answer with concise, actionable steps.
```

## Programmatic Example

```python
from inference_core.services.agents_service import AgentService

service = AgentService(agent_name="support_agent", session_id="support-demo")
try:
    await service.create_agent(
        system_prompt="You are a support assistant. Prefer short answers."
    )
    response = await service.arun_agent_steps("How do I reset my password?")
finally:
    service.close()
```

## Agent Instance Example

```bash
curl -X POST http://localhost:8000/api/v1/agent-instances \
  -H 'Authorization: Bearer <ACCESS_TOKEN>' \
  -H 'Content-Type: application/json' \
  -d '{
    "instance_name": "support_default",
    "display_name": "Support Default",
    "base_agent_name": "support_agent",
    "system_prompt_append": "Always include the next best action."
  }'
```

Use `system_prompt_override` when the instance should replace the base prompt entirely, and `system_prompt_append` when it should extend the template prompt.

Prompt-length limits for `system_prompt_override` and `system_prompt_append` are optional configuration in `inference_core`. They can be set either by runtime code or globally via `INFERENCE_CORE_AGENT_SYSTEM_PROMPT_OVERRIDE_MAX_LENGTH` and `INFERENCE_CORE_AGENT_SYSTEM_PROMPT_APPEND_MAX_LENGTH`. The current effective values can be exposed to clients through `GET /api/v1/config/resolved` under `agent_prompt_limits`; `null` means no limit is currently enforced.

## Removed Legacy Behavior

The following prompt mechanisms are no longer supported:

- `inference_core.llm.prompts`
- `inference_core/custom_prompts/completion/*.j2`
- `inference_core/custom_prompts/chat/*.j2`
- `inference_core/custom_prompts/mcp/*.j2`
- per-call `prompt_name` arguments on completion/chat calls

Keep reusable prompt text in YAML agent config, database-backed agent instances, or application-level code that passes `system_prompt` to `AgentService.create_agent()`.

# LangChain v1 Agents

This project now ships with a first-class agent stack built on LangChain v1. The
stack is designed to live alongside the existing chain-based flows, so legacy
integrations keep working while new agent features are developed.

## Configuration

Agents are configured via `llm_config.yaml` in the `agents` section.
The configuration includes models, tools, skills, and subagents:

```yaml
# llm_config.yaml

agents:
  browser_researcher:
    primary: gpt-5
    description: 'Specialized researcher with browser access'
    local_tool_providers: [research_bundle]
    skills: ['./skills/web-research/']
    subagents: ['web_searcher']
    mcp_profile: 'web-browsing' # Optional MCP integration
```

Key concepts:

- **Skills** – on-demand instructions loaded from `SKILL.md` files; allow complex workflows without filling the context window.
- **Subagents** – pointers to other agents defined in `llm_config.yaml` that can be invoked via the `task()` tool.
- **Local Tool Providers** – reusable bundles of tools registered in the code.
- **MCP Profiles** – configuration for Model Context Protocol servers.
- **Agent Memory** – integrated long-term memory for persistence across sessions.

## Tool Providers & MCP

Local tools are registered through `inference_core.agents.tools.register_agent_tool_provider`.
A default `internet_research` provider exposes the Tavily search tool. Configured
`mcp_profiles` reuse the existing MCP manager, so RBAC/permissions remain
consistent with LLM tasks.

## API Router

New endpoints live under `/api/v1/agents`:

- `POST /api/v1/agents/chat` – invoke a configured agent with optional
  `session_id` (checkpointing/history) and `prompt_context` (Jinja vars).

Router behaviour:

1. Resolves the agent from YAML.
2. Loads local + MCP tools respecting allowlists/denylists.
3. Renders the configured prompt.
4. Builds middleware stack (YAML + automatic usage middleware).
5. Instantiates `AgentService` or `DeepAgentService` with the resolved model.

## Usage Logging

Agent runs log to a dedicated `agent_request_logs` table via the new
`AgentUsageMiddleware`. The middleware consumes LangChain's standardized usage
metadata and aggregates tokens/costs per run. `AgentUsageService` exposes
aggregated stats and recent logs similar to the existing `llm_usage_service`.

For per-step LLM accounting, `CostTrackingMiddleware` now persists each
`LLMRequestLog` inside the same model-call node that produced the response.
This avoids losing billing or token records when execution is cancelled between
the model node and later middleware hooks.

## Cancellation

`AgentService.run_agent_steps()` and `AgentService.arun_agent_steps()` accept an
optional `cancel_check` callback. The callback is evaluated after streaming
chunks and before model or tool calls. When it returns `True`, the service
raises `AgentCancelled` and stops the run.

```python
from inference_core.services._cancel import AgentCancelled
from inference_core.services.agents_service import AgentService

service = AgentService(agent_name="my_agent")
await service.create_agent()

try:
  response = await service.arun_agent_steps(
    "Summarize the latest release notes",
    cancel_check=lambda: should_stop,
  )
except AgentCancelled:
  # Partial usage logs for completed model calls are already persisted.
  response = None
```

## Agent Memory

Agents can store and recall long-term memories about users using the integrated
memory system. This enables personalization across conversations.

### Configuration

Enable memory in `.env`:

```bash
AGENT_MEMORY_ENABLED=true
VECTOR_BACKEND=qdrant  # Required: memory uses vector store
```

### Usage

Enable memory when creating an agent:

```python
from inference_core.services.agents_service import AgentService

service = AgentService(
    agent_name="my_agent",
    enable_memory=True,  # Adds memory tools + middleware
    user_id=user_uuid,   # Required for memory namespace
)
await service.create_agent()
response = service.run_agent_steps("Hello, I prefer dark mode UI")
```

### Memory Tools

When `enable_memory=True`, agents receive two tools:

- **save_memory**: Store user preferences, facts, or instructions

  ```
  save_memory("User prefers dark mode", "preference", "ui")
  ```

- **recall_memories**: Search stored memories
  ```
  recall_memories("UI preferences", "preference")
  ```

### Memory Types

- `preference` – User preferences (UI, language, communication style)
- `fact` – Facts about user (name, location, occupation)
- `instruction` – User-specific instructions
- `context` – Conversation context to remember
- `general` – General memories

### Middleware Auto-Recall

When `AGENT_MEMORY_AUTO_RECALL=true` (default), the `MemoryMiddleware`
automatically recalls relevant memories at the start of each agent run and
injects them as context. This happens transparently without agent tool calls.

### Deduplication

Set `AGENT_MEMORY_UPSERT_BY_SIMILARITY=true` to check similarity before saving.
If a memory with similarity ≥ `AGENT_MEMORY_SIMILARITY_THRESHOLD` (default 0.85)
exists, the save is skipped to avoid duplicates.

## Migration Notes

- Legacy chain-based endpoints remain untouched.
- `agents_config.yaml` is loaded automatically on startup (override path with
  `AGENTS_CONFIG_PATH`).
- When enabling checkpointing per agent, ensure the configured database is
  reachable from the API/Celery containers.

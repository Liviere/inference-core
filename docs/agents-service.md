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

Both methods also accept `graceful_cancel=True` by default. In this mode the
service injects a callback that raises `GraphInterrupt` from inside the active
LLM stream on the next token. This changes cancellation semantics in two useful
ways:

- the provider HTTP stream is closed immediately, so no extra output tokens are billed after cancellation
- the LangGraph parent run completes as a clean interrupt, so LangSmith shows the graph-level trace as success instead of failure

After the interrupt, the service drains the small number of remaining internal
LangGraph bookkeeping events before returning control. Set
`graceful_cancel=False` only if you prefer an immediate hard close and accept a
failed LangSmith trace for that run.

```python
from inference_core.services._cancel import AgentCancelled
from inference_core.services.agents_service import AgentService

service = AgentService(agent_name="my_agent")
await service.create_agent()

try:
  response = await service.arun_agent_steps(
    "Summarize the latest release notes",
    cancel_check=lambda: should_stop,
    graceful_cancel=True,
  )
except AgentCancelled:
  # Partial usage logs for completed model calls are already persisted.
  response = None
```

## Token Streaming

`AgentService.run_agent_steps()` and `AgentService.arun_agent_steps()` also
accept an optional `on_token` callback for integrations that need token-level
updates while the agent is running.

Passing `on_token` switches the internal stream mode from `"updates"` to
`["updates", "messages"]`. The callback receives `(text, meta)` where `text`
is the streamed fragment and `meta` contains:

- `type` – `text`, `reasoning`, or `tool_call`
- `node` – LangGraph node that emitted the fragment
- `agent_name` – optional sub-agent name when available

This makes it possible to stream normal answer tokens, reasoning blocks, and
partial tool-call arguments without losing the existing step/update stream.
Tokens emitted by middleware nodes are filtered out, so UI integrations only
receive the main agent/model output.

```python
from inference_core.services.agents_service import AgentService

service = AgentService(agent_name="my_agent")
await service.create_agent()


def handle_token(text: str, meta: dict[str, str]) -> None:
    print(meta.get("type"), meta.get("node"), text)


response = await service.arun_agent_steps(
    "Research the latest LangChain agent changes",
    on_token=handle_token,
)
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

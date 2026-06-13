# LangChain v1 Agents

This project now ships with a first-class agent stack built on LangChain v1. The
stack is designed to live alongside the existing chain-based flows, so legacy
integrations keep working while new agent features are developed.

## Configuration

Agents are configured via `llm_config.yaml` in the `agents` section.
The configuration includes models, runtime fallback models, local tool
providers, MCP profiles, skills, subagents, and optional tool-call limits:

```yaml
# llm_config.yaml

tool_providers:
  email_tools:
    class_path: 'inference_core.agents.tools.email_provider:EmailToolsProvider'

mcp:
  profiles:
    browser-readonly:
      servers: [playwright]
      include_tools: [browser_navigate, browser_navigate_back]

    browser-actions:
      servers: [playwright]
      include_tools: [browser_navigate, browser_navigate_back, browser_click]

agents:
  browser_workflow_coordinator:
    primary: gpt-5-mini
    fallback: [gemini-2.5-flash]
    description: 'Coordinator for browser-heavy workflows that delegates MCP browser work'
    subagents: [browser_readonly_worker, browser_action_worker]

  browser_readonly_worker:
    primary: gpt-5-mini
    description: 'Read-only browser worker backed by Playwright MCP'
    mcp_profile: browser-readonly

  browser_action_worker:
    primary: gpt-5-mini
    description: 'Mutating browser worker backed by Playwright MCP'
    mcp_profile: browser-actions

  email_reader_worker:
    primary: gpt-5-mini
    description: 'Email worker backed by the built-in email provider'
    local_tool_providers: [email_tools]
    allowed_tools:
      [
        list_email_accounts,
        read_unseen_emails,
        search_emails,
        get_email,
        summarize_email,
      ]
```

Key concepts:

- **Runtime Model Fallbacks** – `fallback` lists alternate model names used by
  LangChain `ModelFallbackMiddleware` when the active model call fails. User
  instances can override this chain through `config_overrides.fallback` or the
  compatibility alias `fallback_models`; an empty list disables inherited
  fallback for that instance.
- **Skills** – on-demand instructions loaded from `SKILL.md` files; allow complex workflows without filling the context window.
- **Subagents** – pointers to other agents defined in `llm_config.yaml` that can be invoked via the `task()` tool. In coordinator workflows, keep the parent tool surface narrow and let specialists handle raw browser/email execution.
- **Local Tool Providers** – reusable bundles of tools registered in the code. Inference Core ships `weather_agent_tools`, `default_agent_tools`, `demo_tool_calling`, and `email_tools`; application-specific providers can be registered alongside them.
- **MCP Profiles** – configuration for Model Context Protocol servers. Use these for browser-style tools instead of inventing a local browser provider name.
- **Agent Memory** – integrated long-term memory for persistence across sessions.
- **Tool-Call Limits** – per-agent policies that cap tool usage per run or
  across a whole conversation thread.

## Repeated Tool-Error Guard

`RepeatedToolErrorGuardMiddleware` is a provider-agnostic safety net that
detects when the model issues the **same** tool call with the **same** (often
empty or malformed) arguments and gets the **same** validation error back
repeatedly — burning the entire tool-call run limit without making progress.

It is applied automatically to every agent (both local and remote Agent Server
graphs). When the same erroring tool-call signature appears 3 times in a row
(default threshold), the middleware short-circuits the agent loop with an
explanatory message instead of letting the model retry the identical call yet
again.

The guard is intentionally fail-soft: any unexpected error in the scan is
swallowed and the loop proceeds normally (the existing tool-call limit remains
the backstop).

Source: `inference_core/agents/middleware/repeated_tool_error_guard.py`

## Tool-Call Limits

Agents can opt into `ToolCallLimitMiddleware` directly from `llm_config.yaml`:

```yaml
agents:
  browser_researcher:
    primary: gpt-5
    tool_call_limits:
      global_limit:
        run_limit: 30
        thread_limit: 120
        exit_behavior: continue
      per_tool:
        - tool_name: fetch_url
          run_limit: 5
```

Use this when an agent has access to expensive or loop-prone tools.

- `run_limit` resets for each new agent invocation.
- `thread_limit` persists across the whole conversation thread.
- `exit_behavior: continue` blocks the tool call but lets the model finish the
  response using the context it already has.
- `exit_behavior: error` raises instead of letting the model recover.

When limits are configured, the agent system also appends a static policy block
to the system prompt so the model knows how to behave after a limit is hit.
The same configuration is applied in local `AgentService` runs, deep-agent
subagents, and remote LangGraph Agent Server graphs.

## Tool Providers & MCP

Local providers come from the global registry in `inference_core.llm.tools`.
For Agent Server graphs, `agent_graphs.py` registers entries declared under
`tool_providers:` in `llm_config.yaml`. Local `AgentService` runs still work
best when embedding applications register providers during startup, but the
service now has a fallback that can populate the registry from `llm_config.yaml`
when none of an agent's configured `local_tool_providers` are registered yet.

Inference Core ships built-in provider classes for weather/search, demo
tool-calling, and email workflows. Browser tooling is exposed through MCP
profiles rather than a built-in local browser provider, so coordinator
examples should use `mcp_profile` for Playwright-backed workers and reserve
`local_tool_providers` for built-ins or application-defined extensions. See
`docs/mcp-integration.md` and `docs/pluggable-tool-providers.md` for the two
integration paths.

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

When you call `AgentService` from an async request handler or background
coroutine, prefer `async with AgentService(...) as service:` or call
`await service.aclose()` in a `finally` block. This ensures model, MCP, and
tool-provider clients are released as soon as the run finishes.

```python
from inference_core.services._cancel import AgentCancelled
from inference_core.services.agents_service import AgentService

async with AgentService(agent_name="my_agent") as service:
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
- `ns` – optional LangGraph subgraph namespace path for remote Agent Server v2 streams

This makes it possible to stream normal answer tokens, reasoning blocks, and
partial tool-call arguments without losing the existing step/update stream.
Tokens emitted by middleware nodes are filtered out, so UI integrations only
receive the main agent/model output.

For remote Agent Server execution, `stream_remote()` now consumes LangGraph
Platform v2 stream events (`version="v2"`) with `stream_subgraphs=True`.
When a token or step originates from a subgraph, the callback metadata carries
`ns` so UI layers can distinguish parent-agent output from subagent output.

```python
from inference_core.services.agents_service import AgentService

def handle_token(text: str, meta: dict[str, str]) -> None:
  print(meta.get("type"), meta.get("node"), text)


async with AgentService(agent_name="my_agent") as service:
  await service.create_agent()
  response = await service.arun_agent_steps(
    "Research the latest LangChain agent changes",
    on_token=handle_token,
  )
```

## Custom LangGraph Orchestration

A custom LangGraph orchestration can place the existing `AgentService` inside
your own LangGraph `StateGraph`. This is useful when you need deterministic
preparation or reporting nodes around the agent run while still reusing the
existing `AgentService` integration surface for the middle step.

This is different from the Agent Server path below:

- custom orchestration: your graph calls `AgentService` as one node in a larger workflow
- Agent Server: `agent_graphs.py` and `build_agent_graph()` compile configured
  agents for `langgraph dev` / `langgraph up`
- remote delegation: `AgentService.arun_agent_steps()` can still delegate the
  inner run when `AGENT_SERVER_ENABLED=true` and the selected agent has
  `execution_mode: remote`

Start with the no-LLM teaching path:

```bash
poetry run python examples/langgraph_agentservice_tutorial.py \
  --mock-agent \
  --prompt "Explain how this bridge works" \
  --outer-stream
```

The tutorial exposes `--trace-mode` for trace linkage control:

- `separate` keeps the inner `AgentService` trace separate from the outer graph
- `nested` reuses the outer `RunnableConfig` when parent callbacks are available
- `nested-only` refuses to create a standalone inner trace when parent trace context is missing

The tutorial also exposes `--agent-execution-mode` because nested LangSmith
traces require in-process callback propagation. Remote Agent Server execution
crosses an HTTP boundary and remains a separate root trace.

- `auto` uses the configured agent execution for `separate`, and forces local
  execution for `nested` / `nested-only`
- `configured` honors `llm_config.yaml`; nested modes fail fast if that resolves
  to remote Agent Server execution
- `local` always bypasses Agent Server for this tutorial run

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

async with AgentService(
  agent_name="my_agent",
  enable_memory=True,  # Adds memory tools + middleware
  user_id=user_uuid,   # Required for memory namespace
) as service:
  await service.create_agent()
  response = await service.arun_agent_steps("Hello, I prefer dark mode UI")
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

### Post-Run Memory Analysis

When `AGENT_MEMORY_POSTRUN_ANALYSIS_ENABLED=true` (default), the same
`MemoryMiddleware` performs a best-effort extraction pass after the run ends.
It uses a dedicated memory model resolved via the following precedence:

1. `AGENT_MEMORY_POSTRUN_ANALYSIS_MODEL` (finer-grained override)
2. `AGENT_MEMORY_MODEL` (dedicated memory model, default: `gemini-3.1-flash-lite-preview`)
3. The agent's own session model (emergency fallback)

The model is instantiated through the global model factory so provider and
parameters from `llm_config.yaml` are applied. The middleware asks the model to
inspect semantically similar memories first and then call `save_memory_store`,
`update_memory_store`, or `recall_memories_store` as needed. It executes those
tool calls and can save multiple memories in one pass when the conversation
contains several durable facts, preferences, or corrections.

### Memory LLM Cost Attribution

Memory-handling LLM calls (post-run analysis) are billed to the triggering
session via `LLMRequestLog` rows tagged `task_type="agent_memory"`. The
`session_id` and `request_id` from the parent agent run are forwarded so the
billing layer (`CreditBillingMiddleware`) can charge the session for memory
work. This is fail-soft: any billing error is logged and swallowed.

The post-run analysis runs with a dedicated system prompt so the transcript and
prefetched memories are treated as data, not instructions. This is the main
guard against prompt-injection attempts embedded in user content or stored
memories.

This keeps manual `save_memory_store` calls for explicit user-facing saves,
while still capturing completed session summaries automatically.

### Deduplication

Set `AGENT_MEMORY_UPSERT_BY_SIMILARITY=true` to check similarity before saving.
If a memory with similarity ≥ `AGENT_MEMORY_SIMILARITY_THRESHOLD` (default 0.85)
exists, the save is skipped to avoid duplicates.

The post-run analysis also prefetches semantically similar memories and asks the
model to update an existing record instead of creating a duplicate when the new
conversation content clearly extends or corrects what is already stored.
Only memories with a sufficiently strong similarity score are included in that
prefetch context, so loosely related entries do not influence deduplication.

## Remote Execution (LangGraph Agent Server)

Agents can be executed remotely via the [LangGraph Platform](https://docs.langchain.com/langsmith/agent-server) instead of running in-process. This is controlled per-agent in `llm_config.yaml`:

```yaml
agents:
  default_agent:
    primary: 'gemini-3-flash-preview'
    execution_mode: 'remote' # delegates to Agent Server
    # remote_graph_id: 'custom_id'  # optional, defaults to agent name
```

### How It Works

1. `AgentService._is_remote` checks both `AGENT_SERVER_ENABLED=true` and the agent's `execution_mode: 'remote'`.
2. `arun_agent_steps()` delegates to `_arun_agent_steps_remote()` which calls `agent_server_client.run_remote()` or `stream_remote()`.
3. The Agent Server runs the same graph (built by `graph_builder.py` from the same YAML config).
4. Results are wrapped in the standard `AgentResponse` — callers don't need to know whether execution was local or remote.

The sync `run_agent_steps()` raises `RuntimeError` for remote agents — use `arun_agent_steps()` instead.

Remote streaming uses LangGraph Platform v2 events and enables
`stream_subgraphs=True`, so `on_token` / `on_step` callbacks may receive an
`ns` field in metadata for subgraph-originated output.

### Development Setup

```bash
# Start Agent Server (lightweight, no Docker, hot reload)
poetry run langgraph dev --no-browser

# In .env
AGENT_SERVER_ENABLED=true
AGENT_SERVER_URL=http://localhost:2024
```

Graphs are exposed through `langgraph.json` → `agent_graphs.py` → `graph_registry.py` → `graph_builder.py`.

The Agent Server entry point is now YAML-driven:

- top-level `tool_providers:` entries declare provider classes to import and register before graph compilation
- each agent decides whether it is exposed by the Agent Server via `server_graph` or the default `execution_mode: 'remote'` rule
- each compiled graph decides whether memory middleware is included via `use_memory` or auto-detection from the agent's `memory_*` hints

That keeps the remote bootstrap path aligned with the same `llm_config.yaml` used by local agent execution.

When the set of exposed Agent Server graphs changes, regenerate `langgraph.json` from YAML:

```bash
# Rewrite only langgraph.json -> graphs from llm_config.yaml
python scripts/sync_langgraph_json.py

# CI / validation mode: fail when langgraph.json is out of sync
python scripts/sync_langgraph_json.py --check
```

The `tool_providers:` dictionary key must match the provider instance's `name`, because agents still reference that logical key from `local_tool_providers`.

### Architecture

```
FastAPI / Celery Task
    → AgentService.arun_agent_steps()
        → _is_remote? ─── Yes ──→ agent_server_client.run_remote()
        │                              → langgraph-sdk HTTP → Agent Server (port 2024/8123)
        └── No ──→ local LangGraph execution (create_agent + stream)
```

### Testing

- Unit tests mock `agent_server_client` — no server needed (`tests/unit/services/test_agent_server_client.py`)
- Integration tests require a running server and use `@pytest.mark.agent_server` (`tests/integration/test_agent_server.py`)
- Integration tests auto-skip when the server is not reachable

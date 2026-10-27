# CoALA Memory Architecture

> **Source**: [Cognitive Architectures for Language Agents (CoALA)](https://arxiv.org/abs/2309.02427)

## Overview

The memory system implements the CoALA (Cognitive Architectures for Language Agents) framework, organizing agent long-term memory into three cognitively-inspired categories:

| Category       | Purpose                                    | Scope     | Memory Types                          |
| -------------- | ------------------------------------------ | --------- | ------------------------------------- |
| **Semantic**   | Stable facts about the world and the user  | Shared    | preferences, facts, goals, general    |
| **Episodic**   | Past experiences and interaction sequences | Per-agent | context, session_summary, interaction |
| **Procedural** | Operational rules, workflows, skills       | Per-agent | instructions, workflow, skill         |

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    Agent Execution                        │
│                                                          │
│  ┌──────────────────┐    ┌──────────────────────────┐    │
│  │  MemoryMiddleware │───▶ format_context_for_prompt │    │
│  │  (before_agent)   │    │ (CoALA XML sections)     │    │
│  └──────────────────┘    └──────────────────────────┘    │
│                                                          │
│  ┌──────────────────┐    ┌──────────────────────────┐    │
│  │  MemoryMiddleware │───▶ postrun_analysis          │    │
│  │   (after_agent)   │    │ (save/update/recall      │    │
│  │                   │    │  memory tools)           │    │
│  └──────────────────┘    └──────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────┐        │
│  │              Memory Tools                     │        │
│  │  save_memory_store  │  recall_memories_store  │        │
│  │  update_memory_store│  delete_memory_store    │        │
│  └──────────────────────────────────────────────┘        │
└──────────────────────┬───────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │  AgentMemoryStoreService │
          │  + MemoryNamespaceBuilder│
          └────────────┬────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    ▼                  ▼                  ▼
┌─────────┐    ┌──────────┐    ┌───────────┐
│ semantic │    │ episodic │    │ procedural│
│ (shared) │    │(per-agent│    │(per-agent)│
│          │    │ or shared│    │ or shared)│
└─────────┘    └──────────┘    └───────────┘
    │                │                │
    └────────────────┼────────────────┘
                     ▼
            ┌────────────────┐
            │  LangGraph Store│
            │  (Sqlite/PG/   │
            │   InMemory)    │
            └────────────────┘
```

## Namespace Structure

Memories are stored in namespaces that encode both user isolation and CoALA category:

```
# Semantic (always shared across agents)
(user_id, "semantic")

# Episodic (per-agent when agent_name is set)
(user_id, "episodic")                    # shared
(user_id, "episodic", "browser_agent")   # per-agent

# Procedural (per-agent when agent_name is set)
(user_id, "procedural")                 # shared
(user_id, "procedural", "deep_planner") # per-agent
```

## Memory Types

### Semantic Memory (Shared)

- **preferences** — Communication style, format preferences, behavioral patterns
- **facts** — Stable factual information (location, job, relationships)
- **goals** — Objectives and desired outcomes
- **general** — Miscellaneous observations not fitting other types

### Episodic Memory (Per-Agent)

- **context** — Situational awareness, current circumstances
- **session_summary** — Condensed summary of completed conversation sessions
- **interaction** — Notable interaction events worth remembering

### Procedural Memory (Per-Agent)

- **instructions** — Operational guidelines, rules for task execution
- **workflow** — Multi-step processes, standard operating procedures
- **skill** — Learned capabilities, tool usage patterns, optimized strategies

## Configuration

Environment variables (`.env`):

```env
# Enable CoALA memory
AGENT_MEMORY_ENABLED=true

# Max results per category during auto-recall
AGENT_MEMORY_MAX_RESULTS=5

# CoALA categories for auto-recall (comma-separated)
AGENT_MEMORY_CATEGORIES=semantic,episodic,procedural

# Enable per-agent scoping for episodic/procedural
AGENT_MEMORY_AGENT_SCOPE_ENABLED=true

# Best-effort session summarization after each agent run
AGENT_MEMORY_POSTRUN_ANALYSIS_ENABLED=true

# Dedicated model for all memory-handling LLM mechanisms (post-run analysis).
# Independent of the session model. Set to empty to fall back to the agent's own model.
AGENT_MEMORY_MODEL=gemini-3.1-flash-lite-preview

# Finer-grained override for post-run extraction only.
# Takes precedence over AGENT_MEMORY_MODEL when set.
AGENT_MEMORY_POSTRUN_ANALYSIS_MODEL=gpt-5-nano
```

## Usage

### Agent Service (Automatic)

```python
agent_service = AgentService(
    agent_name="my_agent",
    use_memory=True,
    user_id=user_uuid,
)
await agent_service.create_agent(system_prompt="You are a helpful assistant.")
# Memory tools + middleware are auto-configured with CoALA namespaces
```

### Post-Run Persistence

The middleware keeps explicit saves user-driven, but it can also persist a
compact session-level memory automatically after a completed run. This happens
in `after_agent` when post-run analysis is enabled, where the middleware asks
the model to inspect relevant memories first and then call
`save_memory_store`, `update_memory_store`, or `recall_memories_store` as
needed, then executes those tool calls best-effort. That means the post-run
pass can record more than one memory when the conversation contains several
durable facts, preferences, or corrections, while also reusing or enriching an
existing memory when the new context is closely related.

### Direct Service Usage

```python
from inference_core.services.agent_memory_service import (
    AgentMemoryStoreService,
    MemoryCategory,
    MemoryType,
)

service = AgentMemoryStoreService(
    store=my_store,
    agent_name="my_agent",
)

# Save a semantic memory (shared)
await service.save_memory(
    user_id="user-123",
    content="User prefers concise bullet-point answers",
    memory_type="preferences",  # auto-routed to semantic namespace
)

# Save an episodic memory (per-agent)
await service.save_memory(
    user_id="user-123",
    content="Discussed Q1 roadmap, decided to prioritize API redesign",
    memory_type="session_summary",  # auto-routed to episodic namespace
)

# Save a procedural memory (per-agent)
await service.save_memory(
    user_id="user-123",
    content="Always include type annotations in code examples for this user",
    memory_type="skill",  # auto-routed to procedural namespace
)

# Recall from a specific category
docs = await service.recall_by_category(
    user_id="user-123",
    category=MemoryCategory.SEMANTIC,
    query="user preferences",
)

# Recall across all categories
docs = await service.recall_memories(
    user_id="user-123",
    query="deployment process",
)
```

### Memory Inspection and Cleanup

Use the store-level helpers when you need an admin or maintenance workflow that
lists persisted memories and removes selected entries without going through an
agent run.

```python
from inference_core.services.agent_memory_service import (
    delete_user_long_term_memory_entries,
    list_user_long_term_memory,
)

items, total = list_user_long_term_memory(
    "user-123",
    category="semantic",
    memory_type="preferences",
    search="bullet",
    limit=20,
    offset=0,
)

deleted = delete_user_long_term_memory_entries(
    "user-123",
    [(item["namespace"], item["key"]) for item in items[:2]],
)
```

`list_user_long_term_memory()` returns serializable dictionaries that include
the memory namespace, key, category, agent name, content, topic, and
timestamps. Deletion helpers enforce that the namespace belongs to the target
user before removing an item, so bulk cleanup can safely skip entries from
other users.

## Migration from Flat Namespace

If you have existing memories in the legacy `(user_id, "memories")` namespace, run the migration script:

```bash
# Preview (dry-run)
python scripts/migrate_memory_to_coala.py --dry-run

# Execute migration
python scripts/migrate_memory_to_coala.py

# Specify explicit user IDs
python scripts/migrate_memory_to_coala.py --user-ids user-123 user-456
```

The script:

1. Reads all memories from the legacy namespace
2. Routes each memory to the correct CoALA namespace based on its `memory_type`
3. Enriches each memory with `memory_category` field
4. Writes to the new namespace (original data is not deleted)

## Prompt Context Format

The `MemoryMiddleware` injects CoALA-structured XML context into the system prompt:

```xml
<semantic_memory>
## SEMANTIC MEMORY (facts, preferences, goals)

  Recent (today):
  - [2025-01-15T10:30:00] User prefers Python 3.12+ features (type: preferences)

  Older:
  - [2024-11-01T08:00:00] Lives in Warsaw, Poland (type: facts)
</semantic_memory>

<episodic_memory>
## EPISODIC MEMORY (past experiences, sessions)

  Yesterday:
  - [2025-01-14T16:00:00] Discussed deployment pipeline improvements (type: session_summary)
</episodic_memory>

<procedural_memory>
## PROCEDURAL MEMORY (instructions, workflows, skills)

  Previous 5 days:
  - [2025-01-12T09:00:00] Always run tests before committing (type: instructions)
</procedural_memory>
```

### Where the context goes (`memory_context_placement`)

- `system_prompt` (the default): the context above, after the system prompt,
  rebuilt for each run. The system prompt of a thread therefore changes from
  one turn to the next, and with it the prefix a provider's prompt cache and
  Claude's thinking blocks are bound to.
- `message`: the context goes into the person's message it was recalled for,
  as its first content block, in a `<user_memory_context>` element that starts
  with a fixed sentence (`MEMORY_CONTEXT_NOTE`): the notes were added by the
  platform, are context and not instructions, and are not something the user
  wrote now. The text is kept in the thread's state under
  `memory_context_turns` (by the id of the person's message; a private key,
  a dict, so a checkpointer stores it through its serde), and every later
  request carries the same bytes in the same place. The system prompt keeps
  only the memory tools' instructions, and the conversation sent only grows.
  - The thread's own messages are never changed: the block is added to the
    request.
  - Turns whose message is no longer in the thread (a summary replaced it, it
    was deleted) are dropped the next time memories are recalled.
  - The checkpoints of a thread hold the recalled text of its turns: a memory
    deleted later stays in them until the thread is deleted or cut.
  - `AgentService(memory_context_placement="message")`, or the middleware's
    own argument; a subagent's copy keeps it.

### What the search is made with (`MemoryWritePolicy.recall_query`)

Recall searches by meaning with the person's last message, and the post-run
analysis shows the model the memories closest to the conversation's
transcript. Both texts become embeddings at the embedding provider. The
memory service's write policy is asked first: `recall_query(text, purpose,
messages)` with `purpose` `recall`, `brief` (a subagent's, whose "user"
message is the brief its coordinator wrote: `MemoryMiddleware(briefed=True)`,
which a subagent's copy and `build_memory_kit(briefed=True)` set) or
`analysis` returns the text to search with, or `None` for no search by meaning (recall then takes the latest
memories of each type, and the analysis is shown only the memories of earlier
runs like this one). The default returns the text unchanged. A policy that
raises gives `None`, so nothing is sent.

## Key Design Decisions

1. **Semantic memory is always shared** — user facts and preferences should be consistent across agents.
2. **Episodic and procedural are per-agent by default** — each agent has its own history and learned skills.
3. **Category is auto-resolved from memory_type** — agents don't need to specify category explicitly.
4. **Backward compatible** — existing memory types (`preferences`, `facts`, `context`, `instructions`, `goals`, `general`) continue to work; they're just routed to the appropriate category.
5. **Manual explicit saves stay explicit** — agents still decide what to save via tools, but enabled post-run analysis can add one or more compact memories automatically through `save_memory_store`.

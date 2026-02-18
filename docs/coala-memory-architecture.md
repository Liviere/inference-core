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

## Key Design Decisions

1. **Semantic memory is always shared** — user facts and preferences should be consistent across agents.
2. **Episodic and procedural are per-agent by default** — each agent has its own history and learned skills.
3. **Category is auto-resolved from memory_type** — agents don't need to specify category explicitly.
4. **Backward compatible** — existing memory types (`preferences`, `facts`, `context`, `instructions`, `goals`, `general`) continue to work; they're just routed to the appropriate category.
5. **Manual episodic saves** — agents explicitly decide what to save as episodic memory (no auto-save of transcripts).

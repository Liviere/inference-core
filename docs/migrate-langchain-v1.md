# Migration to LangChain v1.0+

Goal: migrate to the new LangChain 1.0 API (agents, unified content blocks, simplified namespace) while maintaining backward compatibility via langchain-classic in areas that are not yet modernized.

## TL;DR – what's changing in v1

- New agents standard: `langchain.agents.create_agent` + middleware (PII, summarization, HITL).
- Unified content blocks with types: reasoning, text, tool_call – a common API across providers.
- Simplified namespace in the "new" LangChain:
  - classic modules (chains, retrievers, hub, indexing) are moved to a separate package: `langchain-classic` (import: `langchain_classic.*`).
  - Many modern primitives are available on `langchain.*` (agents, messages, tools, init_chat_model / init_embeddings).
- It is recommended to install `langchain` and – temporarily for compatibility – `langchain-classic`.

Source: see the "Context7 Sources" section below.

## Repo scan – where we touch LangChain

Identified imports:

- `langchain_core.*` – prompts, runnables, output_parsers, messages; this remains (core package) – no immediate migration required.
- `langchain_openai`, `langchain_google_genai`, `langchain_anthropic` – provider integrations; only ensure compatible versions (>=1.x where available) and initialization options.
- `langchain_community.chat_message_histories.SQLChatMessageHistory` – chat history; remains, requires compatible community versions.
- `langchain.agents.AgentExecutor`, `create_openai_tools_agent` (optionally used) – these are classic elements; in v1 prefer `create_agent` + middleware. If needed, use classics via `langchain_classic.agents.*`.

Key file paths in the codebase:

- `inference_core/llm/models.py` – model factory (OpenAI, Gemini, Claude) – OK on core/provider packages.
- `inference_core/llm/chains.py` – `RunnableWithMessageHistory` + `SQLChatMessageHistory` – compatible; no changes initially.
- `inference_core/llm/prompts.py` – `ChatPromptTemplate`, `PromptTemplate` – from core, OK.
- `inference_core/llm/callbacks.py` – `UsageMetadataCallbackHandler` – from core, OK; consider extending for `content_blocks` (reasoning/tool_call) later.
- `inference_core/llm/streaming.py` – astream_events/astream – should work unchanged; later can support content blocks.
- `inference_core/services/llm_service.py` – agent usages (optional block): migrate or cover with a classic fallback.

## Migration strategy – phased

1. Phase 0: Safe compatibility lane

- Add dependency: `langchain-classic` (import `langchain_classic.*`) – use only where we currently use classic items (AgentExecutor / create_openai_tools_agent / chains, if present).
- Do not change logic based on `langchain_core.*` yet – tests should pass.

2. Phase 1: Compatibility layer (optional shim)

- Add module `inference_core/llm/compat/` with lightweight import aliases:
  - `from langchain_core.prompts import ChatPromptTemplate, PromptTemplate`
  - `from langchain_core.output_parsers import StrOutputParser`
  - `from langchain_core.runnables.history import RunnableWithMessageHistory`
  - History: `SQLChatMessageHistory` from `langchain_community.*`
- Goal: unify imports across our code and ease future replacements.

3. Phase 2: Agents → new standard

- In `llm_service.py` replace the old agent creation path (`create_openai_tools_agent`/`AgentExecutor`) with the new (`langchain.agents.create_agent`).
- Add middleware support (PII redaction, summarization, HITL) as needed.
- Keep a fallback: if `create_agent` is unavailable / not configurable – use `langchain_classic.agents` (log a warning + tests).

4. Phase 3: Usage & streaming – content blocks

- Extend `LLMUsageCallbackHandler` to handle `AIMessage.content_blocks` (reasoning/text/tool_call) – logging and optional metrics.
- Streaming: optionally emit metadata events based on content blocks (without breaking the SSE contract).

5. Phase 4: Namespace cleanup

- Where imports use `langchain` (new) – adopt the new API (messages, tools, init_chat_model) only where justified.
- Avoid mixing classics (`langchain_classic`) with new features in the same module; keep them in the compatibility layer or separate code paths.

## Dependency changes (proposal)

- Add: `langchain>=1.0,<2.0` (if not already transitively available).
- Add: `langchain-classic>=1.0,<2.0` – only until agents and classic APIs are fully migrated.
- Keep and upgrade provider integrations to compatible versions:
  - `langchain-openai>=1.0` (if available – currently we use `>=0.3.29,<0.4.0`, verify compatibility with v1).
  - `langchain-community` – a version that supports v1 (currently `>=0.3.x` may be OK, but check release notes).
  - `langchain-google-genai` / `langchain-anthropic` – ensure versions are supported with v1.

Note: the current `pyproject.toml` does not have a direct `langchain` dependency, but there are `langchain-*` packages. The v1 "new" `langchain` will be needed for agents/middlewares. `langchain_core` is provided transitively – we don't change it.

## Test plan and risks

- Unit/integration tests should pass after adding `langchain-classic` (Phase 0). Typical risks:
  - Provider version changes (e.g., OpenAI, Anthropic) – params/timeouts.
  - Agent path – if code enters the new path; therefore keep fallback and a feature flag.
- After Phase 2 (new agents) add tests:
  - Happy path with a simple tools configuration (`tools`).
  - Middleware path (e.g., PII redaction) – stubbed.
  - Handling tool call vs structured output.
- Streaming and usage – regression tests for SSE and event snapshots.

## Deployment checklist

- [ ] Add `langchain-classic` to dependencies.
- [ ] (optional) Add `inference_core/llm/legacy/` and switch internal imports to aliases.
- [ ] Refactor agents in `llm_service.py` (including a classic/new mode flag).
- [ ] Extend usage callback for content blocks.
- [ ] Run full test suite (unit + integration) and fix regressions.

## Mapping changes → our modules

- `services/llm_service.py` – the largest scope of changes (agents); add a new agent builder and classic fallback. Optionally add `response_format` handling (structured output) if needed.
- `llm/chains.py` – no immediate changes; in the future consider initializing models via `init_chat_model`.
- `llm/callbacks.py` – extend for content blocks and avoid double counting usage.
- `llm/streaming.py` – compatible; add detection for reasoning/tool_call in the stream (optional, without changing SSE contract).

## Context7 Sources

- LangChain OSS Python – v1 release highlights
  - ID: `/websites/langchain_oss_python_releases_langchain-v1`
  - Topics: "langchain v1 release highlights and migration"

If you want, I can immediately add a pinned dependency and a skeleton compatibility layer (without changing runtime behavior) in a separate PR.

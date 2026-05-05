# LangChain v1 Migration Status

The LangChain v1 migration is complete for the runtime surface of this project. The backend no longer ships the legacy chain-based `LLMService` API, a direct `langchain-classic` dependency, or any `langchain_classic.*` imports.

## Current Runtime Surface

- Agent execution uses `AgentService` and `DeepAgentService` built on `langchain.agents.create_agent`.
- Remote execution uses the LangGraph Agent Server path described in `agents-service.md`.
- User-facing agent customization flows through `/api/v1/agent-instances`.
- Provider/model creation, parameter policy, usage logging, MCP tools, local tool providers, embeddings, vector stores, and batch APIs remain shared infrastructure.
- Provider-native batch endpoints remain mounted under `/api/v1/llm/batch` because they are independent from the removed chain runtime.

## Removed Legacy Surface

- `inference_core.services.llm_service`
- `inference_core.api.v1.routes.llm`
- `inference_core.llm.chains`
- `inference_core.llm.streaming`
- `inference_core.llm.prompts`
- `inference_core.celery.tasks.llm_tasks`
- `/api/v1/llm/completion`, `/api/v1/llm/chat`, `/api/v1/llm/*/stream`, `/api/v1/llm/models`, `/api/v1/llm/stats`, `/api/v1/llm/health`, and `/api/v1/llm/param-policy`
- direct `langchain-classic` dependency and any `langchain_classic.*` imports

Note: `langchain-classic` can still appear in `poetry.lock` as a transitive dependency of `langchain-community`. The project keeps `langchain-community` for integrations such as DeepInfra, but runtime code no longer imports classic modules directly.

## Migration Guidance For Integrations

- Replace completion/chat calls with AgentService runs or Agent Instance runs.
- Move reusable prompts into `agents:` configuration or Agent Instance prompt overrides.
- Move local tools from task-level config to agent-level `local_tool_providers`.
- Use `LLMUsageService` and existing cost middleware for usage/cost reporting instead of the removed stats endpoint.
- Keep direct model factory usage limited to infrastructure code; application behavior should generally enter through agents.

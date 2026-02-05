# Documentation Index

Central index of project documentation

## Sections

| Area                               | Purpose                                     | File                                 |
| ---------------------------------- | ------------------------------------------- | ------------------------------------ |
| Quickstart & Overview              | Entry point for newcomers                   | `../README.md`                       |
| Configuration Reference            | All environment variables grouped by domain | `configuration.md`                   |
| LLM Configuration                  | Reference for `llm_config.yaml`             | `llm_config_summary.md`              |
| Dynamic & DB Configuration         | DB overrides & user-specific preferences    | `dynamic-configuration.md`           |
| Repository Structure Snapshot      | Auto/Manual snapshot of directories         | `structure.md`                       |
| Custom Prompts (Jinja2)            | How to add your own prompt templates        | `custom-prompts.md`                  |
| LangChain Agents                   | YAML-configured agents, tools, middleware   | `agents.md`                          |
| Agent Memory                       | Long-term memory for agents                 | `agents.md#agent-memory`             |
| Vector Store Guide                 | Semantic search & ingestion                 | `vector-store.md`                    |
| Provider Extensions (OpenAI batch) | Provider specific behavior                  | `providers/openai-batch-provider.md` |
| Docker Test Environments           | Isolated test setup                         | `testing-docker.md`                  |
| LLM Usage Logging                  | Cost / usage persistence & privacy notes    | `observability/llm-usage-logging.md` |

## How to Navigate

1. Start with the root `README.md` for a quick run.
2. Jump to `configuration.md` when setting up env vars.
3. Use guides (Vector / Batch) for feature-specific integration.
4. Consult design docs (`docs/issues/*`) for architectural rationale.

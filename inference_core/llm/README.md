# LLM Module

This package now contains shared LLM infrastructure, not a standalone completion/chat API. Agent execution lives in `inference_core.services.agents_service`, while this package owns model configuration, provider factories, parameter policy, tool-provider registry, MCP helpers, and usage logging primitives.

## Components

- `config.py`: loads and validates `llm_config.yaml` providers, models, agents, batch settings, MCP profiles, and usage logging config.
- `models.py`: creates provider-specific chat model instances for AgentService, batch, embeddings, and vector workflows.
- `param_policy.py`: normalizes provider/model parameters and blocks deprecated parameters for models that no longer accept them.
- `tools.py`: registry for local LangChain-compatible tool providers used by AgentService.
- `mcp_tools.py`: MCP integration helpers used by configured agents.
- `usage_logging.py` and `callbacks.py`: token and cost tracking primitives.

## Runtime Entry Points

- Programmatic agent runs: `AgentService` and `DeepAgentService`.
- HTTP agent runs and customization: `/api/v1/agent-instances`.
- Remote LangGraph execution: Agent Server graphs built from `agent_graphs.py` and `inference_core/agents/graph_builder.py`.
- Provider-native batch jobs: `/api/v1/llm/batch`.
- Embeddings and vector workflows: `/api/v1/embeddings/*` and `/api/v1/vector/*`.

## Parameter Normalization

Parameter normalization happens automatically in `LLMModelFactory`. The policy layer accepts common parameters, applies provider/model-specific renames, drops unsupported values, and raises when a caller supplies parameters that a model explicitly rejects.

Configuration lives in `llm_config.yaml` under `param_policies`:

```yaml
param_policies:
  settings:
    passthrough_prefixes: ['x_', 'ext_']
  providers:
    openai:
      patch:
        allowed: ['logit_bias']
  models:
    gpt-5:
      replace:
        allowed: ['reasoning_effort', 'verbosity']
        dropped:
          - temperature
          - top_p
          - frequency_penalty
          - presence_penalty
          - max_tokens
          - request_timeout
```

## Agent Configuration

Use `llm_config.yaml` to define models and agents:

```yaml
agents:
  default_agent:
    primary: gpt-5-mini
    description: General-purpose assistant.
    local_tool_providers: []
    reasoning_output: false
```

For HTTP usage, list templates with `GET /api/v1/agent-instances/templates`, create a user instance with `POST /api/v1/agent-instances`, then run it with `POST /api/v1/agent-instances/{instance_id}/run`.

## Fireworks Reasoning Responses

Fireworks reasoning models can emit `reasoning_content` alongside the normal assistant content. The local `ChatFireworksReasoning` adapter preserves that field for standard responses and streaming chunks, and serializes it into later requests when reasoning history should be preserved.

Example model config:

```yaml
accounts/fireworks/models/kimi-k2p5:
  provider: 'fireworks'
  max_tokens: 8192
  reasoning_config:
    reasoning_effort: 'medium'
    reasoning_history: 'preserved'
```

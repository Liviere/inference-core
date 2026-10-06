# Custom Chat Model Providers

The built-in providers (`openai`, `claude`, `gemini`, `mistral`, `ollama`, …) are a fixed set. An application embedding `inference-core` can add its own: it registers a provider name together with a function that builds the chat model, and from then on that name works wherever a built-in one does.

Reach for this when completions are not served by one of the supported HTTP clients: an in-house gateway with its own protocol, a runtime on the user's device, a model reached over a message bus.

## Registering a provider

```python
from langchain_core.language_models.chat_models import BaseChatModel

from inference_core.llm.config import ModelConfig
from inference_core.llm.provider_registry import register_chat_model_provider


def build_gateway_model(config: ModelConfig, params: dict) -> BaseChatModel:
    return GatewayChatModel(
        model=config.name,
        tenant=config.tenant,  # any extra field of the model entry
        temperature=params.get("temperature"),
        timeout=params.get("request_timeout"),
    )


register_chat_model_provider("in_house_gateway", build_gateway_model)
```

The builder receives:

- `config` — the model's `ModelConfig`, including any extra fields its entry carries;
- `params` — the call parameters after normalization (model defaults, agent `generation_params`, caller overrides).

It returns a LangChain `BaseChatModel`. If it raises, the factory logs the error and returns no model, as it does for a built-in provider that cannot be constructed.

A built-in provider name cannot be registered. Registering the same custom name again replaces the earlier entry, so a module that registers at import time is safe to import twice.

## Using it

In `llm_config.yaml`, with no entry under `providers` required:

```yaml
models:
  gateway-large:
    provider: 'in_house_gateway'
    max_tokens: 4096
    tenant: 'acme'

agents:
  assistant_agent:
    primary: 'gateway-large'
```

Or at runtime, on a per-request copy of the config:

```python
config = get_llm_config().with_overrides(
    agent_overrides={"assistant_agent": {"primary": "gateway-large"}}
)
config.add_custom_model(ModelConfig(name="gateway-large", provider="in_house_gateway"))
```

**Register first.** A model entry naming a provider that is neither built in nor registered is rejected — in YAML when the config loads, at runtime when the `ModelConfig` is created. An application that names its provider in YAML has to register it before the first `get_llm_config()` call.

## Parameters

Without a policy of its own, a registered provider is given the sampling parameters every chat API shares (`temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`) and `request_timeout`. Anything else is dropped with a warning.

Pass a `ProviderParamPolicy` to change that:

```python
from inference_core.llm.param_policy import ProviderParamPolicy

register_chat_model_provider(
    "in_house_gateway",
    build_gateway_model,
    param_policy=ProviderParamPolicy(
        allowed={"temperature", "max_tokens", "top_k", "timeout"},
        renamed={"request_timeout": "timeout"},
        dropped={"frequency_penalty", "presence_penalty"},
    ),
)
```

## What the rest of the stack sees

- `LLMConfig.build_model_provider_map()` and usage logs carry the registered name as the provider.
- `LLMConfig.is_model_available()` reports such a model as available: there is no key or endpoint in the config to judge it by.
- Model fallback, tool-model overrides and subagent models build it through the same factory path.
- A model with no `pricing` is logged with its token counts and no cost.

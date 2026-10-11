# Model Catalog

Which models exist is otherwise known only from what somebody wrote under `models:` in the LLM config. Providers publish that list themselves, and several say in it when a model is deprecated or will be shut down. The model catalog reads those listings, keeps them, and compares them with the config, so a new model or an approaching retirement is seen here and not first in a failed request.

**The catalog is advisory.** A model can be used when it has an entry under `models:`, exactly as without the catalog. Nothing a provider lists becomes usable by being listed.

## What it covers

A provider is part of the catalog when all three hold:

- the config has at least one model of it,
- it has a model lister (built in for `openai`, `claude`, `gemini`, `mistral`, `xai`, `deepinfra`, `fireworks` and `custom_openai_compatible`; not for `ollama`),
- it can be asked: its API key is set, or it needs none. A `custom_openai_compatible` provider also needs its `base_url`.

To leave a provider out, switch it off in its entry:

```yaml
providers:
  deepinfra:
    name: 'DeepInfra'
    api_key_env: 'DEEPINFRA_API_TOKEN'
    requires_api_key: true
    catalog:
      enabled: false
```

A `catalog:` block that cannot be read is ignored with a warning. It never stops the provider's models from loading.

Providers do not say the same things about their models:

| Provider                   | Deprecation / retirement | Aliases | Prices | Limits |
| -------------------------- | ------------------------ | ------- | ------ | ------ |
| `openai`                   | shutdown date            | –       | –      | –      |
| `claude`                   | lifecycle and both dates | by id   | –      | yes    |
| `gemini`                   | –                        | –       | –      | yes    |
| `mistral`                  | deprecation, replacement | yes     | –      | yes    |
| `xai`                      | –                        | yes     | yes    | –      |
| `deepinfra`                | deprecation, replacement | –       | yes    | yes    |
| `fireworks`                | retirement date          | –       | –      | yes    |
| `custom_openai_compatible` | –                        | –       | –      | –      |

No provider notifies about changes, so the catalog asks on a schedule.

## Switching it on

| Variable                                     | Default | Description                                                                                             |
| -------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------- |
| `LLM_MODEL_CATALOG_ENABLED`                  | false   | Read the listings on a schedule                                                                         |
| `LLM_MODEL_CATALOG_REFRESH_INTERVAL_SECONDS` | 86400   | How often a provider is read; also how far apart two readings must be before a missing model is removed |
| `LLM_MODEL_CATALOG_HTTP_TIMEOUT_SECONDS`     | 20      | Read timeout of one listing request                                                                     |

With the flag on, the Celery app built by `create_celery_app()` schedules `llm.catalog_refresh` on the `default` queue every 15 minutes. Most runs read nothing: a provider is read when its interval has passed, or an hour after a reading that failed. The worker needs the provider API keys and a way out to the providers; proxy settings are taken from the environment (`HTTPS_PROXY`).

The three tables (`llm_catalog_models`, `llm_catalog_events`, `llm_catalog_provider_state`) are created with the rest of the schema. The command line below works with the flag off: the flag is about the schedule only.

## Command line

```bash
python -m inference_core.llm.catalog refresh [--provider NAME ...]
python -m inference_core.llm.catalog status
python -m inference_core.llm.catalog drift [--check]
python -m inference_core.llm.catalog events [--since 7d] [--provider NAME]
```

- `refresh` reads the providers now, whenever they were last read. Exits with 1 when a provider failed.
- `status` shows, per provider, how many models are configured and listed, when it was last read and how that went.
- `drift` lists the configured models that need a look (see below). With `--check` it exits with 1 when there is any, for use in a deploy check.
- `events` lists what changed in the listings, newest first.

`status`, `drift` and `events` read what is stored and ask no provider.

## Drift

Drift is about configured models only:

| State        | Meaning                                                                                                |
| ------------ | ------------------------------------------------------------------------------------------------------ |
| `missing`    | The provider does not list the model and does not know it by id. A typo in the config looks like this. |
| `retiring`   | The provider has set the date the model stops answering, or has retired it already.                    |
| `deprecated` | The provider has deprecated the model without a date.                                                  |

A configured alias counts as the model it stands for. Models a provider offers that nobody configured are never drift: they show up as `added` events when they appear.

## Events

| Event               | When                                                                                    |
| ------------------- | --------------------------------------------------------------------------------------- |
| `added`             | A model is in the listing for the first time                                            |
| `removed`           | A model is gone (see the rules below)                                                   |
| `reappeared`        | A removed model is listed again                                                         |
| `lifecycle_changed` | Its lifecycle, deprecation date, retirement date or replacement changed, with old and new |

The rules keep the log quiet unless something happened:

- The first reading of a provider is a baseline and reports nothing.
- A reading that fails changes nothing but the provider's state. So does a listing of the wrong shape, one cut short, or an empty one from a provider that had models: a listing is whole or it is an error.
- A model missing from one listing is not gone. It counts as removed after two readings without it that are a refresh interval apart.
- A configured model the listing does not show is asked about by id. That is how an alias or a retired model is found, and when the provider answers that there is no such model, the model is removed at once.

## Metrics

Written by `llm.catalog_refresh` on every run, from what is stored:

| Metric                                                          | Meaning                                                              |
| --------------------------------------------------------------- | -------------------------------------------------------------------- |
| `llm_catalog_publish_timestamp`                                 | When the gauges were last written                                    |
| `llm_catalog_provider_last_success_timestamp{provider}`         | Last successful reading (0 = never)                                  |
| `llm_catalog_provider_stale{provider}`                          | 1 when a covered provider had no successful reading for 3 intervals  |
| `llm_catalog_provider_models{provider}`                         | Models in the latest listing                                         |
| `llm_catalog_events_recent{provider,type}`                      | Changes in the last 24 hours                                         |
| `llm_catalog_configured_models{provider,state}`                 | Configured models in drift, by state                                 |
| `llm_catalog_configured_retirement_soonest_timestamp{provider}` | Earliest retirement date among configured models (0 = none)          |

Every series exists for every provider that has a lister, with 0 for nothing to report. None carries a model name: with Prometheus' multiprocess mode a labelled series cannot be taken back, so an alert for one model would stay up after that model left the config. Alert on the counts, and ask `drift` which models are meant.

Example alert expressions:

```promql
llm_catalog_configured_models{state="missing"} > 0
(llm_catalog_configured_retirement_soonest_timestamp > 0) - time() < 30 * 86400
llm_catalog_provider_stale > 0
time() - max(llm_catalog_publish_timestamp) > 3600
```

## A lister for a provider of your own

A provider registered with `register_chat_model_provider` (see [`custom-model-providers.md`](custom-model-providers.md)) joins the catalog when it has a lister:

```python
import httpx

from inference_core.llm.catalog import (
    DiscoveredModel,
    ModelLifecycle,
    register_model_lister,
)


async def list_gateway_models(runtime: dict, client: httpx.AsyncClient):
    response = await client.get("https://gateway.internal/models")
    response.raise_for_status()
    return [
        DiscoveredModel(
            provider="in_house_gateway",
            model_id=item["id"],
            lifecycle=ModelLifecycle.ACTIVE,
        )
        for item in response.json()["models"]
    ]


register_model_lister("in_house_gateway", list_gateway_models)
```

`runtime` is the provider's entry from the config with `api_key` resolved. Return the whole listing or raise: a partial list would read as models having been removed. An optional `get_model(runtime, client, model_id)` answers for one model, with `None` for "no such model".

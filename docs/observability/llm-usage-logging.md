# LLM Usage & Cost Logging

The inference-core application includes comprehensive LLM usage and cost logging to help track token consumption, costs, and performance across all LLM interactions.

## Overview

The usage logging system captures detailed metrics for every LLM request, including:

- **Token usage**: Input, output, and total tokens with support for provider-specific extras (reasoning tokens, cache tokens, etc.)
- **Cost tracking**: Precise USD cost calculations based on configurable pricing per token dimension
- **Performance metrics**: Request latency, success/failure status, error classification
- **Request metadata**: Task type, model, provider, session correlation
- **Audit trails**: Raw usage data and pricing snapshots for transparency

## Key Features

### Lean Core + Flexible Extras

The system uses a "lean core + flexible extras" approach:

- **Core tokens** (input/output) are stored as first-class database columns for efficient querying
- **Extra dimensions** (reasoning, cache, etc.) are stored in JSON fields for flexibility
- No schema migrations needed when providers add new token types

### Pricing Flexibility

- **Per-1M or per-1K pricing** for all dimensions, normalized internally to per-1K values
- **Key aliases** to normalize provider-specific usage keys
- **billed_separately** marks additive extras such as cache write charges
- **Context tier multipliers** for pricing based on input size
- **Unpriced token passthrough** captures counts even without pricing
- **Breakdown keys** (`extras_policy.breakdown_keys`) name usage keys that are part of the core counts and do not mark a cost estimated

### Cost Transparency

- **Pricing snapshots** stored with each log for audit trails
- **Raw usage data** preserved for debugging and verification
- **Cost estimation flags** when some tokens lack pricing

## Configuration

### Pricing Configuration

Add pricing information to models in `llm_config.yaml`:

```yaml
models:
  gpt-5-mini:
    provider: 'openai'
    pricing:
      currency: USD
      input:
        cost_per_1m: 0.15
      output:
        cost_per_1m: 0.60
      extras:
        reasoning_tokens:
          cost_per_1m: 2.40
          billed_separately: true
        cache_read_tokens:
          cost_per_1m: 0.03
        cache_write_tokens:
          cost_per_1m: 1.25
          billed_separately: true
      key_aliases:
        prompt_tokens: input_tokens
        completion_tokens: output_tokens
      context_tiers:
        - max_context: 128000
          multiplier: 1.0
        - max_context: 1000000
          multiplier: 1.5
      rounding:
        decimals: 6
      extras_policy:
        passthrough_unpriced: true
```

#### Context tiers

`context_tiers` multiplies every price of the entry by the multiplier of the
first tier the request fits in (a request over the last tier pays the last
one). Which tokens choose the tier is `context_tier_basis`:

- `total` (the default): input and output tokens together.
- `input`: the prompt alone, its cache reads and writes included
  (`input_tokens` already counts them). Use it for a provider that prices a
  request by the length of its prompt, such as Anthropic's models priced
  higher over 100K tokens.

With `input`, a request of 95,000 prompt tokens and 10,000 output tokens on a
model priced at 0.10 / 0.50 per 1M up to 100K and five times that above costs
0.0095 + 0.005 = 0.0145 USD; with `total` the same request is over 100K and
costs 0.0475 + 0.025 = 0.0725 USD.

```yaml
      context_tiers:
        - max_context: 100000
          multiplier: 1.0
        - max_context: 1000000
          multiplier: 5.0
      context_tier_basis: input
```

Standard LangChain-style extras are treated as token breakdowns already included
in input_tokens or output_tokens. Use billed_separately: true when a provider
charges for an extra dimension on top of the core token count. If a provider
uses nonstandard detail names, map them with key_aliases to a normalized extra
key whose name already makes the parent dimension clear.

#### Usage keys as the cost tracking names them

The agent middleware and the usage callback turn every key of LangChain's
`input_token_details` and `output_token_details` into `<key>_tokens`, and keep
`total_tokens`. The names your `extras` and `key_aliases` must match are these:

| Detail from LangChain | Usage key |
|---|---|
| `input_token_details.cache_read` | `cache_read_tokens` |
| `input_token_details.cache_creation` (Anthropic, without a TTL breakdown) | `cache_creation_tokens` |
| `input_token_details.ephemeral_5m_input_tokens` (Anthropic) | `ephemeral_5m_input_tokens_tokens` |
| `input_token_details.ephemeral_1h_input_tokens` (Anthropic) | `ephemeral_1h_input_tokens_tokens` |
| `output_token_details.reasoning` | `reasoning_tokens` |

`langchain-anthropic` counts cache reads and cache writes in `input_tokens`, and
reasoning in `output_tokens`. A cache write priced on its own is therefore a
breakdown, not `billed_separately`. Map the Anthropic write keys to an extra whose
name contains `cache` so it is taken out of the input tokens:

```yaml
extras:
  cache_write_tokens:
    cost_per_1m: 5      # 5-minute write, 1.25x the input price
  cache_write_1h_tokens:
    cost_per_1m: 8      # 1-hour write, 2x the input price
  cache_read_tokens:
    cost_per_1m: 0.2
key_aliases:
  ephemeral_5m_input_tokens_tokens: cache_write_tokens
  ephemeral_1h_input_tokens_tokens: cache_write_1h_tokens
  cache_creation_tokens: cache_write_tokens   # with prompt_cache: 5m
```

When the response has the TTL breakdown, `cache_creation` is 0 and is left out,
so only one of the two keys that map to a write price is ever present.

A streamed response has no TTL breakdown: `langchain-anthropic` reads the usage
of a stream from its `message_delta` event, which names only
`cache_creation_input_tokens`. The model factory streams every call, so
`cache_creation_tokens` is what nearly every call logs. A model whose
`prompt_cache` is `1h` asks only for an hour's cache, so it maps that key to
the hour's price:

```yaml
  cache_creation_tokens: cache_write_1h_tokens   # with prompt_cache: 1h
```

Mapped to the 5-minute price, an hour's write would be logged at 1.25x the
input price instead of 2x. When the file is read, a model whose
`cache_creation_tokens` is priced at another rate than its `prompt_cache`
lifetime's (1.25x for `5m`, 2x for `1h`) is logged with a warning; its
pricing is used as it is.

#### Keys that only break down input or output

A usage key that ends in `_tokens` and has no price in `extras` marks the cost
estimated (`cost_estimated`). `total_tokens`, which comes with every call, is
one of them. So is `reasoning_tokens` when the reasoning is billed at the output
price and already counted in `output_tokens`.

`extras_policy.breakdown_keys` names the keys that only break down the input or
output tokens. They leave the cost as it is and do not mark it estimated:

```yaml
extras_policy:
  breakdown_keys: [total_tokens, reasoning_tokens]
```

The list is empty by default, and every unpriced key then marks the cost
estimated. Leave `reasoning_tokens` out of the list for a provider that bills
reasoning at a price of its own: price it in `extras` instead.

### Global Settings

Configure usage logging behavior in the `settings` section:

```yaml
settings:
  usage_logging:
    enabled: true
    base_currency: USD
    fail_open: true
    default_rounding_decimals: 6
```

### Environment Variables

Override logging settings:

- `LLM_USAGE_LOGGING_ENABLED`: Enable/disable logging (true/false)
- `LLM_USAGE_FAIL_OPEN`: Continue on logging errors (true/false)

## Security & Privacy

- **No prompt storage**: Only token counts and metadata are logged, never actual content
- **No request parameter persistence**: Model parameters and prompts are not stored
- **User attribution**: Optional user association for multi-tenant deployments
- **Error message truncation**: Error messages limited to 500 characters

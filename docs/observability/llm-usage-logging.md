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
  cache_creation_tokens: cache_write_tokens
```

When the response has the TTL breakdown, `cache_creation` is 0 and is left out,
so only one of the two keys that map to `cache_write_tokens` is ever present.

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

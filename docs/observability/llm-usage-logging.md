# LLM Usage & Cost Logging

The inference-core application includes comprehensive LLM usage and cost logging to help track token consumption, costs, and performance across all LLM interactions.

## Overview

The usage logging system captures detailed metrics for every LLM request, including:

- **Token usage**: Input, output, and total tokens with support for provider-specific extras (reasoning tokens, cache tokens, etc.)
- **Cost tracking**: Precise USD cost calculations based on configurable per-1K token pricing
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

- **Per-1K token pricing** for all dimensions
- **Key aliases** to normalize provider-specific usage keys
- **Context tier multipliers** for pricing based on input size
- **Unpriced token passthrough** captures counts even without pricing

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
        cost_per_1k: 0.15
      output:
        cost_per_1k: 0.60
      extras:
        reasoning_token:
          cost_per_1k: 2.40
        cache_write_token:
          cost_per_1k: 1.25
      key_aliases:
        reasoning_tokens: reasoning_token
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

## Usage Statistics API

The `/api/v1/llm/stats` endpoint returns comprehensive usage and cost statistics with backward compatibility.

## Security & Privacy

- **No prompt storage**: Only token counts and metadata are logged, never actual content
- **No request parameter persistence**: Model parameters and prompts are not stored
- **User attribution**: Optional user association for multi-tenant deployments
- **Error message truncation**: Error messages limited to 500 characters
# Issue 002: Extend llm_config.yaml for Batch Settings

## Summary

Add a `batch` section to `llm_config.yaml` enabling provider/model capability declaration and operational parameters (poll interval, max concurrency, retry defaults, per-model limits).

## Scope

- YAML schema extension & validation (Pydantic settings model update).
- Documentation of new fields.
- Fallback defaults when section missing.

## Proposed YAML Structure

```yaml
batch:
  enabled: true
  default_poll_interval_seconds: 30
  max_concurrent_provider_polls: 5
  defaults:
    retry:
      max_attempts: 5
      base_delay: 2
      max_delay: 60
  providers:
    openai:
      enabled: true
      models:
        - name: gpt-4o-mini
          mode: chat
          max_prompts_per_batch: 20
          pricing_tier: batch
    gemini:
      enabled: true
      models:
        - name: gemini-1.5-flash
          mode: chat
          max_prompts_per_batch: 100
    claude:
      enabled: true
      models:
        - name: claude-3-5-sonnet
          mode: chat
          max_prompts_per_batch: 100
```

## Acceptance Criteria

- Application loads config; disables feature cleanly if `enabled: false`.
- Invalid model entries raise clear validation error.
- Unit tests cover: missing batch section, invalid provider name, invalid numeric bounds.

## Technical Notes

- Provide accessor helper: `BatchConfig.get_provider_models(provider)`.
- Support dynamic reload (optional; out-of-scope if complexity high).

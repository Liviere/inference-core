# Issue 011: Cost & Usage Aggregation

## Summary

Collect usage metrics (token counts, provider cost hints) after batch completion and persist aggregated cost estimates.

## Scope

- Extend `BatchJob` with fields: input_tokens, output_tokens, estimated_cost_minor_units.
- Provider adapters parse usage metadata (when available) or estimate from tokenized inputs + model pricing table in config.
- Expose cost summary in job detail API.

## Acceptance Criteria

- After completion, job has non-null token counts (when provider supplies data).
- Unit test cost calculation for a fixed pricing example.

## Risks

- Provider pricing changes; mitigate by externalizing to config with versioning.

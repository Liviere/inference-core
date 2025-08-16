# Issue 010: Idempotency & Partial Retry Support

## Summary

Add optional idempotency keying and selective re-batching of failed items to improve resilience and cost efficiency.

## Scope

- Hash function (model + sorted inputs) → idempotency key stored on BatchJob.
- Lookup existing completed job by key (if items identical) to short-circuit new submission.
- Command to spawn new BatchJob containing only failed items from a previous job.

## Acceptance Criteria

- Creating identical batch twice returns reference to prior job (if idempotency enabled via config flag).
- Partial retry job links to parent (parent_job_id field) and only contains previously failed items.

## Risks

- Drift if provider normalizes inputs (document requirement that inputs are canonicalized beforehand).

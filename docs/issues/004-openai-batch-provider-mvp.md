# Issue 004: OpenAI Batch Provider (MVP)

## Summary

First concrete provider using OpenAI `/v1/batches` (or equivalent current endpoint) implementing submission, polling, and result retrieval for chat/completion style prompts.

## Scope

- JSONL construction with `custom_id` = BatchItem.id.
- File upload (Files API) if required by current OpenAI batch contract.
- Submit request; capture provider_batch_id.
- Poll status mapping to internal statuses.
- Fetch final result file; parse; map outputs to BatchItems.
- Basic error mapping.

## Out of Scope

- Embeddings batch (follow-up issue if needed).
- Cost extraction (later issue).

## Acceptance Criteria

- End-to-end flow succeeds for ≥3 prompts (integration test with mocked OpenAI HTTP responses).
- Handles provider status transitions (queued→in_progress→completed / failed).
- Partial failure scenario recorded (one item fails, others succeed).

## Technical Notes

- Use HTTP client already in project (validate existing patterns) or `openai` SDK if installed.
- Keep raw provider responses for debug in verbose log at TRACE/DEBUG.

## Risks

- Provider contract shifts (add smoke test in CI against sandbox if feasible).

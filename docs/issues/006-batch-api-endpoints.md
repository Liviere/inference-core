# Issue 006: Public Batch API Endpoints

## Summary

Expose REST endpoints to create batch jobs, query status, list/filter items, and cancel a job.

## Endpoints (Proposed)

- POST `/api/v1/llm/batch` → create job `{provider, model, items:[{input, custom_id?}], params?}`.
- GET `/api/v1/llm/batch/{job_id}` → job detail (aggregated counts, status timeline optional).
- GET `/api/v1/llm/batch/{job_id}/items?status=failed` → paginated items.
- POST `/api/v1/llm/batch/{job_id}/cancel` → attempt provider + local cancel.

## Scope

- Request/response schemas (Pydantic) with validation vs config-supported provider/model.
- Pagination for items (limit/offset or cursor).
- Input size limits (number of items, prompt length) with clear error.

## Acceptance Criteria

- 201 on create returns job_id + initial status.
- Unauthorized requests blocked (reuse existing auth layer).
- Cancel updates status appropriately (if provider supports cancel).

## Risks

- Large payloads; consider streaming or requiring client side chunking.

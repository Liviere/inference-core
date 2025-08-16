# Issue 001: Batch Domain Model & Database Migration

## Summary

Introduce persistent entities for batch processing: `BatchJob`, `BatchItem`, optional `BatchEvent`. Provide initial Alembic migration (or equivalent) and Pydantic schemas where needed for API.

## Scope

- SQLAlchemy models + metadata.
- Migration script creation.
- Basic repository/service helper methods (create job, add items, update statuses, query pending jobs).
- Indexes for common queries (status, provider, created_at).

## Out of Scope

- Provider interaction logic.
- API endpoints (handled in later issue).

## Data Model (Proposed Fields)

`BatchJob`:

- id (UUID PK)
- provider (string)
- model (string)
- status (enum)
- provider_batch_id (nullable)
- mode (string: chat|embedding|completion|custom)
- submitted_at, completed_at
- request_count, success_count, error_count
- config_json (JSON)
- error_summary (text)
- result_uri (string / nullable)
- metadata_json (JSON)

`BatchItem`:

- id (UUID PK)
- batch_job_id (FK)
- sequence_index (int)
- custom_external_id (string nullable)
- input_payload (JSON / text)
- output_payload (JSON / text nullable)
- status (enum: queued|sent|completed|failed)
- error_detail (text nullable)

`BatchEvent` (optional initial):

- id, batch_job_id, type, old_status, new_status, event_ts, meta_json

## Acceptance Criteria

- Migration applies cleanly on fresh DB & existing DB.
- CRUD operations covered by unit tests.
- Enum statuses centralized to avoid divergence.

## Technical Notes

- Consider small composite index: (status, provider) for polling queries.
- Use UTC timestamps; set default with server_default=func.now().
- Keep JSON fields provider-agnostic (avoid nesting provider names).

## Tests

- Create job with N items; assert counts.
- Update item statuses; aggregate counts propagate to job.
- Query pending jobs returns only expected statuses.

## Risks

- Migration conflicts: coordinate with other pending migrations.

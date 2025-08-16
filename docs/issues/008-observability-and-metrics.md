# Issue 008: Observability & Metrics for Batch System

## Summary

Add structured logging, metrics, and optional Sentry breadcrumbs for batch lifecycle events.

## Scope

- Metrics: jobs_total{status}, items_total{status}, job_duration_seconds (histogram), poll_cycle_seconds, provider_latency_seconds.
- Logging: standardized fields (job_id, provider, status_transition).
- Optional: Sentry breadcrumb on status change; error capture with enriched context.

## Acceptance Criteria

- Metrics endpoint shows new counters after at least one job run (if Prometheus enabled in project).
- Log entries easily filterable by `batch_job` tag.
- Error sample includes provider_batch_id when available.

## Risks

- Excessive log volume on large jobs; implement log rate limiting for item-level debug.

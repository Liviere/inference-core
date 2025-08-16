# Issue 005: Celery Tasks for Batch Lifecycle

## Summary

Add Celery tasks orchestrating batch job lifecycle: submission, polling, fetching results, finalization, and repair/retry. Integrate with configuration (interval, concurrency limits).

## Scope

- Tasks: `batch_submit(job_id)`, `batch_poll()`, `batch_fetch(job_id)`, `batch_retry_failed(job_id)`.
- Celery Beat schedule for `batch_poll` (config-driven interval).
- Concurrency guard (e.g. Redis lock/semaphore) to prevent duplicate pollers.
- Logging + metrics for task durations and outcomes.

## Acceptance Criteria

- Submitting a job enqueues submission task automatically.
- Poller discovers in-progress jobs and advances status.
- Fetch runs once per terminal provider status.
- Idempotent execution (rerun poll or fetch does not corrupt state).

## Technical Notes

- Use exponential backoff retry for transient provider errors (respect config retry policy).
- Consider marking large jobs with dynamic poll interval (future enhancement).

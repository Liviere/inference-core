# Issue 009: Retry & Rate Limit Policies

## Summary

Implement configurable retry & rate limit handling for submission, polling, and fetch phases with exponential backoff and jitter.

## Scope

- Policy classes: `RetryPolicy`, `RateLimitPolicy`.
- Central wrapper to apply policy around provider calls.
- Distinguish permanent vs transient provider errors (mapping table).
- Respect per-provider global concurrency limits (config-based).

## Acceptance Criteria

- Simulated 429 triggers backoff and eventual success within max attempts (test).
- Permanent error (e.g. 400 invalid model) fails fast (no >1 retry).
- Metrics include retry counts.

## Risks

- Over-aggressive retries worsening rate limits; enforce cap.

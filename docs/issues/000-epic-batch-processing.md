# EPIC: Provider-Native LLM Batch Processing Framework

## Summary

Design and implement an extensible, provider-agnostic batch processing subsystem for OpenAI, Gemini, Claude (initial set), enabling cost‑efficient, large-scale asynchronous LLM workloads using provider native batch APIs (vs current per-request LangChain batching). Includes domain model, provider abstraction, Celery orchestration, configuration, observability, and follow-on enhancements.

## Motivation

Current codebase supports synchronous or client-side micro-batching via LangChain `.batch()`, which does not leverage provider server-side batch price/performance advantages (discounted token rates, higher throughput, offline processing). Introducing a unified batch job framework unlocks: lower cost for bulk tasks, predictable large-scale offline processing, and a path to add future providers (e.g., Grok) with minimal code duplication.

## Goals

- Unified domain model (BatchJob / BatchItem / BatchEvent).
- Pluggable provider strategy layer (OpenAI, Gemini, Claude baseline).
- Celery-driven lifecycle (submit → poll → fetch → finalize) with retry & resilience.
- YAML-driven configuration (enable/disable providers, per-model limits, polling cadence, retry policy, token cost hints).
- Minimal public API endpoints for creating & querying batch jobs.
- Observability (logs, metrics, structured status events) and error transparency.

## Non-Goals (Now)

- Real-time streaming of provider batch progress (can follow later via websockets/SSE).
- UI frontend components.
- Advanced scheduling optimization or cost prediction ML.
- Cross-job dependency graphs.

## High-Level Deliverables

1. DB schema + migration.
2. Configuration schema extension (`llm_config.yaml`).
3. Provider base & OpenAI provider (MVP path).
4. Celery tasks (submit, poll, fetch, finalize, repair).
5. Public API endpoints.
6. Gemini & Claude provider implementations.
7. Observability & metrics.
8. Retry & rate limit policies.
9. Idempotency & partial retry capabilities.
10. Adaptive polling & performance improvements.
11. Cost & usage aggregation, reporting.
12. Privacy & PII masking safeguards.

## Acceptance Criteria (Epic Completion)

- All listed deliverables merged; integration tests pass; documentation added.
- Can submit a batch (≥ 10 prompts) to OpenAI, see status progress, retrieve outputs.
- At least one successful end-to-end batch for each supported provider.
- Metrics exposed for job counts & durations.
- Error scenarios (provider 4xx, 5xx, partial failures) covered with tests.

## Risks / Mitigations

| Risk                   | Impact                | Mitigation                                |
| ---------------------- | --------------------- | ----------------------------------------- |
| Provider API evolution | Breaks adapters       | Versioned provider modules + smoke tests  |
| Rate limit bursts      | Job failures / delays | Backoff policy + concurrency caps         |
| Large result files     | Memory spikes         | Streamed parsing / chunked writes         |
| Partial failures       | Data inconsistency    | Distinct item status + selective re-batch |
| PII logging            | Compliance risk       | Central redact helper on logging pipeline |

## Success Metrics

- ≥ 95% batch jobs complete without manual intervention.
- Cost per 1K tokens reduced vs synchronous path (tracked after deployment).
- Median fetch-to-finalize latency < configurable SLA (e.g. 2× provider median).

## Follow-Up Extensions (Post-Epic)

- Webhook callbacks / SSE notifications.
- Multi-stage pipelines (chain of batch jobs).
- Dynamic model selection based on cost/latency targets.
- Self-service cost dashboard.

## References

- Internal analysis notes (LangChain vs native batch APIs).
- Provider docs: OpenAI (`/v1/batches`), Gemini Batch Mode, Anthropic Batch Processing.

## Tracking

Use child issues `001`–`013` for initial execution; add more as needed.

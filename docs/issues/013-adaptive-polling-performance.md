# Issue 013: Adaptive Polling & Performance Optimizations

## Summary

Optimize polling strategy to reduce unnecessary provider requests and improve latency for fast-completing batches.

## Scope

- Adaptive interval algorithm (e.g., exponential increase until first progress, then clamp).
- Batch size aware poll scheduling (larger jobs less frequent, small jobs more frequent early).
- Skip polling for jobs already scheduled for fetch.

## Acceptance Criteria

- Simulation test: adaptive strategy reduces poll calls vs fixed interval baseline (>20% reduction) while keeping average completion detection delay within SLA.
- Config flag to revert to fixed interval.

## Risks

- Complexity vs real gain; ensure metrics prove benefit before defaulting on.

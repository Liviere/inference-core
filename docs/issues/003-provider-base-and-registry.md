# Issue 003: Provider Base Class & Registry

## Summary

Implement `BaseBatchProvider` strategy interface and a registry mechanism to dynamically look up provider implementations using provider string keys.

## Scope

- Define abstract methods: supports_model, prepare_payloads, submit, poll_status, fetch_results, cancel.
- Introduce typed DTOs (PreparedSubmission, ProviderSubmitResult, ProviderStatus, ProviderResultRow).
- Implement a simple `BatchProviderRegistry` with register/get/list.
- Error classes: ProviderTransientError, ProviderPermanentError.

## Acceptance Criteria

- Registry can register ≥1 provider and retrieve by name.
- Unregistered provider access raises defined exception.
- Unit tests for interface contract (mock subclass).

## Technical Notes

- Keep provider modules under `app/llm/batch/providers/`.
- Add `PROVIDER_NAME` constant per provider.

## Risks

- Overdesign early; keep DTOs minimal and evolve.

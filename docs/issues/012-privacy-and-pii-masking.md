# Issue 012: Privacy & PII Masking

## Summary

Introduce centralized masking/redaction for sensitive prompt content in logs, events, and error traces.

## Scope

- Redaction utility (patterns: emails, phone numbers, UUIDs, custom regex list from config).
- Apply in: logging interceptors, provider error logs, status events.
- Config flag to disable (for local dev).

## Acceptance Criteria

- Log lines containing test email/phone are masked (unit test with capture).
- Raw stored input/output payloads remain unmodified (only logs sanitized).

## Risks

- Over-redaction harming debuggability (provide debug override env var for secure sandbox only).

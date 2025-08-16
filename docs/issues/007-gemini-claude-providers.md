# Issue 007: Gemini & Claude Batch Providers

## Summary

Add provider implementations for Google Gemini and Anthropic Claude native batch modes, reusing base provider abstractions.

## Scope

- Implement submission, polling, result fetch & parsing.
- Handle provider-specific limits (max prompts per batch, file size, model-specific constraints).
- Update docs & config examples.

## Acceptance Criteria

- Integration tests (mocked HTTP) for each provider complete success & partial failure paths.
- Config toggling (disable provider) prevents submission.

## Technical Notes

- Normalize output to internal schema: `output_text`, `raw_metadata`.
- Distinguish transient vs permanent errors based on HTTP codes/rate-limit headers.

## Risks

- Divergent authentication flows (ensure environment variables / settings documented).

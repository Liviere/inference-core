#!/usr/bin/env python
"""Run the `agent_server_direct` Locust performance profile with optional overrides."""

from __future__ import annotations

from scripts.perf_tests.perf_launcher import main

if __name__ == "__main__":
    raise SystemExit(main(default_profile="agent_server_direct"))

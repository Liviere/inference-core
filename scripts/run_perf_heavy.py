#!/usr/bin/env python
"""Run the `heavy` Locust performance profile with optional overrides."""

from __future__ import annotations

from perf_launcher import main

if __name__ == "__main__":
    raise SystemExit(main(default_profile="heavy"))

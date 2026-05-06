"""Helpers for selecting the project dotenv file consistently.

WHY: Local entrypoints (FastAPI, Celery, LangGraph-related imports) need one
explicit rule for choosing between ``.env`` and ``.env.test``. Resolving the
path from the repository root keeps that behavior stable regardless of the
current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_dotenv_path() -> Path:
    """Return the dotenv file path chosen by the early ``ENVIRONMENT`` flag.

    WHY: Tests and test-like local tasks rely on setting ``ENVIRONMENT`` before
    importing application modules. This helper preserves that contract while
    avoiding duplicated path-selection logic.
    """

    env_filename = ".env.test" if os.getenv("ENVIRONMENT") == "testing" else ".env"
    return PROJECT_ROOT / env_filename


def load_project_dotenv(*, override: bool = False) -> Path:
    """Load the selected project dotenv file for CLI entrypoints and workers.

    WHY: ``python-dotenv`` resolves relative paths against the working
    directory. Loading the absolute repository-root file keeps service startup
    predictable for VS Code tasks, tests, and direct CLI execution.
    """

    dotenv_path = get_project_dotenv_path()
    load_dotenv(dotenv_path=dotenv_path, override=override)
    return dotenv_path

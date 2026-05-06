#!/usr/bin/env python
"""
Regenerate the ``graphs`` sections of LangGraph config files from ``llm_config.yaml``.

WHY: ``agent_graphs.py`` is YAML-driven — every agent selected by
``server_graph`` / ``execution_mode: remote`` is exposed as a module-level
attribute. For the LangGraph CLI to serve them, each local config file must
list the same names under ``graphs``. Running this script after editing
``llm_config.yaml`` keeps the default and testing configs in sync without
hand-editing JSON.

Usage:
    python scripts/sync_langgraph_json.py            # write changes
    python scripts/sync_langgraph_json.py --check    # fail if out of sync (CI)

The script only touches the ``graphs`` field; every other field in each
config file is preserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the project root importable when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from inference_core.agents.graph_registry import (  # noqa: E402
    _should_build_server_graph,
)
from inference_core.llm.config import get_llm_config  # noqa: E402

LANGGRAPH_CONFIGS = (
    _PROJECT_ROOT / "langgraph.json",
    _PROJECT_ROOT / "langgraph.test.json",
)
AGENT_GRAPHS_MODULE = "./agent_graphs.py"


def compute_graphs_section() -> dict[str, str]:
    """Return the desired ``graphs`` mapping derived from YAML."""
    config = get_llm_config()
    selected = {
        name: f"{AGENT_GRAPHS_MODULE}:{name}"
        for name, agent_config in (config.agent_configs or {}).items()
        if _should_build_server_graph(agent_config)
    }
    return dict(sorted(selected.items()))


def load_langgraph_configs() -> dict[Path, dict[str, object]]:
    """Load each managed LangGraph config so graphs can be updated consistently.

    WHY: The testing task uses a separate config file from normal development.
    Updating both files together prevents their graph registries from drifting.
    """

    loaded_configs: dict[Path, dict[str, object]] = {}
    for config_path in LANGGRAPH_CONFIGS:
        if not config_path.exists():
            print(f"ERROR: {config_path} not found", file=sys.stderr)
            raise SystemExit(2)

        with config_path.open("r", encoding="utf-8") as file_handle:
            loaded_configs[config_path] = json.load(file_handle)

    return loaded_configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when langgraph.json is out of sync; no changes written.",
    )
    args = parser.parse_args()

    desired_graphs = compute_graphs_section()
    loaded_configs = load_langgraph_configs()

    out_of_sync_configs = {
        config_path: current
        for config_path, current in loaded_configs.items()
        if current.get("graphs", {}) != desired_graphs
    }

    if not out_of_sync_configs:
        print("LangGraph config files are already in sync with llm_config.yaml.")
        return 0

    if args.check:
        print("LangGraph config files are OUT OF SYNC with llm_config.yaml.", file=sys.stderr)
        for config_path, current in out_of_sync_configs.items():
            current_graphs = current.get("graphs", {})
            print(f"  {config_path.name} current graphs: {sorted(current_graphs.keys())}", file=sys.stderr)
            print(f"  {config_path.name} desired graphs: {sorted(desired_graphs.keys())}", file=sys.stderr)
        print("  Run: python scripts/sync_langgraph_json.py", file=sys.stderr)
        return 1

    for config_path, current in out_of_sync_configs.items():
        current["graphs"] = desired_graphs
        with config_path.open("w", encoding="utf-8") as file_handle:
            json.dump(current, file_handle, indent="\t", ensure_ascii=False)
            file_handle.write("\n")

    updated_names = ", ".join(config_path.name for config_path in out_of_sync_configs)
    print(f"Updated {updated_names} with graphs: {sorted(desired_graphs.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

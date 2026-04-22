#!/usr/bin/env python
"""
Regenerate the ``graphs`` section of ``langgraph.json`` from ``llm_config.yaml``.

WHY: ``agent_graphs.py`` is YAML-driven — every agent selected by
``server_graph`` / ``execution_mode: remote`` is exposed as a module-level
attribute. For the LangGraph CLI to serve them, ``langgraph.json`` must
list the same names under ``graphs``. Running this script after editing
``llm_config.yaml`` keeps both in sync without hand-editing JSON.

Usage:
    python scripts/sync_langgraph_json.py            # write changes
    python scripts/sync_langgraph_json.py --check    # fail if out of sync (CI)

The script only touches the ``graphs`` field; every other field in
``langgraph.json`` is preserved.
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

LANGGRAPH_JSON = _PROJECT_ROOT / "langgraph.json"
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when langgraph.json is out of sync; no changes written.",
    )
    args = parser.parse_args()

    if not LANGGRAPH_JSON.exists():
        print(f"ERROR: {LANGGRAPH_JSON} not found", file=sys.stderr)
        return 2

    with LANGGRAPH_JSON.open("r", encoding="utf-8") as f:
        current = json.load(f)

    desired_graphs = compute_graphs_section()
    current_graphs = current.get("graphs", {})

    if current_graphs == desired_graphs:
        print("langgraph.json already in sync with llm_config.yaml.")
        return 0

    if args.check:
        print("langgraph.json is OUT OF SYNC with llm_config.yaml.", file=sys.stderr)
        print(f"  current graphs: {sorted(current_graphs.keys())}", file=sys.stderr)
        print(f"  desired graphs: {sorted(desired_graphs.keys())}", file=sys.stderr)
        print(
            "  Run: python scripts/sync_langgraph_json.py",
            file=sys.stderr,
        )
        return 1

    current["graphs"] = desired_graphs
    with LANGGRAPH_JSON.open("w", encoding="utf-8") as f:
        json.dump(current, f, indent="\t", ensure_ascii=False)
        f.write("\n")

    print(f"Updated {LANGGRAPH_JSON} with graphs: {sorted(desired_graphs.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""A draft ``models:`` entry for a model in the catalog.

Adding a model to the config stays a decision somebody makes. This only
saves the typing: it writes down what the provider's listing says and marks
what it does not say, prices above all, since a model without prices is used
without its cost being known.
"""

import textwrap
from typing import Any, Dict, List

import yaml

from inference_core.database.sql.models.model_catalog import LLMCatalogModel

from .reconcile import as_utc
from .types import ModelLifecycle


def render_model_entry(row: LLMCatalogModel) -> str:
    """YAML for ``row`` as an entry under ``models:``, with notes as comments."""
    attributes: Dict[str, Any] = row.attributes or {}
    capabilities = attributes.get("capabilities") or {}
    pricing = attributes.get("pricing") or {}

    entry: Dict[str, Any] = {"provider": row.provider}
    if attributes.get("display_name"):
        entry["display_name"] = attributes["display_name"]
    if attributes.get("max_output_tokens"):
        entry["max_tokens"] = attributes["max_output_tokens"]
    if isinstance(capabilities.get("vision"), bool):
        entry["multimodal"] = capabilities["vision"]
    if "input" in pricing and "output" in pricing:
        entry["pricing"] = {
            "currency": "USD",
            "input": {"cost_per_1m": pricing["input"]},
            "output": {"cost_per_1m": pricing["output"]},
        }
        if "cache_read" in pricing:
            entry["pricing"]["extras"] = {
                "cache_read_tokens": {"cost_per_1m": pricing["cache_read"]}
            }

    notes: List[str] = [f"From the listing of '{row.provider}'. Check it before use."]
    if "max_tokens" not in entry:
        notes.append("max_tokens: not in the listing; the default applies.")
    if attributes.get("context_window"):
        notes.append(f"Context window: {attributes['context_window']} tokens.")
    if "pricing" not in entry:
        notes.append(
            "pricing: not in the listing. Add it from the provider's price list, "
            "or the model's calls are logged without a cost."
        )
    if capabilities.get("reasoning") is True:
        notes.append("The model reasons; add reasoning_config if agents should use it.")
    warning = _lifecycle_warning(row)
    if warning:
        notes.append(warning)

    comments = "\n".join(
        textwrap.fill(note, width=76, initial_indent="# ", subsequent_indent="#   ")
        for note in notes
    )
    body = yaml.safe_dump(
        {row.model_id: entry}, sort_keys=False, default_flow_style=False
    )
    return textwrap.indent(f"{comments}\n{body}", "  ")


def _lifecycle_warning(row: LLMCatalogModel) -> str:
    retires_at = as_utc(row.retires_at)
    if row.lifecycle == ModelLifecycle.RETIRED.value:
        return "WARNING: the provider has retired this model."
    if retires_at is not None:
        return f"WARNING: the provider retires this model on {retires_at:%Y-%m-%d}."
    if row.lifecycle == ModelLifecycle.DEPRECATED.value:
        replacement = f"; it names {row.replacement} instead" if row.replacement else ""
        return f"WARNING: the provider has deprecated this model{replacement}."
    return ""

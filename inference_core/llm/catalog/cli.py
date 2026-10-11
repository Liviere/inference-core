"""Command line for the model catalog.

    python -m inference_core.llm.catalog refresh [--provider NAME ...]
    python -m inference_core.llm.catalog status
    python -m inference_core.llm.catalog drift [--check]
    python -m inference_core.llm.catalog events [--since 7d] [--provider NAME]

``refresh`` reads the providers now, whether or not the scheduled refresh is
switched on and whenever they were last read. The others read what is stored
and ask no provider. ``drift --check`` exits with 1 when a configured model
needs a look, for use in a deploy check.
"""

import argparse
import asyncio
import re
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence

from inference_core.core.env import load_project_dotenv

_SINCE = re.compile(r"(\d+)([dh])")


def _since(value: str) -> timedelta:
    match = _SINCE.fullmatch(value.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError("use a number of days or hours: 7d, 24h")
    amount = int(match.group(1))
    return timedelta(days=amount) if match.group(2) == "d" else timedelta(hours=amount)


def _when(value: Optional[datetime]) -> str:
    from .reconcile import as_utc

    value = as_utc(value)
    return value.strftime("%Y-%m-%d %H:%M") if value else "-"


def _day(value: Optional[datetime]) -> str:
    return value.strftime("%Y-%m-%d") if value else "-"


def _table(rows: Sequence[Sequence[str]]) -> None:
    if not rows:
        return
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())


async def _refresh(args: argparse.Namespace) -> int:
    from inference_core.core.config import get_settings
    from inference_core.llm.config import get_llm_config

    from .service import OUTCOME_FAILED, refresh

    settings = get_settings()
    results = await refresh(
        get_llm_config(),
        providers=args.provider or None,
        interval=timedelta(seconds=settings.llm_model_catalog_refresh_interval_seconds),
        read_timeout=float(settings.llm_model_catalog_http_timeout_seconds),
    )
    if not results:
        print("No provider with a configured model has a model lister.")
        return 0
    rows = []
    for result in results:
        if result.outcome == "refreshed":
            detail = f"{result.model_count} models, {result.events} changes"
        else:
            detail = result.detail or ""
        rows.append([result.provider, result.outcome, detail])
    _table(rows)
    return 1 if any(result.outcome == OUTCOME_FAILED for result in results) else 0


async def _status(args: argparse.Namespace) -> int:
    from inference_core.llm.config import get_llm_config

    from .service import plan_providers, provider_states

    plans = plan_providers(get_llm_config())
    states = {state.provider: state for state in await provider_states()}
    if not plans:
        print("No provider with a configured model has a model lister.")
        return 0
    rows = [["PROVIDER", "CONFIGURED", "LISTED", "LAST READ", "STATE"]]
    for plan in plans:
        state = states.get(plan.provider)
        if plan.skip_reason:
            note = f"skipped: {plan.skip_reason}"
        elif state is None or state.last_attempt_at is None:
            note = "not read yet"
        elif state.last_error:
            note = f"failed: {state.last_error}"
        else:
            note = "ok"
        rows.append(
            [
                plan.provider,
                str(len(plan.configured)),
                str(state.model_count) if state and state.last_success_at else "-",
                _when(state.last_success_at if state else None),
                note,
            ]
        )
    _table(rows)
    return 0


async def _drift(args: argparse.Namespace) -> int:
    from inference_core.llm.config import get_llm_config

    from .drift import STATE_DEPRECATED, STATE_MISSING, compute_drift
    from .service import catalog_models, plan_providers, provider_states

    config = get_llm_config()
    states = await provider_states()
    entries = compute_drift(config, await catalog_models(), states)

    read = {state.provider for state in states if state.baselined_at is not None}
    unread = [
        plan.provider
        for plan in plan_providers(config)
        if not plan.skip_reason and plan.provider not in read
    ]
    if unread:
        # No drift from a provider nobody read is not the same as no drift.
        print(f"Not read yet, so not checked: {', '.join(unread)}", file=sys.stderr)

    if not entries:
        print("Every configured model of the providers read is current.")
        return 0

    now = datetime.now(timezone.utc)
    rows = [["PROVIDER", "MODEL", "STATE", "DETAIL"]]
    for entry in entries:
        if entry.state == STATE_MISSING:
            detail = "the provider does not know this model"
        elif entry.state == STATE_DEPRECATED:
            detail = "deprecated"
            if entry.deprecated_at is not None:
                detail += f" {_day(entry.deprecated_at)}"
        elif entry.retires_at is not None and entry.retires_at <= now:
            detail = f"retirement date {_day(entry.retires_at)} has passed"
        elif entry.retires_at is not None:
            detail = f"retires {_day(entry.retires_at)}"
        else:
            detail = "retired"
        if entry.replacement:
            detail += f", replaced by {entry.replacement}"
        rows.append([entry.provider, entry.model, entry.state, detail])
    _table(rows)
    return 1 if args.check else 0


async def _events(args: argparse.Namespace) -> int:
    from .service import catalog_events

    events = await catalog_events(
        since=datetime.now(timezone.utc) - args.since, provider=args.provider
    )
    if not events:
        print("No changes in that time.")
        return 0
    _table(
        [
            [_when(event.detected_at), event.provider, event.event_type, event.model_id]
            for event in events
        ]
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inference_core.llm.catalog",
        description="What the providers' model listings say about the LLM config.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    refresh = commands.add_parser("refresh", help="read the providers' listings now")
    refresh.add_argument(
        "--provider",
        action="append",
        metavar="NAME",
        help="read only this provider (may be given more than once)",
    )
    refresh.set_defaults(run=_refresh)

    status = commands.add_parser("status", help="when each provider was last read")
    status.set_defaults(run=_status)

    drift = commands.add_parser(
        "drift", help="configured models that are missing, deprecated or retiring"
    )
    drift.add_argument(
        "--check", action="store_true", help="exit with 1 when there is any"
    )
    drift.set_defaults(run=_drift)

    events = commands.add_parser("events", help="what changed in the listings")
    events.add_argument(
        "--since",
        type=_since,
        default=timedelta(days=7),
        metavar="AGE",
        help="how far back, as days or hours: 7d (default), 24h",
    )
    events.add_argument("--provider", metavar="NAME", help="only this provider")
    events.set_defaults(run=_events)

    return parser


async def _run(args: argparse.Namespace) -> int:
    from inference_core.database.sql.connection import close_database

    try:
        return await args.run(args)
    finally:
        await close_database()


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    load_project_dotenv()
    return asyncio.run(_run(args))

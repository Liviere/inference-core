#!/usr/bin/env python
"""Launch Locust performance profiles through short scenario wrappers.

WHY:
    Operators currently have to copy long Locust commands with repeated env
    flags and report-path conventions. This launcher centralizes that assembly
    so each public wrapper script can expose a compact, scenario-focused CLI
    without duplicating Locust invocation logic.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERFORMANCE_DIR = PROJECT_ROOT / "tests" / "performance"
LOCUSTFILE_PATH = PERFORMANCE_DIR / "locustfile.py"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "performance"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env.test"


def _load_performance_config_module() -> ModuleType:
    """Load the shared performance config module from its file path.

    WHY:
        The wrapper scripts live outside `tests/performance`, so loading the
        module by path avoids implicit import-path mutations and keeps static
        analysis aligned with the actual runtime behavior.
    """

    config_path = PERFORMANCE_DIR / "config.py"
    module_spec = importlib.util.spec_from_file_location(
        "performance_config",
        config_path,
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Could not load performance config from {config_path}")

    config_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(config_module)
    return config_module


PERFORMANCE_CONFIG = _load_performance_config_module()
LOAD_PROFILES = PERFORMANCE_CONFIG.LOAD_PROFILES
get_host_url = PERFORMANCE_CONFIG.get_host_url
get_llm_mock_allowed_embedding_backends = (
    PERFORMANCE_CONFIG.get_llm_mock_allowed_embedding_backends
)
get_profile = PERFORMANCE_CONFIG.get_profile


def _strip_inline_comment(value: str) -> str:
    """Remove unquoted inline comments from a dotenv value.

    WHY:
        The launcher loads `.env.test` without depending on an extra dotenv
        package, so it needs a small parser that still respects quoted values
        and ignores trailing comments written for humans.
    """

    quote_char: str | None = None
    escaped = False

    for index, character in enumerate(value):
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if quote_char is None and character in {"'", '"'}:
            quote_char = character
            continue
        if quote_char == character:
            quote_char = None
            continue
        if quote_char is None and character == "#":
            return value[:index].rstrip()

    return value.strip()


def _unquote_env_value(value: str) -> str:
    """Unwrap single- or double-quoted dotenv values.

    WHY:
        Test env files commonly quote strings with spaces or punctuation. The
        launcher only needs predictable plain-string values for subprocess env
        injection and default host resolution.
    """

    stripped = value.strip()
    if (
        len(stripped) >= 2
        and stripped[0] == stripped[-1]
        and stripped[0]
        in {
            "'",
            '"',
        }
    ):
        return stripped[1:-1]
    return stripped


def _load_env_file(env_file_path: Path) -> dict[str, str]:
    """Load key-value pairs from a dotenv-style file when it exists.

    WHY:
        Performance wrapper scripts should inherit the dedicated test runtime
        configuration from `.env.test` automatically, so operators do not need
        to repeat host and guardrail settings on every run.
    """

    if not env_file_path.exists():
        return {}

    loaded_values: dict[str, str] = {}
    for raw_line in env_file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            continue
        normalized_value = _unquote_env_value(_strip_inline_comment(value))
        loaded_values[normalized_key] = normalized_value

    return loaded_values


def _apply_env_defaults(env_values: dict[str, str]) -> None:
    """Seed process environment defaults from the loaded test env file.

    WHY:
        Parser defaults such as the target host are resolved before the Locust
        subprocess exists. Applying env defaults early keeps CLI help, runtime
        defaults, and subprocess configuration aligned.
    """

    for key, value in env_values.items():
        os.environ.setdefault(key, value)

    if os.environ.get("TARGET_HOST"):
        return

    host = os.environ.get("HOST", "").strip()
    port = os.environ.get("PORT", "").strip()
    scheme = os.environ.get("TARGET_SCHEME", "http").strip() or "http"
    if not host and not port:
        return
    normalized_host = host or "localhost"
    if normalized_host in {"0.0.0.0", "::", "[::]"}:
        normalized_host = "localhost"
    normalized_port = port or "8000"
    os.environ.setdefault(
        "TARGET_HOST", f"{scheme}://{normalized_host}:{normalized_port}"
    )


def _build_dated_reports_dir() -> Path:
    """Return the default dated directory for generated performance artifacts.

    WHY:
        Grouping reports by day reduces manual cleanup and keeps repeated runs
        easier to browse than a single flat `reports/performance` directory.
    """

    today = date.today()
    return DEFAULT_REPORTS_DIR / f"{today:%Y}" / f"{today:%m}" / f"{today:%d}"


def _resolve_output_path(path: Path) -> Path:
    """Resolve report paths from the project root for stable launcher output.

    WHY:
        Operators may run wrapper scripts from any working directory. Resolving
        relative paths against the repository root keeps report locations
        predictable across shells, CI, and IDE tasks.
    """

    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _normalize_name_suffix(name_suffix: Optional[str]) -> str:
    """Return a filesystem-friendly suffix for generated report names.

    WHY:
        Wrapper scripts should support lightweight run labeling without forcing
        operators to handcraft full report paths for every smoke or comparison
        run.
    """

    if not name_suffix:
        return ""
    cleaned = "-".join(name_suffix.strip().split())
    return f"_{cleaned}" if cleaned else ""


def _build_default_report_path(
    profile_name: str,
    output_dir: Path,
    name_suffix: Optional[str],
    rotation_suffix: str,
) -> Path:
    """Build the default HTML report path for a given profile.

    WHY:
        The existing repo convention names reports by profile. Preserving that
        convention in the wrapper keeps generated artifacts easy to find while
        still allowing per-run suffixes when operators need separate outputs.
    """

    suffix = _normalize_name_suffix(name_suffix)
    return output_dir / f"{profile_name}{suffix}_load_report{rotation_suffix}.html"


def _build_default_csv_prefix(
    profile_name: str,
    output_dir: Path,
    name_suffix: Optional[str],
    rotation_suffix: str,
) -> Path:
    """Build the default Locust CSV prefix for a given profile.

    WHY:
        Locust's `--csv` flag expects a prefix rather than a final filename.
        A shared builder keeps HTML and CSV outputs aligned under the same
        profile naming convention.
    """

    suffix = _normalize_name_suffix(name_suffix)
    return output_dir / f"{profile_name}{suffix}_results{rotation_suffix}"


def _csv_prefix_has_existing_artifacts(prefix_path: Path) -> bool:
    """Return whether a Locust CSV prefix already has generated files.

    WHY:
        Locust materializes several CSV files from one prefix. Rotation should
        treat that family as one artifact set rather than checking only the raw
        prefix path, which is not normally created as a standalone file.
    """

    if prefix_path.exists():
        return True
    return any(prefix_path.parent.glob(f"{prefix_path.name}*.csv"))


def _resolve_rotation_suffix(
    html_base_path: Path | None,
    csv_base_prefix: Path | None,
) -> str:
    """Choose a shared rotation suffix for the current report artifact set.

    WHY:
        When a profile is run multiple times on the same day, the HTML and CSV
        outputs should stay grouped under the same rotated name instead of each
        artifact family incrementing independently.
    """

    attempt = 1
    while True:
        suffix = "" if attempt == 1 else f"_{attempt:02d}"
        html_candidate = None
        csv_candidate = None

        if html_base_path is not None:
            html_candidate = html_base_path.with_name(
                f"{html_base_path.stem}{suffix}{html_base_path.suffix}"
            )
        if csv_base_prefix is not None:
            csv_candidate = csv_base_prefix.with_name(f"{csv_base_prefix.name}{suffix}")

        html_exists = bool(html_candidate and html_candidate.exists())
        csv_exists = bool(
            csv_candidate and _csv_prefix_has_existing_artifacts(csv_candidate)
        )
        if not html_exists and not csv_exists:
            return suffix
        attempt += 1


def _create_parser(default_profile: str | None) -> argparse.ArgumentParser:
    """Create the CLI parser shared by all performance wrapper scripts.

    WHY:
        Every scenario wrapper should expose the same override surface so users
        can switch between profiles without relearning argument names.
    """

    description = (
        f"Run the '{default_profile}' Locust performance profile with optional overrides."
        if default_profile
        else "Run a Locust performance profile with optional overrides."
    )
    parser = argparse.ArgumentParser(description=description)

    if default_profile is None:
        parser.add_argument(
            "--profile",
            choices=sorted(LOAD_PROFILES.keys()),
            required=True,
            help="Performance profile to run.",
        )

    parser.add_argument(
        "--host",
        default=get_host_url(),
        help=(
            "Target API host. Defaults to TARGET_HOST or derives from HOST and PORT, "
            "including values loaded from .env.test."
        ),
    )
    parser.add_argument(
        "--users",
        type=int,
        default=None,
        help="Override the profile's default user count.",
    )
    parser.add_argument(
        "--spawn-rate",
        type=float,
        default=None,
        help="Override the profile's default user spawn rate.",
    )
    parser.add_argument(
        "--duration",
        default=None,
        help="Override the profile's default Locust run time (for example 30s, 5m).",
    )

    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        default=True,
        help="Run Locust headlessly. This is the default.",
    )
    execution_mode.add_argument(
        "--web-ui",
        dest="headless",
        action="store_false",
        help="Run Locust with the web UI instead of headless mode.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_build_dated_reports_dir(),
        help=(
            "Directory for generated HTML and CSV reports. Defaults to "
            "reports/performance/YYYY/MM/DD."
        ),
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Explicit HTML report path. Relative paths are resolved from the repo root.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Disable the default HTML report output for this run.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Generate Locust CSV output using the default profile-based prefix.",
    )
    parser.add_argument(
        "--csv-prefix",
        type=Path,
        default=None,
        help="Explicit CSV output prefix. Relative paths are resolved from the repo root.",
    )
    parser.add_argument(
        "--name-suffix",
        default=None,
        help="Optional suffix appended to default report filenames.",
    )
    parser.add_argument(
        "--skip-auth-preflight",
        action="store_true",
        help="Set LOCUST_SKIP_AUTH_PREFLIGHT=true for this run.",
    )

    allowed_embedding_backends = get_llm_mock_allowed_embedding_backends()
    parser.add_argument(
        "--embedding-backend",
        choices=allowed_embedding_backends,
        default=None,
        help=(
            "Embedding backend override for llm_mock. Defaults to the current "
            "EMBEDDING_BACKEND env var or fake when unset."
        ),
    )
    parser.add_argument(
        "--allow-unsafe-llm-traffic",
        action="store_true",
        help="Set LOCUST_ALLOW_UNSAFE_LLM_TRAFFIC=true for llm_mock runs.",
    )

    return parser


def _validate_args(args: argparse.Namespace, profile_name: str) -> None:
    """Reject scenario-specific flags that do not apply to the selected profile.

    WHY:
        Fast failure keeps the wrapper CLI honest. A user should not think that
        `--embedding-backend local` affected `light` when the selected profile
        has no LLM mock traffic at all.
    """

    if profile_name != "llm_mock":
        if args.embedding_backend is not None:
            raise SystemExit(
                "--embedding-backend is only supported for the llm_mock profile"
            )
        if args.allow_unsafe_llm_traffic:
            raise SystemExit(
                "--allow-unsafe-llm-traffic is only supported for the llm_mock profile"
            )


def _build_locust_env(
    profile_name: str,
    args: argparse.Namespace,
) -> dict[str, str]:
    """Build the environment variables passed to the Locust subprocess.

    WHY:
        Performance presets need a small amount of environment orchestration
        (`LOAD_PROFILE`, llm_mock guardrails, auth preflight flags) that should
        stay consistent regardless of how the wrapper is invoked.
    """

    env = os.environ.copy()
    env["LOAD_PROFILE"] = profile_name

    if args.skip_auth_preflight:
        env["LOCUST_SKIP_AUTH_PREFLIGHT"] = "true"

    if profile_name == "llm_mock":
        embedding_backend = (
            args.embedding_backend or env.get("EMBEDDING_BACKEND") or "fake"
        )
        env["LLM_EMULATION_ENABLED"] = env.get("LLM_EMULATION_ENABLED", "true")
        env["EMBEDDING_BACKEND"] = embedding_backend
        if args.allow_unsafe_llm_traffic:
            env["LOCUST_ALLOW_UNSAFE_LLM_TRAFFIC"] = "true"

    return env


def _build_locust_command(
    profile_name: str,
    args: argparse.Namespace,
) -> tuple[list[str], Path | None, Path | None]:
    """Build the final Locust command line and derived report paths.

    WHY:
        Wrapper scripts should be a thin orchestration layer over Locust rather
        than a forked execution model. Returning the exact command keeps runtime
        behavior transparent and easy to debug.
    """

    profile = get_profile(profile_name)
    output_dir = _resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_html_base = (
        None
        if args.no_html
        else output_dir
        / (f"{profile_name}{_normalize_name_suffix(args.name_suffix)}_load_report.html")
    )
    default_csv_base = None
    if args.csv or args.csv_prefix is not None:
        default_csv_base = output_dir / (
            f"{profile_name}{_normalize_name_suffix(args.name_suffix)}_results"
        )

    rotation_suffix = ""
    if args.html is None and args.csv_prefix is None:
        rotation_suffix = _resolve_rotation_suffix(default_html_base, default_csv_base)

    html_path: Path | None = None
    if not args.no_html:
        html_path = (
            _resolve_output_path(args.html)
            if args.html is not None
            else _build_default_report_path(
                profile_name,
                output_dir,
                args.name_suffix,
                rotation_suffix,
            )
        )
        html_path.parent.mkdir(parents=True, exist_ok=True)

    csv_prefix: Path | None = None
    if args.csv or args.csv_prefix is not None:
        csv_prefix = (
            _resolve_output_path(args.csv_prefix)
            if args.csv_prefix is not None
            else _build_default_csv_prefix(
                profile_name,
                output_dir,
                args.name_suffix,
                rotation_suffix,
            )
        )
        csv_prefix.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(LOCUSTFILE_PATH),
        "--host",
        args.host,
    ]

    if args.headless:
        command.extend(
            [
                "--headless",
                "--users",
                str(args.users if args.users is not None else profile.users),
                "--spawn-rate",
                str(
                    args.spawn_rate
                    if args.spawn_rate is not None
                    else profile.spawn_rate
                ),
                "--run-time",
                str(args.duration if args.duration is not None else profile.run_time),
            ]
        )

    if html_path is not None:
        command.extend(["--html", str(html_path)])

    if csv_prefix is not None:
        command.extend(["--csv", str(csv_prefix)])

    return command, html_path, csv_prefix


def _print_run_summary(
    profile_name: str,
    args: argparse.Namespace,
    command: Sequence[str],
    html_path: Path | None,
    csv_prefix: Path | None,
    env: dict[str, str],
    loaded_env_file: Path | None,
) -> None:
    """Print the effective launcher configuration before execution.

    WHY:
        Wrapper scripts hide command assembly on purpose, so a compact summary
        gives operators enough visibility to debug overrides without reopening
        the implementation.
    """

    print(f"Profile: {profile_name}")
    if loaded_env_file is not None:
        print(f"Env file: {loaded_env_file}")
    print(f"Mode: {'headless' if args.headless else 'web-ui'}")
    print(f"Host: {args.host}")
    if args.headless:
        profile = get_profile(profile_name)
        print(f"Users: {args.users if args.users is not None else profile.users}")
        print(
            "Spawn rate: "
            f"{args.spawn_rate if args.spawn_rate is not None else profile.spawn_rate}"
        )
        print(
            f"Duration: {args.duration if args.duration is not None else profile.run_time}"
        )
    if html_path is not None:
        print(f"HTML report: {html_path}")
    if csv_prefix is not None:
        print(f"CSV prefix: {csv_prefix}")
    if profile_name == "llm_mock":
        print(f"Embedding backend: {env['EMBEDDING_BACKEND']}")
        print(f"LLM emulation: {env['LLM_EMULATION_ENABLED']}")
        if env.get("LOCUST_ALLOW_UNSAFE_LLM_TRAFFIC") == "true":
            print("Unsafe LLM traffic override: enabled")
    if env.get("LOCUST_SKIP_AUTH_PREFLIGHT") == "true":
        print("Auth preflight: skipped")
    print(f"Command: {shlex.join(command)}")


def main(
    default_profile: str | None = None,
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Run a Locust performance profile through the shared launcher.

    WHY:
        Every public preset wrapper should call one place that owns argument
        parsing, guardrail env setup, report naming, and subprocess execution.
    """

    loaded_env_file: Path | None = None
    env_file_values = _load_env_file(DEFAULT_ENV_FILE)
    if env_file_values:
        _apply_env_defaults(env_file_values)
        loaded_env_file = DEFAULT_ENV_FILE

    parser = _create_parser(default_profile)
    args = parser.parse_args(list(argv) if argv is not None else None)

    profile_name = default_profile or args.profile
    _validate_args(args, profile_name)
    env = _build_locust_env(profile_name, args)
    command, html_path, csv_prefix = _build_locust_command(profile_name, args)
    _print_run_summary(
        profile_name,
        args,
        command,
        html_path,
        csv_prefix,
        env,
        loaded_env_file,
    )

    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

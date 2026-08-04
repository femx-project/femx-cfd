"""Command-line interface for portable femx-cfd cases."""

from __future__ import annotations

import argparse
from functools import lru_cache
from importlib import resources
import json
from pathlib import Path
import sys
import traceback
from typing import Optional, Sequence

import yaml

from femx_cfd.cases import load_case
from femx_cfd.runtime.inspection import inspect_case
from femx_cfd.runtime.runner import run_case


def _run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("case", help="path to a portable case YAML file")
    parser.add_argument(
        "-o",
        "--output",
        help="result directory relative to the current directory (overrides output.directory)",
    )
    parser.add_argument(
        "--progress",
        choices=("console", "json"),
        default="console",
        help="progress output format",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show a traceback when the run fails",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="femx-cfd",
        description="Validate and run portable femx-cfd cases.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    _run_arguments(commands.add_parser("run", help="run one case"))
    validate = commands.add_parser("validate", help="validate one case")
    validate.add_argument("case", help="path to a portable case YAML file")
    inspect = commands.add_parser("inspect", help="inspect one case without running it")
    inspect.add_argument("case", help="path to a portable case YAML file")
    inspect.add_argument(
        "-o",
        "--output",
        help="override the configured result directory for this inspection",
    )
    example = commands.add_parser("example", help="run a bundled example case")
    example.add_argument("kind", choices=tuple(_example_cases()))
    example.add_argument("-o", "--output")
    example.add_argument(
        "--progress",
        choices=("console", "json"),
        default="console",
    )
    example.add_argument("--debug", action="store_true")
    return parser


@lru_cache
def _example_cases() -> dict[str, Path]:
    """Load bundled example names and case paths from their manifest."""

    root = Path(str(resources.files("femx_cfd.examples"))).resolve()
    manifest = yaml.safe_load((root / "index.yaml").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise RuntimeError("invalid bundled example manifest version")
    entries = manifest.get("examples")
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError("bundled example manifest must define examples")

    cases = {}
    for name, value in entries.items():
        if not isinstance(name, str) or not name:
            raise RuntimeError("bundled example names must be nonempty strings")
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"bundled example '{name}' must define a case path")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"bundled example '{name}' has an invalid case path")
        path = (root / relative).resolve()
        if not path.is_file():
            raise RuntimeError(f"bundled example '{name}' case does not exist: {path}")
        cases[name] = path
    return cases


def _report_error(error: Exception, progress_kind: str, debug: bool) -> None:
    if progress_kind == "json":
        print(json.dumps({"type": "error", "message": str(error)}), flush=True)
    else:
        print(f"error: {error}", file=sys.stderr, flush=True)
    if debug:
        traceback.print_exc()


def _execute_run(
    args: argparse.Namespace,
) -> int:
    try:
        run_case(
            args.case,
            args.output,
            progress_kind=args.progress,
            finalize_runtime=True,
        )
        return 0
    except Exception as error:
        _report_error(error, args.progress, args.debug)
        return 1


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    """Run the generic femx-cfd CLI and return its process status."""

    args = _parser().parse_args(argv)
    if args.command == "validate":
        try:
            case = load_case(args.case)
        except Exception as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        message = f"valid: {case.format} version {case.source_version}"
        if case.source_version != case.version:
            message += f" (migrates to version {case.version})"
        print(message)
        return 0
    if args.command == "inspect":
        try:
            report = inspect_case(args.case, args.output)
        except Exception as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        version = str(report["source_version"])
        if report["source_version"] != report["version"]:
            version += f" -> {report['version']}"
        rows = (
            ("case", report["case"]),
            ("format", report["format"]),
            ("version", version),
            ("method", report["method"]),
            (
                "mesh",
                f"{report['mesh_nodes']} nodes, {report['mesh_elements']} elements",
            ),
            ("steps", report["steps"]),
            ("output", report["output_directory"]),
            ("output state", report["output_state"]),
            ("output every", report["output_every"]),
            ("output interval", f"{report['output_interval']:.6g} s"),
            ("output alignment", "exact"),
            ("output levels", report["output_levels"]),
            ("Maximum inlet speed", f"{report['peak_inlet_speed']:.6g} m/s"),
            (
                "Estimated hydraulic diameter",
                f"{report['inlet_characteristic_length']:.6g} m",
            ),
            (
                "Estimated maximum Reynolds number",
                f"{report['reynolds']:.6g}",
            ),
            (
                "Estimated mean Reynolds number",
                f"{report['mean_reynolds']:.6g}",
            ),
            ("Estimated maximum inlet CFL", f"{report['cfl']:.6g}"),
        )
        width = max(len(label) for label, _ in rows)
        for label, value in rows:
            print(f"{label:<{width}}  {value}")
        return 1 if report["output_state"] == "occupied" else 0
    if args.command == "example":
        args.case = str(_example_cases()[args.kind])
        if args.output is None:
            args.output = str(Path.cwd() / "run" / f"{args.kind}-example")
        return _execute_run(args)
    return _execute_run(args)


def main() -> None:
    raise SystemExit(run_cli())


__all__ = [
    "main",
    "run_cli",
]


if __name__ == "__main__":
    main()

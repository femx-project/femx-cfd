"""Command-line interface for portable femx-cfd cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Optional, Sequence

from femx_cfd.cases import FORWARD_FORMAT, load_case
from femx_cfd.runtime.runner import run_case


def _run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("case", help="path to a portable case YAML file")
    parser.add_argument(
        "-o",
        "--output",
        help="result directory (default: result beside the case file)",
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
    example = commands.add_parser("example", help="run a bundled example case")
    example.add_argument("kind", choices=("forward", "var", "envar"))
    example.add_argument("-o", "--output")
    example.add_argument(
        "--progress",
        choices=("console", "json"),
        default="console",
    )
    example.add_argument("--debug", action="store_true")
    return parser


def _forward_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="femx-forward",
        description="Run one femx-cfd forward-simulation case.",
    )
    _run_arguments(parser)
    return parser


def _example_case(kind: str) -> Path:
    examples = Path(__file__).parent / "examples"
    if kind == "forward":
        return examples / "forward" / "straight_tube" / "case.yaml"
    return examples / "inverse" / "straight_tube" / f"{kind}.yaml"


def _report_error(error: Exception, progress_kind: str, debug: bool) -> None:
    if progress_kind == "json":
        print(json.dumps({"type": "error", "message": str(error)}), flush=True)
    else:
        print(f"error: {error}", file=sys.stderr, flush=True)
    if debug:
        traceback.print_exc()


def _execute_run(
    args: argparse.Namespace,
    expected_format: str | None = None,
    expected_method: str | None = None,
) -> int:
    try:
        loaded = load_case(args.case)
        if expected_format is not None and loaded.format != expected_format:
            raise ValueError(
                f"expected case format '{expected_format}', got '{loaded.format}'"
            )
        if (
            expected_method is not None
            and loaded.resolved.get("method") != expected_method
        ):
            raise ValueError(
                f"expected inverse method '{expected_method}', "
                f"got '{loaded.resolved.get('method')}'"
            )
        run_case(
            loaded.path,
            args.output,
            progress_kind=args.progress,
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
        print(f"valid: {case.format} version {case.version}")
        return 0
    if args.command == "example":
        args.case = str(_example_case(args.kind))
        if args.output is None:
            args.output = str(Path.cwd() / "run" / f"{args.kind}-example")
        return _execute_run(args)
    return _execute_run(args)


def run_forward_cli(argv: Optional[Sequence[str]] = None) -> int:
    """Run the backward-compatible femx-forward command."""

    args = _forward_parser().parse_args(argv)
    return _execute_run(args, FORWARD_FORMAT)


def run_inverse_cli(
    method: str,
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Run a method-specific compatibility command for an inverse case."""

    parser = argparse.ArgumentParser(
        prog=f"femx-{method}",
        description=f"Run one femx-cfd {method} inverse case.",
    )
    _run_arguments(parser)
    args = parser.parse_args(argv)
    return _execute_run(args, "femx-cfd-inverse", method)


def main() -> None:
    raise SystemExit(run_cli())


def forward_main() -> None:
    raise SystemExit(run_forward_cli())


def var_main() -> None:
    raise SystemExit(run_inverse_cli("var"))


def envar_main() -> None:
    raise SystemExit(run_inverse_cli("envar"))


__all__ = [
    "envar_main",
    "forward_main",
    "main",
    "run_cli",
    "run_forward_cli",
    "run_inverse_cli",
    "var_main",
]


if __name__ == "__main__":
    main()

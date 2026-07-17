"""Run a portable femx-cfd forward-simulation case."""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from omegaconf import OmegaConf

from apps.common import artifacts, parallel, progress
from femx_cfd.forward import load_forward_case, solve_forward


STATUS_FORMAT = "femx-cfd-forward-status"
RESULT_FORMAT = "femx-cfd-forward-result"
OUTPUT_VERSION = 1


def _parser():
    parser = argparse.ArgumentParser(
        prog="femx-forward",
        description="Run one femx-cfd forward-simulation case.",
    )
    parser.add_argument("case", help="path to the forward case YAML file")
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
    return parser


def _output_path(case_path, value):
    if value is None:
        return case_path.parent / "result"
    output = Path(value).expanduser()
    if not output.is_absolute():
        output = case_path.parent / output
    return output.resolve()


def _write_json(path, value):
    with artifacts._atomic_path(path) as temporary:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _write_status(output, state, **values):
    _write_json(
        output / "status.json",
        {
            "format": STATUS_FORMAT,
            "version": OUTPUT_VERSION,
            "state": state,
            **values,
        },
    )


def _prepare_output(output, case):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"result directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with artifacts._atomic_path(output / "case.yaml") as temporary:
        OmegaConf.save(OmegaConf.create(case), temporary)
    _write_status(output, "running")


def _write_result(output, solution):
    artifacts.write_truth(output / "flow.h5", solution.model, solution.trajectory)
    _write_json(
        output / "result.json",
        {
            "format": RESULT_FORMAT,
            "version": OUTPUT_VERSION,
            "case": "case.yaml",
            "flow": "flow.xdmf",
            "data": "flow.h5",
        },
    )
    _write_status(output, "complete", result="result.json")


def _progress_sink(kind, enabled):
    if not enabled:
        return progress.NullProgress()
    if kind == "json":
        return progress.JsonProgress()
    return progress.ConsoleProgress()


def run_cli(argv=None):
    args = _parser().parse_args(argv)
    case_path = Path(args.case).expanduser().resolve()
    output = _output_path(case_path, args.output)
    runtime = None
    output_owned = False

    try:
        case = load_forward_case(case_path)
        petsc_options = case["solver"].get("petsc_options")
        if petsc_options:
            os.environ.setdefault("PETSC_OPTIONS", petsc_options)

        runtime = parallel.for_case(case)
        if runtime.is_root:
            _prepare_output(output, case)
            output_owned = True

        with _progress_sink(args.progress, runtime.is_root) as prog:
            prog.stage("forward: solving Navier-Stokes trajectory")
            solution = solve_forward(case, progress=prog)
            runtime.barrier()
            if runtime.is_root:
                prog.stage("output: writing flow.h5 and flow.xdmf")
                _write_result(output, solution)
                prog.stage(f"complete: {output}")
        return 0
    except Exception as error:
        if runtime is None or runtime.is_root:
            if output_owned:
                _write_status(output, "failed", error=str(error))
            if args.progress == "json":
                print(
                    json.dumps(
                        {
                            "type": "error",
                            "message": str(error),
                        }
                    ),
                    flush=True,
                )
            else:
                print(f"error: {error}", file=sys.stderr, flush=True)
            if args.debug:
                traceback.print_exc()
        return 1
    finally:
        if runtime is not None:
            runtime.finalize()


def main():
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()

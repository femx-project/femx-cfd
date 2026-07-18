"""Run the Var or EnVar pulsatile-flow demo."""

from __future__ import annotations

import subprocess
import sys
from typing import Optional


_COMMON_STEPS = (
    (
        "mesh",
        "apps.mesh.main",
        "demo/mesh",
        ("case.mesh.overwrite=true",),
    ),
    ("truth inlet", "apps.inlet.main", "demo/truth_inlet", ()),
    ("observations", "apps.prepare.main", "demo/prepare", ()),
    ("initial inlet", "apps.inlet.main", "demo/inlet", ()),
)

_ANALYSIS_STEPS = {
    "var": ("Var", "apps.var.main", "demo/var", ()),
    "envar": ("EnVar", "apps.envar.main", "demo/envar", ()),
}

_RULE = "=" * 64


def commands(
    analysis: str,
    python: Optional[str] = None,
) -> list[tuple[str, list[str]]]:
    """Return the subprocess commands used by one analysis demo."""
    try:
        analysis_step = _ANALYSIS_STEPS[analysis]
    except KeyError as error:
        raise ValueError(f"unknown demo analysis: {analysis}") from error

    steps = (*_COMMON_STEPS, analysis_step)
    executable = sys.executable if python is None else python
    return [
        (
            label,
            [
                executable,
                "-m",
                module,
                "--config-name",
                config_name,
                *overrides,
            ],
        )
        for label, module, config_name, overrides in steps
    ]


def _run(analysis: str) -> int:
    name = f"femx-{analysis}-demo"
    steps = commands(analysis)
    for index, (label, command) in enumerate(steps, start=1):
        print(
            f"\n{_RULE}\n{name}  [{index}/{len(steps)}]  {label}\n{_RULE}\n",
            flush=True,
        )
        result = subprocess.run(command, check=False)
        if result.returncode:
            print(
                f"\n{name}: {label} failed (exit code {result.returncode})\n",
                flush=True,
            )
            return result.returncode
        print(f"\nCompleted: {label}\n", flush=True)

    print(f"{_RULE}\n{name} complete\n{_RULE}", flush=True)
    return 0


def var_main() -> int:
    """Run preparation followed by Var."""
    return _run("var")


def envar_main() -> int:
    """Run preparation followed by EnVar."""
    return _run("envar")


if __name__ == "__main__":
    raise SystemExit(var_main())

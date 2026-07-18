"""Regenerate the small packaged Var and EnVar example inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from femx_cfd.data import save_inlet, save_velocity
from femx_cfd.prep.inlet import make_inlet
from synthetic.mesh import write_straight_tube
from synthetic.observation import make_velocity_observations


def _fixed_conditions() -> list[dict[str, object]]:
    return [
        {
            "boundary": "wall",
            "field": "velocity",
            "components": "all",
            "space": {"kind": "uniform", "value": [0.0, 0.0]},
            "time": {"kind": "uniform"},
        },
        {
            "boundary": "outlet",
            "field": "pressure",
            "space": {"kind": "uniform", "value": 0.0},
            "time": {"kind": "uniform"},
        },
    ]


def _portable_boundaries() -> list[dict[str, object]]:
    return [
        {"boundary": "wall", "kind": "no_slip"},
        {"boundary": "outlet", "kind": "pressure", "value": 0.0},
    ]


def _base_case(method: str) -> dict[str, object]:
    case: dict[str, object] = {
        "format": "femx-cfd-inverse",
        "version": 1,
        "method": method,
        "mesh": {"path": "mesh.msh", "length_unit": "m"},
        "fluid": {"rho": 1.0, "mu": 0.01},
        "time": {"period": 1.0, "dt": 0.25, "end_time": 1.0},
        "control": {"boundary": "inlet", "kind": "normal"},
        "boundary_conditions": _portable_boundaries(),
        "observations": {"kind": "velocity", "path": "observations.h5"},
        "initialization": {"inlet": {"source": "file", "path": "initial_inlet.h5"}},
        "solver": {"backend": "dense"},
    }
    if method == "var":
        case["algorithm"] = {
            "optimizer": "scipy",
            "max_iterations": 3,
            "gradient_tolerance": 1.0e-3,
            "function_tolerance": 1.0e-12,
            "control_bounds": [0.0, 0.2],
            "regularization": {
                "background": 0.01,
                "space": 0.001,
                "time": 0.01,
            },
        }
    else:
        case["algorithm"] = {"prior_weight": 0.25}
        case["ensemble"] = {
            "workers": 1,
            "mode_scales": {
                "flow_mean": 0.2,
                "flow_sin1": 0.2,
            },
        }
    return case


def build(output: Path, force: bool = False) -> None:
    """Create deterministic prepared inputs and portable case files."""

    expected = (
        output / "mesh.msh",
        output / "observations.h5",
        output / "initial_inlet.h5",
        output / "var.yaml",
        output / "envar.yaml",
    )
    existing = [path for path in expected if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"example files already exist: {names}; pass --force")
    output.mkdir(parents=True, exist_ok=True)

    mesh_path = output / "mesh.msh"
    velocity_path = output / "observations.h5"
    inlet_path = output / "initial_inlet.h5"
    write_straight_tube(
        mesh_path,
        length=4.0,
        diameter=1.0,
        cell_size=0.25,
    )

    truth_case = {
        "mesh": str(mesh_path),
        "fluid": {"rho": 1.0, "mu": 0.01},
        "time": {"T": 1.0, "dt": 0.25},
        "simulation": {"end_time": 2.0},
        "control": {"boundary": "inlet", "kind": "normal", "normal": "auto"},
        "bc": [
            {
                "boundary": "inlet",
                "field": "velocity",
                "components": "all",
                "direction": "normal",
                "space": {"kind": "poisson", "peak": 0.1},
                "time": {
                    "kind": "harmonics",
                    "mean": 1.0,
                    "terms": [{"kind": "sin", "order": 1, "amplitude": 0.3}],
                },
            },
            *_fixed_conditions(),
        ],
        "inlet": {"source": "analytic", "T": 1.0, "dt": 0.25},
        "solver": {"backend": "dense"},
        "observation": {
            "lower": [-1.75, -0.4],
            "upper": [1.75, 0.4],
            "counts": [8, 5],
            "phases": 4,
        },
    }
    truth_inlet = make_inlet(truth_case)
    save_velocity(
        velocity_path,
        make_velocity_observations(truth_case, truth_inlet),
    )

    inlet_case = {
        "mesh": str(mesh_path),
        "time": {"T": 1.0},
        "control": {"boundary": "inlet", "kind": "normal", "normal": "auto"},
        "inlet": {
            "source": "velocity",
            "file": str(velocity_path),
            "T": 1.0,
            "projection": {
                "depth": 0.5,
                "axial_decay": 0.25,
                "laplacian_modes": 3,
                "ridge": 0.001,
            },
        },
    }
    save_inlet(inlet_path, make_inlet(inlet_case))

    for method in ("var", "envar"):
        path = output / f"{method}.yaml"
        path.write_text(
            yaml.safe_dump(_base_case(method), sort_keys=False),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "examples"
            / "inverse"
            / "straight_tube"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.output.expanduser().resolve(), force=args.force)


if __name__ == "__main__":
    main()

"""Configuration-driven synthetic inverse-case generation."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from femx_cfd._typing import Config, Dynamic, PathLike
from femx_cfd.data import add_velocity_noise, save_inlet, save_velocity
from femx_cfd.prep.inlet import make_inlet, make_inlet_velocity

from .mesh import write_straight_tube
from .observation import make_velocity_observations


SYNTHETIC_FORMAT = "femx-cfd-synthetic"
SYNTHETIC_VERSION = 1


def _mapping(value: Dynamic, name: str) -> Config:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive(value: Dynamic, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive number") from error
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive number")
    return number


def _nonnegative(value: Dynamic, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a nonnegative number") from error
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a nonnegative number")
    return number


def _positive_integer(value: Dynamic, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if number <= 0 or number != value:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _unit_direction(value: Dynamic, name: str) -> Dynamic:
    if isinstance(value, str):
        if value != "auto":
            raise ValueError(f"{name} must be 'auto' or a vector")
        return value
    try:
        direction = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be 'auto' or a vector") from error
    if direction.shape not in ((2,), (3,)) or not np.all(np.isfinite(direction)):
        raise ValueError(f"{name} must be 'auto' or a finite vector")
    length = np.linalg.norm(direction)
    if length == 0.0:
        raise ValueError(f"{name} must be nonzero")
    return (direction / length).tolist()


def _pair(value: Dynamic, name: str, *, integers: bool = False) -> list[Dynamic]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    if integers:
        return [
            _positive_integer(component, f"{name}[{index}]")
            for index, component in enumerate(value)
        ]
    try:
        pair = [float(component) for component in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain two finite numbers") from error
    if not np.all(np.isfinite(pair)):
        raise ValueError(f"{name} must contain two finite numbers")
    return pair


def validate_synthetic_config(value: Dynamic) -> Config:
    """Validate and normalize one synthetic-generation configuration."""

    cfg = deepcopy(_mapping(value, "synthetic config"))
    if cfg.get("format") != SYNTHETIC_FORMAT:
        raise ValueError(f"format must be '{SYNTHETIC_FORMAT}'")
    if cfg.get("version") != SYNTHETIC_VERSION:
        raise ValueError(f"version must be {SYNTHETIC_VERSION}")
    if "analysis" in cfg:
        raise ValueError("analysis settings belong in independent Var/EnVar cases")

    mesh = _mapping(cfg.get("mesh"), "mesh")
    if mesh.get("kind", "straight_tube") != "straight_tube":
        raise ValueError("mesh.kind must be 'straight_tube'")
    mesh["kind"] = "straight_tube"
    for name in ("length", "diameter", "cell_size"):
        mesh[name] = _positive(mesh.get(name), f"mesh.{name}")

    fluid = _mapping(cfg.get("fluid"), "fluid")
    fluid["rho"] = _positive(fluid.get("rho"), "fluid.rho")
    fluid["mu"] = _positive(fluid.get("mu"), "fluid.mu")

    time = _mapping(cfg.get("time"), "time")
    if "analysis_cycles" in time:
        raise ValueError("analysis time belongs in independent Var/EnVar cases")
    time["period"] = _positive(time.get("period"), "time.period")
    time["dt"] = _positive(time.get("dt"), "time.dt")
    ratio = time["period"] / time["dt"]
    if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1.0e-10):
        raise ValueError("time.period must be an integer multiple of time.dt")
    time["truth_cycles"] = _positive_integer(
        time.get("truth_cycles", 2),
        "time.truth_cycles",
    )
    truth = _mapping(cfg.get("truth"), "truth")
    if "normal" in truth:
        raise ValueError("truth.normal was replaced by truth.direction")
    truth["direction"] = _unit_direction(
        truth.get("direction", "auto"),
        "truth.direction",
    )
    truth["profile"] = _mapping(truth.get("profile"), "truth.profile")
    truth["waveform"] = _mapping(truth.get("waveform"), "truth.waveform")
    truth["solver"] = _mapping(
        truth.get("solver", {"backend": "dense"}),
        "truth.solver",
    )

    observations = _mapping(cfg.get("observations"), "observations")
    observations["lower"] = _pair(observations.get("lower"), "observations.lower")
    observations["upper"] = _pair(observations.get("upper"), "observations.upper")
    if any(
        lower >= upper
        for lower, upper in zip(observations["lower"], observations["upper"])
    ):
        raise ValueError("observations bounds must be increasing")
    observations["counts"] = _pair(
        observations.get("counts"),
        "observations.counts",
        integers=True,
    )
    observations["phases"] = _positive_integer(
        observations.get("phases"),
        "observations.phases",
    )
    noise = _mapping(observations.get("noise", {}), "observations.noise")
    noise["std"] = _nonnegative(noise.get("std", 0.0), "observations.noise.std")
    seed = noise.get("seed", 0)
    if isinstance(seed, bool):
        raise ValueError("observations.noise.seed must be an integer")
    try:
        seed = int(seed)
    except (TypeError, ValueError) as error:
        raise ValueError("observations.noise.seed must be an integer") from error
    if seed != noise.get("seed", 0):
        raise ValueError("observations.noise.seed must be an integer")
    noise["seed"] = seed
    observations["noise"] = noise

    reconstruction = _mapping(cfg.get("reconstruction"), "reconstruction")
    reconstruction["normal"] = _unit_direction(
        reconstruction.get("normal", "auto"),
        "reconstruction.normal",
    )
    for name in ("depth", "axial_decay"):
        reconstruction[name] = _positive(
            reconstruction.get(name),
            f"reconstruction.{name}",
        )
    reconstruction["laplacian_modes"] = _positive_integer(
        reconstruction.get("laplacian_modes"),
        "reconstruction.laplacian_modes",
    )
    for name in ("ridge", "temporal_weight"):
        reconstruction[name] = _nonnegative(
            reconstruction.get(name, 0.0),
            f"reconstruction.{name}",
        )

    return cfg


def load_synthetic_config(path: PathLike) -> Config:
    """Load and validate a synthetic-generation YAML file."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"synthetic config does not exist: {source}")
    return validate_synthetic_config(yaml.safe_load(source.read_text(encoding="utf-8")))


def _fixed_conditions() -> list[Config]:
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


def build_synthetic_case(
    value: Dynamic,
    output: PathLike,
    *,
    force: bool = False,
) -> Path:
    """Generate a mesh, observations, and an inlet reconstruction."""

    cfg = validate_synthetic_config(value)
    output = Path(output).expanduser().resolve()
    expected = tuple(
        output / name
        for name in (
            "mesh.msh",
            "observations.h5",
            "initial_inlet.h5",
        )
    )
    existing = [path for path in expected if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"generated files already exist: {names}; pass --force")
    output.mkdir(parents=True, exist_ok=True)

    mesh_path = output / "mesh.msh"
    velocity_path = output / "observations.h5"
    inlet_path = output / "initial_inlet.h5"
    mesh = cfg["mesh"]
    write_straight_tube(
        mesh_path,
        length=mesh["length"],
        diameter=mesh["diameter"],
        cell_size=mesh["cell_size"],
    )

    time = cfg["time"]
    truth = cfg["truth"]
    truth_case: Config = {
        "mesh": str(mesh_path),
        "fluid": cfg["fluid"],
        "time": {"T": time["period"], "dt": time["dt"]},
        "simulation": {"end_time": time["truth_cycles"] * time["period"]},
        "control": {
            "boundary": "inlet",
            "kind": "normal",
            "normal": truth["direction"],
        },
        "bc": [
            {
                "boundary": "inlet",
                "field": "velocity",
                "components": "all",
                "direction": "normal",
                "space": truth["profile"],
                "time": truth["waveform"],
            },
            *_fixed_conditions(),
        ],
        "inlet": {
            "source": "analytic",
            "T": time["period"],
            "dt": time["dt"],
        },
        "solver": truth["solver"],
        "observation": {
            name: cfg["observations"][name]
            for name in ("lower", "upper", "counts", "phases")
        },
    }
    truth_inlet = make_inlet(truth_case)
    velocity = make_velocity_observations(truth_case, truth_inlet)
    noise = cfg["observations"]["noise"]
    if noise["std"] > 0.0:
        velocity = add_velocity_noise(
            velocity,
            std=noise["std"],
            seed=noise["seed"],
        )
    save_velocity(velocity_path, velocity)

    reconstruction = cfg["reconstruction"]
    inlet_case = {
        "mesh": str(mesh_path),
        "time": {"T": time["period"]},
        "control": {
            "boundary": "inlet",
            "kind": "normal",
            "normal": reconstruction["normal"],
        },
        "inlet": {
            "source": "velocity",
            "file": str(velocity_path),
            "T": time["period"],
            "projection": {
                name: reconstruction[name]
                for name in (
                    "depth",
                    "axial_decay",
                    "laplacian_modes",
                    "ridge",
                    "temporal_weight",
                )
            },
        },
    }
    save_inlet(inlet_path, make_inlet_velocity(inlet_case))

    return output


def build_from_file(
    config: PathLike,
    output: PathLike,
    *,
    force: bool = False,
) -> Path:
    """Generate a synthetic case from a YAML configuration file."""

    return build_synthetic_case(
        load_synthetic_config(config),
        output,
        force=force,
    )


__all__ = [
    "SYNTHETIC_FORMAT",
    "SYNTHETIC_VERSION",
    "build_from_file",
    "build_synthetic_case",
    "load_synthetic_config",
    "validate_synthetic_config",
]

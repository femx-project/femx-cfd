"""Configuration-driven synthetic inverse-case generation."""

from copy import deepcopy
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional

import h5py
import numpy as np
import yaml

from femx_cfd._typing import Config, Dynamic, OptionalProgress, PathLike
from femx_cfd.cases.validation import (
    mapping,
    nonnegative,
    positive,
    positive_integer,
    reject_unknown,
    validate_fluid,
    validate_mesh as validate_file_mesh,
    validate_profile,
    validate_solver,
    validate_waveform,
)
from femx_cfd.data import add_velocity_noise, save_inlet, save_velocity
from femx_cfd.mesh import mesh_in_meters
from femx_cfd.prep.inlet import make_inlet, make_inlet_velocity
from femx_cfd.runtime import parallel
from femx_cfd.runtime.output import (
    write_truth_samples,
    write_velocity_series,
)
from femx_cfd.workflows.boundaries import fixed_conditions

from .mesh import write_straight_tube
from .observation import observation_times, solve_truth_observations


SYNTHETIC_FORMAT = "femx-cfd-synthetic"
SYNTHETIC_VERSION = 2


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
            positive_integer(component, f"{name}[{index}]")
            for index, component in enumerate(value)
        ]
    try:
        pair = [float(component) for component in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain two finite numbers") from error
    if not np.all(np.isfinite(pair)):
        raise ValueError(f"{name} must contain two finite numbers")
    return pair


def _synthetic_mesh(value: Dynamic, base_dir: Optional[PathLike]) -> Config:
    """Validate either a generated primitive or an existing mesh file."""

    if isinstance(value, str):
        return validate_file_mesh(value, base_dir)

    mesh = mapping(value, "mesh")
    if "path" in mesh:
        generated = {"kind", "length", "diameter", "cell_size"}.intersection(mesh)
        if generated:
            names = ", ".join(sorted(generated))
            raise ValueError(
                "mesh.path cannot be combined with generated mesh options: " + names
            )
        reject_unknown(mesh, {"path", "length_unit"}, "mesh")
        return validate_file_mesh(mesh, base_dir)

    if mesh.get("kind", "straight_tube") != "straight_tube":
        raise ValueError("mesh.kind must be 'straight_tube'")
    reject_unknown(mesh, {"kind", "length", "diameter", "cell_size"}, "mesh")
    mesh["kind"] = "straight_tube"
    for name in ("length", "diameter", "cell_size"):
        mesh[name] = positive(mesh.get(name), f"mesh.{name}")
    return mesh


def _validate_boundary_conditions(value: Dynamic) -> list[Config]:
    """Validate one analytic inlet and its fixed boundary conditions."""

    if not isinstance(value, list) or not value:
        raise ValueError("boundary_conditions must be a nonempty list")

    names = set()
    inlet_count = 0
    for index, item in enumerate(value):
        condition = mapping(item, f"boundary_conditions[{index}]")
        boundary = condition.get("boundary")
        if not isinstance(boundary, str) or not boundary:
            raise ValueError(
                f"boundary_conditions[{index}].boundary must be a nonempty string"
            )
        if boundary in names:
            raise ValueError(f"duplicate boundary condition: {boundary}")
        names.add(boundary)

        kind = condition.get("kind")
        analytic = any(
            name in condition for name in ("direction", "profile", "waveform")
        )
        if analytic:
            reject_unknown(
                condition,
                {"boundary", "kind", "direction", "profile", "waveform"},
                f"boundary_conditions[{index}]",
            )
            if kind != "velocity":
                raise ValueError(
                    f"boundary_conditions[{index}] with profile or waveform "
                    "must have kind: velocity"
                )
            if "value" in condition:
                raise ValueError(
                    f"boundary_conditions[{index}] cannot combine value with "
                    "profile or waveform"
                )
            condition["direction"] = _unit_direction(
                condition.get("direction", "auto"),
                f"boundary_conditions[{index}].direction",
            )
            condition["profile"] = validate_profile(
                condition.get("profile"),
                f"boundary_conditions[{index}].profile",
            )
            condition["waveform"] = validate_waveform(
                condition.get("waveform"),
                f"boundary_conditions[{index}].waveform",
            )
            inlet_count += 1
            continue

        if kind == "velocity":
            reject_unknown(
                condition,
                {"boundary", "kind", "value"},
                f"boundary_conditions[{index}]",
            )
            try:
                vector = np.asarray(condition.get("value"), dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"boundary_conditions[{index}].value must contain "
                    "2 or 3 finite values"
                ) from error
            if (
                vector.ndim != 1
                or vector.size not in (2, 3)
                or not np.all(np.isfinite(vector))
            ):
                raise ValueError(
                    f"boundary_conditions[{index}].value must contain "
                    "2 or 3 finite values"
                )
            condition["value"] = vector.tolist()
            continue
        if kind == "pressure":
            reject_unknown(
                condition,
                {"boundary", "kind", "value"},
                f"boundary_conditions[{index}]",
            )
            try:
                pressure = float(condition.get("value"))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"boundary_conditions[{index}].value must be finite"
                ) from error
            if not np.isfinite(pressure):
                raise ValueError(f"boundary_conditions[{index}].value must be finite")
            condition["value"] = pressure
            continue
        raise ValueError(
            f"boundary_conditions[{index}].kind must be 'velocity' or 'pressure'"
        )

    if inlet_count != 1:
        raise ValueError(
            "boundary_conditions must contain exactly one velocity condition "
            "with profile and waveform"
        )
    return value


def _inlet_condition(cfg: Config) -> Config:
    return next(
        condition for condition in cfg["boundary_conditions"] if "profile" in condition
    )


def _fixed_boundary_case(cfg: Config) -> Config:
    return {
        "boundary_conditions": [
            condition
            for condition in cfg["boundary_conditions"]
            if "profile" not in condition
        ]
    }


def validate_synthetic_config(
    value: Dynamic,
    base_dir: Optional[PathLike] = None,
) -> Config:
    """Validate and normalize one synthetic-generation configuration."""

    cfg = deepcopy(mapping(value, "synthetic config"))
    if cfg.get("format") != SYNTHETIC_FORMAT:
        raise ValueError(f"format must be '{SYNTHETIC_FORMAT}'")
    if cfg.get("version") != SYNTHETIC_VERSION:
        raise ValueError(f"version must be {SYNTHETIC_VERSION}")
    allowed = {
        "format",
        "version",
        "output",
        "mesh",
        "fluid",
        "time",
        "boundary_conditions",
        "solver",
        "observations",
        "reconstruction",
    }
    reject_unknown(cfg, allowed, "synthetic config")

    cfg["mesh"] = _synthetic_mesh(cfg.get("mesh"), base_dir)

    fluid = validate_fluid(cfg.get("fluid"))
    cfg["fluid"] = fluid

    time = mapping(cfg.get("time"), "time")
    reject_unknown(time, {"period", "dt", "truth_cycles"}, "time")
    if "analysis_cycles" in time:
        raise ValueError("analysis time belongs in independent Var/EnVar cases")
    time["period"] = positive(time.get("period"), "time.period")
    time["dt"] = positive(time.get("dt"), "time.dt")
    ratio = time["period"] / time["dt"]
    if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1.0e-10):
        raise ValueError("time.period must be an integer multiple of time.dt")
    time["truth_cycles"] = positive_integer(
        time.get("truth_cycles", 2),
        "time.truth_cycles",
    )
    period_steps = int(round(ratio))

    cfg["boundary_conditions"] = _validate_boundary_conditions(
        cfg.get("boundary_conditions"),
    )

    output = mapping(cfg.get("output", {}), "output")
    reject_unknown(output, {"directory", "every"}, "output")
    output["every"] = positive_integer(output.get("every", 1), "output.every")
    if period_steps % output["every"] != 0:
        raise ValueError(
            f"output.every ({output['every']}) must divide the "
            f"{period_steps} integration steps per period"
        )
    directory = output.get("directory")
    if directory is not None:
        if not isinstance(directory, str) or not directory:
            raise ValueError("output.directory must be a nonempty path")
        path = Path(directory).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        output["directory"] = str(path.resolve())
    cfg["output"] = output

    cfg["solver"] = validate_solver(
        cfg.get("solver", {"backend": "dense"}),
    )

    observations = mapping(cfg.get("observations"), "observations")
    reject_unknown(
        observations,
        {
            "lower",
            "upper",
            "counts",
            "mask",
            "phases",
            "selected_phases",
            "noise",
        },
        "observations",
    )
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
    mask = observations.get("mask", "none")
    if mask not in ("none", "mesh"):
        raise ValueError("observations.mask must be 'none' or 'mesh'")
    observations["mask"] = mask
    observations["phases"] = positive_integer(
        observations.get("phases"),
        "observations.phases",
    )
    selected_phases = observations.get(
        "selected_phases",
        list(range(1, observations["phases"] + 1)),
    )
    if not isinstance(selected_phases, list) or not selected_phases:
        raise ValueError("observations.selected_phases must be a nonempty list")
    selected_phases = [
        positive_integer(phase, f"observations.selected_phases[{index}]")
        for index, phase in enumerate(selected_phases)
    ]
    if selected_phases != sorted(set(selected_phases)):
        raise ValueError("observations.selected_phases must be unique and increasing")
    if selected_phases[-1] > observations["phases"]:
        raise ValueError("observations.selected_phases must not exceed phases")
    observations["selected_phases"] = selected_phases
    noise = mapping(observations.get("noise", {}), "observations.noise")
    reject_unknown(
        noise,
        {"std", "snr", "reference", "seed"},
        "observations.noise",
    )
    if "std" in noise and "snr" in noise:
        raise ValueError("observations.noise must specify either std or snr, not both")
    if "snr" in noise:
        noise["snr"] = positive(noise["snr"], "observations.noise.snr")
        reference = noise.get("reference", "rms")
        if reference not in ("rms", "max"):
            raise ValueError("observations.noise.reference must be 'rms' or 'max'")
        noise["reference"] = reference
    else:
        if "reference" in noise:
            raise ValueError("observations.noise.reference requires snr")
        noise["std"] = nonnegative(
            noise.get("std", 0.0),
            "observations.noise.std",
        )
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

    reconstruction = mapping(cfg.get("reconstruction"), "reconstruction")
    reconstruction["normal"] = _unit_direction(
        reconstruction.get("normal", "auto"),
        "reconstruction.normal",
    )
    kind = reconstruction.get("kind", "laplacian")
    if kind not in ("laplacian", "poisson_fourier"):
        raise ValueError("reconstruction.kind must be 'laplacian' or 'poisson_fourier'")
    reconstruction["kind"] = kind
    reconstruction["depth"] = positive(
        reconstruction.get("depth"),
        "reconstruction.depth",
    )
    if kind == "poisson_fourier":
        reject_unknown(
            reconstruction,
            {"kind", "normal", "depth", "fourier_order", "axial_decay"},
            "reconstruction",
        )
        reconstruction["fourier_order"] = positive_integer(
            reconstruction.get("fourier_order"),
            "reconstruction.fourier_order",
        )
        if "axial_decay" in reconstruction:
            reconstruction["axial_decay"] = positive(
                reconstruction["axial_decay"],
                "reconstruction.axial_decay",
            )
    else:
        reject_unknown(
            reconstruction,
            {
                "kind",
                "normal",
                "depth",
                "axial_decay",
                "laplacian_modes",
                "ridge",
                "temporal_weight",
            },
            "reconstruction",
        )
        reconstruction["axial_decay"] = positive(
            reconstruction.get("axial_decay"),
            "reconstruction.axial_decay",
        )
        reconstruction["laplacian_modes"] = positive_integer(
            reconstruction.get("laplacian_modes"),
            "reconstruction.laplacian_modes",
        )
        for name in ("ridge", "temporal_weight"):
            reconstruction[name] = nonnegative(
                reconstruction.get(name, 0.0),
                f"reconstruction.{name}",
            )

    return cfg


def load_synthetic_config(path: PathLike) -> Config:
    """Load and validate a synthetic-generation YAML file."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"synthetic config does not exist: {source}")
    return validate_synthetic_config(
        yaml.safe_load(source.read_text(encoding="utf-8")),
        base_dir=source.parent,
    )


def _materialize_mesh(mesh: Config, target: Path) -> None:
    """Generate or copy one validated synthetic mesh in meter units."""

    if "path" not in mesh:
        write_straight_tube(
            target,
            length=mesh["length"],
            diameter=mesh["diameter"],
            cell_size=mesh["cell_size"],
        )
        return

    source = Path(mesh["path"])
    if source.resolve() == target.resolve():
        if mesh["length_unit"] != "m":
            raise ValueError(
                "mesh.path inside the output directory must use length_unit: m"
            )
        return
    with mesh_in_meters(mesh) as source_in_meters:
        shutil.copyfile(source_in_meters, target)


def _noise_parameters(
    velocity: Dynamic,
    noise: Config,
    *,
    model: Dynamic = None,
    trajectory: Dynamic = None,
) -> tuple[float, float]:
    """Return the selected clean-signal scale and Gaussian noise deviation."""

    if "snr" not in noise:
        signal_scale = float(np.sqrt(np.mean(np.square(velocity.vals))))
    elif noise.get("reference", "rms") == "rms":
        signal_scale = float(np.sqrt(np.mean(np.square(velocity.vals))))
    else:
        if model is None or trajectory is None:
            raise ValueError("max noise reference requires the clean truth trajectory")
        dimension = int(model.mesh.dimension)
        velocity_dofs = np.asarray(model.velocity_dofs, dtype=np.int64).reshape(
            model.mesh.num_nodes,
            dimension,
        )
        signal_scale = max(
            float(np.max(np.linalg.norm(state[velocity_dofs], axis=-1)))
            for state in trajectory.values
        )
    if not np.isfinite(signal_scale):
        raise ValueError("synthetic observation signal scale must be finite")
    if "snr" not in noise:
        return signal_scale, noise["std"]
    if signal_scale == 0.0:
        raise ValueError("observations.noise.snr requires a nonzero signal scale")
    return signal_scale, signal_scale / noise["snr"]


def _write_noise_metadata(
    path: Path,
    noise: Config,
    *,
    signal_scale: float,
    noise_std: float,
) -> None:
    """Record the configured and resolved observation-noise parameters."""

    with h5py.File(path, "a") as h5:
        attrs = h5["velocity"].attrs
        attrs["noise_model"] = "independent_gaussian"
        attrs["noise_specification"] = "snr" if "snr" in noise else "std"
        attrs["noise_seed"] = noise["seed"]
        attrs["noise_std"] = noise_std
        if "snr" in noise:
            attrs["noise_snr"] = noise["snr"]
            attrs["noise_reference"] = noise["reference"]
            attrs["signal_scale"] = signal_scale


def build_synthetic_case(
    value: Dynamic,
    output: Optional[PathLike] = None,
    *,
    force: bool = False,
    progress: OptionalProgress = None,
    finalize_runtime: bool = False,
) -> Path:
    """Generate a mesh, truth fields, observations, and an inlet reconstruction."""

    cfg = validate_synthetic_config(value)
    time = cfg["time"]
    inlet_condition = _inlet_condition(cfg)
    petsc_options = cfg["solver"].get("petsc_options")
    if petsc_options:
        os.environ.setdefault("PETSC_OPTIONS", petsc_options)
    runtime = parallel.for_case(cfg)
    if output is None:
        output = cfg["output"].get("directory")
        if output is None:
            raise ValueError(
                "synthetic output requires output.directory or an explicit output"
            )
    output = Path(output).expanduser().resolve()
    mesh = cfg["mesh"]
    materialize_mesh = "path" not in mesh or mesh["length_unit"] != "m"
    mesh_path = (
        output / "mesh" / "mesh.msh"
        if materialize_mesh
        else Path(mesh["path"]).resolve()
    )
    expected = tuple(
        output / name
        for name in (
            "reference/flow_full.h5",
            "reference/flow_full.xdmf",
            "reference/flow_window.h5",
            "reference/flow_window.xdmf",
            "reference/flow_observation.h5",
            "reference/flow_observation.xdmf",
            "observation/velocity.h5",
            "observation/velocity.pvd",
            "inlet/initial_velocity.h5",
        )
    )
    if materialize_mesh:
        expected = (mesh_path, *expected)
    existing = [path for path in expected if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"generated files already exist: {names}; pass --force")
    velocity_path = output / "observation" / "velocity.h5"
    inlet_path = output / "inlet" / "initial_velocity.h5"
    if runtime.is_root:
        output.mkdir(parents=True, exist_ok=True)
        velocity_path.parent.mkdir(parents=True, exist_ok=True)
        inlet_path.parent.mkdir(parents=True, exist_ok=True)
        if materialize_mesh:
            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            _materialize_mesh(mesh, mesh_path)
    runtime.barrier()

    truth_case: Config = {
        "mesh": str(mesh_path),
        "fluid": cfg["fluid"],
        "time": {"T": time["period"], "dt": time["dt"]},
        "simulation": {"end_time": time["truth_cycles"] * time["period"]},
        "control": {
            "boundary": inlet_condition["boundary"],
            "kind": "normal",
            "normal": inlet_condition["direction"],
        },
        "bc": [
            {
                "boundary": inlet_condition["boundary"],
                "field": "velocity",
                "components": "all",
                "direction": "normal",
                "space": inlet_condition["profile"],
                "time": inlet_condition["waveform"],
            },
            *fixed_conditions(
                _fixed_boundary_case(cfg),
                len(cfg["observations"]["lower"]),
            ),
        ],
        "inlet": {
            "source": "analytic",
            "T": time["period"],
            "dt": time["dt"],
        },
        "solver": cfg["solver"],
        "observation": {
            name: cfg["observations"][name]
            for name in (
                "lower",
                "upper",
                "counts",
                "mask",
                "phases",
                "selected_phases",
            )
        },
    }
    truth_inlet = make_inlet(truth_case)
    model, trajectory, velocity = solve_truth_observations(
        truth_case,
        truth_inlet,
        progress if runtime.is_root else None,
    )
    runtime.barrier()
    try:
        if runtime.is_root:
            noise = cfg["observations"]["noise"]
            signal_scale, noise_std = _noise_parameters(
                velocity,
                noise,
                model=model,
                trajectory=trajectory,
            )
            reference_dir = output / "reference"
            reference_dir.mkdir(parents=True, exist_ok=True)
            window_steps = int(round(time["period"] / time["dt"]))
            every = cfg["output"]["every"]
            full_steps = time["truth_cycles"] * window_steps
            full_times = np.arange(0, full_steps + 1, every) * time["dt"]
            window_times = np.arange(0, window_steps + 1, every) * time["dt"]
            window_offset = (time["truth_cycles"] - 1) * time["period"]
            phase_times = observation_times(truth_case)
            with tempfile.TemporaryDirectory(prefix="femx-cfd-truth-") as tmp_dir:
                complete_truth = Path(tmp_dir) / "flow_complete.h5"
                model.write_xdmf(complete_truth.with_suffix(".xdmf"), trajectory)
                write_truth_samples(
                    complete_truth,
                    reference_dir / "flow_full.h5",
                    full_times,
                    time["dt"],
                )
                write_truth_samples(
                    complete_truth,
                    reference_dir / "flow_window.h5",
                    window_times,
                    time["dt"],
                    source_times=window_times + window_offset,
                )
                write_truth_samples(
                    complete_truth,
                    reference_dir / "flow_observation.h5",
                    phase_times,
                    time["dt"],
                    source_times=phase_times + window_offset,
                )
            if noise_std > 0.0:
                velocity = add_velocity_noise(
                    velocity,
                    std=noise_std,
                    seed=noise["seed"],
                )
            save_velocity(velocity_path, velocity)
            _write_noise_metadata(
                velocity_path,
                noise,
                signal_scale=signal_scale,
                noise_std=noise_std,
            )
            write_velocity_series(velocity_path.with_suffix(".pvd"), velocity)

            reconstruction = cfg["reconstruction"]
            projection = reconstruction.copy()
            projection.pop("normal")
            inlet_case = {
                "mesh": str(mesh_path),
                "time": {"T": time["period"]},
                "control": {
                    "boundary": inlet_condition["boundary"],
                    "kind": "normal",
                    "normal": reconstruction["normal"],
                },
                "inlet": {
                    "source": "velocity",
                    "file": str(velocity_path),
                    "T": time["period"],
                    "dt": time["dt"],
                    "projection": projection,
                },
            }
            save_inlet(inlet_path, make_inlet_velocity(inlet_case))
        runtime.barrier()
        return output
    finally:
        if finalize_runtime:
            runtime.finalize()
        else:
            runtime.collect()


def build_from_file(
    config: PathLike,
    output: Optional[PathLike] = None,
    *,
    force: bool = False,
    progress: OptionalProgress = None,
    finalize_runtime: bool = False,
) -> Path:
    """Generate a synthetic case from a YAML configuration file."""

    source = Path(config).expanduser().resolve()
    cfg = load_synthetic_config(source)
    if output is None:
        output = cfg["output"].get("directory", source.parent)
    return build_synthetic_case(
        cfg,
        output,
        force=force,
        progress=progress,
        finalize_runtime=finalize_runtime,
    )


__all__ = [
    "SYNTHETIC_FORMAT",
    "SYNTHETIC_VERSION",
    "build_from_file",
    "build_synthetic_case",
    "load_synthetic_config",
    "validate_synthetic_config",
]

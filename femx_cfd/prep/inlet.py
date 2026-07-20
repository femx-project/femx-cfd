"""Create a common inlet boundary condition from a formula or MRI."""

from typing import Optional

import femx
import numpy as np

from femx_cfd._typing import Config, Dynamic
from femx_cfd.data import InletData, InletVelocityData, load_velocity

from .mri import project_to_inlet


def inlet_normal(
    mesh: Dynamic,
    surface: Dynamic,
    value: Dynamic = "auto",
) -> np.ndarray:
    """Return a unit inlet normal directed into the volume mesh."""

    dim = mesh.dimension

    if not isinstance(value, str):
        normal = np.asarray(value, dtype=float)
        if normal.shape != (dim,):
            raise ValueError(f"inlet normal must contain {dim} values")
        length = np.linalg.norm(normal)
        if not np.isfinite(length) or length == 0.0:
            raise ValueError("inlet normal must be finite and nonzero")
        return normal / length

    if value != "auto":
        raise ValueError("inlet normal must be 'auto' or a vector")

    coords = np.asarray(surface.coordinates)[:, :dim]
    center = coords.mean(axis=0)
    centered = coords - center

    _, axes = np.linalg.eigh(centered.T @ centered)
    normal = axes[:, 0]

    mesh_center = np.asarray(mesh.coordinates)[:, :dim].mean(axis=0)
    if (mesh_center - center) @ normal < 0.0:
        normal = -normal

    return normal


def _periodic(
    times: np.ndarray,
    vals: np.ndarray,
    period: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isclose(times[0], 0.0):
        raise ValueError("periodic inlet data must start at t=0")
    if times[-1] > period or np.isclose(times[-1], period):
        if not np.isclose(times[-1], period):
            raise ValueError("inlet times must not exceed the period")
        vals = vals.copy()
        vals[-1] = vals[0]
        return times, vals
    return (
        np.append(times, period),
        np.concatenate((vals, vals[:1]), axis=0),
    )


def _waveform(times: np.ndarray, period: float, cfg: Config) -> np.ndarray:
    if cfg.get("kind", "harmonics") == "uniform":
        return np.full(times.size, float(cfg.get("value", 1.0)))
    if cfg.get("kind", "harmonics") != "harmonics":
        raise ValueError("inlet time kind must be 'uniform' or 'harmonics'")

    vals = np.full(times.size, float(cfg.get("mean", 1.0)))
    omega = 2.0 * np.pi / period
    for harmonic in cfg.get("terms", cfg.get("harmonics", [])):
        order = int(harmonic["order"])
        amp = float(harmonic["amplitude"])
        phase = float(harmonic.get("phase", 0.0))
        angle = order * omega * times + phase
        kind = harmonic.get("kind", "sin")
        if kind == "sin":
            vals += amp * np.sin(angle)
        elif kind == "cos":
            vals += amp * np.cos(angle)
        else:
            raise ValueError("harmonic kind must be 'sin' or 'cos'")
    return vals


def analytic_inlet(
    surface: Dynamic,
    cfg: Config,
    normal: np.ndarray,
    boundary: str,
    *,
    space: Config,
    time: Config,
) -> InletData:
    """Evaluate an analytic inlet specified by spatial and temporal profiles."""

    period = float(cfg["T"])
    dt = float(cfg["dt"])
    count = int(round(period / dt))
    if count <= 0 or not np.isclose(count * dt, period):
        raise ValueError("inlet period must be an integer multiple of dt")
    times = np.linspace(0.0, period, count + 1)

    if space["kind"] == "poisson":
        profile = surface.poisson_profile(normalize=False)
        skew = float(space.get("skew", 0.0))
        if skew:
            coord = np.asarray(surface.coordinates)[:, 1]
            coord = coord - 0.5 * (coord.min() + coord.max())
            coord /= np.max(np.abs(coord))
            profile *= 1.0 + skew * coord
        profile *= float(space["peak"]) / profile.max()
    elif space["kind"] == "uniform":
        profile = np.full(
            np.asarray(surface.coordinates).shape[0],
            float(space["value"]),
        )
    else:
        raise ValueError("analytic inlet space kind must be 'poisson' or 'uniform'")
    vals = _waveform(times, period, time)[:, None] * profile
    active = surface.interior_node_ids

    return InletData(
        boundary=boundary,
        times=times,
        node_ids=np.asarray(surface.mesh_node_ids)[active],
        coords=np.asarray(surface.coordinates)[active],
        normal=normal,
        vals=vals[:, active],
        periodic=True,
        source="analytic",
    )


def repeat_inlet(
    inlet: InletData,
    cycles: int,
    end_time: Optional[float] = None,
) -> InletData:
    """Repeat one periodic inlet over multiple cycles."""

    period = inlet.times[-1] - inlet.times[0]
    times = [inlet.times]
    vals = [inlet.vals]
    for cycle in range(1, cycles):
        times.append(inlet.times[1:] + cycle * period)
        vals.append(inlet.vals[1:])
    times = np.concatenate(times)
    if end_time is not None:
        times[-1] = end_time

    return InletData(
        boundary=inlet.boundary,
        times=times,
        node_ids=inlet.node_ids,
        coords=inlet.coords,
        normal=inlet.normal,
        vals=np.vstack(vals),
        periodic=False,
        source=inlet.source,
    )


def normal_inlet(
    inlet: InletVelocityData,
    normal: np.ndarray,
    boundary: Optional[str] = None,
) -> InletData:
    """Extract a prepared scalar normal control from inlet vector velocity."""

    if not isinstance(inlet, InletVelocityData):
        raise TypeError("inlet must be InletVelocityData")
    if boundary is None:
        boundary = inlet.boundary
    if inlet.boundary != boundary:
        raise ValueError("inlet HDF5 boundary does not match the configured boundary")
    normal = np.asarray(normal, dtype=float)
    if normal.shape != (inlet.vals.shape[2],):
        raise ValueError("inlet normal and vector velocity dimensions do not match")
    length = np.linalg.norm(normal)
    if not np.isfinite(length) or length == 0.0:
        raise ValueError("inlet normal must be finite and nonzero")
    normal = normal / length
    return InletData(
        boundary=boundary,
        times=inlet.times,
        node_ids=inlet.node_ids,
        coords=inlet.coords,
        normal=normal,
        vals=np.einsum("tnd,d->tn", inlet.vals, normal),
        periodic=inlet.periodic,
        source=inlet.source,
    )


def vector_inlet(inlet: InletData) -> InletVelocityData:
    """Expand a prepared normal control into its equivalent vector velocity."""

    if not isinstance(inlet, InletData):
        raise TypeError("inlet must be InletData")
    return InletVelocityData(
        boundary=inlet.boundary,
        times=inlet.times,
        node_ids=inlet.node_ids,
        coords=inlet.coords,
        vals=inlet.vals[..., None] * inlet.normal,
        periodic=inlet.periodic,
        source=inlet.source,
    )


def reconstruct_velocity_inlet(
    surface: Dynamic,
    cfg: Config,
    normal: np.ndarray,
    boundary: Optional[str] = None,
) -> InletVelocityData:
    """Reconstruct MRI vector velocity on the inlet surface nodes."""

    velocity = load_velocity(cfg["file"])
    profiles = project_to_inlet(
        surface,
        velocity.points,
        velocity.times,
        velocity.vals,
        normal,
        cfg["projection"],
    )
    times, vals = _periodic(
        velocity.times,
        profiles,
        float(cfg["T"]),
    )
    active = surface.interior_node_ids
    if boundary is None:
        boundary = cfg["boundary"]
    return InletVelocityData(
        boundary=boundary,
        times=times,
        node_ids=np.asarray(surface.mesh_node_ids)[active],
        coords=np.asarray(surface.coordinates)[active],
        vals=vals[:, active],
        periodic=True,
        source="velocity",
    )


def velocity_inlet(
    surface: Dynamic,
    cfg: Config,
    normal: np.ndarray,
    boundary: Optional[str] = None,
) -> InletData:
    """Reconstruct vector velocity and extract its configured normal control."""

    reconstructed = reconstruct_velocity_inlet(
        surface,
        cfg,
        normal,
        boundary,
    )
    return normal_inlet(reconstructed, normal, boundary)


def make_inlet(cfg: Config) -> InletData:
    """Build inlet data using the source selected in a plain config dict."""

    mesh = femx.Mesh.read(cfg["mesh"])
    opts = cfg["inlet"]
    control = cfg["control"]
    if control["kind"] != "normal":
        raise ValueError("control.kind must be 'normal'")
    boundary = control["boundary"]
    surface = mesh.boundary(boundary)
    normal = inlet_normal(mesh, surface, control.get("normal", "auto"))

    if opts["source"] == "analytic":
        matching = [
            item
            for item in cfg["bc"]
            if item["boundary"] == boundary and item["field"] == "velocity"
        ]
        if len(matching) != 1:
            raise ValueError("analytic inlet requires exactly one velocity bc")
        inlet_bc = matching[0]
        if inlet_bc.get("direction") != "normal":
            raise ValueError("analytic inlet requires direction: normal")
        return analytic_inlet(
            surface,
            opts,
            normal,
            boundary,
            space=inlet_bc["space"],
            time=inlet_bc["time"],
        )
    if opts["source"] == "velocity":
        return velocity_inlet(surface, opts, normal, boundary)
    raise ValueError("inlet source must be 'analytic' or 'velocity'")


def make_inlet_velocity(cfg: Config) -> InletVelocityData:
    """Build workflow-independent inlet vector velocity from a plain config."""

    opts = cfg["inlet"]
    if opts["source"] == "analytic":
        return vector_inlet(make_inlet(cfg))
    if opts["source"] != "velocity":
        raise ValueError("inlet source must be 'analytic' or 'velocity'")

    mesh = femx.Mesh.read(cfg["mesh"])
    control = cfg["control"]
    if control["kind"] != "normal":
        raise ValueError("control.kind must be 'normal'")
    boundary = control["boundary"]
    surface = mesh.boundary(boundary)
    normal = inlet_normal(mesh, surface, control.get("normal", "auto"))
    return reconstruct_velocity_inlet(surface, opts, normal, boundary)

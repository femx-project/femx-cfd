"""HDF5 persistence for prepared CFD and data-assimilation inputs."""

from dataclasses import dataclass
from pathlib import Path

import femx
import h5py
import numpy as np


FORMAT = "femx-cfd"
LEGACY_FORMATS = {"femx-inv"}
FORMAT_VERSION = 1
H5_SUFFIXES = {".h5", ".hdf5"}


@dataclass(frozen=True)
class InletData:
    """Normal or vector velocity sampled on FEM inlet nodes and time levels."""

    boundary: str
    times: np.ndarray
    node_ids: np.ndarray
    coords: np.ndarray
    normal: np.ndarray
    vals: np.ndarray
    periodic: bool = True
    source: str = ""
    kind: str = "normal"

    def __post_init__(self):
        times = np.asarray(self.times, dtype=float)
        node_ids = np.asarray(self.node_ids, dtype=np.int64)
        coords = np.asarray(self.coords, dtype=float)
        normal = np.asarray(self.normal, dtype=float)
        vals = np.asarray(self.vals, dtype=float)
        if (
            times.ndim != 1
            or times.size == 0
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("inlet times must be increasing")
        if node_ids.ndim != 1 or node_ids.size == 0:
            raise ValueError("inlet node_ids must be a nonempty vector")
        if (
            coords.ndim != 2
            or coords.shape[0] != node_ids.size
            or coords.shape[1] not in (2, 3)
        ):
            raise ValueError("inlet coordinates have an incompatible shape")
        if self.kind not in ("normal", "vector"):
            raise ValueError("inlet kind must be 'normal' or 'vector'")
        expected = (
            (times.size, node_ids.size)
            if self.kind == "normal"
            else (times.size, node_ids.size, normal.size)
        )
        if vals.shape != expected:
            raise ValueError("inlet velocity has an incompatible shape")
        if normal.size not in (2, 3) or not np.all(np.isfinite(normal)):
            raise ValueError("inlet normal must be a finite 2D or 3D vector")
        length = np.linalg.norm(normal)
        if length == 0.0:
            raise ValueError("inlet normal must be nonzero")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(coords)):
            raise ValueError("inlet coordinates and times must be finite")
        if not np.all(np.isfinite(vals)):
            raise ValueError("inlet velocity must be finite")
        if self.periodic and not np.allclose(vals[0], vals[-1]):
            raise ValueError("periodic inlet must repeat its first velocity at the end")

        object.__setattr__(self, "times", np.ascontiguousarray(times))
        object.__setattr__(self, "node_ids", np.ascontiguousarray(node_ids))
        object.__setattr__(self, "coords", np.ascontiguousarray(coords))
        object.__setattr__(self, "normal", np.ascontiguousarray(normal / length))
        object.__setattr__(self, "vals", np.ascontiguousarray(vals))

    @property
    def params(self):
        """Flattened independent time levels for the configured control."""

        vals = self.vals[:-1] if self.periodic else self.vals
        return vals.reshape(-1).copy()

    def params_for(self, prob):
        """Order the inlet values for a compatible Navier-Stokes problem."""

        if not np.allclose(prob.ctr.times, self.times):
            raise ValueError("problem and inlet times do not match")
        local = {int(node): idx for idx, node in enumerate(self.node_ids)}
        try:
            cols = [local[int(node)] for node in prob.ctr_mesh_node_ids]
        except KeyError as error:
            raise ValueError("problem and inlet nodes do not match") from error
        vals = self.vals[:-1] if self.periodic else self.vals
        if self.kind == "normal":
            if len(cols) != self.node_ids.size:
                raise ValueError("problem and inlet nodes do not match")
            return np.ascontiguousarray(vals[:, cols]).reshape(-1)

        node_ids = np.asarray(prob.ctr_mesh_node_ids, dtype=np.int64)
        dim = self.normal.size
        if node_ids.size != self.node_ids.size * dim:
            raise ValueError("problem and inlet vector nodes do not match")
        component = np.empty(node_ids.size, dtype=np.int64)
        seen = {}
        for idx, node in enumerate(node_ids):
            count = seen.get(int(node), 0)
            component[idx] = count
            seen[int(node)] = count + 1
        if np.any(component >= dim):
            raise ValueError("problem has an incompatible vector control ordering")
        return np.ascontiguousarray(vals[:, cols, component]).reshape(-1)


@dataclass(frozen=True)
class VelocityData:
    """Discrete velocity samples shared by synthetic and MRI cases."""

    points: np.ndarray
    times: np.ndarray
    vals: np.ndarray

    def __post_init__(self):
        points = np.asarray(self.points, dtype=float)
        times = np.asarray(self.times, dtype=float)
        vals = np.asarray(self.vals, dtype=float)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("velocity points must have shape (num_points, 2 or 3)")
        if (
            times.ndim != 1
            or times.size == 0
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("velocity times must be increasing")
        if vals.shape != (times.size, points.shape[0], points.shape[1]):
            raise ValueError("velocity values have an incompatible shape")
        if not all(np.all(np.isfinite(array)) for array in (points, times, vals)):
            raise ValueError("velocity data must be finite")

        object.__setattr__(self, "points", np.ascontiguousarray(points))
        object.__setattr__(self, "times", np.ascontiguousarray(times))
        object.__setattr__(self, "vals", np.ascontiguousarray(vals))


def _path(path):
    path = Path(path)
    if path.suffix.lower() not in H5_SUFFIXES:
        raise ValueError("data path must end with .h5 or .hdf5")
    return path


def _write_header(h5, kind):
    h5.attrs["format"] = FORMAT
    h5.attrs["format_version"] = FORMAT_VERSION
    h5.attrs["kind"] = kind


def _check_header(h5, kind):
    if h5.attrs.get("format") not in {FORMAT, *LEGACY_FORMATS}:
        raise ValueError("not a femx-cfd HDF5 file")
    if h5.attrs.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported femx-cfd HDF5 format version")
    if h5.attrs.get("kind") != kind:
        raise ValueError(f"HDF5 data is not {kind}")


def _group(h5, name, datasets):
    if name not in h5 or not isinstance(h5[name], h5py.Group):
        raise ValueError(f"HDF5 data is missing group '{name}'")
    group = h5[name]
    missing = datasets.difference(group.keys())
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"HDF5 group '{name}' is missing: {names}")
    return group


def save_velocity(path, velocity):
    """Save discrete velocity samples in the common prepared-data format."""

    if not isinstance(velocity, VelocityData):
        raise TypeError("velocity must be VelocityData")
    with h5py.File(_path(path), "w") as h5:
        _write_header(h5, "velocity")
        group = h5.create_group("velocity")
        group.create_dataset("points", data=velocity.points)
        group.create_dataset("times", data=velocity.times)
        group.create_dataset("values", data=velocity.vals, compression="gzip")


def load_velocity(path):
    """Load discrete velocity samples from the common prepared-data format."""

    with h5py.File(_path(path), "r") as h5:
        _check_header(h5, "velocity")
        group = _group(h5, "velocity", {"points", "times", "values"})
        return VelocityData(
            points=group["points"][...],
            times=group["times"][...],
            vals=group["values"][...],
        )


def make_velocity_obs(velocity, model):
    """Create unit-weight observations from prepared velocity samples."""

    components = np.arange(velocity.points.shape[1])
    op = femx.VelocityObservationOperator(
        model,
        points=velocity.points,
        times=velocity.times,
        components=components,
    )
    return femx.GaussianObservation(op, velocity.vals.reshape(-1), std=1.0)


def add_velocity_noise(velocity, std, seed):
    """Add reproducible independent Gaussian noise to velocity samples."""

    rng = np.random.default_rng(seed)
    vals = velocity.vals + rng.normal(0.0, std, velocity.vals.shape)

    return VelocityData(
        points=velocity.points,
        times=velocity.times,
        vals=vals,
    )


def save_inlet(path, inlet):
    """Save a prepared inlet boundary condition."""

    if not isinstance(inlet, InletData):
        raise TypeError("inlet must be InletData")
    with h5py.File(_path(path), "w") as h5:
        _write_header(h5, "inlet")
        h5.attrs["boundary"] = inlet.boundary
        h5.attrs["periodic"] = inlet.periodic
        h5.attrs["source"] = inlet.source
        h5.attrs["control_kind"] = inlet.kind
        group = h5.create_group("inlet")
        group.create_dataset("times", data=inlet.times)
        group.create_dataset("node_ids", data=inlet.node_ids)
        group.create_dataset("coordinates", data=inlet.coords)
        group.create_dataset("normal", data=inlet.normal)
        group.create_dataset(
            "normal_velocity" if inlet.kind == "normal" else "velocity",
            data=inlet.vals,
        )


def load_inlet(path):
    """Load a prepared inlet boundary condition."""

    with h5py.File(_path(path), "r") as h5:
        _check_header(h5, "inlet")
        kind = str(h5.attrs.get("control_kind", "normal"))
        value_name = "normal_velocity" if kind == "normal" else "velocity"
        group = _group(h5, "inlet", {"times", "node_ids", "coordinates", "normal", value_name})
        return InletData(
            boundary=str(h5.attrs["boundary"]),
            times=group["times"][...],
            node_ids=group["node_ids"][...],
            coords=group["coordinates"][...],
            normal=group["normal"][...],
            vals=group[value_name][...],
            periodic=bool(h5.attrs["periodic"]),
            source=str(h5.attrs.get("source", "")),
            kind=kind,
        )

"""HDF5 persistence for prepared CFD and data-assimilation inputs."""

from dataclasses import dataclass
from pathlib import Path

import femx
import h5py
import numpy as np

from femx_cfd._typing import Dynamic, PathLike


FORMAT = "femx-cfd"
FORMAT_VERSION = 1
INLET_FORMAT_VERSION = 2
H5_SUFFIXES = {".h5", ".hdf5"}


@dataclass(frozen=True)
class InletData:
    """Prepared normal control sampled on FEM inlet nodes and time levels."""

    boundary: str
    times: np.ndarray
    node_ids: np.ndarray
    coords: np.ndarray
    normal: np.ndarray
    vals: np.ndarray
    periodic: bool = True
    source: str = ""

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        node_ids = np.asarray(self.node_ids, dtype=np.int64)
        coords = np.asarray(self.coords, dtype=float)
        normal = np.asarray(self.normal, dtype=float)
        vals = np.asarray(self.vals, dtype=float)
        if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("inlet times must be increasing")
        if node_ids.ndim != 1 or node_ids.size == 0:
            raise ValueError("inlet node_ids must be a nonempty vector")
        if (
            coords.ndim != 2
            or coords.shape[0] != node_ids.size
            or coords.shape[1] not in (2, 3)
        ):
            raise ValueError("inlet coordinates have an incompatible shape")
        expected = (times.size, node_ids.size)
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
    def params(self) -> np.ndarray:
        """Flattened independent time levels for the configured control."""

        vals = self.vals[:-1] if self.periodic else self.vals
        return vals.reshape(-1).copy()

    def params_for(self, prob: Dynamic) -> np.ndarray:
        """Order the inlet values for a compatible Navier-Stokes problem."""

        cols = self._param_columns(prob)
        vals = self.vals[:-1] if self.periodic else self.vals
        return np.ascontiguousarray(vals[:, cols]).reshape(-1)

    def with_params_for(self, prob: Dynamic, params: Dynamic) -> "InletData":
        """Return inlet data updated from problem-ordered control parameters."""

        cols = self._param_columns(prob)
        levels = self.times.size - 1 if self.periodic else self.times.size
        params = np.asarray(params, dtype=float)
        expected = levels * len(cols)
        if params.ndim != 1 or params.size != expected:
            raise ValueError("normal inlet parameters have an incompatible size")
        vals = np.empty((levels, self.node_ids.size))
        vals[:, cols] = params.reshape(levels, len(cols))
        if self.periodic:
            vals = np.vstack((vals, vals[:1]))
        return InletData(
            boundary=self.boundary,
            times=self.times,
            node_ids=self.node_ids,
            coords=self.coords,
            normal=self.normal,
            vals=vals,
            periodic=self.periodic,
            source=self.source,
        )

    def _param_columns(self, prob: Dynamic) -> np.ndarray:
        if not np.allclose(prob.ctr.times, self.times):
            raise ValueError("problem and inlet times do not match")
        local = {int(node): idx for idx, node in enumerate(self.node_ids)}
        try:
            cols = np.asarray(
                [local[int(node)] for node in prob.ctr_mesh_node_ids],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError("problem and inlet nodes do not match") from error
        if len(cols) != self.node_ids.size:
            raise ValueError("problem and inlet nodes do not match")
        return cols


@dataclass(frozen=True)
class InletVelocityData:
    """Vector velocity reconstructed on inlet nodes, independent of a control."""

    boundary: str
    times: np.ndarray
    node_ids: np.ndarray
    coords: np.ndarray
    vals: np.ndarray
    periodic: bool = True
    source: str = ""

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        node_ids = np.asarray(self.node_ids, dtype=np.int64)
        coords = np.asarray(self.coords, dtype=float)
        vals = np.asarray(self.vals, dtype=float)
        if not isinstance(self.boundary, str) or not self.boundary:
            raise ValueError("inlet boundary must be a nonempty string")
        if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("inlet times must be increasing")
        if node_ids.ndim != 1 or node_ids.size == 0:
            raise ValueError("inlet node_ids must be a nonempty vector")
        if (
            coords.ndim != 2
            or coords.shape[0] != node_ids.size
            or coords.shape[1] not in (2, 3)
        ):
            raise ValueError("inlet coordinates have an incompatible shape")
        if (
            vals.ndim != 3
            or vals.shape[:2] != (times.size, node_ids.size)
            or vals.shape[2] not in (2, 3)
        ):
            raise ValueError("inlet vector velocity has an incompatible shape")
        if not all(np.all(np.isfinite(array)) for array in (times, coords, vals)):
            raise ValueError("inlet data must be finite")
        if self.periodic and not np.allclose(vals[0], vals[-1]):
            raise ValueError("periodic inlet must repeat its first velocity at the end")

        object.__setattr__(self, "times", np.ascontiguousarray(times))
        object.__setattr__(self, "node_ids", np.ascontiguousarray(node_ids))
        object.__setattr__(self, "coords", np.ascontiguousarray(coords))
        object.__setattr__(self, "vals", np.ascontiguousarray(vals))

    @property
    def params(self) -> np.ndarray:
        """Flattened independent time levels for a vector control."""

        vals = self.vals[:-1] if self.periodic else self.vals
        return vals.reshape(-1).copy()

    def params_for(self, prob: Dynamic) -> np.ndarray:
        """Order vector inlet values for a compatible velocity control."""

        columns, components = self._param_entries(prob)
        vals = self.vals[:-1] if self.periodic else self.vals
        ordered = vals[:, columns, components]
        params = np.ascontiguousarray(ordered).reshape(-1)
        if params.size != prob.num_ctr_param:
            raise ValueError("problem and inlet vector parameter counts do not match")
        return params

    def with_params_for(
        self,
        prob: Dynamic,
        params: Dynamic,
    ) -> "InletVelocityData":
        """Return inlet data updated from problem-ordered control parameters."""

        columns, components = self._param_entries(prob)
        levels = self.times.size - 1 if self.periodic else self.times.size
        params = np.asarray(params, dtype=float)
        expected = levels * columns.size
        if params.ndim != 1 or params.size != expected:
            raise ValueError("vector inlet parameters have an incompatible size")
        vals = np.empty((levels, self.node_ids.size, self.vals.shape[2]))
        vals[:, columns, components] = params.reshape(levels, columns.size)
        if self.periodic:
            vals = np.concatenate((vals, vals[:1]), axis=0)
        return InletVelocityData(
            boundary=self.boundary,
            times=self.times,
            node_ids=self.node_ids,
            coords=self.coords,
            vals=vals,
            periodic=self.periodic,
            source=self.source,
        )

    def _param_entries(self, prob: Dynamic) -> tuple[np.ndarray, np.ndarray]:
        if getattr(prob.ctr, "kind", None) != "vector":
            raise ValueError("problem must use a vector velocity control")
        if not np.allclose(prob.ctr.times, self.times):
            raise ValueError("problem and inlet times do not match")

        dimension = int(prob.model.mesh.dimension)
        if self.vals.shape[2] != dimension:
            raise ValueError("problem and inlet velocity dimensions do not match")
        node_ids = np.asarray(prob.ctr_mesh_node_ids, dtype=np.int64)
        state_dofs = np.asarray(prob.ctr_state_dofs, dtype=np.int64)
        if node_ids.shape != state_dofs.shape:
            raise ValueError("problem vector control has incompatible node mapping")

        local = {int(node): idx for idx, node in enumerate(self.node_ids)}
        try:
            columns = np.asarray(
                [local[int(node)] for node in node_ids],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError("problem and inlet nodes do not match") from error

        velocity_dofs = np.asarray(prob.model.velocity_dofs, dtype=np.int64)
        component_by_dof = {
            int(dof): index % dimension for index, dof in enumerate(velocity_dofs)
        }
        try:
            components = np.asarray(
                [component_by_dof[int(dof)] for dof in state_dofs],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                "vector control contains a non-velocity state dof"
            ) from error

        pairs = set(zip(columns.tolist(), components.tolist()))
        expected_pairs = {
            (node, component)
            for node in range(self.node_ids.size)
            for component in range(dimension)
        }
        if pairs != expected_pairs or len(pairs) != node_ids.size:
            raise ValueError("problem and inlet vector control nodes do not match")
        return columns, components


@dataclass(frozen=True)
class VelocityData:
    """Discrete velocity samples shared by synthetic and MRI cases."""

    points: np.ndarray
    times: np.ndarray
    vals: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        times = np.asarray(self.times, dtype=float)
        vals = np.asarray(self.vals, dtype=float)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("velocity points must have shape (num_points, 2 or 3)")
        if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("velocity times must be increasing")
        if vals.shape != (times.size, points.shape[0], points.shape[1]):
            raise ValueError("velocity values have an incompatible shape")
        if not all(np.all(np.isfinite(array)) for array in (points, times, vals)):
            raise ValueError("velocity data must be finite")

        object.__setattr__(self, "points", np.ascontiguousarray(points))
        object.__setattr__(self, "times", np.ascontiguousarray(times))
        object.__setattr__(self, "vals", np.ascontiguousarray(vals))


def _path(path: PathLike) -> Path:
    path = Path(path)
    if path.suffix.lower() not in H5_SUFFIXES:
        raise ValueError("data path must end with .h5 or .hdf5")
    return path


def _write_header(
    h5: h5py.File,
    kind: str,
    version: int = FORMAT_VERSION,
) -> None:
    h5.attrs["format"] = FORMAT
    h5.attrs["format_version"] = version
    h5.attrs["kind"] = kind


def _check_header(
    h5: h5py.File,
    kind: str,
    versions: tuple[int, ...] = (FORMAT_VERSION,),
) -> int:
    if h5.attrs.get("format") != FORMAT:
        raise ValueError("not a femx-cfd HDF5 file")
    version = h5.attrs.get("format_version")
    if version not in versions:
        raise ValueError("unsupported femx-cfd HDF5 format version")
    if h5.attrs.get("kind") != kind:
        raise ValueError(f"HDF5 data is not {kind}")
    return int(version)


def _group(h5: h5py.File, name: str, datasets: set[str]) -> h5py.Group:
    if name not in h5 or not isinstance(h5[name], h5py.Group):
        raise ValueError(f"HDF5 data is missing group '{name}'")
    group = h5[name]
    missing = datasets.difference(group.keys())
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"HDF5 group '{name}' is missing: {names}")
    return group


def save_velocity(path: PathLike, velocity: VelocityData) -> None:
    """Save discrete velocity samples in the common prepared-data format."""

    if not isinstance(velocity, VelocityData):
        raise TypeError("velocity must be VelocityData")
    with h5py.File(_path(path), "w") as h5:
        _write_header(h5, "velocity")
        group = h5.create_group("velocity")
        group.create_dataset("points", data=velocity.points)
        group.create_dataset("times", data=velocity.times)
        group.create_dataset("values", data=velocity.vals, compression="gzip")


def load_velocity(path: PathLike) -> VelocityData:
    """Load discrete velocity samples from the common prepared-data format."""

    with h5py.File(_path(path), "r") as h5:
        _check_header(h5, "velocity")
        group = _group(h5, "velocity", {"points", "times", "values"})
        return VelocityData(
            points=group["points"][...],
            times=group["times"][...],
            vals=group["values"][...],
        )


def make_velocity_obs(velocity: VelocityData, model: Dynamic) -> Dynamic:
    """Create unit-weight observations from prepared velocity samples."""

    components = np.arange(velocity.points.shape[1])
    op = femx.VelocityObservationOperator(
        model,
        points=velocity.points,
        times=velocity.times,
        components=components,
    )
    return femx.GaussianObservation(op, velocity.vals.reshape(-1), std=1.0)


def add_velocity_noise(
    velocity: VelocityData,
    std: float,
    seed: int,
) -> VelocityData:
    """Add reproducible independent Gaussian noise to velocity samples."""

    rng = np.random.default_rng(seed)
    vals = velocity.vals + rng.normal(0.0, std, velocity.vals.shape)

    return VelocityData(
        points=velocity.points,
        times=velocity.times,
        vals=vals,
    )


def save_inlet(path: PathLike, inlet: Dynamic) -> None:
    """Save inlet vector velocity, expanding a legacy normal control if needed."""

    if isinstance(inlet, InletData):
        inlet = InletVelocityData(
            boundary=inlet.boundary,
            times=inlet.times,
            node_ids=inlet.node_ids,
            coords=inlet.coords,
            vals=inlet.vals[..., None] * inlet.normal,
            periodic=inlet.periodic,
            source=inlet.source,
        )
    if not isinstance(inlet, InletVelocityData):
        raise TypeError("inlet must be InletVelocityData or InletData")
    with h5py.File(_path(path), "w") as h5:
        _write_header(h5, "inlet", INLET_FORMAT_VERSION)
        h5.attrs["boundary"] = inlet.boundary
        h5.attrs["periodic"] = inlet.periodic
        h5.attrs["source"] = inlet.source
        group = h5.create_group("inlet")
        group.create_dataset("times", data=inlet.times)
        group.create_dataset("node_ids", data=inlet.node_ids)
        group.create_dataset("coordinates", data=inlet.coords)
        group.create_dataset("velocity", data=inlet.vals, compression="gzip")


def load_inlet(path: PathLike) -> InletVelocityData:
    """Load inlet vector velocity, upgrading legacy normal-only data."""

    with h5py.File(_path(path), "r") as h5:
        version = _check_header(
            h5,
            "inlet",
            (FORMAT_VERSION, INLET_FORMAT_VERSION),
        )
        datasets = {"times", "node_ids", "coordinates", "velocity"}
        if version == FORMAT_VERSION:
            datasets = {
                "times",
                "node_ids",
                "coordinates",
                "normal",
                "normal_velocity",
            }
        group = _group(h5, "inlet", datasets)
        if version == FORMAT_VERSION:
            vals = group["normal_velocity"][...][..., None] * group["normal"][...]
        else:
            vals = group["velocity"][...]
        return InletVelocityData(
            boundary=str(h5.attrs["boundary"]),
            times=group["times"][...],
            node_ids=group["node_ids"][...],
            coords=group["coordinates"][...],
            vals=vals,
            periodic=bool(h5.attrs["periodic"]),
            source=str(h5.attrs.get("source", "")),
        )

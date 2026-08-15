"""Write case lifecycle, visualization, and analysis artifacts."""

import base64
from copy import deepcopy
import json
import os
import shutil
import stat
import struct
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np
import yaml

from femx_cfd._typing import Config, Dynamic, PathLike
from femx_cfd.data import VelocityData
from femx_cfd.models.navier_stokes.wss import (
    WallShearStressEvaluator,
    WallShearStressTimeAverage,
)


FORWARD_STATUS_FORMAT = "femx-cfd-forward-status"
FORWARD_RESULT_FORMAT = "femx-cfd-forward-result"
INVERSE_STATUS_FORMAT = "femx-cfd-inverse-status"
INVERSE_RESULT_FORMAT = "femx-cfd-inverse-result"
OUTPUT_VERSION = 1
INVERSE_RESULT_VERSION = 3


@contextmanager
def _atomic_path(path: PathLike) -> Iterator[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    handle, tmp_name = tempfile.mkstemp(dir=path.parent)
    os.close(handle)
    tmp = Path(tmp_name)
    try:
        yield tmp
        tmp.chmod(mode)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path: PathLike, value: Dynamic) -> None:
    """Atomically write one human-readable JSON document."""

    with _atomic_path(path) as temporary:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def write_case(path: PathLike, case: Config) -> None:
    """Atomically snapshot one portable case document as YAML."""

    with _atomic_path(path) as temporary:
        temporary.write_text(
            yaml.safe_dump(case, sort_keys=False),
            encoding="utf-8",
        )


def write_status(
    output: PathLike,
    state: str,
    *,
    status_format: str = FORWARD_STATUS_FORMAT,
    **values: Dynamic,
) -> None:
    """Write the machine-readable lifecycle state for one case run."""

    write_json(
        Path(output) / "status.json",
        {
            "format": status_format,
            "version": OUTPUT_VERSION,
            "state": state,
            **values,
        },
    )


def result_directory_state(path: PathLike) -> str:
    """Return whether a result directory is new, empty, or occupied."""

    path = Path(path)
    if not path.exists():
        return "new"
    if not path.is_dir() or any(path.iterdir()):
        return "occupied"
    return "empty"


def require_available_result_directory(path: PathLike) -> None:
    """Reject a file or nonempty directory before a run starts."""

    path = Path(path)
    if result_directory_state(path) == "occupied":
        raise FileExistsError(f"result directory is not empty: {path}")


def prepare_run(
    output: PathLike,
    case: Config,
    *,
    status_format: str = FORWARD_STATUS_FORMAT,
) -> Path:
    """Create an empty result directory and snapshot its source case."""

    output = Path(output)
    require_available_result_directory(output)
    output.mkdir(parents=True, exist_ok=True)
    write_case(output / "case.yaml", case)
    write_status(output, "running", status_format=status_format)
    return output


def trajectory_levels(
    num_steps: int,
    every: int,
) -> np.ndarray:
    """Select uniformly spaced persisted solver levels."""

    if isinstance(every, bool) or not isinstance(every, int) or every <= 0:
        raise ValueError("output.every must be a positive integer")
    if num_steps < 0:
        raise ValueError("trajectory step count must be nonnegative")
    if num_steps % every != 0:
        raise ValueError(
            f"output.every ({every}) must divide the {num_steps} integration steps"
        )
    return np.arange(0, num_steps + 1, every, dtype=int)


class ForwardStreamWriter:
    """Write sampled forward states without retaining a complete trajectory."""

    def __init__(
        self,
        output: PathLike,
        *,
        every: int = 1,
        wall_shear_stress: Dynamic = None,
        wall_shear_stress_statistics: Dynamic = None,
    ) -> None:
        if isinstance(every, bool) or not isinstance(every, int) or every <= 0:
            raise ValueError("output.every must be a positive integer")
        if wall_shear_stress_statistics is not None and wall_shear_stress is None:
            raise ValueError("WSS statistics require wall shear stress output")
        self._output = Path(output)
        self._partial_h5 = self._output / "flow.partial.h5"
        self._final_h5 = self._output / "flow.h5"
        self._h5: Optional[h5py.File] = None
        self._model: Dynamic = None
        self._steps: list[tuple[str, float]] = []
        self._mesh_dimension = 0
        self._num_nodes = 0
        self._num_elements = 0
        self._nodes_per_element = 0
        self._last_level = -1
        self._last_observed_level = -1
        self._every = every
        self._wall_shear_stress_options = wall_shear_stress
        self._wall_shear_stress: Optional[WallShearStressEvaluator] = None
        self._wall_shear_stress_statistics_options = wall_shear_stress_statistics
        self._wall_shear_stress_time_average: Optional[WallShearStressTimeAverage] = (
            None
        )
        self._wall_shear_stress_nodal_time_average: Optional[
            WallShearStressTimeAverage
        ] = None

    @property
    def has_wall_shear_stress(self) -> bool:
        """Return whether this stream includes wall shear stress."""

        return self._wall_shear_stress_options is not None

    @property
    def has_wall_shear_stress_statistics(self) -> bool:
        """Return whether this stream includes TAWSS."""

        return self._wall_shear_stress_statistics_options is not None

    @property
    def requires_every_step(self) -> bool:
        """Return whether the callback must observe every solver level."""

        return self.has_wall_shear_stress_statistics

    def _open(self, model: Dynamic) -> None:
        mesh = model.mesh
        coordinates = np.asarray(mesh.coordinates, dtype=float)
        elements = np.asarray(mesh.elements, dtype=np.int32)
        if coordinates.shape != (mesh.num_nodes, 3):
            raise ValueError("mesh coordinates must have shape (num_nodes, 3)")
        if (
            elements.ndim != 2
            or elements.shape[0] != mesh.num_elements
            or elements.shape[1] == 0
        ):
            raise ValueError("mesh elements must have a uniform connectivity")

        self._output.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(self._partial_h5, "x")
        self._model = model
        self._mesh_dimension = int(mesh.dimension)
        self._num_nodes = int(mesh.num_nodes)
        self._num_elements = int(mesh.num_elements)
        self._nodes_per_element = int(elements.shape[1])
        self._h5.attrs["format"] = "femx-cfd-forward-stream"
        self._h5.attrs["version"] = OUTPUT_VERSION
        self._h5.attrs["complete"] = False
        mesh_group = self._h5.create_group("mesh")
        mesh_group.create_dataset("geometry", data=coordinates)
        mesh_group.create_dataset("topology", data=elements)
        self._h5.create_group("data")
        if self._wall_shear_stress_options is not None:
            boundary = self._wall_shear_stress_options["boundary"]
            self._wall_shear_stress = WallShearStressEvaluator(
                mesh,
                boundary,
                model.mu,
            )
            wall = self._h5.create_group("wall_shear_stress")
            wall.attrs["boundary"] = boundary
            wall.attrs["dynamic_viscosity"] = float(model.mu)
            wall.attrs["units"] = "Pa"
            wall.attrs["location"] = "boundary_facet_center"
            wall.attrs["nodal_location"] = "boundary_node"
            wall.attrs["nodal_projection"] = "continuous_P1_L2"
            wall.attrs["normal_convention"] = "outward_from_fluid"
            wall.attrs["traction_convention"] = "wall_on_fluid"
            wall_mesh = wall.create_group("mesh")
            wall_mesh.create_dataset(
                "geometry",
                data=self._wall_shear_stress.coordinates,
            )
            wall_mesh.create_dataset(
                "topology",
                data=self._wall_shear_stress.elements,
            )
            wall_mesh.create_dataset(
                "mesh_node_ids",
                data=self._wall_shear_stress.mesh_node_ids,
            )
            wall_mesh.create_dataset(
                "adjacent_cells",
                data=self._wall_shear_stress.adjacent_cells,
            )
            wall_mesh.create_dataset(
                "normal",
                data=self._wall_shear_stress.normals,
            )
            wall.create_group("data")
            if self._wall_shear_stress_statistics_options is not None:
                options = self._wall_shear_stress_statistics_options
                self._wall_shear_stress_time_average = WallShearStressTimeAverage(
                    self._wall_shear_stress.elements.shape[0],
                    options["start_time"],
                    options["end_time"],
                )
                self._wall_shear_stress_nodal_time_average = WallShearStressTimeAverage(
                    self._wall_shear_stress.coordinates.shape[0],
                    options["start_time"],
                    options["end_time"],
                )
        self._h5.flush()

    def sample(self, model: Dynamic, level: int, state: Dynamic) -> None:
        """Observe one solver level and persist it when selected by ``every``."""

        level = int(level)
        if level <= self._last_observed_level:
            raise ValueError("observed solver levels must be strictly increasing")
        if self._h5 is None:
            self._open(model)
        elif model is not self._model:
            raise ValueError("streamed states must belong to one model")

        if self._wall_shear_stress_time_average is not None:
            velocity, _ = model.state_fields(state)
            velocity = np.asarray(velocity, dtype=float)
            if velocity.shape != (self._num_nodes, 3):
                raise ValueError("velocity output has an inconsistent shape")
            assert self._wall_shear_stress is not None
            facet_values = self._wall_shear_stress.evaluate(velocity)
            self._wall_shear_stress_time_average.add(
                level * float(model.dt),
                facet_values,
            )
            assert self._wall_shear_stress_nodal_time_average is not None
            self._wall_shear_stress_nodal_time_average.add(
                level * float(model.dt),
                self._wall_shear_stress.project_to_nodes(facet_values),
            )

        if level % self._every == 0:
            self.append(model, level, state)
        self._last_observed_level = level

    def append(self, model: Dynamic, level: int, state: Dynamic) -> None:
        """Append one solver level to the partial HDF5 trajectory."""

        level = int(level)
        if level <= self._last_level:
            raise ValueError("streamed solver levels must be strictly increasing")
        if self._h5 is None:
            self._open(model)
        elif model is not self._model:
            raise ValueError("streamed states must belong to one model")

        velocity, pressure = model.state_fields(state)
        velocity = np.asarray(velocity, dtype=float)
        pressure = np.asarray(pressure, dtype=float)
        if velocity.shape != (self._num_nodes, 3):
            raise ValueError("velocity output has an inconsistent shape")
        if pressure.shape != (self._num_nodes,):
            raise ValueError("pressure output has an inconsistent shape")

        name = f"step_{len(self._steps):06d}"
        group = self._h5["data"].create_group(name)
        group.attrs["solver_level"] = level
        group.attrs["time"] = level * float(model.dt)
        group.create_dataset("velocity", data=velocity)
        group.create_dataset("pressure", data=pressure)
        if self._wall_shear_stress is not None:
            values = self._wall_shear_stress.evaluate(velocity)
            nodal_values = self._wall_shear_stress.project_to_nodes(values)
            wall_group = self._h5["wall_shear_stress/data"].create_group(name)
            wall_group.attrs["solver_level"] = level
            wall_group.attrs["time"] = level * float(model.dt)
            vector = wall_group.create_dataset("vector", data=values)
            vector.attrs["units"] = "Pa"
            magnitude = wall_group.create_dataset(
                "magnitude",
                data=np.linalg.norm(values, axis=1),
            )
            magnitude.attrs["units"] = "Pa"
            nodal_vector = wall_group.create_dataset(
                "nodal_vector",
                data=nodal_values,
            )
            nodal_vector.attrs["units"] = "Pa"
            nodal_magnitude = wall_group.create_dataset(
                "nodal_magnitude",
                data=np.linalg.norm(nodal_values, axis=1),
            )
            nodal_magnitude.attrs["units"] = "Pa"
        self._steps.append((name, level * float(model.dt)))
        self._last_level = level
        self._h5.attrs["last_solver_level"] = level
        self._h5.flush()

    def finish(self) -> None:
        """Finalize HDF5 and XDMF output after a successful solve."""

        if self._h5 is None or not self._steps:
            raise RuntimeError("forward stream contains no output levels")
        if self._wall_shear_stress_time_average is not None:
            values = self._wall_shear_stress_time_average.finish()
            assert self._wall_shear_stress_nodal_time_average is not None
            nodal_values = self._wall_shear_stress_nodal_time_average.finish()
            options = self._wall_shear_stress_statistics_options
            statistics = self._h5["wall_shear_stress"].create_group("statistics")
            statistics.attrs["time_average"] = options["time_average"]
            statistics.attrs["start_time"] = options["start_time"]
            statistics.attrs["end_time"] = options["end_time"]
            statistics.attrs["duration"] = options["end_time"] - options["start_time"]
            statistics.attrs["integration"] = "trapezoidal"
            statistics.attrs["source"] = "all_solver_levels"
            statistics.attrs["nodal_source"] = (
                "instantaneous_P1_L2_projection_at_all_solver_levels"
            )
            tawss = statistics.create_dataset("tawss", data=values)
            tawss.attrs["units"] = "Pa"
            nodal_tawss = statistics.create_dataset(
                "nodal_tawss",
                data=nodal_values,
            )
            nodal_tawss.attrs["units"] = "Pa"
        self._h5.attrs["complete"] = True
        self._h5.flush()
        self._h5.close()
        self._h5 = None
        os.replace(self._partial_h5, self._final_h5)
        _write_stream_xdmf(
            self._output / "flow.xdmf",
            self._final_h5.name,
            self._steps,
            mesh_dimension=self._mesh_dimension,
            num_nodes=self._num_nodes,
            num_elements=self._num_elements,
            nodes_per_element=self._nodes_per_element,
        )
        if self._wall_shear_stress is not None:
            _write_wall_shear_stress_xdmf(
                self._output / "wall_shear_stress.xdmf",
                self._final_h5.name,
                self._steps,
                num_nodes=self._wall_shear_stress.coordinates.shape[0],
                num_elements=self._wall_shear_stress.elements.shape[0],
                nodes_per_element=self._wall_shear_stress.elements.shape[1],
            )
            if self.has_wall_shear_stress_statistics:
                _write_wall_shear_stress_statistics_xdmf(
                    self._output / "wall_shear_stress_statistics.xdmf",
                    self._final_h5.name,
                    num_nodes=self._wall_shear_stress.coordinates.shape[0],
                    num_elements=self._wall_shear_stress.elements.shape[0],
                    nodes_per_element=self._wall_shear_stress.elements.shape[1],
                )

    def abort(self) -> None:
        """Close a failed stream while retaining its partial HDF5 file."""

        if self._h5 is not None:
            self._h5.flush()
            self._h5.close()
            self._h5 = None


def _stream_topology_type(mesh_dimension: int, nodes_per_element: int) -> str:
    if mesh_dimension == 2 and nodes_per_element == 3:
        return "Triangle"
    if mesh_dimension == 2 and nodes_per_element == 4:
        return "Quadrilateral"
    if mesh_dimension == 3 and nodes_per_element == 4:
        return "Tetrahedron"
    raise ValueError("streamed output has an unsupported mesh element type")


def _write_stream_xdmf(
    path: PathLike,
    h5_name: str,
    steps: list[tuple[str, float]],
    *,
    mesh_dimension: int,
    num_nodes: int,
    num_elements: int,
    nodes_per_element: int,
) -> None:
    topology_type = _stream_topology_type(
        mesh_dimension,
        nodes_per_element,
    )
    root = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(root, "Domain")
    collection = ET.SubElement(
        domain,
        "Grid",
        Name="time_series",
        GridType="Collection",
        CollectionType="Temporal",
    )
    for name, time in steps:
        grid = ET.SubElement(
            collection,
            "Grid",
            Name=name,
            GridType="Uniform",
        )
        ET.SubElement(grid, "Time", Value=f"{time:.17g}")
        topology = ET.SubElement(
            grid,
            "Topology",
            TopologyType=topology_type,
            NumberOfElements=str(num_elements),
        )
        topology_data = ET.SubElement(
            topology,
            "DataItem",
            Dimensions=f"{num_elements} {nodes_per_element}",
            NumberType="Int",
            Precision="4",
            Format="HDF",
        )
        topology_data.text = f"{h5_name}:/mesh/topology"
        geometry = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
        geometry_data = ET.SubElement(
            geometry,
            "DataItem",
            Dimensions=f"{num_nodes} 3",
            NumberType="Float",
            Precision="8",
            Format="HDF",
        )
        geometry_data.text = f"{h5_name}:/mesh/geometry"
        velocity = ET.SubElement(
            grid,
            "Attribute",
            Name="velocity",
            AttributeType="Vector",
            Center="Node",
        )
        velocity_data = ET.SubElement(
            velocity,
            "DataItem",
            Dimensions=f"{num_nodes} 3",
            NumberType="Float",
            Precision="8",
            Format="HDF",
        )
        velocity_data.text = f"{h5_name}:/data/{name}/velocity"
        pressure = ET.SubElement(
            grid,
            "Attribute",
            Name="pressure",
            AttributeType="Scalar",
            Center="Node",
        )
        pressure_data = ET.SubElement(
            pressure,
            "DataItem",
            Dimensions=str(num_nodes),
            NumberType="Float",
            Precision="8",
            Format="HDF",
        )
        pressure_data.text = f"{h5_name}:/data/{name}/pressure"

    ET.indent(root, space="  ")
    with _atomic_path(path) as temporary:
        ET.ElementTree(root).write(
            temporary,
            encoding="utf-8",
            xml_declaration=True,
        )


def _wall_topology_type(nodes_per_element: int) -> str:
    if nodes_per_element == 2:
        return "Polyline"
    if nodes_per_element == 3:
        return "Triangle"
    raise ValueError("wall output has an unsupported facet type")


def _write_wall_shear_stress_xdmf(
    path: PathLike,
    h5_name: str,
    steps: list[tuple[str, float]],
    *,
    num_nodes: int,
    num_elements: int,
    nodes_per_element: int,
) -> None:
    topology_type = _wall_topology_type(nodes_per_element)

    root = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(root, "Domain")
    collection = ET.SubElement(
        domain,
        "Grid",
        Name="wall_shear_stress",
        GridType="Collection",
        CollectionType="Temporal",
    )
    for name, time in steps:
        grid = ET.SubElement(
            collection,
            "Grid",
            Name=name,
            GridType="Uniform",
        )
        ET.SubElement(grid, "Time", Value=f"{time:.17g}")
        topology = ET.SubElement(
            grid,
            "Topology",
            TopologyType=topology_type,
            NumberOfElements=str(num_elements),
        )
        if nodes_per_element == 2:
            topology.set("NodesPerElement", "2")
        topology_data = ET.SubElement(
            topology,
            "DataItem",
            Dimensions=f"{num_elements} {nodes_per_element}",
            NumberType="Int",
            Precision="4",
            Format="HDF",
        )
        topology_data.text = f"{h5_name}:/wall_shear_stress/mesh/topology"
        geometry = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
        geometry_data = ET.SubElement(
            geometry,
            "DataItem",
            Dimensions=f"{num_nodes} 3",
            NumberType="Float",
            Precision="8",
            Format="HDF",
        )
        geometry_data.text = f"{h5_name}:/wall_shear_stress/mesh/geometry"

        for attribute_name, dataset, attribute_type, dimensions, center in (
            (
                "wall_shear_stress",
                "vector",
                "Vector",
                f"{num_elements} 3",
                "Cell",
            ),
            (
                "wall_shear_stress_magnitude",
                "magnitude",
                "Scalar",
                str(num_elements),
                "Cell",
            ),
            (
                "wall_shear_stress_nodal",
                "nodal_vector",
                "Vector",
                f"{num_nodes} 3",
                "Node",
            ),
            (
                "wall_shear_stress_magnitude_nodal",
                "nodal_magnitude",
                "Scalar",
                str(num_nodes),
                "Node",
            ),
            (
                "wall_normal",
                "normal",
                "Vector",
                f"{num_elements} 3",
                "Cell",
            ),
        ):
            attribute = ET.SubElement(
                grid,
                "Attribute",
                Name=attribute_name,
                AttributeType=attribute_type,
                Center=center,
            )
            data = ET.SubElement(
                attribute,
                "DataItem",
                Dimensions=dimensions,
                NumberType="Float",
                Precision="8",
                Format="HDF",
            )
            if dataset == "normal":
                data.text = f"{h5_name}:/wall_shear_stress/mesh/normal"
            else:
                data.text = f"{h5_name}:/wall_shear_stress/data/{name}/{dataset}"

    ET.indent(root, space="  ")
    with _atomic_path(path) as temporary:
        ET.ElementTree(root).write(
            temporary,
            encoding="utf-8",
            xml_declaration=True,
        )


def _write_wall_shear_stress_statistics_xdmf(
    path: PathLike,
    h5_name: str,
    *,
    num_nodes: int,
    num_elements: int,
    nodes_per_element: int,
) -> None:
    topology_type = _wall_topology_type(nodes_per_element)
    root = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(root, "Domain")
    grid = ET.SubElement(
        domain,
        "Grid",
        Name="wall_shear_stress_statistics",
        GridType="Uniform",
    )
    topology = ET.SubElement(
        grid,
        "Topology",
        TopologyType=topology_type,
        NumberOfElements=str(num_elements),
    )
    if nodes_per_element == 2:
        topology.set("NodesPerElement", "2")
    topology_data = ET.SubElement(
        topology,
        "DataItem",
        Dimensions=f"{num_elements} {nodes_per_element}",
        NumberType="Int",
        Precision="4",
        Format="HDF",
    )
    topology_data.text = f"{h5_name}:/wall_shear_stress/mesh/topology"
    geometry = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
    geometry_data = ET.SubElement(
        geometry,
        "DataItem",
        Dimensions=f"{num_nodes} 3",
        NumberType="Float",
        Precision="8",
        Format="HDF",
    )
    geometry_data.text = f"{h5_name}:/wall_shear_stress/mesh/geometry"

    for attribute_name, dataset, attribute_type, dimensions, center in (
        (
            "TAWSS",
            "statistics/tawss",
            "Scalar",
            str(num_elements),
            "Cell",
        ),
        (
            "TAWSS_nodal",
            "statistics/nodal_tawss",
            "Scalar",
            str(num_nodes),
            "Node",
        ),
        (
            "wall_normal",
            "mesh/normal",
            "Vector",
            f"{num_elements} 3",
            "Cell",
        ),
    ):
        attribute = ET.SubElement(
            grid,
            "Attribute",
            Name=attribute_name,
            AttributeType=attribute_type,
            Center=center,
        )
        data = ET.SubElement(
            attribute,
            "DataItem",
            Dimensions=dimensions,
            NumberType="Float",
            Precision="8",
            Format="HDF",
        )
        data.text = f"{h5_name}:/wall_shear_stress/{dataset}"

    ET.indent(root, space="  ")
    with _atomic_path(path) as temporary:
        ET.ElementTree(root).write(
            temporary,
            encoding="utf-8",
            xml_declaration=True,
        )


def write_streamed_forward_result(
    output: PathLike,
    writer: ForwardStreamWriter,
    *,
    linear_systems: bool = False,
) -> None:
    """Finalize a streamed forward trajectory and its result manifest."""

    output = Path(output)
    writer.finish()
    result = {
        "format": FORWARD_RESULT_FORMAT,
        "version": OUTPUT_VERSION,
        "case": "case.yaml",
        "flow": "flow.xdmf",
        "data": "flow.h5",
    }
    if writer.has_wall_shear_stress:
        result["wall_shear_stress"] = "wall_shear_stress.xdmf"
    if writer.has_wall_shear_stress_statistics:
        result["wall_shear_stress_statistics"] = "wall_shear_stress_statistics.xdmf"
    if linear_systems:
        result["linear_systems"] = "linear_systems.h5"
    write_json(
        output / "result.json",
        result,
    )
    write_status(output, "complete", result="result.json")


def _copy_trajectory_levels(
    source_xdmf: PathLike,
    target_xdmf: PathLike,
    levels: np.ndarray,
) -> None:
    """Copy selected flow levels into a compact HDF5/XDMF time series."""

    source_xdmf = Path(source_xdmf).with_suffix(".xdmf")
    target_xdmf = Path(target_xdmf).with_suffix(".xdmf")
    source_h5 = source_xdmf.with_suffix(".h5")
    target_h5 = target_xdmf.with_suffix(".h5")
    levels = np.asarray(levels, dtype=int)

    root = ET.parse(source_xdmf).getroot()
    collection = root.find("./Domain/Grid")
    if collection is None:
        raise ValueError("flow XDMF is missing its temporal collection")
    source_grids = list(collection)
    if (
        levels.ndim != 1
        or levels.size == 0
        or np.any(np.diff(levels) <= 0)
        or levels[0] < 0
        or levels[-1] >= len(source_grids)
    ):
        raise ValueError("output levels are incompatible with the trajectory")

    with h5py.File(source_h5, "r") as input_h5:
        with _atomic_path(target_h5) as temporary:
            with h5py.File(temporary, "w") as output_h5:
                mesh = output_h5.create_group("mesh")
                input_h5.copy("Mesh/Geometry", mesh, name="geometry")
                input_h5.copy("Mesh/Topology", mesh, name="topology")
                data = output_h5.create_group("data")
                for target_level, source_level in enumerate(levels):
                    source_name = source_grids[int(source_level)].attrib["Name"]
                    input_h5.copy(
                        f"Data/{source_name}",
                        data,
                        name=f"step_{target_level:06d}",
                    )

    for grid in source_grids:
        collection.remove(grid)
    for target_level, source_level in enumerate(levels):
        grid = deepcopy(source_grids[int(source_level)])
        source_name = grid.attrib["Name"]
        target_name = f"step_{target_level:06d}"
        grid.attrib["Name"] = target_name
        for item in grid.iter("DataItem"):
            if item.text:
                item.text = item.text.replace(
                    f"{source_h5.name}:",
                    f"{target_h5.name}:",
                )
                item.text = item.text.replace(
                    "/Mesh/Geometry",
                    "/mesh/geometry",
                )
                item.text = item.text.replace(
                    "/Mesh/Topology",
                    "/mesh/topology",
                )
                item.text = item.text.replace(
                    f"/Data/{source_name}/",
                    f"/data/{target_name}/",
                )
        collection.append(grid)
    ET.indent(root, space="  ")
    with _atomic_path(target_xdmf) as temporary:
        ET.ElementTree(root).write(temporary, encoding="utf-8", xml_declaration=True)


def _write_trajectory(
    path: PathLike,
    model: Dynamic,
    trajectory: Dynamic,
    levels: np.ndarray,
) -> None:
    with tempfile.TemporaryDirectory(prefix="femx-cfd-flow-") as tmp_dir:
        source = Path(tmp_dir) / "flow.xdmf"
        model.write_xdmf(source, trajectory)
        _copy_trajectory_levels(source, Path(path).with_suffix(".xdmf"), levels)


def write_forward_result(
    output: PathLike,
    solution: Dynamic,
    *,
    every: int = 1,
) -> None:
    """Write a forward trajectory and its result manifest."""

    output = Path(output)
    levels = trajectory_levels(solution.trajectory.num_steps, every)
    _write_trajectory(
        output / "flow.xdmf",
        solution.model,
        solution.trajectory,
        levels,
    )
    write_json(
        output / "result.json",
        {
            "format": FORWARD_RESULT_FORMAT,
            "version": OUTPUT_VERSION,
            "case": "case.yaml",
            "flow": "flow.xdmf",
            "data": "flow.h5",
        },
    )
    write_status(output, "complete", result="result.json")


def write_inverse_result(
    output: PathLike,
    solution: Dynamic,
    *,
    every: int = 1,
    ensemble_basis: str = "final",
) -> None:
    """Write inverse analysis data, flow trajectory, and its manifest."""

    if ensemble_basis not in ("final", "all"):
        raise ValueError("ensemble_basis must be 'final' or 'all'")
    output = Path(output)
    estimate = output / "estimate"
    write_result(
        estimate,
        solution.model,
        solution.method,
        solution.values,
        solution.trajectory,
        solution.metrics,
        every=every,
        required_times=solution.predicted_velocity.times,
    )
    bases = tuple(solution.ensemble_bases)
    first_outer_index = 0
    if ensemble_basis == "final" and bases:
        first_outer_index = len(bases) - 1
        bases = bases[-1:]
    ensemble_basis_views = write_ensemble_bases(
        estimate / "result.h5",
        estimate / "flow_window.xdmf",
        solution.model,
        bases,
        first_outer_index=first_outer_index,
    )
    write_velocity_series(
        estimate / "observation" / "velocity.pvd",
        solution.predicted_velocity,
    )
    manifest = {
        "format": INVERSE_RESULT_FORMAT,
        "version": INVERSE_RESULT_VERSION,
        "method": solution.method,
        "case": "case.yaml",
        "flow_window": "estimate/flow_window.xdmf",
        "flow_observation": "estimate/flow_observation.xdmf",
        "data": "estimate/result.h5",
        "observation": "estimate/observation/velocity.pvd",
        "initial_inlet": "inlet/initial_velocity.h5",
    }
    if ensemble_basis_views:
        manifest["ensemble_basis"] = [
            str(path.relative_to(output)) for path in ensemble_basis_views
        ]
    write_json(output / "result.json", manifest)
    write_status(
        output,
        "complete",
        status_format=INVERSE_STATUS_FORMAT,
        result="result.json",
    )


def _save_result(
    path: PathLike,
    method: str,
    vals: Config,
    levels: np.ndarray,
    dt: float,
) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "femx-cfd"
        h5.attrs["format_version"] = INVERSE_RESULT_VERSION
        h5.attrs["kind"] = f"{method}_result"
        estimate = h5.create_group("estimate")
        for name, val in vals.items():
            if name == "coefficient_names":
                val = np.asarray(val, dtype=h5py.string_dtype())
            estimate.create_dataset(name, data=val)
        estimate.create_dataset("state_levels", data=levels)
        estimate.create_dataset("state_times", data=levels * dt)


def _save_metrics(group: h5py.Group, metrics: Config) -> None:
    for name, val in metrics.items():
        group.attrs[name] = val


def bundle_result(
    src_result: PathLike,
    src_xdmf: PathLike,
    out: PathLike,
    metrics: Optional[Config] = None,
    *,
    data_name: str = "result.h5",
    flow_name: str = "flow.xdmf",
) -> None:
    """Merge femx visualization data into the analysis result HDF5 file."""
    src_result = Path(src_result)
    src_xdmf = Path(src_xdmf)
    out = Path(out)

    xdmf = src_xdmf.read_text(encoding="utf-8")
    heavy_name = next(
        word.split(":", 1)[0]
        for word in xdmf.replace(">", " ").replace("<", " ").split()
        if ".h5:" in word
    )
    heavy_file = src_xdmf.parent / heavy_name

    result_file = out / data_name
    flow_file = out / flow_name
    if result_file.exists() or flow_file.exists():
        raise FileExistsError(f"run output already exists: {out}")

    with _atomic_path(result_file) as tmp:
        shutil.copyfile(src_result, tmp)
        with h5py.File(tmp, "a") as result, h5py.File(heavy_file) as flow:
            flow.copy("mesh", result)
            estimate = result.require_group("estimate")
            flow.copy("data", estimate, name="fields")
            if metrics:
                _save_metrics(estimate.create_group("metrics"), metrics)

    with _atomic_path(flow_file) as tmp:
        tmp.write_text(
            xdmf.replace(f"{heavy_name}:", f"{data_name}:").replace(
                "/data/", "/estimate/fields/"
            ),
            encoding="utf-8",
        )


def _bundle_sampled_flow(
    source_h5: PathLike,
    source_xdmf: PathLike,
    result_h5: PathLike,
    target_xdmf: PathLike,
) -> None:
    """Append independently sampled flow fields to an inverse result."""

    source_h5 = Path(source_h5).with_suffix(".h5")
    source_xdmf = Path(source_xdmf).with_suffix(".xdmf")
    result_h5 = Path(result_h5).with_suffix(".h5")
    target_xdmf = Path(target_xdmf).with_suffix(".xdmf")
    with h5py.File(result_h5, "a") as result, h5py.File(source_h5) as samples:
        estimate = result.require_group("estimate")
        if "observation_fields" in estimate:
            raise FileExistsError(f"observation flow already exists: {result_h5}")
        samples.copy("data", estimate, name="observation_fields")

    root = ET.parse(source_xdmf).getroot()
    for item in root.iter("DataItem"):
        if item.text:
            item.text = item.text.replace(
                f"{source_h5.name}:",
                f"{result_h5.name}:",
            )
            item.text = item.text.replace(
                "/data/",
                "/estimate/observation_fields/",
            )
    ET.indent(root, space="  ")
    with _atomic_path(target_xdmf) as temporary:
        ET.ElementTree(root).write(temporary, encoding="utf-8", xml_declaration=True)


def write_result(
    out: PathLike,
    model: Dynamic,
    method: str,
    vals: Config,
    traj: Dynamic,
    metrics: Config,
    *,
    every: int = 1,
    required_times: Optional[np.ndarray] = None,
) -> None:
    """Write one inverse result and its selected flow time series."""
    levels = trajectory_levels(traj.num_steps, every)
    with tempfile.TemporaryDirectory(prefix="femx-cfd-result-") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        result_file = tmp_dir / "result.h5"
        complete_flow = tmp_dir / "complete.xdmf"
        flow_file = tmp_dir / "flow.xdmf"
        observation_file = tmp_dir / "observation.h5"
        _save_result(result_file, method, vals, levels, model.dt)
        model.write_xdmf(complete_flow, traj)
        if required_times is not None:
            write_truth_samples(
                complete_flow,
                observation_file,
                required_times,
                model.dt,
            )
        _copy_trajectory_levels(complete_flow, flow_file, levels)
        bundle_result(
            result_file,
            flow_file,
            out,
            metrics,
            flow_name="flow_window.xdmf",
        )
        if required_times is not None:
            _bundle_sampled_flow(
                observation_file,
                observation_file.with_suffix(".xdmf"),
                Path(out) / "result.h5",
                Path(out) / "flow_observation.xdmf",
            )


def _state_fields(
    model: Dynamic,
    states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert state vectors to node-indexed velocity and pressure fields."""

    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != model.num_states:
        raise ValueError("ensemble basis states do not match the model")
    dimension = int(model.mesh.dimension)
    num_nodes = int(model.mesh.num_nodes)
    velocity_dofs = np.asarray(model.velocity_dofs, dtype=np.int64).reshape(
        num_nodes,
        dimension,
    )
    velocity = np.zeros((states.shape[0], num_nodes, 3))
    velocity[..., :dimension] = states[:, velocity_dofs]

    pressure_mask = np.ones(model.num_states, dtype=bool)
    pressure_mask[velocity_dofs.reshape(-1)] = False
    pressure_dofs = np.flatnonzero(pressure_mask)
    if pressure_dofs.size != num_nodes:
        raise ValueError("model pressure degrees of freedom do not match the mesh")
    pressure = states[:, pressure_dofs]
    return velocity, pressure


def _write_basis_fields(
    group: h5py.Group,
    model: Dynamic,
    states: np.ndarray,
) -> None:
    velocity, pressure = _state_fields(model, states)
    fields = group.create_group("fields")
    fields.create_dataset("velocity", data=velocity, compression="gzip")
    fields.create_dataset("pressure", data=pressure, compression="gzip")


def _xdmf_hyperslab(
    dataset: str,
    index: int,
    full_shape: tuple[int, ...],
) -> ET.Element:
    selected_shape = full_shape[1:]
    dimensions = " ".join(str(size) for size in selected_shape)
    item = ET.Element(
        "DataItem",
        ItemType="HyperSlab",
        Dimensions=dimensions,
        Type="HyperSlab",
    )
    rank = len(full_shape)
    selector = ET.SubElement(
        item,
        "DataItem",
        Dimensions=f"3 {rank}",
        Format="XML",
    )
    start = [index, *(0 for _ in selected_shape)]
    stride = [1] * rank
    count = [1, *selected_shape]
    selector.text = (
        "\n"
        + "\n".join(
            " ".join(str(value) for value in row) for row in (start, stride, count)
        )
        + "\n"
    )
    source = ET.SubElement(
        item,
        "DataItem",
        Dimensions=" ".join(str(size) for size in full_shape),
        NumberType="Float",
        Precision="8",
        Format="HDF",
    )
    source.text = dataset
    return item


def _append_basis_attribute(
    grid: ET.Element,
    *,
    name: str,
    attribute_type: str,
    reference: str,
    perturbation: Optional[str],
    index: int,
    full_shape: tuple[int, ...],
) -> None:
    attribute = ET.SubElement(
        grid,
        "Attribute",
        Name=name,
        AttributeType=attribute_type,
        Center="Node",
    )
    reference_item = _xdmf_hyperslab(reference, index, full_shape)
    if perturbation is None:
        attribute.append(reference_item)
        return
    dimensions = " ".join(str(size) for size in full_shape[1:])
    function = ET.SubElement(
        attribute,
        "DataItem",
        ItemType="Function",
        Function="$0 + $1",
        Dimensions=dimensions,
    )
    function.append(reference_item)
    function.append(_xdmf_hyperslab(perturbation, index, full_shape))


def _basis_member_grid(
    template: ET.Element,
    *,
    name: str,
    reference_root: str,
    perturbation_root: Optional[str],
    time_index: int,
    num_times: int,
    num_nodes: int,
) -> ET.Element:
    grid = deepcopy(template)
    grid.attrib["Name"] = name
    time = grid.find("Time")
    if time is not None:
        grid.remove(time)
    for attribute in list(grid.findall("Attribute")):
        grid.remove(attribute)

    velocity_shape = (num_times, num_nodes, 3)
    pressure_shape = (num_times, num_nodes)
    _append_basis_attribute(
        grid,
        name="velocity",
        attribute_type="Vector",
        reference=f"{reference_root}/velocity",
        perturbation=(
            None if perturbation_root is None else f"{perturbation_root}/velocity"
        ),
        index=time_index,
        full_shape=velocity_shape,
    )
    _append_basis_attribute(
        grid,
        name="pressure",
        attribute_type="Scalar",
        reference=f"{reference_root}/pressure",
        perturbation=(
            None if perturbation_root is None else f"{perturbation_root}/pressure"
        ),
        index=time_index,
        full_shape=pressure_shape,
    )
    if perturbation_root is not None:
        _append_basis_attribute(
            grid,
            name="velocity_perturbation",
            attribute_type="Vector",
            reference=f"{perturbation_root}/velocity",
            perturbation=None,
            index=time_index,
            full_shape=velocity_shape,
        )
        _append_basis_attribute(
            grid,
            name="pressure_perturbation",
            attribute_type="Scalar",
            reference=f"{perturbation_root}/pressure",
            perturbation=None,
            index=time_index,
            full_shape=pressure_shape,
        )
    return grid


def _write_ensemble_basis_view(
    path: Path,
    result_path: Path,
    flow_xdmf: Path,
    outer_name: str,
    coefficient_names: tuple[str, ...],
    state_times: np.ndarray,
    num_nodes: int,
) -> None:
    root = ET.parse(flow_xdmf).getroot()
    collection = root.find("./Domain/Grid")
    if collection is None:
        raise ValueError("flow XDMF is missing its temporal collection")
    source_grids = list(collection)
    if len(source_grids) != state_times.size:
        raise ValueError("ensemble basis and flow output times do not match")
    source_times = []
    for grid in source_grids:
        time = grid.find("Time")
        if time is None:
            raise ValueError("flow XDMF time level has no time")
        source_times.append(float(time.attrib["Value"]))
    if not np.allclose(source_times, state_times, rtol=0.0, atol=1.0e-10):
        raise ValueError("ensemble basis and flow output times do not match")

    collection.attrib["Name"] = outer_name
    for grid in source_grids:
        collection.remove(grid)
    result_name = result_path.name
    basis_root = f"{result_name}:/ensemble_basis/{outer_name}"
    reference_root = f"{basis_root}/reference/fields"
    for time_index, (time_value, template) in enumerate(zip(state_times, source_grids)):
        step = ET.Element(
            "Grid",
            Name=f"step_{time_index:06d}",
            GridType="Collection",
            CollectionType="Spatial",
        )
        ET.SubElement(step, "Time", Value=f"{float(time_value):.17g}")
        step.append(
            _basis_member_grid(
                template,
                name="reference",
                reference_root=reference_root,
                perturbation_root=None,
                time_index=time_index,
                num_times=state_times.size,
                num_nodes=num_nodes,
            )
        )
        for mode_index, coefficient_name in enumerate(coefficient_names):
            perturbation_root = (
                f"{basis_root}/perturbations/mode_{mode_index:03d}/fields"
            )
            step.append(
                _basis_member_grid(
                    template,
                    name=coefficient_name,
                    reference_root=reference_root,
                    perturbation_root=perturbation_root,
                    time_index=time_index,
                    num_times=state_times.size,
                    num_nodes=num_nodes,
                )
            )
        collection.append(step)

    ET.indent(root, space="  ")
    with _atomic_path(path) as temporary:
        ET.ElementTree(root).write(
            temporary,
            encoding="utf-8",
            xml_declaration=True,
        )


def write_ensemble_bases(
    result_path: PathLike,
    flow_xdmf: PathLike,
    model: Dynamic,
    bases: Dynamic,
    *,
    first_outer_index: int = 0,
) -> list[Path]:
    """Append full EnVar trajectory bases and write reconstructing XDMF views."""

    if not bases:
        return []
    if (
        isinstance(first_outer_index, bool)
        or not isinstance(first_outer_index, int)
        or first_outer_index < 0
    ):
        raise ValueError("first_outer_index must be a nonnegative integer")
    result_path = Path(result_path)
    flow_xdmf = Path(flow_xdmf)
    outputs = []
    with h5py.File(result_path, "a") as h5:
        ensemble_group = h5.create_group("ensemble_basis")
        for outer_index, basis in enumerate(bases, start=first_outer_index):
            outer_name = f"outer_{outer_index:03d}"
            outer = ensemble_group.create_group(outer_name)
            names = tuple(basis.coefficient_names)
            levels = np.asarray(basis.state_levels, dtype=np.int64)
            times = np.asarray(basis.state_times, dtype=float)
            reference = np.asarray(basis.reference, dtype=float)
            perturbations = np.asarray(basis.perturbations, dtype=float)
            if perturbations.shape != (
                len(names),
                reference.shape[0],
                reference.shape[1],
            ):
                raise ValueError("ensemble basis perturbations have invalid shape")
            if levels.shape != times.shape or levels.size != reference.shape[0]:
                raise ValueError("ensemble basis output levels have invalid shape")
            outer.create_dataset(
                "coefficient_names",
                data=np.asarray(names, dtype=h5py.string_dtype()),
            )
            outer.create_dataset("state_levels", data=levels)
            outer.create_dataset("state_times", data=times)
            _write_basis_fields(outer.create_group("reference"), model, reference)
            perturbation_group = outer.create_group("perturbations")
            for mode_index, states in enumerate(perturbations):
                mode = perturbation_group.create_group(f"mode_{mode_index:03d}")
                _write_basis_fields(mode, model, states)

            view = result_path.with_name(f"ensemble_basis_{outer_name}.xdmf")
            _write_ensemble_basis_view(
                view,
                result_path,
                flow_xdmf,
                outer_name,
                names,
                times,
                int(model.mesh.num_nodes),
            )
            outputs.append(view)
    return outputs


def write_truth_samples(
    source: PathLike,
    path: PathLike,
    times: np.ndarray,
    dt: float,
    *,
    source_times: Optional[np.ndarray] = None,
) -> None:
    """Write fields sampled from a complete truth trajectory at given times.

    ``source`` is the complete HDF5/XDMF trajectory.  Values between solver
    levels are linearly interpolated, matching time interpolation used for
    velocity observations. ``times`` label the sampled output and
    ``source_times`` optionally gives their locations in the source trajectory.
    ``path`` names the sampled HDF5 file.
    """

    source = Path(source).with_suffix(".h5")
    path = Path(path).with_suffix(".h5")
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("truth sample times must be increasing and nonempty")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("truth sample dt must be positive")

    if source_times is None:
        source_times = times
    source_times = np.asarray(source_times, dtype=float)
    if source_times.shape != times.shape or not np.all(np.isfinite(source_times)):
        raise ValueError("truth source sample times must match output times")

    source_xdmf = source.with_suffix(".xdmf")
    target_xdmf = path.with_suffix(".xdmf")
    with h5py.File(source, "r") as input_h5:
        source_steps = len(input_h5["Data"])
        positions = source_times / dt
        if positions[0] < -1.0e-10 or positions[-1] > source_steps - 1 + 1.0e-10:
            raise ValueError("truth sample times are outside the trajectory")
        positions = np.clip(positions, 0.0, source_steps - 1)

        with _atomic_path(path) as temporary:
            with h5py.File(temporary, "w") as output_h5:
                mesh = output_h5.create_group("mesh")
                input_h5.copy("Mesh/Geometry", mesh, name="geometry")
                input_h5.copy("Mesh/Topology", mesh, name="topology")
                data = output_h5.create_group("data")
                for sample, position in enumerate(positions):
                    lower = int(np.floor(position))
                    upper = min(lower + 1, source_steps - 1)
                    weight = position - lower
                    before = input_h5[f"Data/Step{lower:05d}"]
                    after = input_h5[f"Data/Step{upper:05d}"]
                    step = data.create_group(f"step_{sample:06d}")
                    for field in before:
                        values = (1.0 - weight) * before[field][...] + weight * after[
                            field
                        ][...]
                        step.create_dataset(field, data=values)

    root = ET.parse(source_xdmf).getroot()
    collection = root.find("./Domain/Grid")
    if collection is None:
        raise ValueError("truth XDMF is missing its temporal collection")
    template = next(iter(collection), None)
    if template is None:
        raise ValueError("truth XDMF has no time levels")
    for grid in list(collection):
        collection.remove(grid)
    for sample, time in enumerate(times):
        grid = deepcopy(template)
        old_name = grid.attrib["Name"]
        new_name = f"step_{sample:06d}"
        grid.attrib["Name"] = new_name
        time_element = grid.find("Time")
        if time_element is None:
            raise ValueError("truth XDMF time level has no time")
        time_element.attrib["Value"] = f"{time:.17g}"
        for item in grid.iter("DataItem"):
            if item.text:
                item.text = item.text.replace(f"{source.name}:", f"{path.name}:")
                item.text = item.text.replace(
                    "/Mesh/Geometry",
                    "/mesh/geometry",
                )
                item.text = item.text.replace(
                    "/Mesh/Topology",
                    "/mesh/topology",
                )
                item.text = item.text.replace(
                    f"/Data/{old_name}/", f"/data/{new_name}/"
                )
        collection.append(grid)
    ET.indent(root, space="  ")
    with _atomic_path(target_xdmf) as temporary:
        ET.ElementTree(root).write(temporary, encoding="utf-8", xml_declaration=True)


def _vtk_binary(values: np.ndarray, dtype: Dynamic) -> str:
    values = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    payload = values.tobytes()
    header = struct.pack("<Q", len(payload))
    return base64.b64encode(header + payload).decode("ascii")


def _vtk_data(
    parent: ET.Element,
    values: np.ndarray,
    vtk_type: str,
    dtype: Dynamic,
    **attrs: str,
) -> None:
    data = ET.SubElement(
        parent,
        "DataArray",
        type=vtk_type,
        format="binary",
        **attrs,
    )
    data.text = _vtk_binary(values, dtype)


def _write_observation_vtu(
    path: PathLike,
    points: np.ndarray,
    values: np.ndarray,
) -> None:
    num_points, dimension = points.shape
    points_3d = np.zeros((num_points, 3))
    values_3d = np.zeros((num_points, 3))
    points_3d[:, :dimension] = points
    values_3d[:, :dimension] = values

    root = ET.Element(
        "VTKFile",
        type="UnstructuredGrid",
        version="1.0",
        byte_order="LittleEndian",
        header_type="UInt64",
    )
    grid = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(
        grid,
        "Piece",
        NumberOfPoints=str(num_points),
        NumberOfCells=str(num_points),
    )
    point_data = ET.SubElement(piece, "PointData", Vectors="velocity")
    _vtk_data(
        point_data,
        values_3d,
        "Float64",
        "<f8",
        Name="velocity",
        NumberOfComponents="3",
    )
    ET.SubElement(piece, "CellData")

    vtk_points = ET.SubElement(piece, "Points")
    _vtk_data(
        vtk_points,
        points_3d,
        "Float64",
        "<f8",
        NumberOfComponents="3",
    )

    cells = ET.SubElement(piece, "Cells")
    _vtk_data(
        cells,
        np.arange(num_points),
        "Int64",
        "<i8",
        Name="connectivity",
    )
    _vtk_data(
        cells,
        np.arange(1, num_points + 1),
        "Int64",
        "<i8",
        Name="offsets",
    )
    _vtk_data(
        cells,
        np.ones(num_points),
        "UInt8",
        "u1",
        Name="types",
    )

    ET.indent(root, space="  ")
    with _atomic_path(path) as tmp:
        ET.ElementTree(root).write(
            tmp,
            encoding="utf-8",
            xml_declaration=True,
        )


def write_velocity_series(path: PathLike, velocity: VelocityData) -> Path:
    """Write binary VTU point clouds and their ParaView time-series index."""
    pvd_file = Path(path).with_suffix(".pvd")
    data_dir = pvd_file.with_suffix("")

    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True)

    root = ET.Element(
        "VTKFile",
        type="Collection",
        version="1.0",
        byte_order="LittleEndian",
    )
    collection = ET.SubElement(root, "Collection")
    for step, time in enumerate(velocity.times):
        name = f"velocity_{step:05d}.vtu"
        _write_observation_vtu(
            data_dir / name,
            velocity.points,
            velocity.vals[step],
        )
        ET.SubElement(
            collection,
            "DataSet",
            timestep=f"{time:.17g}",
            group="",
            part="0",
            file=f"{data_dir.name}/{name}",
        )

    ET.indent(root, space="  ")
    with _atomic_path(pvd_file) as tmp:
        ET.ElementTree(root).write(
            tmp,
            encoding="utf-8",
            xml_declaration=True,
        )

    return pvd_file

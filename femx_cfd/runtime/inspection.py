"""Read-only case inspection and preflight diagnostics."""

from __future__ import annotations

from typing import Optional

import femx
import numpy as np

from femx_cfd._typing import Config, PathLike
from femx_cfd.cases import FORWARD_FORMAT, LoadedCase, load_case
from femx_cfd.cases.forward import inlet_boundary_condition
from femx_cfd.mesh import mesh_in_meters
from femx_cfd.prep.inlet import analytic_inlet, inlet_normal
from femx_cfd.workflows.forward import _file_inlet
from femx_cfd.workflows.inverse import _initial_inlet

from .output import result_directory_state, trajectory_levels
from .runner import result_path


def _surface_element_measures(surface: object) -> np.ndarray:
    """Return the length or area of every inlet boundary element."""

    coordinates = np.asarray(surface.coordinates, dtype=float)
    elements = np.asarray(surface.elements, dtype=int)
    if elements.ndim != 2 or elements.size == 0:
        raise ValueError("inlet boundary has no elements")

    if int(surface.dimension) == 1:
        return np.linalg.norm(
            coordinates[elements[:, 1]] - coordinates[elements[:, 0]],
            axis=1,
        )

    measures = np.zeros(elements.shape[0], dtype=float)
    for element_index, element in enumerate(elements):
        origin = coordinates[int(element[0])]
        for index in range(1, element.size - 1):
            first = coordinates[int(element[index])] - origin
            second = coordinates[int(element[index + 1])] - origin
            measures[element_index] += 0.5 * float(
                np.linalg.norm(np.cross(first, second))
            )
    return measures


def _surface_scales(surface: object) -> tuple[float, float]:
    coordinates = np.asarray(surface.coordinates, dtype=float)
    elements = np.asarray(surface.elements, dtype=int)
    if elements.ndim != 2 or elements.size == 0:
        raise ValueError("inlet boundary has no elements")

    edge_counts: dict[tuple[int, int], int] = {}
    edge_lengths: dict[tuple[int, int], float] = {}
    for element in elements:
        if element.size == 2:
            pairs = [(int(element[0]), int(element[1]))]
        else:
            pairs = [
                (int(element[index]), int(element[(index + 1) % element.size]))
                for index in range(element.size)
            ]
        for first, second in pairs:
            edge = tuple(sorted((first, second)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            edge_lengths[edge] = float(
                np.linalg.norm(coordinates[first] - coordinates[second])
            )

    positive_edges = [length for length in edge_lengths.values() if length > 0.0]
    if not positive_edges:
        raise ValueError("inlet boundary has no positive edge lengths")
    min_edge = min(positive_edges)

    if int(surface.dimension) == 1:
        characteristic_length = sum(positive_edges)
    else:
        area = float(np.sum(_surface_element_measures(surface)))
        perimeter = sum(
            edge_lengths[edge] for edge, count in edge_counts.items() if count == 1
        )
        if area <= 0.0:
            raise ValueError("inlet boundary has no positive area")
        characteristic_length = (
            4.0 * area / perimeter
            if perimeter > 0.0
            else float(np.sqrt(4.0 * area / np.pi))
        )
    return characteristic_length, min_edge


def _mean_bulk_speed(
    surface: object,
    node_ids: np.ndarray,
    values: np.ndarray,
    times: np.ndarray,
) -> float:
    """Return the time mean of the area-averaged normal inlet speed."""

    surface_node_ids = np.asarray(surface.mesh_node_ids, dtype=np.int64)
    node_index = {int(node_id): index for index, node_id in enumerate(surface_node_ids)}
    try:
        columns = np.asarray(
            [node_index[int(node_id)] for node_id in node_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("inlet values reference a node outside the inlet") from error

    values = np.asarray(values, dtype=float)
    nodal_values = np.zeros((values.shape[0], surface_node_ids.size), dtype=float)
    nodal_values[:, columns] = values

    elements = np.asarray(surface.elements, dtype=int)
    measures = _surface_element_measures(surface)
    total_measure = float(np.sum(measures))
    if total_measure <= 0.0:
        raise ValueError("inlet boundary has no positive measure")
    element_values = np.mean(nodal_values[:, elements], axis=2)
    bulk_velocity = element_values @ measures / total_measure
    speeds = np.abs(bulk_velocity)
    times = np.asarray(times, dtype=float)
    if speeds.size != times.size:
        raise ValueError("inlet velocity and times have incompatible lengths")
    if speeds.size == 1:
        return float(speeds[0])

    intervals = np.diff(times)
    duration = float(times[-1] - times[0])
    if duration <= 0.0 or np.any(intervals <= 0.0):
        raise ValueError("inlet times must be increasing")
    integral = np.sum(0.5 * (speeds[:-1] + speeds[1:]) * intervals)
    return float(integral / duration)


def _inlet_speeds(case: Config, mesh: object) -> tuple[float, float, object]:
    if case["format"] == FORWARD_FORMAT:
        inlet_options = inlet_boundary_condition(case)
        surface = mesh.boundary(inlet_options["boundary"])
        normal = inlet_normal(mesh, surface, inlet_options["normal"])
        if inlet_options["source"] == "file":
            inlet = _file_inlet(case, normal)
        else:
            inlet = analytic_inlet(
                surface,
                {"T": inlet_options["period"], "dt": case["time"]["dt"]},
                normal,
                inlet_options["boundary"],
                space=inlet_options["profile"],
                time=inlet_options["waveform"],
            )
        maximum_speed = float(np.max(np.abs(inlet.vals)))
        mean_speed = _mean_bulk_speed(
            surface,
            inlet.node_ids,
            inlet.vals,
            inlet.times,
        )
        return maximum_speed, mean_speed, surface

    _, velocity = _initial_inlet(case, mesh)
    surface = mesh.boundary(case["control"]["boundary"])
    normal_value = (
        case["control"].get("normal", "auto")
        if case["control"]["kind"] == "normal"
        else "auto"
    )
    normal = inlet_normal(mesh, surface, normal_value)
    normal_values = np.einsum("tnd,d->tn", velocity.vals, normal)
    maximum_speed = float(np.max(np.linalg.norm(velocity.vals, axis=-1)))
    mean_speed = _mean_bulk_speed(
        surface,
        velocity.node_ids,
        normal_values,
        velocity.times,
    )
    return maximum_speed, mean_speed, surface


def _inspect_loaded_case(
    case: LoadedCase,
    destination: Optional[PathLike] = None,
) -> Config:
    """Inspect one loaded case without creating output or starting a solve."""

    resolved = case.resolved
    output_path = result_path(case, destination)
    dt = resolved["time"]["dt"]
    end_time = resolved["time"]["end_time"]
    steps = int(round(end_time / dt))
    levels = trajectory_levels(steps, resolved["output"]["every"])

    with mesh_in_meters(resolved["mesh"]) as mesh_path:
        mesh = femx.Mesh.read(mesh_path)
        maximum_speed, mean_speed, inlet = _inlet_speeds(resolved, mesh)
        characteristic_length, min_edge = _surface_scales(inlet)
        mesh_nodes = int(mesh.num_nodes)
        mesh_elements = int(mesh.num_elements)

    reynolds_scale = (
        resolved["fluid"]["rho"] * characteristic_length / resolved["fluid"]["mu"]
    )

    return {
        "case": str(case.path),
        "format": case.format,
        "source_version": case.source_version,
        "version": case.version,
        "method": resolved.get("method", "forward"),
        "mesh_nodes": mesh_nodes,
        "mesh_elements": mesh_elements,
        "steps": steps,
        "output_directory": str(output_path),
        "output_state": result_directory_state(output_path),
        "output_every": resolved["output"]["every"],
        "output_interval": resolved["output"]["every"] * dt,
        "output_aligned": True,
        "output_levels": int(levels.size),
        "peak_inlet_speed": maximum_speed,
        "mean_inlet_speed": mean_speed,
        "inlet_characteristic_length": characteristic_length,
        "inlet_min_edge": min_edge,
        "reynolds": reynolds_scale * maximum_speed,
        "mean_reynolds": reynolds_scale * mean_speed,
        "cfl": maximum_speed * dt / min_edge,
    }


def inspect_case(
    case_path: PathLike,
    destination: Optional[PathLike] = None,
) -> Config:
    """Load and inspect one portable case."""

    return _inspect_loaded_case(load_case(case_path), destination)


__all__ = ["inspect_case"]

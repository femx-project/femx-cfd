"""Wall-shear-stress evaluation on conforming Navier--Stokes meshes."""

from itertools import combinations
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from femx_cfd._typing import Dynamic


def _local_faces(dimension: int, nodes_per_element: int) -> tuple[tuple[int, ...], ...]:
    if dimension == 2 and nodes_per_element == 3:
        return ((0, 1), (1, 2), (2, 0))
    if dimension == 2 and nodes_per_element == 4:
        return ((0, 1), (1, 2), (2, 3), (3, 0))
    if dimension == 3 and nodes_per_element == 4:
        return tuple(combinations(range(4), 3))
    raise ValueError(
        "wall shear stress supports triangular or quadrilateral 2D elements "
        "and tetrahedral 3D elements"
    )


def _simplex_gradients(coordinates: np.ndarray) -> np.ndarray:
    interpolation = np.column_stack((np.ones(coordinates.shape[0]), coordinates))
    try:
        coefficients = np.linalg.inv(interpolation)
    except np.linalg.LinAlgError as error:
        raise ValueError("wall boundary has a degenerate adjacent element") from error
    return coefficients[1:, :].T


def _quad_gradients(
    coordinates: np.ndarray,
    local_face: tuple[int, ...],
) -> np.ndarray:
    reference = np.asarray(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    xi, eta = np.mean(reference[list(local_face)], axis=0)
    signs_x = reference[:, 0]
    signs_y = reference[:, 1]
    reference_gradients = np.column_stack(
        (
            0.25 * signs_x * (1.0 + signs_y * eta),
            0.25 * signs_y * (1.0 + signs_x * xi),
        )
    )
    jacobian = coordinates.T @ reference_gradients
    try:
        inverse_jacobian = np.linalg.inv(jacobian)
    except np.linalg.LinAlgError as error:
        raise ValueError("wall boundary has a degenerate adjacent element") from error
    return reference_gradients @ inverse_jacobian


def _shape_gradients(
    coordinates: np.ndarray,
    local_face: tuple[int, ...],
) -> np.ndarray:
    dimension = coordinates.shape[1]
    if coordinates.shape[0] == dimension + 1:
        return _simplex_gradients(coordinates)
    if dimension == 2 and coordinates.shape[0] == 4:
        return _quad_gradients(coordinates, local_face)
    raise ValueError("unsupported element for wall shear stress")


def _outward_normal(
    facet_coordinates: np.ndarray,
    cell_coordinates: np.ndarray,
) -> np.ndarray:
    dimension = cell_coordinates.shape[1]
    if dimension == 2:
        tangent = facet_coordinates[1] - facet_coordinates[0]
        normal = np.asarray([tangent[1], -tangent[0]])
    elif dimension == 3:
        normal = np.cross(
            facet_coordinates[1] - facet_coordinates[0],
            facet_coordinates[2] - facet_coordinates[0],
        )
    else:
        raise ValueError("wall shear stress requires a 2D or 3D mesh")

    length = float(np.linalg.norm(normal))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("wall boundary contains a degenerate facet")
    normal /= length
    outward = np.mean(facet_coordinates, axis=0) - np.mean(
        cell_coordinates,
        axis=0,
    )
    orientation = float(normal @ outward)
    if np.isclose(orientation, 0.0):
        raise ValueError("cannot orient a wall facet from its adjacent element")
    if orientation < 0.0:
        normal *= -1.0
    return normal


class WallShearStressEvaluator:
    """Evaluate facet-centered Newtonian WSS on one named no-slip wall."""

    def __init__(
        self,
        mesh: Dynamic,
        boundary: str,
        dynamic_viscosity: float,
    ) -> None:
        self.boundary = boundary
        self.dynamic_viscosity = float(dynamic_viscosity)
        if not np.isfinite(self.dynamic_viscosity) or self.dynamic_viscosity <= 0.0:
            raise ValueError("dynamic viscosity must be positive")

        self.dimension = int(mesh.dimension)
        if self.dimension not in (2, 3):
            raise ValueError("wall shear stress requires a 2D or 3D mesh")
        self.num_mesh_nodes = int(mesh.num_nodes)
        self._coordinates = np.asarray(mesh.coordinates, dtype=float)[
            :, : self.dimension
        ]
        self._cells = np.asarray(mesh.elements, dtype=int)
        if self._cells.ndim != 2 or self._cells.shape[0] == 0:
            raise ValueError("wall shear stress requires a nonempty uniform mesh")

        surface = mesh.boundary(boundary)
        self.coordinates = np.asarray(surface.coordinates, dtype=float)
        self.elements = np.asarray(surface.elements, dtype=np.int32)
        self.mesh_node_ids = np.asarray(surface.mesh_node_ids, dtype=np.int32)
        if self.elements.ndim != 2 or self.elements.shape[0] == 0:
            raise ValueError("wall shear stress requires a nonempty wall boundary")

        global_facets = self.mesh_node_ids[self.elements]
        faces = _local_faces(self.dimension, self._cells.shape[1])
        cell_by_face: dict[frozenset[int], tuple[int, tuple[int, ...]]] = {}
        duplicated: set[frozenset[int]] = set()
        for cell_id, cell in enumerate(self._cells):
            for local_face in faces:
                key = frozenset(int(cell[index]) for index in local_face)
                if key in cell_by_face:
                    duplicated.add(key)
                else:
                    cell_by_face[key] = (cell_id, local_face)
        for key in duplicated:
            cell_by_face.pop(key, None)

        adjacent_cells = []
        local_wall_faces = []
        normals = []
        for facet in global_facets:
            key = frozenset(int(node) for node in facet)
            try:
                cell_id, local_face = cell_by_face[key]
            except KeyError as error:
                raise ValueError(
                    f"boundary {boundary!r} facet has no unique adjacent element"
                ) from error
            cell = self._cells[cell_id]
            normals.append(
                _outward_normal(
                    self._coordinates[facet],
                    self._coordinates[cell],
                )
            )
            adjacent_cells.append(cell_id)
            local_wall_faces.append(local_face)

        self.adjacent_cells = np.asarray(adjacent_cells, dtype=np.int32)
        self._local_faces = tuple(local_wall_faces)
        self.normals = np.zeros((len(normals), 3))
        self.normals[:, : self.dimension] = np.asarray(normals)
        self._facet_cells = self._cells[self.adjacent_cells]
        self._shape_gradients = np.asarray(
            [
                _shape_gradients(
                    self._coordinates[cell],
                    local_face,
                )
                for cell, local_face in zip(
                    self._facet_cells,
                    self._local_faces,
                )
            ]
        )
        self._prepare_nodal_projection()

    def _prepare_nodal_projection(self) -> None:
        nodes_per_facet = self.elements.shape[1]
        if nodes_per_facet == 2:
            delta = (
                self.coordinates[self.elements[:, 1]]
                - self.coordinates[self.elements[:, 0]]
            )
            measures = np.linalg.norm(delta, axis=1)
        elif nodes_per_facet == 3:
            first = (
                self.coordinates[self.elements[:, 1]]
                - self.coordinates[self.elements[:, 0]]
            )
            second = (
                self.coordinates[self.elements[:, 2]]
                - self.coordinates[self.elements[:, 0]]
            )
            measures = 0.5 * np.linalg.norm(np.cross(first, second), axis=1)
        else:
            raise ValueError("nodal WSS projection requires line or triangle facets")
        if not np.all(np.isfinite(measures)) or np.any(measures <= 0.0):
            raise ValueError("wall boundary contains a degenerate facet")

        local_template = np.ones((nodes_per_facet, nodes_per_facet))
        local_template[np.diag_indices(nodes_per_facet)] = 2.0
        local_mass = (
            measures[:, None, None]
            * local_template[None, :, :]
            / (nodes_per_facet * (nodes_per_facet + 1))
        )
        rows = np.repeat(self.elements, nodes_per_facet, axis=1).ravel()
        columns = np.tile(self.elements, (1, nodes_per_facet)).ravel()
        mass = sparse.coo_matrix(
            (local_mass.ravel(), (rows, columns)),
            shape=(self.coordinates.shape[0], self.coordinates.shape[0]),
        ).tocsc()
        try:
            self._solve_projection = sparse_linalg.factorized(mass)
        except RuntimeError as error:
            raise ValueError(
                "wall boundary has a singular nodal mass matrix"
            ) from error
        self._projection_rhs_weights = measures / nodes_per_facet

    def evaluate(self, velocity: Dynamic) -> np.ndarray:
        """Return wall-on-fluid shear traction at every wall facet center."""

        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (self.num_mesh_nodes, 3):
            raise ValueError(
                "velocity for wall shear stress must have shape (num_nodes, 3)"
            )
        if not np.all(np.isfinite(velocity)):
            raise ValueError("velocity for wall shear stress must be finite")

        cell_velocity = velocity[self._facet_cells, : self.dimension]
        velocity_gradient = np.einsum(
            "fna,fnb->fab",
            cell_velocity,
            self._shape_gradients,
        )
        viscous_stress = self.dynamic_viscosity * (
            velocity_gradient + velocity_gradient.transpose(0, 2, 1)
        )
        normal = self.normals[:, : self.dimension]
        traction = np.einsum("fab,fb->fa", viscous_stress, normal)
        traction -= normal * np.einsum("fa,fa->f", traction, normal)[:, None]
        values = np.zeros((self.elements.shape[0], 3))
        values[:, : self.dimension] = traction
        return values

    def project_to_nodes(self, facet_values: Dynamic) -> np.ndarray:
        """L2-project one facet-constant vector field to wall P1 nodes."""

        facet_values = np.asarray(facet_values, dtype=float)
        if facet_values.shape != (self.elements.shape[0], 3):
            raise ValueError(
                "facet values for nodal WSS projection must have shape (num_facets, 3)"
            )
        if not np.all(np.isfinite(facet_values)):
            raise ValueError("facet values for nodal WSS projection must be finite")

        right_hand_side = np.zeros((self.coordinates.shape[0], 3))
        weighted = self._projection_rhs_weights[:, None] * facet_values
        for local_node in range(self.elements.shape[1]):
            np.add.at(
                right_hand_side,
                self.elements[:, local_node],
                weighted,
            )
        return np.column_stack(
            [
                self._solve_projection(right_hand_side[:, component])
                for component in range(3)
            ]
        )


class WallShearStressTimeAverage:
    """Integrate one discrete WSS field over one closed time window."""

    def __init__(
        self,
        num_locations: int,
        start_time: float,
        end_time: float,
    ) -> None:
        self.num_locations = int(num_locations)
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        if self.num_locations <= 0:
            raise ValueError("WSS time averaging requires at least one location")
        if (
            not np.isfinite(self.start_time)
            or not np.isfinite(self.end_time)
            or self.end_time <= self.start_time
        ):
            raise ValueError("WSS time averaging requires a positive time window")

        self._previous_time: Optional[float] = None
        self._previous_values: Optional[np.ndarray] = None
        self._integral_magnitude = np.zeros(self.num_locations)

    def _is_close(self, left: float, right: float) -> bool:
        tolerance = 1.0e-10 * max(1.0, abs(self.end_time))
        return bool(np.isclose(left, right, rtol=0.0, atol=tolerance))

    def add(self, time: float, values: Dynamic) -> None:
        """Add one solver-level WSS field to the trapezoidal integral."""

        time = float(time)
        if not np.isfinite(time):
            raise ValueError("WSS sample time must be finite")
        if time < self.start_time and not self._is_close(time, self.start_time):
            return
        if time > self.end_time and not self._is_close(time, self.end_time):
            return
        if self._is_close(time, self.start_time):
            time = self.start_time
        if self._is_close(time, self.end_time):
            time = self.end_time

        values = np.asarray(values, dtype=float)
        if values.shape != (self.num_locations, 3):
            raise ValueError(
                "WSS time-average samples must have shape (num_locations, 3)"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("WSS time-average samples must be finite")

        if self._previous_time is None:
            if time != self.start_time:
                raise ValueError(
                    "WSS time averaging is missing its initial window sample"
                )
        else:
            step = time - self._previous_time
            if step <= 0.0:
                raise ValueError(
                    "WSS time-average sample times must be strictly increasing"
                )
            assert self._previous_values is not None
            self._integral_magnitude += (
                0.5
                * (
                    np.linalg.norm(self._previous_values, axis=1)
                    + np.linalg.norm(values, axis=1)
                )
                * step
            )
        self._previous_time = time
        self._previous_values = values.copy()

    def finish(self) -> np.ndarray:
        """Return TAWSS after the complete window."""

        if self._previous_time is None or not self._is_close(
            self._previous_time,
            self.end_time,
        ):
            raise RuntimeError("WSS time averaging is missing its final window sample")

        duration = self.end_time - self.start_time
        return self._integral_magnitude / duration


__all__ = ["WallShearStressEvaluator", "WallShearStressTimeAverage"]

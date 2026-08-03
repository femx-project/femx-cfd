import unittest
from pathlib import Path

import femx
import numpy as np

from femx_cfd.models.navier_stokes.wss import (
    WallShearStressEvaluator,
    WallShearStressTimeAverage,
)


MESH = Path(__file__).resolve().parents[1] / "data" / "2d_tiny_tube.msh"


class _Surface:
    dimension = 2
    coordinates = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    elements = np.asarray([[0, 1, 2]])
    mesh_node_ids = np.asarray([0, 1, 2])


class _TetrahedralMesh:
    dimension = 3
    num_nodes = 4
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    elements = np.asarray([[0, 1, 2, 3]])

    def boundary(self, name):
        if name != "wall":
            raise ValueError(name)
        return _Surface()


class WallShearStressTest(unittest.TestCase):
    def test_evaluates_tangential_newtonian_traction_on_quadrilateral_wall(self):
        mesh = femx.Mesh.read(MESH)
        evaluator = WallShearStressEvaluator(mesh, "wall", 0.5)
        coordinates = np.asarray(mesh.coordinates)
        velocity = np.zeros((mesh.num_nodes, 3))
        y = coordinates[:, 1]
        velocity[:, 0] = 4.0 * y * (1.0 - y)

        values = evaluator.evaluate(velocity)

        np.testing.assert_allclose(values[:, 0], -1.0)
        np.testing.assert_allclose(values[:, 1:], 0.0, atol=1.0e-14)
        np.testing.assert_allclose(
            np.sum(values * evaluator.normals, axis=1),
            0.0,
            atol=1.0e-14,
        )
        nodal_values = evaluator.project_to_nodes(values)
        self.assertEqual(nodal_values.shape, evaluator.coordinates.shape)
        np.testing.assert_allclose(nodal_values[:, 0], -1.0)
        np.testing.assert_allclose(nodal_values[:, 1:], 0.0, atol=1.0e-14)
        self.assertEqual(evaluator.elements.shape, (4, 2))

    def test_evaluates_tangential_newtonian_traction_on_tetrahedral_wall(self):
        mesh = _TetrahedralMesh()
        evaluator = WallShearStressEvaluator(mesh, "wall", 0.25)
        velocity = np.zeros((mesh.num_nodes, 3))
        velocity[:, 0] = mesh.coordinates[:, 2]

        values = evaluator.evaluate(velocity)

        np.testing.assert_allclose(evaluator.normals, [[0.0, 0.0, -1.0]])
        np.testing.assert_allclose(values, [[-0.25, 0.0, 0.0]])
        np.testing.assert_allclose(
            evaluator.project_to_nodes(values),
            [[-0.25, 0.0, 0.0]] * 3,
        )

    def test_integrates_tawss(self):
        average = WallShearStressTimeAverage(1, 0.0, 1.0)
        average.add(0.0, [[1.0, 0.0, 0.0]])
        average.add(0.5, [[-1.0, 0.0, 0.0]])
        average.add(1.0, [[1.0, 0.0, 0.0]])

        np.testing.assert_allclose(average.finish(), [1.0])

    def test_integrates_only_the_requested_time_window(self):
        average = WallShearStressTimeAverage(1, 1.0, 2.0)
        average.add(0.0, [[100.0, 0.0, 0.0]])
        average.add(1.0, [[2.0, 0.0, 0.0]])
        average.add(1.5, [[2.0, 0.0, 0.0]])
        average.add(2.0, [[2.0, 0.0, 0.0]])

        np.testing.assert_allclose(average.finish(), [2.0])


if __name__ == "__main__":
    unittest.main()

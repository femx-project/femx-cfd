import unittest
from pathlib import Path

import femx
import numpy as np

from femx_cfd.prep.mri import project_to_inlet


MESH = Path(__file__).resolve().parents[1] / "data" / "2d_tiny_tube.msh"


class MriProjectionTest(unittest.TestCase):
    def setUp(self):
        mesh = femx.Mesh.read(MESH)
        self.surface = mesh.boundary("inlet")
        self.points = np.column_stack((np.full(5, 0.1), np.linspace(0.0, 1.0, 5)))
        self.times = np.asarray([0.0, 0.5, 1.0])
        self.velocity = np.empty((3, 5, 2))
        self.velocity[:, :, 0] = np.asarray([1.0, -1.0, 1.0])[:, None]
        self.velocity[:, :, 1] = 2.0
        self.options = {
            "depth": 0.2,
            "axial_decay": 0.2,
            "laplacian_modes": 1,
            "ridge": 0.001,
        }

    def project(self, temporal_weight=None):
        options = self.options.copy()
        if temporal_weight is not None:
            options["temporal_weight"] = temporal_weight
        return project_to_inlet(
            self.surface,
            self.points,
            self.times,
            self.velocity,
            np.asarray([1.0, 0.0]),
            options,
        )

    def test_reconstructs_all_velocity_components(self):
        reconstructed = self.project()

        self.assertEqual(reconstructed.shape, (3, 3, 2))
        self.assertGreater(np.max(np.abs(reconstructed[:, :, 1])), 0.0)
        np.testing.assert_allclose(reconstructed, self.project(0.0))

    def test_temporal_regularization_reduces_time_variation(self):
        unregularized = self.project(0.0)
        regularized = self.project(10.0)

        self.assertLess(
            np.linalg.norm(np.diff(regularized, axis=0)),
            np.linalg.norm(np.diff(unregularized, axis=0)),
        )


if __name__ == "__main__":
    unittest.main()

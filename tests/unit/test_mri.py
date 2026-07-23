import unittest
from pathlib import Path

import femx
import numpy as np

from femx_cfd.prep.mri import (
    project_poisson_fourier_to_inlet,
    project_to_inlet,
)


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

    def test_recovers_poisson_profile_and_fourier_waveform(self):
        normal = np.asarray([1.0, 0.0])
        points = np.asarray(self.surface.coordinates)[:, :2] + 0.1 * normal
        times = np.asarray([0.0, 0.25, 0.5, 0.75])
        output_times = np.linspace(0.0, 1.0, 9)
        _, _, load = self.surface.laplacian_matrices()
        profile = float(np.sum(load)) * self.surface.poisson_profile(normalize=True)
        waveform = 0.2 + 0.05 * np.sin(2.0 * np.pi * times)
        waveform += 0.03 * np.cos(2.0 * np.pi * times)
        velocity = waveform[:, None, None] * profile[None, :, None] * normal
        velocity[..., 1] += 2.0

        reconstructed, coeff = project_poisson_fourier_to_inlet(
            self.surface,
            points,
            times,
            velocity,
            normal,
            {"depth": 0.2, "fourier_order": 1},
            period=1.0,
            output_times=output_times,
        )

        expected_waveform = 0.2 + 0.05 * np.sin(2.0 * np.pi * output_times)
        expected_waveform += 0.03 * np.cos(2.0 * np.pi * output_times)
        expected = expected_waveform[:, None, None] * profile[None, :, None] * normal
        expected[-1] = expected[0]
        np.testing.assert_allclose(coeff, [0.2, 0.05, 0.03], atol=1.0e-12)
        np.testing.assert_allclose(reconstructed, expected, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()

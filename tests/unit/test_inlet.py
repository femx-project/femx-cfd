import unittest

import numpy as np

from femx_cfd.prep.inlet import analytic_inlet, inlet_normal, repeat_inlet


class Surface:
    interior_node_ids = np.array([1, 2])
    mesh_node_ids = np.array([10, 11, 12, 13])
    coordinates = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    @staticmethod
    def poisson_profile(normalize=False):
        return np.array([0.0, 0.5, 1.0, 0.0])


class Mesh2D:
    dimension = 2
    coordinates = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )


class Mesh3D:
    dimension = 3
    coordinates = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )


class Surface3D:
    coordinates = Mesh3D.coordinates[[2, 0, 3, 1]]


class InletNormalTest(unittest.TestCase):
    def test_estimates_inward_normal_in_2d(self):
        np.testing.assert_allclose(
            inlet_normal(Mesh2D(), Surface(), "auto"),
            [1.0, 0.0],
        )

    def test_estimates_inward_normal_in_3d_independent_of_node_order(self):
        np.testing.assert_allclose(
            inlet_normal(Mesh3D(), Surface3D(), "auto"),
            [0.0, 0.0, 1.0],
        )

    def test_manual_normal_overrides_estimate(self):
        np.testing.assert_allclose(
            inlet_normal(Mesh2D(), Surface(), [-2.0, 0.0]),
            [-1.0, 0.0],
        )


class AnalyticInletTest(unittest.TestCase):
    def test_evaluates_profile_and_waveform(self):
        cfg = {
            "T": 1.0,
            "dt": 0.25,
        }
        space = {"kind": "poisson", "peak": 2.0}
        time = {
            "kind": "harmonics",
                "mean": 1.0,
                "terms": [
                    {"kind": "sin", "order": 1, "amplitude": 0.5}
                ],
        }

        inlet = analytic_inlet(
            Surface(),
            cfg,
            normal=[1.0, 0.0],
            boundary="inlet",
            space=space,
            time=time,
        )

        np.testing.assert_array_equal(inlet.node_ids, [11, 12])
        np.testing.assert_allclose(inlet.vals[0], [1.0, 2.0])
        np.testing.assert_allclose(inlet.vals[1], [1.5, 3.0])
        np.testing.assert_allclose(inlet.vals[-1], inlet.vals[0])
        self.assertEqual(inlet.params.size, 8)

    def test_repeats_periodic_inlet_without_duplicate_seams(self):
        cfg = {
            "T": 1.0,
            "dt": 0.5,
        }
        inlet = repeat_inlet(
            analytic_inlet(
                Surface(),
                cfg,
                normal=[1.0, 0.0],
                boundary="inlet",
                space={"kind": "poisson", "peak": 2.0},
                time={"kind": "uniform"},
            ),
            2,
        )

        np.testing.assert_allclose(inlet.times, [0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertFalse(inlet.periodic)
        self.assertEqual(inlet.params.size, 10)

    def test_repeat_uses_exact_solver_end_time(self):
        cfg = {
            "T": 19.634954084936208,
            "dt": 0.19634954084936207,
        }
        inlet = analytic_inlet(
            Surface(),
            cfg,
            normal=[1.0, 0.0],
            boundary="inlet",
            space={"kind": "poisson", "peak": 2.0},
            time={"kind": "uniform"},
        )
        end_time = 98.17477042468103

        repeated = repeat_inlet(inlet, 5, end_time=end_time)

        self.assertEqual(repeated.times[-1], end_time)

    def test_evaluates_uniform_space_and_time(self):
        inlet = analytic_inlet(
            Surface(),
            {"T": 1.0, "dt": 0.5},
            normal=[1.0, 0.0],
            boundary="inlet",
            space={"kind": "uniform", "value": 2.0},
            time={"kind": "uniform", "value": 0.5},
        )

        np.testing.assert_allclose(inlet.vals, 1.0)


if __name__ == "__main__":
    unittest.main()

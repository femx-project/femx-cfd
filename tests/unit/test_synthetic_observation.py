"""Tests for synthetic observation generation helpers."""

import unittest
from pathlib import Path

import numpy as np

from tools.synthetic.observation import observation_points, observation_times


MESH = Path(__file__).resolve().parents[1] / "data" / "2d_tiny_tube.msh"


class SyntheticObservationTest(unittest.TestCase):
    def test_observation_points_form_y_major_cartesian_lattice(self):
        case = {
            "observation": {
                "lower": [0.0, 0.0],
                "upper": [2.0, 1.0],
                "counts": [3, 2],
                "mask": "none",
            }
        }

        np.testing.assert_allclose(
            observation_points(case),
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ],
        )

    def test_mesh_mask_removes_outside_points(self):
        case = {
            "mesh": str(MESH),
            "observation": {
                "lower": [0.25, 0.25],
                "upper": [2.0, 0.75],
                "counts": [3, 2],
                "mask": "mesh",
            },
        }

        points = observation_points(case)

        np.testing.assert_array_equal(points, [[0.25, 0.25], [0.25, 0.75]])

    def test_observation_phases_cover_one_cycle(self):
        case = {"time": {"T": 1.0}, "observation": {"phases": 10}}

        np.testing.assert_allclose(observation_times(case), np.arange(10) / 10.0)
        np.testing.assert_allclose(
            observation_times(case, cycle=2),
            1.0 + np.arange(10) / 10.0,
        )

    def test_selects_one_based_observation_phases(self):
        case = {
            "time": {"T": 1.0},
            "observation": {"phases": 4, "selected_phases": [2, 4]},
        }

        np.testing.assert_allclose(observation_times(case), [0.25, 0.75])
        np.testing.assert_allclose(observation_times(case, cycle=2), [1.25, 1.75])


if __name__ == "__main__":
    unittest.main()

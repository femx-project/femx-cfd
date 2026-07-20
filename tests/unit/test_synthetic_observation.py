"""Tests for synthetic observation generation helpers."""

import unittest

import numpy as np

from tools.synthetic.observation import observation_times


class SyntheticObservationTest(unittest.TestCase):
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
